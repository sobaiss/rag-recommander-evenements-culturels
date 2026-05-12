import datetime
import json
import logging
import os
import re
from typing import Any, Callable

from langchain_classic.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from utils.config import (
    EMBEDDING_MODEL,
    VECTOR_DB_DIR,
)

# ---------------------------------------------------------------------------
# Détection des expressions temporelles (pour injection de contexte date)
# ---------------------------------------------------------------------------

_TEMPORAL_RE = re.compile(
    r"\bce\s+mois\b|\baujourd['\s]?hui\b|\bdemain\b|\bce\s+week[\s-]?end\b|\bweekend\b"
    r"|\bsemaine\s+prochaine\b|\bcette\s+semaine\b|\bmois\s+prochain\b"
    r"|\bà\s+venir\b|\bprochains?\b|\bprochainement\b|\bfuturs?\b|\ben\s+cours\b|\bencores?\s+actifs?\b"
    r"|\b(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)"
    r"\s+\d{4}\b"
    r"|\b\d{4}\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# SelfQueryRetriever configuration
# ---------------------------------------------------------------------------

DOCUMENT_CONTENT_DESCRIPTION = (
    "Description d'un événement culturel public en France "
    "(concert, exposition, festival, spectacle, atelier, conférence, etc.). "
    "Les événements peuvent s'étaler sur plusieurs mois : start_date est le début, "
    "end_date est la fin. Un événement est actif si start_date <= date_cherchée <= end_date."
)

METADATA_FIELD_INFO = [
    AttributeInfo(
        name="city",
        description="Ville où se déroule l'événement (ex: Paris, Lyon, Toulouse)",
        type="string",
    ),
    AttributeInfo(
        name="region",
        description="Région où se déroule l'événement (ex: Île-de-France, Occitanie)",
        type="string",
    ),
    AttributeInfo(
        name="start_date",
        description=(
            "Date de début de l'événement au format YYYY-MM-DD (ex: 2025-05-06). "
            "IMPORTANT : un événement est EN COURS pendant une période [D1, D2] si start_date <= D2. "
            "Pour chercher les événements actifs en mai 2026, utiliser start_date <= '2026-05-31'."
        ),
        type="string",
    ),
    AttributeInfo(
        name="end_date",
        description=(
            "Date de fin de l'événement au format YYYY-MM-DD (ex: 2026-05-31). "
            "IMPORTANT : un événement est EN COURS pendant une période [D1, D2] si end_date >= D1. "
            "Pour chercher les événements actifs en mai 2026, utiliser end_date >= '2026-05-01'."
        ),
        type="string",
    ),
    AttributeInfo(
        name="is_free",
        description="Vrai si l'événement est gratuit, Faux sinon",
        type="boolean",
    ),
    AttributeInfo(
        name="keywords",
        description="Mots-clés et thématiques de l'événement (ex: jazz, nature, patrimoine, danse)",
        type="string",
    ),
]


class MetadataModel:
    embedding_model: str
    created_at: str
    num_documents: int
    cities: list[str]


# ---------------------------------------------------------------------------
# Custom FAISS translator (callable filter — supporte tous les opérateurs)
# ---------------------------------------------------------------------------


class FaissCallableTranslator(Visitor):
    """Traduit un StructuredQuery en callable Python pour le filtre FAISS.

    FAISS accepte un callable `(metadata: dict) -> bool` comme filtre.
    Cette approche supporte EQ, NE, GT, GTE, LT, LTE, LIKE et les
    opérateurs AND/OR/NOT, y compris les plages de dates ISO (comparaison
    lexicographique correcte sur le format YYYY-MM-DD).
    """

    allowed_comparators = [
        Comparator.EQ,
        Comparator.NE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.LIKE,
    ]
    allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]

    def visit_comparison(self, comparison: Comparison) -> Callable:
        attr = comparison.attribute
        comparator = comparison.comparator
        value = comparison.value

        def _normalize(v: Any) -> str:
            """Normalise les valeurs pour la comparaison (gère bool, str, int, dict date)."""
            if isinstance(v, dict):
                return v.get(
                    "date", str(v)
                )  # {'date': '2026-05-01', 'type': 'date'} → '2026-05-01'
            if isinstance(v, bool):
                return str(v).lower()  # True→"true", False→"false"
            if isinstance(v, str) and v.lower() in ("true", "false"):
                return v.lower()
            return str(v)

        def check(metadata: dict) -> bool:
            meta_val = metadata.get(attr)
            if meta_val is None:
                return False
            mv = _normalize(meta_val)
            vv = _normalize(value)
            if comparator == Comparator.EQ:
                return mv == vv
            if comparator == Comparator.NE:
                return mv != vv
            if comparator == Comparator.GT:
                return mv > vv
            if comparator == Comparator.GTE:
                return mv >= vv
            if comparator == Comparator.LT:
                return mv < vv
            if comparator == Comparator.LTE:
                return mv <= vv
            if comparator == Comparator.LIKE:
                return vv.lower() in mv.lower()
            return False

        return check

    def visit_operation(self, operation: Operation) -> Callable:
        filters = [arg.accept(self) for arg in operation.arguments]
        if operation.operator == Operator.AND:
            return lambda m: all(f(m) for f in filters)
        if operation.operator == Operator.OR:
            return lambda m: any(f(m) for f in filters)
        # NOT
        return lambda m: not filters[0](m)

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> tuple[str, dict]:
        if structured_query.filter is not None:
            filter_func = structured_query.filter.accept(self)
            return structured_query.query, {"filter": filter_func}
        return structured_query.query, {}


# ---------------------------------------------------------------------------
# VectorStoreManager
# ---------------------------------------------------------------------------


class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index FAISS LangChain.

    Utilise SelfQueryRetriever pour extraire des filtres structurés (ville, dates,
    gratuité) depuis la requête en langage naturel avant la recherche vectorielle.
    """

    def __init__(
        self,
        embedding_model: str | None = None,
        vector_db_dir: str = VECTOR_DB_DIR,
    ):
        self._vector_db_dir = vector_db_dir
        self._index_metadata_file = os.path.join(vector_db_dir, "index_metadata.json")

        saved_meta = self._read_metadata()
        self.embedding_model = (
            embedding_model
            or (saved_meta.get("embedding_model") if saved_meta else None)
            or EMBEDDING_MODEL
        )

        self._faiss_store: LangchainFAISS | None = None
        self._llm: ChatMistralAI | None = None

        self._load_index_and_chunks()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _read_metadata(self) -> dict[str, Any] | None:
        if os.path.exists(self._index_metadata_file):
            try:
                with open(self._index_metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def get_metadata(self) -> dict[str, Any] | None:
        return self._read_metadata()

    def _save_metadata(
        self, num_documents: int, cities: list[str] | None = None
    ) -> None:
        os.makedirs(os.path.dirname(self._index_metadata_file), exist_ok=True)
        meta = {
            "embedding_model": self.embedding_model,
            "created_at": datetime.datetime.now().isoformat(),
            "num_documents": num_documents,
            "cities": sorted(cities) if cities else [],
        }
        with open(self._index_metadata_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Embedding / LLM helpers
    # ------------------------------------------------------------------

    def _is_hf_model(self) -> bool:
        return not self.embedding_model.startswith("mistral")

    def _get_langchain_embeddings(self):
        if self._is_hf_model():
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            except ImportError as exc:
                raise ImportError(
                    "Le package sentence-transformers n'est pas installé. "
                    "Exécutez: uv add sentence-transformers"
                ) from exc
            return HuggingFaceEmbeddings(model_name=self.embedding_model)
        return MistralAIEmbeddings(model=self.embedding_model)

    def _get_llm(self) -> ChatMistralAI:
        if self._llm is None:
            self._llm = ChatMistralAI(temperature=0)
        return self._llm

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _load_index_and_chunks(self) -> None:
        index_path = os.path.join(self._vector_db_dir, "index.faiss")
        if os.path.exists(index_path):
            try:
                logging.info(
                    f"Chargement de l'index LangChain FAISS depuis {self._vector_db_dir}."
                )
                embeddings = self._get_langchain_embeddings()
                self._faiss_store = LangchainFAISS.load_local(
                    self._vector_db_dir,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                logging.info(
                    f"Index chargé ({self._faiss_store.index.ntotal} vecteurs)."
                )
            except Exception as e:
                logging.error(f"Erreur lors du chargement de l'index FAISS: {e}")
                self._faiss_store = None
        else:
            self._faiss_store = None
            logging.info("Aucun index existant trouvé. Initialisation d'un index vide.")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_index(
        self,
        documents: list[Document],
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Construit l'index FAISS LangChain à partir des documents."""

        def _progress(msg: str) -> None:
            logging.info(msg)
            if progress_callback:
                progress_callback(msg)

        if not documents:
            logging.warning("Aucun document fourni pour la construction de l'index.")
            return

        _progress(
            f"Génération des embeddings pour {len(documents)} documents (modèle: {self.embedding_model})..."
        )
        embeddings = self._get_langchain_embeddings()

        _progress("Construction de l'index FAISS...")
        self._faiss_store = LangchainFAISS.from_documents(documents, embeddings)

        _progress("Sauvegarde de l'index sur le disque...")
        os.makedirs(self._vector_db_dir, exist_ok=True)
        self._faiss_store.save_local(self._vector_db_dir)
        cities = sorted(
            {
                doc.metadata.get("city", "")
                for doc in documents
                if doc.metadata.get("city")
            }
        )
        self._save_metadata(num_documents=len(documents), cities=cities)
        _progress(f"Index sauvegardé — {len(documents)} documents indexés.")

    # ------------------------------------------------------------------
    # Search (SelfQueryRetriever + scores)
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        k: int = 5,
        min_score: float | None = None,
    ) -> list[dict]:
        """Recherche avec SelfQueryRetriever : filtres structurés + similarité cosinus.

        Le LLM extrait les filtres (ville, dates, gratuité) de la requête naturelle,
        puis FAISS retourne les documents les plus similaires parmi ceux qui passent
        ces filtres.
        """
        if self._faiss_store is None:
            logging.warning("Recherche impossible : aucun index chargé.")
            return []

        fetch_k = k * 3 if min_score is not None else k
        semantic_query = query_text
        search_kwargs: dict = {}

        # --- Parsing structuré via SelfQueryRetriever ---
        try:
            retriever = SelfQueryRetriever.from_llm(
                llm=self._get_llm(),
                vectorstore=self._faiss_store,
                document_contents=DOCUMENT_CONTENT_DESCRIPTION,
                metadata_field_info=METADATA_FIELD_INFO,
                structured_query_translator=FaissCallableTranslator(),
                enable_limit=True,
                verbose=True,
            )
            constructor_query = query_text
            if _TEMPORAL_RE.search(query_text):
                today = datetime.date.today()
                first_day = today.replace(day=1)
                last_day = (today.replace(day=28) + datetime.timedelta(days=4)).replace(
                    day=1
                ) - datetime.timedelta(days=1)
                date_ctx = (
                    f"[Date du jour: {today.isoformat()}. Règles de filtrage selon l'expression temporelle: "
                    f"'à venir' / 'prochains' → end_date >= '{today.isoformat()}'; "
                    f"'aujourd\\'hui' → start_date <= '{today.isoformat()}' ET end_date >= '{today.isoformat()}'; "
                    f"'ce mois-ci' / 'en cours' → start_date <= '{last_day.isoformat()}' ET end_date >= '{first_day.isoformat()}'; "
                    f"'mois YYYY' → start_date <= dernier_jour_du_mois ET end_date >= premier_jour_du_mois. "
                    f"Un événement commencé avant la période peut encore être actif si end_date est dans la période.]"
                )
                constructor_query = f"{query_text} {date_ctx}"
            structured_query = retriever.query_constructor.invoke(
                {"query": constructor_query}
            )
            semantic_query, search_kwargs = retriever._prepare_query(
                query_text, structured_query
            )
            logging.debug(f"semantic query: {semantic_query}")

            if not semantic_query or not semantic_query.strip():
                meta = self._read_metadata()
                cities = meta.get("cities", []) if meta else []
                semantic_query = (
                    "événements " + " ".join(cities) if cities else query_text
                )
            logging.info(
                f"SelfQueryRetriever — requête sémantique: {semantic_query!r} | "
                f"filtre LLM: {structured_query.filter!r}"
            )
        except Exception as e:
            logging.warning(
                f"SelfQueryRetriever parsing échoué, fallback sémantique : {e}"
            )

        # --- Recherche vectorielle avec scores ---
        faiss_filter = search_kwargs.get("filter")
        total_docs = self._faiss_store.index.ntotal
        # Quand un filtre metadata est actif, scanner TOUS les documents : le filtre est appliqué
        # APRÈS le ranking vectoriel, donc un document pertinent par date mais loin sémantiquement
        # serait raté si on ne balaie pas l'index en entier.
        # fetch_k = nombre de candidats bruts que FAISS évalue AVANT d'appliquer le filtre callable.
        # Par défaut LangChain passe fetch_k=20, ce qui rend le filtre inopérant sur un grand index.
        # Avec un filtre actif, on force fetch_k=total_docs pour évaluer tous les documents.
        effective_fetch_k = total_docs if faiss_filter is not None else fetch_k
        try:
            docs_with_scores = (
                self._faiss_store.similarity_search_with_relevance_scores(
                    semantic_query,
                    k=effective_fetch_k,
                    filter=faiss_filter,
                    fetch_k=effective_fetch_k,
                )
            )
            if not docs_with_scores and faiss_filter is not None:
                logging.warning(
                    "Filtre structuré sans résultat, fallback recherche sémantique pure."
                )
                docs_with_scores = (
                    self._faiss_store.similarity_search_with_relevance_scores(
                        semantic_query, k=fetch_k
                    )
                )
        except Exception as e:
            logging.warning(
                f"Recherche avec filtre échouée, fallback sans filtre : {e}"
            )
            docs_with_scores = (
                self._faiss_store.similarity_search_with_relevance_scores(
                    query_text, k=fetch_k
                )
            )

        # --- Filtrage par min_score et formatage ---
        results = []

        for doc, score in docs_with_scores:
            similarity = float(score) * 100
            if min_score is not None and similarity < min_score * 100:
                logging.debug(
                    f"Document filtré (score {similarity:.2f}% < minimum {min_score * 100:.2f}%)"
                )
                continue
            results.append(
                {
                    "score": similarity,
                    "raw_score": float(score),
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        if len(results) > k:
            results = results[:k]

        logging.info(f"{len(results)} documents pertinents trouvés.")
        return results

    # ------------------------------------------------------------------
    # Compatibility property
    # ------------------------------------------------------------------

    @property
    def index(self):
        """Accès au FAISS index sous-jacent (pour Chat.py : vector_store.index.ntotal)."""
        if self._faiss_store is None:
            return None
        return self._faiss_store.index

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear_index(self) -> None:
        """Supprime l'index et les métadonnées du disque."""
        self._faiss_store = None
        self._llm = None
        for fname in ["index.faiss", "index.pkl", "index_metadata.json"]:
            path = os.path.join(self._vector_db_dir, fname)
            if os.path.exists(path):
                os.remove(path)
                logging.info(f"Fichier supprimé: {path}")
        logging.info("Index réinitialisé.")
