import datetime
import json
import logging
import os
import pickle
from typing import Callable

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai.client import Mistral

from utils.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_CHUNKS_FILE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    INDEX_METADATA_FILE,
    MISTRAL_API_KEY,
)


class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss.

    Supporte les modèles Mistral (via API) et les modèles HuggingFace
    (via sentence-transformers, installation optionnelle).
    """

    def __init__(
        self,
        embedding_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        saved_meta = self._read_metadata()

        self.embedding_model = (
            embedding_model
            or (saved_meta.get("embedding_model") if saved_meta else None)
            or EMBEDDING_MODEL
        )
        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else (saved_meta.get("chunk_size") if saved_meta else None) or CHUNK_SIZE
        )
        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else (saved_meta.get("chunk_overlap") if saved_meta else None) or CHUNK_OVERLAP
        )

        self.index: faiss.Index | None = None
        self.document_chunks: list[dict] = []
        self._hf_model = None
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None

        self._load_index_and_chunks()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _read_metadata(self) -> dict | None:
        if os.path.exists(INDEX_METADATA_FILE):
            try:
                with open(INDEX_METADATA_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def get_metadata(self) -> dict | None:
        return self._read_metadata()

    def _save_metadata(self, num_documents: int, num_chunks: int) -> None:
        os.makedirs(os.path.dirname(INDEX_METADATA_FILE), exist_ok=True)
        meta = {
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "created_at": datetime.datetime.now().isoformat(),
            "num_documents": num_documents,
            "num_chunks": num_chunks,
        }
        with open(INDEX_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # HuggingFace model helpers
    # ------------------------------------------------------------------

    def _is_hf_model(self) -> bool:
        return not self.embedding_model.startswith("mistral")

    def _get_hf_model(self):
        if self._hf_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Le package sentence-transformers n'est pas installé. "
                    "Exécutez: uv add sentence-transformers"
                ) from exc
            logging.info(f"Chargement du modèle HuggingFace: {self.embedding_model}")
            self._hf_model = SentenceTransformer(self.embedding_model)
        return self._hf_model

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _load_index_and_chunks(self) -> None:
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE):
            try:
                logging.info(f"Chargement de l'index Faiss depuis {FAISS_INDEX_FILE}.")
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                with open(DOCUMENT_CHUNKS_FILE, "rb") as f:
                    self.document_chunks = pickle.load(f)
                logging.info(
                    f"Index ({self.index.ntotal} vecteurs) et "
                    f"{len(self.document_chunks)} chunks chargés."
                )
            except Exception as e:
                logging.error(f"Erreur lors du chargement de l'index/chunks Faiss: {e}")
                self.index = None
                self.document_chunks = []
        else:
            self.index = None
            self.document_chunks = []
            logging.info("Aucun index existant trouvé. Initialisation d'un index vide.")

    def _save_index_and_chunks(self) -> None:
        if self.index is None or not self.document_chunks:
            logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return
        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)
        try:
            logging.info(f"Sauvegarde de l'index Faiss dans {FAISS_INDEX_FILE}...")
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logging.info(f"Sauvegarde des chunks dans {DOCUMENT_CHUNKS_FILE}...")
            with open(DOCUMENT_CHUNKS_FILE, "wb") as f:
                pickle.dump(self.document_chunks, f)
            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'index/chunks: {e}")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_index(
        self,
        documents: list[Document],
        progress_callback: Callable[[str], None] = None,
    ) -> None:
        """Construit l'index Faiss à partir des documents."""

        def _progress(msg: str) -> None:
            logging.info(msg)
            if progress_callback:
                progress_callback(msg)

        if not documents:
            logging.warning("Aucun document fourni pour la construction de l'index.")
            return

        _progress(f"Découpage des documents en chunks (taille={self.chunk_size}, overlap={self.chunk_overlap})...")
        self.document_chunks = self._split_documents_to_chunks(documents)
        if not self.document_chunks:
            logging.warning("Aucun chunk généré à partir des documents fournis.")
            return

        _progress(f"Génération des embeddings pour {len(self.document_chunks)} chunks (modèle: {self.embedding_model})...")
        embeddings = self._generate_embeddings(self.document_chunks, progress_callback)
        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            logging.error("Échec de la génération des embeddings. L'index ne sera pas construit.")
            self.document_chunks = []
            self.index = None
            for path in [FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE]:
                if os.path.exists(path):
                    os.remove(path)
            return

        _progress("Construction de l'index FAISS (IndexFlatIP, similarité cosinus)...")
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logging.info(f"Index Faiss créé avec {self.index.ntotal} vecteurs.")

        _progress("Sauvegarde de l'index et des chunks sur le disque...")
        self._save_index_and_chunks()
        self._save_metadata(num_documents=len(documents), num_chunks=len(self.document_chunks))
        _progress(f"Index sauvegardé — {len(documents)} documents, {len(self.document_chunks)} chunks.")

    def _split_documents_to_chunks(self, documents: list[Document]) -> list[dict]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        all_chunks = []
        for doc_counter, doc in enumerate(documents):
            chunks = text_splitter.split_documents([doc])
            logging.info(
                f"  Document '{doc.metadata.get('uid', 'N/A')}' découpé en {len(chunks)} chunks."
            )
            for i, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "id": f"doc{doc_counter}_{i}",
                        "text": chunk.page_content,
                        "metadata": {
                            **chunk.metadata,
                            "chunk_id_in_doc": i,
                            "start_index": chunk.metadata.get("start_index", -1),
                        },
                    }
                )
        return all_chunks

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def _generate_embeddings(
        self,
        chunks: list[dict],
        progress_callback: Callable[[str], None] = None,
    ) -> np.ndarray | None:
        if self._is_hf_model():
            return self._generate_embeddings_hf(chunks)
        return self._generate_embeddings_mistral(chunks, progress_callback)

    def _generate_embeddings_hf(self, chunks: list[dict]) -> np.ndarray | None:
        try:
            model = self._get_hf_model()
            texts = [chunk["text"] for chunk in chunks]
            logging.info(f"Génération HuggingFace de {len(texts)} embeddings...")
            embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.astype("float32")
        except Exception as e:
            logging.error(f"Erreur lors de la génération HuggingFace: {e}")
            raise

    def _generate_embeddings_mistral(
        self,
        chunks: list[dict],
        progress_callback: Callable[[str], None] = None,
    ) -> np.ndarray | None:
        if not MISTRAL_API_KEY or not self.mistral_client:
            logging.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
            return None
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        all_embeddings = []
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_chunks = chunks[i : i + EMBEDDING_BATCH_SIZE]
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]

            msg = f"Lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)..."
            logging.info(msg)
            if progress_callback:
                progress_callback(msg)

            try:
                response = self.mistral_client.embeddings.create(
                    model=self.embedding_model,
                    inputs=texts_to_embed,
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logging.error(f"Erreur inattendue lors de la génération d'embeddings (lot {batch_num}): {e}")
                if all_embeddings:
                    dim = len(all_embeddings[0])
                    logging.warning(f"Ajout de {len(texts_to_embed)} vecteurs nuls (dim={dim}).")
                    all_embeddings.extend([np.zeros(dim, dtype="float32")] * len(texts_to_embed))
                else:
                    continue

        if not all_embeddings:
            logging.error("Aucun embedding généré avec succès.")
            return None

        embeddings_array = np.array(all_embeddings, dtype="float32")
        logging.info(f"Génération terminée. Forme finale: {embeddings_array.shape}")
        return embeddings_array

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        k: int = 5,
        min_score: float = None,
    ) -> list[dict]:
        """Recherche les k chunks les plus pertinents pour une requête."""
        if self.index is None or not self.document_chunks:
            logging.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []

        try:
            if self._is_hf_model():
                model = self._get_hf_model()
                query_embedding = model.encode(
                    [query_text], convert_to_numpy=True
                ).astype("float32")
            else:
                if not self.mistral_client:
                    logging.error("Recherche impossible: MISTRAL_API_KEY manquante.")
                    return []
                response = self.mistral_client.embeddings.create(
                    model=self.embedding_model,
                    inputs=[query_text],
                )
                query_embedding = np.array([response.data[0].embedding]).astype("float32")

            faiss.normalize_L2(query_embedding)
            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            results = []
            if indices.size > 0:
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.document_chunks):
                        chunk = self.document_chunks[idx]
                        raw_score = float(scores[0][i])
                        similarity = raw_score * 100
                        min_score_percent = min_score * 100 if min_score is not None else 0
                        if min_score is not None and similarity < min_score_percent:
                            logging.debug(
                                f"Document filtré (score {similarity:.2f}% < minimum {min_score_percent:.2f}%)"
                            )
                            continue
                        results.append(
                            {
                                "score": similarity,
                                "raw_score": raw_score,
                                "text": chunk["text"],
                                "metadata": chunk["metadata"],
                            }
                        )
                    else:
                        logging.warning(
                            f"Index Faiss {idx} hors limites "
                            f"(taille des chunks: {len(self.document_chunks)})."
                        )

            results.sort(key=lambda x: x["score"], reverse=True)
            if len(results) > k:
                results = results[:k]

            logging.info(f"{len(results)} chunks pertinents trouvés.")
            return results

        except Exception as e:
            logging.error(f"Erreur inattendue lors de la recherche: {e}")
            return []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear_index(self) -> None:
        """Supprime l'index, les chunks et les métadonnées du disque."""
        self.index = None
        self.document_chunks = []
        self._hf_model = None
        for path in [FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, INDEX_METADATA_FILE]:
            if os.path.exists(path):
                os.remove(path)
                logging.info(f"Fichier supprimé: {path}")
        logging.info("Index et chunks réinitialisés.")
