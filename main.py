import datetime
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, status
from mistralai.client import Mistral
from pydantic import BaseModel, Field

from utils.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_CHUNKS_FILE,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    INDEX_METADATA_FILE,
    MISTRAL_API_KEY,
)
from utils.load_data import build_openagenda_url, load_documents_from_url_paginated, save_documents_to_json
from utils.prompts import direct_system_prompt, rag_no_results_system_prompt, rag_system_prompt
from utils.query_classifier import QueryClassifier
from utils.query_utils import expand_temporal_query
from utils.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# État partagé de l'application (chargé une fois au démarrage)
# ──────────────────────────────────────────────────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Démarrage de l'API — chargement des ressources...")
    _state["vector_store"] = VectorStoreManager()
    _state["mistral_client"] = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None
    _state["query_classifier"] = QueryClassifier()
    logging.info("Ressources chargées.")
    yield
    _state.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Application FastAPI
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Puls-Events RAG API",
    description="""
API REST pour le système RAG de recommandation d'événements culturels **Puls-Events**.

## Endpoints principaux

| Méthode | Chemin | Description |
|---------|--------|-------------|
| `POST` | `/ask` | Pose une question au système RAG |
| `POST` | `/rebuild` | Reconstruit la base vectorielle FAISS |
| `GET` | `/health` | État de santé de l'API |

## Flux RAG

1. La requête est **classifiée** (RAG vs réponse directe).
2. En mode RAG, les documents les plus proches sont récupérés depuis l'index **FAISS**.
3. Une réponse est générée via **Mistral** en utilisant les documents comme contexte.
""",
    version="1.0.0",
    contact={"name": "Puls-Events"},
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────────────
# Modèles Pydantic (requêtes / réponses)
# ──────────────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="La question posée par l'utilisateur.",
        examples=["Quels concerts sont prévus à Paris ce mois-ci ?"],
    )
    k: int = Field(5, ge=1, le=20, description="Nombre de documents à récupérer depuis l'index FAISS.")
    min_score: float = Field(0.75, ge=0.0, le=1.0, description="Score de similarité minimum (entre 0 et 1).")
    model: str = Field(
        "mistral-large-latest",
        description="Identifiant du modèle Mistral.",
        examples=["mistral-small-latest", "mistral-large-latest"],
    )


class SourceModel(BaseModel):
    text: str = Field(..., description="Extrait du document source.")
    score: float = Field(..., description="Score de similarité en pourcentage (0–100).")
    metadata: dict = Field(..., description="Métadonnées du document (url, ville, dates, prix…).")


class AskResponse(BaseModel):
    answer: str = Field(..., description="Réponse générée par le modèle Mistral.")
    mode: str = Field(..., description="Mode de traitement utilisé : `RAG` ou `DIRECT`.")
    confidence: float = Field(..., description="Indice de confiance de la classification (0–1).")
    reason: str = Field(..., description="Explication de la classification.")
    sources: list[SourceModel] = Field(
        default=[],
        description="Documents sources utilisés (vide en mode DIRECT).",
    )
    model_used: str = Field(..., description="Modèle Mistral ayant généré la réponse.")


class RebuildRequest(BaseModel):
    cities: list[str] = Field(
        default=[],
        description="Liste de villes à filtrer (vide = toute la France).",
        examples=[["Paris", "Lyon"]],
    )
    begin_date: str | None = Field(
        None,
        description="Date de début des événements au format `YYYY-MM-DD`. Seuls le mois et l'année sont utilisés.",
        examples=["2025-01-01"],
    )
    embedding_model: str = Field(EMBEDDING_MODEL, description="Modèle d'embedding à utiliser.")
    chunk_size: int = Field(CHUNK_SIZE, ge=100, le=10000, description="Taille des chunks en caractères.")
    chunk_overlap: int = Field(CHUNK_OVERLAP, ge=0, le=2000, description="Chevauchement des chunks en caractères.")
    max_records: int = Field(120, ge=20, le=500, description="Nombre maximum d'événements à récupérer depuis l'API.")


class RebuildResponse(BaseModel):
    status: str = Field(..., description="Résultat de l'opération : `success`.")
    message: str = Field(..., description="Message descriptif du résultat.")
    num_documents: int = Field(..., description="Nombre de documents indexés.")
    num_chunks: int = Field(..., description="Nombre de chunks créés.")
    embedding_model: str = Field(..., description="Modèle d'embedding utilisé.")
    data_file: str = Field(..., description="Chemin du fichier de données sauvegardé dans `data/`.")


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    num_vectors: int
    index_metadata: dict | None
    mistral_configured: bool


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="État de santé de l'API",
    tags=["Système"],
)
def health() -> HealthResponse:
    """Retourne l'état de l'API, le nombre de vecteurs indexés et les métadonnées de l'index."""
    vector_store: VectorStoreManager | None = _state.get("vector_store")
    meta = vector_store.get_metadata() if vector_store else None
    return HealthResponse(
        status="ok",
        index_loaded=vector_store is not None and vector_store.index is not None,
        num_vectors=vector_store.index.ntotal if vector_store and vector_store.index else 0,
        index_metadata=meta,
        mistral_configured=_state.get("mistral_client") is not None,
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Poser une question au système RAG",
    tags=["RAG"],
)
def ask(request: AskRequest) -> AskResponse:
    """
    Pose une question au système RAG et retourne une réponse générée par Mistral.

    **Flux de traitement :**

    1. La requête est **classifiée** (RAG nécessaire ou réponse directe).
    2. En mode **RAG** : recherche sémantique dans l'index FAISS, construction du contexte.
    3. Appel à **Mistral** avec le prompt système + contexte des documents.
    4. Retour de la réponse, des sources et des métadonnées de classification.
    """
    mistral_client: Mistral | None = _state.get("mistral_client")
    vector_store: VectorStoreManager = _state.get("vector_store")
    query_classifier: QueryClassifier = _state.get("query_classifier")

    if not mistral_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La clé API Mistral (MISTRAL_API_KEY) n'est pas configurée.",
        )

    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_month = now.strftime("%B %Y")

    # 1. Classification
    needs_rag, confidence, reason = query_classifier.needs_rag(request.question)
    mode = "RAG" if needs_rag else "DIRECT"
    logging.info(f"/ask — mode={mode} confiance={confidence:.2f} raison='{reason}'")

    # 2. Recherche vectorielle (mode RAG uniquement)
    sources: list[SourceModel] = []
    if needs_rag:
        search_query = expand_temporal_query(request.question, today=datetime.date.today())
        if search_query != request.question:
            logging.info(f"/ask — requête augmentée : {search_query!r}")
        retrieved = vector_store.search(search_query, k=request.k, min_score=request.min_score)
        sources = [
            SourceModel(text=doc["text"], score=doc["score"], metadata=doc["metadata"])
            for doc in retrieved
        ]

    # 3. Construction du prompt système
    if needs_rag and sources:
        context_str = "\n\n---\n\n".join(
            f"Source: {s.metadata.get('source', 'Inconnue')} (Score: {s.score:.2f}%)\nContenu: {s.text}"
            for s in sources
        )
        system_prompt = rag_system_prompt(context_str, current_date, current_month)
    elif needs_rag:
        system_prompt = rag_no_results_system_prompt(current_date, current_month)
    else:
        system_prompt = direct_system_prompt(current_date, current_month)

    # 4. Génération de la réponse
    try:
        chat_response = mistral_client.chat.complete(
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question},
            ],
            temperature=0.1,
        )
        answer = chat_response.choices[0].message.content
    except Exception as exc:
        logging.error(f"Erreur API Mistral: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Erreur lors de la génération de la réponse : {exc}",
        )

    return AskResponse(
        answer=answer,
        mode=mode,
        confidence=confidence,
        reason=reason,
        sources=sources,
        model_used=request.model,
    )


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    summary="Reconstruire la base vectorielle FAISS",
    tags=["Administration"],
    status_code=status.HTTP_200_OK,
)
def rebuild(request: RebuildRequest) -> RebuildResponse:
    """
    Reconstruit la base vectorielle FAISS à partir des données OpenAgenda.

    **Étapes exécutées :**

    1. Validation des paramètres (chunk_overlap < chunk_size).
    2. Construction de l'URL OpenAgenda avec les filtres ville / date fournis.
    3. Récupération des événements avec **pagination automatique**.
    4. Sauvegarde des données brutes dans `data/`.
    5. Suppression de l'ancien index FAISS.
    6. Génération des embeddings et construction du nouvel index.
    7. Mise à jour de l'instance `VectorStoreManager` en mémoire.

    ⚠️ Cette opération peut prendre plusieurs minutes selon le nombre d'événements et le modèle d'embedding choisi.
    """
    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="chunk_overlap doit être strictement inférieur à chunk_size.",
        )

    # 1. Construction de l'URL
    url = build_openagenda_url(request.cities, request.begin_date)
    logging.info(f"/rebuild — URL: {url[:120]}...")

    # 2. Récupération des données
    try:
        documents = load_documents_from_url_paginated(url, max_records=request.max_records)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Erreur lors de la récupération des données OpenAgenda : {exc}",
        )

    if not documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aucun événement trouvé pour les critères fournis. Essayez d'autres filtres.",
        )

    # 3. Sauvegarde dans data/
    data_file = save_documents_to_json(documents)
    logging.info(f"Données sauvegardées : {data_file}")

    # 4. Suppression de l'ancien index
    for path in [FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, INDEX_METADATA_FILE]:
        if os.path.exists(path):
            os.remove(path)
            logging.info(f"Supprimé : {path}")

    # 5. Construction du nouvel index
    try:
        new_store = VectorStoreManager(
            embedding_model=request.embedding_model,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        new_store.build_index(documents)
    except Exception as exc:
        logging.error(f"Erreur lors de la construction de l'index : {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la construction de l'index FAISS : {exc}",
        )

    # 6. Mise à jour de l'état partagé
    _state["vector_store"] = new_store

    meta = new_store.get_metadata() or {}
    logging.info(f"/rebuild — succès : {meta.get('num_documents')} docs, {meta.get('num_chunks')} chunks")

    return RebuildResponse(
        status="success",
        message=f"{len(documents)} événements indexés avec succès.",
        num_documents=meta.get("num_documents", len(documents)),
        num_chunks=meta.get("num_chunks", 0),
        embedding_model=request.embedding_model,
        data_file=data_file,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
