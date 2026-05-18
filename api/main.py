import datetime
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from core.config import EMBEDDING_MODEL
from core.container import AppContainer, build_container
from core.load_data import (
    build_openagenda_url,
    load_documents_from_url_paginated,
    save_documents_to_json,
)
from core.rag_pipeline import RAGPipeline
from core.vector_store import VectorStoreManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ──────────────────────────────────────────────────────────────────────────────
# État partagé de l'application (chargé une fois au démarrage)
# ──────────────────────────────────────────────────────────────────────────────
_state: dict = {}


def _get_container() -> AppContainer:
    container: AppContainer | None = _state.get("container")
    if container is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service non initialisé.",
        )
    return container


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Démarrage de l'API — chargement des ressources...")
    _state["container"] = build_container()
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
| `GET` | `/metadata` | Métadonnées de l'index vectoriel |
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
    k: int = Field(
        5,
        ge=1,
        le=20,
        description="Nombre de documents à récupérer depuis l'index FAISS.",
    )
    min_score: float = Field(
        0.5, ge=0.0, le=1.0, description="Score de similarité minimum (entre 0 et 1)."
    )
    model: str = Field(
        "mistral-large-latest",
        description="Identifiant du modèle Mistral.",
        examples=["mistral-small-latest", "mistral-large-latest"],
    )
    temperature: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Température de génération (0 = déterministe, 1 = créatif).",
    )
    format: Literal["text", "json"] = Field(
        "text",
        description=(
            "Format de la réponse : `text` (markdown) ou `json` "
            "(liste d'événements structurés avec title, description, location, "
            "city, start_date, end_date, price, is_free)."
        ),
    )


class SourceModel(BaseModel):
    text: str = Field(..., description="Extrait du document source.")
    score: float = Field(..., description="Score de similarité en pourcentage (0–100).")
    metadata: dict = Field(
        ..., description="Métadonnées du document (url, ville, dates, prix…)."
    )


class AskResponse(BaseModel):
    answer: Any = Field(..., description="Réponse générée : texte markdown (format=text) ou objet JSON (format=json).")
    mode: str = Field(
        ..., description="Mode de traitement utilisé : `RAG` ou `DIRECT`."
    )
    confidence: float = Field(
        ..., description="Indice de confiance de la classification (0–1)."
    )
    reason: str = Field(..., description="Explication de la classification.")
    sources: list[SourceModel] = Field(
        default=[],
        description="Documents sources utilisés (vide en mode DIRECT).",
    )
    model_used: str = Field(..., description="Modèle Mistral ayant généré la réponse.")


class RebuildRequest(BaseModel):
    cities: list[str] = Field(
        default=["Paris"],
        description="Liste de villes à filtrer (vide = toute la France).",
        examples=[["Paris", "Lyon"]],
    )
    begin_date: str | None = Field(
        None,
        description="Date de début des événements au format `YYYY-MM-DD`. Seuls le mois et l'année sont utilisés.",
        examples=["2025-01-01"],
    )
    embedding_model: str = Field(
        EMBEDDING_MODEL, description="Modèle d'embedding à utiliser."
    )
    max_records: int = Field(
        120,
        ge=20,
        le=1000,
        description="Nombre maximum d'événements à récupérer depuis l'API.",
    )


class RebuildResponse(BaseModel):
    status: str = Field(..., description="Résultat de l'opération : `success`.")
    message: str = Field(..., description="Message descriptif du résultat.")
    num_documents: int = Field(..., description="Nombre de documents indexés.")
    embedding_model: str = Field(..., description="Modèle d'embedding utilisé.")
    data_file: str = Field(
        ..., description="Chemin du fichier de données sauvegardé dans `data/`."
    )


class MetadataResponse(BaseModel):
    embedding_model: str = Field(
        ..., description="Modèle d'embedding utilisé pour l'index."
    )
    created_at: str = Field(
        ..., description="Date et heure de création de l'index (ISO 8601)."
    )
    num_documents: int = Field(..., description="Nombre de documents indexés.")
    cities: list[str] = Field(
        default=[], description="Liste des villes couvertes par l'index."
    )


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
    container: AppContainer | None = _state.get("container")
    vector_store = container.vector_store if container else None
    meta = vector_store.get_metadata() if vector_store else None
    return HealthResponse(
        status="ok",
        index_loaded=vector_store is not None and vector_store.index is not None,
        num_vectors=vector_store.index.ntotal if vector_store and vector_store.index else 0,
        index_metadata=meta,
        mistral_configured=container is not None and container.mistral_client is not None,
    )


@app.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Métadonnées de l'index vectoriel",
    tags=["Système"],
)
def get_metadata() -> MetadataResponse:
    """Retourne les métadonnées de l'index FAISS actuel (modèle, date de création, villes couvertes)."""
    container: AppContainer | None = _state.get("container")
    vector_store = container.vector_store if container else None
    meta = vector_store.get_metadata() if vector_store else None
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aucun index trouvé. Lancez d'abord une indexation via POST /rebuild.",
        )
    return MetadataResponse(
        embedding_model=meta.get("embedding_model", ""),
        created_at=meta.get("created_at", ""),
        num_documents=meta.get("num_documents", 0),
        cities=meta.get("cities", []),
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
    container = _get_container()
    if not container.mistral_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La clé API Mistral (MISTRAL_API_KEY) n'est pas configurée.",
        )

    as_json = request.format == "json"

    try:
        result = RAGPipeline(container.query_classifier, container.vector_store, container.mistral_client).run(
            question=request.question,
            k=request.k,
            min_score=request.min_score,
            model=request.model,
            temperature=request.temperature,
            as_json=as_json,
        )
    except Exception as exc:
        logging.error(f"Erreur API Mistral: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Erreur lors de la génération de la réponse : {exc}",
        )

    answer: Any = result.answer
    if as_json:
        try:
            answer = json.loads(result.answer)
        except Exception:
            logging.warning("Impossible de parser la réponse JSON du LLM.")

    return AskResponse(
        answer=answer,
        mode=result.mode,
        confidence=result.confidence,
        reason=result.reason,
        sources=[SourceModel(**s) for s in result.sources],
        model_used=result.model_used,
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

    1. Construction de l'URL OpenAgenda avec les filtres ville / date fournis.
    2. Récupération des événements avec **pagination automatique**.
    3. Suppression de l'ancien index FAISS.
    4. Génération des embeddings et construction du nouvel index.
    5. Sauvegarde des données brutes dans `data/`.
    6. Mise à jour de l'instance `VectorStoreManager` en mémoire.

    ⚠️ Cette opération peut prendre plusieurs minutes selon le nombre d'événements et le modèle d'embedding choisi.
    """
    # 1. Construction de l'URL
    begin_date = request.begin_date or (
        datetime.date.today().replace(year=datetime.date.today().year - 1).isoformat()
    )
    url = build_openagenda_url(request.cities, begin_date)
    logging.info(f"/rebuild — begin_date effectif: {begin_date}")
    logging.info(f"/rebuild — URL: {url[:120]}...")

    # 2. Récupération des données
    try:
        documents = load_documents_from_url_paginated(
            url, max_records=request.max_records
        )
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

    # 3. Suppression de l'ancien index
    container = _get_container()
    container.vector_store.clear_index()

    # 4. Construction du nouvel index
    try:
        new_store = VectorStoreManager(embedding_model=request.embedding_model)
        new_store.build_index(documents)
    except Exception as exc:
        logging.error(f"Erreur lors de la construction de l'index : {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la construction de l'index FAISS : {exc}",
        )

    # 5. Sauvegarde dans data/
    data_file = save_documents_to_json(documents)
    logging.info(f"Données sauvegardées : {data_file}")

    # 6. Mise à jour de l'état partagé
    container.vector_store = new_store

    meta = new_store.get_metadata() or {}
    logging.info(f"/rebuild — succès : {meta.get('num_documents')} docs")

    return RebuildResponse(
        status="success",
        message=f"{len(documents)} événements indexés avec succès.",
        num_documents=meta.get("num_documents", len(documents)),
        embedding_model=request.embedding_model,
        data_file=data_file,
    )


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
