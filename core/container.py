import logging
from dataclasses import dataclass

from mistralai.client import Mistral

from core.config import MISTRAL_API_KEY
from core.query_classifier import QueryClassifier
from core.vector_store import VectorStoreManager


@dataclass
class AppContainer:
    vector_store: VectorStoreManager
    mistral_client: Mistral | None
    query_classifier: QueryClassifier


def build_container() -> AppContainer:
    logging.info("Initialisation du conteneur applicatif...")
    return AppContainer(
        vector_store=VectorStoreManager(),
        mistral_client=Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None,
        query_classifier=QueryClassifier(),
    )
