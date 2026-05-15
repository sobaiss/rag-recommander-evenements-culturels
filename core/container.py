import logging
from dataclasses import dataclass

from mistralai.client import Mistral

from core.config import EMBEDDING_MODEL, MISTRAL_API_KEY, VECTOR_DB_DIR
from core.query_classifier import QueryClassifier
from core.vector_store import VectorStoreManager


@dataclass
class AppContainer:
    vector_store: VectorStoreManager
    mistral_client: Mistral | None
    query_classifier: QueryClassifier

    @property
    def has_mistral(self) -> bool:
        return self.mistral_client is not None


def build_container() -> AppContainer:
    logging.info("Initialisation du conteneur applicatif...")
    return AppContainer(
        vector_store=VectorStoreManager(
            embedding_model=EMBEDDING_MODEL,
            vector_db_dir=VECTOR_DB_DIR,
        ),
        mistral_client=Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None,
        query_classifier=QueryClassifier(),
    )
