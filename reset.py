import logging

from utils.database import reset_database
from utils.vector_store import VectorStoreManager


def reset_all():
    """Réinitialise la base de données et le vector store."""
    logging.info("Réinitialisation de la base de données et du vector store...")
    vector_store = VectorStoreManager()
    reset_database()
    vector_store.clear_index()
    logging.info("Réinitialisation terminée.")


if __name__ == "__main__":
    reset_all()
