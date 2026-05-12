import argparse
import logging

from utils.load_data import (
    load_documents_from_file,
    load_documents_from_url,
    save_documents_to_json,
)
from utils.vector_store import VectorStoreManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_indexing(
    input_file: str | None = None,
    data_url: str | None = None,
    vector_db_dir: str | None = None,
):
    """
    Fonction principale pour exécuter le processus d'indexation.
    Args:
        input_file (str): Chemin vers le fichier local à indexer.
        data_url (str): URL externe pour récupérer les données.
        vector_db_dir (str): Dossier de destination de l'index FAISS.
    """
    logging.info("Démarrage du processus d'indexation.")

    if data_url:
        documents = load_documents_from_url(data_url)
    elif input_file:
        logging.info(f"Chargement et parsing du fichier: {input_file}")
        documents = load_documents_from_file(input_file)
    else:
        raise ValueError("Veuillez spécifier --input-file ou --data-url")

    if not documents:
        logging.warning("Aucun document trouvé pour l'indexation.")
        return

    kwargs = {"vector_db_dir": vector_db_dir} if vector_db_dir else {}
    logging.info("Initialisation du gestionnaire de Vector Store...")
    vector_store = VectorStoreManager(**kwargs)

    logging.info("Construction de l'index Faiss (cela peut prendre du temps)...")
    vector_store.build_index(documents)

    logging.info("--- Processus d'indexation terminé avec succès ---")
    logging.info(f"Nombre de documents traités: {len(documents)}")
    if vector_store.index:
        logging.info(f"Nombre de chunks indexés: {vector_store.index.ntotal}")
        save_path = save_documents_to_json(documents)
        logging.info(f"✅ Fichier sauvegardé : `{save_path}`")
    else:
        logging.warning("L'index final n'a pas pu être créé ou est vide.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script d'indexation pour l'application RAG"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Chemin vers le fichier local à indexer (JSON ou CSV)",
    )
    parser.add_argument(
        "--data-url",
        type=str,
        default=None,
        help="URL externe (API) pour récupérer les données JSON à indexer",
    )
    parser.add_argument(
        "--vector-db-dir",
        type=str,
        default=None,
        help="Dossier de destination de l'index FAISS (défaut: vector_db/)",
    )
    args = parser.parse_args()

    if not args.input_file and not args.data_url:
        parser.error("Veuillez spécifier --input-file ou --data-url")

    run_indexing(
        input_file=args.input_file,
        data_url=args.data_url,
        vector_db_dir=args.vector_db_dir,
    )
