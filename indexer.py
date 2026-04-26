import argparse
import logging

from utils.vector_store import VectorStoreManager
from utils.load_data import load_documents_from_file, load_documents_from_url

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_indexing(input_file: str = None, data_url: str = None):
    """
    Fonction principale pour exécuter le processus d'indexation.
    Args:
        input_file (str): Chemin vers le fichier à indexer.
    """
    logging.info("Démarrage du processus d'indexation.")

    # Déterminer la source des données
    if data_url:
        logging.info(f"Chargement des données depuis l'URL: {data_url}")
        documents = load_documents_from_url(data_url)
    elif input_file:
        logging.info(f"Chargement et parsing du fichier: {input_file}")
        documents = load_documents_from_file(input_file)
    else:
        raise ValueError("Veuillez spécifier --input-file ou --data-url")

    if not documents:
        logging.warning("Aucun document trouvé pour l'indexation.")
        return

    # --- Étape 3: Création/Mise à jour de l'index Vectoriel ---
    logging.info("Initialisation du gestionnaire de Vector Store...")
    vector_store = VectorStoreManager()

    logging.info("Construction de l'index Faiss (cela peut prendre du temps)...")
    # Cette méthode va splitter, générer les embeddings, créer l'index et sauvegarder
    vector_store.build_index(documents)

    logging.info("--- Processus d'indexation terminé avec succès ---")
    logging.info(f"Nombre de documents traités: {len(documents)}")
    if vector_store.index:
        logging.info(f"Nombre de chunks indexés: {vector_store.index.ntotal}")
    else:
        logging.warning("L'index final n'a pas pu être créé ou est vide.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'indexation pour l'application RAG")
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Chemin vers le fichier local à indexer (JSON ou CSV)"
    )
    parser.add_argument(
        "--data-url",
        type=str,
        default=None,
        help="URL externe (API) pour récupérer les données JSON à indexer"
    )
    args = parser.parse_args()

    # Vérifier qu'au moins une source de données est fournie
    if not args.input_file and not args.data_url:
        parser.error("Veuillez spécifier --input-file ou --data-url")

    run_indexing(input_file=args.input_file, data_url=args.data_url)
