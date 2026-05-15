import argparse
import logging

from core.load_data import (
    build_openagenda_url,
    load_documents_from_file,
    load_documents_from_url,
    load_documents_from_url_paginated,
    save_documents_to_json,
)
from core.vector_store import VectorStoreManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _validate_begin_date(date_str: str) -> None:
    import re

    if not re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", date_str):
        raise ValueError(
            f"Format de date invalide : {date_str!r}. "
            "Formats acceptés : YYYY, YYYY-MM, YYYY-MM-DD"
        )


def run_indexing(
    input_file: str | None = None,
    data_url: str | None = None,
    vector_db_dir: str | None = None,
    embedding_model: str | None = None,
    locations: list[str] | None = None,
    begin_date: str | None = None,
):
    logging.info("Démarrage du processus d'indexation.")

    if data_url:
        documents = load_documents_from_url(data_url)
    elif input_file:
        logging.info(f"Chargement et parsing du fichier: {input_file}")
        documents = load_documents_from_file(input_file)
    elif locations or begin_date:
        if begin_date:
            _validate_begin_date(begin_date)
        url = build_openagenda_url(locations or [], begin_date)
        logging.info(f"Chargement paginé depuis OpenAgenda: {url}")
        documents = load_documents_from_url_paginated(url, max_records=1000)
    else:
        raise ValueError(
            "Veuillez spécifier --input-file, --data-url, ou --locations/--begin-date"
        )

    if not documents:
        logging.warning("Aucun document trouvé pour l'indexation.")
        return

    kwargs = {}
    if vector_db_dir:
        kwargs["vector_db_dir"] = vector_db_dir
    if embedding_model:
        kwargs["embedding_model"] = embedding_model
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
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Modèle d'embedding à utiliser (ex: mistral-embed, ollama:nomic-embed-text)",
    )
    parser.add_argument(
        "--locations",
        type=str,
        default=None,
        help="Villes à filtrer, séparées par des virgules (ex: Paris,Lyon,Toulouse)",
    )
    parser.add_argument(
        "--begin-date",
        type=str,
        default=None,
        help="Date de début au format YYYY, YYYY-MM ou YYYY-MM-DD",
    )
    args = parser.parse_args()

    locations = (
        [loc.strip() for loc in args.locations.split(",") if loc.strip()]
        if args.locations
        else None
    )

    if (
        not args.input_file
        and not args.data_url
        and not locations
        and not args.begin_date
    ):
        parser.error(
            "Veuillez spécifier --input-file, --data-url, ou --locations/--begin-date"
        )

    run_indexing(
        input_file=args.input_file,
        data_url=args.data_url,
        vector_db_dir=args.vector_db_dir,
        embedding_model=args.embedding_model,
        locations=locations,
        begin_date=args.begin_date,
    )
