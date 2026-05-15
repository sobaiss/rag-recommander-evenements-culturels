import datetime
import json
import logging
import os
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def clean_html(html_content):
    if not html_content:
        return ""
    # Utilisation de BeautifulSoup pour extraire uniquement le texte
    soup = BeautifulSoup(html_content, "html.parser")
    # On peut aussi ajouter un espace entre les paragraphes pour garder la lisibilité
    return soup.get_text(separator=" ", strip=True)


def extract_metadata(record: dict) -> dict:
    """Extrait les métadonnées d'un enregistrement JSON."""
    metadata = {}

    # On extrait les infos utiles pour le filtrage et l'affichage
    metadata["uid"] = record.get("uid")
    metadata["city"] = record.get("location_city", "")
    metadata["region"] = record.get("location_region", "")
    metadata["start_date"] = str(record.get("firstdate_begin", ""))[:10]
    metadata["end_date"] = str(record.get("lastdate_begin"))[:10]
    metadata["price"] = record.get("conditions_fr")
    keywords = record.get("keywords_fr", "")
    metadata["keywords"] = (
        ", ".join(keywords) if isinstance(keywords, list) else keywords
    )
    # Nettoyage optionnel du prix pour un filtrage futur
    conditions = str(record.get("conditions_fr", "")).lower()
    metadata["is_free"] = "tarif" not in conditions
    metadata["source"] = record.get("canonicalurl", "")

    # Le champ registration peut être une chaîne JSON ou déjà un dict
    registration = record.get("registration", "")
    if registration:
        try:
            # Si c'est déjà un dict, pas besoin de le parser
            if isinstance(registration, dict):
                registration_data = registration
            else:
                registration_data = json.loads(registration)
            if isinstance(registration_data, list) and len(registration_data) > 0:
                metadata["registration_link"] = registration_data[0].get("value", "")
            else:
                metadata["registration"] = ""
        except (json.JSONDecodeError, TypeError):
            metadata["registration_link"] = ""

    return metadata


def create_document_from_record(record: dict) -> Document:
    """Crée un Document langchain à partir d'un enregistrement JSON."""
    # La description contient des balises HTML, ils faut les supprimier pour éviter d'avoir du bruit dans les embeddings
    description = clean_html(record.get("longdescription_fr", ""))
    metadata = extract_metadata(record)

    page_content = (
        f"NOM DE L'ÉVÉNEMENT: {record.get('title_fr', '')}\n"
        f"DATE : du {metadata['start_date']} au {metadata['end_date']}\n"
        f"LIEU: {record.get('location_name', '')} ({record.get('location_city', '')})\n"
        f"TARIF: {'Gratuit' if metadata['is_free'] else metadata['price']}\n"
        f"DESCRIPTION: {description}\n"
    )

    return Document(page_content=page_content, metadata=metadata)


_OPENAGENDA_BASE = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets"
    "/evenements-publics-openagenda/records"
)


def build_openagenda_url(
    cities: list[str],
    begin_date: "datetime.date | str | None" = None,
    limit: int = 100,
    offset: int = 0,
) -> str:
    """Construit l'URL OpenAgenda v2.1 (/records) avec les filtres ville et date.

    Args:
        cities: Liste de villes à filtrer (vide = toute la France).
        begin_date: Date de début sous forme de ``datetime.date`` ou chaîne
            ``YYYY-MM-DD``. Seuls le mois et l'année sont utilisés.
        limit: Nombre de résultats par page (pagination via ``offset``).
        offset: Position de départ (omis de l'URL si égal à 1).
    """
    params: list[tuple] = [("limit", limit)]
    if offset > 0:
        params.append(("offset", offset))
    for city in cities:
        params.append(("refine", f'location_city:"{city}"'))
    if begin_date:
        if isinstance(begin_date, datetime.date):
            refine_date = begin_date.strftime("%Y/%m")
        else:
            parts = begin_date.split("-")
            refine_date = parts[0] if len(parts) == 1 else f"{parts[0]}/{parts[1]}"
        params.append(("refine", f'firstdate_begin:"{refine_date}"'))

    return _OPENAGENDA_BASE + "?" + urlencode(params)


def load_documents_from_file(input_file: str) -> list[Document]:
    """Charge les documents depuis un fichier local (JSON ou CSV)."""
    if input_file.endswith(".json"):
        return _load_from_json_file(input_file)
    else:
        # Pour CSV, utiliser le loader standard
        loader = CSVLoader(file_path=input_file)
        return loader.load()


def load_documents_from_url(url: str) -> list[Document]:
    """
    Charge les documents depuis une URL externe (API).

    Args:
        url (str): URL de l'API retournant du JSON

    Returns:
        list: Liste de Documents LangChain
    """
    logging.info(f"Chargement des données depuis l'URL: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Transformer chaque enregistrement en Document
        documents: list[Document] = []
        if isinstance(data, list):
            for item in data:
                doc = create_document_from_record(item)
                documents.append(doc)
        elif isinstance(data, dict):
            # Gérer différents formats de réponse API
            if "results" in data:
                for item in data["results"]:
                    doc = create_document_from_record(item)
                    documents.append(doc)
            else:
                logging.warning(
                    "Format de données inattendu: ni liste ni dict avec 'results'."
                )

        logging.info(f"Chargé {len(documents)} documents depuis l'URL")
        return documents

    except requests.RequestException as e:
        logging.error(f"Erreur lors du chargement depuis l'URL: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Erreur lors du parsing du JSON depuis l'URL: {e}")
        raise


def _load_from_json_file(input_file: str) -> list[Document]:
    """Charge les documents depuis un fichier JSON local."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents: list[Document] = []
    if isinstance(data, list):
        for item in data:
            doc = create_document_from_record(item)
            documents.append(doc)
    elif isinstance(data, dict) and "results" in data:
        for item in data["results"]:
            doc = create_document_from_record(item)
            documents.append(doc)

    return documents


def load_documents_from_url_paginated(
    base_url: str, max_records: int = 120
) -> list[Document]:
    """Charge les documents depuis l'API OpenAgenda avec pagination automatique.

    Args:
        base_url: URL de base (déjà construite avec les filtres région/date).
        max_records: Nombre maximum de documents à récupérer.

    Returns:
        Liste de Documents LangChain.
    """
    documents: list[Document] = []
    limit_per_page = 100
    offset = 0
    total_count: int | None = None

    parsed = urlparse(base_url)
    base_params = parse_qs(parsed.query, keep_blank_values=True)

    while len(documents) < max_records:
        # Reconstruire l'URL avec offset/limit mis à jour (API v2.1)
        page_params = {k: (v[0] if len(v) == 1 else v) for k, v in base_params.items()}
        page_params["limit"] = str(limit_per_page)
        if offset != 0:
            page_params["offset"] = str(offset)
        url = urlunparse(parsed._replace(query=urlencode(page_params, doseq=True)))

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logging.error(f"Erreur lors du chargement (page offset={offset}): {e}")
            raise

        if total_count is None:
            total_count = data.get("total_count", 0)
            logging.info(f"Total disponible sur l'API: {total_count} enregistrements")

        records = data.get("results", [])
        if not records:
            break

        for record in records:
            doc = create_document_from_record(record)
            documents.append(doc)
            if len(documents) >= max_records:
                break

        offset += len(records)
        if total_count is not None and offset >= total_count:
            break

    logging.info(
        f"Chargement terminé: {len(documents)} documents récupérés "
        f"(sur {total_count} disponibles)."
    )
    return documents


def save_documents_to_json(documents: list, folder: str = "data") -> str:
    """Sauvegarde les documents dans un fichier JSON horodaté.

    Args:
        documents: Liste de Documents LangChain.
        folder: Dossier de destination.

    Returns:
        Chemin absolu du fichier créé.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(folder, f"openagenda_{timestamp}.json")

    records = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logging.info(f"Données sauvegardées dans {filepath} ({len(documents)} documents).")
    return filepath
