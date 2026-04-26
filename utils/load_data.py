import json
import logging
import requests

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    metadata["url"] = record.get("canonicalurl")
    metadata["ville"] = record.get("location_city")
    metadata["date_debut"] = record.get("firstdate_begin")
    metadata["date_fin"] = record.get("lastdate_begin")
    metadata["prix"] = record.get("conditions_fr")
    metadata["image"] = record.get("thumbnail")
    metadata["status"] = record.get("status")
    # Nettoyage optionnel du prix pour un filtrage futur
    conditions = str(record.get("conditions_fr", "")).lower()
    metadata["est_gratuit"] = "gratuit" in conditions or "07,00" not in conditions
    
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
                metadata["lien_inscription"] = registration_data[0].get("value", "")
            else:
                metadata["lien_inscription"] = ""
        except (json.JSONDecodeError, TypeError):
            metadata["lien_inscription"] = ""

    # Le champ location_coordinates peut être une chaîne JSON ou déjà un dict
    coordinates = record.get("location_coordinates", "")
    if coordinates:
        try:
            # Si c'est déjà un dict, pas besoin de le parser
            if isinstance(coordinates, dict):
                coord_data = coordinates
            else:
                coord_data = json.loads(coordinates)
            metadata["latitude"] = coord_data.get("lat")
            metadata["longitude"] = coord_data.get("lon")
        except (json.JSONDecodeError, TypeError):
            pass

    return metadata

def create_document_from_record(record: dict) -> Document:
    """Crée un Document langchain à partir d'un enregistrement JSON."""
    # La description contient des balises HTML, ils faut les supprimier pour éviter d'avoir du bruit dans les embeddings
    description = clean_html(record.get("longdescription_fr", ""))

    page_content = (
        f"TITRE: {record.get('title_fr', '')}\n"
        f"LIEU: {record.get('location_name', '')} ({record.get('location_city', '')})\n"
        f"DESCRIPTION: {description}\n"
        f"TAGS: {record.get('keywords_fr', '')}"
    )

    metadata = extract_metadata(record)
    metadata["source"] = record.get("canonicalurl", "")

    return Document(page_content=page_content, metadata=metadata)

def load_documents_from_file(input_file: str):
    """Charge les documents depuis un fichier local (JSON ou CSV)."""
    if input_file.endswith(".json"):
        return _load_from_json_file(input_file)
    else:
        # Pour CSV, utiliser le loader standard
        loader = CSVLoader(file_path=input_file)
        return loader.load()


def load_documents_from_url(url: str) -> list:
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
        documents = []
        if isinstance(data, list):
            for item in data:
                doc = create_document_from_record(item)
                documents.append(doc)
        elif isinstance(data, dict):
            # Gérer différents formats de réponse API
            if "records" in data:
                for item in data["records"]:
                    if "fields" in item:
                        doc = create_document_from_record(item["fields"])
                        documents.append(doc)
            else:
                logging.warning("Format de données inattendu: ni liste ni dict avec 'records'.")
        
        logging.info(f"Chargé {len(documents)} documents depuis l'URL")
        return documents
        
    except requests.RequestException as e:
        logging.error(f"Erreur lors du chargement depuis l'URL: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Erreur lors du parsing du JSON depuis l'URL: {e}")
        raise


def _load_from_json_file(input_file: str) -> list:
    """Charge les documents depuis un fichier JSON local."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    if isinstance(data, list):
        for item in data:
            doc = create_document_from_record(item)
            documents.append(doc)
    elif isinstance(data, dict) and "results" in data:
        for item in data["results"]:
            doc = create_document_from_record(item)
            documents.append(doc)

    return documents
