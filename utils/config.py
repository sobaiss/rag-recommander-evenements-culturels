import os

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ La variable d'environnement MISTRAL_API_KEY n'est pas définie dans le fichier .env.")

INPUT_DIR = "data"                # Dossier pour les données sources après extraction

VECTOR_DB_DIR = "vector_db"         # Dossier pour stocker l'index Faiss et les chunks
FAISS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss_index.idx")
DOCUMENT_CHUNKS_FILE = os.path.join(VECTOR_DB_DIR, "document_chunks.pkl")

CHUNK_SIZE = 2000                   # Taille des chunks en *caractères* (vise ~512 tokens)
CHUNK_OVERLAP = 200                 # Chevauchement en *caractères*
EMBEDDING_BATCH_SIZE = 32  # Taille des lots pour l'API d'embedding

# --- Modèles Mistral ---
EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-small-latest"  # Ou un autre modèle comme mistral-large-latest

MODEL_NAME = "mistral-small-latest"  # Ou un autre modèle comme mistral-large-latest

# --- Configuration de la Recherche ---
SEARCH_K = 5  # Nombre de documents à récupérer par défaut

# --- Configuration de la Base de Données ---
DATABASE_DIR = "database"
DATABASE_FILE = os.path.join(DATABASE_DIR, "interactions.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"  # URL pour SQLAlchemy

# --- Configuration de l'Application ---
APP_TITLE = "Assistant recommandation des événements culturels"
COMPANY_NAME = "Puls-Events"  # Nom à personnaliser dans l'interface
