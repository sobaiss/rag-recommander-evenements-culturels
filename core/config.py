import os

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print(
        "⚠️ La variable d'environnement MISTRAL_API_KEY n'est pas définie dans le fichier .env."
    )

INPUT_DIR = "data"  # Dossier pour les données sources après extraction

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "vector_db")
FAISS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss_index.idx")
DOCUMENT_CHUNKS_FILE = os.path.join(VECTOR_DB_DIR, "document_chunks.pkl")
INDEX_METADATA_FILE = os.path.join(VECTOR_DB_DIR, "index_metadata.json")

EMBEDDING_BATCH_SIZE = 32  # Taille des lots pour l'API d'embedding

# --- Modèles Mistral ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")
EMBEDDING_MODEL_OLLAMA = "ollama:bge-m3"
CHAT_MODEL = "mistral-large-latest"  # Ou un autre modèle comme mistral-large-latest

MODEL_NAME = "mistral-large-latest"  # Ou un autre modèle comme mistral-large-latest

# --- Modèles d'embedding disponibles (libres et gratuits) ---
AVAILABLE_EMBEDDING_MODELS = {
    "mistral-embed": "Mistral Embed (via API Mistral — recommandé)",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "MiniLM Multilingue · local · ~117 MB",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "MPNet Multilingue · local · ~438 MB",
    "BAAI/bge-m3": "BGE-M3 Multilingue · local · ~2.2 GB",
    "ollama:nomic-embed-text": "Nomic Embed Text · Ollama · 2048 tokens",
    "ollama:bge-m3": "BGE-M3 Multilingue · Ollama · 8192 tokens · ~1.2 GB",
    "ollama:snowflake-arctic-embed2": "Snowflake Arctic Embed 2 · Ollama · 8192 tokens · ~670 MB",
}

# --- Configuration de la Recherche ---
SEARCH_K = 5  # Nombre de documents à récupérer par défaut

# --- Configuration de la Base de Données ---
DATABASE_DIR = "database"
DATABASE_FILE = os.path.join(DATABASE_DIR, "interactions.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"  # URL pour SQLAlchemy

# --- Configuration de l'Application ---
APP_TITLE = "Assistant recommandation des événements culturels"
COMPANY_NAME = "Puls-Events"  # Nom à personnaliser dans l'interface

# --- Métriques d'évaluation RAG ---
# Source unique pour les clés, seuils, labels et descriptions.
# Modifier ici pour répercuter les changements dans l'UI et dans evaluate_rag.py.
EVAL_METRICS: list[dict] = [
    {
        "key": "faithfulness",
        "label": "Fidélité",
        "description": "Réponses ancrées dans les sources ?",
        "threshold": 0.8,
    },
    {
        "key": "factual_correctness(mode=f1)",
        "label": "Exactitude factuelle",
        "description": "Faits factuellement corrects ?",
        "threshold": 0.6,
    },
    {
        "key": "llm_context_precision_with_reference",
        "label": "Précision contexte",
        "description": "Documents récupérés pertinents ?",
        "threshold": 0.8,
    },
    {
        "key": "context_recall",
        "label": "Rappel contexte",
        "description": "Tous les docs pertinents retrouvés ?",
        "threshold": 0.8,
    },
    {
        "key": "nv_accuracy",
        "label": "Précision réponse",
        "description": "La réponse est-elle correcte par rapport à la référence ?",
        "threshold": 0.7,
    },
]

THRESHOLDS: dict[str, float] = {m["key"]: m["threshold"] for m in EVAL_METRICS}
METRIC_LABELS: dict[str, str] = {m["key"]: m["label"] for m in EVAL_METRICS}
METRIC_DESCRIPTIONS: dict[str, str] = {m["key"]: m["description"] for m in EVAL_METRICS}


# --- Régions françaises disponibles pour le filtre OpenAgenda ---
FRENCH_CITIES = [
    "Paris",
    "Toulouse",
    "Marseille",
    "Genève",
    "Nantes",
    "Lille",
    "Bordeaux",
    "Rennes",
    "Orléans",
    "Issy-les-Moulineaux",
    "Roubaix",
    "Montreuil",
    "Arles",
    "Limoges",
    "Rouen",
    "Versailles",
    "Albi",
    "Lyon",
    "Villeneuve-d'Ascq",
    "Strasbourg",
    "Tourcoing",
    "Aix-en-Provence",
    "Meudon",
    "Nevers",
    "Châtellerault",
    "Avignon",
    "Beaugency",
    "Villeurbanne",
    "Cenon",
    "Martigues",
    "Berlin",
    "Le Mans",
    "Reims",
    "Bègles",
    "Nancy",
    "Angers",
    "Trappes",
    "Montpellier",
    "Tours",
    "Guérande",
    "Saint-Étienne",
    "Brest",
    "Metz",
    "Saint-Denis",
    "Grenoble",
    "Caen",
    "Dijon",
    "Poitiers",
    "Boulogne-Billancourt",
    "Alès",
    "Pantin",
    "Olivet",
    "Le Havre",
    "Saint-Dizier",
    "Mérignac",
    "Nice",
    "Wattrelos",
    "Nîmes",
    "Ramonville-Saint-Agne",
    "Mulhouse",
    "Quimper",
    "Massy",
    "Chambéry",
    "Perpignan",
    "Senlis",
    "Colomiers",
    "Lisieux",
    "Blagnac",
    "Besançon",
    "Clermont-Ferrand",
    "Erquy",
    "Saint-Germain-en-Laye",
    "La Rochelle",
    "Villenave-d'Ornon",
    "Concarneau",
    "Valence",
    "Guyancourt",
    "Amiens",
    "Bondy",
    "Troyes",
    "Cambrai",
    "Saint-Étienne-du-Rouvray",
    "Lormont",
    "Charleville-Mézières",
    "Saint-Médard-en-Jalles",
    "Les Lilas",
    "Annecy",
    "Gif-sur-Yvette",
    "Nanterre",
    "Romainville",
    "Castanet-Tolosan",
    "Fontainebleau",
    "Chartres",
    "Carbon-Blanc",
    "Ambérieu-en-Bugey",
    "Grande-Synthe",
    "Talence",
    "Saint-Brieuc",
    "Vieux",
    "Montigny-le-Bretonneux",
    "Balma",
    "Cornebarrieu",
    "Bourges",
    "Toulon",
    "Pessac",
    "Rambouillet",
    "Noisy-le-Sec",
    "Saint-Nazaire",
    "Élancourt",
    "Laval",
    "Vannes",
    "Pau",
    "Saint-Herblain",
    "Épinal",
    "Vitry-sur-Seine",
    "Châlons-en-Champagne",
    "Arcueil",
    "Floirac",
    "Cergy",
    "La Roche-sur-Yon",
    "Draguignan",
    "Bagnolet",
    "Orsay",
    "Saint-Pierre",
    "Le Haillan",
    "Louvres",
    "Angoulême",
    "Blois",
    "Libourne",
    "Lorient",
    "Menton",
    "Dunkerque",
    "Montaigu-Vendée",
    "Langres",
    "Rochefort",
    "Armentières",
    "Laon",
    "Carvin",
    "Montmorillon",
    "Aubervilliers",
    "La Ciotat",
    "Auterive",
    "Bayonne",
    "Jouy-en-Josas",
    "Chantepie",
    "Colmar",
    "Antibes",
    "Saint-Ouen-sur-Seine",
    "Arras",
    "Labège",
    "Ivry-sur-Seine",
    "Elbeuf",
    "Lens",
    "Narbonne",
    "Périgueux",
    "Grasse",
    "Rezé",
    "Orleans",
    "Niort",
    "Saint-Chamond",
    "La turballe",
    "Chaumont",
    "Calais",
    "Béziers",
    "Montauban",
    "Dole",
    "Amilly",
    "Fleury-les-Aubrais",
    "Évreux",
    "Pérols",
    "Brive-la-Gaillarde",
    "Chalon-sur-Saône",
    "Quimperlé",
    "Bobigny",
    "Sèvres",
    "Saint-Malo",
    "La baule",
    "Saclay",
    "Fontenay-sous-Bois",
    "Carcassonne",
    "Pontoise",
    "Cherbourg-en-Cotentin",
    "Tournefeuille",
    "Beauvais",
    "Roanne",
    "Saint-Quentin",
    "Douai",
    "Châtenay-Malabry",
    "Luxeuil-les-Bains",
    "Vitré",
    "Rodez",
    "Abbeville",
    "Boulogne-sur-Mer",
    "Sète",
    "Fougères",
    "Le croisic",
    "Eysines",
    "Royan",
    "Livarot-Pays-d'Auge",
    "Saint-Pierre-en-Auge",
]
