# Puls-Events — RAG pour la recommandation d'événements culturels

Système de recommandation d'événements culturels basé sur une architecture **RAG** (Retrieval-Augmented Generation). Les données proviennent de l'API [OpenAgenda](https://openagenda.com), indexées dans un index vectoriel **FAISS** via des embeddings **Mistral**, et interrogées via un chatbot **Streamlit** ou une **API REST FastAPI**.

---

## Architecture

```
OpenAgenda API
      │
      ▼
  indexer.py          ← Chargement, découpage, embeddings, construction FAISS
      │
      ▼
 vector_db/           ← Index FAISS + métadonnées (persistés sur disque)
      │
  ┌───┴────────────────────────┐
  │                            │
  ▼                            ▼
Chat.py                     main.py
Streamlit chatbot           FastAPI REST API
  :8501                       :8000
  │                            │
  └──────────┬─────────────────┘
             │
             ▼
       FeedbackViewer.py
       Streamlit dashboard
         :8502
```

### Flux RAG

1. **Classification** — `QueryClassifier` détermine si la requête nécessite une recherche documentaire (RAG) ou une réponse directe du LLM.
2. **Recherche vectorielle** — `VectorStoreManager` (avec `SelfQueryRetriever` de LangChain) extrait des filtres structurés de la requête (ville, dates, gratuité) et effectue une recherche sémantique dans l'index FAISS.
3. **Génération** — `RAGPipeline` construit le prompt système avec le contexte récupéré et appelle **Mistral** pour générer la réponse.
4. **Persistance** — Chaque interaction est enregistrée dans une base SQLite via `database.py`. Le feedback utilisateur (👍/👎) y est également stocké.

### Modules clés

| Fichier | Rôle |
|---|---|
| `utils/config.py` | Constantes globales (chemins, modèles, paramètres) |
| `utils/vector_store.py` | Gestion de l'index FAISS + `SelfQueryRetriever` |
| `utils/query_classifier.py` | Classification RAG vs DIRECT (regex → mots-clés → LLM) |
| `utils/rag_pipeline.py` | Orchestration classify → search → prompt → LLM |
| `utils/container.py` | `AppContainer` : instanciation partagée des dépendances |
| `utils/database.py` | ORM SQLAlchemy sur SQLite, logs d'interactions et feedback |
| `utils/load_data.py` | Chargement et parsing des événements OpenAgenda |
| `utils/prompts.py` | Templates de prompts système (RAG, JSON, direct) |

---

## Prérequis

- Python ≥ 3.13
- [`uv`](https://docs.astral.sh/uv/) (gestionnaire de paquets)
- Clé API Mistral

---

## Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd rag-recommander-evenements-culturels

# Installer les dépendances
make install

# Configurer la clé API
cp .env.example .env   # puis éditer .env
```

Contenu du fichier `.env` :

```env
MISTRAL_API_KEY=<votre_clé>
```

---

## Utilisation

### 1. Indexer les données

```bash
# Depuis un fichier JSON local
make index input-file=data/evenements-publics-openagenda.json

# Depuis l'API OpenAgenda (avec une URL construite)
make index data-url="<url_openagenda>"
```

### 2. Lancer les services

```bash
make chat       # Chatbot Streamlit → http://localhost:8501
make api        # API REST FastAPI  → http://localhost:8000
make feedback   # Dashboard feedback → http://localhost:8502
```

---

## API REST

Documentation interactive disponible sur `http://localhost:8000/docs`.

### Endpoints

| Méthode | Chemin | Description |
|---|---|---|
| `GET` | `/health` | État de l'API et de l'index |
| `GET` | `/metadata` | Métadonnées de l'index FAISS |
| `POST` | `/ask` | Interroger le système RAG |
| `POST` | `/rebuild` | Reconstruire l'index depuis OpenAgenda |

### Exemple — `POST /ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels concerts sont prévus à Paris ce mois-ci ?",
    "k": 5,
    "min_score": 0.5,
    "format": "text"
  }'
```

Le champ `format` accepte :
- `"text"` — réponse en markdown (défaut)
- `"json"` — réponse structurée : `{"events": [{"title", "description", "location", "city", "start_date", "end_date", "price", "is_free"}]}`

---

## Docker

### Démarrage rapide

```bash
# Build et démarrage des 3 services
docker compose up --build

# Mode détaché
docker compose up -d --build
```

Les trois services démarrent dans des conteneurs séparés et partagent les mêmes données via des bind mounts :

| Service | Port | Conteneur |
|---|---|---|
| API FastAPI | `8000` | `api` |
| Chatbot Streamlit | `8501` | `chatbot` |
| Dashboard feedback | `8502` | `feedback` |

Les répertoires `vector_db/`, `database/` et `data/` sont montés depuis le host — les données persistent entre les redémarrages.

```bash
docker compose logs -f api      # suivre les logs de l'API
docker compose down             # arrêter tous les services
```

### Variable d'environnement `MISTRAL_API_KEY`

Le fichier `.env` est lu par `docker-compose.yml` via `env_file: .env`. Il n'est **jamais** copié dans l'image Docker (exclu via `.dockerignore`).

---

## Commandes utiles

```bash
make install     # Installer les dépendances
make index       # Indexer les données
make chat        # Lancer le chatbot
make api         # Lancer l'API REST
make feedback    # Lancer le dashboard feedback
make test        # Lancer les tests fonctionnels
make lint        # Vérifier le style (Ruff)
make lint-fix    # Corriger automatiquement le style
make reset       # Réinitialiser l'index et la base SQLite
make clean       # Supprimer les caches Python
make eval-build  # Construire l'index d'évaluation
make eval        # Lancer l'évaluation Ragas
```

---

## Évaluation (Ragas)

```bash
make eval-build   # Construit un index dédié depuis data/eval_events.json
make eval         # Lance l'évaluation et génère data/eval_report.json
```

Les métriques évaluées : fidélité, pertinence de la réponse, rappel contextuel, précision contextuelle.

---

## Structure du projet

```
.
├── Chat.py                  # Chatbot Streamlit
├── FeedbackViewer.py        # Dashboard feedback Streamlit
├── main.py                  # API REST FastAPI
├── indexer.py               # Script d'indexation
├── evaluate_rag.py          # Évaluation Ragas
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── Makefile
├── utils/
│   ├── config.py
│   ├── container.py
│   ├── database.py
│   ├── load_data.py
│   ├── prompts.py
│   ├── query_classifier.py
│   ├── rag_pipeline.py
│   └── vector_store.py
├── data/                    # Données OpenAgenda (JSON)
├── vector_db/               # Index FAISS + métadonnées
├── database/                # Base SQLite (interactions + feedback)
└── tests/
    └── test_api.py
```
