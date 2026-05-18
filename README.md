# Puls-Events — RAG pour la recommandation d'événements culturels

Système de recommandation d'événements culturels basé sur une architecture **RAG** (Retrieval-Augmented Generation). Les données proviennent de l'API [OpenAgenda](https://openagenda.com), indexées dans un index vectoriel **FAISS** via des embeddings **Mistral**, et interrogées via un chatbot **Streamlit** ou une **API REST FastAPI**.

---

## Architecture

```
OpenAgenda API
      │
      ▼
  scripts/indexer.py    ← Chargement, embeddings, construction FAISS (LangChain)
      │
      ▼
 vector_db/             ← Index FAISS + métadonnées (persistés sur disque)
      │
  ┌───┴────────────────────────┐
  │                            │
  ▼                            ▼
app/Chat.py               api/main.py
Streamlit chatbot         FastAPI REST API
  :8501                     :8000
  │                            │
  └──────────┬─────────────────┘
             │
             ▼
       app/FeedbackViewer.py
       Streamlit dashboard
         :8502
```

### Flux RAG

1. **Classification** — `QueryClassifier` détermine si la requête nécessite une recherche documentaire (RAG) ou une réponse directe du LLM (regex → mots-clés).
2. **Recherche vectorielle** — `VectorStoreManager` utilise le `SelfQueryRetriever` de LangChain pour extraire des filtres structurés de la requête (ville, dates, gratuité) et effectue une recherche sémantique dans l'index FAISS.
3. **Génération** — `RAGPipeline` construit le prompt système avec le contexte récupéré et appelle **Mistral** pour générer la réponse.
4. **Persistance** — Chaque interaction est enregistrée dans une base SQLite via `db/database.py`. Le feedback utilisateur (👍/👎) y est également stocké.

### Modules clés

| Fichier | Rôle |
|---|---|
| `core/config.py` | Constantes globales (chemins, modèles, métriques d'évaluation) |
| `core/vector_store.py` | Gestion de l'index FAISS + `SelfQueryRetriever` |
| `core/query_classifier.py` | Classification RAG vs DIRECT (regex → mots-clés) |
| `core/rag_pipeline.py` | Orchestration classify → search → prompt → LLM |
| `core/container.py` | `AppContainer` : instanciation partagée des dépendances |
| `core/load_data.py` | Chargement et parsing des événements OpenAgenda |
| `core/prompts.py` | Templates de prompts système (RAG, JSON, direct) |
| `db/database.py` | ORM SQLAlchemy sur SQLite, logs d'interactions et feedback |

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

# Depuis l'API OpenAgenda
make index data-url="<url_openagenda>"
```

### 2. Lancer les services

```bash
make chat       # Chatbot Streamlit    → http://localhost:8501
make api        # API REST FastAPI     → http://localhost:8000
make feedback   # Dashboard feedback   → http://localhost:8502
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

Les trois services partagent les mêmes données via des bind mounts :

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

La clé `MISTRAL_API_KEY` est lue depuis le fichier `.env` par `docker-compose.yml`. Elle n'est jamais copiée dans l'image Docker (exclue via `.dockerignore`).

---

## Commandes utiles

```bash
make install     # Installer les dépendances
make index       # Indexer les données
make chat        # Lancer le chatbot Streamlit
make api         # Lancer l'API REST
make feedback    # Lancer le dashboard feedback
make test        # Lancer les tests fonctionnels
make lint        # Vérifier le style (Ruff)
make lint-fix    # Corriger automatiquement le style
make reset       # Réinitialiser l'index et la base SQLite (avec confirmation)
make clean       # Supprimer les caches Python
make eval-build  # Construire l'index d'évaluation (data/eval_events.json)
make eval        # Lancer l'évaluation Ragas
```

---

## Évaluation (Ragas)

```bash
make eval-build   # Construit un index FAISS dédié depuis data/eval_events.json
make eval         # Lance l'évaluation et génère report/eval_report.json
```

Les métriques évaluées (seuils configurables dans `core/config.py`) :

| Métrique | Description |
|---|---|
| Fidélité (`faithfulness`) | Les réponses sont-elles ancrées dans les sources ? |
| Exactitude factuelle (`factual_correctness`) | Les faits sont-ils corrects ? |
| Précision contexte (`llm_context_precision_with_reference`) | Les documents récupérés sont-ils pertinents ? |
| Rappel contexte (`context_recall`) | Tous les documents pertinents ont-ils été retrouvés ? |
| Précision réponse (`nv_accuracy`) | La réponse est-elle correcte par rapport à la référence ? |
| Pertinence réponse (`answer_relevancy`) | La réponse est-elle pertinente par rapport à la question ? |

Le rapport est visualisable directement dans le chatbot Streamlit (bouton **Évaluation RAG** dans la barre latérale) : cartes métriques, barres de progression et graphique radar scores vs seuils.

Un workflow **GitHub Actions** (`.github/workflows/evaluate_rag.yml`) relance automatiquement l'évaluation à chaque push sur `main` et chaque lundi à 8h UTC. L'index FAISS est mis en cache entre les exécutions pour éviter les appels inutiles à l'API d'embeddings.

---

## Structure du projet

```
.
├── api/
│   └── main.py              # API REST FastAPI
├── app/
│   ├── Chat.py              # Chatbot Streamlit (+ visualisation rapport évaluation)
│   └── FeedbackViewer.py    # Dashboard feedback Streamlit
├── core/
│   ├── config.py            # Constantes globales et métriques d'évaluation
│   ├── container.py         # AppContainer — instanciation des dépendances
│   ├── load_data.py         # Parsing des événements OpenAgenda
│   ├── prompts.py           # Templates de prompts système
│   ├── query_classifier.py  # Classification RAG vs DIRECT
│   ├── query_utils.py       # Utilitaires de requête
│   ├── rag_pipeline.py      # Orchestration du pipeline RAG
│   └── vector_store.py      # Index FAISS + SelfQueryRetriever
├── db/
│   └── database.py          # ORM SQLAlchemy / SQLite
├── evaluation/
│   ├── evaluate_rag.py      # Évaluation Ragas (6 métriques)
│   └── testset_generator.py # Génération du dataset d'évaluation
├── scripts/
│   ├── indexer.py           # Script d'indexation
│   └── reset.py             # Script de remise à zéro
├── tests/
│   └── test_api.py          # Tests fonctionnels API
├── .github/
│   └── workflows/
│       └── evaluate_rag.yml # CI : évaluation automatique
├── data/                    # Données OpenAgenda (JSON) + dataset d'évaluation
├── vector_db/               # Index FAISS (index.faiss + index.pkl + metadata)
├── vector_db_eval/          # Index FAISS dédié à l'évaluation
├── database/                # Base SQLite (interactions + feedback)
├── report/                  # Rapports d'évaluation (eval_report.json)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── Makefile
```

### Persistence

| Chemin | Contenu |
|---|---|
| `vector_db/index.faiss` | Index FAISS (LangChain) |
| `vector_db/index.pkl` | Document store associé |
| `vector_db/index_metadata.json` | Métadonnées (modèle, date, villes) |
| `database/interactions.db` | SQLite — interactions + feedback |
| `report/eval_report.json` | Dernier rapport d'évaluation Ragas |
