# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot for recommending French cultural events ("Puls-Events"). It ingests OpenAgenda event data, builds a FAISS vector index using Mistral embeddings, then serves a Streamlit chat UI that classifies queries and routes them through RAG or direct LLM answers.

## Setup

Requires a `.env` file with:
```
MISTRAL_API_KEY=<your_key>
```

```bash
make install    # install deps via uv
make index input-file=data/evenements-publics-openagenda.json   # build vector index
make chat       # launch Streamlit app at localhost:8501
```

## Common Commands

```bash
make index                          # index from default file
make index data-url=<api_url>       # index from OpenAgenda API URL
make chat                           # run chat UI (app/Chat.py via streamlit)
make feedback                       # run feedback viewer (app/FeedbackViewer.py)
make lint                           # ruff check
make lint-fix                       # ruff check --fix
make reset                          # wipe vector store + SQLite DB (prompts confirmation)
make clean                          # remove __pycache__, .ruff_cache, .pyc files
```

All `make` commands that invoke Python set `PYTHONPATH=.` inline so packages resolve from the project root.

## Architecture

### Directory Structure

```
├── api/              # FastAPI REST server (api/main.py)
├── app/              # Streamlit UIs (app/Chat.py, app/FeedbackViewer.py)
├── scripts/          # CLI scripts (scripts/indexer.py, scripts/reset.py)
├── core/             # Business logic (config, vector_store, rag_pipeline, …)
├── db/               # Persistence layer (db/database.py — SQLAlchemy/SQLite)
└── evaluation/       # Ragas evaluation (evaluate_rag.py, test_generator.py)
```

### Data Flow

1. **Indexing** (`scripts/indexer.py`): loads JSON/CSV event data → generates Mistral embeddings → builds LangChain FAISS index → saves to `vector_db/`.

2. **Query** (`app/Chat.py`): user message → `QueryClassifier.needs_rag()` → if RAG: `VectorStoreManager.search()` embeds query, searches FAISS, filters by `min_score` → builds system prompt with context → `mistral.chat.complete()` → response displayed + logged to SQLite.

### Key Modules

- **`core/config.py`** — all constants: paths, model names, chunk sizes, DB URL. Change model or chunk settings here.
- **`core/vector_store.py`** — `VectorStoreManager`: builds/loads FAISS index, generates embeddings, runs semantic search. Scores are cosine similarity × 100 (displayed as %).
- **`core/query_classifier.py`** — `QueryClassifier.needs_rag()`: two-tier classification — (1) regex for greetings/farewells, (2) keyword matching against event-related terms. Returns `(bool, confidence, reason)`.
- **`core/rag_pipeline.py`** — `RAGPipeline.run()`: shared classify → search → prompt → LLM flow used by both the Streamlit UI and the FastAPI server.
- **`core/load_data.py`** — parses OpenAgenda JSON or CSV into LangChain `Document` objects; strips HTML from `longdescription_fr`; extracts metadata (city, dates, price, coordinates).
- **`core/prompts.py`** — system prompt builders for RAG, RAG-no-results, JSON and direct modes.
- **`core/container.py`** — `AppContainer` dataclass + `build_container()` factory that wires VectorStore, Mistral client and QueryClassifier.
- **`db/database.py`** — SQLAlchemy ORM over SQLite (`database/interactions.db`); logs every interaction (query, response, sources, mode metadata) and stores thumbs-up/down feedback via `update_feedback()`.
- **`api/main.py`** — FastAPI REST server (`/ask`, `/rebuild`, `/metadata`, `/health`). Run with `PYTHONPATH=. uv run uvicorn api.main:app`.

### Persistence

| Path | Contents |
|---|---|
| `vector_db/index.faiss` | FAISS index (rebuilt by `make index`) |
| `vector_db/index.pkl` | Document store (parallel to index) |
| `vector_db/index_metadata.json` | Index metadata (model, date, cities) |
| `database/interactions.db` | SQLite — all chat interactions + feedback |

### Streamlit UI (`app/Chat.py`)

- `@st.cache_resource` caches `VectorStoreManager`, `Mistral` client, and `QueryClassifier` across reruns.
- Sidebar controls: model selection (`mistral-small-latest` / `mistral-large-latest`), number of docs (k), minimum similarity score threshold.
- After each assistant response, `streamlit_feedback` (thumbs) writes back to SQLite via `update_feedback()`.
- Three system prompt modes: RAG with results, RAG without results, Direct (no retrieval).

## Configuration Defaults

| Setting | Value |
|---|---|
| Embedding model | `mistral-embed` |
| Chat model | `mistral-small-latest` |
| Chunk size | 2000 chars |
| Chunk overlap | 200 chars |
| Embedding batch size | 32 |
| Default search k | 5 |
| Default min score | 75% |
