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
make chat                           # run chat UI (Chat.py via streamlit)
make feedback                       # run feedback viewer (pages/Feedback_Viewer.py)
make lint                           # ruff check
make lint-fix                       # ruff check --fix
make reset                          # wipe vector store + SQLite DB (prompts confirmation)
make clean                          # remove __pycache__, .ruff_cache, .pyc files
```

No test suite exists in the project.

## Architecture

### Data Flow

1. **Indexing** (`indexer.py`): loads JSON/CSV event data → splits into 2000-char chunks (200-char overlap) → generates Mistral embeddings in batches of 32 → builds `faiss.IndexFlatIP` (cosine similarity via normalized L2) → saves to `vector_db/faiss_index.idx` and `vector_db/document_chunks.pkl`.

2. **Query** (`Chat.py`): user message → `QueryClassifier.needs_rag()` → if RAG: `VectorStoreManager.search()` embeds query, searches FAISS, filters by `min_score` → builds system prompt with context → `mistral.chat.complete()` → response displayed + logged to SQLite.

### Key Modules

- **`utils/config.py`** — all constants: paths, model names, chunk sizes, DB URL. Change model or chunk settings here.
- **`utils/vector_store.py`** — `VectorStoreManager`: builds/loads FAISS index, generates embeddings, runs semantic search. Scores are cosine similarity × 100 (displayed as %).
- **`utils/query_classifier.py`** — `QueryClassifier.needs_rag()`: three-tier classification — (1) regex for greetings/farewells, (2) keyword matching against event-related terms, (3) LLM fallback (`mistral-small-latest`) for ambiguous queries. Returns `(bool, confidence, reason)`.
- **`utils/load_data.py`** — parses OpenAgenda JSON or CSV into LangChain `Document` objects; strips HTML from `longdescription_fr`; extracts metadata (city, dates, price, coordinates).
- **`utils/database.py`** — SQLAlchemy ORM over SQLite (`database/interactions.db`); logs every interaction (query, response, sources, mode metadata) and stores thumbs-up/down feedback via `update_feedback()`.

### Persistence

| Path | Contents |
|---|---|
| `vector_db/faiss_index.idx` | FAISS index (rebuilt by `make index`) |
| `vector_db/document_chunks.pkl` | Chunk text + metadata (parallel to index) |
| `database/interactions.db` | SQLite — all chat interactions + feedback |

### Streamlit UI (`Chat.py`)

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
