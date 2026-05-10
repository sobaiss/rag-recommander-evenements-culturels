"""
Tests fonctionnels de l'API RAG Puls-Events.

Les dépendances externes (Mistral API, FAISS, OpenAgenda) sont mockées
pour que les tests s'exécutent sans clé API ni index réel.

Stratégie d'isolation : après démarrage du lifespan, on écrase directement
`main._state` avec des mocks pour éviter tout appel réseau.
"""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

import main as _main
from main import app


# ──────────────────────────────────────────────────────────────────────────────
# Factories de mocks réutilisables
# ──────────────────────────────────────────────────────────────────────────────
def make_vector_store(search_results=None):
    store = MagicMock()
    store.index = MagicMock()
    store.index.ntotal = 42
    store.get_metadata.return_value = {
        "embedding_model": "mistral-embed",
        "chunk_size": 2000,
        "chunk_overlap": 200,
        "num_documents": 10,
        "num_chunks": 42,
        "created_at": "2025-05-01T10:00:00",
    }
    store.search.return_value = search_results or [
        {
            "text": "TITRE: Concert Jazz à Paris\nLIEU: Salle Pleyel (Paris)\nDATES : du 2026-05-15 au 2026-05-15",
            "score": 85.0,
            "metadata": {
                "source": "https://openagenda.com/concert-jazz",
                "ville": "Paris",
                "date_debut": "2026-05-15",
            },
        }
    ]
    return store


def make_mistral_client(answer="Concert Jazz à Paris le 15 mai 2026."):
    client = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = answer
    client.chat.complete.return_value = response
    return client


def make_classifier(needs_rag=True, confidence=0.9, reason="Mots-clés événements"):
    classifier = MagicMock()
    classifier.needs_rag.return_value = (needs_rag, confidence, reason)
    return classifier


# ──────────────────────────────────────────────────────────────────────────────
# Fixture principale : client HTTP avec mocks injectés dans _state
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
async def http_client():
    """
    Client HTTP branché sur l'app FastAPI.
    Injecte les mocks directement dans main._state après le démarrage
    du lifespan, évitant tout appel réseau ou lecture de fichiers.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        _main._state["vector_store"] = make_vector_store()
        _main._state["mistral_client"] = make_mistral_client()
        _main._state["query_classifier"] = make_classifier()
        yield ac
    _main._state.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Tests GET /health
# ──────────────────────────────────────────────────────────────────────────────
async def test_health_ok(http_client):
    response = await http_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["index_loaded"] is True
    assert data["num_vectors"] == 42
    assert data["mistral_configured"] is True
    assert data["index_metadata"]["embedding_model"] == "mistral-embed"
    assert data["index_metadata"]["num_chunks"] == 42


# ──────────────────────────────────────────────────────────────────────────────
# Tests POST /ask — mode RAG
# ──────────────────────────────────────────────────────────────────────────────
async def test_ask_rag_returns_answer_and_sources(http_client):
    response = await http_client.post("/ask", json={"question": "Quels concerts à Paris ?"})

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "RAG"
    assert data["confidence"] == 0.9
    assert len(data["sources"]) == 1
    assert data["sources"][0]["score"] == 85.0
    assert "Concert Jazz" in data["sources"][0]["text"]
    assert data["answer"] != ""
    assert data["model_used"] == "mistral-large-latest"


async def test_ask_rag_custom_model(http_client):
    mock_mistral = make_mistral_client()
    _main._state["mistral_client"] = mock_mistral

    response = await http_client.post(
        "/ask",
        json={"question": "Événements gratuits à Lyon ?", "model": "mistral-large-latest"},
    )

    assert response.status_code == 200
    assert response.json()["model_used"] == "mistral-large-latest"
    call_kwargs = mock_mistral.chat.complete.call_args
    assert call_kwargs.kwargs["model"] == "mistral-large-latest"


async def test_ask_rag_custom_k_and_min_score(http_client):
    mock_store = make_vector_store()
    _main._state["vector_store"] = mock_store

    await http_client.post(
        "/ask",
        json={"question": "Expositions ce mois ?", "k": 10, "min_score": 0.5},
    )

    call = mock_store.search.call_args
    assert call.kwargs["k"] == 10
    assert call.kwargs["min_score"] == 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Tests POST /ask — mode DIRECT
# ──────────────────────────────────────────────────────────────────────────────
async def test_ask_direct_no_sources(http_client):
    _main._state["query_classifier"] = make_classifier(needs_rag=False, confidence=0.95, reason="Salutation")
    mock_store = make_vector_store()
    _main._state["vector_store"] = mock_store

    response = await http_client.post("/ask", json={"question": "Bonjour !"})

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "DIRECT"
    assert data["sources"] == []
    mock_store.search.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# Tests POST /ask — validation et erreurs
# ──────────────────────────────────────────────────────────────────────────────
async def test_ask_empty_question_rejected(http_client):
    response = await http_client.post("/ask", json={"question": ""})
    assert response.status_code == 422


async def test_ask_missing_question_rejected(http_client):
    response = await http_client.post("/ask", json={})
    assert response.status_code == 422


async def test_ask_k_out_of_range_rejected(http_client):
    assert (await http_client.post("/ask", json={"question": "Test", "k": 0})).status_code == 422
    assert (await http_client.post("/ask", json={"question": "Test", "k": 21})).status_code == 422


async def test_ask_no_mistral_key_returns_503(http_client):
    _main._state["mistral_client"] = None

    response = await http_client.post("/ask", json={"question": "Test"})

    assert response.status_code == 503
    assert "MISTRAL_API_KEY" in response.json()["detail"]


async def test_ask_mistral_api_error_returns_502(http_client):
    mock_mistral = make_mistral_client()
    mock_mistral.chat.complete.side_effect = RuntimeError("API timeout")
    _main._state["mistral_client"] = mock_mistral

    response = await http_client.post("/ask", json={"question": "Concerts ?"})

    assert response.status_code == 502
    assert "génération" in response.json()["detail"]


# ──────────────────────────────────────────────────────────────────────────────
# Tests POST /rebuild — validation
# ──────────────────────────────────────────────────────────────────────────────
async def test_rebuild_chunk_overlap_gte_chunk_size_rejected(http_client):
    response = await http_client.post(
        "/rebuild", json={"chunk_size": 500, "chunk_overlap": 500}
    )
    assert response.status_code == 422
    assert "chunk_overlap" in response.json()["detail"]


async def test_rebuild_no_events_returns_404(http_client):
    with patch("main.load_documents_from_url_paginated", return_value=[]):
        response = await http_client.post(
            "/rebuild", json={"cities": ["VilleFantôme"], "begin_date": "2099-01-01"}
        )
    assert response.status_code == 404


async def test_rebuild_openagenda_network_error_returns_502(http_client):
    with patch("main.load_documents_from_url_paginated", side_effect=ConnectionError("timeout")):
        response = await http_client.post("/rebuild", json={})
    assert response.status_code == 502


# ──────────────────────────────────────────────────────────────────────────────
# Tests POST /rebuild — succès
# ──────────────────────────────────────────────────────────────────────────────
async def test_rebuild_success(http_client):
    mock_doc = MagicMock()
    mock_doc.page_content = "TITRE: Festival Test\nLIEU: Paris"
    mock_doc.metadata = {"ville": "Paris"}

    mock_new_store = MagicMock()
    mock_new_store.get_metadata.return_value = {
        "num_documents": 1,
        "num_chunks": 3,
        "embedding_model": "mistral-embed",
    }

    with (
        patch("main.load_documents_from_url_paginated", return_value=[mock_doc]),
        patch("main.save_documents_to_json", return_value="data/openagenda_20260506.json"),
        patch("main.VectorStoreManager", return_value=mock_new_store),
    ):
        response = await http_client.post(
            "/rebuild",
            json={"cities": ["Paris"], "begin_date": "2025-05-01"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["num_documents"] == 1
    assert data["num_chunks"] == 3
    assert data["embedding_model"] == "mistral-embed"
    assert "data/" in data["data_file"]
    # Vérifie que le vecteur store en mémoire a été mis à jour
    assert _main._state["vector_store"] is mock_new_store


async def test_rebuild_updates_in_memory_vector_store(http_client):
    """Après /rebuild, les requêtes /ask utilisent le nouvel index."""
    mock_new_store = MagicMock()
    mock_new_store.get_metadata.return_value = {"num_documents": 5, "num_chunks": 15, "embedding_model": "mistral-embed"}
    mock_new_store.search.return_value = []

    mock_doc = MagicMock()
    mock_doc.page_content = "TITRE: Expo"
    mock_doc.metadata = {}

    with (
        patch("main.load_documents_from_url_paginated", return_value=[mock_doc]),
        patch("main.save_documents_to_json", return_value="data/test.json"),
        patch("main.VectorStoreManager", return_value=mock_new_store),
    ):
        await http_client.post("/rebuild", json={})

    # Le vecteur store en mémoire est maintenant le nouveau
    assert _main._state["vector_store"] is mock_new_store


# ──────────────────────────────────────────────────────────────────────────────
# Tests POST /rebuild — construction de l'URL OpenAgenda
# ──────────────────────────────────────────────────────────────────────────────
async def test_rebuild_url_contains_city_filter(http_client):
    with patch("main.load_documents_from_url_paginated", return_value=[]) as mock_fetch:
        await http_client.post("/rebuild", json={"cities": ["Bordeaux"]})
        called_url = mock_fetch.call_args[0][0]

    assert "refine.location_city=Bordeaux" in called_url


async def test_rebuild_url_contains_date_filter_formatted(http_client):
    """La date YYYY-MM-DD doit être convertie en YYYY%2FMM dans l'URL."""
    with patch("main.load_documents_from_url_paginated", return_value=[]) as mock_fetch:
        await http_client.post("/rebuild", json={"begin_date": "2026-03-15"})
        called_url = mock_fetch.call_args[0][0]

    assert "refine.firstdate_begin=2026%2F03" in called_url


async def test_rebuild_url_no_filter_when_no_city(http_client):
    """Sans villes sélectionnées, l'URL ne doit pas contenir refine.location_city."""
    with patch("main.load_documents_from_url_paginated", return_value=[]) as mock_fetch:
        await http_client.post("/rebuild", json={"cities": []})
        called_url = mock_fetch.call_args[0][0]

    assert "refine.location_city" not in called_url
