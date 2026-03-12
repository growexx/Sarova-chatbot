import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app


# ---------------------------------------------------------
# Test Client Fixture (Disable lifespan to avoid DB init)
# ---------------------------------------------------------

@pytest.fixture
def client():
    app.router.lifespan_context = None
    return TestClient(app)


# ---------------------------------------------------------
# 1️⃣ chat_inquiry
# ---------------------------------------------------------

def test_chat_inquiry(client):
    payload = {
        "user_id": "u1",
        "chat_id": "c1",
        "user_message": "hello"
    }

    with patch("app.api.routes.chat.service") as mock_service:
        mock_service.handle_inquiry = AsyncMock(
            return_value={"status": 1}
        )

        response = client.post(
            "/api/v1/chat/chat-inquiry",
            json=payload
        )

        assert response.status_code == 200
        assert response.json() == {"status": 1}


# ---------------------------------------------------------
# 2️⃣ load_chat_history
# ---------------------------------------------------------

def test_load_chat_history(client):
    payload = {
        "user_id": "u1",
        "chat_id": "c1"
    }

    with patch("app.api.routes.chat.service") as mock_service:
        mock_service.load_chat_history.return_value = {"status": 1}

        response = client.post(
            "/api/v1/chat/load-historical-chat",
            json=payload
        )

        assert response.status_code == 200
        assert response.json() == {"status": 1}


# ---------------------------------------------------------
# 3️⃣ load_chat_previews
# ---------------------------------------------------------

def test_load_chat_previews(client):
    payload = {
        "user_id": "u1"
    }

    with patch("app.api.routes.chat.service") as mock_service:
        mock_service.load_user_chats_previews.return_value = {"status": 1}

        response = client.post(
            "/api/v1/chat/load-chats-preview",
            json=payload
        )

        assert response.status_code == 200
        assert response.json() == {"status": 1}


# ---------------------------------------------------------
# 4️⃣ user_signout
# ---------------------------------------------------------

def test_user_signout(client):
    payload = {
        "user_id": "u1"
    }

    with patch("app.api.routes.chat.service") as mock_service:
        mock_service.chat_runtime_cleanup.return_value = {"status": 1}

        response = client.post(
            "/api/v1/chat/user-signout",
            json=payload
        )

        assert response.status_code == 200
        assert response.json() == {"status": 1}


# ---------------------------------------------------------
# 5️⃣ delete_chats
# ---------------------------------------------------------

def test_delete_chats(client):
    payload = {
        "user_id": "u1",
        "chat_ids": ["c1", "c2"]
    }

    with patch("app.api.routes.chat.service") as mock_service:
        mock_service.delete_chat_history.return_value = {"status": 1}

        response = client.request(
            "DELETE",
            "/api/v1/chat/delete-chats",
            json=payload
        )

        assert response.status_code == 200
        assert response.json() == {"status": 1}
# ---------------------------------------------------------
# 6️⃣ delete_all_chats
# ---------------------------------------------------------
def test_delete_all_chats(client):
    payload = {
        "user_id": "u1"
    }

    with patch("app.api.routes.chat.service") as mock_service:
        mock_service.delete_all_chats_for_user.return_value = {"status": 1}

        response = client.request(
            "DELETE",
            "/api/v1/chat/delete-all_chats",
            json=payload
        )

        assert response.status_code == 200
        assert response.json() == {"status": 1}

# ---------------------------------------------------------
# 7️⃣ view_state
# ---------------------------------------------------------

def test_view_state(client):
    response = client.get("/api/v1/chat/view-state")

    assert response.status_code == 200

    data = response.json()
    assert "_state" in data
    assert "last_user_chat" in data["_state"]
    assert "chat_history" in data["_state"]
    assert "last_sql_queries" in data["_state"]