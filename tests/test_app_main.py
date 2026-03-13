import sys
from unittest.mock import MagicMock, patch

# ------------------------------------------------------------------
# Mock heavy dependencies BEFORE importing app
# ------------------------------------------------------------------
sys.modules["seaborn"] = MagicMock()

# ------------------------------------------------------------------
# Mock ChatService BEFORE importing app
# IMPORTANT: patch where it is USED (app.main.chat.service)
# ------------------------------------------------------------------
mock_adb_client = MagicMock()
mock_service = MagicMock()
mock_service.adb_client = mock_adb_client

with patch("app.main.chat.service", mock_service):
    from fastapi.testclient import TestClient
    from app.main import app


# ---------------------------------------------------------
# Basic App Tests
# ---------------------------------------------------------

def test_app_starts():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 404


def test_app_title():
    assert app.title == "Americana Audit Bot Fast API"


# ---------------------------------------------------------
# Middleware Tests
# ---------------------------------------------------------

def test_cors_middleware_configured():
    cors_middleware = next(
        (m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"),
        None
    )

    assert cors_middleware is not None
    assert cors_middleware.kwargs["allow_origins"] == ["*"]
    assert cors_middleware.kwargs["allow_methods"] == ["*"]
    assert cors_middleware.kwargs["allow_headers"] == ["*"]
    assert cors_middleware.kwargs["allow_credentials"] is True


# ---------------------------------------------------------
# App State Tests (UPDATED)
# ---------------------------------------------------------

def test_app_state_initialized():
    assert isinstance(app.state.last_user_chat, dict)
    assert isinstance(app.state.chat_history, dict)
    assert isinstance(app.state.last_sql_queries, dict)

    assert app.state.last_user_chat == {}
    assert app.state.chat_history == {}
    assert app.state.last_sql_queries == {}


# ---------------------------------------------------------
# Router Tests
# ---------------------------------------------------------

def test_chat_router_registered():
    routes = [route.path for route in app.routes]
    assert any(path.startswith("/api/v1/chat") for path in routes)


# ---------------------------------------------------------
# Lifespan Tests (UPDATED)
# ---------------------------------------------------------

def test_lifespan_startup_and_shutdown():
    """
    Verify init_pool() is called on startup
    and close_pool() is called on shutdown.
    """

    with patch("app.main.chat.service.adb_client") as mock_adb_client:

        from app.main import app  # import AFTER patch

        with TestClient(app):
            mock_adb_client.init_pool.assert_called_once()

        mock_adb_client.close_pool.assert_called_once()