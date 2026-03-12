import builtins
from unittest.mock import MagicMock, mock_open, patch

from app.services.title import create_new_chat_title


# ------------------------------------------------------------------
# Helper: Create mock service with required attributes
# ------------------------------------------------------------------

def get_mock_service():
    service = MagicMock()

    service.llm_inference_client = MagicMock()
    service.llm_response_extractor = MagicMock()
    service.prompt_generator_client = MagicMock()
    service.sql_loader = MagicMock()
    service.adb_client = MagicMock()

    return service


# ------------------------------------------------------------------
# 1️⃣ Happy Path Test
# ------------------------------------------------------------------

def test_create_new_chat_title_success():
    service = get_mock_service()
    conn = MagicMock()

    service.llm_inference_client.inference_single_input.return_value = "mocked_llm_response"
    service.llm_response_extractor.get.return_value = "Generated Title"
    service.sql_loader.insert_user_chat.return_value = {
        "insert_user_chat": "INSERT SQL"
    }

    m = mock_open(read_data="Title prompt {user_message}")

    with patch.object(builtins, "open", m):
        user_id, chat_id, title = create_new_chat_title(
            conn,
            service,
            user_id="user1",
            chat_id="chat123456",
            user_query="Hello world"
        )

    assert title == "Generated Title"
    service.llm_inference_client.inference_single_input.assert_called_once()
    service.adb_client.execute_single_non_query.assert_called_once()


# ------------------------------------------------------------------
# 2️⃣ LLM Returns Tuple
# ------------------------------------------------------------------

def test_create_new_chat_title_llm_returns_tuple():
    service = get_mock_service()
    conn = MagicMock()

    service.llm_inference_client.inference_single_input.return_value = ("tuple_title",)
    service.llm_response_extractor.get.return_value = "Tuple Extracted"

    m = mock_open(read_data="Prompt {user_message}")

    with patch.object(builtins, "open", m):
        _, _, title = create_new_chat_title(
            conn,
            service,
            user_id=None,
            chat_id="chat123456",
            user_query="Hi"
        )

    assert title == "Tuple Extracted"


# ------------------------------------------------------------------
# 3️⃣ Extractor Fails → Fallback Title
# ------------------------------------------------------------------

def test_create_new_chat_title_extractor_exception():
    service = get_mock_service()
    conn = MagicMock()

    service.llm_inference_client.inference_single_input.return_value = "bad_response"
    service.llm_response_extractor.set_data.side_effect = Exception("Extractor failure")

    m = mock_open(read_data="Prompt {user_message}")

    with patch.object(builtins, "open", m):
        _, chat_id, title = create_new_chat_title(
            conn,
            service,
            user_id=None,
            chat_id="chatABCDEFGH",
            user_query="Hi"
        )

    assert title == "Chat_chatABCD"


# ------------------------------------------------------------------
# 4️⃣ DB Insert Fails But Function Still Returns
# ------------------------------------------------------------------

def test_create_new_chat_title_db_exception():
    service = get_mock_service()
    conn = MagicMock()

    service.llm_inference_client.inference_single_input.return_value = "ok"
    service.llm_response_extractor.get.return_value = "Safe Title"

    service.sql_loader.insert_user_chat.return_value = {
        "insert_user_chat": "INSERT SQL"
    }

    service.adb_client.execute_single_non_query.side_effect = Exception("DB error")

    m = mock_open(read_data="Prompt {user_message}")

    with patch.object(builtins, "open", m):
        _, _, title = create_new_chat_title(
            conn,
            service,
            user_id="user1",
            chat_id="chat123456",
            user_query="Hi"
        )

    # Should not crash
    assert title == "Safe Title"


# ------------------------------------------------------------------
# 5️⃣ user_id is None → DB Not Called
# ------------------------------------------------------------------

def test_create_new_chat_title_without_user_id():
    service = get_mock_service()
    conn = MagicMock()

    service.llm_inference_client.inference_single_input.return_value = "ok"
    service.llm_response_extractor.get.return_value = "No DB Title"

    m = mock_open(read_data="Prompt {user_message}")

    with patch.object(builtins, "open", m):
        _, _, title = create_new_chat_title(
            conn,
            service,
            user_id=None,
            chat_id="chat123456",
            user_query="Hi"
        )

    assert title == "No DB Title"
    service.adb_client.execute_single_non_query.assert_not_called()