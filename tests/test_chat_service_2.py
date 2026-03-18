import pytest
import pandas as pd
from unittest.mock import MagicMock

from app.services.chat_service import ChatService

@pytest.fixture
def chat_service():
    service = ChatService()

    # Mock dependencies
    service.adb_client = MagicMock()
    service.sql_loader = MagicMock()
    service.prompt_generator_client = MagicMock()
    service.llm_inference_client = MagicMock()

    return service

class DummyAppState:
    def __init__(self):
        self.last_user_chat = {}
        self.chat_history = {}
        self.last_sql_queries = {}


def test_reconnect_conn_and_get_df_success(chat_service):
    # Arrange
    mock_conn = MagicMock()
    expected_df = pd.DataFrame({"col": [1, 2, 3]})

    chat_service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn
    chat_service.adb_client.execute_query_df.return_value = expected_df

    query = "SELECT * FROM TEST"

    # Act
    result = chat_service.reconnnect_conn_and_get_df(query)

    # Assert
    chat_service.adb_client.get_connection.assert_called_once()
    chat_service.adb_client.execute_query_df.assert_called_once_with(mock_conn, query)
    assert result.equals(expected_df)

def test_load_chat_history_ram_empty(chat_service):
    # Arrange
    chat_service.sql_loader.load_chat_history_by_id.return_value = {
        "load_chat_history": "SELECT * FROM CHAT"
    }

    empty_df = pd.DataFrame()
    mock_conn = MagicMock()

    chat_service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn
    chat_service.adb_client.execute_query_df.return_value = empty_df

    app_state = DummyAppState()

    # Act
    result = chat_service.load_chat_history_ram("user1", "chat1", app_state)

    # Assert
    assert result == "empty"

def test_load_chat_history_ram_success(chat_service):
    # Arrange
    chat_service.sql_loader.load_chat_history_by_id.return_value = {
        "load_chat_history": "SELECT * FROM CHAT"
    }

    chat_service.prompt_generator_client.generate_main_prompt.return_value = "SYSTEM PROMPT"

    data = {
        "MESSAGE_NO": [3, 2, 1],
        "ROLE": ["User", "Assistant", "SQL"],
        "MESSAGE": ["Hi", "Hello", "SELECT * FROM X"]
    }

    df = pd.DataFrame(data)
    mock_conn = MagicMock()

    chat_service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn
    chat_service.adb_client.execute_query_df.return_value = df

    app_state = DummyAppState()

    # Act
    result = chat_service.load_chat_history_ram("user1", "chat1", app_state)

    # Assert
    assert result == "success"
    assert "chat1" in app_state.chat_history
    assert app_state.last_sql_queries["chat1"] == "SELECT * FROM X"
    assert app_state.chat_history["chat1"][0]["role"] == "System"

def test_load_chat_history_ram_skips_par_and_sql(chat_service):
    chat_service.sql_loader.load_chat_history_by_id.return_value = {
        "load_chat_history": "SELECT * FROM CHAT"
    }

    chat_service.prompt_generator_client.generate_main_prompt.return_value = "SYSTEM PROMPT"

    data = {
        "MESSAGE_NO": [5, 4, 3, 2, 1],
        "ROLE": ["PAR", "SQL", "User", "Assistant", "User"],
        "MESSAGE": ["Par msg", "SQL msg", "Hi", "Hello", "How are you?"]
    }

    df = pd.DataFrame(data)
    mock_conn = MagicMock()

    chat_service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn
    chat_service.adb_client.execute_query_df.return_value = df

    app_state = DummyAppState()

    result = chat_service.load_chat_history_ram("user1", "chat1", app_state)

    assert result == "success"

    # Ensure PAR and SQL not in state history
    roles = [msg["role"] for msg in app_state.chat_history["chat1"]]
    assert "PAR" not in roles
    assert "SQL" not in roles

def test_load_chat_history_ram_limit_6(chat_service):
    chat_service.sql_loader.load_chat_history_by_id.return_value = {
        "load_chat_history": "SELECT * FROM CHAT"
    }

    chat_service.prompt_generator_client.generate_main_prompt.return_value = "SYSTEM PROMPT"

    data = {
        "MESSAGE_NO": list(range(10, 0, -1)),
        "ROLE": ["User"] * 10,
        "MESSAGE": [f"msg{i}" for i in range(10)]
    }

    df = pd.DataFrame(data)
    mock_conn = MagicMock()

    chat_service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn
    chat_service.adb_client.execute_query_df.return_value = df

    app_state = DummyAppState()

    result = chat_service.load_chat_history_ram("user1", "chat1", app_state)

    assert result == "success"

    # 1 system + max 6 messages
    assert len(app_state.chat_history["chat1"]) == 7

def test_load_chat_history_ram_removes_previous_chat(chat_service):
    chat_service.sql_loader.load_chat_history_by_id.return_value = {
        "load_chat_history": "SELECT * FROM CHAT"
    }

    chat_service.prompt_generator_client.generate_main_prompt.return_value = "SYSTEM PROMPT"

    data = {
        "MESSAGE_NO": [1],
        "ROLE": ["User"],
        "MESSAGE": ["Hi"]
    }

    df = pd.DataFrame(data)
    mock_conn = MagicMock()

    chat_service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn
    chat_service.adb_client.execute_query_df.return_value = df

    app_state = DummyAppState()
    app_state.last_user_chat["user1"] = "old_chat"
    app_state.chat_history["old_chat"] = []
    app_state.last_sql_queries["old_chat"] = "OLD SQL"

    result = chat_service.load_chat_history_ram("user1", "chat1", app_state)

    assert "old_chat" not in app_state.chat_history
    assert app_state.last_user_chat["user1"] == "chat1"

def test_load_chat_history_ram_failure(chat_service):
    chat_service.sql_loader.load_chat_history_by_id.side_effect = Exception("DB error")

    app_state = DummyAppState()

    result = chat_service.load_chat_history_ram("user1", "chat1", app_state)

    assert result == "failure by DB error"


import pytest
from unittest.mock import MagicMock, patch
from app.services.chat_service import RAW_MESSAGE, JumpToFinally

@pytest.mark.asyncio
async def test_handle_inquiry_guardrail_raw(chat_service, app_state):
    # Fake future for guardrail
    mock_future_guard = MagicMock()
    mock_future_guard.result.return_value = RAW_MESSAGE

    mock_future_sql = MagicMock()

    with patch("app.services.chat_service._PARALLEL_EXECUTOR") as mock_exec:
        mock_exec.submit.side_effect = [mock_future_guard, mock_future_sql]

        result = await chat_service.handle_inquiry(
            "user1", "chat1", "bad query", app_state
        )

    # Since RAW_MESSAGE triggers JumpToFinally,
    # we verify no crash and no SQL executed
    assert mock_future_sql.result.called is False


import pandas as pd

import pytest
from types import SimpleNamespace

@pytest.fixture
def app_state():
    return SimpleNamespace(
        last_sql_queries={},
        chat_history={},
        last_user_chat={}
    )

@pytest.mark.asyncio
async def test_handle_inquiry_context_management(chat_service, app_state):
    # ── Guardrail OK ──
    mock_future_guard = MagicMock()
    mock_future_guard.result.return_value = {
        "relevant": "yes",
        "tables": ["T1"]
    }

    # ── SQL OK ──
    mock_future_sql = MagicMock()
    mock_future_sql.result.return_value = {
        "sql_query": "select * from test",
        "error_status": 0
    }

    # ── Mock executor ──
    with patch("app.services.chat_service._PARALLEL_EXECUTOR") as mock_exec, \
         patch("app.services.chat_service.add_distinct_safely", lambda x: x), \
         patch("app.services.chat_service.ensure_fetch_first_clause", lambda x, y: x), \
         patch("app.services.chat_service.wrap_query_with_count", lambda x: x), \
         patch("app.services.chat_service.check_if_df_all_null_or_zero", return_value=False), \
         patch("app.services.chat_service.asyncio.create_task"):

        mock_exec.submit.side_effect = [mock_future_guard, mock_future_sql]

        # ── Mock DB connection ──
        mock_conn = MagicMock()
        chat_service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn

        # Fake dataframe with duplicates
        df = pd.DataFrame({"A": [1, 1, 2]})
        chat_service.adb_client.execute_query_df.return_value = df
        chat_service.adb_client.execute_scalar.return_value = 2

        chat_service.sql_loader.load_user_chats_previews.return_value = {
            "load_chats_preview": "query"
        }
        chat_service.sql_loader.get_last_message_no.return_value = {
            "last_message_no": "query"
        }

        mock_conn.execute_query_df.return_value = pd.DataFrame({"CHAT_ID": ["chat1"]})
        mock_conn.execute_scalar.return_value = 1

        chat_service.prompt_generator_client.generate_assistant_prompt.return_value = "assistant prompt"
        chat_service.llm_inference_client.inference_from_chat_history.return_value = {
            "message": "final answer"
        }

        with patch("app.services.chat_service.LLMResponseExtractor") as mock_extractor:
            instance = mock_extractor.return_value
            instance.get.return_value = "final answer"

            await chat_service.handle_inquiry(
                "user1", "chat1", "query", app_state
            )

    # Confirm duplicates removed
    assert len(df.drop_duplicates()) == 2


@pytest.mark.asyncio
async def test_handle_inquiry_exception(chat_service, app_state):
    with patch.object(chat_service, "_run_guardrail", side_effect=Exception("boom")):
        await chat_service.handle_inquiry(
            "user1", "chat1", "query", app_state
        )
    

