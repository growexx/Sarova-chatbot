import os
import json
import time
import pytest
import pandas as pd
import numpy as np

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.chat_service import (
    ChatService,
    prepare_metadata_string,
    check_if_df_all_null_or_zero,
    add_distinct_safely,
    _load_all_metadata,
)
# ============================================
# Guardrail / Text2SQL
# ============================================

def test_guard_rail(chat_service):
    chat_service.prompt_generator_client.guardrail_check_inference_call.return_value = {"r":"yes","t":["T1"]}
    chat_service.llm_response_extractor.get_many.return_value = ("yes", ["T1"])
    relevant, tables = chat_service.guard_rail("hello")
    assert relevant == "yes"
    assert tables == ["T1"]


# ============================================
# load_chat_history
# ============================================

def test_load_chat_history_empty(chat_service, app_state):
    chat_service.sql_loader.load_chat_history_by_id.return_value = {"load_chat_history":"SELECT 1"}
    chat_service.adb_client.execute_query_df.return_value = pd.DataFrame()
    res = chat_service.load_chat_history("u1","c1",app_state)
    assert res["status"] == 0


# # ============================================
# # load_user_chats_previews
# # ============================================

def test_load_user_chats_previews_success(chat_service):
    df = pd.DataFrame([{"chat":"c1"}])
    chat_service.sql_loader.load_user_chats_previews.return_value = {"load_chats_preview":"SELECT 1"}
    chat_service.adb_client.execute_query_df.return_value = df
    res = chat_service.load_user_chats_previews("u1",SimpleNamespace())
    assert res["status"] == 1

def test_load_user_chats_previews_error(chat_service):
    chat_service.sql_loader.load_user_chats_previews.side_effect = Exception("fail")
    res = chat_service.load_user_chats_previews("u1",SimpleNamespace())
    assert res["status"] == 0

# ============================================
# chat_runtime_cleanup
# ============================================

def test_chat_runtime_cleanup_nodata(chat_service):
    state = SimpleNamespace(chat_history={}, last_sql_query={}, user_chats={})
    res = chat_service.chat_runtime_cleanup("u1", state)
    assert res["status"] == 0

# ============================================
# delete_chat_history
# ============================================

def test_delete_chat_history_success(chat_service):
    state = SimpleNamespace(
        chat_history={"c1":[]},
        last_sql_queries={"c1":"SQL"}
    )
    chat_service.sql_loader.delete_chat_queries.return_value = {"delete_chat_history":"Q1","delete_chat_preview":"Q2"}
    chat_service.adb_client.execute_single_non_query.return_value = None
    res = chat_service.delete_chat_history("u1",["c1"],state)
    assert res["status"] == 1

def test_delete_chat_history_error(chat_service):
    state = SimpleNamespace(chat_history={"c1":[]}, last_sql_queries={"c1":"SQL"})
    chat_service.sql_loader.delete_chat_queries.side_effect = Exception("fail")
    res = chat_service.delete_chat_history("u1",["c1"],state)
    assert res["status"] == 0


# ============================================
# text_2_sql
# ============================================

def test_text_2_sql_success(chat_service, monkeypatch):
    user_message = "show all users"
    tables = ["users"]
    last_sql_query = "SELECT * FROM users"
    metadata = "ignored"  # method recomputes metadata internally

    # mock prepare_metadata_string
    monkeypatch.setattr(
        "app.services.chat_service.prepare_metadata_string",
        MagicMock(return_value="META")
    )

    # mock prompt generator
    chat_service.prompt_generator_client.generate_sql_prompt.return_value = "SQL_PROMPT"

    # mock llm inference
    chat_service.llm_inference_client.inference_single_input.return_value = {
        "raw": "llm_output"
    }

    # mock extractor
    chat_service.llm_response_extractor.get_many.return_value = (
        "SELECT * FROM users",
        0
    )

    sql_query, error_status = chat_service.text_2_sql(
        user_message=user_message,
        tables=tables,
        last_sql_query=last_sql_query,
        metadata=metadata
    )

    assert sql_query == "SELECT * FROM users"
    assert error_status == 0

    chat_service.prompt_generator_client.generate_sql_prompt.assert_called_once_with(
        user_message, "META", last_sql_query
    )

    chat_service.llm_inference_client.inference_single_input.assert_called_once_with(
        user_message, "SQL_PROMPT"
    )

    chat_service.llm_response_extractor.set_data.assert_called_once()
    chat_service.llm_response_extractor.get_many.assert_called_once_with(
        ["sql_query", "error_status"],
        {"sql_query": "", "error_status": 0}
    )


# ============================================
# delete_all_chats_for_user
# ============================================

# def test_delete_all_chats_for_user_success(chat_service):
#     user_id = "u1"
#     app_state = SimpleNamespace()

#     # mock delete previews SQL
#     chat_service.sql_loader.delete_all_chats_for_user.return_value = {
#         "delete_all_chats_for_user": "DELETE FROM chat_previews"
#     }

#     # mock bulk delete SQL
#     chat_service.sql_loader.delete_chat_history_bulk.return_value = "BULK DELETE SQL"

#     # adb methods return None on success
#     chat_service.adb_client.execute_single_non_query.return_value = None
#     chat_service.adb_client.execute_multiple_non_query.return_value = None

#     result = chat_service.delete_all_chats_for_user(user_id, app_state)

#     assert result["status"] == 1
#     assert f"user {user_id}" in result["message"]

#     chat_service.sql_loader.load_user_chats_previews.assert_called_once_with(user_id)
#     chat_service.sql_loader.delete_all_chats_for_user.assert_called_once_with(user_id)
#     chat_service.sql_loader.delete_chat_history_bulk.assert_called_once()

#     chat_service.adb_client.execute_single_non_query.assert_called_once()
#     chat_service.adb_client.execute_multiple_non_query.assert_called_once()

def test_delete_all_chats_for_user_success(chat_service):
    user_id = "u1"
    app_state = SimpleNamespace()

    # Mock SQL loader responses
    chat_service.sql_loader.delete_all_chats_for_user.return_value = {
        "delete_all_chats_for_user": "DELETE FROM chat_previews WHERE user_id = :1"
    }

    chat_service.sql_loader.delete_chat_history_bulk.return_value = "BULK DELETE SQL"

    # Mock DB connection context manager
    mock_conn = MagicMock()
    chat_service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn
    chat_service.adb_client.get_connection.return_value.__exit__.return_value = None

    # Mock DB executions
    chat_service.adb_client.execute_single_non_query.return_value = None
    chat_service.adb_client.execute_multiple_non_query.return_value = None

    # Act
    result = chat_service.delete_all_chats_for_user(user_id, app_state)

    # Assert response
    assert result["status"] == 1
    assert f"user {user_id}" in result["message"]

    # Assert SQL calls
    chat_service.sql_loader.delete_all_chats_for_user.assert_called_once_with(user_id)
    chat_service.sql_loader.delete_chat_history_bulk.assert_called_once()

    # Assert DB calls
    chat_service.adb_client.execute_single_non_query.assert_called_once_with(
        mock_conn,
        "DELETE FROM chat_previews WHERE user_id = :1"
    )

    chat_service.adb_client.execute_multiple_non_query.assert_called_once_with(
        mock_conn,
        "BULK DELETE SQL"
    )

def test_delete_all_chats_for_user_error(chat_service):
    user_id = "u1"
    app_state = SimpleNamespace()

    # force exception early
    chat_service.sql_loader.delete_all_chats_for_user.side_effect = Exception("DB error")

    result = chat_service.delete_all_chats_for_user(user_id, app_state)

    assert result["status"] == 0
    assert "error_message" in result


# ============================================
# chat_runtime_cleanup (exception flow)
# ============================================

def test_chat_runtime_cleanup_exception(chat_service):
    user_id = "u1"

    class BrokenState:
        @property
        def chat_history(self):
            raise Exception("state access failed")

    broken_state = BrokenState()

    result = chat_service.chat_runtime_cleanup(user_id, broken_state)

    assert result["status"] == 0
    assert "error_message" in result




def test_load_chat_history_outer_exception(chat_service, app_state):
    chat_service.sql_loader.load_chat_history_by_id.side_effect = Exception("DB down")

    result = chat_service.load_chat_history("u1", "c1", app_state)

    assert result["status"] == 0
    assert "error_message" in result


# =========================================================
# Imports
# =========================================================

from app.services.chat_service import (
    ChatService,
    prepare_metadata_string,
    check_if_df_all_null_or_zero,
    add_distinct_safely,
    _load_all_metadata,
)

# =========================================================
# Fixtures
# =========================================================

@pytest.fixture
def chat_service():
    service = ChatService()
    service.llm_inference_client = MagicMock()
    service.llm_response_extractor = MagicMock()
    service.prompt_generator_client = MagicMock()
    service.adb_client = MagicMock()
    service.sql_loader = MagicMock()
    return service

@pytest.fixture
def app_state():
    return SimpleNamespace(
        chat_history={},
        last_sql_queries={},
        user_chats={},
    )


# =========================================================
# Utility Function Tests
# =========================================================

# def test_is_count_query():
#     assert is_count_query("SELECT COUNT(id) FROM users")
#     assert is_count_query("select count(id) from users")
#     assert is_count_query("   SELECT   COUNT ( id ) FROM users")
#     assert not is_count_query("SELECT id FROM users")


def test_add_distinct_safely():
    assert add_distinct_safely("SELECT DISTINCT id FROM users") == "SELECT DISTINCT id FROM users"

    result = add_distinct_safely("SELECT COUNT(id) FROM users")
    assert "COUNT(DISTINCT id)" in result

    result = add_distinct_safely("SELECT id FROM users")
    assert result.startswith("SELECT DISTINCT")


def test_log_time(chat_service):
    with patch("time.perf_counter", side_effect=[1.0, 2.5]), \
         patch("builtins.print") as mock_print:

        start = time.perf_counter()
        chat_service.log_time("TestLabel", start)

        printed = mock_print.call_args[0][0]
        assert "[TIMER] TestLabel took" in printed
        assert "1.5000" in printed


def test_check_if_df_all_null_or_zero():
    df1 = pd.DataFrame({"a": [0, None], "b": [0, None]})
    df2 = pd.DataFrame({"a": [1, 0], "b": [0, 0]})
    assert check_if_df_all_null_or_zero(df1)
    assert not check_if_df_all_null_or_zero(df2)


# =========================================================
# Metadata Loading
# =========================================================

def test_prepare_metadata_string(tmp_path, monkeypatch):
    table_dir = tmp_path / "table_metadata"
    table_dir.mkdir()
    file_path = table_dir / "table1.json"
    file_path.write_text(json.dumps({"col": "val"}))

    monkeypatch.chdir(tmp_path)

    result = prepare_metadata_string(["table1"])

    assert "TABLE1" in result
    assert '"col": "val"' in result


def test_load_all_metadata_file_not_found():
    with patch("app.services.chat_service._ALL_TABLES", ["missing"]), \
         patch("builtins.open", side_effect=FileNotFoundError):

        result = _load_all_metadata()

    assert result == ""


# =========================================================
# prepare_data_response
# =========================================================


# =========================================================
# prepare_last_sql_query
# =========================================================


def test_prepare_last_sql_query_existing(chat_service):
    result = chat_service.prepare_last_sql_query("SQL", "c1")
    assert result == "SQL"


# =========================================================
# _run_guardrail
# =========================================================

def test_run_guardrail(chat_service):
    chat_service.prompt_generator_client.guardrail_check_inference_call.return_value = "raw"

    with patch("app.services.chat_service.LLMResponseExtractor") as mock_ext:
        instance = MagicMock()
        instance.get_many.return_value = ("yes", ["T1"])
        mock_ext.return_value = instance

        result = chat_service._run_guardrail("hello")

    assert result == {"relevant": "yes", "tables": ["T1"]}


# =========================================================
# _run_text_2_sql
# =========================================================

def test_run_text_2_sql(chat_service):
    chat_service.prompt_generator_client.generate_sql_prompt.return_value = "PROMPT"
    chat_service.llm_inference_client.inference_single_input.return_value = "raw"

    with patch("app.services.chat_service.LLMResponseExtractor") as mock_ext, \
         patch("app.services.chat_service._ALL_METADATA_STRING", "META"):

        instance = MagicMock()
        instance.get_many.return_value = ("SELECT 1", 0)
        mock_ext.return_value = instance

        result = chat_service._run_text_2_sql("msg", None)

    assert result == {"sql_query": "SELECT 1", "error_status": 0}


def test_prepare_message_no_empty_df(chat_service):
    empty_df = pd.DataFrame()

    result = chat_service.prepare_message_no(empty_df)

    assert result == 1


def test_prepare_message_no_non_empty_df(chat_service):
    df = pd.DataFrame([{"MESSAGE_NO": 5}])

    result = chat_service.prepare_message_no(df)


    assert result == 6



class FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value



def test_chat_runtime_cleanup_success(chat_service):
    state = SimpleNamespace(
        chat_history={"c1": ["m1"], "c2": ["m2"]},
        last_sql_queries={"c1": "SQL1", "c2": "SQL2"},
        user_chats={"u1": ["c1", "c2"]},
    )

    result = chat_service.chat_runtime_cleanup("u1", state)

    assert result["status"] == 1
    assert "Deleted 2 chat history" in result["message"]

    # Ensure deletion happened
    assert "c1" not in state.chat_history
    assert "c2" not in state.chat_history
    assert "c1" not in state.last_sql_queries
    assert "c2" not in state.last_sql_queries




def test_prepare_local_file_and_par_url(chat_service):
    with patch("app.services.chat_service.datetime") as mock_datetime, \
         patch("app.services.chat_service.uuid") as mock_uuid:

        # Mock utcnow() to return real datetime
        mock_datetime.utcnow.return_value = datetime(2025, 1, 1, 12, 0, 0)

        # Mock uuid
        mock_uuid.uuid4.return_value.hex = "abcdef1234567890"

        local_file, par_url = chat_service._prepare_local_file_and_par_url()

    assert local_file.startswith("result_data_20250101_120000_abcdef12")
    assert local_file.endswith(".xlsx")
    assert local_file in par_url




@pytest.mark.asyncio
async def test_background_message_insert(chat_service):
    rows = [{"msg": "hello"}]

    chat_service.sql_loader.insert_chat_history_bulk.return_value = {
        "query": "INSERT",
        "params": ["p1"]
    }

    mock_conn = MagicMock()
    chat_service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn

    chat_service.log_time = MagicMock()

    await chat_service._background_message_insert(rows)

    chat_service.sql_loader.insert_chat_history_bulk.assert_called_once_with(rows)
    chat_service.adb_client.execute_multiple_non_query.assert_called_once_with(
        mock_conn, "INSERT", ["p1"]
    )
    chat_service.log_time.assert_called_once()

@pytest.mark.asyncio
async def test_background_file_write_success(chat_service, monkeypatch):

    monkeypatch.setattr(
        "app.services.chat_service.ensure_fetch_first_clause",
        MagicMock(return_value="LIMITED_SQL")
    )

    mock_oci = MagicMock()
    monkeypatch.setattr(
        "app.services.chat_service.OCIObjectStorageClient",
        MagicMock(return_value=mock_oci)
    )

    mock_conn = MagicMock()
    chat_service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn

    chat_service.adb_client.stream_query_chunks.return_value = iter([["row1"]])

    await chat_service._background_file_write("SELECT * FROM T", "file.xlsx")

    chat_service.adb_client.stream_query_chunks.assert_called_once_with(
        mock_conn, "LIMITED_SQL"
    )

    mock_oci.stream_chunks_to_excel_and_upload.assert_called_once()

@pytest.mark.asyncio
async def test_background_file_write_exception(chat_service, monkeypatch):

    monkeypatch.setattr(
        "app.services.chat_service.ensure_fetch_first_clause",
        MagicMock(return_value="SQL")
    )

    monkeypatch.setattr(
        "app.services.chat_service.OCIObjectStorageClient",
        MagicMock(side_effect=Exception("OCI fail"))
    )

    result = await chat_service._background_file_write("SELECT * FROM T", "file.xlsx")

    assert result["status"] == 0
    assert result["llm_response"] == "Failed to load data"

def test_chat_runtime_cleanup_user_not_in_state(chat_service):
    state = SimpleNamespace(
        chat_history={},
        last_sql_queries={},
        user_chats={"other_user": ["c1"]}  # u1 NOT present
    )

    result = chat_service.chat_runtime_cleanup("u1", state)

    assert result["status"] == 0
    assert result["message"] == "No chat history found for user u1 in state"

def test_load_chat_history_success(chat_service):
    user_id = "u1"
    chat_id = "c_new"

    # -----------------------------------
    # Mock SQL loader
    # -----------------------------------
    chat_service.sql_loader.load_chat_history_by_id.return_value = {
        "load_chat_history": "SELECT * FROM chat_history"
    }

    # -----------------------------------
    # Create mock dataframe (covers PAR, SQL, NORMAL)
    # -----------------------------------
    df = pd.DataFrame([
        {"MESSAGE_NO": 3, "ROLE": "PAR", "MESSAGE": "par message"},
        {"MESSAGE_NO": 2, "ROLE": "SQL", "MESSAGE": "SELECT * FROM T"},
        {"MESSAGE_NO": 1, "ROLE": "USER", "MESSAGE": "hello"},
    ])

    chat_service.adb_client.get_connection.return_value.__enter__.return_value = MagicMock()
    chat_service.adb_client.execute_query_df.return_value = df

    # -----------------------------------
    # Mock dependencies
    # -----------------------------------
    chat_service.prompt_generator_client.generate_main_prompt.return_value = "MAIN_PROMPT"

    # SQL branch dependency
    chat_service.reconnnect_conn_and_get_df = MagicMock(
        return_value=pd.DataFrame([{"col": "value"}])
    )

    # -----------------------------------
    # Mock ensure_fetch_first_clause
    # -----------------------------------
    with patch("app.services.chat_service.ensure_fetch_first_clause", return_value="LIMITED_SQL"):

        # -----------------------------------
        # Prepare app_state with existing last_user_chat (to cover deletion branch)
        # -----------------------------------
        app_state = SimpleNamespace(
            last_user_chat={"u1": "old_chat"},
            chat_history={"old_chat": []},
            last_sql_queries={"old_chat": "OLD_SQL"}
        )

        result = chat_service.load_chat_history(user_id, chat_id, app_state)

    # -----------------------------------
    # Assertions
    # -----------------------------------
    assert result["status"] == 1
    assert result["chat_id"] == chat_id
    assert isinstance(result["system_response"], list)

    # State updated correctly
    assert app_state.last_user_chat[user_id] == chat_id
    assert chat_id in app_state.chat_history
    assert chat_id in app_state.last_sql_queries



def test_load_chat_history_sql_success_branch(chat_service):
    user_id = "u1"
    chat_id = "c1"

    chat_service.sql_loader.load_chat_history_by_id.return_value = {
        "load_chat_history": "SELECT 1"
    }

    # One SQL row only
    df = pd.DataFrame([
        {"MESSAGE_NO": 1, "ROLE": "SQL", "MESSAGE": "SELECT * FROM T"}
    ])

    chat_service.adb_client.get_connection.return_value.__enter__.return_value = MagicMock()
    chat_service.adb_client.execute_query_df.return_value = df

    chat_service.prompt_generator_client.generate_main_prompt.return_value = "MAIN_PROMPT"

    # Mock SQL execution success
    chat_service.reconnnect_conn_and_get_df = MagicMock(
        return_value=pd.DataFrame([{"col": "value"}])
    )

    app_state = SimpleNamespace(
        last_user_chat={},
        chat_history={},
        last_sql_queries={}
    )

    with patch("app.services.chat_service.ensure_fetch_first_clause", return_value="LIMITED_SQL"):
        result = chat_service.load_chat_history(user_id, chat_id, app_state)

    assert result["status"] == 1
    assert result["system_response"][0]["role"] == "SQL"
    assert result["system_response"][0]["message"] == [{"col": "value"}]






def test_load_chat_history_triggers_n_zero_break(monkeypatch):
    service = ChatService()

    service.sql_loader = MagicMock()
    service.adb_client = MagicMock()
    service.prompt_generator_client = MagicMock()

    service.sql_loader.load_chat_history_by_id.return_value = {
        "load_chat_history": "SELECT * FROM DUAL"
    }

    # 6 USER rows (to exhaust n)
    data = []
    for i in range(6):
        data.append({
            "MESSAGE_NO": i,
            "ROLE": "USER",
            "MESSAGE": f"Message {i}"
        })

    # Add ONE SQL row so sql_record doesn't fail
    data.append({
        "MESSAGE_NO": 7,
        "ROLE": "SQL",
        "MESSAGE": "SELECT * FROM T"
    })

    df = pd.DataFrame(data)

    service.adb_client.get_connection.return_value.__enter__.return_value = MagicMock()
    service.adb_client.execute_query_df.return_value = df

    service.prompt_generator_client.generate_main_prompt.return_value = "System Prompt"

    # Mock SQL execution success
    service.reconnnect_conn_and_get_df = MagicMock(
        return_value=pd.DataFrame([{"col": 1}])
    )

    monkeypatch.setattr(
        "app.services.chat_service.ensure_fetch_first_clause",
        lambda x, y: x
    )

    class AppState:
        chat_history = {}
        last_sql_queries = {}
        last_user_chat = {}

    app_state = AppState()

    result = service.load_chat_history("user1", "chat1", app_state)

    assert result["status"] == 1


def test_load_chat_history_sql_exception(monkeypatch):
    service = ChatService()

    service.sql_loader = MagicMock()
    service.adb_client = MagicMock()
    service.prompt_generator_client = MagicMock()

    service.sql_loader.load_chat_history_by_id.return_value = {
        "load_chat_history": "SELECT * FROM DUAL"
    }

    # One SQL row
    df = pd.DataFrame([
        {
            "MESSAGE_NO": 1,
            "ROLE": "SQL",
            "MESSAGE": "SELECT * FROM TEST"
        }
    ])

    service.adb_client.get_connection.return_value.__enter__.return_value = MagicMock()
    service.adb_client.execute_query_df.return_value = df

    service.prompt_generator_client.generate_main_prompt.return_value = "System Prompt"

    # Force SQL fetch to fail
    service.reconnnect_conn_and_get_df = MagicMock(side_effect=Exception("DB Error"))

    # Mock ensure_fetch_first_clause
    monkeypatch.setattr(
        "app.services.chat_service.ensure_fetch_first_clause",
        lambda x, y: x
    )

    class AppState:
        chat_history = {}
        last_sql_queries = {}
        last_user_chat = {}

    app_state = AppState()

    result = service.load_chat_history("user1", "chat1", app_state)

    assert result["status"] == 1

    # Ensure fallback message was inserted
    assert result["system_response"][0]["role"] == "SQL"
    assert result["system_response"][0]["message"] == "Failed to deliver message due to SQL error"



@pytest.mark.asyncio
async def test_handle_inquiry_guardrail_timeout(chat_service, app_state):
    with patch.object(chat_service, "_run_guardrail", return_value="LLM timeout error"), \
         patch("app.services.chat_service._PARALLEL_EXECUTOR.submit") as mock_submit:

        mock_submit.side_effect = lambda fn, *a: FakeFuture(fn(*a))

        result = await chat_service.handle_inquiry(
            "u1", "c1", "test message", app_state
        )

    assert result["status"] == 2
    assert "time out" in result["llm_response"]

@pytest.mark.asyncio
async def test_handle_inquiry_irrelevant(chat_service, app_state):
    guardrail_result = {"relevant": "no", "tables": []}

    with patch.object(chat_service, "_run_guardrail", return_value=guardrail_result), \
         patch.object(chat_service, "_run_text_2_sql", return_value={}), \
         patch("app.services.chat_service._PARALLEL_EXECUTOR.submit") as mock_submit:

        mock_submit.side_effect = lambda fn, *a: FakeFuture(fn(*a))

        result = await chat_service.handle_inquiry(
            "u1", "c1", "irrelevant query", app_state
        )

    assert result["status"] == 2
    assert "rejected" in result["llm_response"]

@pytest.mark.asyncio
async def test_handle_inquiry_sql_error_status(chat_service, app_state):
    guardrail = {"relevant": "yes", "tables": []}
    sql_result = {"sql_query": "SELECT 1", "error_status": 1}

    with patch.object(chat_service, "_run_guardrail", return_value=guardrail), \
         patch.object(chat_service, "_run_text_2_sql", return_value=sql_result), \
         patch("app.services.chat_service._PARALLEL_EXECUTOR.submit") as mock_submit:

        mock_submit.side_effect = lambda fn, *a: FakeFuture(fn(*a))

        result = await chat_service.handle_inquiry(
            "u1", "c1", "bad query", app_state
        )

    assert result["status"] == 2
    assert "Failed to generate query" in result["llm_response"]

@pytest.mark.asyncio
async def test_handle_inquiry_empty_df(chat_service, app_state):
    guardrail = {"relevant": "yes", "tables": []}
    sql_result = {"sql_query": "SELECT 1", "error_status": 0}

    empty_df = pd.DataFrame()

    with patch.object(chat_service, "_run_guardrail", return_value=guardrail), \
         patch.object(chat_service, "_run_text_2_sql", return_value=sql_result), \
         patch("app.services.chat_service._PARALLEL_EXECUTOR.submit") as mock_submit, \
         patch("app.services.chat_service.check_if_df_all_null_or_zero", return_value=True):

        mock_submit.side_effect = lambda fn, *a: FakeFuture(fn(*a))

        chat_service.adb_client.get_connection.return_value.__enter__.return_value = MagicMock()
        chat_service.adb_client.execute_query_df.return_value = empty_df
        chat_service.adb_client.execute_scalar.return_value = 0

        result = await chat_service.handle_inquiry(
            "u1", "c1", "query", app_state
        )

    assert result["status"] == 3
    assert "No data found" in result["llm_response"]

from app.services.chat_service import RAW_MESSAGE


def test_run_guardrail_returns_raw_message(chat_service):
    # Arrange
    chat_service.prompt_generator_client.guardrail_check_inference_call.return_value = RAW_MESSAGE

    # Act
    result = chat_service._run_guardrail("some query")

    # Assert
    assert result == RAW_MESSAGE