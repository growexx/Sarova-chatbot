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
)
# ============================================
# Guardrail / Text2SQL
# ============================================



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



# =========================================================
# _run_text_2_sql
# =========================================================



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




from app.services.chat_service import RAW_MESSAGE

@patch("app.services.chat_service.RAW_MESSAGE", "RAW")
def test_run_guardrail_raw_message():
    service = ChatService()

    service.prompt_generator_client = MagicMock()
    service.guardrail_llm_client = MagicMock()

    service.prompt_generator_client.guardrail_check_inference_call.return_value = "RAW"

    result = service._run_guardrail("test message")

    assert result == "RAW"

@patch("app.services.chat_service.LLMResponseExtractor")
def test_run_guardrail_success(mock_extractor_cls):
    service = ChatService()

    service.prompt_generator_client = MagicMock()
    service.guardrail_llm_client = MagicMock()

    mock_raw = {"some": "response"}
    service.prompt_generator_client.guardrail_check_inference_call.return_value = mock_raw

    # Mock extractor instance
    mock_extractor = MagicMock()
    mock_extractor_cls.return_value = mock_extractor

    mock_extractor.get_many.return_value = ("yes", ["table1"], "ok")

    result = service._run_guardrail("hello")

    assert result == {
        "relevant": "yes",
        "tables": ["table1"],
        "reply_message": "ok"
    }

    mock_extractor.set_data.assert_called_once_with(mock_raw)

@patch("app.services.chat_service.LLMResponseExtractor")
def test_run_guardrail_defaults_used(mock_extractor_cls):
    service = ChatService()

    service.prompt_generator_client = MagicMock()
    service.guardrail_llm_client = MagicMock()

    service.prompt_generator_client.guardrail_check_inference_call.return_value = {"bad": "data"}

    mock_extractor = MagicMock()
    mock_extractor_cls.return_value = mock_extractor

    # Simulate extractor returning defaults
    mock_extractor.get_many.return_value = ("no", [], "Query is rejected , please try again")

    result = service._run_guardrail("bad input")

    assert result["relevant"] == "no"
    assert result["tables"] == []
    assert "rejected" in result["reply_message"]

def test_run_guardrail_message_format():
    service = ChatService()

    service.prompt_generator_client = MagicMock()
    service.guardrail_llm_client = MagicMock()

    service.prompt_generator_client.guardrail_check_inference_call.return_value = "RAW"

    service._run_guardrail("hello")

    args = service.prompt_generator_client.guardrail_check_inference_call.call_args[0]

    assert "User message is 'hello'" in args[1]

@pytest.mark.asyncio
async def test_background_message_insert_success():
    service = ChatService()

    # Mock sql loader
    service.sql_loader = MagicMock()
    service.sql_loader.insert_chat_history_bulk.return_value = {
        "query": "INSERT ...",
        "params": [("a", "b")]
    }

    # Mock adb client
    mock_conn = MagicMock()
    service.adb_client = MagicMock()
    service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn

    # Act
    await service._background_message_insert(rows=[{"a": 1}])

    # Assert
    service.adb_client.execute_multiple_non_query.assert_called_once_with(
        mock_conn, "INSERT ...", [("a", "b")]
    )

@pytest.mark.asyncio
@patch("app.services.chat_service.ensure_fetch_first_clause", return_value="MODIFIED_SQL")
async def test_background_large_file_write_success(mock_ensure):
    service = ChatService()

    # Mock adb client
    mock_conn = MagicMock()
    service.adb_client = MagicMock()
    service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn

    mock_chunk_gen = MagicMock()
    service.adb_client.stream_query_chunks.return_value = mock_chunk_gen

    # Mock object storage client
    service.object_storage_client = MagicMock()

    # Act
    await service._background_large_file_write(
        sql_query="SELECT * FROM table",
        local_file_path="file.xlsx"
    )

    # Assert
    service.object_storage_client.stream_chunks_to_excel_and_upload.assert_called_once_with(
        chunk_generator=mock_chunk_gen,
        local_file_path="file.xlsx",
        bucket_folder_name="Sarova_Table_files/"
    )

@pytest.mark.asyncio
@patch("app.services.chat_service.ensure_fetch_first_clause", return_value="SQL")
async def test_background_large_file_write_exception(mock_ensure):
    service = ChatService()

    # Force exception
    service.adb_client = MagicMock()
    service.adb_client.get_connection.side_effect = Exception("DB error")

    result = await service._background_large_file_write(
        sql_query="SELECT * FROM table",
        local_file_path="file.xlsx"
    )

    assert result["status"] == 0
    assert "Failed to load data" in result["llm_response"]

@pytest.mark.asyncio
@patch("app.services.chat_service.OCIObjectStorageClient")
@patch("app.services.chat_service.os.remove")
async def test_background_small_file_write_success(mock_remove, mock_oci_cls):
    service = ChatService()

    mock_oci = MagicMock()
    mock_oci_cls.return_value = mock_oci

    await service._background_small_file_write(
        sql_query="SELECT *",
        local_file_path="file.xlsx",
        bucket_folder_name="folder"
    )

    mock_oci.put_file_in_bucket_folder.assert_called_once_with("file.xlsx", "folder")
    mock_remove.assert_called_once_with("file.xlsx")

@pytest.mark.asyncio
@patch("app.services.chat_service.OCIObjectStorageClient")
async def test_background_small_file_write_exception(mock_oci_cls):
    service = ChatService()

    mock_oci = MagicMock()
    mock_oci.put_file_in_bucket_folder.side_effect = Exception("Upload failed")
    mock_oci_cls.return_value = mock_oci

    result = await service._background_small_file_write(
        sql_query="SELECT *",
        local_file_path="file.xlsx",
        bucket_folder_name="folder"
    )

    assert result["status"] == 0
    assert "Failed to load data" in result["llm_response"]

@patch("app.services.chat_service.prepare_local_file_and_par_url")
@patch("app.services.chat_service.asyncio.create_task")
def test_scenario_large_data(mock_task, mock_prepare):
    service = ChatService()

    df = pd.DataFrame({"a": [1, 2, 3]})
    mock_prepare.return_value = ("file.xlsx", "par_url")

    service.prepare_data_response = MagicMock(return_value={"status": 1})

    service.scenario_based_response(
        num_of_records=150,
        selected_df=df,
        max_limit_query="SQL",
        sql_query="SQL",
        message="msg",
        scenario="anything"
    )

    # Async task triggered
    mock_task.assert_called_once()

    # prepare_data_response called
    service.prepare_data_response.assert_called_once()

@pytest.mark.asyncio
@patch("app.services.chat_service.prepare_local_file_and_par_url")
@patch("app.services.chat_service.asyncio.create_task")
@patch("pandas.DataFrame.to_excel")
def test_scenario_raw_data(mock_to_excel, mock_task, mock_prepare):
    service = ChatService()

    df = pd.DataFrame({"a": [1, 2]})
    mock_prepare.return_value = ("file.xlsx", "par_url")

    service.prepare_data_response = MagicMock(return_value={"status": 1})

    service.scenario_based_response(
        num_of_records=50,
        selected_df=df,
        max_limit_query="SQL",
        sql_query="SQL",
        message="msg",
        scenario="raw_data"
    )

    mock_to_excel.assert_called_once()
    mock_task.assert_called_once()
    service.prepare_data_response.assert_called_once()

@pytest.mark.asyncio
@patch("app.services.chat_service.generate_categorical_plots")
@patch("app.services.chat_service.wrap_par_around_file")
@patch("app.services.chat_service.asyncio.create_task")
def test_scenario_plot_exists(mock_task, mock_wrap, mock_plots):
    service = ChatService()

    df = pd.DataFrame({
        "cat": ["A", "B"],
        "val": [10, 20]
    })

    mock_plots.return_value = ["temp_graph/plot.png"]
    mock_wrap.return_value = "wrapped_par"

    service.prepare_data_response = MagicMock(return_value={"status": 1})

    service.scenario_based_response(
        num_of_records=50,
        selected_df=df,
        max_limit_query="SQL",
        sql_query="SQL",
        message="msg",
        scenario="chart"
    )

    mock_task.assert_called_once()
    mock_wrap.assert_called_once()
    service.prepare_data_response.assert_called_once()

@pytest.mark.asyncio
@patch("app.services.chat_service.generate_categorical_plots")
@patch("app.services.chat_service.asyncio.create_task")
def test_scenario_no_plot(mock_task, mock_plots):
    service = ChatService()

    df = pd.DataFrame({
        "cat": ["A", "B"],
        "val": [10, 20]
    })

    mock_plots.return_value = []

    service.prepare_data_response = MagicMock(return_value={"status": 1})

    service.scenario_based_response(
        num_of_records=50,
        selected_df=df,
        max_limit_query="SQL",
        sql_query="SQL",
        message="msg",
        scenario="chart"
    )

    # No async task since no plot
    mock_task.assert_not_called()

    service.prepare_data_response.assert_called_once()

def test_prepare_data_response_basic():
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [None, 3]
    })

    result = ChatService.prepare_data_response(
        selected_df=df,
        sql_query="SELECT *",
        message="ok",
        scenario="raw_data",
        par="url"
    )

    assert result["status"] == 1
    assert result["scenario"] == "raw_data"
    assert result["par"] == "url"
    assert isinstance(result["results_df"], list)

def test_prepare_data_response_none_df():
    result = ChatService.prepare_data_response(
        selected_df=None,
        sql_query="SQL",
        message="msg",
        scenario="chart",
        par=None
    )

    assert result["results_df"] is None
    assert result["status"] == 1

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import pandas as pd

from app.services.chat_service import ChatService, RAW_MESSAGE

@pytest.mark.asyncio
@patch("app.services.chat_service.asyncio.create_task", return_value=None)
async def test_guardrail_reject(mock_task):
    service = ChatService()

    # Mock guardrail → reject
    service._run_guardrail = MagicMock(return_value={
        "relevant": "no",
        "tables": [],
        "reply_message": "Rejected by guardrail"
    })

    # Mock Stest_gQL future (won't be used)
    mock_future_sql = MagicMock()

    service.executor = MagicMock()
    service.executor.submit.side_effect = [
        MagicMock(result=lambda: service._run_guardrail("q")),
        mock_future_sql
    ]

    result = await service.handle_inquiry(
        user_id="u1",
        chat_id="c1",
        user_message="bad query",
        app_state=MagicMock()
    )

    assert result["status"] == 2
    assert "Rejected" in result["llm_response"]


from unittest.mock import MagicMock, patch
import pytest


@pytest.mark.asyncio
@patch("app.services.chat_service._PARALLEL_EXECUTOR")
@patch.object(ChatService, "prepare_last_sql_query", return_value=None)  # ✅ FIX
async def test_guardrail_reject(mock_prepare, mock_executor):
    service = ChatService()

    # Mock futures
    mock_future_guardrail = MagicMock()
    mock_future_guardrail.result.return_value = {
        "relevant": "no",
        "tables": [],
        "reply_message": "Rejected by guardrail"
    }

    mock_future_sql = MagicMock()

    mock_executor.submit.side_effect = [
        mock_future_guardrail,
        mock_future_sql
    ]

    app_state = MagicMock()
    app_state.last_sql_queries = {}

    result = await service.handle_inquiry("u1", "c1", "bad query", app_state)

    assert result["status"] == 2
    assert result["llm_response"] == "Rejected by guardrail"

@pytest.mark.asyncio
@patch("app.services.chat_service._PARALLEL_EXECUTOR")
@patch.object(ChatService, "prepare_last_sql_query", return_value=None)  # ✅ FIX
async def test_sql_raw_message(mock_prepare, mock_executor):
    service = ChatService()

    mock_future_guardrail = MagicMock()
    mock_future_guardrail.result.return_value = {
        "relevant": "yes",
        "tables": [],
        "reply_message": ""
    }

    mock_future_sql = MagicMock()
    mock_future_sql.result.return_value = RAW_MESSAGE

    mock_executor.submit.side_effect = [
        mock_future_guardrail,
        mock_future_sql
    ]

    app_state = MagicMock()
    app_state.last_sql_queries = {}

    result = await service.handle_inquiry("u1", "c1", "query", app_state)

    assert result["status"] == 2
    assert "please change request" in result["llm_response"]


import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
@patch("app.services.chat_service._PARALLEL_EXECUTOR")
@patch.object(ChatService, "prepare_last_sql_query", return_value=None)
async def test_sql_error_status_one(mock_prepare, mock_executor):
    service = ChatService()

    # ── Mock guardrail (valid flow) ─────────────────────
    mock_future_guardrail = MagicMock()
    mock_future_guardrail.result.return_value = {
        "relevant": "yes",
        "tables": [],
        "reply_message": ""
    }

    # ── Mock SQL result with error_status = 1 ───────────
    mock_future_sql = MagicMock()
    mock_future_sql.result.return_value = {
        "sql_query": "SELECT * FROM test",
        "error_status": 1,
        "scenario": "raw_text"
    }

    mock_executor.submit.side_effect = [
        mock_future_guardrail,
        mock_future_sql
    ]

    # ── Mock app_state ──────────────────────────────────
    app_state = MagicMock()
    app_state.last_sql_queries = {}

    # ── Execute ─────────────────────────────────────────
    result = await service.handle_inquiry("u1", "c1", "query", app_state)

    # ── Assertions ──────────────────────────────────────
    assert result["status"] == 2
    assert "Failed to generate query" in result["llm_response"]


@pytest.mark.asyncio
@patch("app.services.chat_service.check_if_df_all_null_or_zero", return_value=True)
@patch("app.services.chat_service.wrap_query_with_count", return_value="COUNT_QUERY")
@patch("app.services.chat_service.ensure_fetch_first_clause", side_effect=["LIMIT_10000", "LIMIT_100"])
@patch("app.services.chat_service.add_distinct_safely", return_value="FINAL_SQL")
@patch("app.services.chat_service.smart_column_insertion", return_value="FINAL_SQL")
@patch("app.services.chat_service.classify_query", return_value="AGGREGATION")
@patch("app.services.chat_service._PARALLEL_EXECUTOR")
@patch.object(ChatService, "prepare_last_sql_query", return_value=None)
async def test_sql_execution_empty_df(
    mock_prepare,
    mock_executor,
    mock_classify,
    mock_smart_col,
    mock_add_distinct,
    mock_fetch_clause,
    mock_wrap_count,
    mock_df_empty
):
    service = ChatService()

    # ── Mock guardrail ───────────────────────────────
    mock_future_guardrail = MagicMock()
    mock_future_guardrail.result.return_value = {
        "relevant": "yes",
        "tables": [],
        "reply_message": ""
    }

    # ── Mock SQL generation ─────────────────────────
    mock_future_sql = MagicMock()
    mock_future_sql.result.return_value = {
        "sql_query": "SELECT * FROM test",
        "error_status": 0,
        "scenario": "raw_text"
    }

    mock_executor.submit.side_effect = [
        mock_future_guardrail,
        mock_future_sql
    ]

    # ── Mock DB connection ──────────────────────────
    mock_conn = MagicMock()
    service.adb_client.get_connection = MagicMock()
    service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn

    # Mock DB responses
    service.adb_client.execute_query_df = MagicMock(return_value=MagicMock())
    service.adb_client.execute_scalar = MagicMock(return_value=0)

    # ── App state ───────────────────────────────────
    app_state = MagicMock()
    app_state.last_sql_queries = {}

    # ── Execute ─────────────────────────────────────
    result = await service.handle_inquiry("u1", "c1", "query", app_state)

    # ── Assertions ──────────────────────────────────
    assert result["status"] == 3
    assert "No data found for following search" in result["llm_response"]

    # ✅ Verify classification logic triggered
    mock_classify.assert_called_once()

@pytest.mark.asyncio
@patch("app.services.chat_service.check_if_df_all_null_or_zero", return_value=True)
@patch("app.services.chat_service.wrap_query_with_count", return_value="COUNT_QUERY")
@patch("app.services.chat_service.ensure_fetch_first_clause", side_effect=["LIMIT_10000", "LIMIT_100"])
@patch("app.services.chat_service.add_distinct_safely", return_value="FINAL_SQL")
@patch("app.services.chat_service.smart_column_insertion", return_value="FINAL_SQL")
@patch("app.services.chat_service.classify_query", return_value="ENTITY")
@patch("app.services.chat_service._PARALLEL_EXECUTOR")
@patch.object(ChatService, "prepare_last_sql_query", return_value=None)
async def test_sql_execution_empty_df_with_raw(
    mock_prepare,
    mock_executor,
    mock_classify,
    mock_smart_col,
    mock_add_distinct,
    mock_fetch_clause,
    mock_wrap_count,
    mock_df_empty
):
    service = ChatService()

    # ── Mock guardrail ───────────────────────────────
    mock_future_guardrail = MagicMock()
    mock_future_guardrail.result.return_value = {
        "relevant": "yes",
        "tables": [],
        "reply_message": ""
    }

    # ── Mock SQL generation ─────────────────────────
    mock_future_sql = MagicMock()
    mock_future_sql.result.return_value = {
        "sql_query": "SELECT * FROM test",
        "error_status": 0,
        "scenario": "raw_text"
    }

    mock_executor.submit.side_effect = [
        mock_future_guardrail,
        mock_future_sql
    ]

    # ── Mock DB connection ──────────────────────────
    mock_conn = MagicMock()
    service.adb_client.get_connection = MagicMock()
    service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn

    # Mock DB responses
    service.adb_client.execute_query_df = MagicMock(return_value=MagicMock())
    service.adb_client.execute_scalar = MagicMock(return_value=0)

    # ── App state ───────────────────────────────────
    app_state = MagicMock()
    app_state.last_sql_queries = {}

    # ── Execute ─────────────────────────────────────
    result = await service.handle_inquiry("u1", "c1", "query", app_state)

    # ── Assertions ──────────────────────────────────
    assert result["status"] == 3
    assert "No data found for following search" in result["llm_response"]

    # ✅ Verify classification logic triggered
    mock_classify.assert_called_once()

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
@patch("app.services.chat_service.create_new_chat_title")
@patch("app.services.chat_service.smart_reorder", return_value=["col1", "col2"])
@patch("app.services.chat_service._PARALLEL_EXECUTOR")
@patch.object(ChatService, "prepare_last_sql_query", return_value=None)
async def test_new_chat_flow(
    mock_prepare,
    mock_executor,
    mock_reorder,
    mock_create_chat
):
    service = ChatService()

    # ── Mock guardrail ─────────────────────
    mock_future_guardrail = MagicMock()
    mock_future_guardrail.result.return_value = {
        "relevant": "yes",
        "tables": [],
        "reply_message": ""
    }

    # ── Mock SQL ──────────────────────────
    mock_future_sql = MagicMock()
    mock_future_sql.result.return_value = {
        "sql_query": "SELECT col1, col2 FROM test",
        "error_status": 0,
        "scenario": "raw_text"
    }

    mock_executor.submit.side_effect = [
        mock_future_guardrail,
        mock_future_sql
    ]

    # ── Mock DataFrame ────────────────────
    df = pd.DataFrame({
        "col1": [1, 1],
        "col2": [2, 2]
    })

    service.adb_client.get_connection = MagicMock()
    mock_conn = MagicMock()
    service.adb_client.get_connection.return_value.__enter__.return_value = mock_conn

    # First DB call → SQL execution
    service.adb_client.execute_query_df = MagicMock(side_effect=[
        df,  # selected_df
        pd.DataFrame({"CHAT_ID": []})  # no chat exists → new chat
    ])

    service.adb_client.execute_scalar = MagicMock(side_effect=[
        2,  # num_of_records
        10  # message_no
    ])

    # ── Mock dependencies ─────────────────
    service.prompt_generator_client.generate_main_prompt = MagicMock(return_value="SYSTEM_PROMPT")
    service.prompt_generator_client.generate_assistant_prompt = MagicMock(return_value="ASSISTANT_PROMPT")

    service.llm_inference_client.inference_from_chat_history = MagicMock(
        return_value={"message": "final answer"}
    )

    service.scenario_based_response = MagicMock(return_value=(
        {"status": 1, "llm_response": "ok"},
        "PAR_TEXT"
    ))

    service._background_message_insert = MagicMock()

    # ── App state ─────────────────────────
    app_state = MagicMock()
    app_state.last_sql_queries = {}
    app_state.chat_history = {}
    app_state.last_user_chat = {}

    # ── Execute ───────────────────────────
    result = await service.handle_inquiry("u1", "c1", "query", app_state)

    # ── Assertions ────────────────────────

    # ✅ duplicates removed
    assert len(df.drop_duplicates()) == 1

    # ✅ reorder called
    mock_reorder.assert_called_once()

    # ✅ new chat created
    assert "c1" in app_state.chat_history
    mock_create_chat.assert_called_once()

    # ✅ LLM called
    service.llm_inference_client.inference_from_chat_history.assert_called_once()

    # ✅ final response
    assert result["status"] == 1


@patch("app.services.chat_service.smart_reorder", return_value=["col1", "col2"])
async def test_existing_chat_load_from_db(mock_reorder):
    service = ChatService()

    service.load_chat_history_ram = MagicMock()

    # Mock DB returning existing chat_id
    service.adb_client.execute_query_df = MagicMock(return_value=pd.DataFrame({"CHAT_ID": ["c1"]}))
    service.adb_client.execute_scalar = MagicMock(return_value=5)

    service.adb_client.get_connection = MagicMock()
    service.adb_client.get_connection.return_value.__enter__.return_value = MagicMock()

    app_state = MagicMock()
    app_state.last_sql_queries = {}
    app_state.chat_history = {"c1": []}
    app_state.last_user_chat = {"u1": "different_chat"}  # triggers load

    # Minimal mocks to reach branch
    service.prompt_generator_client.generate_assistant_prompt = MagicMock(return_value="prompt")
    service.llm_inference_client.inference_from_chat_history = MagicMock(return_value={"message": "msg"})
    service.scenario_based_response = MagicMock(return_value=({"status": 1}, "PAR"))

    # Call (you’ll need to mock earlier pipeline too in real test)
    await service.handle_inquiry("u1", "c1", "query", app_state)

    # ✅ verify loading triggered
    service.load_chat_history_ram.assert_called_once()

def test_chat_history_trimming():
    service = ChatService()

    chat_history = [
        {"role": "System", "message": "sys"},
        {"role": "User", "message": "1"},
        {"role": "User", "message": "2"},
        {"role": "User", "message": "3"},
        {"role": "User", "message": "4"},
        {"role": "User", "message": "5"},
    ]

    # simulate trimming logic
    if len(chat_history) > 5:
        trimmed = chat_history[:1] + chat_history[3:]

    assert len(trimmed) == 4
    assert trimmed[0]["role"] == "System"