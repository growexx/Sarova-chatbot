import pytest
from code_modules.sql_queries_loader import SqlQueryLoader


# -------------------------------------------------------------------
# load_chat_history_by_id
# -------------------------------------------------------------------

def test_load_chat_history_by_id():
    result = SqlQueryLoader.load_chat_history_by_id("123")

    assert "load_chat_history" in result
    assert "FROM CHAT_MESSAGES" in result["load_chat_history"]
    assert "chat_id = '123'" in result["load_chat_history"]


# -------------------------------------------------------------------
# load_user_chats_previews
# -------------------------------------------------------------------

def test_load_user_chats_previews():
    result = SqlQueryLoader.load_user_chats_previews("user1")

    assert "load_chats_preview" in result
    assert "FROM USER_CHATS" in result["load_chats_preview"]
    assert "user_id = 'user1'" in result["load_chats_preview"]


# -------------------------------------------------------------------
# delete_chat_queries
# -------------------------------------------------------------------

def test_delete_chat_queries():
    result = SqlQueryLoader.delete_chat_queries("chat123")

    assert "delete_chat_history" in result
    assert "delete_chat_preview" in result
    assert "CHAT_MESSAGES" in result["delete_chat_history"]
    assert "USER_CHATS" in result["delete_chat_preview"]
    assert "chat123" in result["delete_chat_history"]
    assert "chat123" in result["delete_chat_preview"]


# -------------------------------------------------------------------
# insert_user_chat
# -------------------------------------------------------------------

def test_insert_user_chat_basic():
    result = SqlQueryLoader.insert_user_chat("user1", "chat1", "My Chat")

    query = result["insert_user_chat"]

    assert "INSERT INTO USER_CHATS" in query
    assert "'user1'" in query
    assert "'chat1'" in query
    assert "'My Chat'" in query


def test_insert_user_chat_escapes_single_quotes():
    result = SqlQueryLoader.insert_user_chat("user1", "chat1", "John's Chat")

    query = result["insert_user_chat"]

    # SQL escaping: ' becomes ''
    assert "John''s Chat" in query


# -------------------------------------------------------------------
# insert_chat_message
# -------------------------------------------------------------------

def test_insert_chat_message_basic():
    result = SqlQueryLoader.insert_chat_message(
        "chat1", 1, "USER", "Hello"
    )

    query = result["insert_chat_message"]

    assert "INSERT INTO CHAT_MESSAGES" in query
    assert "'chat1'" in query
    assert "1" in query
    assert "'USER'" in query
    assert "'Hello'" in query


def test_insert_chat_message_escapes_single_quotes():
    result = SqlQueryLoader.insert_chat_message(
        "chat1", 1, "USER", "It's working"
    )

    query = result["insert_chat_message"]

    assert "It''s working" in query


# -------------------------------------------------------------------
# last_sql_query_for_chat
# -------------------------------------------------------------------

def test_last_sql_query_for_chat():
    result = SqlQueryLoader.last_sql_query_for_chat("chat99")

    query = result["last_sql_query_of_chat"]

    assert "UPPER(ROLE) = 'SQL'" in query
    assert "FETCH FIRST 1 ROW ONLY" in query
    assert "chat_id = 'chat99'" in query


# -------------------------------------------------------------------
# insert_chat_history_bulk
# -------------------------------------------------------------------

def test_insert_chat_history_bulk():
    rows = [
        ("chat1", 1, "Hello", "USER"),
        ("chat1", 2, "Hi", "BOT"),
    ]

    result = SqlQueryLoader.insert_chat_history_bulk(rows)

    assert "query" in result
    assert "params" in result

    assert "INSERT INTO chat_messages" in result["query"]
    assert result["params"] == rows


# -------------------------------------------------------------------
# delete_chat_history_bulk
# -------------------------------------------------------------------

def test_delete_chat_history_bulk():
    rows = [("chat1",), ("chat2",)]

    result = SqlQueryLoader.delete_chat_history_bulk(rows)

    assert "query" in result
    assert "params" in result

    assert "DELETE FROM  chat_messages" in result["query"]
    assert result["params"] == rows


# -------------------------------------------------------------------
# get_last_message_no
# -------------------------------------------------------------------

def test_get_last_message_no():
    result = SqlQueryLoader.get_last_message_no("chat777")

    query = result["last_message_no"]

    assert "ORDER BY MESSAGE_NO DESC" in query.upper()
    assert "FETCH FIRST 1 ROW ONLY" in query.upper()
    assert "chat_id = 'chat777'" in query


# -------------------------------------------------------------------
# delete_all_chats_for_user
# -------------------------------------------------------------------

def test_delete_all_chats_for_user():
    result = SqlQueryLoader.delete_all_chats_for_user("user999")

    assert "delete_all_chats_for_user" in result
    assert "DELETE FROM USER_CHATS" in result["delete_all_chats_for_user"]
    assert "user_id = 'user999'" in result["delete_all_chats_for_user"]