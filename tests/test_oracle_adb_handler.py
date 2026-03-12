# import pytest
# import pandas as pd
# from unittest.mock import MagicMock, patch

# from code_modules.oracle_adb_handler import OracleADBClient


# @pytest.fixture
# def mock_config():
#     """Mock ADWConfig object"""
#     config = MagicMock()
#     config.username = "user"
#     config.password = "password"
#     config.dsn = "dsn"
#     config.config_dir = "/config"
#     config.wallet_loc = "/wallet"
#     config.wallet_pw = "wallet_pw"
#     return config

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from code_modules.oracle_adb_handler import OracleADBClient


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.username = "user"
    config.password = "pass"
    config.dsn = "dsn"
    config.config_dir = "config_dir"
    config.wallet_loc = "wallet_loc"
    config.wallet_pw = "wallet_pw"
    return config


@pytest.fixture
def client(mock_config):
    return OracleADBClient(mock_config)


@pytest.fixture
def mock_pool():
    return MagicMock()


@pytest.fixture
def mock_connection():
    return MagicMock()


# -------------------------------------------------------------------
# init_pool / close_pool
# -------------------------------------------------------------------

def test_init_pool_creates_pool(client, mock_config):
    with patch("oracledb.create_pool") as mock_create_pool:
        mock_create_pool.return_value = MagicMock()

        client.init_pool()

        assert client._pool is not None
        mock_create_pool.assert_called_once()


def test_init_pool_noop_if_already_initialized(client):
    client._pool = MagicMock()

    with patch("oracledb.create_pool") as mock_create_pool:
        client.init_pool()
        mock_create_pool.assert_not_called()


def test_close_pool(client):
    mock_pool = MagicMock()
    client._pool = mock_pool

    client.close_pool()

    mock_pool.close.assert_called_once()
    assert client._pool is None


def test_close_pool_noop_if_none(client):
    client._pool = None
    client.close_pool()  # Should not raise


# -------------------------------------------------------------------
# get_connection context manager
# -------------------------------------------------------------------

def test_get_connection_success(client, mock_pool, mock_connection):
    client._pool = mock_pool
    mock_pool.acquire.return_value = mock_connection

    with client.get_connection() as conn:
        assert conn == mock_connection

    mock_pool.release.assert_called_once_with(mock_connection)


def test_get_connection_raises_if_pool_not_initialized(client):
    with pytest.raises(RuntimeError):
        with client.get_connection():
            pass


def test_get_connection_releases_on_exception(client, mock_pool, mock_connection):
    client._pool = mock_pool
    mock_pool.acquire.return_value = mock_connection

    with pytest.raises(ValueError):
        with client.get_connection():
            raise ValueError("boom")

    mock_pool.release.assert_called_once_with(mock_connection)


# -------------------------------------------------------------------
# execute_query_df
# -------------------------------------------------------------------

def test_execute_query_df_success(client, mock_connection):
    df = pd.DataFrame({"a": [1, 2]})

    with patch("pandas.read_sql", return_value=df) as mock_read_sql:
        result = client.execute_query_df(mock_connection, "SELECT 1")

        assert result.equals(df)
        mock_read_sql.assert_called_once()


def test_execute_query_df_with_params(client, mock_connection):
    df = pd.DataFrame({"a": [1]})

    with patch("pandas.read_sql", return_value=df) as mock_read_sql:
        client.execute_query_df(mock_connection, "SELECT 1 WHERE a=:1", params=[1])
        mock_read_sql.assert_called_once()


def test_execute_query_df_exception(client, mock_connection):
    with patch("pandas.read_sql", side_effect=Exception("fail")):
        with pytest.raises(Exception):
            client.execute_query_df(mock_connection, "SELECT 1")


# -------------------------------------------------------------------
# execute_query_row_count
# -------------------------------------------------------------------

def test_execute_query_row_count_success(client, mock_connection):
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (5,)
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    count = client.execute_query_row_count(mock_connection, "SELECT * FROM t")

    assert count == 5
    mock_cursor.execute.assert_called_once()


def test_execute_query_row_count_with_params(client, mock_connection):
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (3,)
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    count = client.execute_query_row_count(
        mock_connection,
        "SELECT * FROM t WHERE a=:1",
        params=[1]
    )

    assert count == 3


def test_execute_query_row_count_no_rows(client, mock_connection):
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    count = client.execute_query_row_count(mock_connection, "SELECT * FROM t")

    assert count == 0


def test_execute_query_row_count_exception(client, mock_connection):
    mock_connection.cursor.side_effect = Exception("fail")

    with pytest.raises(Exception):
        client.execute_query_row_count(mock_connection, "SELECT * FROM t")


# -------------------------------------------------------------------
# stream_query_chunks
# -------------------------------------------------------------------

def test_stream_query_chunks_success(client, mock_connection):
    mock_cursor = MagicMock()
    mock_cursor.description = [("col1",), ("col2",)]

    mock_cursor.fetchmany.side_effect = [
        [(1, "a"), (2, "b")],
        [(3, "c")],
        []
    ]

    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    chunks = list(client.stream_query_chunks(mock_connection, "SELECT * FROM t", chunk_size=2))

    assert len(chunks) == 2
    assert chunks[0].shape == (2, 2)
    assert chunks[1].shape == (1, 2)


def test_stream_query_chunks_with_params(client, mock_connection):
    mock_cursor = MagicMock()
    mock_cursor.description = [("col1",)]
    mock_cursor.fetchmany.side_effect = [[(1,)], []]

    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    chunks = list(
        client.stream_query_chunks(
            mock_connection,
            "SELECT * FROM t WHERE a=:1",
            params=[1]
        )
    )

    assert len(chunks) == 1


def test_stream_query_chunks_exception(client, mock_connection):
    mock_connection.cursor.side_effect = Exception("fail")

    with pytest.raises(Exception):
        list(client.stream_query_chunks(mock_connection, "SELECT * FROM t"))


# -------------------------------------------------------------------
# execute_multiple_non_query
# -------------------------------------------------------------------

def test_execute_multiple_non_query_success(client, mock_connection):
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    client.execute_multiple_non_query(
        mock_connection,
        "INSERT INTO t VALUES (:1)",
        params=[(1,), (2,)]
    )

    mock_cursor.executemany.assert_called_once()
    mock_connection.commit.assert_called_once()


def test_execute_multiple_non_query_no_params(client, mock_connection):
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    client.execute_multiple_non_query(mock_connection, "DELETE FROM t")

    mock_cursor.executemany.assert_called_once()
    mock_connection.commit.assert_called_once()


def test_execute_multiple_non_query_exception(client, mock_connection):
    mock_cursor = MagicMock()
    mock_cursor.executemany.side_effect = Exception("fail")
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    with pytest.raises(Exception):
        client.execute_multiple_non_query(
            mock_connection,
            "INSERT INTO t VALUES (:1)",
            params=[(1,)]
        )

    mock_connection.rollback.assert_called_once()


# -------------------------------------------------------------------
# execute_single_non_query
# -------------------------------------------------------------------

def test_execute_single_non_query_success(client, mock_connection):
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    client.execute_single_non_query(
        mock_connection,
        "INSERT INTO t VALUES (:1)",
        params=[1]
    )

    mock_cursor.execute.assert_called_once()
    mock_connection.commit.assert_called_once()


def test_execute_single_non_query_no_params(client, mock_connection):
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    client.execute_single_non_query(mock_connection, "DELETE FROM t")

    mock_cursor.execute.assert_called_once()
    mock_connection.commit.assert_called_once()


def test_execute_single_non_query_exception(client, mock_connection):
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = Exception("fail")
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    with pytest.raises(Exception):
        client.execute_single_non_query(
            mock_connection,
            "INSERT INTO t VALUES (:1)",
            params=[1]
        )

    mock_connection.rollback.assert_called_once()

import pytest
from unittest.mock import MagicMock
from code_modules.oracle_adb_handler import OracleADBClient


# ---------------------------------------------------------
# Helper to create mock connection + cursor
# ---------------------------------------------------------

def create_mock_conn(fetchone_result=None, execute_side_effect=None):
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = fetchone_result

    if execute_side_effect:
        mock_cursor.execute.side_effect = execute_side_effect

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    return mock_conn, mock_cursor


# ---------------------------------------------------------
# 1️⃣ Successful scalar execution
# ---------------------------------------------------------

def test_execute_scalar_success():
    mock_conn, mock_cursor = create_mock_conn(fetchone_result=(5,))

    result = OracleADBClient.execute_scalar(
        conn=mock_conn,
        query="SELECT COUNT(*) FROM DUAL",
    )

    assert result == 5
    mock_cursor.execute.assert_called_once()
    mock_cursor.fetchone.assert_called_once()


# ---------------------------------------------------------
# 2️⃣ fetchone() returns None → return 0
# ---------------------------------------------------------

def test_execute_scalar_no_result():
    mock_conn, mock_cursor = create_mock_conn(fetchone_result=None)

    result = OracleADBClient.execute_scalar(
        conn=mock_conn,
        query="SELECT COUNT(*) FROM DUAL",
    )

    assert result == 0


# ---------------------------------------------------------
# 3️⃣ Empty query raises ValueError
# ---------------------------------------------------------

def test_execute_scalar_empty_query():
    mock_conn = MagicMock()

    with pytest.raises(ValueError):
        OracleADBClient.execute_scalar(mock_conn, "")


# ---------------------------------------------------------
# 4️⃣ Whitespace query raises ValueError
# ---------------------------------------------------------

def test_execute_scalar_whitespace_query():
    mock_conn = MagicMock()

    with pytest.raises(ValueError):
        OracleADBClient.execute_scalar(mock_conn, "   ")


# ---------------------------------------------------------
# 5️⃣ Parameters passed correctly
# ---------------------------------------------------------

def test_execute_scalar_with_params():
    mock_conn, mock_cursor = create_mock_conn(fetchone_result=(10,))

    OracleADBClient.execute_scalar(
        conn=mock_conn,
        query="SELECT COUNT(*) FROM table WHERE id=:1",
        params=[123]
    )

    mock_cursor.execute.assert_called_once_with(
        "SELECT COUNT(*) FROM table WHERE id=:1",
        [123]
    )


# ---------------------------------------------------------
# 6️⃣ Exception inside execute → re-raised
# ---------------------------------------------------------

def test_execute_scalar_execute_exception():
    mock_conn, mock_cursor = create_mock_conn(
        execute_side_effect=Exception("DB error")
    )

    with pytest.raises(Exception):
        OracleADBClient.execute_scalar(
            conn=mock_conn,
            query="SELECT COUNT(*) FROM DUAL",
        )


# ---------------------------------------------------------
# 7️⃣ Exception inside fetchone → re-raised
# ---------------------------------------------------------

def test_execute_scalar_fetchone_exception():
    mock_conn, mock_cursor = create_mock_conn(fetchone_result=(1,))
    mock_cursor.fetchone.side_effect = Exception("Fetch error")

    with pytest.raises(Exception):
        OracleADBClient.execute_scalar(
            conn=mock_conn,
            query="SELECT COUNT(*) FROM DUAL",
        )