"""
This is Oracle adb handler module, All code interacting with
Oracle Autonoumous Database Warehourse.
"""
from typing import Optional, Sequence, Generator
from contextlib import contextmanager
import pandas as pd
import oracledb

from config_loader import ADWConfig , load_adw_config

_CHUNK_SIZE = 500
_POOL_MIN = 2
_POOL_MAX = 5
_POOL_INCREMENT = 1
_POOL_PING_INTERVAL = 60

class OracleADBClient:
    """
    Client for interacting with Oracle Autonomous Database (ADB).

    Responsibilities:
    - Execute SELECT queries and return results as pandas DataFrames
    - Execute DML/DDL statements with no returned output
    """

    def __init__(self, config: ADWConfig):
        self._config = config
        self._pool = None

    def init_pool(self) -> None:
        """
        Create the connection pool.
        Call this ONCE at application startup (e.g. from a FastAPI lifespan
        handler) so the pool is ready before the first request arrives and
        min connections are pre-warmed — eliminating first-request latency.
        Calling it more than once is a no-op.
        """
        if self._pool is not None:
            return

        print(
            f"[OracleADBClient] Creating connection pool "
            f"(min={_POOL_MIN}, max={_POOL_MAX}, "
            f"ping_interval={_POOL_PING_INTERVAL}s)"
        )
        self._pool = oracledb.create_pool(
            user=self._config.username,
            password=self._config.password,
            dsn=self._config.dsn,
            config_dir=self._config.config_dir,
            wallet_location=self._config.wallet_loc,
            wallet_password=self._config.wallet_pw,
            min=_POOL_MIN,
            max=_POOL_MAX,
            increment=_POOL_INCREMENT,
            ping_interval=_POOL_PING_INTERVAL,
        )
        print("[OracleADBClient] Pool created successfully")

    def close_pool(self) -> None:
        """
        Drain and close the connection pool.

        Call this at application shutdown (e.g. from a FastAPI lifespan
        handler) to release all DB connections cleanly.
        """
        if self._pool is not None:
            self._pool.close()
            self._pool = None
            print("[OracleADBClient] Pool closed")

    @contextmanager
    def get_connection(self):
        """
        Context manager that acquires a live connection from the pool and
        releases it back when the ``with`` block exits — even on exception.

        The pool automatically pings the connection before handing it over
        (controlled by ping_interval), so callers always receive a live,
        working connection. There is no need to manually reconnect after
        inactivity periods.

        Usage::

            with self.adb_client.get_connection() as conn:
                df = self.adb_client.execute_query_df(conn, sql)

        Raises:
            RuntimeError: If the pool has not been initialised yet.
        """
        if self._pool is None:
            raise RuntimeError(
                "Connection pool is not initialised. "
                "Call OracleADBClient.init_pool() at application startup."
            )
        conn = self._pool.acquire()
        try:
            yield conn
        finally:
            self._pool.release(conn)

    @staticmethod
    def execute_scalar(
        conn: oracledb.Connection,
        query: str,
        params: Optional[Sequence] = None,
    ) -> int:
        """
        Execute a SELECT query that returns a single scalar value
        (e.g., COUNT(*)).

        Args:
            conn (oracledb.Connection): Active DB connection
            query (str): SQL query returning single value
            params (Optional[Sequence]): Bind parameters

        Returns:
            int: Single scalar result
        """
        print("Executing scalar query")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params or [])
                result = cursor.fetchone()

                if result is None:
                    return 0

                value = result[0]

                print(f"Scalar query executed successfully, value: {value}")
                return value

        except Exception:
            print("Failed to execute scalar query")
            raise

    @staticmethod
    def execute_query_df(
        conn: oracledb.Connection,
        query: str,
        params: Optional[Sequence] = None,
    ):
        """
        Execute a SELECT query and return results as a pandas DataFrame.

        Args:
            query (str): SQL SELECT query
            params (Optional[Sequence]): Query bind parameters

        Returns:
            pd.DataFrame: Query result
        """
        print("Executing SELECT query")

        try:
            df = pd.read_sql(query, conn, params=params)
            print("Query executed successfully, rows fetched: %d", len(df))
            return df
        except Exception:
            print("Failed to execute SELECT query")
            raise
        # finally:
            # conn.close()

    @staticmethod
    def execute_query_row_count(
        conn: oracledb.Connection,
        query: str,
        params: Optional[Sequence] = None,
    ) -> int:
        """
        Execute a SELECT query and return only the row count without loading
        all data into memory.

        Args:
            conn: Active Oracle connection.
            query (str): SQL SELECT query.
            params (Optional[Sequence]): Query bind parameters.

        Returns:
            int: Number of rows the query would return.
        """
        count_query = f"SELECT COUNT(*) FROM ({query})"
        print("Executing row count query")
        try:
            with conn.cursor() as cursor:
                if params:
                    cursor.execute(count_query, params)
                else:
                    cursor.execute(count_query)
                row = cursor.fetchone()
                count = int(row[0]) if row else 0
            print(f"Row count: {count}")
            return count
        except Exception:
            print("Failed to execute row count query")
            raise

    @staticmethod
    def stream_query_chunks(
        conn: oracledb.Connection,
        query: str,
        params: Optional[Sequence] = None,
        chunk_size: int = _CHUNK_SIZE,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Stream a SELECT query result in fixed-size DataFrame chunks to avoid
        exhausting memory on large result sets.

        Each yielded DataFrame contains at most `chunk_size` rows. Column
        names are taken from the cursor description so the schema is
        preserved across chunks.

        Args:
            conn: Active Oracle connection.
            query (str): SQL SELECT query.
            params (Optional[Sequence]): Query bind parameters.
            chunk_size (int): Number of rows per chunk (default: 500).

        Yields:
            pd.DataFrame: A chunk of query results.
        """
        print(f"Streaming query in chunks of {chunk_size}")
        try:
            with conn.cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                columns = [col[0] for col in cursor.description]

                while True:
                    rows = cursor.fetchmany(chunk_size)
                    if not rows:
                        break
                    yield pd.DataFrame(rows, columns=columns)

        except Exception:
            print("Failed to stream query chunks")
            raise

    @staticmethod
    def execute_multiple_non_query(
        conn: oracledb.Connection,
        query: str,
        params: Optional[Sequence] = None,
    ) -> None:
        """
        Execute a non-SELECT query (INSERT, UPDATE, DELETE, DDL).

        Args:
            query (str): SQL statement
            params (Optional[Sequence]): Query bind parameters
        """
        print("Executing non-SELECT query")

        try:
            with conn.cursor() as cursor:
                if params:
                    cursor.executemany(query, params)
                else:
                    cursor.executemany(query)

            conn.commit()
            print("Query committed successfully")
        except Exception:
            conn.rollback()
            print("Failed to execute non-SELECT query")
            raise
        # finally:
        #     conn.close()

    @staticmethod
    def execute_single_non_query(
        conn: oracledb.Connection,
        query: str,
        params: Optional[Sequence] = None,
    ) -> None:
        """
        Execute a non-SELECT query (INSERT, UPDATE, DELETE, DDL).

        Args:
            query (str): SQL statement
            params (Optional[Sequence]): Query bind parameters
        """
        print("Executing non-SELECT query")

        try:
            with conn.cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

            conn.commit()
            print("Query committed successfully")
        except Exception:
            conn.rollback()
            print("Failed to execute non-SELECT query")
            raise
        # finally:
        #     conn.close()
