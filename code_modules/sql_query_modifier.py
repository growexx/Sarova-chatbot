"""
This Module is focusedd on modifying laready existing queries for execution.
"""
import re

def is_count_query(sql: str) -> bool:
    """
    Determine whether a SQL statement is a COUNT query.

    This function checks if the provided SQL string begins with a
    `SELECT COUNT(...)` expression (case-insensitive), allowing for
    leading whitespace.

    Args:
        sql (str): The SQL query string to evaluate.

    Returns:
        bool: True if the query appears to be a COUNT query,
              otherwise False.

    Notes:
        - The check is pattern-based using a regular expression.
        - It does not perform full SQL parsing.
        - Complex queries (e.g., nested SELECT COUNT in subqueries)
          may not be accurately detected.
    """
    pattern = r'^\s*select\s+count\s*\('
    return re.search(pattern, sql, re.IGNORECASE) is not None

def add_distinct_safely(sql: str) -> str:
    """
    Safely add DISTINCT to a SQL SELECT query.

    This function modifies a SQL query string to ensure that results
    are distinct while avoiding duplication of an existing DISTINCT
    clause.

    Behavior:
        1. If the query already contains `SELECT DISTINCT`, it is returned unchanged.
        2. If the query is a COUNT query (e.g., SELECT COUNT(column)),
           it is transformed into COUNT(DISTINCT column).
        3. Otherwise, DISTINCT is inserted after the SELECT keyword.

    Args:
        sql (str): The original SQL query string.

    Returns:
        str: The modified SQL query with DISTINCT applied where appropriate.

    Notes:
        - The transformation is based on regular expressions and does not
          perform full SQL parsing.
        - Only simple COUNT(column) patterns are supported.
        - Complex queries, nested subqueries, or advanced SQL constructs
          may not be handled correctly.
        - Input SQL is stripped of leading/trailing whitespace before processing.
    """
    sql = sql.strip()

    # If already DISTINCT
    if re.search(r'^\s*select\s+distinct', sql, re.IGNORECASE):
        return sql

    # Case 1: COUNT query
    if is_count_query(sql):
        return re.sub(
            r'count\s*\(\s*(\w+)\s*\)',
            r'COUNT(DISTINCT \1)',
            sql,
            flags=re.IGNORECASE
        )
    # Case 2: Normal select
    return re.sub(
        r'^\s*select\s+',
        'SELECT DISTINCT ',
        sql,
        count=1,
        flags=re.IGNORECASE
    )

def ensure_fetch_first_clause(query: str, limit: int = 100) -> str:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    query_clean = query.strip().rstrip(";")
    # Normalize whitespace
    query_clean = re.sub(r"\s+", " ", query_clean)

    fetch_pattern = re.compile(
        r"fetch\s+(first|next)\s+\d+\s+rows?\s+only\s*$",
        re.IGNORECASE
    )

    offset_pattern = re.compile(
        r"\boffset\s+\d+\s+rows\b",
        re.IGNORECASE
    )
    rownum_pattern = re.compile(
        r"\brownum\b",
        re.IGNORECASE
    )
    # If query already limited
    if (
        fetch_pattern.search(query_clean)
        or offset_pattern.search(query_clean)
        or rownum_pattern.search(query_clean)
    ):
        return query_clean

    # Otherwise append limit
    return f"{query_clean} FETCH FIRST {limit} ROWS ONLY"

def wrap_query_with_count(query: str) -> str:
    """
    Wraps a SQL query inside:
    SELECT COUNT(*) FROM (<query>)

    Args:
        query (str): Original SQL query

    Returns:
        str: Count wrapped query
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    query_clean = query.strip().rstrip(";")

    return f"SELECT COUNT(*) FROM ({query_clean})"

