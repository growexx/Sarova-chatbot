# tests/test_sql_query_modifier.py
import pytest
from code_modules.sql_query_modifier import ensure_fetch_first_clause, wrap_query_with_count

# -----------------------------
# Tests for ensure_fetch_first_clause
# -----------------------------
def test_ensure_fetch_adds_fetch_if_missing():
    query = "SELECT * FROM employees"
    result = ensure_fetch_first_clause(query, limit=50)
    assert result == "SELECT * FROM employees FETCH FIRST 50 ROWS ONLY"

def test_ensure_fetch_keeps_existing_fetch():
    query = "SELECT * FROM employees FETCH FIRST 10 ROWS ONLY"
    result = ensure_fetch_first_clause(query)
    # Should keep the first fetch and not duplicate
    assert result == "SELECT * FROM employees FETCH FIRST 10 ROWS ONLY"

def test_ensure_fetch_removes_extra_fetches():
    query = "SELECT * FROM employees FETCH FIRST 5 ROWS ONLY FETCH NEXT 10 ROWS ONLY"
    result = ensure_fetch_first_clause(query)
    # Only the first FETCH should remain
    assert result == "SELECT * FROM employees FETCH FIRST 5 ROWS ONLY"

def test_ensure_fetch_respects_rownum():
    query = "SELECT * FROM employees WHERE ROWNUM < 5"
    result = ensure_fetch_first_clause(query)
    # If ROWNUM exists, it should not add FETCH
    assert result == "SELECT * FROM employees WHERE ROWNUM < 5"

def test_ensure_fetch_respects_offset():
    query = "SELECT * FROM employees OFFSET 10 ROWS"
    result = ensure_fetch_first_clause(query)
    # If OFFSET exists, it should not add FETCH
    assert result == "SELECT * FROM employees OFFSET 10 ROWS"

def test_ensure_fetch_handles_extra_whitespace_and_semicolon():
    query = "   SELECT   *   FROM employees;   "
    result = ensure_fetch_first_clause(query, limit=20)
    assert result == "SELECT * FROM employees FETCH FIRST 20 ROWS ONLY"

def test_ensure_fetch_empty_query_raises():
    with pytest.raises(ValueError):
        ensure_fetch_first_clause("")

# -----------------------------
# Tests for wrap_query_with_count
# -----------------------------
def test_wrap_query_basic():
    query = "SELECT * FROM employees"
    result = wrap_query_with_count(query)
    assert result == "SELECT COUNT(*) FROM (SELECT * FROM employees)"

def test_wrap_query_removes_trailing_semicolon():
    query = "SELECT * FROM employees;"
    result = wrap_query_with_count(query)
    assert result == "SELECT COUNT(*) FROM (SELECT * FROM employees)"

def test_wrap_query_empty_query_raises():
    with pytest.raises(ValueError):
        wrap_query_with_count("")

def test_wrap_query_with_complex_query():
    query = "SELECT id, name FROM employees WHERE dept_id = 10 ORDER BY name"
    result = wrap_query_with_count(query)
    assert result == "SELECT COUNT(*) FROM (SELECT id, name FROM employees WHERE dept_id = 10 ORDER BY name)"