import pytest
from code_modules.llm_response_extractor import extract_json, LLMResponseExtractor


def test_extract_json_plain_json():
    text = '{"a": 1, "b": "test"}'
    result = extract_json(text)
    assert result == {"a": 1, "b": "test"}


def test_extract_json_embedded_json():
    text = "Here is some text {\"x\": 10, \"y\": 20} end"
    result = extract_json(text)
    assert result == {"x": 10, "y": 20}


def test_extract_json_fenced_json():
    text = """
    ```json
    {
        "name": "ashish",
        "age": 25
    }
    ```
    """
    result = extract_json(text)
    assert result["name"] == "ashish"
    assert result["age"] == 25


def test_extract_json_invalid_json():
    text = "this is not json"

    result = extract_json(text)

    assert "response" in result
    assert "error" in result
    assert result["response"] == text
    assert isinstance(result["error"], str)


def test_llm_response_extractor_set_and_get():
    text = '{"key1": "value1", "key2": 2}'
    extractor = LLMResponseExtractor()
    extractor.set_data(text)

    assert extractor.get("key1") == "value1"
    assert extractor.get("missing") is None
    assert extractor.get("missing", "default") == "default"


def test_llm_response_extractor_get_many():
    text = '{"a": 1, "b": 2}'
    extractor = LLMResponseExtractor()
    extractor.set_data(text)

    result = extractor.get_many(["a", "b", "c"], defaults={"c": 3})
    assert result == (1, 2, 3)




import pytest

from code_modules.llm_response_extractor import (
    smart_reorder,
    smart_column_insertion,
    extract_select_columns,
    normalize_column,
    extract_alias,
    classify_query
)

@pytest.mark.parametrize("query,expected", [
    ("SELECT * FROM table", "ENTITY"),
    ("SELECT COUNT(*) FROM table", "AGGREGATION"),
    ("SELECT SUM(x) FROM table", "AGGREGATION"),
    ("SELECT x, COUNT(*) OVER(PARTITION BY y) FROM table", "WINDOW"),
    ("SELECT CASE WHEN x > 1 THEN COUNT(*) END FROM table", "KPI"),
    ("SELECT * FROM a JOIN b ON a.id=b.id", "JOIN"),
    ("SELECT * FROM table FETCH FIRST 10 ROWS ONLY", "TOP_N"),
    ("SELECT * FROM table ORDER BY col", "ORDERED_ENTITY"),
])
def test_classify_query(query, expected):
    assert classify_query(query) == expected

def test_extract_select_columns_basic():
    query = "SELECT col1, col2 FROM table"
    result = extract_select_columns(query)

    assert result == ["col1", "col2"]


def test_extract_select_columns_distinct():
    query = "SELECT DISTINCT col1, col2 FROM table"
    result = extract_select_columns(query)

    assert result == ["col1", "col2"]


def test_extract_select_columns_no_match():
    query = "DELETE FROM table"
    result = extract_select_columns(query)

    assert result is None

@pytest.mark.parametrize("input_col,expected", [
    ("COL1", "col1"),
    (" table.col1 ", "col1"),
    ("alias.column_name", "column_name"),
])
def test_normalize_column(input_col, expected):
    assert normalize_column(input_col) == expected

def test_extract_alias_basic():
    query = "SELECT * FROM inventory_consumption_data t"
    assert extract_alias(query, "inventory_consumption_data") == "t"


def test_extract_alias_no_alias():
    query = "SELECT * FROM inventory_consumption_data"
    assert extract_alias(query, "inventory_consumption_data") is None


def test_extract_alias_sql_keyword():
    query = "SELECT * FROM inventory_consumption_data WHERE col = 1"
    assert extract_alias(query, "inventory_consumption_data") is None

from unittest.mock import patch


@patch("code_modules.llm_response_extractor.inventory_consumption_data_prirority", {
    "col1": 1,
    "col2": 2
})
def test_smart_reorder_inventory():
    columns = ["col2", "col1", "col3"]
    query = "SELECT * FROM inventory_consumption_data"

    result = smart_reorder(query, columns)

    assert result[0] == "col1"
    assert result[1] == "col2"
    assert result[-1] == "col3"  # unknown goes last


def test_smart_reorder_no_match():
    columns = ["b", "a"]
    query = "SELECT * FROM unknown_table"

    result = smart_reorder(query, columns)

    # No priority → default sort keeps order stable
    assert result == sorted(columns, key=lambda x: 999)

@patch("code_modules.llm_response_extractor.classify_query", return_value="AGGREGATION")
def test_smart_column_insertion_early_exit(mock_classify):
    query = "SELECT COUNT(*) FROM table"

    result = smart_column_insertion(query)

    assert result == query

@patch("code_modules.llm_response_extractor.classify_query", return_value="ENTITY")
@patch("code_modules.llm_response_extractor.extract_alias", return_value="t")
@patch("code_modules.llm_response_extractor.extract_select_columns", return_value=["t.col1"])
@patch("code_modules.llm_response_extractor.inventory_consumption_data_necessary_fields", ["col1", "col2"])
def test_smart_column_insertion_adds_columns(mock_extract_select, mock_alias, mock_classify):
    query = "SELECT col1 FROM inventory_consumption_data t"

    result = smart_column_insertion(query)

    assert "t.col2" in result
    assert "SELECT DISTINCT" in result

@patch("code_modules.llm_response_extractor.classify_query", return_value="ENTITY")
@patch("code_modules.llm_response_extractor.extract_alias", return_value=None)
@patch("code_modules.llm_response_extractor.extract_select_columns", return_value=["col1", "col2"])
@patch("code_modules.llm_response_extractor.inventory_consumption_data_necessary_fields", ["col1", "col2"])
def test_smart_column_insertion_no_change(mock_extract_select, mock_alias, mock_classify):
    query = "SELECT col1, col2 FROM inventory_consumption_data"

    result = smart_column_insertion(query)

    assert result == query

@patch("code_modules.llm_response_extractor.classify_query", return_value="ENTITY")
@patch("code_modules.llm_response_extractor.extract_select_columns", return_value=["col1"])
def test_smart_column_insertion_unknown_table(mock1, mock2):
    query = "SELECT col1 FROM unknown_table"

    # Should not crash
    result = smart_column_insertion(query)

    assert isinstance(result, str)

from unittest.mock import patch
from code_modules.llm_response_extractor import smart_reorder


@patch("code_modules.llm_response_extractor.occupancy_revenue_data_priority", {
    "col_a": 1,
    "col_b": 2
})
def test_smart_reorder_occupancy_revenue_data():
    columns = ["col_b", "col_a", "col_x"]
    query = "SELECT * FROM occupancy_revenue_data"

    result = smart_reorder(query, columns)

    # Expected: based on priority → col_a first, then col_b, unknown last
    assert result == ["col_a", "col_b", "col_x"]

@patch("code_modules.llm_response_extractor.supplier_scoring_data_priority", {
    "score": 1,
    "supplier_id": 2
})
def test_smart_reorder_supplier_scoring_data():
    columns = ["supplier_id", "score", "extra"]
    query = "SELECT * FROM supplier_scoring_data"

    result = smart_reorder(query, columns)

    assert result == ["score", "supplier_id", "extra"]



from unittest.mock import patch
from code_modules.llm_response_extractor import smart_column_insertion


@patch("code_modules.llm_response_extractor.classify_query", return_value="ENTITY")
@patch("code_modules.llm_response_extractor.extract_alias", return_value="t")
@patch("code_modules.llm_response_extractor.extract_select_columns", return_value=["t.col1"])
@patch("code_modules.llm_response_extractor.occupancy_revenue_data_necessary_fields", ["col1", "col2"])
def test_smart_column_insertion_occupancy(mock_select, mock_alias, mock_classify):
    query = "SELECT col1 FROM occupancy_revenue_data t"

    result = smart_column_insertion(query)

    # New column should be added with alias
    assert "t.col2" in result
    assert "SELECT DISTINCT" in result

@patch("code_modules.llm_response_extractor.classify_query", return_value="ENTITY")
@patch("code_modules.llm_response_extractor.extract_alias", return_value=None)
@patch("code_modules.llm_response_extractor.extract_select_columns", return_value=["col1"])
@patch("code_modules.llm_response_extractor.supplier_scoring_data_necessary_fields", ["col1", "col2"])
def test_smart_column_insertion_supplier(mock_select, mock_alias, mock_classify):
    query = "SELECT col1 FROM supplier_scoring_data"

    result = smart_column_insertion(query)

    # No alias → plain column insertion
    assert "col2" in result
    assert "SELECT DISTINCT" in result




@patch("code_modules.llm_response_extractor.classify_query", return_value="ENTITY")
@patch("code_modules.llm_response_extractor.extract_alias", return_value=None)  # 👈 IMPORTANT
@patch("code_modules.llm_response_extractor.extract_select_columns", return_value=["col1"])
@patch("code_modules.llm_response_extractor.supplier_scoring_data_necessary_fields", ["col1", "col2"])
def test_missing_columns_without_alias(mock_select, mock_alias, mock_classify):
    query = "SELECT col1 FROM supplier_scoring_data"

    result = smart_column_insertion(query)

    # 👇 This ensures plain column (no alias) was appended
    assert "col2" in result
    assert "t.col2" not in result  # extra safety
    assert "SELECT DISTINCT" in result


import pytest
from unittest.mock import patch, MagicMock

