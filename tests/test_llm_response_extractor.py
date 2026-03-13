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
    ap_column_prirority,
    gl_column_priority,
    user_column_priority
)


# ---------------------------------------------------------
# 1️⃣ AP AUDIT DATA REORDER
# ---------------------------------------------------------

def test_smart_reorder_ap_table():
    query = "select * from ap_audit_data"
    columns = [
        "INVOICE_AMOUNT",
        "BUSINESS_UNIT",
        "VENDOR_NAME"
    ]

    result = smart_reorder(query, columns)

    assert result == [
        "BUSINESS_UNIT",
        "VENDOR_NAME",
        "INVOICE_AMOUNT"
    ]


# ---------------------------------------------------------
# 2️⃣ GL AUDIT DATA REORDER
# ---------------------------------------------------------

def test_smart_reorder_gl_table():
    query = "select * from gl_audit_data"
    columns = [
        "JOURNAL_CR",
        "LEDGER_NAME",
        "PERIOD_NAME"
    ]

    result = smart_reorder(query, columns)

    assert result == [
        "LEDGER_NAME",
        "PERIOD_NAME",
        "JOURNAL_CR"
    ]


# ---------------------------------------------------------
# 3️⃣ USER AUDIT DATA REORDER
# ---------------------------------------------------------

def test_smart_reorder_user_table():
    query = "select * from user_audit_data"
    columns = [
        "USER_NAME",
        "APPLICATION_ID",
        "CREATION_DATE"
    ]

    result = smart_reorder(query, columns)

    assert result == [
        "APPLICATION_ID",
        "USER_NAME",
        "CREATION_DATE"
    ]


# ---------------------------------------------------------
# 4️⃣ UNKNOWN TABLE → NO REORDER
# ---------------------------------------------------------

def test_smart_reorder_unknown_table():
    query = "select * from some_other_table"
    columns = ["B", "A", "C"]

    result = smart_reorder(query, columns)

    # No priority → original order preserved
    assert result == ["B", "A", "C"]


# ---------------------------------------------------------
# 5️⃣ MIXED KNOWN + UNKNOWN COLUMNS
# ---------------------------------------------------------

def test_smart_reorder_mixed_columns():
    query = "select * from ap_audit_data"
    columns = [
        "UNKNOWN_COL",
        "VENDOR_NAME",
        "BUSINESS_UNIT"
    ]

    result = smart_reorder(query, columns)

    # Known columns first by priority, unknown last
    assert result == [
        "BUSINESS_UNIT",
        "VENDOR_NAME",
        "UNKNOWN_COL"
    ]


# ---------------------------------------------------------
# 6️⃣ CASE INSENSITIVE TABLE NAME
# ---------------------------------------------------------

def test_smart_reorder_case_insensitive():
    query = "SELECT * FROM AP_AUDIT_DATA"
    columns = [
        "VENDOR_NAME",
        "BUSINESS_UNIT"
    ]

    result = smart_reorder(query, columns)

    assert result == [
        "BUSINESS_UNIT",
        "VENDOR_NAME"
    ]


# ---------------------------------------------------------
# 7️⃣ EMPTY COLUMN LIST
# ---------------------------------------------------------

def test_smart_reorder_empty_columns():
    query = "select * from ap_audit_data"
    columns = []

    result = smart_reorder(query, columns)

    assert result == []


# ---------------------------------------------------------
# 8️⃣ ALL UNKNOWN COLUMNS
# ---------------------------------------------------------

def test_smart_reorder_all_unknown_columns():
    query = "select * from ap_audit_data"
    columns = ["X", "Y", "Z"]

    result = smart_reorder(query, columns)

    # All priority = 999 → order preserved
    assert result == ["X", "Y", "Z"]