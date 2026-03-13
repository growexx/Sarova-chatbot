"""
The following module is used to extarct json from LLM response
"""
import re
import json
from typing import Dict, Any
import traceback

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from:
    - pure JSON string
    - text containing JSON
    - ```json fenced blocks
    """
    try:
        text = text.strip()

        # Case 1: fenced ```json
        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        # Case 2: embedded JSON
        embedded = re.search(r"\{.*\}", text, re.DOTALL)
        if embedded:
            return json.loads(embedded.group(0))

        # Case 3: plain JSON
        result = json.loads(text)
    except Exception as e:
        traceback.print_exc()
        result = {"response":text,
                "error":str(e)}
    return result

class LLMResponseExtractor:
    """
    We Declare json data with set_data and get single or multiple keys
    """
    def set_data(self, text:str):
        self.data = extract_json(text)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def get_many(self, fields, defaults=None):
        defaults = defaults or {}
        return tuple(self.get(f, defaults.get(f)) for f in fields)


ap_column_prirority ={
    "BUSINESS_UNIT":1,
    "VENDOR_NAME":2,
    "INVOICE_NUM":3,
    "INVOICE_DATE":4,
    "INVOICE_AMOUNT":5,
    "INVOICE_CURRENCY_CODE":6,
    "WFAPPROVAL_STATUS":7,
    "PAYMENT_STATUS_FLAG":8,
    "INV_CREATOR_NAME":9,
    "APPROVER_FULL_NAME":10
}

gl_column_priority={
"LEDGER_ID":1,
"LEDGER_NAME":2,
"LEDGER_CATEGORY_CODE":3,
"LEDGER_CREATED_BY":4,
"PERIOD_NAME":5,
"JE_SOURCE":6,
"JE_CATEGORY":7,
"BATCH_NAME":8,
"JOURNAL_NAME":9,
"CURRENCY_CODE":10,
"POSTED_DATE":11,
"JOURNAL_DR":12,
"JOURNAL_CR":13,
"APPROVAL_STATUS_CODE":14,
"JOURNAL_STATUS":15
}

user_column_priority = {
"APPLICATION_ID":1,
"RESPONSIBILITY_NAME":2,
"USER_ID":3,
"USER_NAME":4,
"DESCRIPTION":5,
"ROLE_ACCESS":6,
"CREATION_DATE":7,
"LAST_UPDATE_DATE":8
}

SQL_KEYWORDS = {
    "where","join","left","right","inner","outer","group",
    "order","fetch","limit","union","on"
}

gl_necessary_fields = ["ledger_id" ,"ledger_name" , "ledger_category_code"  , "ledger_created_by" ]
user_necessary_fields = ["user_id" , "user_name" ,"description" ,"creation_date" "role_access"]

def smart_reorder(query, columns):
    """
    Takes column names and query and rearranges them to fit the order
    """
    print(f"smart reorder start with input columns {columns}")
    query_lower = query.lower()

    if "ap_audit_data" in query_lower:
        table_column_priority = ap_column_prirority
        print('ap audit data reorder started')
    elif "gl_audit_data" in query_lower:
        print('gl audit data reorder started')
        table_column_priority = gl_column_priority
    elif "user_audit_data" in query_lower:
        print('user audit data reorder started')
        table_column_priority = user_column_priority
    else:
        table_column_priority = {}

    # Sort columns based on priority (unknown columns go last)
    ordered_columns = sorted(
        columns,
        key=lambda col: table_column_priority.get(col, 999)
    )
    print(f"Soted columns are {ordered_columns}")

    return ordered_columns


def smart_column_insertion(query):
    print('_'*50, "smart column insertion started",'_'*50)
    query_type =  classify_query(query)
    query_lower = query.lower()
    if "ap_audit_data" in query_lower or query_type in ("AGGREGATION", "KPI","WINDOW","JOIN"):
        print(f"Early exit triggered either by sql_type {query_type} or ap_audit_table found")
        print('_'*50, "smart column insertion ended early",'_'*50)
        return query
    elif "gl_audit_data" in query_lower:
        print('query references gl audit data')
        insertion_fields = gl_necessary_fields
        alias_name = extract_alias(query,"gl_audit_data")
    elif "user_audit_data" in query_lower:
        print('query references user audit data')
        insertion_fields = user_necessary_fields
        alias_name = extract_alias(query,"user_audit_data")
    else:
        print('No idea what to do')
    present_cols = extract_select_columns(query)
    # present_cols = extract_select_columns(query)
    normalized_present = {normalize_column(c) for c in present_cols}
    print(f"{alias_name} is the alias name and present columns are {present_cols}")

    missing_columns = []
    for field in insertion_fields:
        if field.lower() not in normalized_present:
            if alias_name:
                missing_columns.append(f"{alias_name}.{field}")
            else:
                missing_columns.append(field)

    if not missing_columns:
        return query
    new_columns = present_cols + missing_columns
    new_select_clause = "SELECT DISTINCT " + ", ".join(new_columns)

    # Replace old SELECT clause
    new_query = re.sub(
        r"select\s+(distinct\s+)?(.*?)\s+from",
        new_select_clause + " FROM",
        query,
        flags=re.IGNORECASE | re.DOTALL
    )
    print(f"new query is  {new_query}, while_old query was {query}")   
    print('_'*50, "smart column insertion ended",'_'*50)
    return new_query
    
    
def extract_select_columns(query):
    match = re.search(r"select\s+distinct\s+(.*?)\s+from", query, re.IGNORECASE | re.DOTALL)
    if not match:
        match = re.search(r"select\s+(.*?)\s+from", query, re.IGNORECASE | re.DOTALL)

    if not match:
        return None

    columns = match.group(1)
    return [c.strip() for c in columns.split(",")]


def normalize_column(col):
    col = col.lower().strip()
    if "." in col:
        col = col.split(".")[-1]
    return col

def extract_alias(query, table_name):

    pattern = rf"from\s+{table_name}\s+(\w+)"
    match = re.search(pattern, query, re.IGNORECASE)

    if not match:
        return None

    alias = match.group(1)

    if alias.lower() in SQL_KEYWORDS:
        return None

    return alias

def classify_query(sql: str) -> str:
    sql_lower = sql.lower()

    if " over(" in sql_lower:
        sql_type =  "WINDOW"
    elif "case when" in sql_lower and "count(" in sql_lower:
        sql_type =  "KPI"
    elif any(agg in sql_lower for agg in ["count(", "sum(", "avg(", "min(", "max("]):
        sql_type =  "AGGREGATION"
    elif "join " in sql_lower:
        sql_type =  "JOIN"
    elif "fetch first" in sql_lower:
        sql_type =  "TOP_N"
    elif "order by" in sql_lower:
        sql_type =  "ORDERED_ENTITY"
    else:
        sql_type =  "ENTITY"
    return sql_type

