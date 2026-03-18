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


inventory_consumption_data_prirority ={
    "ITEM_GROUP":1 , "ITEM_NAME":2 ,"TOTAL_ITEM_CONSUMED":3 ,  "QUARTER":4, "YEAR":5
}

occupancy_revenue_data_priority={
    "date_col":1, "occupancy":2, "revenue":3 , "data_type":4 ,"year":5
}

supplier_scoring_data_priority = {
    "item_group":1, "item_name":2, "total_qty_supplied":3, "supplier":4, "category":5
}

SQL_KEYWORDS = {
    "where","join","left","right","inner","outer","group",
    "order","fetch","limit","union","on"
}

inventory_consumption_data_necessary_fields = ["item_group" , "item_name" ,"total_item_consumed",  "quarter", "year"]
occupancy_revenue_data_necessary_fields = ["date_col", "occupancy", "revenue" , "data_type" ,"year"]
supplier_scoring_data_necessary_fields = ["item_group", "item_name", "total_qty_supplied", "supplier", "category"]


def smart_reorder(query, columns):
    """
    Takes column names and query and rearranges them to fit the order
    """
    print(f"smart reorder start with input columns {columns}")
    query_lower = query.lower()

    if "inventory_consumption_data" in query_lower:
        table_column_priority = inventory_consumption_data_prirority
        print('invenoryt consumption data reorder started')
    elif "occupancy_revenue_data" in query_lower:
        print('daily occupancy revenue data reorder started')
        table_column_priority = occupancy_revenue_data_priority
    elif "supplier_scoring_data" in query_lower:
        print('supplier_scoring data reorder started')
        table_column_priority = supplier_scoring_data_priority
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
    """
    Smart column inserrion adds columsn to the df
    """
    print('_'*50, "smart column insertion started",'_'*50)
    query_type =  classify_query(query)
    query_lower = query.lower()
    if query_type in ("AGGREGATION", "KPI","WINDOW","JOIN"):
        print(f"Early exit triggered either by sql_type {query_type} found")
        print('_'*50, "smart column insertion ended early",'_'*50)
        return query
    elif "inventory_consumption_data" in query_lower:
        print('query references inventory_consumption_data')
        insertion_fields = inventory_consumption_data_necessary_fields
        alias_name = extract_alias(query,"inventory_consumption_data")
    elif "occupancy_revenue_data" in query_lower:
        print('query references user occupancy_revenue_data')
        insertion_fields = occupancy_revenue_data_necessary_fields
        alias_name = extract_alias(query,"occupancy_revenue_data")
    elif "supplier_scoring_data" in query_lower:
        print('query references supplier_scoring_data')
        insertion_fields = supplier_scoring_data_necessary_fields
        alias_name = extract_alias(query,"supplier_scoring_data")
    else:
        print('No idea what to do')
        return query
    present_cols = extract_select_columns(query)
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
    elif any(agg in sql_lower for agg in ["count(", "sum(", "avg(", "min(", "max(", "distinct("]):
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

