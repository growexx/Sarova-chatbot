"""
Chat service module for the Sarova Web API app.

This module orchestrates the end-to-end workflow for handling user chat inquiries:
- Applies guardrails to validate relevance
- Converts natural language to SQL using LLMs
- Executes SQL queries against Oracle Autonomous Database
- Processes results for analysis or raw data download
- Generates plots and uploads artifacts to OCI Object Storage

It integrates database access, LLM inference, prompt generation,
response parsing, and visualization utilities.
"""
import os
import uuid
from code_modules.oracle_adb_handler import OracleADBClient
from config_loader import load_adw_config
from code_modules.prompt_generator import PromptGenerator
from code_modules.llm_response_extractor import LLMResponseExtractor, extract_reply_message_from_raw 
# , smart_reorder , smart_column_insertion
from code_modules.sql_queries_loader import SqlQueryLoader
from code_modules.oracle_genai_handler import create_llm_client, create_guardrail_llm_client
from code_modules.oci_object_storage import OCIObjectStorageClient
from code_modules.sql_query_modifier import add_distinct_safely ,ensure_fetch_first_clause , wrap_query_with_count
from typing import Optional, Tuple
import json
import pandas as pd
from code_modules.oracle_adb_handler import OracleADBClient
from datetime import datetime
import traceback
import re
import numpy as np
import time
from app.services.title import create_new_chat_title
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns


_PARALLEL_EXECUTOR = ThreadPoolExecutor(max_workers=2)
_ALL_TABLES = ["inventory_consumption_data", "occupancy_revenue_data", "supplier_scoring_data"]
RAW_MESSAGE = "It is found that Oracle Genai has marked this in appropriate content and rejected it."


def _load_all_metadata() -> str:
    """Load and cache all table metadata as a single string at import time."""
    metadata_string = ""
    for table in _ALL_TABLES:
        file_name = f"table_metadata/{table.lower()}.json"
        try:
            with open(file_name, "r") as f:
                metadata = json.load(f)
            metadata_string += f"\n\n### TABLE: {table.upper()}\n"
            metadata_string += json.dumps(metadata, indent=2)
        except FileNotFoundError:
            pass
    return re.sub(r'\s+', ' ', metadata_string).strip()

_ALL_METADATA_STRING: str = _load_all_metadata()


def generate_categorical_plots( df: pd.DataFrame, output_dir: str,file_prefix: str, max_categories: int = 15 ) -> list[str]:
    """
    Generate categorical visualizations from a dataframe.
 
    Depending on the number of categorical columns, this function:
    - Creates a bar chart for a single categorical column
    - Creates a grouped bar chart or heatmap for two categorical columns
 
    The generated plots are saved to disk and their file paths are returned.
 
    Args:
        df (pd.DataFrame): Input dataframe containing categorical and numeric data.
        output_dir (str): Directory where plots will be saved.
        file_prefix (str): Prefix used for plot file names.
        max_categories (int, optional): Maximum number of categories to display.
 
    Returns:
        list[str]: List of generated plot file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []
 
    # Detect columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
 
    if not categorical_cols or not numeric_cols:
        return plot_paths
 
    num_col = numeric_cols[0]
 
    # -----------------------
    # CASE 1: One categorical
    # -----------------------
    if len(categorical_cols) == 1:
        cat = categorical_cols[0]
 
        # Reduce categories
        if df[cat].nunique() > max_categories:
            df = df.nlargest(max_categories, num_col)
 
        df_sorted = df.sort_values(num_col, ascending=False)
 
        plt.figure(figsize=(8, 4))
        sns.barplot(data=df_sorted, x=cat, y=num_col)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{num_col} by {cat}")
 
        path = os.path.join(output_dir, f"{file_prefix}_{cat}_bar.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
 
        plot_paths.append(path)
 
    # -----------------------
    # CASE 2: Two categoricals
    # -----------------------
    elif len(categorical_cols) >= 2:
        cat1, cat2 = categorical_cols[:2]
 
        # Reduce cardinality
        if df[cat1].nunique() > max_categories:
            top_cat1 = (
                df.groupby(cat1)[num_col].sum()
                .nlargest(max_categories)
                .index
            )
            df = df[df[cat1].isin(top_cat1)]
 
        # Few values → Grouped Bar
        if df[cat2].nunique() <= 5:
            plt.figure(figsize=(9, 4))
            sns.barplot(data=df, x=cat1, y=num_col, hue=cat2)
            plt.xticks(rotation=45, ha="right")
            plt.title(f"{num_col} by {cat1} and {cat2}")
 
            path = os.path.join(
                output_dir, f"{file_prefix}_{cat1}_{cat2}_grouped_bar.png"
            )
 
        # Many values → Heatmap
        else:
            pivot = df.pivot_table(
                index=cat2,
                columns=cat1,
                values=num_col,
                aggfunc="sum"
            )
 
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, cmap="Blues")
            plt.title(f"{num_col} Heatmap ({cat1} vs {cat2})")
 
            path = os.path.join(
                output_dir, f"{file_prefix}_{cat1}_{cat2}_heatmap.png"
            )
 
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(path)
 
    return plot_paths
 

def prepare_metadata_string(tables):
    """
    Prepare metadata string for prompt construction."""
    metadata_string = ""

    for table in tables:
        file_name = f"table_metadata/{table.lower()}.json"
        with open(file_name, "r") as f:
            metadata = json.load(f)
        metadata_string += f"\n\n### TABLE: {table.upper()}\n"
        metadata_string += json.dumps(metadata, indent=2)
    return metadata_string

def check_if_df_all_null_or_zero(df: pd.DataFrame) -> bool:
    """
    Check whether all values in the dataframe are null or zero.

    Args:
        df (pd.DataFrame): Dataframe to evaluate.

    Returns:
        bool: True if all values are null or zero, otherwise False.
    """
    return bool(((df.isna()) | (df == 0)).all().all())

class JumpToFinally(Exception):
    """Custom exception just to jump to finally"""
    pass

class ChatService:
    """
    Service layer responsible for handling chat-based BI queries.

    This class manages:
    - Guardrail validation
    - Natural language to SQL conversion
    - Database query execution
    - Context-aware LLM inference
    - Response formatting for analysis or raw data scenarios
    """

    def __init__(self):
        """
        Initialize ChatService with LLM clients, prompt generator,
        and response extractor.
        """
        self.llm_inference_client = create_llm_client()
        self.guardrail_llm_client = create_guardrail_llm_client()
        self.llm_response_extractor = LLMResponseExtractor()
        self.prompt_generator_client = PromptGenerator()
        self.adb_client = OracleADBClient(load_adw_config())
        self.sql_loader = SqlQueryLoader()
        self.object_storage_client = OCIObjectStorageClient()

    @staticmethod
    def log_time(label, start_time):
        elapsed = time.perf_counter() - start_time
        print(f"[TIMER] {label} took {elapsed:.4f} seconds")

    def _run_guardrail(self, user_message: str) -> dict:
        """
        Worker: run guardrail check and return parsed result dict.
        Runs in a background thread — do NOT mutate shared state here.
        """
        user_guard_rail_message = f"User message is '{user_message}'"
        raw = self.prompt_generator_client.guardrail_check_inference_call(
            self.guardrail_llm_client, user_guard_rail_message
        )
        if raw == RAW_MESSAGE:
            return raw
        print(f"Guard rail raw response is{raw}")
        # Use a fresh extractor per thread to avoid shared state mutation
        extractor = LLMResponseExtractor()
        extractor.set_data(raw)
        relevant, tables, reply_message = extractor.get_many(
            ["relevant_question", "tables_related", "reply_message"],
            {"relevant_question": "no", "tables_related": [], "reply_message": ""}
        )
        reply_message = reply_message or ""
        if isinstance(reply_message, str):
            reply_message = reply_message.strip()
        else:
            reply_message = str(reply_message).strip() if reply_message else ""
        if not reply_message and extractor.data.get("response"):
            reply_message = extract_reply_message_from_raw(extractor.data["response"])
        return {"relevant": relevant, "tables": tables, "reply_message": reply_message}

    def _run_text_2_sql(self, user_message: str, last_sql_query: str) -> dict:
        """
        Worker: generate SQL using ALL table metadata (no guardrail dependency).
        Runs in a background thread — do NOT mutate shared state here.

        We intentionally use _ALL_METADATA_STRING (all tables) so this worker
        can start immediately without waiting for guardrail's table list.
        The SQL generated here may reference more tables than strictly needed,
        but the LLM is instructed to only use what's relevant — in practice
        output quality is identical to filtered-metadata calls.
        """
        sql_prompt = self.prompt_generator_client.generate_sql_prompt(
            user_message, _ALL_METADATA_STRING, last_sql_query
        )
        raw = self.llm_inference_client.inference_single_input(user_message, sql_prompt)

        extractor = LLMResponseExtractor()
        extractor.set_data(raw)
        sql_query,scenario, error_status = extractor.get_many(
            ["sql_query","scenario", "error_flag"],
            {"sql_query": "","scenario": "raw_text", "error_flag": 0}
        )
        return {"sql_query": sql_query, "scenario": scenario ,"error_status": error_status}

    @staticmethod
    def _prepare_local_file_and_par_url(file_prefix:str = "result_data", bucket_folder_name: str = "Sarova_Table_files")  -> Tuple[str, str]:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        local_filename = f"{file_prefix}_{timestamp}_{unique_id}.xlsx"
        object_name = bucket_folder_name +'/' + os.path.basename(local_filename)
        par_url=f"https://objectstorage.me-dubai-1.oraclecloud.com/p/oawta1HMX-BgQZkdRtaJUVt6E8lTOa5vEzC3ZqeIuc7i649VOG2VHlBRxinPm8Ny/n/bmb8tbvmgtsy/b/Sarova_extended/o/{object_name}"
        return local_filename, par_url

    async def _background_message_insert(self,rows):
        bulk_insert = self.sql_loader.insert_chat_history_bulk(rows)
        start = time.perf_counter()
        with self.adb_client.get_connection() as conn:
            self.adb_client.execute_multiple_non_query(
                conn, bulk_insert["query"], bulk_insert["params"]
            )
        self.log_time("DB insert chat messages", start)

    async def _background_large_file_write(self,sql_query :str ,local_file_path:str):
        sql_query = ensure_fetch_first_clause(query = sql_query,limit=10000)
        try:
            print("Large file write started.")
            folder = f""
            with self.adb_client.get_connection() as conn:
                chunk_gen = self.adb_client.stream_query_chunks(conn, sql_query)

                self.oci_client.stream_chunks_to_excel_and_upload(
                    chunk_generator=chunk_gen,
                    local_file_path=local_file_path,
                    bucket_folder_name = folder
                )
                print("Large file write completed.")
        except Exception:
            traceback.print_exc()
            print("[prepare_data_response] Failed to upload large result to OCI")
            # Graceful fallback: return the first 100 rows inline
            return {
                "results_df": None,
                "llm_response": "Failed to load data",
                "sql_query": None,
                "status": 0,
                "par": None,
            }

    async def _background_small_file_write(self,sql_query :str ,local_file_path:str,bucket_folder_name: str ):
        # sql_query = ensure_fetch_first_clause(query = sql_query,limit=10000)
        try:
            oci_client = OCIObjectStorageClient()
            

            oci_client.put_file_in_bucket_folder(local_file_path, bucket_folder_name)
            os.remove(local_file_path)

            print("Small file write succeeded")
        except Exception:
            traceback.print_exc()
            print("[prepare_data_response] Failed to upload large result to OCI")
            # Graceful fallback: return the first 100 rows inline
            return {
                "results_df": None,
                "llm_response": "Failed to load data",
                "sql_query": None,
                "status": 0,
                "par": None,
            }

    

    async def handle_inquiry(self, user_id: str, chat_id: str, user_message: str, app_state):
        """
        Handle a user chat inquiry end-to-end.

        Flow
        ────
        call guardrail and text-2-sql in parallel with ALL table metadata (no dependency between them)
        then wait for both to complete and unpack results
        if guardrail says irrelevant → reject immediately (don't care about SQL gen result)
        if guardrail says relevant → continue with SQL gen result then execute SQL, generate response, etc.
        """
        final_response = {
            "chat_id": chat_id,
            "llm_response": "Failed due to error",
            "user_query": "",
            "status": 0
        }

        try:
            # ── Prepare last SQL (needed by SQL gen worker) ────────────────
            last_sql_query_of_chat = app_state.last_sql_queries
            last_sql_query = last_sql_query_of_chat.get(chat_id, None)
            print(last_sql_query)
            last_sql_query = self.prepare_last_sql_query(last_sql_query, chat_id)

            # ── Fire guardrail + SQL gen in parallel ───────────────────────
            print(50 * '═', " PARALLEL: Guardrail + SQL Gen ", 50 * '═')
            parallel_start = time.perf_counter()

            future_guardrail = _PARALLEL_EXECUTOR.submit(
                self._run_guardrail, user_message
            )

            future_sql = _PARALLEL_EXECUTOR.submit(
                self._run_text_2_sql, user_message, last_sql_query
            )

            guardrail_result = future_guardrail.result()
            if guardrail_result == "LLM timeout error":
                final_response = {
                    "chat_id": chat_id,
                    "llm_response": "Oracle Gen ai service is on time out error , Please try again ",
                    "user_query": user_message,
                    "status": 2
                }
                raise JumpToFinally()

            if guardrail_result == RAW_MESSAGE:
                final_response = {
                    "chat_id": chat_id,
                    "llm_response": f"{guardrail_result}, Either please change request , or if it seems appropriate.",
                    "user_query": user_message,
                    "status": 2
                }
                raise JumpToFinally()

            relevant = guardrail_result["relevant"]
            tables = guardrail_result["tables"]
            print(f"[Guardrail] relevant={relevant}, tables={tables}")

            if relevant != "yes":
                print("Query rejected by guardrail — discarding SQL gen result")
                reply_message = guardrail_result.get("reply_message", "").strip()
                final_response = {
                    "chat_id": chat_id,
                    "llm_response": reply_message if reply_message else "Query rejected by guardrail due to irrelevance.",
                    "user_query": user_message,
                    "status": 2
                }
                raise JumpToFinally()

            sql_result = future_sql.result()
            print(f"SQL RAW RESULT IS {sql_result}")
            if sql_result ==  RAW_MESSAGE:
                final_response = {
                    "chat_id": chat_id,
                    "llm_response": f"{sql_result}, Either please change request , or if it seems appropriate , please try again.",
                    "user_query": user_message,
                    "status": 2
                }
                raise JumpToFinally()


            self.log_time("PARALLEL guardrail + SQL gen", parallel_start)
            # ── Unpack SQL gen ─────────────────────────────────────────────
            sql_query = sql_result["sql_query"]
            error_status = sql_result["error_status"]
            scenario = sql_result["scenario"]
            print(f"[SQL Gen] error_status={error_status}")

            if error_status == 1 or error_status == "1":
                final_response = {
                    "chat_id": chat_id,
                    "llm_response": (
                        "Failed to generate query as it extends beyond the scope "
                        "of Data. Please try with another query."
                    ),
                    "user_query": user_message,
                    "status": 2
                }
                raise JumpToFinally()

            print(f"[SQL Gen] Generated SQL: {sql_query} and scenario is {scenario}")

            # ── Execute SQL ────────────────────────────────────────────────
            print(50 * '═', " SQL Execution ", 50 * '═')
            start = time.perf_counter()
            # sql_query = smart_column_insertion(add_distinct_safely(sql_query))
            sql_query = add_distinct_safely(sql_query)
            max_limit_query =  ensure_fetch_first_clause(sql_query,10000)
            limited_query = ensure_fetch_first_clause(sql_query,100)
            count_query = wrap_query_with_count(max_limit_query)

            with self.adb_client.get_connection() as conn:
                selected_df = self.adb_client.execute_query_df(conn, limited_query)
                num_of_records = self.adb_client.execute_scalar(conn, count_query)

            print(f"{num_of_records} records found.")
            self.log_time("SQL Execution", start)

            df_is_empty = check_if_df_all_null_or_zero(selected_df)

            if df_is_empty:
                final_response = {
                    "chat_id": chat_id,
                    "llm_response": "No data found for following search",
                    "sql_query": max_limit_query,
                    "status": 3
                }
                raise JumpToFinally()

            selected_df.drop_duplicates(inplace=True)
            print(f"DataFrame shape: {selected_df.shape}")

            print(50 * '═', " Context Management ", 50 * '═')

            app_state.last_sql_queries[chat_id] = sql_query
            chat_histories = app_state.chat_history
            users_chats = app_state.last_user_chat
            current_user_chat_id = users_chats.get(user_id, None)

            user_df_for_chat_ids_query = self.sql_loader.load_user_chats_previews(user_id)['load_chats_preview']
            last_message_no_query = self.sql_loader.get_last_message_no(chat_id)['last_message_no']

            start = time.perf_counter()

            with self.adb_client.get_connection() as conn:
                user_df_for_chat_ids = self.adb_client.execute_query_df(
                    conn, user_df_for_chat_ids_query
                )["CHAT_ID"].tolist()
                message_no = self.adb_client.execute_scalar(conn, last_message_no_query)

            if chat_id not in user_df_for_chat_ids:
                print(f"New chat — creating title for chat_id={chat_id}")
                chat_histories[chat_id] = [{
                    "role": "System",
                    "message": self.prompt_generator_client.generate_main_prompt()
                }]
                with self.adb_client.get_connection() as conn:
                    create_new_chat_title(conn, self, user_id, chat_id, user_message)
                users_chats[user_id] = chat_id
            elif chat_id != current_user_chat_id:
                print("Chat history not in RAM — loading from DB")
                self.load_chat_history_ram(user_id, chat_id, app_state)
            print("chant context management has ended.")
            chat_history = chat_histories[chat_id]

            if len(chat_history) > 5:
                chat_history = chat_history[:1] + chat_history[3:]

            print(50 * '═', " Main LLM Chat Call ", 50 * '═')
            assistant_prompt = self.prompt_generator_client.generate_assistant_prompt(
                sql_query, selected_df , num_of_records
            )
            chat_history.append({"role": "User", "message": user_message})
            chat_history.append({"role": "User", "message": assistant_prompt})

            start = time.perf_counter()
            result = self.llm_inference_client.inference_from_chat_history(chat_history)
            self.log_time("LLM Chat Inference", start)
            extractor = LLMResponseExtractor()
            extractor.set_data(result)
            message = extractor.get("message", "")

            chat_history.pop()
            chat_history.append({"role": "Assistant", "message": message})

            rows = [(chat_id, message_no + 1, user_message, "user"),
                    (chat_id, message_no + 2, message, "assistant"),
                    (chat_id, message_no + 3, sql_query, "SQL")]

            if scenario == 'analysis':
                plot_path = generate_categorical_plots(selected_df, 'temp_graph','graph')[0]
                final_response = self.prepare_data_response(
                    selected_df, sql_query, message ,scenario,None
                )
            else:

                local_file_path, actual_par = self._prepare_local_file_and_par_url()
                selected_df.to_excel(local_file_path, index=False)
                print("Writing in par file")
                # asyncio.create_task(self._background_file_write(max_limit_query,local_file_path))
                if num_of_records<100:
                    asyncio.create_task(self._background_small_file_write(max_limit_query,local_file_path,"Sarova_Table_files"))
                else:
                    print("File too big to write")
                final_response = self.prepare_data_response(
                    selected_df.head(20), sql_query, message ,scenario, actual_par
                )
                rows.append((chat_id, message_no + 4, actual_par, "PAR"))
            asyncio.create_task(self._background_message_insert(rows))
        except JumpToFinally:
            pass
        except Exception as e:
            traceback.print_exc()
            print(f"Unexpected error: {e}")
        finally:
            print("handle_inquiry complete.")

        return final_response

    def prepare_last_sql_query(self,last_sql_query, chat_id):
        if last_sql_query is None:
            get_last_sql_query = self.sql_loader.last_sql_query_for_chat(chat_id)['last_sql_query_of_chat']
            with self.adb_client.get_connection() as conn:
                last_sql_query = self.adb_client.execute_scalar(conn,get_last_sql_query)
            print(f"Last sql query for chat_id {chat_id} is {last_sql_query}.")
        return last_sql_query

    @staticmethod
    def prepare_message_no(message_no_df):
        if message_no_df.empty:
            message_no = 1
        else:
            message_no = int(message_no_df.iloc[0]["MESSAGE_NO"])
            message_no= message_no+1
        return message_no

    @staticmethod
    def prepare_data_response(selected_df: pd.DataFrame, sql_query: str, message: str,scenario: str,par: Optional[str] = None):
        """
        Prepare a response payload based on the size of the query result.

        Row-count routing
        -----------------
        • **≤ 100 rows** — Return all rows inline as a list of dicts.
          ``par`` is set to ``None``.
        • **> 100 rows** — Do NOT return data inline.  Instead stream the
          full result set from the database in chunks, write it to a
          temporary CSV file on disk, upload it to OCI Object Storage, and
          return a Pre-Authenticated Request (PAR) download URL.
          ``results_df`` is set to ``None``.

        Memory safety for large datasets
        ---------------------------------
        Rather than calling ``execute_query_df`` (which loads every row
        into a single DataFrame), we call ``stream_query_chunks`` which
        fetches the cursor in small batches and writes each batch directly
        to a local CSV file.  Only one batch lives in memory at any time.

        Args:
            selected_df (pd.DataFrame): DataFrame already fetched earlier in
                the pipeline (used for the ≤ 100 row path and for counting).
            sql_query (str): The SQL query that produced ``selected_df``.
            message (str): LLM-generated explanation accompanying the result.

        Returns:
            dict: Response payload with keys:
                ``results_df``, ``llm_response``, ``sql_query``, ``status``,
                ``par``.
        """
        # ── Small result: return inline ────────────────────────────────────
        if selected_df is not None:
            # column_order = smart_reorder(sql_query, selected_df.columns)
            # selected_df = selected_df.replace([np.nan, np.inf, -np.inf], None)[column_order].to_dict(orient="records")
            selected_df = selected_df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")

        return {
            "results_df": selected_df,
            "scenario":scenario,
            "llm_response": message,
            "sql_query": sql_query,
            "status": 1,
            "par": par
        }

    # def guard_rail(self, user_message,):
    #     """
    #     Apply guardrail validation to a user query.

    #     Uses an LLM-based guardrail to determine whether the query
    #     is relevant and which database tables are required.

    #     Args:
    #         user_message (str): User's natural language query.

    #     Returns:
    #         tuple[str, list[str]]: Relevance flag and list of related tables.
    #     """

    #     user_guard_rail_message= f"User message is '{user_message}'"
    #     guard_rail_result = self.prompt_generator_client.guardrail_check_inference_call(self.guardrail_llm_client, user_guard_rail_message)

    #     self.llm_response_extractor.set_data(guard_rail_result)
    #     print(f"Guard rail result is {guard_rail_result}")

    #     relevant, tables = self.llm_response_extractor.get_many(["relevant_question", "tables_related"],{"relevant_question":"no","tables_related":[]})

    #     return relevant , tables

    def text_2_sql(self, user_message, tables, last_sql_query, metadata):
        """
        Convert a natural language query into SQL.

        Uses LLM inference with schema metadata and conversation context
        to generate a SQL query and determine execution scenario.

        Args:
            user_message (str): User's natural language query.
            tables (list[str]): Relevant database tables.
            last_sql_query (str | None): Previously executed SQL for context.
            metadata (str): Table metadata for prompt construction.

        Returns:
            tuple[str, str, int]: SQL query, scenario, and error status.
        """

        metadata_string = prepare_metadata_string(tables)
        print(f"last sql query is {last_sql_query} and its type is {type(last_sql_query)}")
        sql_prompt = self.prompt_generator_client.generate_sql_prompt(user_message, metadata_string, last_sql_query)
        llm_sql_raw = self.llm_inference_client.inference_single_input(user_message, sql_prompt)
        self.llm_response_extractor.set_data(llm_sql_raw)
        sql_query, error_status = self.llm_response_extractor.get_many(
            ["sql_query", "error_status"],
            {"sql_query":"","error_status":0}
        )
        return sql_query, error_status

    def reconnnect_conn_and_get_df(self, given_query):
        with self.adb_client.get_connection() as conn:
            return self.adb_client.execute_query_df(conn, given_query)

    def load_chat_history_ram(self,user_id,chat_id,app_state):
        """
        This message is used to load chat history for a specific user and chat id only in ram
        Assumed that Message not called frequently, So we fetch it from db and not store in state.
        """
        try:

            get_chat_history_query = self.sql_loader.load_chat_history_by_id(chat_id)['load_chat_history']
            with self.adb_client.get_connection() as conn:
                chat_history_df = self.adb_client.execute_query_df(conn,get_chat_history_query)

            # No Chat history Found exit
            if chat_history_df.empty:
                return "empty"

            print("Now we manage and Chat history and store it in state")

            # Load it in load in chat_history

            chat_history_state = [{"role":"System","message":self.prompt_generator_client.generate_main_prompt()}]

            chat_history_df = chat_history_df.sort_values(by="MESSAGE_NO",ascending=False)
            print(chat_history_df)

            i = 0
            n = 6
            print(f"chat history length is {len(chat_history_df)}")
            while i < len(chat_history_df) and n > 0:
                if n==0:
                    print("exit condition triggered")
                    print(f"{i}, {n}")
                    break
                row = chat_history_df.iloc[i]
                if row["ROLE"].upper() == "PAR" or row["ROLE"].upper() == "SQL":
                    print("Message not be selected as it is par or sql")
                else:
                    chat_history_state.insert(1,{
                        "role": row["ROLE"],
                        "message": row["MESSAGE"]
                    })
                    n=n-1
                i+=1

            sql_df = chat_history_df[chat_history_df["ROLE"].str.upper() == "SQL"].tail(1)
            sql_record = sql_df["MESSAGE"].iloc[0] if not sql_df.empty else None

            last_chat_id_of_user = app_state.last_user_chat.get(user_id, None)
            if last_chat_id_of_user is not None:
                del app_state.last_user_chat[user_id]
                del app_state.chat_history[last_chat_id_of_user]
                del app_state.last_sql_queries[last_chat_id_of_user]

            app_state.last_user_chat[user_id] = chat_id
            app_state.chat_history[chat_id] = chat_history_state
            app_state.last_sql_queries[chat_id] = sql_record or None

            return "success"
        except Exception as e:
            traceback.print_exc()
            return f"failure by {e}"

    def load_chat_history(self,user_id,chat_id,app_state):
        """
        This message is used to load chat history for a specific user and chat id
        Assumed that Message not called frequently, So we fetch it from db and not store in state.
        """
        try:

            get_chat_history_query = self.sql_loader.load_chat_history_by_id(chat_id)['load_chat_history']
            with self.adb_client.get_connection() as conn:
                chat_history_df = self.adb_client.execute_query_df(conn,get_chat_history_query)

            # No Chat history Found exit
            if chat_history_df.empty:
                return {
                    "chat_id":chat_id,
                    "systen_message":"No chat history found for chat_id {chat_id}",
                    "status":0
                }

            print("Now we manage and Chat history and store it in state")

            # Load it in load in chat_history
            main_prompt = self.prompt_generator_client.generate_main_prompt()
            chat_history_state = [{"role":"Assistant","message":main_prompt}]
            chat_history_front_end = []

            chat_history_df = chat_history_df.sort_values(by="MESSAGE_NO",ascending=False)
            print(chat_history_df)

            i = 0
            n = 6
            print(f"chat history length is {len(chat_history_df)}")
            while i < len(chat_history_df) and n > 0:
                if n==0:
                    print("exit condition triggerd")
                    print(f"{i}, {n}")
                    break
                row = chat_history_df.iloc[i]
                if row["ROLE"].upper() == "PAR":
                    print(f"par message detected , lets see i and increment it {i}")
                    chat_history_front_end.insert(0,{
                        "role": row["ROLE"].lower(),
                        "message": row["MESSAGE"]
                    })
                    i=i+1
                    print(i)
                elif row["ROLE"].upper() == "SQL":
                    try:
                        print(f"message i is {i}")
                        message_content =self.reconnnect_conn_and_get_df(ensure_fetch_first_clause(row["MESSAGE"],100)).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
                        chat_history_front_end.insert(0,{
                            "role": row["ROLE"],
                            "message": message_content
                        })
                    except Exception as e:
                        print(f"Sql message not generated correctly {e}.")
                        chat_history_front_end.insert(0,{
                            "role": 'SQL',
                            "message": "Failed to deliver message due to SQL error"
                        })
                else:
                    chat_history_front_end.insert(0,{
                        "role": row["ROLE"],
                        "message": row["MESSAGE"]
                    })
                    chat_history_state.insert(1,{
                        "role": row["ROLE"],
                        "message": row["MESSAGE"]
                    })
                    n=n-1
                i+=1

            sql_record = chat_history_df[chat_history_df["ROLE"].str.upper() == "SQL"].tail(1)["MESSAGE"].values[0]

            last_chat_id_of_user = app_state.last_user_chat.get(user_id, None)
            if last_chat_id_of_user is not None:
                del app_state.last_user_chat[user_id]
                del app_state.chat_history[last_chat_id_of_user]
                del app_state.last_sql_queries[last_chat_id_of_user]

            app_state.last_user_chat[user_id] = chat_id
            app_state.chat_history[chat_id] = chat_history_state
            app_state.last_sql_queries[chat_id] = sql_record or None
            return {
                    "chat_id":chat_id,
                    "system_response":chat_history_front_end,
                    "status":1
                }
        except Exception as e:
            traceback.print_exc()
            return {
                "chat_id":chat_id,
                "error_message":e,
                "status":0
            }

    def load_user_chats_previews(self,user_id,app_state):
        """
        This message is used to load chat history for a specific user
        """
        try:
            print("="*100)
            print("Loading user chat ids ")
            get_chat_history_query = self.sql_loader.load_user_chats_previews(user_id)['load_chats_preview']
            with self.adb_client.get_connection() as conn:
                chat_history_df =self.adb_client.execute_query_df(conn,get_chat_history_query)
            print("="*100)
            return {
                    "previous_chat_previous":chat_history_df.to_dict(orient="records"),
                    "status":1
                }
        except Exception as e:
            print(e)
            return {
                "error_message":e,
                "status":0
            }

    @staticmethod
    def chat_runtime_cleanup(user_id,app_state):
        try:
            print("Chat runtime cleanup")
            chat_histories = app_state.chat_history
            last_sql_queries = app_state.last_sql_queries
            user_chats = app_state.user_chats

            user_chat_ids = user_chats.get(user_id)
            if user_chat_ids is None:
                return {
                    "status":0,
                    "message":f"No chat history found for user {user_id} in state"
                }

            n = len(user_chat_ids)
            print(f"user chat ids are {user_chat_ids}")

            for chat_id in user_chat_ids:
                del chat_histories[chat_id]
                del last_sql_queries[chat_id]

            return {
                "status":1,
                "message":f"Deleted {n} chat history for user {user_id}"
            }
        except Exception as e:
            print(e)
            return {
                "error_message":e,
                "status":0
            }

    def delete_chat_history(self,user_id,chat_ids,app_state):
        try:
            for chat_id in chat_ids:
                delete_queries = self.sql_loader.delete_chat_queries(chat_id)
                with self.adb_client.get_connection() as conn:
                    self.adb_client.execute_single_non_query(conn,delete_queries['delete_chat_history'])
                    self.adb_client.execute_single_non_query(conn,delete_queries['delete_chat_preview'])

                app_state.chat_history.pop(chat_id, None)
                app_state.last_sql_queries.pop(chat_id, None)

            return {
                "status":1,
                "message":f"Deleted {len(chat_ids)} chat history for user {user_id}"
            }
        except Exception as e:
            print(e)
            return {
                "error_message":e,
                "status":0
            }

    def delete_all_chats_for_user(self,user_id,app_state):
        try:

            delete_previws = self.sql_loader.delete_all_chats_for_user(user_id)['delete_all_chats_for_user']
            with self.adb_client.get_connection() as conn:
                self.adb_client.execute_single_non_query(conn,delete_previws)

            tuples_previews = [(s,) for s in delete_previws]

            delete_chats_query = self.sql_loader.delete_chat_history_bulk(tuples_previews)
            with self.adb_client.get_connection() as conn:
                self.adb_client.execute_multiple_non_query(conn,delete_chats_query)

            return {
                "status":1,
                "message":f"Deleted all chat history for user {user_id}"
            }
        except Exception as e:
            print(e)
            return {
                "error_message":e,
                "status":0
            }
