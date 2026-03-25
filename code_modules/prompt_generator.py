"""
Prompt generation utilities for the Real Estate Sales chatbot.

This module is responsible for constructing all LLM prompts used by the system,
including:
- Initial system prompts
- Assistant response prompts
- SQL generation prompts
- Guardrail and relevance-check prompts

Prompts are loaded from external template files and dynamically populated
using runtime inputs such as user queries, SQL statements, and data samples.
"""

import re
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
from datetime import date

class PromptGenerator:
    """
    Generates structured prompts for different stages of LLM interaction.

    This class centralizes all prompt creation logic, ensuring consistency
    across LLM calls such as:
    - Conversation initialization
    - SQL generation
    - Assistant explanation generation
    - Guardrail and relevance validation

    Prompt templates are read from the local `prompts/` directory and
    populated using contextual inputs.
    """
    @staticmethod
    def generate_main_prompt():
        """
        Load and return the main system prompt for chatbot initialization.

        This prompt typically defines the assistant's role, behavior,
        and high-level instructions.

        Returns:
            str: The main system prompt text.
        """
        file_path = "prompts/chatbot_start.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        return prompt

    @staticmethod
    def generate_assistant_prompt(sql_query, df,num_of_records):
        """
        Generate the assistant prompt using SQL output and scenario context.

        A subset of the provided DataFrame is included in the prompt to
        prevent excessive token usage. If the DataFrame contains more than
        25 records, only the first 10 rows are included.

        Args:
            sql_query (str): SQL query executed to fetch the data.
            df (pandas.DataFrame): Query result data.
            scenario (str): Description of the analysis or explanation scenario.

        Returns:
            str: A formatted assistant prompt ready for LLM inference.
        """
        num_fields = df.shape[1]
        prompt_template = Path("prompts/chatbot_assistant.txt").read_text()
        assistanct_prompt = prompt_template.format(
            sql_query=sql_query,
            data_records=df.head(25).to_dict(orient="records"),
            num_records = num_of_records,
            num_fields= num_fields,
            date_today = "January 1st , 2026"
        )
        return assistanct_prompt

    @staticmethod
    def guardrail_check_inference_call(llm, user_input: str):
        """Call LLM to see if query is allowed"""
        with open("prompts/guard_rail_and_relevance.txt") as f:
            prompt = f.read()
        return llm.inference_single_input(user_input, prompt)

    @staticmethod
    def generate_sql_prompt(user_query, metadata, last_sql=None):
        """
        Generate a prompt for converting a natural language query into SQL.

        The prompt includes table metadata and optionally the previously
        generated SQL query to support iterative refinement.

        Args:
            user_query (str): Natural language question from the user.
            metadata (str): Database schema or table metadata.
            last_sql (str, optional): Previously generated SQL query.

        Returns:
            str: A formatted text-to-SQL prompt.
        """
        if last_sql:
            print(f"Last SQL: {type(last_sql)}")
            last_sql = f"Previous SQL (if any):\n{last_sql}"
        else:
            last_sql = "────────────────────────"

        template = Path("prompts/text_2_sql_ritu.txt").read_text()
        prompt = template.format(
            user_query=user_query,
            metadata=metadata,
            last_sql_query=last_sql,
            date_today = "January 1st , 2026"
        )
        return prompt


