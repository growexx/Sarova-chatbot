import builtins
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

from code_modules.prompt_generator import PromptGenerator

def test_generate_main_prompt_reads_file():
    generator = PromptGenerator()
    expected_prompt = "Hello chatbot!"

    with patch.object(
        builtins, "open", mock_open(read_data=expected_prompt)
    ):
        result = generator.generate_main_prompt()

    assert result == expected_prompt


def test_generate_assistant_prompt_small_dataframe():
    generator = PromptGenerator()

    df = pd.DataFrame(
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    )

    template = (
        "SQL:{sql_query}\n"
        "DATA:{data_records}\n"
        "ROWS:{num_records}\n"
        "COLS:{num_fields}\n"
    )

    with patch.object(Path, "read_text", return_value=template):
        result = generator.generate_assistant_prompt(
            sql_query="SELECT * FROM test",
            df=df,
            num_of_records=len(df)   # ✅ NEW
        )

    assert "SELECT * FROM test" in result
    assert "ROWS:2" in result
    assert "COLS:2" in result


def test_generate_assistant_prompt_large_dataframe_uses_head():
    generator = PromptGenerator()

    df = pd.DataFrame(
        [{"a": i, "b": i * 2} for i in range(30)]
    )

    template = "DATA:{data_records}"

    with patch.object(Path, "read_text", return_value=template):
        result = generator.generate_assistant_prompt(
            sql_query="SQL",
            df=df,
            num_of_records=len(df)  # ✅ NEW
        )

    records = eval(result.replace("DATA:", ""))

    # New logic returns head(25)
    assert len(records) == 25

def test_guardrail_check_inference_call():
    generator = PromptGenerator()

    mock_llm = MagicMock()
    mock_llm.inference_single_input.return_value = "ALLOWED"

    with patch.object(
        builtins, "open", mock_open(read_data="guard prompt")
    ):
        result = generator.guardrail_check_inference_call(
            llm=mock_llm,
            user_input="Can I see sales data?"
        )

    mock_llm.inference_single_input.assert_called_once_with(
        "Can I see sales data?", "guard prompt"
    )
    assert result == "ALLOWED"


def test_generate_sql_prompt_with_last_sql():
    generator = PromptGenerator()

    template = (
        "Q:{user_query}\n"
        "META:{metadata}\n"
        "LAST:{last_sql_query}"
    )

    with patch.object(Path, "read_text", return_value=template):
        result = generator.generate_sql_prompt(
            user_query="Show sales",
            metadata="table info",
            last_sql="SELECT * FROM sales"
        )

    assert "Previous SQL" in result
    assert "SELECT * FROM sales" in result


def test_generate_sql_prompt_without_last_sql():
    generator = PromptGenerator()

    template = "LAST:{last_sql_query}"

    with patch.object(Path, "read_text", return_value=template):
        result = generator.generate_sql_prompt(
            user_query="Query",
            metadata="Meta"
        )

    assert "────────────────────────" in result
