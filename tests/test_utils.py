import re
import time
import pytest
from unittest.mock import patch

from code_modules.utils import (
    wrap_par_around_file,
    prepare_local_file_and_par_url,
    log_time,_load_all_metadata
)


def test_wrap_par_around_file_basic():
    file_path = "data.xlsx"
    bucket = "test_bucket"

    result = wrap_par_around_file(file_path, bucket)

    assert bucket in result
    assert file_path in result
    assert result.startswith("https://")


def test_wrap_par_around_file_exact():
    file_path = "file.xlsx"
    bucket = "my_folder"

    result = wrap_par_around_file(file_path, bucket)

    expected = f"https://objectstorage.me-dubai-1.oraclecloud.com" \
               f"/p/oawta1HMX-BgQZkdRtaJUVt6E8lTOa5vEzC3ZqeIuc7i649VOG2VHlBRxinPm8Ny" \
               f"/n/bmb8tbvmgtsy/b/Sarova_extended/o/{bucket}/{file_path}"

    assert result == expected

@patch("code_modules.utils.uuid.uuid4")
@patch("code_modules.utils.datetime")
def test_prepare_local_file_and_par_url(mock_datetime, mock_uuid):
    # Mock timestamp
    mock_datetime.utcnow.return_value.strftime.return_value = "20260101_120000"

    # Mock UUID
    mock_uuid.return_value.hex = "abcdef1234567890"

    filename, par_url = prepare_local_file_and_par_url()

    assert filename == "result_data_20260101_120000_abcdef12.xlsx"
    assert "Sarova_Table_files/result_data_20260101_120000_abcdef12.xlsx" in par_url
    assert par_url.startswith("https://")

@patch("code_modules.utils.uuid.uuid4")
@patch("code_modules.utils.datetime")
def test_prepare_local_file_custom_inputs(mock_datetime, mock_uuid):
    mock_datetime.utcnow.return_value.strftime.return_value = "20260101_120000"
    mock_uuid.return_value.hex = "1234567890abcdef"

    filename, par_url = prepare_local_file_and_par_url(
        file_prefix="custom",
        bucket_folder_name="my_bucket"
    )

    assert filename.startswith("custom_20260101_120000_12345678")
    assert "my_bucket/" in par_url

def test_prepare_local_file_format():
    filename, par_url = prepare_local_file_and_par_url()

    # Filename pattern check
    pattern = r"result_data_\d{8}_\d{6}_[a-f0-9]{8}\.xlsx"
    assert re.match(pattern, filename)

    assert filename in par_url

def test_log_time_output(capsys):
    start = time.perf_counter()

    log_time("test_label", start)

    captured = capsys.readouterr()

    assert "[TIMER]" in captured.out
    assert "test_label" in captured.out
    assert "seconds" in captured.out


def test_log_time_elapsed_positive(capsys):
    start = time.perf_counter()

    time.sleep(0.01)  # small delay
    log_time("delay_test", start)

    captured = capsys.readouterr()

    # Extract number from output
    import re
    match = re.search(r"took ([0-9.]+)", captured.out)

    assert match is not None
    assert float(match.group(1)) > 0

import os
import pandas as pd
import pytest

from code_modules.utils import generate_categorical_plots

def test_no_categorical_columns(tmp_path):
    df = pd.DataFrame({
        "num1": [1, 2, 3],
        "num2": [4, 5, 6]
    })

    result = generate_categorical_plots(df, tmp_path, "test")

    assert result == []

def test_no_numeric_columns(tmp_path):
    df = pd.DataFrame({
        "cat1": ["A", "B", "C"]
    })

    result = generate_categorical_plots(df, tmp_path, "test")

    assert result == []

def test_low_variance_numeric(tmp_path):
    df = pd.DataFrame({
        "cat": ["A", "B", "C"],
        "num": [1.0, 1.0, 1.0]  # no variance
    })

    result = generate_categorical_plots(df, tmp_path, "test")

    assert result == []

def test_single_categorical_plot(tmp_path):
    df = pd.DataFrame({
        "category": ["A", "B", "C"],
        "value": [10, 20, 15]
    })

    result = generate_categorical_plots(df, tmp_path, "test")

    assert len(result) == 1
    assert os.path.exists(result[0])
    assert result[0].endswith("_bar.png")

def test_two_categorical_grouped_bar(tmp_path):
    df = pd.DataFrame({
        "cat1": ["A", "A", "B", "B"],
        "cat2": ["X", "Y", "X", "Y"],  # small cardinality
        "value": [10, 20, 15, 25]
    })

    result = generate_categorical_plots(df, tmp_path, "test")

    assert len(result) == 1
    assert os.path.exists(result[0])
    assert "grouped_bar" in result[0]

def test_two_categorical_heatmap(tmp_path):
    df = pd.DataFrame({
        "cat1": ["A", "A", "B", "B", "C", "C"],
        "cat2": ["X", "Y", "Z", "W", "Q", "P"],  # high cardinality (>5)
        "value": [10, 20, 15, 25, 30, 35]
    })

    result = generate_categorical_plots(df, tmp_path, "test")

    assert len(result) == 1
    assert os.path.exists(result[0])
    assert "heatmap" in result[0]

def test_category_string_truncation(tmp_path):
    df = pd.DataFrame({
        "category": ["A" * 100, "B" * 100],
        "value": [10, 20]
    })

    result = generate_categorical_plots(df, tmp_path, "test")

    assert len(result) == 1
    assert os.path.exists(result[0])

def test_max_categories_limit(tmp_path):
    df = pd.DataFrame({
        "category": [f"cat{i}" for i in range(20)],
        "value": list(range(20))
    })

    result = generate_categorical_plots(df, tmp_path, "test", max_categories=5)

    assert len(result) == 1
    assert os.path.exists(result[0])

def test_exception_handling(monkeypatch, tmp_path):
    df = pd.DataFrame({
        "category": ["A", "B"],
        "value": [10, 20]
    })

    # Force failure
    def mock_makedirs(*args, **kwargs):
        raise Exception("fail")

    monkeypatch.setattr("os.makedirs", mock_makedirs)

    result = generate_categorical_plots(df, tmp_path, "test")

    assert result == []

import json
from unittest.mock import mock_open, patch


def test_load_all_metadata_partial_missing(capsys):
    # Mock file content for one table
    mock_file = mock_open(read_data=json.dumps({"col": "value"}))

    def side_effect(file, *args, **kwargs):
        if "table1" in file:
            return mock_file()
        else:
            raise FileNotFoundError

    with patch("code_modules.utils.open", side_effect=side_effect):
        with patch("code_modules.utils._ALL_TABLES", ["table1", "table2"]):
            result = _load_all_metadata()

    captured = capsys.readouterr()

    assert "TABLE: TABLE1" in result
    assert "File not found" in captured.out

import pandas as pd
from unittest.mock import patch
from code_modules.utils import generate_categorical_plots


@patch("code_modules.utils.plt.savefig")
@patch("code_modules.utils.plt.close")
def test_high_cardinality_cat1(mock_close, mock_savefig, tmp_path):
    # Create dataframe with > max_categories unique values
    data = {
        "cat1": [f"A{i}" for i in range(20)],   # 20 unique
        "cat2": ["X"] * 20,                   # small cardinality
        "value": list(range(20))
    }
    df = pd.DataFrame(data)

    result = generate_categorical_plots(
        df,
        output_dir=tmp_path,
        file_prefix="test",
        max_categories=5   # 👈 trigger condition
    )

    # Should still generate plot
    assert len(result) == 1
    mock_savefig.assert_called_once()

from unittest.mock import patch


@patch("code_modules.utils.pd.DataFrame.select_dtypes")
def test_no_categorical_columns_else_branch(mock_select_dtypes, tmp_path):
    import pandas as pd

    df = pd.DataFrame({"num": [1, 2, 3]})

    # First call → categorical_cols
    # Second call → numeric_cols
    mock_select_dtypes.side_effect = [
        pd.DataFrame().columns,   # empty categorical
        ["num"]                   # numeric exists
    ]

    result = generate_categorical_plots(
        df,
        output_dir=tmp_path,
        file_prefix="test"
    )

    assert result == []