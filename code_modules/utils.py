import re
import os
import json
import uuid
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Optional, Tuple

_ALL_TABLES = ["inventory_consumption_data", "occupancy_revenue_data", "supplier_scoring_data"]

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
            print(f"File not found: {file_name}")
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
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        os.makedirs(output_dir, exist_ok=True)
        plot_paths = []

        # Detect columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not categorical_cols or not numeric_cols:
            print("No categorical and Numerical columns found")
            return plot_paths

        num_col = df[numeric_cols].std().idxmax()
        if df[num_col].std() < 0.01:
            print("Values too similar → chart not informative")
            return plot_paths

        # -----------------------
        # CASE 1: One categorical
        # -----------------------
        if len(categorical_cols) == 1:
            cat = categorical_cols[0]
            df[cat] = df[cat].astype(str).str.slice(0, 40)

            # Reduce categories
            if df[cat].nunique() > max_categories:
                df = df.nlargest(max_categories, num_col)

            df_sorted = df.sort_values(num_col, ascending=False)
            plt.figure(figsize=(10, 6))

            sns.barplot( data=df_sorted, y=cat, x=num_col, errorbar=None )

            plt.title(f"{num_col} by {cat}")
            plt.ylabel(cat)
            plt.xlabel(num_col)
            plt.tight_layout()
            path = os.path.join(output_dir, f"_{file_prefix}_{cat}_{timestamp}_{unique_id}_bar.png")
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
                    output_dir, f"{file_prefix}_{cat1}_{cat2}_{timestamp}_{unique_id}_grouped_bar.png"
                )

            # Many values → Heatmap
            else:
                pivot = df.pivot_table( index=cat2, columns=cat1, values=num_col, aggfunc="sum")

                plt.figure(figsize=(10, 6))
                sns.heatmap(pivot, cmap="Blues")
                plt.title(f"{num_col} Heatmap ({cat1} vs {cat2})")
                path = os.path.join(
                    output_dir, f"{file_prefix}_{cat1}_{cat2}_{timestamp}_{unique_id}_heatmap.png"
                )

            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            plot_paths.append(path)

        return plot_paths
    except Exception as e:
        print(f"Failed to genereate plots due to error {e}")
        return []

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

def wrap_par_around_file(file_path,bucket_folder_name):
    return f"https://objectstorage.me-dubai-1.oraclecloud.com/p/oawta1HMX-BgQZkdRtaJUVt6E8lTOa5vEzC3ZqeIuc7i649VOG2VHlBRxinPm8Ny/n/bmb8tbvmgtsy/b/Sarova_extended/o/{bucket_folder_name}/{file_path}"

def prepare_local_file_and_par_url(file_prefix:str = "result_data", bucket_folder_name: str = "Sarova_Table_files")  -> Tuple[str, str]:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    local_filename = f"{file_prefix}_{timestamp}_{unique_id}.xlsx"
    object_name = bucket_folder_name +'/' + os.path.basename(local_filename)
    par_url=f"https://objectstorage.me-dubai-1.oraclecloud.com/p/oawta1HMX-BgQZkdRtaJUVt6E8lTOa5vEzC3ZqeIuc7i649VOG2VHlBRxinPm8Ny/n/bmb8tbvmgtsy/b/Sarova_extended/o/{object_name}"
    return local_filename, par_url

def log_time(label, start_time):
    elapsed = time.perf_counter() - start_time
    print(f"[TIMER] {label} took {elapsed:.4f} seconds")
