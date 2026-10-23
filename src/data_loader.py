"""
src/data_loader.py
Module for locating, loading, and inspecting raw and cleaned hackathon problem statement datasets.
"""

import os
from typing import Optional, Dict, Any, Tuple
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "hackathon_problem_statements.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "hackathon_problem_statements_cleaned.csv")


def get_raw_data_path() -> str:
    """Returns absolute path to raw dataset CSV."""
    return RAW_DATA_PATH


def get_cleaned_data_path() -> str:
    """Returns absolute path to cleaned dataset CSV."""
    return CLEANED_DATA_PATH


def check_dataset_exists(is_cleaned: bool = False) -> bool:
    """Checks whether the requested dataset file exists on disk."""
    path = get_cleaned_data_path() if is_cleaned else get_raw_data_path()
    return os.path.exists(path) and os.path.getsize(path) > 0


def load_raw_data() -> pd.DataFrame:
    """
    Loads the unmodified raw dataset from data/hackathon_problem_statements.csv.
    Raises FileNotFoundError with a clear message if the file is absent.
    """
    path = get_raw_data_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'. Please run the generation script 'scripts/generate_dataset.py'."
        )
    return pd.read_csv(path)


def load_cleaned_data() -> Optional[pd.DataFrame]:
    """
    Loads the cleaned dataset if it exists, otherwise returns None.
    """
    path = get_cleaned_data_path()
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts high-level descriptive metadata from a dataset DataFrame.
    """
    if df is None or df.empty:
        return {
            "rows": 0,
            "columns": 0,
            "memory_kb": 0.0,
            "unique_ids": 0,
            "sourced_count": 0,
            "synthetic_count": 0,
            "missing_cells": 0,
            "duplicate_rows": 0,
        }
    
    unique_ids = df["problem_id"].nunique() if "problem_id" in df.columns else 0
    sourced_count = int((df["data_type"] == "sourced").sum()) if "data_type" in df.columns else 0
    synthetic_count = int((df["data_type"] == "synthetic").sum()) if "data_type" in df.columns else 0
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    mem_kb = round(df.memory_usage(deep=True).sum() / 1024.0, 2)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_kb": mem_kb,
        "unique_ids": unique_ids,
        "sourced_count": sourced_count,
        "synthetic_count": synthetic_count,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
    }
