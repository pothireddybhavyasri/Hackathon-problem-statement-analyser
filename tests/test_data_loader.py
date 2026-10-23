"""
Unit tests for src/data_loader.py
"""

import os
import pytest
import pandas as pd
from src.data_loader import (
    load_raw_data, load_cleaned_data, check_dataset_exists,
    get_dataset_metadata, get_raw_data_path, get_cleaned_data_path
)


def test_paths_valid():
    raw_path = get_raw_data_path()
    clean_path = get_cleaned_data_path()
    assert raw_path.endswith("hackathon_problem_statements.csv")
    assert clean_path.endswith("hackathon_problem_statements_cleaned.csv")


def test_check_dataset_exists():
    assert check_dataset_exists(is_cleaned=False) is True


def test_load_raw_data():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) >= 500
    assert "problem_id" in df.columns
    assert "domain" in df.columns
    assert "data_type" in df.columns


def test_metadata_computation():
    df = load_raw_data()
    meta = get_dataset_metadata(df)
    assert meta["rows"] >= 500
    assert meta["columns"] >= 20
    assert meta["sourced_count"] > 0
    assert meta["synthetic_count"] > 0
    assert meta["memory_kb"] > 0.0


def test_metadata_empty():
    meta = get_dataset_metadata(pd.DataFrame())
    assert meta["rows"] == 0
    assert meta["columns"] == 0
