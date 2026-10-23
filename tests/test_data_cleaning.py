"""
Unit tests for src/data_cleaning.py
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_raw_data
from src.data_cleaning import clean_hackathon_dataset, detect_outliers_iqr, compute_quality_audit


def test_iqr_outlier_detection():
    df = pd.DataFrame({"vals": [10, 11, 12, 10, 11, 12, 100, -50]})
    res = detect_outliers_iqr(df, "vals")
    assert res["outlier_count"] >= 2
    assert res["iqr"] > 0
    assert res["upper_bound"] < 100
    assert res["lower_bound"] > -50


def test_quality_audit_computation():
    df = load_raw_data()
    audit = compute_quality_audit(df)
    assert audit["total_rows"] >= 500
    assert audit["total_columns"] >= 20
    assert "missing_by_column" in audit
    assert "outliers" in audit


def test_clean_hackathon_dataset_execution():
    raw_df = load_raw_data()
    cleaned_df, comparison = clean_hackathon_dataset(raw_df, save_to_disk=False)
    
    # Assert exact duplicates removed
    assert comparison["duplicates_after"] == 0
    assert comparison["rows_removed"] >= 4
    
    # Assert missing values resolved
    assert comparison["missing_after"] == 0
    
    # Assert duplicate IDs resolved
    assert comparison["dup_ids_after"] == 0
    
    # Assert boolean fields are strictly integer 0 or 1
    assert set(cleaned_df["ai_required"].unique()).issubset({0, 1})
    assert set(cleaned_df["hardware_required"].unique()).issubset({0, 1})
    
    # Assert operations log populated
    assert len(comparison["operations_performed"]) > 5
