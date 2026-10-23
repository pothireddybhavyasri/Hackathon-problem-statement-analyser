"""
Unit tests for src/data_validation.py
"""

import pytest
import pandas as pd
from src.data_loader import load_raw_data, load_cleaned_data
from src.data_validation import validate_dataset_schema, EXPECTED_COLUMNS


def test_validation_on_raw_data():
    df = load_raw_data()
    report = validate_dataset_schema(df)
    assert "summary" in report
    assert "checks" in report
    assert report["summary"]["total"] >= 10
    # Raw data has intentional warnings (duplicates, outliers, mixed casing)
    assert report["summary"]["warning"] > 0


def test_validation_on_empty_df():
    report = validate_dataset_schema(pd.DataFrame())
    assert report["summary"]["fail"] > 0


def test_validation_on_minimal_valid_df():
    data = {col: [1 if "days" in col or "score" in col or "size" in col or "words" in col or "number" in col or "year" in col else "test"] for col in EXPECTED_COLUMNS}
    data["year"] = [2023]
    data["estimated_solution_days"] = [15.0]
    data["team_size"] = [4]
    data["ai_required"] = [1]
    data["hardware_required"] = [0]
    data["domain"] = ["Healthcare"]
    data["problem_description"] = ["This is a substantive description of more than five words long."]
    data["source_url"] = ["https://sih.gov.in"]
    data["data_type"] = ["sourced"]
    df = pd.DataFrame(data)
    
    report = validate_dataset_schema(df)
    assert report["summary"]["fail"] == 0
