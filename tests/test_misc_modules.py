"""
Unit tests for src/text_similarity.py, src/database.py, and src/time_series_analysis.py
"""

import pytest
import pandas as pd
from src.data_loader import load_raw_data
from src.data_cleaning import clean_hackathon_dataset
from src.text_similarity import compute_tfidf_matrix, search_similar_problems
from src.database import init_database, run_custom_query, get_sql_demonstrations
from src.time_series_analysis import compute_yearly_trends, fit_arima_forecast


@pytest.fixture(scope="module")
def cleaned_dataset():
    raw_df = load_raw_data()
    df, _ = clean_hackathon_dataset(raw_df, save_to_disk=False)
    return df


def test_tfidf_and_similarity(cleaned_dataset):
    tfidf_data = compute_tfidf_matrix(cleaned_dataset)
    assert tfidf_data["vocabulary_size"] > 100
    assert tfidf_data["corpus_size"] == len(cleaned_dataset)

    # Search
    results = search_similar_problems(
        query_text="wildfire IoT sensor alerts",
        df=cleaned_dataset,
        tfidf_data=tfidf_data,
        top_n=3
    )
    assert len(results) <= 3
    assert "similarity_score" in results.columns
    assert results.iloc[0]["similarity_score"] > 0.0


def test_database_operations(cleaned_dataset):
    init_res = init_database(cleaned_dataset)
    assert init_res["status"] == "success"
    assert init_res["problem_statements_count"] == len(cleaned_dataset)

    # Custom Query
    df, err = run_custom_query("SELECT COUNT(*) as total FROM problem_statements;")
    assert err is None
    assert df.iloc[0]["total"] == len(cleaned_dataset)

    # SQL Demos
    demos = get_sql_demonstrations()
    for name, q in demos.items():
        q_df, q_err = run_custom_query(q["sql"])
        assert q_err is None
        assert q_df is not None


def test_time_series_and_arima(cleaned_dataset):
    yearly = compute_yearly_trends(cleaned_dataset)
    assert not yearly.empty
    assert len(yearly) >= 4
    assert "total_problems" in yearly.columns
    assert "ai_adoption_pct" in yearly.columns

    # ARIMA forecast
    forecast_res = fit_arima_forecast(yearly, steps_ahead=2)
    assert "forecast_values" in forecast_res
    assert len(forecast_res["forecast_values"]) == 2
    assert "in_sample_metrics" in forecast_res
