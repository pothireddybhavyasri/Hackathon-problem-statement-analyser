"""
Unit tests for src/eda.py and src/statistics_analysis.py
"""

import pytest
import pandas as pd
from src.data_loader import load_raw_data
from src.data_cleaning import clean_hackathon_dataset
from src.statistics_analysis import (
    compute_descriptive_stats, compute_categorical_frequencies,
    compute_crosstab_analysis, compute_correlation_matrix, perform_hypothesis_tests
)
from src.eda import (
    classify_variables, extract_technology_frequencies,
    compute_domain_subdomain_breakdown, compute_hardware_ai_summary
)


@pytest.fixture(scope="module")
def cleaned_dataset():
    raw_df = load_raw_data()
    df, _ = clean_hackathon_dataset(raw_df, save_to_disk=False)
    return df


def test_descriptive_stats(cleaned_dataset):
    stats_df = compute_descriptive_stats(cleaned_dataset)
    assert not stats_df.empty
    assert "Mean" in stats_df.columns
    assert "Median" in stats_df.columns
    assert "IQR" in stats_df.columns
    assert "estimated_solution_days" in stats_df.index


def test_categorical_frequencies(cleaned_dataset):
    freq_df = compute_categorical_frequencies(cleaned_dataset, "domain")
    assert not freq_df.empty
    assert "Frequency" in freq_df.columns
    assert "Percentage (%)" in freq_df.columns
    assert abs(freq_df["Percentage (%)"].sum() - 100.0) < 1.0


def test_crosstab_analysis(cleaned_dataset):
    xtab = compute_crosstab_analysis(cleaned_dataset, "domain", "difficulty_level")
    assert "counts" in xtab
    assert "percentages" in xtab
    assert not xtab["counts"].empty


def test_correlation_matrix(cleaned_dataset):
    corr = compute_correlation_matrix(cleaned_dataset)
    assert not corr.empty
    assert corr.shape[0] == corr.shape[1]
    # Diagonal must be 1.0
    for col in corr.columns:
        assert abs(corr.loc[col, col] - 1.0) < 0.001


def test_hypothesis_testing(cleaned_dataset):
    tests = perform_hypothesis_tests(cleaned_dataset)
    assert "anova_domain_days" in tests
    assert "chi2_domain_ai" in tests
    assert "ttest_ai_days" in tests
    assert "f_statistic" in tests["anova_domain_days"]


def test_eda_variable_classification(cleaned_dataset):
    tax = classify_variables(cleaned_dataset)
    assert len(tax) == len(cleaned_dataset.columns)
    assert "Statistical Scale" in tax.columns


def test_eda_tech_frequencies(cleaned_dataset):
    tech_df = extract_technology_frequencies(cleaned_dataset, top_n=10)
    assert not tech_df.empty
    assert "Technology" in tech_df.columns
    assert "Count" in tech_df.columns


def test_eda_hardware_ai_summary(cleaned_dataset):
    summary = compute_hardware_ai_summary(cleaned_dataset)
    assert summary["total_problems"] == len(cleaned_dataset)
    assert "ai_required_count" in summary
