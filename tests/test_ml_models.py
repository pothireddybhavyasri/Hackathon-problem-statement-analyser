"""
Unit tests for src/regression_models.py, src/classification_models.py, and src/clustering.py
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_raw_data
from src.data_cleaning import clean_hackathon_dataset
from src.regression_models import (
    prepare_regression_dataset, train_linear_regression,
    train_decision_tree_regressor, train_random_forest_regressor, compare_regression_models
)
from src.classification_models import (
    prepare_classification_dataset, train_logistic_regression,
    train_decision_tree_classifier, train_random_forest_classifier,
    compute_tree_depth_overfitting_curve, compare_classification_models
)
from src.clustering import (
    run_kmeans_clustering, compute_elbow_and_silhouette, prepare_clustering_data
)


@pytest.fixture(scope="module")
def cleaned_dataset():
    raw_df = load_raw_data()
    df, _ = clean_hackathon_dataset(raw_df, save_to_disk=False)
    return df


def test_regression_pipeline(cleaned_dataset):
    reg_data = prepare_regression_dataset(cleaned_dataset, test_size=0.20, random_state=42)
    assert len(reg_data["X_train"]) > 0
    assert len(reg_data["X_test"]) > 0
    assert len(reg_data["feature_names"]) > 5

    # Linear Regression
    lr_res = train_linear_regression(reg_data)
    assert "test_metrics" in lr_res
    assert "R2" in lr_res["test_metrics"]
    assert "diagnostics" in lr_res
    assert len(lr_res["coefficients_df"]) == len(reg_data["feature_names"])

    # Decision Tree Regressor
    dt_res = train_decision_tree_regressor(reg_data, max_depth=3)
    assert "R2" in dt_res["test_metrics"]

    # Random Forest Regressor
    rf_res = train_random_forest_regressor(reg_data, n_estimators=20, max_depth=4)
    assert "R2" in rf_res["test_metrics"]

    # Comparison Table
    comp = compare_regression_models(reg_data)
    assert len(comp) == 4
    assert "Test R²" in comp.columns


def test_classification_pipeline(cleaned_dataset):
    clf_data = prepare_classification_dataset(cleaned_dataset, test_size=0.20, random_state=42)
    assert len(clf_data["X_train"]) > 0
    assert len(clf_data["X_test"]) > 0

    # Logistic Regression
    lr_res = train_logistic_regression(clf_data)
    assert "test_metrics" in lr_res
    assert "Accuracy" in lr_res["test_metrics"]
    assert "F1_Score" in lr_res["test_metrics"]
    assert "confusion_matrix" in lr_res

    # Decision Tree Classifier
    dt_res = train_decision_tree_classifier(clf_data, max_depth=4)
    assert "Accuracy" in dt_res["test_metrics"]

    # Random Forest Classifier
    rf_res = train_random_forest_classifier(clf_data, n_estimators=20, max_depth=4)
    assert "Accuracy" in rf_res["test_metrics"]

    # Overfitting curve
    curve_df = compute_tree_depth_overfitting_curve(clf_data, max_depth_limit=5)
    assert len(curve_df) == 5
    assert "Train Accuracy" in curve_df.columns
    assert "Test Accuracy" in curve_df.columns

    # Comparison Table
    comp = compare_classification_models(clf_data)
    assert len(comp) == 3
    assert "F1-Score" in comp.columns


def test_clustering_pipeline(cleaned_dataset):
    X, cols, sub_df = prepare_clustering_data(cleaned_dataset)
    assert X.shape[0] > 0
    assert X.shape[1] == len(cols)

    # Elbow and Silhouette
    elbow_df = compute_elbow_and_silhouette(X, k_range=range(2, 5))
    assert len(elbow_df) == 3
    assert "Inertia (WCSS)" in elbow_df.columns
    assert "Silhouette Score" in elbow_df.columns

    # KMeans run
    res = run_kmeans_clustering(cleaned_dataset, n_clusters=3)
    assert res["n_clusters"] == 3
    assert res["silhouette_score"] > -1.0
    assert not res["pca_data"].empty
    assert "PCA_1" in res["pca_data"].columns
    assert len(res["cluster_profiles"]) == 3
