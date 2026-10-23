"""
src/classification_models.py
Module for Logistic Regression, Decision Tree Classifier, and Random Forest Classifier
predicting binary categorical target 'ai_required' (0 or 1).
Includes confusion matrices, precision/recall/F1, ROC curves, and pruning overfitting analysis.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

DEFAULT_NUMERICAL_FEATURES = [
    "team_size",
    "innovation_score",
    "impact_score",
    "problem_length_words",
    "number_of_technologies",
    "estimated_solution_days",
    "hardware_required"
]

DEFAULT_CATEGORICAL_FEATURES = [
    "domain",
    "difficulty_level",
    "organizer_type",
    "data_availability"
]


def prepare_classification_dataset(
    df: pd.DataFrame,
    target_col: str = "ai_required",
    feature_cols: List[str] = None,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Cleans, encodes, and splits data for binary classification without data leakage.
    Target must be 0 or 1.
    """
    if df is None or target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Drop any null target records
    valid_mask = df[target_col].notna()
    df_clean = df[valid_mask].copy()

    y = df_clean[target_col].astype(int)

    if feature_cols is None:
        num_cols = [c for c in DEFAULT_NUMERICAL_FEATURES if c in df_clean.columns]
        cat_cols = [c for c in DEFAULT_CATEGORICAL_FEATURES if c in df_clean.columns]
    else:
        num_cols = [c for c in feature_cols if c in df_clean.select_dtypes(include=[np.number]).columns]
        cat_cols = [c for c in feature_cols if c in df_clean.select_dtypes(include=["object", "category"]).columns]

    X = df_clean[num_cols + cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    cat_names = []
    if cat_cols:
        cat_encoder = preprocessor.named_transformers_["cat"]
        if hasattr(cat_encoder, "get_feature_names_out"):
            cat_names = list(cat_encoder.get_feature_names_out(cat_cols))

    all_feature_names = num_cols + cat_names

    return {
        "X_train": X_train_trans,
        "X_test": X_test_trans,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "feature_names": all_feature_names,
        "preprocessor": preprocessor,
        "raw_X_train": X_train,
        "raw_X_test": X_test,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target_col": target_col,
        "class_counts": y.value_counts().to_dict()
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """Computes Accuracy, Precision, Recall, F1, and ROC-AUC score."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    auc = np.nan
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(auc, 4) if not np.isnan(auc) else "N/A",
        "confusion_matrix": cm
    }


def train_logistic_regression(data_dict: Dict[str, Any], C: float = 1.0) -> Dict[str, Any]:
    """Fits Logistic Regression classifier with L2 regularization."""
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(data_dict["X_train"], data_dict["y_train"])

    y_train_pred = model.predict(data_dict["X_train"])
    y_test_pred = model.predict(data_dict["X_test"])
    y_test_prob = model.predict_proba(data_dict["X_test"])[:, 1]

    train_m = compute_classification_metrics(data_dict["y_train"], y_train_pred)
    test_m = compute_classification_metrics(data_dict["y_test"], y_test_pred, y_test_prob)

    fpr, tpr, _ = roc_curve(data_dict["y_test"], y_test_prob)

    coef_df = pd.DataFrame({
        "Feature": data_dict["feature_names"],
        "Odds_Ratio": np.round(np.exp(model.coef_[0]), 4),
        "Log_Odds_Coef": np.round(model.coef_[0], 4)
    }).sort_values(by="Log_Odds_Coef", ascending=False).reset_index(drop=True)

    return {
        "model_name": "Logistic Regression",
        "model_obj": model,
        "train_metrics": train_m,
        "test_metrics": test_m,
        "coefficients_df": coef_df,
        "y_test_true": data_dict["y_test"],
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "confusion_matrix": test_m["confusion_matrix"],
        "classification_report_str": classification_report(data_dict["y_test"], y_test_pred)
    }


def train_decision_tree_classifier(
    data_dict: Dict[str, Any], max_depth: int = 4, criterion: str = "gini", random_state: int = 42
) -> Dict[str, Any]:
    """Fits Decision Tree Classifier with adjustable depth to demonstrate pruning."""
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)
    model.fit(data_dict["X_train"], data_dict["y_train"])

    y_train_pred = model.predict(data_dict["X_train"])
    y_test_pred = model.predict(data_dict["X_test"])
    y_test_prob = model.predict_proba(data_dict["X_test"])[:, 1]

    train_m = compute_classification_metrics(data_dict["y_train"], y_train_pred)
    test_m = compute_classification_metrics(data_dict["y_test"], y_test_pred, y_test_prob)

    feat_imp = pd.DataFrame({
        "Feature": data_dict["feature_names"],
        "Importance": np.round(model.feature_importances_, 4)
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return {
        "model_name": f"Decision Tree (Depth={max_depth}, Criterion={criterion})",
        "model_obj": model,
        "max_depth": max_depth,
        "train_metrics": train_m,
        "test_metrics": test_m,
        "feature_importance_df": feat_imp,
        "confusion_matrix": test_m["confusion_matrix"],
        "y_test_true": data_dict["y_test"],
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob
    }


def train_random_forest_classifier(
    data_dict: Dict[str, Any], n_estimators: int = 100, max_depth: int = 6, random_state: int = 42
) -> Dict[str, Any]:
    """Fits Random Forest Classifier ensemble."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
    )
    model.fit(data_dict["X_train"], data_dict["y_train"])

    y_train_pred = model.predict(data_dict["X_train"])
    y_test_pred = model.predict(data_dict["X_test"])
    y_test_prob = model.predict_proba(data_dict["X_test"])[:, 1]

    train_m = compute_classification_metrics(data_dict["y_train"], y_train_pred)
    test_m = compute_classification_metrics(data_dict["y_test"], y_test_pred, y_test_prob)

    feat_imp = pd.DataFrame({
        "Feature": data_dict["feature_names"],
        "Importance": np.round(model.feature_importances_, 4)
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return {
        "model_name": f"Random Forest ({n_estimators} Trees)",
        "model_obj": model,
        "train_metrics": train_m,
        "test_metrics": test_m,
        "feature_importance_df": feat_imp,
        "confusion_matrix": test_m["confusion_matrix"],
        "y_test_true": data_dict["y_test"],
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob
    }


def compute_tree_depth_overfitting_curve(data_dict: Dict[str, Any], max_depth_limit: int = 10) -> pd.DataFrame:
    """
    Evaluates Decision Tree across depths 1 to max_depth_limit on both training and test data.
    Demonstrates model complexity, pruning, and empirical overfitting.
    """
    records = []
    for d in range(1, max_depth_limit + 1):
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(data_dict["X_train"], data_dict["y_train"])

        train_acc = accuracy_score(data_dict["y_train"], dt.predict(data_dict["X_train"]))
        test_acc = accuracy_score(data_dict["y_test"], dt.predict(data_dict["X_test"]))
        train_f1 = f1_score(data_dict["y_train"], dt.predict(data_dict["X_train"]), zero_division=0)
        test_f1 = f1_score(data_dict["y_test"], dt.predict(data_dict["X_test"]), zero_division=0)

        records.append({
            "Depth": d,
            "Train Accuracy": round(train_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "Train F1": round(train_f1, 4),
            "Test F1": round(test_f1, 4),
            "Generalization Gap": round(abs(train_acc - test_acc), 4)
        })

    return pd.DataFrame(records)


def compare_classification_models(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """Evaluates all three classifiers on the exact same test partition."""
    lr_res = train_logistic_regression(data_dict)
    dt_res = train_decision_tree_classifier(data_dict, max_depth=5)
    rf_res = train_random_forest_classifier(data_dict, n_estimators=100, max_depth=6)

    table = [
        {
            "Model": "Logistic Regression",
            "Test Accuracy": lr_res["test_metrics"]["Accuracy"],
            "Precision": lr_res["test_metrics"]["Precision"],
            "Recall": lr_res["test_metrics"]["Recall"],
            "F1-Score": lr_res["test_metrics"]["F1_Score"],
            "ROC-AUC": lr_res["test_metrics"]["ROC_AUC"]
        },
        {
            "Model": "Decision Tree (Depth=5)",
            "Test Accuracy": dt_res["test_metrics"]["Accuracy"],
            "Precision": dt_res["test_metrics"]["Precision"],
            "Recall": dt_res["test_metrics"]["Recall"],
            "F1-Score": dt_res["test_metrics"]["F1_Score"],
            "ROC-AUC": dt_res["test_metrics"]["ROC_AUC"]
        },
        {
            "Model": "Random Forest (100 Trees)",
            "Test Accuracy": rf_res["test_metrics"]["Accuracy"],
            "Precision": rf_res["test_metrics"]["Precision"],
            "Recall": rf_res["test_metrics"]["Recall"],
            "F1-Score": rf_res["test_metrics"]["F1_Score"],
            "ROC-AUC": rf_res["test_metrics"]["ROC_AUC"]
        }
    ]
    return pd.DataFrame(table).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
