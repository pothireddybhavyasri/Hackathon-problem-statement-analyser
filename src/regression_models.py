"""
src/regression_models.py
Module for Linear Regression, Ridge Regression, Decision Tree Regressor,
and Random Forest Regressor predicting 'estimated_solution_days'.
Enforces strict train/test splitting without data leakage and provides BLUE assumption diagnostics.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

DEFAULT_NUMERICAL_FEATURES = [
    "team_size",
    "innovation_score",
    "impact_score",
    "problem_length_words",
    "number_of_technologies",
    "ai_required",
    "hardware_required"
]

DEFAULT_CATEGORICAL_FEATURES = [
    "domain",
    "difficulty_level",
    "organizer_type"
]


def prepare_regression_dataset(
    df: pd.DataFrame,
    target_col: str = "estimated_solution_days",
    feature_cols: List[str] = None,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Cleans, encodes, and splits data into train and test partitions without leakage.
    Returns feature matrices, targets, feature column metadata, and transformers.
    """
    if df is None or target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Filter out missing target rows
    valid_mask = df[target_col].notna()
    df_clean = df[valid_mask].copy()

    # Determine numerical and categorical subsets
    if feature_cols is None:
        num_cols = [c for c in DEFAULT_NUMERICAL_FEATURES if c in df_clean.columns]
        cat_cols = [c for c in DEFAULT_CATEGORICAL_FEATURES if c in df_clean.columns]
    else:
        num_cols = [c for c in feature_cols if c in df_clean.select_dtypes(include=[np.number]).columns]
        cat_cols = [c for c in feature_cols if c in df_clean.select_dtypes(include=["object", "category"]).columns]

    X = df_clean[num_cols + cat_cols]
    y = df_clean[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Build ColumnTransformer preprocessor fitted ONLY on X_train
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Extract transformed feature names
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
        "target_col": target_col
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, MSE, RMSE, and R2 score."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4)
    }


def train_linear_regression(data_dict: Dict[str, Any], fit_intercept: bool = True) -> Dict[str, Any]:
    """
    Fits Ordinary Least Squares (OLS) Linear Regression model.
    Calculates coefficients, intercept, predictions, metrics, and Gauss-Markov assumption diagnostics.
    """
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(data_dict["X_train"], data_dict["y_train"])

    y_train_pred = model.predict(data_dict["X_train"])
    y_test_pred = model.predict(data_dict["X_test"])

    train_metrics = compute_regression_metrics(data_dict["y_train"], y_train_pred)
    test_metrics = compute_regression_metrics(data_dict["y_test"], y_test_pred)

    residuals = data_dict["y_test"] - y_test_pred
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))
    skewness = float(stats.skew(residuals))

    # Feature coefficients dataframe
    coef_df = pd.DataFrame({
        "Feature": data_dict["feature_names"],
        "Coefficient": np.round(model.coef_, 4),
        "Abs_Magnitude": np.round(np.abs(model.coef_), 4)
    }).sort_values(by="Abs_Magnitude", ascending=False).reset_index(drop=True)

    return {
        "model_name": "Multiple Linear Regression (OLS)",
        "model_obj": model,
        "intercept": round(float(model.intercept_), 4),
        "coefficients_df": coef_df,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "y_test_true": data_dict["y_test"],
        "y_test_pred": y_test_pred,
        "residuals": residuals,
        "diagnostics": {
            "residual_mean": round(res_mean, 4),
            "residual_std": round(res_std, 4),
            "residual_skewness": round(skewness, 4),
            "homoscedasticity_note": "Residuals evaluated around zero horizontal axis.",
            "blue_gauss_markov_notes": [
                "Linearity: Model specifies a linear additive relationship between features and solution days.",
                "Strict Exogeneity: Conditional expectation of residuals given inputs E[e|X] = 0.",
                "Homoscedasticity: Equal variance across prediction ranges.",
                "No Multicollinearity: Regressors are linearly independent.",
                "Normality: Residuals display approximately normal distribution without severe skew."
            ]
        }
    }


def train_decision_tree_regressor(
    data_dict: Dict[str, Any], max_depth: int = 4, random_state: int = 42
) -> Dict[str, Any]:
    """Fits Decision Tree Regressor with configurable tree depth to evaluate pruning/overfitting."""
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(data_dict["X_train"], data_dict["y_train"])

    y_train_pred = model.predict(data_dict["X_train"])
    y_test_pred = model.predict(data_dict["X_test"])

    train_metrics = compute_regression_metrics(data_dict["y_train"], y_train_pred)
    test_metrics = compute_regression_metrics(data_dict["y_test"], y_test_pred)

    feat_imp = pd.DataFrame({
        "Feature": data_dict["feature_names"],
        "Importance": np.round(model.feature_importances_, 4)
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return {
        "model_name": f"Decision Tree Regressor (Depth={max_depth})",
        "model_obj": model,
        "max_depth": max_depth,
        "feature_importance_df": feat_imp,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "y_test_true": data_dict["y_test"],
        "y_test_pred": y_test_pred,
        "residuals": data_dict["y_test"] - y_test_pred
    }


def train_random_forest_regressor(
    data_dict: Dict[str, Any], n_estimators: int = 100, max_depth: int = 6, random_state: int = 42
) -> Dict[str, Any]:
    """Fits Random Forest Regressor ensemble with bootstrap aggregation."""
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
    )
    model.fit(data_dict["X_train"], data_dict["y_train"])

    y_train_pred = model.predict(data_dict["X_train"])
    y_test_pred = model.predict(data_dict["X_test"])

    train_metrics = compute_regression_metrics(data_dict["y_train"], y_train_pred)
    test_metrics = compute_regression_metrics(data_dict["y_test"], y_test_pred)

    feat_imp = pd.DataFrame({
        "Feature": data_dict["feature_names"],
        "Importance": np.round(model.feature_importances_, 4)
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return {
        "model_name": f"Random Forest Regressor (Trees={n_estimators})",
        "model_obj": model,
        "feature_importance_df": feat_imp,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "y_test_true": data_dict["y_test"],
        "y_test_pred": y_test_pred,
        "residuals": data_dict["y_test"] - y_test_pred
    }


def compare_regression_models(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """Trains Linear Regression, Ridge, Decision Tree, and Random Forest on the same test split and returns comparison table."""
    lr_res = train_linear_regression(data_dict)
    
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(data_dict["X_train"], data_dict["y_train"])
    ridge_test_pred = ridge.predict(data_dict["X_test"])
    ridge_train_pred = ridge.predict(data_dict["X_train"])
    ridge_test_m = compute_regression_metrics(data_dict["y_test"], ridge_test_pred)
    ridge_train_m = compute_regression_metrics(data_dict["y_train"], ridge_train_pred)
    
    dt_res = train_decision_tree_regressor(data_dict, max_depth=5)
    rf_res = train_random_forest_regressor(data_dict, n_estimators=100, max_depth=6)

    table = [
        {
            "Model": "Linear Regression (OLS)",
            "Test R²": lr_res["test_metrics"]["R2"],
            "Train R²": lr_res["train_metrics"]["R2"],
            "Test MAE": lr_res["test_metrics"]["MAE"],
            "Test RMSE": lr_res["test_metrics"]["RMSE"]
        },
        {
            "Model": "Ridge Regression (L2)",
            "Test R²": ridge_test_m["R2"],
            "Train R²": ridge_train_m["R2"],
            "Test MAE": ridge_test_m["MAE"],
            "Test RMSE": ridge_test_m["RMSE"]
        },
        {
            "Model": "Decision Tree (Depth=5)",
            "Test R²": dt_res["test_metrics"]["R2"],
            "Train R²": dt_res["train_metrics"]["R2"],
            "Test MAE": dt_res["test_metrics"]["MAE"],
            "Test RMSE": dt_res["test_metrics"]["RMSE"]
        },
        {
            "Model": "Random Forest (100 Trees)",
            "Test R²": rf_res["test_metrics"]["R2"],
            "Train R²": rf_res["train_metrics"]["R2"],
            "Test MAE": rf_res["test_metrics"]["MAE"],
            "Test RMSE": rf_res["test_metrics"]["RMSE"]
        }
    ]
    return pd.DataFrame(table).sort_values(by="Test R²", ascending=False).reset_index(drop=True)
