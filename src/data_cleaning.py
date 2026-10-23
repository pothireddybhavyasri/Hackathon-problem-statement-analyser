"""
src/data_cleaning.py
Comprehensive data cleaning and data quality analysis module.
Implements IQR outlier detection, missing value imputation, text normalization,
deduplication, type conversion, and before-and-after audit metrics.
"""

import os
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

from src.data_loader import get_cleaned_data_path

# Standard category canonicalization mappings
DOMAIN_NORMALIZATION_MAP = {
    "healthcare": "Healthcare",
    "fintech": "FinTech",
    "agriculture": "Agriculture",
    "smart cities": "Smart Cities",
    "cybersecurity": "Cybersecurity",
    "clean energy": "Clean Energy",
    "disaster management": "Disaster Management",
    "smart education": "Smart Education",
    "transportation": "Transportation",
    "space tech": "Space Tech",
    "ai / ml": "Healthcare",
}

BOOLEAN_TRUE_SET = {1, "1", True, "true", "True", "yes", "YES", "Yes", "t", "T"}
BOOLEAN_FALSE_SET = {0, "0", False, "false", "False", "no", "NO", "No", "f", "F"}


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> Dict[str, Any]:
    """
    Computes IQR-based outlier boundaries and identifies outlier indices and values.
    Formulas:
        Q1 = 25th percentile
        Q3 = 75th percentile
        IQR = Q3 - Q1
        Lower Bound = Q1 - (multiplier * IQR)
        Upper Bound = Q3 + (multiplier * IQR)
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(series) < 4:
        return {"error": "Insufficient data to compute quartiles"}

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = round(q1 - multiplier * iqr, 2)
    upper_bound = round(q3 + multiplier * iqr, 2)

    outliers = series[(series < lower_bound) | (series > upper_bound)]

    return {
        "column": column,
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": int(len(outliers)),
        "outlier_percentage": round(float(len(outliers) / len(series) * 100.0), 2),
        "outlier_values": sorted(outliers.tolist())[:10]
    }


def compute_quality_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a full data quality audit for the dataset, returning per-column missingness,
    data types, unique value counts, duplicate rows, duplicate IDs, and detected IQR outliers.
    """
    if df is None or df.empty:
        return {"error": "Empty dataframe"}

    total_rows = len(df)
    missing_by_col = {}
    dtypes_by_col = {}
    uniques_by_col = {}

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = round(float(null_count / total_rows * 100.0), 2)
        missing_by_col[col] = {"count": null_count, "percentage": null_pct}
        dtypes_by_col[col] = str(df[col].dtype)
        uniques_by_col[col] = int(df[col].nunique(dropna=True))

    exact_duplicates = int(df.duplicated().sum())
    
    dup_ids = 0
    if "problem_id" in df.columns:
        dup_ids = int(df["problem_id"].astype(str).str.strip().duplicated().sum())

    # Outlier detection on core numerical columns
    num_cols = ["estimated_solution_days", "team_size", "innovation_score", "impact_score"]
    outliers_report = {}
    for col in num_cols:
        if col in df.columns:
            outliers_report[col] = detect_outliers_iqr(df, col)

    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "total_missing_cells": int(df.isna().sum().sum()),
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_problem_ids": dup_ids,
        "missing_by_column": missing_by_col,
        "dtypes_by_column": dtypes_by_col,
        "uniques_by_column": uniques_by_col,
        "outliers": outliers_report
    }


def clean_hackathon_dataset(df_raw: pd.DataFrame, save_to_disk: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the complete, reproducible data cleaning pipeline on the raw dataset:
    1. Removes exact duplicate records.
    2. Strips leading and trailing whitespaces across all text columns.
    3. Normalizes category casing and whitespace for 'domain', 'difficulty_level', 'organizer_type'.
    4. Safely parses and converts boolean columns ('ai_required', 'hardware_required') to 0/1 integers.
    5. Imputes missing numerical values with column medians.
    6. Imputes missing categorical values with column modes.
    7. Resolves duplicate primary keys ('problem_id') safely without deleting valid records.
    8. Caps extreme unrealistic outliers (e.g. team_size > 10 capped to median, estimated_solution_days > 100 capped).
    9. Recomputes derived features ('problem_length_words', 'number_of_technologies').
    10. Saves the cleaned dataset to data/hackathon_problem_statements_cleaned.csv.
    Returns: (cleaned_df, before_and_after_audit_dict)
    """
    if df_raw is None or df_raw.empty:
        raise ValueError("Cannot clean an empty or None DataFrame.")

    # Record baseline state
    before_audit = compute_quality_audit(df_raw)
    df = df_raw.copy()
    operations_log: List[str] = []

    # 1. Exact Duplicate Records Removal
    dups_count = int(df.duplicated().sum())
    if dups_count > 0:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        operations_log.append(f"Removed {dups_count} exact duplicate rows.")
    else:
        operations_log.append("No exact duplicate rows found.")

    # 2. String Whitespace Standardization
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)
    operations_log.append(f"Standardized leading/trailing whitespace across {len(str_cols)} text columns.")

    # 3. Category Normalization (domain, organizer_type, difficulty_level)
    if "domain" in df.columns:
        def normalize_domain(val):
            if pd.isna(val):
                return val
            s = str(val).strip().lower()
            return DOMAIN_NORMALIZATION_MAP.get(s, str(val).strip().title())
        df["domain"] = df["domain"].apply(normalize_domain)
        operations_log.append("Normalized domain casing to canonical title format.")

    for cat_col in ["organizer_type", "difficulty_level", "implementation_feasibility", "data_availability"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].apply(lambda x: str(x).strip().title() if pd.notna(x) else x)

    # 4. Boolean Type Normalization
    for bool_col in ["ai_required", "hardware_required"]:
        if bool_col in df.columns:
            def parse_bool(v):
                if pd.isna(v):
                    return 0
                if v in BOOLEAN_TRUE_SET or str(v).strip().lower() in {"true", "yes", "1", "t"}:
                    return 1
                return 0
            df[bool_col] = df[bool_col].apply(parse_bool).astype(int)
            operations_log.append(f"Converted column '{bool_col}' to standard binary integers (0 and 1).")

    # 5. Missing Value Imputation
    # Numerical: median imputation
    num_impute_cols = ["estimated_solution_days", "innovation_score", "impact_score", "team_size"]
    for col in num_impute_cols:
        if col in df.columns and df[col].isna().sum() > 0:
            median_val = round(float(df[col].dropna().median()), 1)
            null_cnt = int(df[col].isna().sum())
            df[col] = df[col].fillna(median_val)
            operations_log.append(f"Imputed {null_cnt} missing values in '{col}' using column median ({median_val}).")

    # Categorical: mode imputation
    cat_impute_cols = ["data_availability", "sub_domain", "technology_tags"]
    for col in cat_impute_cols:
        if col in df.columns and df[col].isna().sum() > 0:
            null_cnt = int(df[col].isna().sum())
            mode_val = df[col].dropna().mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "General"
            df[col] = df[col].fillna(fill_val)
            operations_log.append(f"Imputed {null_cnt} missing values in '{col}' using mode/fallback ('{fill_val}').")

    # 6. Resolve Duplicate Problem IDs Safely
    if "problem_id" in df.columns:
        seen_ids = set()
        new_ids = []
        fixed_id_count = 0
        for idx, current_id in enumerate(df["problem_id"]):
            cleaned_id = str(current_id).strip()
            if cleaned_id in seen_ids or cleaned_id == "nan" or not cleaned_id:
                # Generate unique synthetic key preserving sequential prefix
                new_id = f"PS-{idx + 1:03d}"
                seen_ids.add(new_id)
                new_ids.append(new_id)
                fixed_id_count += 1
            else:
                seen_ids.add(cleaned_id)
                new_ids.append(cleaned_id)
        df["problem_id"] = new_ids
        if fixed_id_count > 0:
            operations_log.append(f"Resolved {fixed_id_count} duplicate/empty problem identifiers with deterministic unique keys.")

    # 7. Outlier Treatment (Conservative domain-aware capping)
    # Team size capping: max realistic team is 6-8, cap 99 outlier to median (4)
    if "team_size" in df.columns:
        abnormal_teams = (df["team_size"] > 10) | (df["team_size"] < 1)
        if abnormal_teams.sum() > 0:
            median_team = int(df["team_size"][~abnormal_teams].median())
            df.loc[abnormal_teams, "team_size"] = median_team
            operations_log.append(f"Treated {abnormal_teams.sum()} extreme team_size outliers by setting them to median ({median_team}).")

    # Estimated solution days capping: cap extreme values > 100 days to 95th percentile
    if "estimated_solution_days" in df.columns:
        p95 = round(float(df["estimated_solution_days"].quantile(0.95)), 1)
        extreme_days = df["estimated_solution_days"] > 100.0
        if extreme_days.sum() > 0:
            df.loc[extreme_days, "estimated_solution_days"] = p95
            operations_log.append(f"Capped {extreme_days.sum()} extreme solution_days outliers (>100) to 95th percentile ({p95} days).")

    # 8. Recompute Derived Features
    if "problem_description" in df.columns:
        df["problem_length_words"] = df["problem_description"].astype(str).str.split().str.len()
        operations_log.append("Recomputed 'problem_length_words' derived feature.")

    if "technology_tags" in df.columns:
        def count_tags(val):
            if pd.isna(val) or not str(val).strip():
                return 0
            return len([t.strip() for t in str(val).split(",") if t.strip()])
        df["number_of_technologies"] = df["technology_tags"].apply(count_tags)
        operations_log.append("Recomputed 'number_of_technologies' derived feature.")

    # 9. Compute Final Cleaned Audit
    after_audit = compute_quality_audit(df)

    # 10. Persist Cleaned Dataset
    if save_to_disk:
        cleaned_path = get_cleaned_data_path()
        df.to_csv(cleaned_path, index=False)
        operations_log.append(f"Saved cleaned dataset to '{cleaned_path}'.")

    comparison = {
        "original_rows": before_audit["total_rows"],
        "cleaned_rows": after_audit["total_rows"],
        "rows_removed": before_audit["total_rows"] - after_audit["total_rows"],
        "original_columns": before_audit["total_columns"],
        "cleaned_columns": after_audit["total_columns"],
        "missing_before": before_audit["total_missing_cells"],
        "missing_after": after_audit["total_missing_cells"],
        "duplicates_before": before_audit["exact_duplicate_rows"],
        "duplicates_after": after_audit["exact_duplicate_rows"],
        "dup_ids_before": before_audit["duplicate_problem_ids"],
        "dup_ids_after": after_audit["duplicate_problem_ids"],
        "operations_performed": operations_log,
        "before_audit": before_audit,
        "after_audit": after_audit
    }

    return df, comparison
