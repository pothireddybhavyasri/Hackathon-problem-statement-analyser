"""
src/statistics_analysis.py
Module for descriptive statistics, cross-tabulations, correlation matrices,
and statistical hypothesis testing (ANOVA, Chi-Square, t-test).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

NUMERICAL_COLUMNS = [
    "estimated_solution_days",
    "team_size",
    "innovation_score",
    "impact_score",
    "problem_length_words",
    "number_of_technologies",
    "year"
]


def compute_descriptive_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Computes rigorous univariate descriptive statistics for numerical columns:
    Mean, Median, Mode, Min, Max, Range, Variance, Std Dev, Q1 (25%), Q2 (50%), Q3 (75%), IQR, Skewness.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    target_cols = [c for c in (columns or NUMERICAL_COLUMNS) if c in df.columns]
    records = []

    for col in target_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue

        mean_val = float(series.mean())
        median_val = float(series.median())
        mode_series = series.mode()
        mode_val = float(mode_series.iloc[0]) if not mode_series.empty else np.nan
        min_val = float(series.min())
        max_val = float(series.max())
        val_range = max_val - min_val
        variance_val = float(series.var())
        std_val = float(series.std())
        q1 = float(series.quantile(0.25))
        q2 = float(series.quantile(0.50))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        skew_val = float(series.skew())
        kurt_val = float(series.kurt())

        records.append({
            "Variable": col,
            "Count": int(len(series)),
            "Mean": round(mean_val, 2),
            "Median": round(median_val, 2),
            "Mode": round(mode_val, 2) if not np.isnan(mode_val) else "N/A",
            "Min": round(min_val, 2),
            "Max": round(max_val, 2),
            "Range": round(val_range, 2),
            "Variance": round(variance_val, 2),
            "Std Dev": round(std_val, 2),
            "Q1 (25%)": round(q1, 2),
            "Q2 (50%)": round(q2, 2),
            "Q3 (75%)": round(q3, 2),
            "IQR": round(iqr, 2),
            "Skewness": round(skew_val, 2),
            "Kurtosis": round(kurt_val, 2),
        })

    return pd.DataFrame(records).set_index("Variable")


def compute_categorical_frequencies(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Computes count, percentage, and cumulative percentage for a categorical column.
    """
    if df is None or column not in df.columns:
        return pd.DataFrame()

    counts = df[column].value_counts(dropna=False)
    pcts = (counts / len(df) * 100.0).round(2)
    cum_pcts = pcts.cumsum().round(2)

    result = pd.DataFrame({
        "Category": counts.index.astype(str),
        "Frequency": counts.values,
        "Percentage (%)": pcts.values,
        "Cumulative (%)": cum_pcts.values
    })
    return result


def compute_crosstab_analysis(df: pd.DataFrame, row_col: str, col_col: str) -> Dict[str, pd.DataFrame]:
    """
    Computes raw frequency and percentage cross-tabulations between two categorical attributes.
    """
    if df is None or row_col not in df.columns or col_col not in df.columns:
        return {"counts": pd.DataFrame(), "percentages": pd.DataFrame()}

    counts = pd.crosstab(df[row_col], df[col_col], margins=True, margins_name="Total")
    percentages = (pd.crosstab(df[row_col], df[col_col], normalize="index") * 100.0).round(1)

    return {
        "counts": counts,
        "percentages": percentages
    }


def compute_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Computes Pearson or Spearman correlation matrix across numerical features.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    available_cols = [c for c in NUMERICAL_COLUMNS if c in df.columns]
    num_df = df[available_cols].apply(pd.to_numeric, errors="coerce")
    return num_df.corr(method=method).round(3)


def perform_hypothesis_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Conducts formal statistical hypothesis inference:
    1. One-Way ANOVA: Does estimated_solution_days vary significantly across domain categories?
    2. Chi-Square Test of Independence: Are domain and ai_required statistically independent?
    3. Independent Samples Two-Tailed t-test: Do AI-required problems take significantly more solution days?
    """
    results = {}
    if df is None or df.empty:
        return results

    # 1. ANOVA: Days vs Domain
    if "estimated_solution_days" in df.columns and "domain" in df.columns:
        domain_groups = [
            group["estimated_solution_days"].dropna().values
            for _, group in df.groupby("domain")
            if len(group["estimated_solution_days"].dropna()) > 2
        ]
        if len(domain_groups) > 1:
            f_stat, p_val = stats.f_oneway(*domain_groups)
            results["anova_domain_days"] = {
                "test_name": "One-Way Analysis of Variance (ANOVA)",
                "dependent_var": "estimated_solution_days",
                "independent_factor": "domain",
                "f_statistic": round(float(f_stat), 3),
                "p_value": float(p_val),
                "p_value_formatted": f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}",
                "significant_at_05": bool(p_val < 0.05),
                "interpretation": (
                    "Statistically significant difference in required solution days across domains (p < 0.05)."
                    if p_val < 0.05 else
                    "No statistically significant difference in required solution days across domains (p >= 0.05)."
                )
            }

    # 2. Chi-Square: Domain vs AI Required
    if "domain" in df.columns and "ai_required" in df.columns:
        contingency = pd.crosstab(df["domain"], df["ai_required"])
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
            results["chi2_domain_ai"] = {
                "test_name": "Pearson Chi-Square Test of Independence",
                "var1": "domain",
                "var2": "ai_required",
                "chi2_statistic": round(float(chi2), 3),
                "degrees_of_freedom": int(dof),
                "p_value": float(p_val),
                "p_value_formatted": f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}",
                "significant_at_05": bool(p_val < 0.05),
                "interpretation": (
                    "AI requirement is statistically dependent on the problem domain (p < 0.05)."
                    if p_val < 0.05 else
                    "AI requirement appears independent of problem domain (p >= 0.05)."
                )
            }

    # 3. Two-Sample T-Test: Days for AI=1 vs AI=0
    if "estimated_solution_days" in df.columns and "ai_required" in df.columns:
        ai_days = df[df["ai_required"] == 1]["estimated_solution_days"].dropna().values
        non_ai_days = df[df["ai_required"] == 0]["estimated_solution_days"].dropna().values
        if len(ai_days) > 5 and len(non_ai_days) > 5:
            t_stat, p_val = stats.ttest_ind(ai_days, non_ai_days, equal_var=False)
            results["ttest_ai_days"] = {
                "test_name": "Welch's Two-Sample t-Test",
                "group_1": "AI Required (1)",
                "group_2": "AI Not Required (0)",
                "mean_group_1": round(float(np.mean(ai_days)), 2),
                "mean_group_2": round(float(np.mean(non_ai_days)), 2),
                "t_statistic": round(float(t_stat), 3),
                "p_value": float(p_val),
                "p_value_formatted": f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}",
                "significant_at_05": bool(p_val < 0.05),
                "interpretation": (
                    f"AI-required problems average {np.mean(ai_days):.1f} days vs {np.mean(non_ai_days):.1f} days for non-AI problems (significant at p < 0.05)."
                    if p_val < 0.05 else
                    "Difference in solution days between AI and non-AI problems is not statistically significant."
                )
            }

    return results
