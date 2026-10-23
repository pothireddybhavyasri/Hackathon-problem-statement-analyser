"""
src/eda.py
Exploratory Data Analysis module for computing domain distributions, technology tag
frequency counts, variable classifications, and textual properties.
"""

from typing import Dict, List, Any
from collections import Counter
import pandas as pd

VARIABLE_TAXONOMY = {
    "problem_id": {"type": "Identifier", "nature": "Categorical/String", "role": "Primary Key"},
    "hackathon_name": {"type": "Nominal", "nature": "Categorical", "role": "Grouping/Filter"},
    "year": {"type": "Discrete/Temporal", "nature": "Numerical/Interval", "role": "Time Series Index"},
    "organizer_type": {"type": "Nominal", "nature": "Categorical", "role": "Demographic Filter"},
    "problem_title": {"type": "Text", "nature": "Unstructured Natural Language", "role": "Title Field"},
    "problem_description": {"type": "Text", "nature": "Unstructured Natural Language", "role": "Vectorization / NLP"},
    "domain": {"type": "Nominal", "nature": "Categorical", "role": "Primary Dimension"},
    "sub_domain": {"type": "Nominal", "nature": "Categorical", "role": "Sub-dimension"},
    "industry": {"type": "Nominal", "nature": "Categorical", "role": "Adopter Sector"},
    "technology_tags": {"type": "Multi-label Text", "nature": "Categorical Set", "role": "Skill Mapping"},
    "ai_required": {"type": "Binary", "nature": "Boolean / Discrete (0, 1)", "role": "Classification Target"},
    "hardware_required": {"type": "Binary", "nature": "Boolean / Discrete (0, 1)", "role": "Feature / Filter"},
    "data_availability": {"type": "Ordinal/Nominal", "nature": "Categorical", "role": "Feasibility Factor"},
    "difficulty_level": {"type": "Ordinal", "nature": "Categorical (Beginner < Int < Adv)", "role": "Complexity Level"},
    "estimated_solution_days": {"type": "Continuous Ratio", "nature": "Numerical (Days)", "role": "Regression Target"},
    "team_size": {"type": "Discrete", "nature": "Numerical (Count of people)", "role": "Predictor Feature"},
    "innovation_score": {"type": "Continuous Interval", "nature": "Numerical (1.0 to 10.0)", "role": "Evaluation Rubric"},
    "impact_score": {"type": "Continuous Interval", "nature": "Numerical (1.0 to 10.0)", "role": "Evaluation Rubric"},
    "implementation_feasibility": {"type": "Ordinal", "nature": "Categorical (Low < Med < High)", "role": "Heuristic Class"},
    "problem_length_words": {"type": "Discrete", "nature": "Numerical (Count of words)", "role": "Derived Text Metric"},
    "number_of_technologies": {"type": "Discrete", "nature": "Numerical (Count of tools)", "role": "Derived Stack Metric"},
    "source_url": {"type": "Metadata / URL", "nature": "String", "role": "Provenance Audit"},
    "data_type": {"type": "Binary Nominal", "nature": "Categorical (sourced, synthetic)", "role": "Integrity Cohort"}
}


def classify_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns an academic taxonomy table classifying every variable in the dataset by its
    statistical data type, scale of measurement, and analytical role.
    """
    records = []
    for col in df.columns:
        tax = VARIABLE_TAXONOMY.get(col, {"type": "Unknown", "nature": str(df[col].dtype), "role": "General"})
        records.append({
            "Column Name": col,
            "Pandas Dtype": str(df[col].dtype),
            "Statistical Scale": tax["type"],
            "Mathematical Nature": tax["nature"],
            "Analytical Role": tax["role"]
        })
    return pd.DataFrame(records)


def extract_technology_frequencies(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """
    Tokenizes and extracts all individual technologies from the comma-separated 'technology_tags' column,
    calculating frequency of occurrence and overall share of problem statements.
    """
    if df is None or "technology_tags" not in df.columns:
        return pd.DataFrame()

    tag_counter = Counter()
    total_records = len(df)

    for tags_str in df["technology_tags"].dropna():
        tags = [t.strip() for t in str(tags_str).split(",") if t.strip()]
        tag_counter.update(tags)

    top_tags = tag_counter.most_common(top_n)
    records = []
    for tech, count in top_tags:
        records.append({
            "Technology": tech,
            "Count": count,
            "Percentage (%)": round(count / total_records * 100.0, 2)
        })

    return pd.DataFrame(records)


def compute_domain_subdomain_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes hierarchical counts and percentages by domain and sub-domain.
    """
    if df is None or "domain" not in df.columns:
        return pd.DataFrame()

    group_cols = ["domain"]
    if "sub_domain" in df.columns:
        group_cols.append("sub_domain")

    grouped = df.groupby(group_cols).size().reset_index(name="Problem Count")
    grouped["Percentage (%)"] = (grouped["Problem Count"] / len(df) * 100.0).round(2)
    return grouped.sort_values(by="Problem Count", ascending=False).reset_index(drop=True)


def compute_hardware_ai_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes contingency counts for AI required vs Hardware required problem statements.
    """
    if df is None or "ai_required" not in df.columns or "hardware_required" not in df.columns:
        return {}

    total = len(df)
    ai_count = int((df["ai_required"] == 1).sum())
    hw_count = int((df["hardware_required"] == 1).sum())
    both_count = int(((df["ai_required"] == 1) & (df["hardware_required"] == 1)).sum())
    pure_sw_count = int(((df["ai_required"] == 0) & (df["hardware_required"] == 0)).sum())

    return {
        "total_problems": total,
        "ai_required_count": ai_count,
        "ai_required_pct": round(ai_count / total * 100.0, 1),
        "hardware_required_count": hw_count,
        "hardware_required_pct": round(hw_count / total * 100.0, 1),
        "both_ai_and_hardware_count": both_count,
        "both_ai_and_hardware_pct": round(both_count / total * 100.0, 1),
        "pure_software_count": pure_sw_count,
        "pure_software_pct": round(pure_sw_count / total * 100.0, 1)
    }
