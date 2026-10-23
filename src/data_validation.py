"""
src/data_validation.py
Module for validating hackathon problem statement datasets against schema, range,
uniqueness, type, and categorical consistency constraints.
"""

from typing import Dict, List, Any
import re
import pandas as pd

EXPECTED_COLUMNS = [
    "problem_id",
    "hackathon_name",
    "year",
    "organizer_type",
    "problem_title",
    "problem_description",
    "domain",
    "sub_domain",
    "industry",
    "technology_tags",
    "ai_required",
    "hardware_required",
    "data_availability",
    "difficulty_level",
    "estimated_solution_days",
    "team_size",
    "innovation_score",
    "impact_score",
    "implementation_feasibility",
    "problem_length_words",
    "number_of_technologies",
    "source_url",
    "data_type",
]

EXPECTED_DOMAINS = {
    "Healthcare", "FinTech", "Agriculture", "Smart Cities", "Cybersecurity",
    "Clean Energy", "Disaster Management", "Smart Education", "Transportation", "Space Tech"
}

EXPECTED_ORGANIZERS = {"Government", "Academic", "Corporate", "Open Source / NGO"}
EXPECTED_DIFFICULTIES = {"Beginner", "Intermediate", "Advanced"}
EXPECTED_FEASIBILITIES = {"High", "Medium", "Low"}
EXPECTED_DATA_AVAILABILITY = {"Open Data", "Restricted", "Simulated", "None"}


def validate_dataset_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes a comprehensive data validation audit on the provided DataFrame.
    Returns a dictionary of check items with status: 'PASS', 'WARNING', or 'FAIL',
    along with diagnostic metrics and error details.
    """
    checks: List[Dict[str, Any]] = []
    
    if df is None or df.empty:
        return {
            "summary": {"pass": 0, "warning": 0, "fail": 1, "total": 1},
            "checks": [{
                "name": "Dataset Existence",
                "status": "FAIL",
                "message": "DataFrame is empty or None.",
                "details": "No data available to validate."
            }]
        }

    # 1. Required Columns Check
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if not missing_cols:
        checks.append({
            "name": "Required Columns",
            "status": "PASS",
            "message": f"All {len(EXPECTED_COLUMNS)} expected columns are present.",
            "details": f"Verified {len(EXPECTED_COLUMNS)} columns."
        })
    else:
        checks.append({
            "name": "Required Columns",
            "status": "FAIL",
            "message": f"Missing {len(missing_cols)} required columns: {', '.join(missing_cols)}",
            "details": f"Missing columns: {missing_cols}"
        })

    # 2. Duplicate Rows Check
    exact_dups = int(df.duplicated().sum())
    if exact_dups == 0:
        checks.append({
            "name": "Exact Duplicate Rows",
            "status": "PASS",
            "message": "Zero exact duplicate rows detected.",
            "details": "0 exact duplicate rows."
        })
    else:
        checks.append({
            "name": "Exact Duplicate Rows",
            "status": "WARNING",
            "message": f"Found {exact_dups} exact duplicate rows that require deduplication.",
            "details": f"{exact_dups} duplicate records identified."
        })

    # 3. Unique Identifiers Check
    if "problem_id" in df.columns:
        stripped_ids = df["problem_id"].astype(str).str.strip()
        id_dup_count = int(stripped_ids.duplicated().sum())
        null_ids = int(df["problem_id"].isna().sum())
        
        if id_dup_count == 0 and null_ids == 0:
            checks.append({
                "name": "Problem ID Uniqueness",
                "status": "PASS",
                "message": f"All {len(df)} problem identifiers are unique and non-null.",
                "details": "100% unique primary keys."
            })
        else:
            checks.append({
                "name": "Problem ID Uniqueness",
                "status": "WARNING" if null_ids == 0 else "FAIL",
                "message": f"Found {id_dup_count} duplicate IDs and {null_ids} missing IDs.",
                "details": f"Duplicates: {id_dup_count}, Nulls: {null_ids}"
            })

    # 4. Year Range Check
    if "year" in df.columns:
        valid_years = pd.to_numeric(df["year"], errors="coerce")
        out_of_range_years = ((valid_years < 2018) | (valid_years > 2026)).sum()
        null_years = valid_years.isna().sum()
        if out_of_range_years == 0 and null_years == 0:
            checks.append({
                "name": "Year Range Validity",
                "status": "PASS",
                "message": f"Years range from {int(valid_years.min())} to {int(valid_years.max())} within 2018-2026.",
                "details": "All years valid."
            })
        else:
            checks.append({
                "name": "Year Range Validity",
                "status": "WARNING",
                "message": f"Found {out_of_range_years} out-of-range years and {null_years} null years.",
                "details": f"Invalid years: {out_of_range_years}"
            })

    # 5. Numerical Range Check: estimated_solution_days
    if "estimated_solution_days" in df.columns:
        days = pd.to_numeric(df["estimated_solution_days"], errors="coerce")
        null_days = int(days.isna().sum())
        outlier_days = int(((days < 1.0) | (days > 100.0)).sum())
        if outlier_days == 0 and null_days == 0:
            checks.append({
                "name": "Estimated Solution Days Range",
                "status": "PASS",
                "message": f"All values are between 1.0 and 100.0 days (Min: {days.min():.1f}, Max: {days.max():.1f}).",
                "details": "Ranges conform to standard student hackathon project durations."
            })
        else:
            checks.append({
                "name": "Estimated Solution Days Range",
                "status": "WARNING",
                "message": f"Detected {outlier_days} unusual outlier values (>100 or <1) and {null_days} missing values.",
                "details": f"Max observed: {days.max():.1f} days, Nulls: {null_days}"
            })

    # 6. Team Size Check
    if "team_size" in df.columns:
        teams = pd.to_numeric(df["team_size"], errors="coerce")
        abnormal_teams = int(((teams < 1) | (teams > 10)).sum())
        if abnormal_teams == 0:
            checks.append({
                "name": "Team Size Validity",
                "status": "PASS",
                "message": f"Team sizes range from {int(teams.min())} to {int(teams.max())} participants.",
                "details": "Standard team constraints verified."
            })
        else:
            checks.append({
                "name": "Team Size Validity",
                "status": "WARNING",
                "message": f"Found {abnormal_teams} records with abnormal team sizes (>10 or <1).",
                "details": f"Abnormal records: {abnormal_teams}"
            })

    # 7. Boolean Fields Consistency: ai_required, hardware_required
    for bool_col in ["ai_required", "hardware_required"]:
        if bool_col in df.columns:
            non_std_vals = df[~df[bool_col].isin([0, 1, True, False, "0", "1"])][bool_col].unique()
            if len(non_std_vals) == 0:
                checks.append({
                    "name": f"Boolean Consistency ({bool_col})",
                    "status": "PASS",
                    "message": f"Column '{bool_col}' contains standardized binary values.",
                    "details": "All values map directly to binary flags."
                })
            else:
                checks.append({
                    "name": f"Boolean Consistency ({bool_col})",
                    "status": "WARNING",
                    "message": f"Column '{bool_col}' has non-standard values: {list(non_std_vals)[:5]}",
                    "details": f"Needs normalization to 0/1 integers."
                })

    # 8. Category Consistency: Domain
    if "domain" in df.columns:
        current_domains = set(df["domain"].dropna().astype(str).str.strip().unique())
        unrecognized_domains = [d for d in current_domains if d not in EXPECTED_DOMAINS]
        if not unrecognized_domains:
            checks.append({
                "name": "Domain Standardization",
                "status": "PASS",
                "message": f"All {len(current_domains)} unique domains conform to standard taxonomy.",
                "details": "Taxonomy standard adhered to."
            })
        else:
            checks.append({
                "name": "Domain Standardization",
                "status": "WARNING",
                "message": f"Found casing variations or unrecognized domains: {unrecognized_domains[:5]}",
                "details": "Requires capitalization normalization."
            })

    # 9. Text Description Quality Check
    if "problem_description" in df.columns:
        desc_lens = df["problem_description"].fillna("").astype(str).str.split().str.len()
        short_descs = int((desc_lens < 5).sum())
        if short_descs == 0:
            checks.append({
                "name": "Problem Description Completeness",
                "status": "PASS",
                "message": f"All problem statements have substantive descriptions (Mean words: {desc_lens.mean():.1f}).",
                "details": "Zero truncated or placeholder descriptions."
            })
        else:
            checks.append({
                "name": "Problem Description Completeness",
                "status": "WARNING",
                "message": f"Found {short_descs} problem statements with empty or trivial descriptions (<5 words).",
                "details": f"Short text entries: {short_descs}"
            })

    # 10. URL Format Check
    if "source_url" in df.columns:
        urls = df["source_url"].dropna().astype(str)
        url_pattern = re.compile(r"^https?://", re.IGNORECASE)
        invalid_urls = int((~urls.str.match(url_pattern)).sum())
        if invalid_urls == 0:
            checks.append({
                "name": "Source URL Format",
                "status": "PASS",
                "message": "All source URLs follow standard HTTP/HTTPS protocol formatting.",
                "details": f"Validated {len(urls)} URLs."
            })
        else:
            checks.append({
                "name": "Source URL Format",
                "status": "WARNING",
                "message": f"Found {invalid_urls} URLs with invalid syntax.",
                "details": f"Malformed URLs: {invalid_urls}"
            })

    pass_count = sum(1 for c in checks if c["status"] == "PASS")
    warn_count = sum(1 for c in checks if c["status"] == "WARNING")
    fail_count = sum(1 for c in checks if c["status"] == "FAIL")

    return {
        "summary": {
            "pass": pass_count,
            "warning": warn_count,
            "fail": fail_count,
            "total": len(checks)
        },
        "checks": checks
    }
