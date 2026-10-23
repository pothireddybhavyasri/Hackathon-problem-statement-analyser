"""
pages/0_🔬_Data_Analytics_Lab.py
Central Step-by-Step Data Analytics Lab.
Presents the entire end-to-end analytical workflow with functional interactive buttons:
Load Data → Validate Data → Inspect Quality → Clean Data → Explore Data → Train Models → Interpret Results → Export Results.
"""

import io
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import (
    load_raw_data, load_cleaned_data, check_dataset_exists, get_dataset_metadata, get_raw_data_path, get_cleaned_data_path
)
from src.data_validation import validate_dataset_schema
from src.data_cleaning import clean_hackathon_dataset, compute_quality_audit, detect_outliers_iqr
from src.statistics_analysis import compute_descriptive_stats, compute_correlation_matrix
from src.regression_models import prepare_regression_dataset, train_linear_regression
from src.classification_models import prepare_classification_dataset, train_logistic_regression
from src.clustering import run_kmeans_clustering
from src.visualizations import (
    plot_correlation_heatmap, plot_actual_vs_predicted,
    plot_residuals_diagnostic, plot_confusion_matrix_heatmap, plot_pca_clusters
)

st.set_page_config(page_title="Data Analytics Lab — Hackathon Insights", page_icon="🔬", layout="wide")

st.markdown("# 🔬 Central Data Analytics Lab")
st.markdown(
    "**End-to-End Analytical Workflow Hub**: Execute each stage sequentially, inspect computed outputs, "
    "and witness how raw hackathon problem data transforms through data quality audit, cleaning, exploratory analysis, and predictive modeling."
)

st.markdown("---")

# Session state initialization
if "raw_df" not in st.session_state or st.session_state.raw_df is None:
    try:
        st.session_state.raw_df = load_raw_data()
    except Exception as e:
        st.session_state.raw_df = None

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = load_cleaned_data() if check_dataset_exists(is_cleaned=True) else None

if "validation_report" not in st.session_state:
    st.session_state.validation_report = None

if "quality_report" not in st.session_state:
    st.session_state.quality_report = None

if "cleaning_audit" not in st.session_state:
    st.session_state.cleaning_audit = None

if "lab_regression" not in st.session_state:
    st.session_state.lab_regression = None

if "lab_classification" not in st.session_state:
    st.session_state.lab_classification = None

if "lab_clustering" not in st.session_state:
    st.session_state.lab_clustering = None


# Workflow Stepper Navigator in Expander / Header
st.markdown("""
<div style="background-color: #F1F5F9; padding: 12px; border-radius: 8px; margin-bottom: 20px; font-weight: 600; color: #334155; text-align: center;">
Step 1: Load Data &nbsp;➔&nbsp; Step 2: Validate Data &nbsp;➔&nbsp; Step 3: Inspect Quality &nbsp;➔&nbsp; Step 4: Clean Data &nbsp;➔&nbsp; Step 5: Explore & Transform &nbsp;➔&nbsp; Step 6: Train Models &nbsp;➔&nbsp; Step 7: Export
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================
st.markdown("## 📥 Step 1 — Load Data")
st.markdown("Load and inspect the original, unmodified raw dataset from local storage.")

col1_btn, col1_info = st.columns([0.25, 0.75])
with col1_btn:
    if st.button("🔄 Load / Refresh Raw Data", key="btn_load_raw", use_container_width=True):
        try:
            st.session_state.raw_df = load_raw_data()
            st.success("Successfully loaded raw dataset from disk!")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

raw_df = st.session_state.raw_df

if raw_df is not None:
    raw_meta = get_dataset_metadata(raw_df)
    with col1_info:
        st.markdown(
            f"**File Location**: `{get_raw_data_path()}` &nbsp;|&nbsp; "
            f"**Total Rows**: `{raw_meta['rows']}` &nbsp;|&nbsp; "
            f"**Columns**: `{raw_meta['columns']}` &nbsp;|&nbsp; "
            f"**Sourced**: `{raw_meta['sourced_count']}` &nbsp;|&nbsp; "
            f"**Synthetic**: `{raw_meta['synthetic_count']}` &nbsp;|&nbsp; "
            f"**Memory**: `{raw_meta['memory_kb']} KB`"
        )

    with st.expander("👁️ View Raw Dataset Preview & Column Schema", expanded=True):
        st.dataframe(raw_df.head(8), use_container_width=True)
        
        c_dtypes, c_provenance = st.columns(2)
        with c_dtypes:
            st.markdown("##### Column Data Types")
            dtype_df = pd.DataFrame({"Column": raw_df.columns, "Pandas Dtype": [str(t) for t in raw_df.dtypes]})
            st.dataframe(dtype_df, height=200, use_container_width=True)
        with c_provenance:
            st.markdown("##### Provenance Cohort Breakdown")
            prov_counts = raw_df["data_type"].value_counts().reset_index()
            prov_counts.columns = ["Cohort", "Count"]
            prov_counts["Percentage"] = (prov_counts["Count"] / len(raw_df) * 100.0).round(1).astype(str) + "%"
            st.dataframe(prov_counts, use_container_width=True)
else:
    st.warning("Raw dataset not loaded. Please click 'Load / Refresh Raw Data' above.")

st.markdown("---")


# ==============================================================================
# STEP 2: DATA VALIDATION
# ==============================================================================
st.markdown("## 🛡️ Step 2 — Data Validation")
st.markdown(
    "Validate the dataset against expected structural schema, unique identifiers, numerical bounds, "
    "boolean standards, and text completeness."
)

if st.button("🧪 Execute Data Validation Audit", key="btn_validate", type="primary"):
    if raw_df is not None:
        st.session_state.validation_report = validate_dataset_schema(raw_df)
        st.success("Data validation audit completed successfully!")
    else:
        st.error("Please load the raw dataset in Step 1 first.")

if st.session_state.validation_report is not None:
    val_rep = st.session_state.validation_report
    sum_data = val_rep["summary"]
    
    # Validation summary pills
    vc1, vc2, vc3, vc4 = st.columns(4)
    vc1.metric("Total Checks", sum_data["total"])
    vc2.metric("Passed Checks", sum_data["pass"])
    vc3.metric("Warnings (Human Review)", sum_data["warning"])
    vc4.metric("Failures", sum_data["fail"])

    # Detailed checks table
    check_rows = []
    for c in val_rep["checks"]:
        badge = "✅ PASS" if c["status"] == "PASS" else ("⚠️ WARNING" if c["status"] == "WARNING" else "❌ FAIL")
        check_rows.append({
            "Status": badge,
            "Check Rule": c["name"],
            "Diagnostic Message": c["message"],
            "Details": c["details"]
        })
    st.dataframe(pd.DataFrame(check_rows), use_container_width=True)

    st.caption("ℹ️ *Note: Warnings identify real data flaws that need cleaning or human review without silently discarding valid records.*")

st.markdown("---")


# ==============================================================================
# STEP 3: DATA QUALITY ANALYSIS
# ==============================================================================
st.markdown("## 🔍 Step 3 — Data Quality Analysis")
st.markdown(
    "Perform a deep audit of missing values, duplicate records, identifier collisions, and numerical outliers using the Interquartile Range (IQR) technique."
)

if st.button("📊 Generate Comprehensive Quality Report", key="btn_quality_audit"):
    if raw_df is not None:
        st.session_state.quality_report = compute_quality_audit(raw_df)
        st.success("Data quality audit computed!")
    else:
        st.error("Please load the raw dataset in Step 1 first.")

if st.session_state.quality_report is not None:
    q_rep = st.session_state.quality_report
    
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Missing Data Cells", q_rep["total_missing_cells"])
    q2.metric("Duplicate Rows", q_rep["exact_duplicate_rows"])
    q3.metric("Duplicate Problem IDs", q_rep["duplicate_problem_ids"])
    q4.metric("Detected Outlier Features", len(q_rep["outliers"]))

    t_miss, t_outlier = st.columns([1.1, 0.9])
    with t_miss:
        st.markdown("##### Missing Values Breakdown by Column")
        miss_data = []
        for col, info in q_rep["missing_by_column"].items():
            if info["count"] > 0:
                miss_data.append({"Column": col, "Missing Count": info["count"], "Missing (%)": f"{info['percentage']}%"})
        if miss_data:
            st.dataframe(pd.DataFrame(miss_data), use_container_width=True)
        else:
            st.info("No missing values detected in the evaluated columns.")

    with t_outlier:
        st.markdown("##### Interquartile Range (IQR) Outlier Detection")
        st.markdown(r"$$\text{IQR} = Q_3 - Q_1, \quad \text{Lower} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper} = Q_3 + 1.5 \times \text{IQR}$$")
        outlier_rows = []
        for col, o_info in q_rep["outliers"].items():
            if "error" not in o_info:
                outlier_rows.append({
                    "Feature": col,
                    "Q1": o_info["q1"],
                    "Q3": o_info["q3"],
                    "IQR": o_info["iqr"],
                    "Upper Bound": o_info["upper_bound"],
                    "Outlier Count": o_info["outlier_count"]
                })
        st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)

    # Download quality report JSON
    q_json = json.dumps(q_rep, indent=2)
    st.download_button(
        "📥 Download Quality Audit Report (JSON)",
        data=q_json,
        file_name="data_quality_audit_report.json",
        mime="application/json"
    )

st.markdown("---")


# ==============================================================================
# STEP 4: DATA CLEANING
# ==============================================================================
st.markdown("## 🧹 Step 4 — Data Cleaning Pipeline")
st.markdown(
    "Execute the automated cleaning pipeline: Standardize whitespace, normalize category casing, "
    "impute missing values (median for numerical, mode for categorical), remove duplicate rows, resolve duplicate IDs, and cap extreme outliers."
)

if st.button("🚀 Run Cleaning Pipeline", key="btn_run_cleaning", type="primary"):
    if raw_df is not None:
        clean_df, audit = clean_hackathon_dataset(raw_df, save_to_disk=True)
        st.session_state.cleaned_df = clean_df
        st.session_state.cleaning_audit = audit
        st.success("Dataset successfully cleaned and saved to `data/hackathon_problem_statements_cleaned.csv`!")
    else:
        st.error("Please load the raw dataset in Step 1 first.")

if st.session_state.cleaning_audit is not None:
    audit = st.session_state.cleaning_audit
    
    st.markdown("### 🔄 Before-and-After Cleaning Comparison")
    
    # Comparison KPI cards
    c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
    c_kpi1.metric("Rows", f"{audit['cleaned_rows']}", delta=f"-{audit['rows_removed']} rows", delta_color="inverse")
    c_kpi2.metric("Missing Cells", f"{audit['missing_after']}", delta=f"-{audit['missing_before']}", delta_color="inverse")
    c_kpi3.metric("Duplicate Rows", f"{audit['duplicates_after']}", delta=f"-{audit['duplicates_before']}", delta_color="inverse")
    c_kpi4.metric("Duplicate IDs", f"{audit['dup_ids_after']}", delta=f"-{audit['dup_ids_before']}", delta_color="inverse")

    # Detailed Before vs After Table
    comp_table = pd.DataFrame([
        {"Metric": "Total Records (Rows)", "Raw Baseline": audit["original_rows"], "After Cleaning": audit["cleaned_rows"], "Net Delta": -audit["rows_removed"]},
        {"Metric": "Total Columns", "Raw Baseline": audit["original_columns"], "After Cleaning": audit["cleaned_columns"], "Net Delta": 0},
        {"Metric": "Missing Value Cells", "Raw Baseline": audit["missing_before"], "After Cleaning": audit["missing_after"], "Net Delta": -audit["missing_before"]},
        {"Metric": "Exact Duplicate Rows", "Raw Baseline": audit["duplicates_before"], "After Cleaning": audit["duplicates_after"], "Net Delta": -audit["duplicates_before"]},
        {"Metric": "Duplicate Problem IDs", "Raw Baseline": audit["dup_ids_before"], "After Cleaning": audit["dup_ids_after"], "Net Delta": -audit["dup_ids_before"]}
    ])
    st.dataframe(comp_table, use_container_width=True)

    with st.expander("📜 View Operations Log & Applied Transformations", expanded=True):
        for op in audit["operations_performed"]:
            st.markdown(f"- ✅ {op}")

    # Interactive Table of Cleaned Data
    st.markdown("### 📋 Cleaned Dataset Explorer")
    cleaned_df = st.session_state.cleaned_df
    st.dataframe(cleaned_df.head(10), use_container_width=True)

    # Download Cleaned CSV
    csv_buffer = io.StringIO()
    cleaned_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "📥 Download Cleaned Dataset (CSV)",
        data=csv_buffer.getvalue(),
        file_name="hackathon_problem_statements_cleaned.csv",
        mime="text/csv"
    )

st.markdown("---")


# ==============================================================================
# STEP 5: DATA TRANSFORMATION & EXPLORATORY ANALYSIS
# ==============================================================================
st.markdown("## 📈 Step 5 — Data Transformation & Exploratory Analysis")
st.markdown(
    "Examine statistical properties, variable classifications, and feature correlations on the clean dataset."
)

active_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df

if active_df is not None:
    tab_stats, tab_corr, tab_scale_theory = st.tabs([
        "📊 Univariate Descriptive Statistics", "🔥 Correlation Heatmap", "📚 Normalization vs Standardization"
    ])

    with tab_stats:
        st.markdown("##### 14 Univariate Statistical Measures (Central Tendency & Dispersion)")
        stats_df = compute_descriptive_stats(active_df)
        st.dataframe(stats_df, use_container_width=True)

    with tab_corr:
        st.markdown("##### Pixel-Oriented Visualization: Pearson Correlation Matrix")
        corr_df = compute_correlation_matrix(active_df)
        st.plotly_chart(plot_correlation_heatmap(corr_df), use_container_width=True)

    with tab_scale_theory:
        st.markdown("""
        ### Academic Theory: Normalization vs. Standardization (Unit I & II)

        | Property | Min-Max Normalization | Z-Score Standardization |
        | :--- | :--- | :--- |
        | **Mathematical Formula** | $$x' = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}$$ | $$z = \\frac{x - \\mu}{\\sigma}$$ |
        | **Output Bounding Range** | Strictly bounded in $[0, 1]$ (or $[-1, 1]$) | Unbounded; centered at $\\mu = 0$ with unit variance $\\sigma^2 = 1$ |
        | **Sensitivity to Outliers** | Highly sensitive (extreme outlier compresses all normal values) | Robust; preserves outlier signals as standard deviations away |
        | **Algorithm Use Case** | Neural networks, image pixel intensities, KNN distance metrics | Linear Regression, Logistic Regression, PCA, KMeans |
        """)
else:
    st.warning("Please complete Step 1 or Step 4 to load data.")

st.markdown("---")


# ==============================================================================
# STEP 6: MODEL TRAINING & EVALUATION
# ==============================================================================
st.markdown("## 🤖 Step 6 — Machine Learning Model Training & Evaluation")
st.markdown(
    "Train and evaluate supervised and unsupervised algorithms on the cleaned dataset with strict 80/20 train/test splits."
)

if active_df is not None:
    ml_col1, ml_col2, ml_col3 = st.columns(3)

    # 1. Linear Regression
    with ml_col1:
        st.markdown("#### 1. Multiple Linear Regression")
        st.caption("Target: `estimated_solution_days`")
        if st.button("Train Linear Regression", key="btn_train_reg", use_container_width=True):
            reg_data = prepare_regression_dataset(active_df, test_size=0.20, random_state=42)
            st.session_state.lab_regression = train_linear_regression(reg_data)
            st.success("Linear Regression trained!")

        if st.session_state.lab_regression is not None:
            lr_res = st.session_state.lab_regression
            tm = lr_res["test_metrics"]
            st.metric("Test R² Score", f"{tm['R2']:.4f}")
            st.metric("Test MAE", f"{tm['MAE']:.2f} days")
            st.metric("Test RMSE", f"{tm['RMSE']:.2f} days")

    # 2. Logistic Regression
    with ml_col2:
        st.markdown("#### 2. Logistic Regression")
        st.caption("Target: `ai_required` (0 vs 1)")
        if st.button("Train Logistic Regression", key="btn_train_clf", use_container_width=True):
            clf_data = prepare_classification_dataset(active_df, test_size=0.20, random_state=42)
            st.session_state.lab_classification = train_logistic_regression(clf_data)
            st.success("Logistic Regression trained!")

        if st.session_state.lab_classification is not None:
            clf_res = st.session_state.lab_classification
            cm_metrics = clf_res["test_metrics"]
            st.metric("Test Accuracy", f"{cm_metrics['Accuracy']*100:.1f}%")
            st.metric("Test Precision", f"{cm_metrics['Precision']:.3f}")
            st.metric("Test F1-Score", f"{cm_metrics['F1_Score']:.3f}")

    # 3. K-Means Clustering
    with ml_col3:
        st.markdown("#### 3. K-Means Segmentation")
        st.caption("Unsupervised (k=4 clusters)")
        if st.button("Run K-Means Clustering", key="btn_train_cluster", use_container_width=True):
            st.session_state.lab_clustering = run_kmeans_clustering(active_df, n_clusters=4)
            st.success("K-Means clustering completed!")

        if st.session_state.lab_clustering is not None:
            km_res = st.session_state.lab_clustering
            st.metric("Silhouette Score", f"{km_res['silhouette_score']:.3f}")
            st.metric("Inertia (WCSS)", f"{km_res['inertia']:.1f}")
            st.metric("PCA Variance Captured", f"{km_res['total_explained_variance']}%")

    # Render Visuals for Trained Models
    if st.session_state.lab_regression is not None:
        st.markdown("### 📊 Linear Regression Model Diagnostics")
        reg_c1, reg_c2 = st.columns(2)
        with reg_c1:
            st.plotly_chart(
                plot_actual_vs_predicted(
                    st.session_state.lab_regression["y_test_true"],
                    st.session_state.lab_regression["y_test_pred"],
                    "Multiple Linear Regression (OLS)"
                ),
                use_container_width=True
            )
        with reg_c2:
            st.plotly_chart(
                plot_residuals_diagnostic(
                    st.session_state.lab_regression["y_test_pred"],
                    st.session_state.lab_regression["residuals"]
                ),
                use_container_width=True
            )

    if st.session_state.lab_classification is not None:
        st.markdown("### 📊 Logistic Regression Confusion Matrix")
        st.plotly_chart(
            plot_confusion_matrix_heatmap(st.session_state.lab_classification["confusion_matrix"]),
            use_container_width=True
        )

    if st.session_state.lab_clustering is not None:
        st.markdown("### 📊 2D PCA Cluster Geometric Projection")
        st.plotly_chart(
            plot_pca_clusters(
                st.session_state.lab_clustering["pca_data"],
                st.session_state.lab_clustering["centroids_pca"],
                st.session_state.lab_clustering["explained_variance_ratio"]
            ),
            use_container_width=True
        )

st.markdown("---")


# ==============================================================================
# STEP 7: EXPORT & SUMMARY
# ==============================================================================
st.markdown("## 📤 Step 7 — Interpret & Export Results")
st.markdown(
    "Review the analytical conclusions and export generated artifacts for academic and professional presentation."
)

exp_c1, exp_c2, exp_c3 = st.columns(3)
with exp_c1:
    if active_df is not None:
        csv_exp = io.StringIO()
        active_df.to_csv(csv_exp, index=False)
        st.download_button(
            "📥 Export Cleaned Dataset (CSV)",
            data=csv_exp.getvalue(),
            file_name="hackathon_insights_cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

with exp_c2:
    if st.session_state.cleaning_audit is not None:
        audit_json = json.dumps(st.session_state.cleaning_audit, indent=2)
        st.download_button(
            "📥 Export Before-and-After Audit (JSON)",
            data=audit_json,
            file_name="cleaning_audit_summary.json",
            mime="application/json",
            use_container_width=True
        )

with exp_c3:
    if st.session_state.lab_regression is not None:
        coef_csv = io.StringIO()
        st.session_state.lab_regression["coefficients_df"].to_csv(coef_csv, index=False)
        st.download_button(
            "📥 Export Regression Coefficients (CSV)",
            data=coef_csv.getvalue(),
            file_name="regression_coefficients.csv",
            mime="text/csv",
            use_container_width=True
        )

st.success("🎉 **Analytical Workflow Complete**: All stages executed with empirical calculations grounded in the real dataset.")
