"""
pages/2_🧹_Data_Quality_and_Cleaning.py
Comprehensive Data Quality Audit and Interactive Cleaning Laboratory.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_raw_data, load_cleaned_data, check_dataset_exists
from src.data_cleaning import clean_hackathon_dataset, compute_quality_audit, detect_outliers_iqr

st.set_page_config(page_title="Data Quality & Cleaning — Hackathon Insights", page_icon="🧹", layout="wide")

st.markdown("# 🧹 Data Quality Audit & Interactive Cleaning Lab")
st.markdown(
    "Detect noise, missing data, schema anomalies, and statistical outliers using the Interquartile Range (IQR) method. "
    "Execute reproducible cleaning operations and inspect before-and-after audit logs."
)

raw_df = load_raw_data()
clean_df = load_cleaned_data() if check_dataset_exists(is_cleaned=True) else None

tab_audit, tab_iqr, tab_cleaning_exec = st.tabs([
    "📊 Data Quality Audit", "📐 IQR Outlier Sandbox", "⚙️ Reusable Cleaning Pipeline"
])

# ------------------------------------------------------------------------------
# TAB 1: DATA QUALITY AUDIT
# ------------------------------------------------------------------------------
with tab_audit:
    st.markdown("### 🔍 Raw Dataset Quality Inspection")
    q_audit = compute_quality_audit(raw_df)

    # Missing values plot
    miss_items = [
        {"Column": col, "Missing Count": info["count"], "Percentage": info["percentage"]}
        for col, info in q_audit["missing_by_column"].items() if info["count"] > 0
    ]
    if miss_items:
        miss_df = pd.DataFrame(miss_items).sort_values(by="Missing Count", ascending=True)
        fig_miss = px.bar(
            miss_df,
            x="Missing Count",
            y="Column",
            orientation="h",
            text="Percentage",
            title="<b>Missing Values by Column in Raw Dataset</b>",
            color="Missing Count",
            color_continuous_scale="Reds"
        )
        fig_miss.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_miss.update_layout(height=350)
        st.plotly_chart(fig_miss, use_container_width=True)
    else:
        st.success("No missing values found.")

    c_dup, c_id = st.columns(2)
    with c_dup:
        st.markdown(f"#### 📑 Exact Duplicate Rows: `{q_audit['exact_duplicate_rows']}`")
        if q_audit["exact_duplicate_rows"] > 0:
            dup_rows = raw_df[raw_df.duplicated(keep=False)]
            st.dataframe(dup_rows[["problem_id", "problem_title", "domain", "hackathon_name"]].head(6), use_container_width=True)
        else:
            st.info("No exact duplicate rows.")

    with c_id:
        st.markdown(f"#### 🔑 Duplicate Problem IDs: `{q_audit['duplicate_problem_ids']}`")
        if q_audit["duplicate_problem_ids"] > 0:
            stripped_ids = raw_df["problem_id"].astype(str).str.strip()
            dup_id_mask = stripped_ids.duplicated(keep=False)
            st.dataframe(raw_df[dup_id_mask][["problem_id", "problem_title", "domain"]].head(6), use_container_width=True)
        else:
            st.info("Zero duplicate problem IDs.")


# ------------------------------------------------------------------------------
# TAB 2: IQR OUTLIER SANDBOX
# ------------------------------------------------------------------------------
with tab_iqr:
    st.markdown("### 📐 Interquartile Range (IQR) Outlier Detection Sandbox")
    st.markdown("""
    Under the Tukey standard for boxplot outlier detection:
    - **First Quartile ($Q_1$)**: 25th percentile of the sorted distribution.
    - **Third Quartile ($Q_3$)**: 75th percentile of the sorted distribution.
    - **Interquartile Range**: $\\text{IQR} = Q_3 - Q_1$.
    - **Fences**: $\\text{Lower Bound} = Q_1 - (k \\times \\text{IQR})$, $\\text{Upper Bound} = Q_3 + (k \\times \\text{IQR})$, where typically $k = 1.5$.
    """)

    iqr_c1, iqr_c2 = st.columns([0.4, 0.6])
    with iqr_c1:
        target_num_col = st.selectbox(
            "Select Numerical Attribute for Outlier Inspection:",
            options=["estimated_solution_days", "team_size", "innovation_score", "impact_score", "problem_length_words"]
        )
        multiplier_k = st.slider("IQR Multiplier (k):", min_value=1.0, max_value=3.0, value=1.5, step=0.5)
        
        out_res = detect_outliers_iqr(raw_df, target_num_col, multiplier=multiplier_k)
        
        st.markdown(f"""
        - **$Q_1$ (25%)**: `{out_res['q1']}`
        - **$Q_3$ (75%)**: `{out_res['q3']}`
        - **IQR**: `{out_res['iqr']}`
        - **Lower Bound**: `{out_res['lower_bound']}`
        - **Upper Bound**: `{out_res['upper_bound']}`
        - **Detected Outliers**: `{out_res['outlier_count']}` ({out_res['outlier_percentage']}%)
        """)
        if out_res["outlier_values"]:
            st.markdown(f"**Sample Outlier Values**: `{out_res['outlier_values'][:8]}`")

    with iqr_c2:
        fig_box = px.box(
            raw_df,
            y=target_num_col,
            points="all",
            title=f"<b>Boxplot & Distribution for '{target_num_col}'</b>",
            color_discrete_sequence=["#2563EB"]
        )
        fig_box.add_hline(y=out_res["upper_bound"], line_dash="dash", line_color="red", annotation_text=f"Upper Fence: {out_res['upper_bound']}")
        fig_box.add_hline(y=out_res["lower_bound"], line_dash="dash", line_color="red", annotation_text=f"Lower Fence: {out_res['lower_bound']}")
        fig_box.update_layout(height=420)
        st.plotly_chart(fig_box, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 3: CLEANING PIPELINE EXECUTION
# ------------------------------------------------------------------------------
with tab_cleaning_exec:
    st.markdown("### ⚙️ Automated Cleaning Execution & Before-and-After Metrics")
    
    if st.button("🚀 Execute Data Cleaning Now", key="clean_page_btn", type="primary"):
        c_df, audit = clean_hackathon_dataset(raw_df, save_to_disk=True)
        st.session_state.cleaned_df = c_df
        st.session_state.cleaning_audit = audit
        st.success("Cleaning pipeline executed and new baseline saved!")

    if "cleaning_audit" in st.session_state and st.session_state.cleaning_audit is not None:
        aud = st.session_state.cleaning_audit
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", f"{aud['original_rows']} → {aud['cleaned_rows']}")
        k2.metric("Missing Cells", f"{aud['missing_before']} → {aud['missing_after']}")
        k3.metric("Duplicate Rows", f"{aud['duplicates_before']} → {aud['duplicates_after']}")
        k4.metric("Duplicate IDs", f"{aud['dup_ids_before']} → {aud['dup_ids_after']}")

        st.markdown("#### Operations Applied:")
        for op in aud["operations_performed"]:
            st.markdown(f"- ✅ {op}")
