"""
pages/8_📖_Methodology_and_SQL.py
SQLite Relational Database Studio, SQL Demonstrations, and Mathematical Methodology.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd

from src.database import (
    get_db_path, init_database, run_custom_query, get_sql_demonstrations
)
from src.data_loader import load_cleaned_data, load_raw_data

st.set_page_config(page_title="Methodology & SQL — Hackathon Insights", page_icon="📖", layout="wide")

st.markdown("# 📖 Relational SQL Studio & Methodology")
st.markdown(
    "Explore the SQLite relational database implementation, execute custom SQL queries, "
    "and review the mathematical and algorithmic methodology supporting the platform."
)

tab_sql, tab_math = st.tabs([
    "🗄️ Relational SQLite Studio", "📐 Mathematical Methodology"
])

# ------------------------------------------------------------------------------
# TAB 1: SQL STUDIO
# ------------------------------------------------------------------------------
with tab_sql:
    st.markdown("### 🗄️ SQLite Database Architecture")
    st.caption(f"Database File: `{get_db_path()}`")

    c_init, c_info = st.columns([0.3, 0.7])
    with c_init:
        if st.button("🔄 Re-initialize SQLite Database", use_container_width=True):
            clean_df = load_cleaned_data()
            res = init_database(clean_df)
            st.success(f"Initialized database with {res['problem_statements_count']} problem statements!")

    # Check table counts
    df_ps, _ = run_custom_query("SELECT COUNT(*) AS total FROM problem_statements;")
    df_tech, _ = run_custom_query("SELECT COUNT(*) AS total FROM problem_technologies;")
    df_dim, _ = run_custom_query("SELECT COUNT(*) AS total FROM domains_dim;")

    ps_cnt = df_ps.iloc[0]["total"] if df_ps is not None else 0
    tech_cnt = df_tech.iloc[0]["total"] if df_tech is not None else 0
    dim_cnt = df_dim.iloc[0]["total"] if df_dim is not None else 0

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("`problem_statements` Rows", ps_cnt)
    sc2.metric("`problem_technologies` Rows", tech_cnt)
    sc3.metric("`domains_dim` Rows", dim_cnt)

    st.markdown("---")
    st.markdown("### 💡 Curated SQL Concept Demonstrations")
    sql_demos = get_sql_demonstrations()
    
    demo_choice = st.selectbox(
        "Select SQL Concept to Inspect & Run:",
        options=list(sql_demos.keys())
    )
    
    selected_demo = sql_demos[demo_choice]
    st.markdown(f"**Demonstrated Concept**: *{selected_demo['concept']}*")
    
    # Query text area
    query_text = st.text_area(
        "SQL Query:",
        value=selected_demo["sql"].strip(),
        height=140
    )

    if st.button("▶️ Execute Query", key="btn_exec_sql", type="primary"):
        res_df, err = run_custom_query(query_text)
        if err:
            st.error(f"SQL Error: {err}")
        elif res_df is not None:
            st.success(f"Query returned {len(res_df)} rows.")
            st.dataframe(res_df, use_container_width=True)
        else:
            st.info("Query returned 0 rows.")


# ------------------------------------------------------------------------------
# TAB 2: MATHEMATICAL METHODOLOGY
# ------------------------------------------------------------------------------
with tab_math:
    st.markdown("### 📐 Formal Mathematical & Algorithmic Methodology")
    
    st.markdown("""
    #### 1. Ordinary Least Squares (OLS) Multiple Regression
    $$\\hat{\\beta} = (X^T X)^{-1} X^T y$$
    - Minimizes the Residual Sum of Squares: $RSS = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 = (y - X\\beta)^T (y - X\\beta)$
    - **Coefficient of Determination ($R^2$)**:
      $$R^2 = 1 - \\frac{SS_{\\text{res}}}{SS_{\\text{tot}}} = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2}$$
      *Note on Negative $R^2$*: On unseen test data, $R^2$ can legitimately be negative if the model predictions perform worse than the horizontal line of the test set mean (i.e., $SS_{\\text{res}} > SS_{\\text{tot}}$). This indicates model overfitting on training data or structural domain shifts.

    #### 2. Logistic Regression (Maximum Likelihood Estimation)
    $$P(Y = 1 | X) = \\sigma(z) = \\frac{1}{1 + e^{-z}} = \\frac{1}{1 + e^{-(\\beta_0 + \\sum \\beta_j x_j)}}$$
    - Log-Odds (Logit): $\\log\\left(\\frac{P}{1 - P}\\right) = X\\beta$
    - Maximizes the Log-Likelihood:
      $$\\ell(\\beta) = \\sum_{i=1}^n \\left[ y_i \\log \\sigma(x_i^T \\beta) + (1 - y_i) \\log(1 - \\sigma(x_i^T \\beta)) \\right]$$

    #### 3. Decision Tree Splitting Criteria
    - **Gini Impurity**:
      $$I_G(p) = 1 - \\sum_{i=1}^C p_i^2$$
    - **Entropy / Information Gain**:
      $$H(X) = -\\sum_{i=1}^C p_i \\log_2 p_i, \\quad IG(T, a) = H(T) - H(T | a)$$

    #### 4. K-Means Clustering & Silhouette Coefficient
    - Objective: Minimize Within-Cluster Sum of Squares (Inertia):
      $$J = \\sum_{k=1}^K \\sum_{x_i \\in C_k} \\|x_i - \\mu_k\\|^2$$
    - **Silhouette Coefficient**:
      $$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$$
      Where $a(i)$ is mean intra-cluster distance, and $b(i)$ is mean nearest-cluster distance. Ranges from $-1$ (incorrect cluster) to $+1$ (dense, well-separated cluster).
    """)
