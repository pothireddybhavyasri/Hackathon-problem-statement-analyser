"""
app.py
Hackathon Insights — Problem Statement Analysis Platform.
Main Streamlit application entry point and Executive Overview Dashboard.
"""

import os
import sys

# Ensure repository root is in sys.path for cloud deployment
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import (
    load_raw_data, load_cleaned_data, check_dataset_exists, get_dataset_metadata
)
from src.data_cleaning import clean_hackathon_dataset
from src.visualizations import (
    plot_domain_distribution_bar, plot_technology_frequency,
    plot_scatter_bivariate, COLOR_PRIMARY, COLOR_ACCENT
)
from src.eda import extract_technology_frequencies, compute_hardware_ai_summary
from src.database import init_database, get_db_path

# Auto-initialize SQLite database if not present
if not os.path.exists(get_db_path()):
    try:
        init_database()
    except Exception:
        pass

# Page Configuration
st.set_page_config(
    page_title="Hackathon Insights — Problem Statement Analysis",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for polished presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "raw_df" not in st.session_state:
    try:
        st.session_state.raw_df = load_raw_data()
    except Exception as e:
        st.session_state.raw_df = None

if "cleaned_df" not in st.session_state:
    if check_dataset_exists(is_cleaned=True):
        st.session_state.cleaned_df = load_cleaned_data()
    elif st.session_state.raw_df is not None:
        df_clean, audit = clean_hackathon_dataset(st.session_state.raw_df, save_to_disk=True)
        st.session_state.cleaned_df = df_clean
        st.session_state.cleaning_audit = audit
    else:
        st.session_state.cleaned_df = None

# Sidebar Navigation and Branding
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.markdown("## **Hackathon Insights**")
    st.markdown("**Problem Statement Analysis Platform**")
    st.markdown("---")
    st.markdown("### 🎓 Academic Curriculum")
    st.caption("B.Tech III-I CSE (R24) • Course: CS513PE Data Analytics")
    st.markdown("---")
    st.markdown("### ⚡ Quick Navigation")
    st.markdown("""
    - [🔬 Data Analytics Lab](/Data_Analytics_Lab)
    - [📊 Data Explorer](/Data_Explorer)
    - [🧹 Data Quality & Cleaning](/Data_Quality_and_Cleaning)
    - [📈 Exploratory Analysis](/Exploratory_Analysis)
    - [🤖 Machine Learning Lab](/Machine_Learning_Lab)
    - [🔍 Text Similarity Search](/Text_Similarity_and_Discovery)
    - [📅 Trends & Forecasting](/Trends_and_Forecasting)
    - [🎓 Syllabus Mapping](/Syllabus_Mapping)
    - [📖 Methodology & SQL](/Methodology_and_SQL)
    """)
    st.markdown("---")
    if st.button("🔄 Reload Raw & Clean Data", use_container_width=True):
        st.session_state.raw_df = load_raw_data()
        st.session_state.cleaned_df = load_cleaned_data()
        st.success("Datasets reloaded successfully!")


# Main Dashboard Header
st.markdown('<div class="main-title">💡 Hackathon Problem Statement Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A complete, presentation-ready Data Analytics platform demonstrating Data Management, Cleaning, EDA, Machine Learning, Text Similarity, and Time-Series Forecasting.</div>',
    unsafe_allow_html=True
)

df_active = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df

if df_active is None:
    st.error("No dataset found. Please generate the dataset via `python scripts/generate_dataset.py`.")
    st.stop()

meta = get_dataset_metadata(df_active)
hw_ai = compute_hardware_ai_summary(df_active)

# Executive KPI Scorecard
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{meta['rows']}</div>
        <div class="metric-lbl">Total Problems</div>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{meta['sourced_count']}</div>
        <div class="metric-lbl">Verified Sourced</div>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{meta['synthetic_count']}</div>
        <div class="metric-lbl">Synthetic Cohort</div>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{hw_ai.get('ai_required_pct', 0)}%</div>
        <div class="metric-lbl">AI / ML Required</div>
    </div>
    """, unsafe_allow_html=True)
with kpi5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{hw_ai.get('hardware_required_pct', 0)}%</div>
        <div class="metric-lbl">Hardware / IoT</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Callout banner to Data Analytics Lab
st.info(
    "🔬 **Primary Workflow Hub**: Navigate directly to the **[Data Analytics Lab](/Data_Analytics_Lab)** "
    "to execute the step-by-step pipeline: **Load Data → Validate Data → Inspect Quality → Clean Data → Explore Data → Train Models → Interpret Results → Export Results**."
)

# Overview Tabs
tab_summary, tab_charts, tab_demo_guide = st.tabs([
    "📋 Executive Overview", "📊 Core Visual Insights", "🎯 Faculty Demonstration Guide"
])

with tab_summary:
    col_left, col_right = st.columns([1.2, 0.8])
    with col_left:
        st.markdown("### 🎯 Problem Statement & Analytical Objectives")
        st.markdown("""
        Hackathons have emerged as the primary vehicle for open innovation across government agencies, academia, and industry.
        However, organizers and student teams frequently grapple with several recurring challenges:
        
        - **Ambiguous Requirements**: How do technical requirements vary across different domains (Healthcare, FinTech, Agriculture)?
        - **Estimation Uncertainty**: What factors drive estimated project implementation duration?
        - **Skill Alignment**: What are the most demanded technologies across hackathon challenges?
        - **Duplication & Reusability**: Can we identify duplicate or semantically related problem statements across hackathons?
        
        This platform delivers an end-to-end analytical framework to solve these questions using rigorous techniques mapped directly to the **CS513PE Data Analytics** curriculum.
        """)
        
        st.markdown("### 🏛️ Dataset Integrity & Provenance")
        st.markdown(f"""
        - **Dataset Size**: **{meta['rows']} unique problem statements** ({meta['columns']} attributes).
        - **Real Sourced Cohort**: **{meta['sourced_count']} records (25.0%)** extracted from legitimate public hackathons:
          *Smart India Hackathon (SIH)*, *NASA Space Apps Challenge*, *MIT Grand Hacks*, *HackMIT*, *ETHIndia*, and *AI for Good Global Summit*.
        - **Synthetic Cohort**: **{meta['synthetic_count']} records (75.0%)** generated via reproducible parametric templates (`seed=42`).
        - **Data Quality Flaws**: Raw dataset deliberately injects missing values, duplicate rows, duplicate IDs, casing variations, and outliers to demonstrate data cleaning.
        """)

    with col_right:
        st.markdown("### 🏗️ Analytical Architecture")
        st.markdown("""
        ```
        Raw Hackathon Data (.csv)
                 │
        [Step 1] Data Loader & Schema
                 │
        [Step 2] Data Validation (10 Checks)
                 │
        [Step 3] Data Quality & IQR Outliers
                 │
        [Step 4] Reusable Cleaning Pipeline
                 │
        [Step 5] Descriptive Stats & Cross-Tabs
                 │
        [Step 6] ML Models (Reg, Clf, Clustering)
                 │
        [Step 7] Text Similarity & Forecasts
                 │
        Relational SQLite & SQL Engine
        ```
        """)
        
        st.markdown("### 📁 Primary Deliverables")
        st.markdown("""
        - 📄 [Data Dictionary](docs/data_dictionary.md)
        - 🔍 [Data Sources & Provenance](docs/data_sources.md)
        - 🎓 [Verified Syllabus Mapping](docs/syllabus_mapping.md)
        - 💡 [Viva Q&A Preparation](docs/viva_questions.md)
        - 📑 [Full Project Report](docs/project_report.md)
        """)

with tab_charts:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.plotly_chart(plot_domain_distribution_bar(df_active), use_container_width=True)
    with col_c2:
        tech_df = extract_technology_frequencies(df_active, top_n=12)
        st.plotly_chart(plot_technology_frequency(tech_df, top_n=12), use_container_width=True)

    st.plotly_chart(
        plot_scatter_bivariate(
            df_active,
            x_col="estimated_solution_days",
            y_col="innovation_score",
            color_col="domain",
            size_col="team_size",
            title="Geometric Projection: Estimated Solution Days vs Innovation Score (by Domain & Team Size)"
        ),
        use_container_width=True
    )

with tab_demo_guide:
    st.markdown("### 🎯 11-Step Faculty Demonstration Sequence")
    st.markdown("""
    Follow this exact sequence when presenting the project to faculty or viva examiners:

    1. **Overview & Problem Statement**: Start on this page. Point out the {0} total records, the {1} verified sourced records from SIH, NASA SpaceApps, and MIT, and explain the project motivation.
    2. **Data Analytics Lab (Step 1 - Load Data)**: Switch to `Data Analytics Lab`. Click **Load Raw Dataset** to show raw data preview, file metadata, and column types.
    3. **Data Validation (Step 2)**: Click **Run Data Validation Audit**. Point out the pass/fail indicators and explain how warnings require human review.
    4. **Data Quality Audit (Step 3)**: Inspect the missing value breakdown, duplicate IDs, and IQR outlier detection on `estimated_solution_days` and `team_size`.
    5. **Data Cleaning (Step 4)**: Click **Run Data Cleaning**. Review the Before-and-After comparison table showing missing values dropping to 0, duplicate rows removed, and review the operations log.
    6. **Exploratory Analysis (Step 5)**: Show the 14 univariate descriptive statistics (Mean, Median, Mode, Variance, IQR, Skewness) and the correlation heatmap.
    7. **Linear Regression (Step 6)**: In the ML Lab, train Multiple Linear Regression predicting `estimated_solution_days`. Point out R², MAE, RMSE, and the Gauss-Markov BLUE residual diagnostics.
    8. **Logistic Regression & Decision Trees**: Train Logistic Regression on `ai_required`. Show Confusion Matrix and precision/recall. Show Decision Tree depth pruning curve demonstrating overfitting.
    9. **Clustering & Text Similarity**: Demonstrate K-Means clustering with 2D PCA projection and search for similar problems using TF-IDF and Cosine Similarity.
    10. **Trends & Forecasting**: Show historical yearly problem volume and the ARIMA(1,1,0) forecast for 2025-2026 with 95% confidence intervals.
    11. **Syllabus Mapping & SQL**: Demonstrate the verified mapping of all 5 units from `CS513PE: DATA ANALYTICS` and run custom SQLite queries.
    """.format(meta['rows'], meta['sourced_count']))
