"""
pages/7_🎓_Syllabus_Mapping.py
Academic Syllabus Verification and Interactive Demonstration Navigator.
Maps the project features directly to the 5 Units of CS513PE: DATA ANALYTICS.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Syllabus Mapping — Hackathon Insights", page_icon="🎓", layout="wide")

st.markdown("# 🎓 Academic Syllabus Mapping & Demonstration Navigator")
st.markdown(
    "**Course Code & Title:** `CS513PE: DATA ANALYTICS (Professional Elective – I)`  \n"
    "**Curriculum Regulation:** B.Tech. III Year I Sem (R24 Regulations)  \n"
    "**Verified Document Source:** `CSE_R24 B.Tech III-I Syllabus 2026-2027 (1).pdf` (Pages 12–13)"
)

st.info(
    "✅ **Verified Implementation**: Every unit, topic, algorithm, and visualization method in the academic syllabus corresponds "
    "to executable Python source code in `src/` and interactive Streamlit components in this application."
)

st.markdown("---")

# Syllabus Units Data
SYLLABUS_RECORDS = [
    # Unit I
    {
        "Unit": "Unit I",
        "Unit Title": "Data Management & Quality",
        "Exact Syllabus Topic": "Data Management & Architecture (manage data for analysis, sources like Sensors/Web/GPS)",
        "Implemented Feature": "Dual data architecture (Raw vs Cleaned separation); CSV & SQLite persistence; Sourced vs Synthetic cohorts.",
        "Source File & Function": "`src/data_loader.py` (`load_raw_data`, `load_cleaned_data`), `src/database.py`",
        "Application Location": "Pages: *Data Analytics Lab*, *Data Explorer*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit I",
        "Unit Title": "Data Management & Quality",
        "Exact Syllabus Topic": "Data Quality (noise, outliers, missing values, duplicate data)",
        "Implemented Feature": "Automated data quality audit; missing value percentage calculator; duplicate row & duplicate ID detection; IQR outlier detection with upper/lower bounds.",
        "Source File & Function": "`src/data_validation.py`, `src/data_cleaning.py` (`detect_outliers_iqr`, `compute_quality_audit`)",
        "Application Location": "Pages: *Step 2 & Step 3 in Lab*, *Data Quality & Cleaning*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit I",
        "Unit Title": "Data Management & Quality",
        "Exact Syllabus Topic": "Data Processing & Preprocessing",
        "Implemented Feature": "Automated multi-stage cleaning pipeline: whitespace stripping, category casing normalization, missing imputation, duplicate pruning, and derived feature recalculation.",
        "Source File & Function": "`src/data_cleaning.py` (`clean_hackathon_dataset`)",
        "Application Location": "Pages: *Step 4 (Run Cleaning) in Lab*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    # Unit II
    {
        "Unit": "Unit II",
        "Unit Title": "Data Analytics & Modeling",
        "Exact Syllabus Topic": "Introduction to Analytics Tools and Environment",
        "Implemented Feature": "Interactive Streamlit platform, Pandas, NumPy, Scikit-learn, Statsmodels, SQLite relational backend.",
        "Source File & Function": "`app.py`, `src/database.py`, `requirements.txt`",
        "Application Location": "All Application Pages & Virtual Environment",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit II",
        "Unit Title": "Data Analytics & Modeling",
        "Exact Syllabus Topic": "Databases & Types of Data and Variables (Numerical, Categorical, SQL)",
        "Implemented Feature": "Variable classification taxonomy (continuous, discrete, nominal, ordinal, binary, text); SQLite database integration with parameterized SELECT, WHERE, GROUP BY, HAVING, ORDER BY, JOIN.",
        "Source File & Function": "`src/database.py`, `src/eda.py` (`classify_variables`)",
        "Application Location": "Pages: *Exploratory Analysis*, *Methodology & SQL*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit II",
        "Unit Title": "Data Analytics & Modeling",
        "Exact Syllabus Topic": "Missing Value Imputation",
        "Implemented Feature": "Documented imputation strategies: median for skewed numerical features (`estimated_solution_days`, `innovation_score`), mode for categorical fields (`data_availability`, `sub_domain`).",
        "Source File & Function": "`src/data_cleaning.py` (`clean_hackathon_dataset`)",
        "Application Location": "Pages: *Data Analytics Lab (Step 4)*, *Data Quality & Cleaning*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit II",
        "Unit Title": "Data Analytics & Modeling",
        "Exact Syllabus Topic": "Descriptive Analysis & Business Modeling",
        "Implemented Feature": "Univariate statistical profiles: Mean, Median, Mode, Min, Max, Range, Variance, Std Dev, Q1, Q2, Q3, IQR, Skewness, Kurtosis, and Cross-tabulations.",
        "Source File & Function": "`src/statistics_analysis.py` (`compute_descriptive_stats`, `compute_crosstab_analysis`)",
        "Application Location": "Pages: *Data Analytics Lab (Step 5)*, *Exploratory Analysis*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    # Unit III
    {
        "Unit": "Unit III",
        "Unit Title": "Regression & Classification",
        "Exact Syllabus Topic": "Regression Concepts, BLUE Property Assumptions, Least Squares Estimation",
        "Implemented Feature": "Multiple Linear Regression (OLS) predicting `estimated_solution_days`; verification of Gauss-Markov BLUE assumptions (Linearity, Homoscedasticity, Normality of residuals).",
        "Source File & Function": "`src/regression_models.py` (`train_linear_regression`)",
        "Application Location": "Pages: *Machine Learning Lab (Tab 1)*, *Lab (Step 6)*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit III",
        "Unit Title": "Regression & Classification",
        "Exact Syllabus Topic": "Variable Rationalization & Model Building (R², MAE, RMSE)",
        "Implemented Feature": "Feature selection, train/test split (80/20) without leakage, coefficient interpretability table, actual vs predicted plot, residual diagnostics.",
        "Source File & Function": "`src/regression_models.py` (`prepare_regression_dataset`, `compute_regression_metrics`)",
        "Application Location": "Pages: *Machine Learning Lab (Tab 1)*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit III",
        "Unit Title": "Regression & Classification",
        "Exact Syllabus Topic": "Logistic Regression: Model Theory, Model Fit Statistics, Applications",
        "Implemented Feature": "Binary classification predicting `ai_required` (0 vs 1); Sigmoid link function; Maximum Likelihood; Confusion Matrix; Accuracy, Precision, Recall, F1, and ROC Curve.",
        "Source File & Function": "`src/classification_models.py` (`train_logistic_regression`)",
        "Application Location": "Pages: *Machine Learning Lab (Tab 2)*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    # Unit IV
    {
        "Unit": "Unit IV",
        "Unit Title": "Segmentation & Time Series",
        "Exact Syllabus Topic": "Object Segmentation: Supervised vs Unsupervised Learning",
        "Implemented Feature": "Direct pedagogical comparison between supervised prediction (Regression/Classification) and unsupervised clustering (K-Means).",
        "Source File & Function": "`src/clustering.py` (`run_kmeans_clustering`), `src/regression_models.py`",
        "Application Location": "Pages: *Machine Learning Lab (Tab 6)*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit IV",
        "Unit Title": "Segmentation & Time Series",
        "Exact Syllabus Topic": "Tree Building: Regression, Classification, Overfitting, Pruning",
        "Implemented Feature": "Decision Tree Classifier & Regressor with tunable depth ($1$ to $10$), feature importances, and empirical train vs test overfitting curve.",
        "Source File & Function": "`src/classification_models.py` (`compute_tree_depth_overfitting_curve`)",
        "Application Location": "Pages: *Machine Learning Lab (Tab 3)*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit IV",
        "Unit Title": "Segmentation & Time Series",
        "Exact Syllabus Topic": "Multiple Decision Trees (Random Forests)",
        "Implemented Feature": "Random Forest Ensemble (100 trees) with bootstrap aggregation and feature importance comparison against single decision tree.",
        "Source File & Function": "`src/classification_models.py` (`train_random_forest_classifier`)",
        "Application Location": "Pages: *Machine Learning Lab (Tab 4)*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit IV",
        "Unit Title": "Segmentation & Time Series",
        "Exact Syllabus Topic": "Time Series Methods: ARIMA, Measures of Forecast Accuracy, STL Approach",
        "Implemented Feature": "Annual problem statement trend tracking; ARIMA(1,1,0) forecast for 2025-2026 with 95% confidence intervals; Backtesting MAE, RMSE, MAPE; STL suitability documentation.",
        "Source File & Function": "`src/time_series_analysis.py` (`fit_arima_forecast`, `compute_yearly_trends`)",
        "Application Location": "Pages: *Trends and Forecasting*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    # Unit V
    {
        "Unit": "Unit V",
        "Unit Title": "Data Visualization Techniques",
        "Exact Syllabus Topic": "Pixel-Oriented Visualization Techniques",
        "Implemented Feature": "Correlation Matrix Heatmap with continuous color gradient; Cross-tabulation frequency density heatmaps.",
        "Source File & Function": "`src/visualizations.py` (`plot_correlation_heatmap`, `plot_crosstab_heatmap`)",
        "Application Location": "Pages: *Exploratory Analysis*, *Data Analytics Lab*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit V",
        "Unit Title": "Data Visualization Techniques",
        "Exact Syllabus Topic": "Geometric Projection Visualization Techniques",
        "Implemented Feature": "Bivariate scatter plots; 2D Principal Component Analysis (PCA) projecting multidimensional features and clusters onto latent orthogonal axes.",
        "Source File & Function": "`src/visualizations.py` (`plot_scatter_bivariate`, `plot_pca_clusters`)",
        "Application Location": "Pages: *Exploratory Analysis*, *Machine Learning Lab*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit V",
        "Unit Title": "Data Visualization Techniques",
        "Exact Syllabus Topic": "Icon-Based Visualization Techniques",
        "Implemented Feature": "KPI metric cards with delta indicators, color-coded badges, and glyph scorecards.",
        "Source File & Function": "`src/visualizations.py`, `app.py`",
        "Application Location": "Executive Dashboard & KPI bars",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit V",
        "Unit Title": "Data Visualization Techniques",
        "Exact Syllabus Topic": "Hierarchical Visualization Techniques",
        "Implemented Feature": "Interactive Plotly Treemap displaying nesting of Organizer Type → Domain → Difficulty with color scaling.",
        "Source File & Function": "`src/visualizations.py` (`plot_hierarchical_treemap`)",
        "Application Location": "Pages: *Exploratory Analysis*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    },
    {
        "Unit": "Unit V",
        "Unit Title": "Data Visualization Techniques",
        "Exact Syllabus Topic": "Visualizing Complex Data and Relations",
        "Implemented Feature": "TF-IDF Vectorization & Cosine Similarity search matching; Actual vs Predicted regression dispersion plot; Residual diagnostics; Confusion matrices.",
        "Source File & Function": "`src/text_similarity.py`, `src/visualizations.py`",
        "Application Location": "Pages: *Text Similarity*, *ML Lab*",
        "Status": "Fully Implemented",
        "Category": "Core Syllabus"
    }
]

df_syl = pd.DataFrame(SYLLABUS_RECORDS)

# Interactive Unit Filter
unit_filter = st.selectbox(
    "Filter by Academic Unit:",
    options=["All Units", "Unit I: Data Management & Quality", "Unit II: Data Analytics & Modeling", "Unit III: Regression & Classification", "Unit IV: Segmentation & Time Series", "Unit V: Data Visualization Techniques"]
)

if unit_filter != "All Units":
    selected_u = unit_filter.split(":")[0]
    display_syl = df_syl[df_syl["Unit"] == selected_u]
else:
    display_syl = df_syl

st.markdown(f"### 📋 Mapped Topics Table ({len(display_syl)} Topics)")
st.dataframe(
    display_syl[["Unit", "Exact Syllabus Topic", "Implemented Feature", "Source File & Function", "Application Location", "Status"]],
    use_container_width=True
)

st.markdown("---")

# Quick Demonstration Navigation Cards
st.markdown("### 🚀 Faculty Demonstration Shortcuts")
d1, d2, d3, d4, d5 = st.columns(5)
with d1:
    st.markdown("**Unit I: Data Quality**")
    st.markdown("Inspect noise, missing values, duplicates, and IQR outliers.")
    st.page_link("pages/0_🔬_Data_Analytics_Lab.py", label="Open Lab Step 3", icon="🔬")

with d2:
    st.markdown("**Unit II: Analytics**")
    st.markdown("Explore 14 univariate stats and data scales.")
    st.page_link("pages/3_📈_Exploratory_Analysis.py", label="Open EDA Page", icon="📈")

with d3:
    st.markdown("**Unit III: Regression**")
    st.markdown("OLS, BLUE assumptions, and Logistic Regression.")
    st.page_link("pages/4_🤖_Machine_Learning_Lab.py", label="Open ML Lab", icon="🤖")

with d4:
    st.markdown("**Unit IV: Decision Trees**")
    st.markdown("Pruning, overfitting curves, and ARIMA forecasts.")
    st.page_link("pages/6_📅_Trends_and_Forecasting.py", label="Open Trends Page", icon="📅")

with d5:
    st.markdown("**Unit V: Visualization**")
    st.markdown("Pixel, geometric, icon, and hierarchical charts.")
    st.page_link("pages/3_📈_Exploratory_Analysis.py", label="Open Treemaps", icon="🌳")
