"""
pages/3_📈_Exploratory_Analysis.py
In-depth Exploratory Data Analysis and Statistical Inference.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_raw_data, load_cleaned_data, check_dataset_exists
from src.statistics_analysis import (
    compute_descriptive_stats, compute_correlation_matrix,
    compute_crosstab_analysis, perform_hypothesis_tests
)
from src.eda import (
    extract_technology_frequencies, compute_domain_subdomain_breakdown,
    compute_hardware_ai_summary, classify_variables
)
from src.visualizations import (
    plot_correlation_heatmap, plot_crosstab_heatmap,
    plot_hierarchical_treemap, plot_domain_distribution_bar, plot_technology_frequency
)

st.set_page_config(page_title="Exploratory Analysis — Hackathon Insights", page_icon="📈", layout="wide")

st.markdown("# 📈 Exploratory Data Analysis (EDA) & Statistical Inference")
st.markdown("Examine multivariate distributions, categorical hierarchies, technology frequencies, and formal hypothesis testing.")

# Load active dataset
raw_df = load_raw_data()
clean_df = load_cleaned_data() if check_dataset_exists(is_cleaned=True) else raw_df
active_df = clean_df

tab_uni, tab_hier, tab_tech, tab_hypo, tab_tax = st.tabs([
    "📊 Descriptive Univariate Stats",
    "🌳 Hierarchical Treemap & Domains",
    "💻 Technology Tag Distribution",
    "🧪 Statistical Hypothesis Tests",
    "📚 Variable Taxonomy"
])

# ------------------------------------------------------------------------------
# TAB 1: DESCRIPTIVE STATS
# ------------------------------------------------------------------------------
with tab_uni:
    st.markdown("### 📊 14 Univariate Statistical Measures")
    st.markdown("Computes Central Tendency (Mean, Median, Mode) and Dispersion (Variance, Std Dev, Range, IQR, Skewness, Kurtosis).")
    stats_df = compute_descriptive_stats(active_df)
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔥 Pixel-Oriented Correlation Heatmap")
    corr_df = compute_correlation_matrix(active_df)
    st.plotly_chart(plot_correlation_heatmap(corr_df), use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: HIERARCHY & DOMAINS
# ------------------------------------------------------------------------------
with tab_hier:
    st.markdown("### 🌳 Hierarchical Visualization Technique: Multi-Level Treemap")
    st.markdown("Nests **Organizer Type → Domain → Difficulty Level** with box color scaled by `estimated_solution_days`.")
    st.plotly_chart(plot_hierarchical_treemap(active_df), use_container_width=True)

    st.markdown("---")
    c_xt1, c_xt2 = st.columns(2)
    with c_xt1:
        st.markdown("#### Cross-Tabulation: Domain vs Difficulty Level")
        xt_diff = compute_crosstab_analysis(active_df, "domain", "difficulty_level")
        st.plotly_chart(plot_crosstab_heatmap(xt_diff["counts"], "Problem Count: Domain vs Difficulty"), use_container_width=True)
    with c_xt2:
        st.markdown("#### Cross-Tabulation: Domain vs Implementation Feasibility")
        xt_feas = compute_crosstab_analysis(active_df, "domain", "implementation_feasibility")
        st.plotly_chart(plot_crosstab_heatmap(xt_feas["counts"], "Problem Count: Domain vs Feasibility"), use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 3: TECHNOLOGY TAGS
# ------------------------------------------------------------------------------
with tab_tech:
    st.markdown("### 💻 Technology Frequency Distribution")
    top_k = st.slider("Select Top N Technologies to display:", min_value=10, max_value=35, value=20)
    tech_freq = extract_technology_frequencies(active_df, top_n=top_k)
    st.plotly_chart(plot_technology_frequency(tech_freq, top_n=top_k), use_container_width=True)
    
    st.markdown("##### Detailed Frequency Table")
    st.dataframe(tech_freq, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: HYPOTHESIS TESTING
# ------------------------------------------------------------------------------
with tab_hypo:
    st.markdown("### 🧪 Formal Inferential Hypothesis Testing")
    st.markdown("Evaluates whether observed group differences and categorical associations reflect genuine population effects or random sampling variance.")

    hypo_results = perform_hypothesis_tests(active_df)

    # 1. ANOVA
    if "anova_domain_days" in hypo_results:
        an = hypo_results["anova_domain_days"]
        with st.container():
            st.markdown(f"#### 1. {an['test_name']}")
            st.markdown(f"**$H_0$ (Null Hypothesis)**: Mean solution days are identical across all domains.")
            st.markdown(f"**$H_1$ (Alternative Hypothesis)**: At least one domain requires a significantly different mean number of days.")
            ac1, ac2, ac3 = st.columns(3)
            ac1.metric("F-Statistic", an["f_statistic"])
            ac2.metric("p-value", an["p_value_formatted"])
            ac3.metric("Significant at α = 0.05", "YES (Reject H0)" if an["significant_at_05"] else "NO (Fail to Reject)")
            st.info(f"**Conclusion**: {an['interpretation']}")

    st.markdown("---")

    # 2. Chi-Square
    if "chi2_domain_ai" in hypo_results:
        ch = hypo_results["chi2_domain_ai"]
        with st.container():
            st.markdown(f"#### 2. {ch['test_name']}")
            st.markdown(f"**$H_0$**: Problem Domain and AI Requirement are statistically independent.")
            st.markdown(f"**$H_1$**: Problem Domain and AI Requirement are statistically dependent.")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Chi-Square Statistic (χ²)", ch["chi2_statistic"])
            cc2.metric("Degrees of Freedom", ch["degrees_of_freedom"])
            cc3.metric("p-value", ch["p_value_formatted"])
            st.info(f"**Conclusion**: {ch['interpretation']}")

    st.markdown("---")

    # 3. Two-Sample T-Test
    if "ttest_ai_days" in hypo_results:
        tt = hypo_results["ttest_ai_days"]
        with st.container():
            st.markdown(f"#### 3. {tt['test_name']}")
            st.markdown(r"**$H_0$**: $\mu_{\text{AI=1}} = \mu_{\text{AI=0}}$ (Mean solution days are identical).")
            st.markdown(r"**$H_1$**: $\mu_{\text{AI=1}} \neq \mu_{\text{AI=0}}$ (Mean solution days differ).")
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("AI Required Mean Days", f"{tt['mean_group_1']} d")
            tc2.metric("Non-AI Mean Days", f"{tt['mean_group_2']} d")
            tc3.metric("t-Statistic", tt["t_statistic"])
            st.info(f"**Conclusion**: {tt['interpretation']}")


# ------------------------------------------------------------------------------
# TAB 5: VARIABLE TAXONOMY
# ------------------------------------------------------------------------------
with tab_tax:
    st.markdown("### 📚 Academic Scale of Measurement Taxonomy (Unit II)")
    tax_df = classify_variables(active_df)
    st.dataframe(tax_df, use_container_width=True)
