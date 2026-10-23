"""
pages/1_📊_Data_Explorer.py
Interactive Data Explorer for filtering, querying, and inspecting hackathon problem statements.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import io
import streamlit as st
import pandas as pd
from src.data_loader import load_raw_data, load_cleaned_data, check_dataset_exists

st.set_page_config(page_title="Data Explorer — Hackathon Insights", page_icon="📊", layout="wide")

st.markdown("# 📊 Interactive Data Explorer")
st.markdown("Search, slice, filter, and inspect hackathon problem statements across multiple dimensions.")

# Load datasets
raw_df = load_raw_data()
clean_df = load_cleaned_data() if check_dataset_exists(is_cleaned=True) else raw_df

# Controls bar
c_mode, c_cohort = st.columns([0.4, 0.6])
with c_mode:
    dataset_mode = st.radio(
        "Select Dataset Version:",
        options=["Cleaned Dataset", "Raw Uncleaned Dataset"],
        horizontal=True
    )

active_df = clean_df if dataset_mode == "Cleaned Dataset" else raw_df

st.markdown("---")

# Sidebar Filters
with st.sidebar:
    st.markdown("### 🎛️ Filter Panel")
    
    # Domain Filter
    domains = ["All"] + sorted(active_df["domain"].dropna().astype(str).unique().tolist())
    selected_domain = st.selectbox("Domain", domains)
    
    # Difficulty Filter
    difficulties = ["All"] + sorted(active_df["difficulty_level"].dropna().astype(str).unique().tolist())
    selected_diff = st.selectbox("Difficulty Level", difficulties)

    # Year Filter
    years = ["All"] + sorted([int(y) for y in active_df["year"].dropna().unique().tolist()])
    selected_year = st.selectbox("Calendar Year", years)

    # Organizer Filter
    organizers = ["All"] + sorted(active_df["organizer_type"].dropna().astype(str).unique().tolist())
    selected_org = st.selectbox("Organizer Type", organizers)

    # Data Type / Provenance Filter
    cohorts = ["All"] + sorted(active_df["data_type"].dropna().astype(str).unique().tolist())
    selected_cohort = st.selectbox("Provenance Cohort", cohorts)

    # AI Required Filter
    ai_choice = st.selectbox("AI Required", ["All", "Yes (1)", "No (0)"])

    # Search keyword
    search_keyword = st.text_input("🔍 Search Title or Description", "")

# Apply Filters
filtered_df = active_df.copy()

if selected_domain != "All":
    filtered_df = filtered_df[filtered_df["domain"] == selected_domain]

if selected_diff != "All":
    filtered_df = filtered_df[filtered_df["difficulty_level"] == selected_diff]

if selected_year != "All":
    filtered_df = filtered_df[filtered_df["year"] == int(selected_year)]

if selected_org != "All":
    filtered_df = filtered_df[filtered_df["organizer_type"] == selected_org]

if selected_cohort != "All":
    filtered_df = filtered_df[filtered_df["data_type"] == selected_cohort]

if ai_choice == "Yes (1)":
    filtered_df = filtered_df[filtered_df["ai_required"].isin([1, "1", "YES", "yes", True])]
elif ai_choice == "No (0)":
    filtered_df = filtered_df[filtered_df["ai_required"].isin([0, "0", "NO", "no", False])]

if search_keyword.strip():
    kw = search_keyword.strip().lower()
    mask = (
        filtered_df["problem_title"].astype(str).str.lower().str.contains(kw) |
        filtered_df["problem_description"].astype(str).str.lower().str.contains(kw) |
        filtered_df["technology_tags"].astype(str).str.lower().str.contains(kw)
    )
    filtered_df = filtered_df[mask]

# Summary of Filtered View
f_col1, f_col2, f_col3, f_col4 = st.columns(4)
f_col1.metric("Matching Records", len(filtered_df), delta=f"{len(filtered_df) - len(active_df)} from total")
f_col2.metric("Sourced Problems", int((filtered_df["data_type"] == "sourced").sum()))
f_col3.metric("Synthetic Problems", int((filtered_df["data_type"] == "synthetic").sum()))
f_col4.metric("Avg Solution Days", f"{filtered_df['estimated_solution_days'].mean():.1f} d" if not filtered_df.empty else "N/A")

st.markdown("### 📋 Filtered Records Table")

# Column Selection
default_cols = [
    "problem_id", "problem_title", "domain", "difficulty_level",
    "estimated_solution_days", "ai_required", "technology_tags", "hackathon_name", "data_type"
]
avail_cols = [c for c in default_cols if c in filtered_df.columns]
all_cols = filtered_df.columns.tolist()

selected_cols = st.multiselect(
    "Configure Visible Columns:",
    options=all_cols,
    default=avail_cols
)

st.dataframe(filtered_df[selected_cols], use_container_width=True, height=400)

# Export Filtered CSV
csv_buff = io.StringIO()
filtered_df.to_csv(csv_buff, index=False)
st.download_button(
    "📥 Export Filtered Subset (CSV)",
    data=csv_buff.getvalue(),
    file_name="filtered_hackathon_problems.csv",
    mime="text/csv"
)

st.markdown("---")

# Record Inspector
st.markdown("### 🔎 Individual Record Deep-Dive Inspector")
if not filtered_df.empty:
    chosen_id = st.selectbox(
        "Select Problem ID to inspect:",
        options=filtered_df["problem_id"].tolist()
    )
    rec = filtered_df[filtered_df["problem_id"] == chosen_id].iloc[0]

    with st.container():
        st.markdown(f"#### `{rec['problem_id']}`: {rec['problem_title']}")
        st.markdown(f"**Description**: {rec['problem_description']}")
        
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(f"- **Domain**: `{rec['domain']}`")
        r1.markdown(f"- **Sub-domain**: `{rec['sub_domain']}`")
        r2.markdown(f"- **Difficulty**: `{rec['difficulty_level']}`")
        r2.markdown(f"- **Feasibility**: `{rec['implementation_feasibility']}`")
        r3.markdown(f"- **Est. Days**: `{rec['estimated_solution_days']}`")
        r3.markdown(f"- **Team Size**: `{rec['team_size']}`")
        r4.markdown(f"- **Innovation**: `{rec['innovation_score']}/10`")
        r4.markdown(f"- **Impact**: `{rec['impact_score']}/10`")

        st.markdown(f"**Technologies**: `{rec['technology_tags']}`")
        st.markdown(f"**Canonical Source**: [{rec['source_url']}]({rec['source_url']}) &nbsp;|&nbsp; **Cohort**: `{rec['data_type']}`")
