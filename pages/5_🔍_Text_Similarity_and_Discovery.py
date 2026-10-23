"""
pages/5_🔍_Text_Similarity_and_Discovery.py
Semantic Problem Statement Discovery using TF-IDF Vectorization and Cosine Similarity.
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
from src.text_similarity import compute_tfidf_matrix, search_similar_problems

st.set_page_config(page_title="Text Similarity & Discovery — Hackathon Insights", page_icon="🔍", layout="wide")

st.markdown("# 🔍 Text Similarity & Problem Statement Discovery")
st.markdown(
    "Search for textually and semantically related hackathon problem statements using Term Frequency–Inverse Document Frequency (TF-IDF) "
    "vectorization and Cosine Similarity."
)

raw_df = load_raw_data()
clean_df = load_cleaned_data() if check_dataset_exists(is_cleaned=True) else raw_df
active_df = clean_df

# Cache TF-IDF matrix
if "tfidf_cache" not in st.session_state:
    with st.spinner("Indexing problem descriptions into TF-IDF vector space..."):
        st.session_state.tfidf_cache = compute_tfidf_matrix(active_df, text_column="problem_description")

tfidf_data = st.session_state.tfidf_cache

st.info(
    f"📚 **Vector Space Indexed**: `{tfidf_data['corpus_size']}` documents vectorized into `{tfidf_data['vocabulary_size']}` "
    f"unique unigram and bigram TF-IDF features."
)

st.markdown("### 🔎 Query Formulation")

query_mode = st.radio(
    "Choose Query Input Method:",
    options=["Custom Search Phrase / Requirements", "Select from Existing Problem Statements"],
    horizontal=True
)

if query_mode == "Custom Search Phrase / Requirements":
    user_query = st.text_area(
        "Enter problem statement topic, technical requirements, or keywords:",
        value="Edge IoT machine learning sensors for real-time forest fire detection and air quality alert"
    )
else:
    sample_titles = active_df["problem_title"].tolist()
    selected_sample = st.selectbox("Select Problem Statement:", sample_titles)
    selected_row = active_df[active_df["problem_title"] == selected_sample].iloc[0]
    user_query = selected_row["problem_description"]
    st.caption(f"Using description of `{selected_row['problem_id']}` as reference vector.")

c_f1, c_f2, c_f3, c_f4 = st.columns(4)
with c_f1:
    domain_opts = ["All"] + sorted(active_df["domain"].dropna().unique().tolist())
    d_filt = st.selectbox("Domain Filter", domain_opts)
with c_f2:
    diff_opts = ["All"] + sorted(active_df["difficulty_level"].dropna().unique().tolist())
    df_filt = st.selectbox("Difficulty Filter", diff_opts)
with c_f3:
    yr_opts = ["All"] + sorted([int(y) for y in active_df["year"].dropna().unique().tolist()])
    y_filt = st.selectbox("Year Filter", yr_opts)
with c_f4:
    top_k = st.slider("Top Results Count:", min_value=3, max_value=15, value=5)

if st.button("🚀 Search Similar Problem Statements", type="primary"):
    if user_query.strip():
        results_df = search_similar_problems(
            query_text=user_query,
            df=active_df,
            tfidf_data=tfidf_data,
            top_n=top_k,
            domain_filter=d_filt,
            difficulty_filter=df_filt,
            year_filter=y_filt
        )

        st.markdown(f"### 🎯 Top {len(results_df)} Closest Matching Problems")
        
        if not results_df.empty:
            for idx, row in results_df.iterrows():
                score = row["similarity_score"]
                color_badge = "#10B981" if score > 0.4 else ("#2563EB" if score > 0.2 else "#64748B")
                
                with st.container():
                    st.markdown(f"""
                    <div style="border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px; background-color: #FFFFFF;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.15rem; font-weight: 700; color: #1E293B;">
                                <code>{row['problem_id']}</code>: {row['problem_title']}
                            </span>
                            <span style="background-color: {color_badge}; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem;">
                                Cosine Similarity: {score:.4f}
                            </span>
                        </div>
                        <p style="color: #475569; margin-top: 8px; font-size: 0.95rem;">{row['problem_description']}</p>
                        <div style="font-size: 0.85rem; color: #64748B;">
                            <b>Domain:</b> {row['domain']} &nbsp;|&nbsp; 
                            <b>Difficulty:</b> {row['difficulty_level']} &nbsp;|&nbsp; 
                            <b>AI Required:</b> {'Yes' if row['ai_required'] == 1 else 'No'} &nbsp;|&nbsp; 
                            <b>Hackathon:</b> {row['hackathon_name']} ({row['year']})
                        </div>
                        <div style="margin-top: 6px; font-size: 0.85rem; color: #2563EB;">
                            <b>Tech Stack:</b> <code>{row['technology_tags']}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Export button
            csv_exp = io.StringIO()
            results_df.to_csv(csv_exp, index=False)
            st.download_button(
                "📥 Export Similarity Search Results (CSV)",
                data=csv_exp.getvalue(),
                file_name="similar_problems_search_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("No problem statements matched the query criteria under the selected filters.")
    else:
        st.warning("Please enter a search query above.")

st.markdown("---")
with st.expander("📚 Mathematical Methodology: TF-IDF & Cosine Similarity"):
    st.markdown(r"""
    1. **Term Frequency–Inverse Document Frequency (TF-IDF)**:
       $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
       Weights words higher if they are frequent in a specific document but rare across the entire problem statement corpus.
    2. **Cosine Similarity**:
       $$\text{Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \frac{\sum_{i=1}^n u_i v_i}{\sqrt{\sum_{i=1}^n u_i^2} \sqrt{\sum_{i=1}^n v_i^2}}$$
       Measures the cosine of the angle between two $n$-dimensional normalized feature vectors in $[0.0, 1.0]$.
    3. **Important Distinction**: A high cosine similarity score indicates lexical and topical proximity in the vectorized feature space, not that the problems are exact duplicates or plagiarism.
    """)
