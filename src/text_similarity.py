"""
src/text_similarity.py
Module for TF-IDF text vectorization, Cosine Similarity matching,
and semantic search across hackathon problem statements.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_matrix(
    df: pd.DataFrame,
    text_column: str = "problem_description",
    max_features: int = 2500,
    ngram_range: tuple = (1, 2)
) -> Dict[str, Any]:
    """
    Fits a TF-IDF vectorizer over the problem text corpus (English stopwords removed).
    Returns the fitted vectorizer and dense/sparse TF-IDF matrix.
    """
    if df is None or text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    corpus = df[text_column].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    return {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "corpus_size": len(corpus)
    }


def search_similar_problems(
    query_text: str,
    df: pd.DataFrame,
    tfidf_data: Optional[Dict[str, Any]] = None,
    text_column: str = "problem_description",
    top_n: int = 5,
    domain_filter: Optional[str] = None,
    difficulty_filter: Optional[str] = None,
    year_filter: Optional[int] = None
) -> pd.DataFrame:
    """
    Transforms the query_text into the fitted TF-IDF feature space and computes
    Cosine Similarity against all problem statement representations.
    Applies metadata filters and returns the Top-N closest matching records.
    """
    if not query_text or not query_text.strip():
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Fit TF-IDF if not passed
    if tfidf_data is None:
        tfidf_data = compute_tfidf_matrix(df, text_column=text_column)

    vectorizer = tfidf_data["vectorizer"]
    tfidf_matrix = tfidf_data["tfidf_matrix"]

    query_vec = vectorizer.transform([query_text])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    results_df = df.copy()
    results_df["similarity_score"] = sim_scores.round(4)

    # Apply filters
    if domain_filter and domain_filter != "All":
        results_df = results_df[results_df["domain"] == domain_filter]
    if difficulty_filter and difficulty_filter != "All":
        results_df = results_df[results_df["difficulty_level"] == difficulty_filter]
    if year_filter and year_filter != "All":
        results_df = results_df[results_df["year"] == int(year_filter)]

    sorted_df = results_df.sort_values(by="similarity_score", ascending=False).head(top_n)

    display_cols = [
        "problem_id",
        "similarity_score",
        "problem_title",
        "domain",
        "difficulty_level",
        "technology_tags",
        "ai_required",
        "hackathon_name",
        "year",
        "problem_description"
    ]
    available_display = [c for c in display_cols if c in sorted_df.columns]

    return sorted_df[available_display].reset_index(drop=True)
