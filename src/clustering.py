"""
src/clustering.py
Module for unsupervised segmentation and K-Means clustering.
Implements Elbow Method (Inertia), Silhouette analysis, PCA 2D projections, and cluster profiling.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

CLUSTERING_FEATURES = [
    "estimated_solution_days",
    "team_size",
    "innovation_score",
    "impact_score",
    "problem_length_words",
    "number_of_technologies",
    "ai_required",
    "hardware_required"
]


def prepare_clustering_data(df: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Extracts, handles missing values, and standardizes features for clustering."""
    cols = [c for c in (feature_cols or CLUSTERING_FEATURES) if c in df.columns]
    sub_df = df[cols].copy().dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub_df)
    return X_scaled, cols, sub_df


def compute_elbow_and_silhouette(X: np.ndarray, k_range: range = range(2, 9)) -> pd.DataFrame:
    """
    Evaluates KMeans over k=2..8 to compute:
    1. Inertia (Within-Cluster Sum of Squares) for the Elbow Method.
    2. Silhouette Score for cluster cohesion and separation quality.
    """
    records = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)
        records.append({
            "Clusters (k)": k,
            "Inertia (WCSS)": round(float(kmeans.inertia_), 2),
            "Silhouette Score": round(float(sil), 4)
        })
    return pd.DataFrame(records)


def run_kmeans_clustering(
    df: pd.DataFrame,
    n_clusters: int = 4,
    feature_cols: List[str] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Runs KMeans clustering, computes 2D PCA projection for geometric visualization,
    and returns cluster assignments, sizes, and tabular profile summaries.
    """
    X_scaled, cols, valid_df = prepare_clustering_data(df, feature_cols)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = float(silhouette_score(X_scaled, labels))

    # Add cluster label back to valid_df
    result_df = valid_df.copy()
    result_df["Cluster"] = [f"Cluster {i+1}" for i in labels]

    # Perform 2D PCA for visual geometric projection
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = [round(float(v) * 100.0, 1) for v in pca.explained_variance_ratio_]

    pca_df = pd.DataFrame({
        "PCA_1": X_pca[:, 0],
        "PCA_2": X_pca[:, 1],
        "Cluster": result_df["Cluster"].values
    })
    
    if "domain" in df.columns:
        pca_df["Domain"] = df.loc[valid_df.index, "domain"].values
    if "problem_title" in df.columns:
        pca_df["Title"] = df.loc[valid_df.index, "problem_title"].values

    # Transform centroids to PCA space
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame({
        "PCA_1": centroids_pca[:, 0],
        "PCA_2": centroids_pca[:, 1],
        "Cluster": [f"Cluster {i+1} Centroid" for i in range(n_clusters)]
    })

    # Cluster summary profiles
    profile_df = result_df.groupby("Cluster")[cols].mean().round(2).reset_index()
    counts = result_df["Cluster"].value_counts().reset_index()
    counts.columns = ["Cluster", "Member Count"]
    profile_df = pd.merge(counts, profile_df, on="Cluster")

    return {
        "n_clusters": n_clusters,
        "silhouette_score": round(sil_score, 4),
        "inertia": round(float(kmeans.inertia_), 2),
        "cluster_profiles": profile_df,
        "pca_data": pca_df,
        "centroids_pca": centroids_df,
        "explained_variance_ratio": var_exp,
        "total_explained_variance": round(sum(var_exp), 1),
        "clustered_indices": valid_df.index.tolist(),
        "cluster_labels": labels.tolist(),
        "interpretation": (
            f"K-Means grouped {len(valid_df)} problem statements into {n_clusters} clusters with an overall Silhouette Score of {sil_score:.3f}. "
            f"The 2D PCA projection captures {sum(var_exp):.1f}% of multidimensional variance."
        )
    }
