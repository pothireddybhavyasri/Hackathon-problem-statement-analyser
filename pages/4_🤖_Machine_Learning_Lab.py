"""
pages/4_🤖_Machine_Learning_Lab.py
Full Machine Learning Laboratory covering Linear Regression, Logistic Regression,
Decision Trees (with pruning & overfitting curves), Random Forests, K-Means Clustering, and Model Comparisons.
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
from src.regression_models import (
    prepare_regression_dataset, train_linear_regression,
    train_decision_tree_regressor, train_random_forest_regressor, compare_regression_models
)
from src.classification_models import (
    prepare_classification_dataset, train_logistic_regression,
    train_decision_tree_classifier, train_random_forest_classifier,
    compute_tree_depth_overfitting_curve, compare_classification_models
)
from src.clustering import (
    run_kmeans_clustering, compute_elbow_and_silhouette, prepare_clustering_data
)
from src.visualizations import (
    plot_actual_vs_predicted, plot_residuals_diagnostic,
    plot_confusion_matrix_heatmap, plot_pca_clusters
)

st.set_page_config(page_title="Machine Learning Lab — Hackathon Insights", page_icon="🤖", layout="wide")

st.markdown("# 🤖 Machine Learning Laboratory")
st.markdown(
    "Train, evaluate, prune, and compare supervised and unsupervised models. "
    "All models enforce strict 80/20 train/test splits with preprocessing fitted exclusively on training data to prevent data leakage."
)

raw_df = load_raw_data()
clean_df = load_cleaned_data() if check_dataset_exists(is_cleaned=True) else raw_df
active_df = clean_df

tab_reg, tab_clf, tab_dt, tab_rf, tab_comp, tab_cluster = st.tabs([
    "📈 Linear Regression & BLUE",
    "🎯 Logistic Regression",
    "🌳 Decision Trees & Pruning",
    "🌲 Random Forest Ensembles",
    "⚖️ Model Comparison Studio",
    "🧩 K-Means Clustering"
])

# ------------------------------------------------------------------------------
# TAB 1: LINEAR REGRESSION
# ------------------------------------------------------------------------------
with tab_reg:
    st.markdown("### 📈 Multiple Linear Regression (OLS) — Unit III")
    st.markdown("Target Variable: **`estimated_solution_days`** (Continuous Ratio)")
    
    reg_prep = prepare_regression_dataset(active_df, test_size=0.20, random_state=42)
    lr_res = train_linear_regression(reg_prep)
    
    m_test = lr_res["test_metrics"]
    m_train = lr_res["train_metrics"]

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Test R² Score", f"{m_test['R2']:.4f}")
    rc2.metric("Train R² Score", f"{m_train['R2']:.4f}")
    rc3.metric("Test MAE", f"{m_test['MAE']:.2f} days")
    rc4.metric("Test RMSE", f"{m_test['RMSE']:.2f} days")

    st.markdown("---")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.plotly_chart(
            plot_actual_vs_predicted(lr_res["y_test_true"], lr_res["y_test_pred"], "Linear Regression (OLS)"),
            use_container_width=True
        )
    with r_col2:
        st.plotly_chart(
            plot_residuals_diagnostic(lr_res["y_test_pred"], lr_res["residuals"]),
            use_container_width=True
        )

    st.markdown("##### Learned Model Coefficients")
    st.dataframe(lr_res["coefficients_df"], use_container_width=True)

    with st.expander("📚 Gauss-Markov Theorem & BLUE Property Assumptions (Unit III)", expanded=True):
        st.markdown("""
        The **Gauss-Markov Theorem** proves that under the classical assumptions, the Ordinary Least Squares (OLS) 
        estimator is the **Best Linear Unbiased Estimator (BLUE)** (minimum variance among all linear unbiased estimators):
        
        1. **Linearity in Parameters**: $y = X\\beta + \\varepsilon$
        2. **Strict Exogeneity**: $E[\\varepsilon | X] = 0$ (error term has conditional mean zero)
        3. **Spherical Errors (Homoscedasticity & No Autocorrelation)**: $\\text{Var}(\\varepsilon | X) = \\sigma^2 I_n$
        4. **No Perfect Multicollinearity**: Matrix $X$ has full column rank.
        5. **Residual Normality**: Residuals $\\varepsilon \\sim \\mathcal{N}(0, \\sigma^2)$ are approximately normally distributed (Residual Mean: `{0}`, Skewness: `{1}`).
        """.format(lr_res["diagnostics"]["residual_mean"], lr_res["diagnostics"]["residual_skewness"]))


# ------------------------------------------------------------------------------
# TAB 2: LOGISTIC REGRESSION
# ------------------------------------------------------------------------------
with tab_clf:
    st.markdown("### 🎯 Logistic Regression Binary Classification — Unit III")
    st.markdown("Target Variable: **`ai_required`** (0 = False, 1 = True)")

    clf_prep = prepare_classification_dataset(active_df, test_size=0.20, random_state=42)
    lr_clf = train_logistic_regression(clf_prep)
    cm_test = lr_clf["test_metrics"]

    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("Test Accuracy", f"{cm_test['Accuracy']*100:.1f}%")
    lc2.metric("Precision", f"{cm_test['Precision']:.4f}")
    lc3.metric("Recall", f"{cm_test['Recall']:.4f}")
    lc4.metric("F1-Score", f"{cm_test['F1_Score']:.4f}")

    st.markdown("---")
    lc_left, lc_right = st.columns(2)
    with lc_left:
        st.plotly_chart(
            plot_confusion_matrix_heatmap(lr_clf["confusion_matrix"]),
            use_container_width=True
        )
    with lc_right:
        # ROC Curve
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=lr_clf["roc_curve"]["fpr"],
            y=lr_clf["roc_curve"]["tpr"],
            mode="lines",
            name=f"ROC Curve (AUC = {cm_test['ROC_AUC']})",
            line=dict(color="#2563EB", width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random Chance",
            line=dict(color="gray", dash="dash")
        ))
        fig_roc.update_layout(title="<b>ROC Curve (Receiver Operating Characteristic)</b>", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=420)
        st.plotly_chart(fig_roc, use_container_width=True)

    with st.expander("💡 Academic Note: The Accuracy Paradox in Classification"):
        st.markdown("""
        In classification problems with class imbalance, a model that simply predicts the majority class can yield high **Accuracy** while having zero recall on the minority class. 
        Therefore, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC** provide a more dependable evaluation of true discriminative power than raw accuracy alone.
        """)


# ------------------------------------------------------------------------------
# TAB 3: DECISION TREES & PRUNING
# ------------------------------------------------------------------------------
with tab_dt:
    st.markdown("### 🌳 Decision Tree Building, Overfitting, & Pruning — Unit IV")
    st.markdown("Experiment with tree depth limits to witness the trade-off between bias, variance, and empirical overfitting.")

    dt_depth = st.slider("Select Max Tree Depth (Pruning Hyperparameter):", min_value=1, max_value=10, value=4)
    
    dt_res = train_decision_tree_classifier(clf_prep, max_depth=dt_depth)
    
    dt_c1, dt_c2, dt_c3, dt_c4 = st.columns(4)
    dt_c1.metric("Tree Depth", dt_depth)
    dt_c2.metric("Train Accuracy", f"{dt_res['train_metrics']['Accuracy']*100:.1f}%")
    dt_c3.metric("Test Accuracy", f"{dt_res['test_metrics']['Accuracy']*100:.1f}%")
    dt_c4.metric("Test F1-Score", f"{dt_res['test_metrics']['F1_Score']:.4f}")

    # Overfitting curve
    st.markdown("##### Empirical Overfitting vs Pruning Curve across Depths (1 to 10)")
    curve_df = compute_tree_depth_overfitting_curve(clf_prep, max_depth_limit=10)
    
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=curve_df["Depth"], y=curve_df["Train Accuracy"],
        mode="lines+markers", name="Train Accuracy", line=dict(color="#2563EB", width=2.5)
    ))
    fig_curve.add_trace(go.Scatter(
        x=curve_df["Depth"], y=curve_df["Test Accuracy"],
        mode="lines+markers", name="Test Accuracy (Generalization)", line=dict(color="#10B981", width=2.5)
    ))
    fig_curve.update_layout(
        title="<b>Decision Tree Complexity: Train vs Test Accuracy</b>",
        xaxis_title="Max Tree Depth",
        yaxis_title="Accuracy",
        height=420
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    st.markdown("##### Feature Importance Ranking in Decision Tree")
    st.dataframe(dt_res["feature_importance_df"].head(8), use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: RANDOM FOREST
# ------------------------------------------------------------------------------
with tab_rf:
    st.markdown("### 🌲 Multiple Decision Trees: Random Forest Ensemble — Unit IV")
    st.markdown("Random Forests combine bootstrap aggregating (bagging) with random feature subspaces to reduce single-tree variance.")

    n_trees = st.select_slider("Number of Decision Trees (Estimators):", options=[25, 50, 100, 150, 200], value=100)
    
    rf_res = train_random_forest_classifier(clf_prep, n_estimators=n_trees, max_depth=6)

    rfc1, rfc2, rfc3, rfc4 = st.columns(4)
    rfc1.metric("Ensemble Size", f"{n_trees} Trees")
    rfc2.metric("Test Accuracy", f"{rf_res['test_metrics']['Accuracy']*100:.1f}%")
    rfc3.metric("Test F1-Score", f"{rf_res['test_metrics']['F1_Score']:.4f}")
    rfc4.metric("Test ROC-AUC", f"{rf_res['test_metrics']['ROC_AUC']}")

    st.markdown("##### Top Predictive Feature Importances (Random Forest)")
    fig_rf_imp = px.bar(
        rf_res["feature_importance_df"].head(10).iloc[::-1],
        x="Importance",
        y="Feature",
        orientation="h",
        title="<b>Random Forest Feature Importance</b>",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    fig_rf_imp.update_layout(height=420)
    st.plotly_chart(fig_rf_imp, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 5: MODEL COMPARISON
# ------------------------------------------------------------------------------
with tab_comp:
    st.markdown("### ⚖️ Rigorous Model Comparison Studio")
    st.markdown("Evaluated using identical training/testing data partitions (80/20 split, seed=42).")

    st.markdown("#### Regression Benchmark (Target: `estimated_solution_days`)")
    reg_comp = compare_regression_models(reg_prep)
    st.dataframe(reg_comp, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Classification Benchmark (Target: `ai_required`)")
    clf_comp = compare_classification_models(clf_prep)
    st.dataframe(clf_comp, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 6: K-MEANS CLUSTERING
# ------------------------------------------------------------------------------
with tab_cluster:
    st.markdown("### 🧩 Unsupervised Segmentation: K-Means Clustering — Unit IV")
    st.markdown("Discovers latent problem clusters without pre-existing labels based on technical requirements, duration, and innovation scores.")

    k_chosen = st.slider("Select Number of Clusters (k):", min_value=2, max_value=8, value=4)
    
    # Compute Elbow curve
    X_clust, clust_cols, _ = prepare_clustering_data(active_df)
    elbow_df = compute_elbow_and_silhouette(X_clust, k_range=range(2, 9))
    
    el_c1, el_c2 = st.columns(2)
    with el_c1:
        fig_elbow = px.line(
            elbow_df, x="Clusters (k)", y="Inertia (WCSS)", markers=True,
            title="<b>Elbow Method: Inertia vs Cluster Count (k)</b>"
        )
        fig_elbow.update_layout(height=380)
        st.plotly_chart(fig_elbow, use_container_width=True)

    with el_c2:
        fig_sil = px.line(
            elbow_df, x="Clusters (k)", y="Silhouette Score", markers=True,
            title="<b>Silhouette Analysis vs Cluster Count (k)</b>",
            color_discrete_sequence=["#10B981"]
        )
        fig_sil.update_layout(height=380)
        st.plotly_chart(fig_sil, use_container_width=True)

    km_output = run_kmeans_clustering(active_df, n_clusters=k_chosen)
    
    st.markdown(f"**Silhouette Score (k={k_chosen})**: `{km_output['silhouette_score']}` &nbsp;|&nbsp; **Inertia**: `{km_output['inertia']}`")

    st.plotly_chart(
        plot_pca_clusters(
            km_output["pca_data"],
            km_output["centroids_pca"],
            km_output["explained_variance_ratio"]
        ),
        use_container_width=True
    )

    st.markdown("##### Cluster Summary Profiles (Mean Feature Values)")
    st.dataframe(km_output["cluster_profiles"], use_container_width=True)
