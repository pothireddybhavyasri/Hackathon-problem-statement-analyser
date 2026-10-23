"""
src/visualizations.py
Interactive data visualization module built with Plotly.
Implements the 5 categories of Unit V:
1. Pixel-Oriented (Heatmaps, correlation matrices)
2. Geometric Projection (Scatter plots, 2D PCA projections)
3. Icon-Based (KPI metric cards, scorecards)
4. Hierarchical (Treemaps, Sunbursts)
5. Complex Data & Relations (Residual diagnostics, ROC curves, Forecast ribbons)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Aesthetic color palette
COLOR_PRIMARY = "#2563EB"    # Royal Blue
COLOR_ACCENT = "#10B981"     # Emerald Green
COLOR_WARNING = "#F59E0B"    # Amber
COLOR_DANGER = "#EF4444"     # Rose Red
COLOR_MUTED = "#6B7280"      # Cool Gray
COLOR_PURPLE = "#8B5CF6"     # Violet

THEME_LAYOUT = {
    "font": {"family": "Inter, -apple-system, sans-serif", "size": 12},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"l": 40, "r": 40, "t": 50, "b": 40},
}


# ==============================================================================
# 1. PIXEL-ORIENTED VISUALIZATION TECHNIQUES (Unit V)
# ==============================================================================

def plot_correlation_heatmap(corr_df: pd.DataFrame, title: str = "Pearson Correlation Matrix") -> go.Figure:
    """
    Renders a pixel-oriented heatmap where pixel color mappings represent
    inter-variable correlation coefficients between -1.0 and +1.0.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            colorbar=dict(title="Correlation (r)"),
            text=np.round(corr_df.values, 2),
            texttemplate="%{text}",
            hoverongaps=False
        )
    )
    fig.update_layout(
        **THEME_LAYOUT,
        title=f"<b>{title}</b>",
        xaxis_title="Features",
        yaxis_title="Features",
        height=520
    )
    return fig


def plot_crosstab_heatmap(
    crosstab_df: pd.DataFrame, title: str = "Cross-Tabulation Frequency Density"
) -> go.Figure:
    """Renders a pixel-oriented frequency density map across two categorical dimensions."""
    # Exclude margin totals if present
    plot_df = crosstab_df.drop(index="Total", columns="Total", errors="ignore")
    fig = go.Figure(
        data=go.Heatmap(
            z=plot_df.values,
            x=plot_df.columns,
            y=plot_df.index,
            colorscale="Viridis",
            text=plot_df.values,
            texttemplate="%{text}",
            hoverongaps=False
        )
    )
    fig.update_layout(
        **THEME_LAYOUT,
        title=f"<b>{title}</b>",
        xaxis_title=plot_df.columns.name or "Sub-Category",
        yaxis_title=plot_df.index.name or "Primary Dimension",
        height=480
    )
    return fig


# ==============================================================================
# 2. GEOMETRIC PROJECTION VISUALIZATION TECHNIQUES (Unit V)
# ==============================================================================

def plot_scatter_bivariate(
    df: pd.DataFrame,
    x_col: str = "estimated_solution_days",
    y_col: str = "innovation_score",
    color_col: str = "domain",
    size_col: Optional[str] = "team_size",
    title: str = "Bivariate Geometric Projection"
) -> go.Figure:
    """
    Plots multidimensional records onto Cartesian coordinates with point size
    and categorical color projections.
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col if size_col in df.columns else None,
        hover_data=["problem_id", "problem_title", "difficulty_level"],
        title=f"<b>{title}</b>",
        labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
        opacity=0.85
    )
    fig.update_layout(**THEME_LAYOUT, height=520)
    return fig


def plot_pca_clusters(
    pca_df: pd.DataFrame,
    centroids_df: Optional[pd.DataFrame] = None,
    var_ratio: Optional[List[float]] = None
) -> go.Figure:
    """
    Projects high-dimensional feature vectors into 2D PCA latent space
    colored by assigned cluster label, with optional cluster centroids.
    """
    x_label = f"Principal Component 1 ({var_ratio[0]}% var)" if var_ratio else "PC 1"
    y_label = f"Principal Component 2 ({var_ratio[1]}% var)" if var_ratio else "PC 2"

    fig = px.scatter(
        pca_df,
        x="PCA_1",
        y="PCA_2",
        color="Cluster",
        hover_data=["Domain", "Title"] if "Domain" in pca_df.columns else None,
        title="<b>Geometric Projection: 2D PCA Cluster Distribution</b>",
        labels={"PCA_1": x_label, "PCA_2": y_label},
        opacity=0.8
    )

    if centroids_df is not None:
        fig.add_trace(
            go.Scatter(
                x=centroids_df["PCA_1"],
                y=centroids_df["PCA_2"],
                mode="markers+text",
                marker=dict(symbol="star", size=16, color="black", line=dict(color="white", width=1.5)),
                text=centroids_df["Cluster"],
                textposition="top center",
                name="Cluster Centroids"
            )
        )

    fig.update_layout(**THEME_LAYOUT, height=540)
    return fig


# ==============================================================================
# 3. ICON-BASED VISUALIZATION TECHNIQUES (Unit V)
# ==============================================================================

def plot_domain_distribution_bar(df: pd.DataFrame) -> go.Figure:
    """Plots clean bar distribution of problem statements across domains with data labels."""
    domain_counts = df["domain"].value_counts().reset_index()
    domain_counts.columns = ["Domain", "Count"]

    fig = px.bar(
        domain_counts,
        x="Count",
        y="Domain",
        orientation="h",
        text="Count",
        title="<b>Problem Statements Distribution by Domain</b>",
        color="Count",
        color_continuous_scale="Blues"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**THEME_LAYOUT, yaxis={"categoryorder": "total ascending"}, height=460)
    return fig


def plot_technology_frequency(tech_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Plots top technologies frequency bar chart."""
    sub_df = tech_df.head(top_n).iloc[::-1]
    fig = px.bar(
        sub_df,
        x="Count",
        y="Technology",
        orientation="h",
        text="Count",
        title=f"<b>Top {top_n} Technologies Associated with Hackathon Problems</b>",
        color="Count",
        color_continuous_scale="Teal"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**THEME_LAYOUT, height=480)
    return fig


# ==============================================================================
# 4. HIERARCHICAL VISUALIZATION TECHNIQUES (Unit V)
# ==============================================================================

def plot_hierarchical_treemap(df: pd.DataFrame) -> go.Figure:
    """
    Renders an interactive hierarchical treemap nesting:
    Organizer Type -> Domain -> Difficulty Level.
    """
    path_cols = ["organizer_type", "domain", "difficulty_level"]
    valid_cols = [c for c in path_cols if c in df.columns]

    fig = px.treemap(
        df,
        path=valid_cols,
        title="<b>Hierarchical Breakdown: Organizer Type → Domain → Difficulty</b>",
        color="estimated_solution_days" if "estimated_solution_days" in df.columns else None,
        color_continuous_scale="RdBu_r"
    )
    fig.update_layout(**THEME_LAYOUT, height=560)
    return fig


# ==============================================================================
# 5. COMPLEX RELATIONS & MODEL DIAGNOSTICS (Unit V)
# ==============================================================================

def plot_actual_vs_predicted(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Linear Regression"
) -> go.Figure:
    """
    Plots Actual vs Predicted values with a 45-degree diagonal reference line
    to diagnose regression dispersion.
    """
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="Predictions",
            marker=dict(color=COLOR_PRIMARY, opacity=0.7, size=8)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal 45° Fit",
            line=dict(color=COLOR_DANGER, dash="dash", width=2)
        )
    )
    fig.update_layout(
        **THEME_LAYOUT,
        title=f"<b>Actual vs Predicted ({model_name})</b>",
        xaxis_title="Actual Solution Days",
        yaxis_title="Predicted Solution Days",
        height=480
    )
    return fig


def plot_residuals_diagnostic(y_pred: np.ndarray, residuals: np.ndarray) -> go.Figure:
    """
    Renders Residuals vs Fitted values to verify BLUE Gauss-Markov homoscedasticity assumption.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(color=COLOR_PURPLE, opacity=0.7, size=7),
            name="Residuals (e = y - ŷ)"
        )
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="red", line_width=1.5)
    fig.update_layout(
        **THEME_LAYOUT,
        title="<b>Gauss-Markov BLUE Diagnostic: Residuals vs Fitted</b>",
        xaxis_title="Fitted / Predicted Values (ŷ)",
        yaxis_title="Residuals (e)",
        height=440
    )
    return fig


def plot_confusion_matrix_heatmap(cm: List[List[int]], labels: List[str] = ["AI Not Required", "AI Required"]) -> go.Figure:
    """Renders formatted Confusion Matrix with True/False Positives & Negatives."""
    z = np.array(cm)
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=f"<b>{z[i, j]}</b>",
                    showarrow=False,
                    font=dict(color="white" if z[i, j] > z.max()/2 else "black", size=14)
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=False
        )
    )
    fig.update_layout(
        **THEME_LAYOUT,
        title="<b>Classification Confusion Matrix</b>",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        annotations=annotations,
        height=420
    )
    return fig


def plot_time_series_forecast(
    historical_years: List[int],
    historical_vals: List[float],
    future_years: List[int],
    forecast_vals: List[float],
    lower_ci: List[float],
    upper_ci: List[float]
) -> go.Figure:
    """
    Visualizes historical annual problem statement volume alongside
    ARIMA forecasts and 95% confidence interval uncertainty ribbons.
    """
    fig = go.Figure()

    # Historical line
    fig.add_trace(
        go.Scatter(
            x=historical_years,
            y=historical_vals,
            mode="lines+markers",
            name="Historical Observations",
            line=dict(color=COLOR_PRIMARY, width=3),
            marker=dict(size=8)
        )
    )

    # Connecting segment from last historical to first forecast
    bridge_x = [historical_years[-1]] + future_years
    bridge_y = [historical_vals[-1]] + forecast_vals
    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines+markers",
            name="ARIMA Forecast",
            line=dict(color=COLOR_ACCENT, width=3, dash="dot"),
            marker=dict(size=8, symbol="diamond")
        )
    )

    # 95% Confidence Interval Ribbon
    ci_x = future_years + future_years[::-1]
    ci_y = upper_ci + lower_ci[::-1]
    fig.add_trace(
        go.Scatter(
            x=ci_x,
            y=ci_y,
            fill="toself",
            fillcolor="rgba(16, 185, 129, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Confidence Interval"
        )
    )

    fig.update_layout(
        **THEME_LAYOUT,
        title="<b>Historical Problem Statement Volume & ARIMA Forecast (with 95% CI)</b>",
        xaxis_title="Calendar Year",
        yaxis_title="Problem Statement Count",
        height=480
    )
    return fig
