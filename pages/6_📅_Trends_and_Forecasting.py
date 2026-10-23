"""
pages/6_📅_Trends_and_Forecasting.py
Time Series Analysis, Historical Trends, and ARIMA Forecasting.
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
from src.time_series_analysis import (
    compute_yearly_trends, compute_domain_trends_by_year, fit_arima_forecast
)
from src.visualizations import plot_time_series_forecast

st.set_page_config(page_title="Trends & Forecasting — Hackathon Insights", page_icon="📅", layout="wide")

st.markdown("# 📅 Trends & Time-Series Forecasting")
st.markdown(
    "Analyze multi-year historical trends in hackathon problem volume, track the surge in Artificial Intelligence adoption, "
    "and evaluate ARIMA autoregressive time-series forecasts."
)

raw_df = load_raw_data()
clean_df = load_cleaned_data() if check_dataset_exists(is_cleaned=True) else raw_df
active_df = clean_df

yearly_df = compute_yearly_trends(active_df)

tab_hist, tab_arima, tab_domain_evo, tab_stl_notes = st.tabs([
    "📈 Historical Trends & AI Adoption",
    "🔮 ARIMA Forecasting Model",
    "🌐 Domain Evolution by Year",
    "📚 STL & Time-Series Methodology"
])

# ------------------------------------------------------------------------------
# TAB 1: HISTORICAL TRENDS
# ------------------------------------------------------------------------------
with tab_hist:
    st.markdown("### 📊 Annual Growth & Technology Adoption (2019 – 2024)")
    st.dataframe(yearly_df, use_container_width=True)

    c_t1, c_t2 = st.columns(2)
    with c_t1:
        fig_vol = px.bar(
            yearly_df,
            x="year",
            y="total_problems",
            text="total_problems",
            title="<b>Annual Problem Statement Volume</b>",
            color="total_problems",
            color_continuous_scale="Blues"
        )
        fig_vol.update_traces(textposition="outside")
        fig_vol.update_layout(xaxis_title="Year", yaxis_title="Total Problem Count", height=420)
        st.plotly_chart(fig_vol, use_container_width=True)

    with c_t2:
        fig_ai = px.line(
            yearly_df,
            x="year",
            y="ai_adoption_pct",
            markers=True,
            title="<b>AI / ML Requirement Adoption Trajectory (%)</b>",
            color_discrete_sequence=["#10B981"]
        )
        fig_ai.update_layout(xaxis_title="Year", yaxis_title="AI Required (%)", height=420)
        st.plotly_chart(fig_ai, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: ARIMA FORECASTING
# ------------------------------------------------------------------------------
with tab_arima:
    st.markdown("### 🔮 Autoregressive Integrated Moving Average (ARIMA) Model — Unit IV")
    st.markdown("""
    An $\\text{ARIMA}(p, d, q)$ model combines autoregressive lags ($p$), degree of differencing ($d$), and moving average errors ($q$):
    $$(1 - \\sum_{i=1}^p \\phi_i L^i) (1 - L)^d Y_t = \\delta + (1 + \\sum_{j=1}^q \\theta_j L^j) \\varepsilon_t$$
    """)

    forecast_steps = st.slider("Select Forecast Horizon (Years Ahead):", min_value=1, max_value=3, value=2)
    arima_res = fit_arima_forecast(yearly_df, target_col="total_problems", steps_ahead=forecast_steps, order=(1, 1, 0))

    if "error" not in arima_res:
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Model Architecture", arima_res["model_name"])
        ac2.metric("Backtest MAE", f"{arima_res['in_sample_metrics']['MAE']} problems")
        ac3.metric("Backtest RMSE", f"{arima_res['in_sample_metrics']['RMSE']}")
        ac4.metric("Backtest MAPE", f"{arima_res['in_sample_metrics']['MAPE_pct']}%")

        st.plotly_chart(
            plot_time_series_forecast(
                arima_res["historical_years"],
                arima_res["historical_values"],
                arima_res["future_years"],
                arima_res["forecast_values"],
                arima_res["lower_ci"],
                arima_res["upper_ci"]
            ),
            use_container_width=True
        )

        st.markdown("##### 📋 Projected Volume with 95% Confidence Intervals")
        st.dataframe(arima_res["forecast_table"], use_container_width=True)
    else:
        st.error(arima_res["error"])


# ------------------------------------------------------------------------------
# TAB 3: DOMAIN EVOLUTION
# ------------------------------------------------------------------------------
with tab_domain_evo:
    st.markdown("### 🌐 Thematic Domain Evolution across Years")
    domain_by_yr = compute_domain_trends_by_year(active_df)
    
    fig_dom_yr = px.bar(
        domain_by_yr,
        title="<b>Domain Composition Across Calendar Years</b>",
        barmode="stack",
        height=480
    )
    fig_dom_yr.update_layout(xaxis_title="Year", yaxis_title="Number of Problems")
    st.plotly_chart(fig_dom_yr, use_container_width=True)

    st.markdown("##### Domain Count Contingency Table")
    st.dataframe(domain_by_yr, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: STL METHODOLOGY
# ------------------------------------------------------------------------------
with tab_stl_notes:
    st.markdown("### 📚 STL Approach & Forecast Accuracy Principles (Unit IV)")
    st.markdown("""
    #### 1. Seasonal and Trend Decomposition using Loess (STL)
    $$\\text{Series } Y_t = T_t + S_t + R_t \\quad (\\text{Additive}) \\quad \\text{or} \\quad Y_t = T_t \\times S_t \\times R_t \\quad (\\text{Multiplicative})$$
    
    - **Theoretical Foundation**: STL decomposes a time series into a smooth trend component ($T_t$), a recurring seasonal cycle ($S_t$), and a stochastic residual remainder ($R_t$).
    - **Data Sufficiency Constraint**: Mathematical STL requires at least **two full periodic seasonal cycles** ($m \\ge 2$). In annual datasets (1 observation per year), sub-annual seasonality cannot be isolated because there are no quarterly or monthly sub-periods.
    - **Honest Academic Conclusion**: Because hackathons typically occur in annual competition cycles without fine-grained quarterly dates in the standard problem schema, applying STL decomposition directly would be statistically unjustified. We therefore utilize the verified ARIMA stochastic trend model.

    #### 2. Forecast Accuracy Measures
    - **Mean Absolute Error (MAE)**: $\\frac{1}{n}\\sum |y_t - \\hat{y}_t|$ (linear penalty, easily interpretable in original count units).
    - **Root Mean Squared Error (RMSE)**: $\\sqrt{\\frac{1}{n}\\sum (y_t - \\hat{y}_t)^2}$ (penalizes large outlier forecast errors).
    - **Mean Absolute Percentage Error (MAPE)**: $\\frac{100\\%}{n}\\sum |\\frac{y_t - \\hat{y}_t}{y_t}|$ (relative scale-independent error).
    """)
