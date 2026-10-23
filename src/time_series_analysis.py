"""
src/time_series_analysis.py
Module for historical trend analysis, Year-over-Year (YoY) growth,
ARIMA forecasting, forecast accuracy evaluation (RMSE, MAPE), and STL methodology.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def compute_yearly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates problem statements by calendar year, computing total counts,
    AI-required counts, Hardware-required counts, and Year-over-Year percentage changes.
    """
    if df is None or "year" not in df.columns:
        return pd.DataFrame()

    yearly = df.groupby("year").agg(
        total_problems=("problem_id", "count"),
        ai_problems=("ai_required", lambda x: int((x == 1).sum())),
        hardware_problems=("hardware_required", lambda x: int((x == 1).sum())),
        avg_solution_days=("estimated_solution_days", "mean"),
        avg_innovation_score=("innovation_score", "mean")
    ).reset_index()

    yearly["ai_adoption_pct"] = (yearly["ai_problems"] / yearly["total_problems"] * 100.0).round(1)
    yearly["yoy_growth_pct"] = yearly["total_problems"].pct_change().mul(100.0).round(1).fillna(0.0)
    yearly["avg_solution_days"] = yearly["avg_solution_days"].round(1)
    yearly["avg_innovation_score"] = yearly["avg_innovation_score"].round(2)

    return yearly.sort_values(by="year").reset_index(drop=True)


def compute_domain_trends_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Computes cross-tabulation of problem counts by domain across calendar years."""
    if df is None or "year" not in df.columns or "domain" not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(df["year"], df["domain"])


def fit_arima_forecast(
    yearly_df: pd.DataFrame,
    target_col: str = "total_problems",
    steps_ahead: int = 2,
    order: Tuple[int, int, int] = (1, 1, 0)
) -> Dict[str, Any]:
    """
    Fits an Autoregressive Integrated Moving Average (ARIMA) model to historical annual counts
    and generates multi-step forward predictions with 95% confidence intervals.
    """
    if yearly_df is None or len(yearly_df) < 4:
        return {
            "error": "Insufficient time-series observations (minimum 4 periods required for ARIMA)."
        }

    series = yearly_df[target_col].astype(float).values
    years = yearly_df["year"].astype(int).values
    last_year = int(years[-1])

    # Fit ARIMA model
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast_res = model_fit.get_forecast(steps=steps_ahead)
        forecast_vals = np.round(forecast_res.predicted_mean, 1)
        conf_int = np.round(forecast_res.conf_int(alpha=0.05), 1)

        future_years = [last_year + i for i in range(1, steps_ahead + 1)]

        # Backtesting: In-sample predictions & accuracy
        in_sample_pred = model_fit.predict(start=1, end=len(series)-1)
        actual_in_sample = series[1:]
        mae = float(np.mean(np.abs(actual_in_sample - in_sample_pred)))
        rmse = float(np.sqrt(np.mean((actual_in_sample - in_sample_pred) ** 2)))
        mape = float(np.mean(np.abs((actual_in_sample - in_sample_pred) / actual_in_sample)) * 100.0)

        forecast_table = pd.DataFrame({
            "Forecast Year": future_years,
            "Predicted Count": forecast_vals,
            "Lower 95% CI": conf_int[:, 0],
            "Upper 95% CI": conf_int[:, 1]
        })

        return {
            "model_name": f"ARIMA{order}",
            "historical_years": years.tolist(),
            "historical_values": series.tolist(),
            "future_years": future_years,
            "forecast_values": forecast_vals.tolist(),
            "lower_ci": conf_int[:, 0].tolist(),
            "upper_ci": conf_int[:, 1].tolist(),
            "forecast_table": forecast_table,
            "in_sample_metrics": {
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "MAPE_pct": round(mape, 2)
            },
            "aic": round(float(model_fit.aic), 2),
            "bic": round(float(model_fit.bic), 2),
            "methodology_notes": [
                f"ARIMA{order} fitted on {len(series)} annual observations ({years[0]} to {years[-1]}).",
                "Model uses first-order differencing (d=1) to achieve mean stationarity.",
                "Confidence intervals reflect 95% forecast uncertainty bands.",
                "Limitations: With annual frequency, sub-annual seasonal patterns (STL decomposition) cannot be mathematically estimated without quarterly/monthly timestamps."
            ]
        }
    except Exception as e:
        return {"error": f"ARIMA fitting exception: {str(e)}"}
