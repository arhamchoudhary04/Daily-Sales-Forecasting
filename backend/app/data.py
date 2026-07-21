"""Synthetic series generation and feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

# lag + calendar features for the xgboost baseline
FEATURE_COLUMNS = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_28", "roll_std_7",
    "dayofweek", "day", "month", "dayofyear",
]


def generate_synthetic_series(
    n_days: int = 730,
    start_date: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Daily series with trend + weekly/yearly seasonality + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start_date, periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = 0.05 * t
    weekly = 8.0 * np.sin(2 * np.pi * t / 7)
    yearly = 20.0 * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0, 3.0, n_days)

    value = 100.0 + trend + weekly + yearly + noise
    return pd.DataFrame({"date": idx, "value": value})


def load_series_from_records(records: list[dict]) -> pd.DataFrame:
    """Build a clean, sorted (date, value) frame from a list of dicts."""
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
    return df[["date", "value"]]


def make_supervised_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a (date, value) series into a supervised table for tree models."""
    out = df.copy()
    for lag in (1, 2, 3, 7, 14, 28):
        out[f"lag_{lag}"] = out["value"].shift(lag)
    out["roll_mean_7"] = out["value"].shift(1).rolling(7).mean()
    out["roll_mean_28"] = out["value"].shift(1).rolling(28).mean()
    out["roll_std_7"] = out["value"].shift(1).rolling(7).std()
    out["dayofweek"] = out["date"].dt.dayofweek
    out["day"] = out["date"].dt.day
    out["month"] = out["date"].dt.month
    out["dayofyear"] = out["date"].dt.dayofyear
    return out.dropna().reset_index(drop=True)
