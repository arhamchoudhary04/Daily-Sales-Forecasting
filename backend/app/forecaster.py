"""Forecasting engines: Chronos-Bolt (zero-shot) and XGBoost. Both return a
point forecast plus an interval via ForecastResult."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS, make_supervised_features


@dataclass
class ForecastResult:
    model: str
    horizon: int
    median: list[float]
    lower: list[float]
    upper: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


# --- metrics ---

def evaluate_forecast(actual, predicted) -> dict[str, float]:
    """Point-forecast error metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = actual - predicted
    denom = np.where(np.abs(actual) < 1e-8, 1e-8, np.abs(actual))
    return {
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "mape": round(float(np.mean(np.abs(err) / denom) * 100), 4),
        "smape": round(float(np.mean(2 * np.abs(err) / (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100), 4),
    }


# --- chronos (zero-shot) ---

CHRONOS_MODEL = os.getenv("CHRONOS_MODEL", "amazon/chronos-bolt-small")


@lru_cache(maxsize=1)
def _get_chronos_pipeline():
    """Load the Chronos pipeline once, on CPU."""
    import torch
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(
        CHRONOS_MODEL, device_map="cpu", torch_dtype=torch.float32
    )


def chronos_forecast(values, horizon: int) -> ForecastResult:
    """Zero-shot forecast, no training."""
    import torch

    pipeline = _get_chronos_pipeline()
    context = torch.tensor(np.asarray(values, dtype="float32"))
    # 2.x renamed context -> inputs
    quantiles, _mean = pipeline.predict_quantiles(
        inputs=context, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
    )
    q = quantiles[0].cpu().numpy()  # -> [horizon, 3]
    return ForecastResult(
        model="chronos-bolt",
        horizon=horizon,
        median=np.round(q[:, 1], 4).tolist(),
        lower=np.round(q[:, 0], 4).tolist(),
        upper=np.round(q[:, 2], 4).tolist(),
    )


# --- xgboost ---

class XGBoostForecaster:
    """XGBoost baseline with recursive multi-step forecasting."""

    def __init__(self, **params):
        from xgboost import XGBRegressor

        defaults = dict(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
        )
        defaults.update(params)
        self.model = XGBRegressor(**defaults)
        self._history: pd.DataFrame | None = None
        self._residual_std: float = 0.0

    def fit(self, df: pd.DataFrame) -> "XGBoostForecaster":
        feats = make_supervised_features(df)
        X, y = feats[FEATURE_COLUMNS], feats["value"]
        self.model.fit(X, y)
        self._residual_std = float(np.std(y.to_numpy() - self.model.predict(X)))
        self._history = df[["date", "value"]].reset_index(drop=True)
        return self

    def _feature_row(self, values: list[float], next_date: pd.Timestamp) -> pd.DataFrame:
        s = pd.Series(values)
        feat = {
            "lag_1": s.iloc[-1], "lag_2": s.iloc[-2], "lag_3": s.iloc[-3],
            "lag_7": s.iloc[-7], "lag_14": s.iloc[-14], "lag_28": s.iloc[-28],
            "roll_mean_7": s.iloc[-7:].mean(),
            "roll_mean_28": s.iloc[-28:].mean(),
            "roll_std_7": s.iloc[-7:].std(),
            "dayofweek": next_date.dayofweek, "day": next_date.day,
            "month": next_date.month, "dayofyear": next_date.dayofyear,
        }
        return pd.DataFrame([feat])[FEATURE_COLUMNS]

    def predict(self, horizon: int) -> ForecastResult:
        if self._history is None:
            raise RuntimeError("Call fit() before predict().")
        values = self._history["value"].tolist()
        dates = list(self._history["date"])

        preds: list[float] = []
        for _ in range(horizon):
            next_date = dates[-1] + pd.Timedelta(days=1)
            yhat = float(self.model.predict(self._feature_row(values, next_date))[0])
            preds.append(yhat)
            values.append(yhat)
            dates.append(next_date)

        band = 1.2816 * self._residual_std  # ~80% prediction interval
        return ForecastResult(
            model="xgboost",
            horizon=horizon,
            median=[round(p, 4) for p in preds],
            lower=[round(p - band, 4) for p in preds],
            upper=[round(p + band, 4) for p in preds],
        )


# --- backtest ---

def compare_models(df: pd.DataFrame, horizon: int) -> dict:
    """Hold out the last `horizon` points, forecast with both, and score."""
    train, test = df.iloc[:-horizon], df.iloc[-horizon:]
    actual = test["value"].tolist()

    chronos = chronos_forecast(train["value"].tolist(), horizon)
    xgb = XGBoostForecaster().fit(train).predict(horizon)

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in test["date"]],
        "actual": [round(v, 4) for v in actual],
        "models": {
            "chronos-bolt": {"forecast": chronos.to_dict(), "metrics": evaluate_forecast(actual, chronos.median)},
            "xgboost": {"forecast": xgb.to_dict(), "metrics": evaluate_forecast(actual, xgb.median)},
        },
    }
