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

# The forecasters emit these quantile levels as (lower, median, upper).
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
# Nominal coverage of the [lower, upper] band: 0.9 - 0.1.
NOMINAL_COVERAGE = 80.0


def evaluate_forecast(actual, predicted) -> dict[str, float]:
    """Point-forecast error metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = actual - predicted
    # MAPE is undefined when actual == 0 (closed days), so average it over the
    # non-zero points only; MAE/RMSE/sMAPE use every point.
    nz = np.abs(actual) > 1e-8
    mape = float(np.mean(np.abs(err[nz] / actual[nz])) * 100) if nz.any() else 0.0
    return {
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "mape": round(mape, 4),
        "smape": round(float(np.mean(2 * np.abs(err) / (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100), 4),
    }


def pinball_loss(actual, predicted, quantile: float) -> float:
    """Asymmetric quantile loss. Lower is better."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = actual - predicted
    return float(np.mean(np.maximum(quantile * err, (quantile - 1) * err)))


def evaluate_interval(actual, lower, median, upper) -> dict[str, float]:
    """Score the forecast band, not just the median.

    Coverage is best *closest to* NOMINAL_COVERAGE rather than highest, since a
    band of +/- infinity would cover 100%. Pinball loss is a proper scoring rule,
    so it's the metric that can't be gamed by widening the band.
    """
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    inside = (actual >= lower) & (actual <= upper)
    losses = [
        pinball_loss(actual, q, level)
        for q, level in zip((lower, median, upper), QUANTILE_LEVELS)
    ]
    return {
        "coverage": round(float(np.mean(inside) * 100), 4),
        "pinball": round(float(np.mean(losses)), 4),
        "interval_width": round(float(np.mean(upper - lower)), 4),
    }


def score_forecast(actual, result: "ForecastResult") -> dict[str, float]:
    return {
        **evaluate_forecast(actual, result.median),
        **evaluate_interval(actual, result.lower, result.median, result.upper),
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
    """XGBoost baseline with recursive multi-step forecasting.

    Hyperparameters were tuned on a rolling-window backtest of the retail
    series (deeper trees + L2 regularisation beat the shallower default here;
    a log-target transform was tried and hurt, so it isn't used).
    """

    def __init__(self, **params):
        from xgboost import XGBRegressor

        defaults = dict(
            n_estimators=500, max_depth=8, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
            reg_lambda=2.0, random_state=42,
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

        # ~80% interval assuming Gaussian residuals of constant width. Chronos
        # predicts its quantiles per step, so it calibrates far better; the
        # coverage column in the backtest shows the gap.
        band = 1.2816 * self._residual_std
        return ForecastResult(
            model="xgboost",
            horizon=horizon,
            median=[round(p, 4) for p in preds],
            lower=[round(p - band, 4) for p in preds],
            upper=[round(p + band, 4) for p in preds],
        )


# --- backtest ---

def ensemble_forecast(a: ForecastResult, b: ForecastResult) -> ForecastResult:
    """Equal-weight average of two forecasts — usually beats either alone,
    because the two models make different mistakes."""
    mean = lambda xs, ys: [round((x + y) / 2, 4) for x, y in zip(xs, ys)]
    return ForecastResult(
        model="ensemble",
        horizon=a.horizon,
        median=mean(a.median, b.median),
        lower=mean(a.lower, b.lower),
        upper=mean(a.upper, b.upper),
    )


MIN_TRAIN_POINTS = 60


def backtest_windows(
    n_points: int,
    horizon: int,
    folds: int,
    step: int | None = None,
    min_train: int = MIN_TRAIN_POINTS,
) -> list[tuple[int, int]]:
    """Rolling-origin windows as (start, end) index pairs, most recent first.

    Each window tests df[start:end] after training on df[:start]. `step` defaults
    to `horizon`, i.e. non-overlapping. Short windows are dropped, so the result
    can be shorter than `folds`.
    """
    step = horizon if step is None else step
    windows: list[tuple[int, int]] = []
    for i in range(folds):
        end = n_points - i * step
        start = end - horizon
        if start < min_train:
            break
        windows.append((start, end))
    return windows


def compare_models(df: pd.DataFrame, horizon: int, folds: int = 4) -> dict:
    """Rolling-window backtest: score each model over the last `folds`
    non-overlapping windows and average, and return the most-recent window
    (actual + forecast) for plotting.

    A single last-window holdout is misleading here — the series ends on the
    volatile pre-Christmas peak — so averaging several windows gives a fairer
    read. Chronos is optional: if its model can't load, only XGBoost is scored
    (and the ensemble is skipped, since it needs both).
    """
    scores: dict[str, list[dict]] = {"chronos-bolt": [], "xgboost": [], "ensemble": []}
    recent: dict = {}
    recent_test = None

    windows = backtest_windows(len(df), horizon, folds)
    for i, (start, end) in enumerate(windows):
        train, test = df.iloc[:start], df.iloc[start:end]
        actual = test["value"].tolist()

        # Each fold trains on its own prefix, so these fits can't come from the
        # registry cache.
        xgb = XGBoostForecaster().fit(train).predict(horizon)
        scores["xgboost"].append(score_forecast(actual, xgb))

        chronos = None
        try:
            chronos = chronos_forecast(train["value"].tolist(), horizon)
            scores["chronos-bolt"].append(score_forecast(actual, chronos))
            ens = ensemble_forecast(chronos, xgb)
            scores["ensemble"].append(score_forecast(actual, ens))
        except Exception:  # noqa: BLE001 - Chronos is best-effort
            pass

        if i == 0:  # most recent window -> what the chart shows
            recent_test = test
            recent["xgboost"] = xgb
            if chronos is not None:
                recent["chronos-bolt"] = chronos
                recent["ensemble"] = ensemble_forecast(chronos, xgb)

    used = len(windows)
    if recent_test is None:
        raise ValueError("Series too short to build a single backtest window.")

    def _avg(rows: list[dict]) -> dict:
        return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}

    models: dict = {}
    if scores["chronos-bolt"] and "chronos-bolt" in recent:
        models["chronos-bolt"] = {"forecast": recent["chronos-bolt"].to_dict(), "metrics": _avg(scores["chronos-bolt"])}
    models["xgboost"] = {"forecast": recent["xgboost"].to_dict(), "metrics": _avg(scores["xgboost"])}
    if scores["ensemble"] and "ensemble" in recent:
        models["ensemble"] = {"forecast": recent["ensemble"].to_dict(), "metrics": _avg(scores["ensemble"])}

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in recent_test["date"]],
        "actual": [round(v, 4) for v in recent_test["value"].tolist()],
        "folds": used,
        # So the UI can judge coverage against its target instead of hardcoding it.
        "nominal_coverage": NOMINAL_COVERAGE,
        "models": models,
    }
