"""FastAPI service: zero-shot (Chronos-Bolt) vs trained (XGBoost) forecasting."""
from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .data import load_retail_series, load_series_from_records
from .forecaster import XGBoostForecaster, chronos_forecast, compare_models, ensemble_forecast
from .registry import fingerprint_series, get_bundled_forecaster, get_model_info
from .schemas import CompareRequest, ForecastRequest

app = FastAPI(
    title="Sales Forecasting API",
    description="Forecast daily online-retail sales with a zero-shot foundation model (Chronos-Bolt) and a trained XGBoost baseline.",
    version="1.1.0",
)

# The browser normally never calls this API cross-origin — Next's rewrites()
# proxy /api same-origin — so this only matters for direct callers (curl, the
# Swagger UI, a separate frontend). Comma-separated allowlist via env var;
# defaults to local dev origins only, never a wildcard.
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _resolve_df(series) -> tuple[pd.DataFrame, bool]:
    """Return (frame, is_bundled); a caller's own series needs its own fit."""
    if series:
        df = load_series_from_records([{"date": p.date, "value": p.value} for p in series])
        if len(df) < 60:
            raise HTTPException(400, "Provide at least 60 data points for reliable forecasting.")
        return df, False
    return load_retail_series(), True


def _xgb_forecast(df: pd.DataFrame, horizon: int, is_bundled: bool):
    if is_bundled:
        return get_bundled_forecaster().predict(horizon)
    return XGBoostForecaster().fit(df).predict(horizon)


@lru_cache(maxsize=16)
def _compare_bundled(horizon: int, folds: int, _fingerprint: str) -> dict:
    """Memoised: a backtest refits XGBoost and reruns Chronos once per fold.

    The fingerprint is in the cache key so a regenerated CSV invalidates it.
    """
    return compare_models(load_retail_series(), horizon, folds=folds)


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.get("/api/model-info", tags=["ops"])
def model_info():
    """Which XGBoost model is serving: a loaded artifact, or one fitted here."""
    return asdict(get_model_info())


@app.get("/api/demo-series", tags=["data"])
def demo_series(n_days: int = 730):
    """Recent daily sales history for the frontend to plot."""
    df = load_retail_series().tail(n_days)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in df["date"]],
        "values": [round(v, 4) for v in df["value"]],
    }


@app.post("/api/forecast", tags=["forecast"])
def forecast(req: ForecastRequest):
    """Forecast future values (no ground truth) with one, both, or the ensemble."""
    df, is_bundled = _resolve_df(req.series)
    last = df["date"].iloc[-1]
    future_dates = [(last + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(req.horizon)]

    # Compute whichever base models the request needs (ensemble needs both).
    chronos = None
    if req.model in ("chronos", "both", "ensemble"):
        try:
            chronos = chronos_forecast(df["value"].tolist(), req.horizon)
        except Exception as exc:  # noqa: BLE001 - Chronos is best-effort
            if req.model == "chronos":
                raise HTTPException(503, "Chronos model is unavailable in this environment.") from exc

    xgb = None
    if req.model in ("xgboost", "both", "ensemble"):
        xgb = _xgb_forecast(df, req.horizon, is_bundled)

    out: dict = {"horizon": req.horizon, "future_dates": future_dates, "models": {}}
    if req.model in ("chronos", "both") and chronos is not None:
        out["models"]["chronos-bolt"] = chronos.to_dict()
    if req.model in ("xgboost", "both") and xgb is not None:
        out["models"]["xgboost"] = xgb.to_dict()
    if req.model == "ensemble":
        if chronos is not None and xgb is not None:
            out["models"]["ensemble"] = ensemble_forecast(chronos, xgb).to_dict()
        elif xgb is not None:  # Chronos unavailable -> fall back to XGBoost
            out["models"]["xgboost"] = xgb.to_dict()
    return out


@app.post("/api/compare", tags=["forecast"])
def compare(req: CompareRequest):
    """Backtest: hold out recent windows and score both models against the truth."""
    df, is_bundled = _resolve_df(req.series)
    if len(df) <= req.horizon + 40:
        raise HTTPException(400, "Series too short for this horizon; add more history or reduce the horizon.")
    if is_bundled:
        return _compare_bundled(req.horizon, req.folds, fingerprint_series(df))
    return compare_models(df, req.horizon, folds=req.folds)
