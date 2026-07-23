"""FastAPI service: zero-shot (Chronos-Bolt) vs trained (XGBoost) forecasting."""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .data import load_retail_series, load_series_from_records
from .forecaster import XGBoostForecaster, chronos_forecast, compare_models
from .schemas import CompareRequest, ForecastRequest

app = FastAPI(
    title="Sales Forecasting API",
    description="Forecast daily online-retail sales with a zero-shot foundation model (Chronos-Bolt) and a trained XGBoost baseline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_df(series) -> pd.DataFrame:
    """Use the caller's series if given, else the built-in retail sales series."""
    if series:
        df = load_series_from_records([{"date": p.date, "value": p.value} for p in series])
        if len(df) < 60:
            raise HTTPException(400, "Provide at least 60 data points for reliable forecasting.")
        return df
    return load_retail_series()


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


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
    """Forecast future values (no ground truth) with one or both models."""
    df = _resolve_df(req.series)
    last = df["date"].iloc[-1]
    future_dates = [(last + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(req.horizon)]

    out: dict = {"horizon": req.horizon, "future_dates": future_dates, "models": {}}
    if req.model in ("chronos", "both"):
        try:
            out["models"]["chronos-bolt"] = chronos_forecast(df["value"].tolist(), req.horizon).to_dict()
        except Exception as exc:  # noqa: BLE001 - Chronos is best-effort
            if req.model == "chronos":
                raise HTTPException(503, "Chronos model is unavailable in this environment.") from exc
    if req.model in ("xgboost", "both"):
        out["models"]["xgboost"] = XGBoostForecaster().fit(df).predict(req.horizon).to_dict()
    return out


@app.post("/api/compare", tags=["forecast"])
def compare(req: CompareRequest):
    """Backtest: hold out recent points and score both models against the truth."""
    df = _resolve_df(req.series)
    if len(df) <= req.horizon + 40:
        raise HTTPException(400, "Series too short for this horizon; add more history or reduce the horizon.")
    return compare_models(df, req.horizon)
