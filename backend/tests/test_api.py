"""API tests. Chronos is skipped by default so CI stays fast and offline."""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

POINT_METRICS = ("mae", "rmse", "mape", "smape")
INTERVAL_METRICS = ("coverage", "pinball", "interval_width")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_demo_series():
    r = client.get("/api/demo-series?n_days=120")
    assert r.status_code == 200
    body = r.json()
    assert len(body["dates"]) == 120
    assert len(body["values"]) == 120


def test_forecast_xgboost():
    r = client.post("/api/forecast", json={"horizon": 7, "model": "xgboost"})
    assert r.status_code == 200
    body = r.json()
    assert body["horizon"] == 7
    assert len(body["future_dates"]) == 7
    assert len(body["models"]["xgboost"]["median"]) == 7


def test_forecast_returns_an_ordered_band():
    r = client.post("/api/forecast", json={"horizon": 7, "model": "xgboost"})
    f = r.json()["models"]["xgboost"]
    for lo, mid, hi in zip(f["lower"], f["median"], f["upper"]):
        assert lo <= mid <= hi


def test_forecast_rejects_bad_horizon():
    r = client.post("/api/forecast", json={"horizon": 999, "model": "xgboost"})
    assert r.status_code == 422  # validation error


def test_forecast_rejects_unknown_model():
    r = client.post("/api/forecast", json={"horizon": 7, "model": "prophet"})
    assert r.status_code == 422


def test_forecast_rejects_too_short_a_custom_series():
    series = [{"date": f"2024-01-{d:02d}", "value": 10.0} for d in range(1, 11)]
    r = client.post("/api/forecast", json={"horizon": 7, "model": "xgboost", "series": series})
    assert r.status_code == 400
    assert "60" in r.json()["detail"]


# --- model registry ---

def test_model_info_reports_provenance():
    r = client.get("/api/model-info")
    assert r.status_code == 200
    info = r.json()
    assert info["source"] in ("artifact", "fit-on-demand")
    assert info["n_train_points"] > 0
    assert info["data_fingerprint"]
    assert "lag_7" in info["feature_columns"]


def test_model_info_is_strictly_json_serialisable():
    """Regression: `missing: nan` in the params used to 500 this endpoint."""
    r = client.get("/api/model-info")
    assert r.status_code == 200
    json.dumps(r.json(), allow_nan=False)
    assert r.json()["params"]["missing"] is None


def test_repeated_forecasts_reuse_the_same_cached_model():
    before = client.get("/api/model-info").json()
    client.post("/api/forecast", json={"horizon": 7, "model": "xgboost"})
    client.post("/api/forecast", json={"horizon": 14, "model": "xgboost"})
    after = client.get("/api/model-info").json()
    # trained_at is stamped once, when the model is resolved
    assert before["trained_at"] == after["trained_at"]


def test_bundled_forecast_is_deterministic():
    a = client.post("/api/forecast", json={"horizon": 7, "model": "xgboost"}).json()
    b = client.post("/api/forecast", json={"horizon": 7, "model": "xgboost"}).json()
    assert a["models"]["xgboost"]["median"] == b["models"]["xgboost"]["median"]


# --- backtest ---

def test_compare_reports_point_and_interval_metrics():
    r = client.post("/api/compare", json={"horizon": 14, "folds": 2})
    assert r.status_code == 200
    body = r.json()

    assert body["folds"] == 2
    assert body["nominal_coverage"] == 80.0
    assert len(body["dates"]) == 14
    assert len(body["actual"]) == 14

    metrics = body["models"]["xgboost"]["metrics"]
    for key in POINT_METRICS + INTERVAL_METRICS:
        assert key in metrics, f"missing metric {key}"
    assert 0.0 <= metrics["coverage"] <= 100.0
    assert metrics["interval_width"] >= 0.0


def test_compare_rejects_a_horizon_the_series_cannot_support():
    series = [{"date": f"2024-{m:02d}-{d:02d}", "value": 10.0 + d}
              for m in (1, 2, 3) for d in range(1, 26)]
    r = client.post("/api/compare", json={"horizon": 90, "folds": 2, "series": series})
    assert r.status_code == 400


def test_compare_rejects_out_of_range_folds():
    assert client.post("/api/compare", json={"horizon": 14, "folds": 0}).status_code == 422
    assert client.post("/api/compare", json={"horizon": 14, "folds": 99}).status_code == 422


@pytest.mark.skipif(
    os.getenv("RUN_CHRONOS_TESTS") != "1",
    reason="Downloads the Chronos model; set RUN_CHRONOS_TESTS=1 to enable.",
)
def test_forecast_chronos():
    r = client.post("/api/forecast", json={"horizon": 7, "model": "chronos"})
    assert r.status_code == 200
    assert len(r.json()["models"]["chronos-bolt"]["median"]) == 7


@pytest.mark.skipif(
    os.getenv("RUN_CHRONOS_TESTS") != "1",
    reason="Downloads the Chronos model; set RUN_CHRONOS_TESTS=1 to enable.",
)
def test_compare_includes_the_ensemble_when_chronos_is_available():
    r = client.post("/api/compare", json={"horizon": 14, "folds": 2})
    models = r.json()["models"]
    assert {"chronos-bolt", "xgboost", "ensemble"} <= set(models)
