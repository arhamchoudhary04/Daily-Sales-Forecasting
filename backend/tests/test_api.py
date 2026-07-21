"""API tests. Chronos is skipped by default so CI stays fast and offline."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


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


def test_forecast_rejects_bad_horizon():
    r = client.post("/api/forecast", json={"horizon": 999, "model": "xgboost"})
    assert r.status_code == 422  # validation error


@pytest.mark.skipif(
    os.getenv("RUN_CHRONOS_TESTS") != "1",
    reason="Downloads the Chronos model; set RUN_CHRONOS_TESTS=1 to enable.",
)
def test_forecast_chronos():
    r = client.post("/api/forecast", json={"horizon": 7, "model": "chronos"})
    assert r.status_code == 200
    assert len(r.json()["models"]["chronos-bolt"]["median"]) == 7
