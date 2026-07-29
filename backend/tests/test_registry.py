"""Artifact round-trip and staleness detection.

Uses a stub forecaster rather than a real fit, so these stay fast.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from app.registry import (
    ARTIFACT_VERSION,
    _load_artifact,
    fingerprint_series,
    json_safe,
    save_model,
)


class _StubModel:
    def get_params(self):
        # nan mirrors what XGBRegressor really returns for `missing`
        return {"n_estimators": 500, "max_depth": 8, "missing": float("nan")}


class _StubForecaster:
    """Stands in for XGBoostForecaster; only .model.get_params() is used."""

    def __init__(self) -> None:
        self.model = _StubModel()


def _series(n: int = 100, offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "value": [float(i) + offset for i in range(n)],
        }
    )


# --- JSON safety ---

@pytest.mark.parametrize(
    "value, expected",
    [
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        (1.5, 1.5),
        (7, 7),
        (True, True),
        (None, None),
        ("x", "x"),
    ],
)
def test_json_safe_scalars(value, expected):
    assert json_safe(value) == expected


def test_json_safe_recurses_into_containers():
    out = json_safe({"a": [float("nan"), 1.0], "b": {"c": float("inf")}})
    assert out == {"a": [None, 1.0], "b": {"c": None}}


def test_json_safe_stringifies_exotic_values():
    class Weird:
        def __repr__(self):
            return "<weird>"

    assert json_safe(Weird()) == "<weird>"


def test_json_safe_output_is_strict_json_serialisable():
    # allow_nan=False is what FastAPI uses
    payload = json_safe({"missing": float("nan"), "depth": 8})
    json.dumps(payload, allow_nan=False)


# --- fingerprinting ---

def test_fingerprint_is_deterministic():
    assert fingerprint_series(_series()) == fingerprint_series(_series())


def test_fingerprint_changes_when_a_value_changes():
    a = _series()
    b = _series()
    b.loc[50, "value"] = 999.0
    assert fingerprint_series(a) != fingerprint_series(b)


def test_fingerprint_changes_when_length_changes():
    assert fingerprint_series(_series(100)) != fingerprint_series(_series(101))


def test_fingerprint_changes_when_dates_shift():
    a = _series()
    b = _series()
    b["date"] = b["date"] + pd.Timedelta(days=1)
    assert fingerprint_series(a) != fingerprint_series(b)


# --- artifact round-trip ---

def test_save_then_load_round_trips_with_provenance(tmp_path):
    df = _series()
    path = tmp_path / "model.joblib"
    save_model(_StubForecaster(), df, path=path, train_metrics={"holdout_mae": 12.5})

    assert path.exists()
    loaded = _load_artifact(path, fingerprint_series(df))
    assert loaded is not None
    model, info = loaded

    assert isinstance(model, _StubForecaster)
    assert info.source == "artifact"
    assert info.artifact_version == ARTIFACT_VERSION
    assert info.n_train_points == len(df)
    assert info.train_start == "2024-01-01"
    assert info.train_end == str(df["date"].iloc[-1].date())
    assert info.data_fingerprint == fingerprint_series(df)
    assert info.train_metrics == {"holdout_mae": 12.5}
    assert info.params["max_depth"] == 8
    assert info.params["missing"] is None
    json.dumps(info.params, allow_nan=False)


def test_save_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "model.joblib"
    save_model(_StubForecaster(), _series(), path=path)
    assert path.exists()


# --- staleness / validation ---

def test_load_rejects_an_artifact_trained_on_different_data(tmp_path):
    trained_on = _series(offset=0.0)
    now_serving = _series(offset=5.0)  # same shape, different values
    path = tmp_path / "model.joblib"
    save_model(_StubForecaster(), trained_on, path=path)

    with pytest.raises(ValueError, match="different data"):
        _load_artifact(path, fingerprint_series(now_serving))


def test_load_rejects_a_stale_artifact_version(tmp_path):
    import joblib

    path = tmp_path / "model.joblib"
    df = _series()
    save_model(_StubForecaster(), df, path=path)

    payload = joblib.load(path)
    payload["artifact_version"] = ARTIFACT_VERSION + 1
    joblib.dump(payload, path)

    with pytest.raises(ValueError, match="artifact_version"):
        _load_artifact(path, fingerprint_series(df))


def test_load_rejects_mismatched_feature_columns(tmp_path):
    import joblib

    path = tmp_path / "model.joblib"
    df = _series()
    save_model(_StubForecaster(), df, path=path)

    payload = joblib.load(path)
    payload["feature_columns"] = ["lag_1", "lag_2"]
    joblib.dump(payload, path)

    with pytest.raises(ValueError, match="feature columns"):
        _load_artifact(path, fingerprint_series(df))


def test_load_rejects_a_bare_pickled_model(tmp_path):
    """Pre-registry artifacts were a bare model, not a payload dict."""
    import joblib

    path = tmp_path / "model.joblib"
    joblib.dump(_StubForecaster(), path)

    with pytest.raises(ValueError, match="registry payload"):
        _load_artifact(path, "whatever")
