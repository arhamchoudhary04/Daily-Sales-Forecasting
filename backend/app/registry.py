"""Loads the trained XGBoost artifact so the API doesn't refit per request.

Falls back to fitting once, in memory, if there's no usable artifact. An artifact
trained on different data than the CSV on disk is treated as unusable, so a stale
model can't quietly serve wrong forecasts.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .data import FEATURE_COLUMNS, load_retail_series

ARTIFACT_VERSION = 1
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "xgb_model.joblib"


def model_path() -> Path:
    override = os.getenv("MODEL_PATH")
    return Path(override) if override else DEFAULT_MODEL_PATH


def fingerprint_series(df: pd.DataFrame) -> str:
    """Content hash of a (date, value) series, for staleness checks."""
    values = df["value"].to_numpy(dtype="float64")
    digest = hashlib.sha256()
    digest.update(f"{len(df)}|".encode())
    digest.update(f"{df['date'].iloc[0]}|{df['date'].iloc[-1]}|".encode())
    digest.update(values.round(4).tobytes())
    return digest.hexdigest()[:16]


def json_safe(value: Any) -> Any:
    """Strip values JSON can't encode; XGBoost reports `missing: nan`."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return str(value)


@dataclass
class ModelInfo:
    source: str  # "artifact" | "fit-on-demand"
    artifact_version: int | None = None
    model_path: str | None = None
    trained_at: str | None = None
    n_train_points: int | None = None
    train_start: str | None = None
    train_end: str | None = None
    data_fingerprint: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    train_metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_model(forecaster, df: pd.DataFrame, path: Path | None = None,
               train_metrics: dict[str, float] | None = None) -> Path:
    """Persist a fitted forecaster with the provenance needed to validate it."""
    import joblib

    out = Path(path) if path else model_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "artifact_version": ARTIFACT_VERSION,
        "model": forecaster,
        "trained_at": _now(),
        "n_train_points": int(len(df)),
        "train_start": str(df["date"].iloc[0].date()),
        "train_end": str(df["date"].iloc[-1].date()),
        "data_fingerprint": fingerprint_series(df),
        "params": json_safe(forecaster.model.get_params()),
        "feature_columns": list(FEATURE_COLUMNS),
        "train_metrics": json_safe(train_metrics or {}),
    }, out)
    return out


def _load_artifact(path: Path, expected_fingerprint: str) -> tuple[Any, ModelInfo]:
    """Load an artifact, raising if it can't be trusted for this data."""
    import joblib

    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("artifact is not a registry payload")
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"artifact_version {payload.get('artifact_version')} != {ARTIFACT_VERSION}"
        )
    if payload.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("artifact feature columns do not match the current code")
    if payload.get("data_fingerprint") != expected_fingerprint:
        raise ValueError(
            "artifact was trained on different data than the bundled series "
            f"({payload.get('data_fingerprint')} != {expected_fingerprint})"
        )

    info = ModelInfo(
        source="artifact",
        artifact_version=payload.get("artifact_version"),
        model_path=str(path),
        trained_at=payload.get("trained_at"),
        n_train_points=payload.get("n_train_points"),
        train_start=payload.get("train_start"),
        train_end=payload.get("train_end"),
        data_fingerprint=payload.get("data_fingerprint"),
        params=json_safe(payload.get("params", {})),
        train_metrics=json_safe(payload.get("train_metrics", {})),
    )
    return payload["model"], info


class _BundledModelRegistry:
    """Process-wide cache of the model that serves the bundled series."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._info: ModelInfo | None = None

    def get(self) -> tuple[Any, ModelInfo]:
        if self._model is not None and self._info is not None:
            return self._model, self._info
        with self._lock:
            if self._model is None or self._info is None:
                self._model, self._info = self._resolve()
            return self._model, self._info

    def _resolve(self) -> tuple[Any, ModelInfo]:
        from .forecaster import XGBoostForecaster

        df = load_retail_series()
        expected = fingerprint_series(df)
        path = model_path()
        notes: list[str] = []

        if path.exists():
            try:
                return _load_artifact(path, expected)
            except Exception as exc:  # noqa: BLE001 - fall back to fitting
                notes.append(f"artifact at {path} unusable, refit instead: {exc}")
        else:
            notes.append(
                f"no artifact at {path}; fitted on demand. "
                "Run `python -m training.train_xgboost` to create one."
            )

        model = XGBoostForecaster().fit(df)
        info = ModelInfo(
            source="fit-on-demand",
            model_path=str(path),
            trained_at=_now(),
            n_train_points=int(len(df)),
            train_start=str(df["date"].iloc[0].date()),
            train_end=str(df["date"].iloc[-1].date()),
            data_fingerprint=expected,
            params=json_safe(model.model.get_params()),
            notes=notes,
        )
        return model, info

    def reset(self) -> None:
        with self._lock:
            self._model = None
            self._info = None


_registry = _BundledModelRegistry()


def get_bundled_forecaster():
    return _registry.get()[0]


def get_model_info() -> ModelInfo:
    return _registry.get()[1]


def reset_cache() -> None:
    _registry.reset()
