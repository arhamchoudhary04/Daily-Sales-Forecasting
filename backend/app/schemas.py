"""Request/response schemas for the forecasting API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SeriesPoint(BaseModel):
    date: str = Field(..., description="ISO date, e.g. 2024-01-31")
    value: float


class ForecastRequest(BaseModel):
    horizon: int = Field(default=14, ge=1, le=90, description="Days to forecast ahead")
    model: str = Field(default="both", pattern="^(chronos|xgboost|both)$")
    series: list[SeriesPoint] | None = Field(
        default=None,
        description="Your history (>=60 points). If omitted, built-in synthetic demo data is used.",
    )


class CompareRequest(BaseModel):
    horizon: int = Field(default=14, ge=1, le=90)
    series: list[SeriesPoint] | None = None
