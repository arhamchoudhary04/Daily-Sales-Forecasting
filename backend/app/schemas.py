"""Request/response schemas for the forecasting API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SeriesPoint(BaseModel):
    date: str = Field(..., description="ISO date, e.g. 2024-01-31")
    value: float


SERIES_DESCRIPTION = (
    "Your own history (>=60 points). If omitted, the bundled daily "
    "online-retail sales series is used."
)


class ForecastRequest(BaseModel):
    horizon: int = Field(default=14, ge=1, le=90, description="Days to forecast ahead")
    model: str = Field(default="both", pattern="^(chronos|xgboost|both|ensemble)$")
    series: list[SeriesPoint] | None = Field(default=None, description=SERIES_DESCRIPTION)


class CompareRequest(BaseModel):
    horizon: int = Field(default=14, ge=1, le=90)
    folds: int = Field(
        default=4,
        ge=1,
        le=8,
        description=(
            "Rolling windows to average over. Each fold refits XGBoost and "
            "reruns Chronos, so higher is slower but fairer."
        ),
    )
    series: list[SeriesPoint] | None = Field(default=None, description=SERIES_DESCRIPTION)
