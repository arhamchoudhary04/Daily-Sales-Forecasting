"""Metrics checked against hand-computed values, not against current output."""
from __future__ import annotations

import numpy as np
import pytest

from app.forecaster import (
    NOMINAL_COVERAGE,
    ForecastResult,
    evaluate_forecast,
    evaluate_interval,
    pinball_loss,
    score_forecast,
)

# errors of [-10, 20, -30, -20]; the trailing 0 is a closed-shop day, which MAPE
# has to skip.
ACTUAL = [100.0, 200.0, 300.0, 0.0]
PREDICTED = [110.0, 180.0, 330.0, 20.0]


def test_mae_and_rmse_are_exact():
    m = evaluate_forecast(ACTUAL, PREDICTED)
    # |err| = [10, 20, 30, 20] -> 80 / 4
    assert m["mae"] == pytest.approx(20.0)
    # err^2 = [100, 400, 900, 400] -> sqrt(1800 / 4)
    assert m["rmse"] == pytest.approx(np.sqrt(450.0), rel=1e-6)


def test_mape_skips_zero_actuals():
    m = evaluate_forecast(ACTUAL, PREDICTED)
    # every non-zero point is off by exactly 10%
    assert m["mape"] == pytest.approx(10.0)


def test_mape_is_zero_when_every_actual_is_zero():
    m = evaluate_forecast([0.0, 0.0], [5.0, 7.0])
    assert m["mape"] == 0.0
    assert not np.isnan(m["mae"])


def test_smape_uses_every_point_including_zeros():
    m = evaluate_forecast(ACTUAL, PREDICTED)
    expected = (
        np.mean([20 / 210, 40 / 380, 60 / 630, 40 / 20]) * 100
    )  # 2|err| / (|a| + |p|)
    assert m["smape"] == pytest.approx(expected, rel=1e-5)


def test_perfect_forecast_scores_zero():
    m = evaluate_forecast(ACTUAL, ACTUAL)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["smape"] == pytest.approx(0.0, abs=1e-6)


# --- interval metrics ---

I_ACTUAL = [100.0, 200.0, 300.0, 400.0]
I_LOWER = [90.0, 250.0, 280.0, 100.0]
I_MEDIAN = [100.0, 200.0, 300.0, 300.0]
I_UPPER = [110.0, 260.0, 320.0, 500.0]


def test_coverage_counts_actuals_inside_the_band():
    m = evaluate_interval(I_ACTUAL, I_LOWER, I_MEDIAN, I_UPPER)
    # 200 is below its lower bound of 250; the other three are inside
    assert m["coverage"] == pytest.approx(75.0)


def test_interval_width_is_the_mean_band_width():
    m = evaluate_interval(I_ACTUAL, I_LOWER, I_MEDIAN, I_UPPER)
    # widths [20, 10, 40, 400] -> 470 / 4
    assert m["interval_width"] == pytest.approx(117.5)


@pytest.mark.parametrize(
    "quantile, predicted, expected",
    [
        (0.1, I_LOWER, 19.5),   # (1 + 45 + 2 + 30) / 4
        (0.5, I_MEDIAN, 12.5),  # (0 + 0 + 0 + 50) / 4
        (0.9, I_UPPER, 4.75),   # (1 + 6 + 2 + 10) / 4
    ],
)
def test_pinball_loss_per_quantile(quantile, predicted, expected):
    assert pinball_loss(I_ACTUAL, predicted, quantile) == pytest.approx(expected)


def test_pinball_is_the_mean_over_the_three_quantiles():
    m = evaluate_interval(I_ACTUAL, I_LOWER, I_MEDIAN, I_UPPER)
    assert m["pinball"] == pytest.approx((19.5 + 12.5 + 4.75) / 3)


def test_pinball_at_median_is_half_the_mae():
    mae = evaluate_forecast(I_ACTUAL, I_MEDIAN)["mae"]
    assert pinball_loss(I_ACTUAL, I_MEDIAN, 0.5) == pytest.approx(mae / 2)


def test_widening_the_band_buys_coverage_but_costs_pinball():
    """Why the UI ranks coverage by closeness to nominal, not by size."""
    tight = evaluate_interval(I_ACTUAL, I_LOWER, I_MEDIAN, I_UPPER)
    absurd = evaluate_interval(
        I_ACTUAL,
        [-1e6] * 4,
        I_MEDIAN,
        [1e6] * 4,
    )
    assert absurd["coverage"] == 100.0
    assert absurd["coverage"] > tight["coverage"]
    assert absurd["pinball"] > tight["pinball"]
    assert absurd["interval_width"] > tight["interval_width"]


def test_nominal_coverage_matches_the_emitted_quantiles():
    assert NOMINAL_COVERAGE == pytest.approx(80.0)


def test_score_forecast_merges_point_and_interval_metrics():
    result = ForecastResult(
        model="test", horizon=4, median=I_MEDIAN, lower=I_LOWER, upper=I_UPPER
    )
    scored = score_forecast(I_ACTUAL, result)
    for key in ("mae", "rmse", "mape", "smape", "coverage", "pinball", "interval_width"):
        assert key in scored
    assert scored["mae"] == pytest.approx(evaluate_forecast(I_ACTUAL, I_MEDIAN)["mae"])
