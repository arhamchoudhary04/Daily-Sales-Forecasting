"""Backtest window arithmetic and the ensemble.

An off-by-one in `backtest_windows` would either leak training data into a test
window or silently score fewer folds than reported.
"""
from __future__ import annotations

import pytest

from app.forecaster import (
    MIN_TRAIN_POINTS,
    ForecastResult,
    backtest_windows,
    ensemble_forecast,
)

RETAIL_LENGTH = 739  # rows in the bundled series


def test_returns_the_requested_number_of_folds():
    windows = backtest_windows(RETAIL_LENGTH, horizon=21, folds=6)
    assert len(windows) == 6


def test_most_recent_window_comes_first_and_ends_at_the_series_end():
    windows = backtest_windows(RETAIL_LENGTH, horizon=21, folds=4)
    # the API plots windows[0], so it has to be the newest data
    assert windows[0] == (RETAIL_LENGTH - 21, RETAIL_LENGTH)
    starts = [s for s, _ in windows]
    assert starts == sorted(starts, reverse=True)


@pytest.mark.parametrize("horizon", [7, 14, 21, 30])
def test_every_window_is_exactly_horizon_long(horizon):
    for start, end in backtest_windows(RETAIL_LENGTH, horizon=horizon, folds=4):
        assert end - start == horizon


def test_default_step_makes_windows_non_overlapping_and_contiguous():
    windows = backtest_windows(RETAIL_LENGTH, horizon=21, folds=5)
    for (newer_start, _), (_, older_end) in zip(windows, windows[1:]):
        assert newer_start == older_end


def test_training_prefix_never_reaches_into_the_test_window():
    for start, end in backtest_windows(RETAIL_LENGTH, horizon=21, folds=6):
        assert start < end
        assert start >= MIN_TRAIN_POINTS
        # asserted explicitly so a future "start - 1" can't slip through
        assert not set(range(0, start)) & set(range(start, end))


def test_stops_early_rather_than_returning_windows_without_enough_training_data():
    # the second window here would train on only 58 points
    windows = backtest_windows(100, horizon=21, folds=10, min_train=60)
    assert len(windows) == 1
    assert windows[0] == (79, 100)


def test_returns_empty_when_the_series_is_too_short_for_one_window():
    assert backtest_windows(70, horizon=21, folds=4, min_train=60) == []


def test_explicit_step_allows_overlapping_windows():
    windows = backtest_windows(RETAIL_LENGTH, horizon=21, folds=3, step=7)
    assert [w[1] for w in windows] == [739, 732, 725]
    assert windows[1][1] > windows[0][0]  # step < horizon, so they overlap


# --- ensemble ---

def _result(model: str, median, lower, upper) -> ForecastResult:
    return ForecastResult(
        model=model, horizon=len(median), median=median, lower=lower, upper=upper
    )


def test_ensemble_averages_each_band_elementwise():
    a = _result("a", [100.0, 200.0], [80.0, 180.0], [120.0, 220.0])
    b = _result("b", [200.0, 400.0], [120.0, 220.0], [280.0, 580.0])
    ens = ensemble_forecast(a, b)

    assert ens.model == "ensemble"
    assert ens.horizon == 2
    assert ens.median == [150.0, 300.0]
    assert ens.lower == [100.0, 200.0]
    assert ens.upper == [200.0, 400.0]


def test_ensemble_of_identical_forecasts_is_unchanged():
    a = _result("a", [10.0, 20.0], [5.0, 15.0], [15.0, 25.0])
    ens = ensemble_forecast(a, a)
    assert ens.median == a.median
    assert ens.lower == a.lower
    assert ens.upper == a.upper


def test_ensemble_error_cannot_exceed_the_worse_of_its_parents():
    """The mean sits between the two predictions, so error is bounded."""
    from app.forecaster import evaluate_forecast

    actual = [100.0, 100.0, 100.0]
    a = _result("a", [80.0, 130.0, 100.0], [70.0] * 3, [90.0] * 3)
    b = _result("b", [130.0, 90.0, 100.0], [120.0] * 3, [140.0] * 3)

    mae_a = evaluate_forecast(actual, a.median)["mae"]
    mae_b = evaluate_forecast(actual, b.median)["mae"]
    mae_ens = evaluate_forecast(actual, ensemble_forecast(a, b).median)["mae"]

    assert mae_ens <= max(mae_a, mae_b)
