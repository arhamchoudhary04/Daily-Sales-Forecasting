"""Feature engineering, mostly guarding against lookahead leakage.

If a feature on row i can see row i's own target, every backtest score in the
project becomes optimistic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data import FEATURE_COLUMNS, make_supervised_features


def _series(n: int = 120) -> pd.DataFrame:
    """Strictly increasing, so an off-by-one in a window is visible."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "value": np.arange(n, dtype=float) * 10.0,
        }
    )


def test_produces_expected_columns():
    feats = make_supervised_features(_series())
    for col in FEATURE_COLUMNS:
        assert col in feats.columns, f"missing feature {col}"
    assert "value" in feats.columns


def test_drops_rows_with_incomplete_history():
    # Longest lookback is 28, so the first 28 rows can't have full features.
    n = 120
    feats = make_supervised_features(_series(n))
    assert len(feats) == n - 28
    assert feats["date"].iloc[0] == pd.Timestamp("2024-01-29")


@pytest.mark.parametrize("lag", [1, 2, 3, 7, 14, 28])
def test_lag_features_look_strictly_backwards(lag):
    raw = _series()
    feats = make_supervised_features(raw)
    lookup = dict(zip(raw["date"], raw["value"]))
    for _, row in feats.iterrows():
        expected = lookup[row["date"] - pd.Timedelta(days=lag)]
        assert row[f"lag_{lag}"] == pytest.approx(expected)


def test_rolling_features_exclude_the_current_row():
    raw = _series()
    feats = make_supervised_features(raw)
    values = raw["value"].to_numpy()
    index_of = {d: i for i, d in enumerate(raw["date"])}

    for _, row in feats.iterrows():
        i = index_of[row["date"]]
        assert row["roll_mean_7"] == pytest.approx(values[i - 7 : i].mean())
        assert row["roll_mean_28"] == pytest.approx(values[i - 28 : i].mean())
        assert row["roll_std_7"] == pytest.approx(values[i - 7 : i].std(ddof=1))
        # The leaky window must differ, or the assertions above prove nothing.
        assert row["roll_mean_7"] != pytest.approx(values[i - 6 : i + 1].mean())


def test_no_feature_is_identical_to_the_target():
    feats = make_supervised_features(_series())
    for col in FEATURE_COLUMNS:
        assert not np.allclose(feats[col].to_numpy(), feats["value"].to_numpy()), (
            f"{col} equals the target"
        )


def test_calendar_features_match_the_row_date():
    feats = make_supervised_features(_series())
    for _, row in feats.iterrows():
        d = row["date"]
        assert row["dayofweek"] == d.dayofweek
        assert row["day"] == d.day
        assert row["month"] == d.month
        assert row["dayofyear"] == d.dayofyear


def test_no_nans_survive():
    feats = make_supervised_features(_series())
    assert not feats[FEATURE_COLUMNS].isna().any().any()
