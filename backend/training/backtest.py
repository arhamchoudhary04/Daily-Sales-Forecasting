"""Rolling-window backtest of both models on the real sales series.

Holds out several recent windows (not just the last one) and averages the
scores, which is far more honest than a single holdout. Usage:

    python -m training.backtest --horizon 21 --folds 4
"""
from __future__ import annotations

import argparse

import numpy as np

from app.data import load_retail_series
from app.forecaster import XGBoostForecaster, chronos_forecast, evaluate_forecast


def main() -> None:
    ap = argparse.ArgumentParser(description="Rolling-window backtest.")
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--step", type=int, default=21)
    ap.add_argument("--skip-chronos", action="store_true")
    args = ap.parse_args()

    df = load_retail_series()
    scores: dict[str, list[dict]] = {"chronos-bolt": [], "xgboost": []}

    for i in range(args.folds):
        end = len(df) - i * args.step
        start = end - args.horizon
        if start < 60:
            break
        train, test = df.iloc[:start], df.iloc[start:end]
        actual = test["value"].tolist()

        xgb = XGBoostForecaster().fit(train).predict(args.horizon)
        scores["xgboost"].append(evaluate_forecast(actual, xgb.median))
        if not args.skip_chronos:
            ch = chronos_forecast(train["value"].tolist(), args.horizon)
            scores["chronos-bolt"].append(evaluate_forecast(actual, ch.median))

        print(f"fold {i + 1}: {test['date'].iloc[0].date()} -> {test['date'].iloc[-1].date()}")

    def avg(rows: list[dict]) -> dict:
        return {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0]} if rows else {}

    print(f"\n== rolling backtest: horizon={args.horizon}, folds={args.folds} (averaged) ==")
    for name, rows in scores.items():
        if rows:
            print(f"{name:14s} {avg(rows)}")


if __name__ == "__main__":
    main()
