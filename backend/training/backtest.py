"""Rolling-window backtest of both models and their ensemble.

Averages over several recent windows instead of a single holdout. Shares
`backtest_windows` with /api/compare so the script and the app agree. Usage:

    python -m training.backtest --horizon 21 --folds 6
    python -m training.backtest --horizon 21 --folds 6 --skip-chronos
"""
from __future__ import annotations

import argparse

import numpy as np

from app.data import load_retail_series
from app.forecaster import (
    NOMINAL_COVERAGE,
    XGBoostForecaster,
    backtest_windows,
    chronos_forecast,
    ensemble_forecast,
    score_forecast,
)

COLUMNS = ["mae", "rmse", "mape", "smape", "coverage", "pinball", "interval_width"]
HEADERS = ["MAE", "RMSE", "MAPE %", "sMAPE %", "Cover %", "Pinball", "Width"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Rolling-window backtest.")
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--step", type=int, default=None,
                    help="Window stride (defaults to --horizon, i.e. non-overlapping).")
    ap.add_argument("--skip-chronos", action="store_true")
    args = ap.parse_args()

    df = load_retail_series()
    scores: dict[str, list[dict]] = {"chronos-bolt": [], "xgboost": [], "ensemble": []}

    windows = backtest_windows(len(df), args.horizon, args.folds, step=args.step)
    if not windows:
        raise SystemExit("Series too short for even one window at this horizon.")

    for i, (start, end) in enumerate(windows):
        train, test = df.iloc[:start], df.iloc[start:end]
        actual = test["value"].tolist()

        xgb = XGBoostForecaster().fit(train).predict(args.horizon)
        scores["xgboost"].append(score_forecast(actual, xgb))

        if not args.skip_chronos:
            ch = chronos_forecast(train["value"].tolist(), args.horizon)
            scores["chronos-bolt"].append(score_forecast(actual, ch))
            scores["ensemble"].append(score_forecast(actual, ensemble_forecast(ch, xgb)))

        print(f"fold {i + 1}: {test['date'].iloc[0].date()} -> {test['date'].iloc[-1].date()}")

    def avg(rows: list[dict]) -> dict:
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}

    print(f"\n== rolling backtest: horizon={args.horizon}, folds={len(windows)} (averaged) ==")
    print(f"   lower is better, except coverage (target {NOMINAL_COVERAGE:.0f}%)\n")
    print("   " + f"{'model':<14}" + "".join(f"{h:>12}" for h in HEADERS))
    for name, rows in scores.items():
        if not rows:
            continue
        m = avg(rows)
        print("   " + f"{name:<14}" + "".join(f"{m[c]:>12,.2f}" for c in COLUMNS))


if __name__ == "__main__":
    main()
