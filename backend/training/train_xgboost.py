"""Train the XGBoost baseline, backtest it against Chronos-Bolt, log to MLflow.

Usage:
    python -m training.train_xgboost --horizon 14 --mlflow
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from app.data import generate_synthetic_series
from app.forecaster import XGBoostForecaster, chronos_forecast, evaluate_forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + backtest forecasting models.")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--n-days", type=int, default=730)
    parser.add_argument("--output", type=str, default="app/models/xgb_model.joblib")
    parser.add_argument("--mlflow", action="store_true", help="Log the run to MLflow")
    parser.add_argument("--skip-chronos", action="store_true", help="Skip the Chronos backtest (faster/offline)")
    args = parser.parse_args()

    df = generate_synthetic_series(n_days=args.n_days)
    train, test = df.iloc[:-args.horizon], df.iloc[-args.horizon:]
    actual = test["value"].tolist()

    xgb_metrics = evaluate_forecast(actual, XGBoostForecaster().fit(train).predict(args.horizon).median)
    print("XGBoost :", xgb_metrics)

    chronos_metrics = {}
    if not args.skip_chronos:
        chronos_metrics = evaluate_forecast(actual, chronos_forecast(train["value"].tolist(), args.horizon).median)
        print("Chronos :", chronos_metrics)

    # Refit on ALL data before saving so the served model uses full history.
    model = XGBoostForecaster().fit(df)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"Saved model -> {out.resolve()}")

    if args.mlflow:
        import mlflow

        mlflow.set_experiment("timeseries-forecasting")
        with mlflow.start_run():
            mlflow.log_params({"horizon": args.horizon, "n_days": args.n_days})
            for k, v in xgb_metrics.items():
                mlflow.log_metric(f"xgb_{k}", v)
            for k, v in chronos_metrics.items():
                mlflow.log_metric(f"chronos_{k}", v)
            mlflow.log_artifact(str(out))
            print("Logged run to MLflow.")


if __name__ == "__main__":
    main()
