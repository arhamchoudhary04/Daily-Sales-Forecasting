"""Train the XGBoost baseline, backtest it against Chronos-Bolt, log to MLflow.

Writes the artifact the API loads at serve time (see app/registry.py). Usage:

    python -m training.train_xgboost --horizon 21 --mlflow
    python -m training.train_xgboost --skip-chronos     # fast / offline
"""
from __future__ import annotations

import argparse

from app.data import load_retail_series
from app.forecaster import XGBoostForecaster, chronos_forecast, evaluate_forecast
from app.registry import model_path, save_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + backtest forecasting models.")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Artifact path (defaults to the registry location / $MODEL_PATH).",
    )
    parser.add_argument("--mlflow", action="store_true", help="Log the run to MLflow")
    parser.add_argument("--skip-chronos", action="store_true", help="Skip the Chronos backtest (faster/offline)")
    args = parser.parse_args()

    df = load_retail_series()
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
    out = save_model(
        model,
        df,
        path=args.output or model_path(),
        train_metrics={f"holdout_{k}": v for k, v in xgb_metrics.items()},
    )
    print(f"Saved model -> {out.resolve()}")
    print("The API will load this on next start (GET /api/model-info to confirm).")

    if args.mlflow:
        import mlflow

        mlflow.set_experiment("sales-forecasting")
        with mlflow.start_run():
            mlflow.log_params({"horizon": args.horizon, "n_days": len(df)})
            for k, v in xgb_metrics.items():
                mlflow.log_metric(f"xgb_{k}", v)
            for k, v in chronos_metrics.items():
                mlflow.log_metric(f"chronos_{k}", v)
            mlflow.log_artifact(str(out))
            print("Logged run to MLflow.")


if __name__ == "__main__":
    main()
