"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import type { ChartPoint } from "@/components/ForecastChart";
import MetricsTable from "@/components/MetricsTable";
import {
  getDemoSeries,
  getModelInfo,
  runForecast,
  runCompare,
  type DemoSeries,
  type CompareResponse,
  type ModelInfo,
} from "@/lib/api";

// Recharts needs the DOM, so keep the chart client-only.
const ForecastChart = dynamic(() => import("@/components/ForecastChart"), {
  ssr: false,
  loading: () => <div className="section-title">Loading chart…</div>,
});

const CHRONOS = "chronos-bolt";
const XGBOOST = "xgboost";
const ENSEMBLE = "ensemble";
const HISTORY_WINDOW = 90;
const PREVIEW_WINDOW = 120;

type Mode = "idle" | "forecast" | "compare";

function clampHorizon(value: number): number {
  if (Number.isNaN(value)) return 1;
  return Math.min(90, Math.max(1, Math.round(value)));
}

export default function Page() {
  const [demo, setDemo] = useState<DemoSeries | null>(null);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [horizon, setHorizon] = useState(21);
  const [model, setModel] = useState<"both" | "chronos" | "xgboost" | "ensemble">("both");

  const [chart, setChart] = useState<ChartPoint[]>([]);
  const [compare, setCompare] = useState<CompareResponse | null>(null);
  const [mode, setMode] = useState<Mode>("idle");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDemoSeries(730)
      .then((d) => {
        setDemo(d);
        const start = Math.max(0, d.dates.length - PREVIEW_WINDOW);
        setChart(
          d.dates.slice(start).map((date, i) => ({
            date,
            history: d.values[start + i],
          }))
        );
      })
      .catch((e) => setError(e?.message ?? String(e)));

    // informational only, so a failure just hides the line
    getModelInfo()
      .then(setInfo)
      .catch(() => setInfo(null));
  }, []);

  async function onRunForecast() {
    if (!demo) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runForecast(horizon, model);

      const start = Math.max(0, demo.dates.length - HISTORY_WINDOW);
      const hist: ChartPoint[] = demo.dates.slice(start).map((date, i) => ({
        date,
        history: demo.values[start + i],
      }));

      const chronos = res.models[CHRONOS];
      const xgb = res.models[XGBOOST];
      const ens = res.models[ENSEMBLE];

      // connect the model lines to the last history point
      const bridge = hist[hist.length - 1];
      if (bridge) {
        if (chronos) bridge.chronos = bridge.history;
        if (xgb) bridge.xgboost = bridge.history;
        if (ens) bridge.ensemble = bridge.history;
      }

      const future: ChartPoint[] = res.future_dates.map((date, i) => {
        const point: ChartPoint = { date };
        if (chronos) point.chronos = chronos.median[i];
        if (xgb) point.xgboost = xgb.median[i];
        if (ens) point.ensemble = ens.median[i];
        const band = chronos ?? xgb ?? ens; // interval from whichever we have
        if (band) point.band = [band.lower[i], band.upper[i]];
        return point;
      });

      setCompare(null);
      setMode("forecast");
      setChart([...hist, ...future]);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onCompare() {
    if (!demo) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runCompare(horizon);

      // The backend backtests the same bundled retail series we plotted, so its
      // most-recent holdout window lines up with the tail of `demo`.
      const holdout = res.dates.length;
      const histEnd = Math.max(0, demo.dates.length - holdout);
      const start = Math.max(0, histEnd - HISTORY_WINDOW);
      const hist: ChartPoint[] = demo.dates
        .slice(start, histEnd)
        .map((date, i) => ({ date, history: demo.values[start + i] }));

      const chronos = res.models[CHRONOS]?.forecast;
      const xgb = res.models[XGBOOST]?.forecast;
      const ens = res.models[ENSEMBLE]?.forecast;

      const bridge = hist[hist.length - 1];
      if (bridge) {
        bridge.actual = bridge.history;
        if (chronos) bridge.chronos = bridge.history;
        if (xgb) bridge.xgboost = bridge.history;
        if (ens) bridge.ensemble = bridge.history;
      }

      const holdoutWindow: ChartPoint[] = res.dates.map((date, i) => {
        const point: ChartPoint = { date, actual: res.actual[i] };
        if (chronos) point.chronos = chronos.median[i];
        if (xgb) point.xgboost = xgb.median[i];
        if (ens) point.ensemble = ens.median[i];
        return point;
      });

      setCompare(res);
      setMode("compare");
      setChart([...hist, ...holdoutWindow]);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  const chartTitle =
    mode === "compare"
      ? `Most recent ${compare?.dates.length ?? horizon}-day window (actual vs forecast)`
      : mode === "forecast"
      ? `Sales forecast — next ${horizon} days`
      : "Daily sales history (UCI Online Retail II)";

  return (
    <main className="container">
      <header className="header">
        <h1>Daily Sales Forecasting</h1>
        <p>Zero-shot Chronos-Bolt vs a trained XGBoost baseline, on ~2 years of real online-retail sales.</p>
      </header>

      <section className="card">
        <div className="controls">
          <div className="field">
            <label htmlFor="horizon">Horizon (days)</label>
            <input
              id="horizon"
              type="number"
              min={1}
              max={90}
              value={horizon}
              onChange={(e) => setHorizon(clampHorizon(e.target.valueAsNumber))}
            />
          </div>
          <div className="field">
            <label htmlFor="model">Model</label>
            <select
              id="model"
              value={model}
              onChange={(e) => setModel(e.target.value as typeof model)}
            >
              <option value="both">Both</option>
              <option value="chronos">Chronos-Bolt (zero-shot)</option>
              <option value="xgboost">XGBoost (trained)</option>
              <option value="ensemble">Ensemble (most accurate)</option>
            </select>
          </div>
          <div className="actions">
            <button
              className="btn-primary"
              onClick={onRunForecast}
              disabled={loading || !demo}
            >
              {loading ? "Working…" : "Run forecast"}
            </button>
            <button
              className="btn-secondary"
              onClick={onCompare}
              disabled={loading || !demo}
            >
              Backtest &amp; compare
            </button>
          </div>
        </div>

        {info && (
          <div className="provenance">
            <span className={info.source === "artifact" ? "tag" : "tag warn"}>
              {info.source === "artifact" ? "trained artifact" : "fitted on demand"}
            </span>
            <span>
              XGBoost model
              {info.source === "artifact"
                ? " loaded from the registry"
                : " fitted at startup (no artifact found)"}
              {info.n_train_points !== null && ` · ${info.n_train_points} training days`}
              {info.train_end && ` through ${info.train_end}`}
            </span>
            {info.data_fingerprint && (
              <code title="Hash of the training series — the API refuses a model trained on different data">
                data {info.data_fingerprint}
              </code>
            )}
          </div>
        )}
      </section>

      {error && <div className="error">{error}</div>}

      <section className="card">
        <h2 className="section-title">{chartTitle}</h2>
        <ForecastChart data={chart} />
      </section>

      {mode === "compare" && compare && (
        <section className="card">
          <h2 className="section-title">
            Backtest metrics — averaged over {compare.folds ?? "several"} rolling windows (lower is better)
          </h2>
          <MetricsTable data={compare} />
        </section>
      )}

      <footer className="footer">
        Data: UCI Online Retail II (daily sales revenue). Models: Chronos-Bolt
        (zero-shot) vs XGBoost (trained), served with FastAPI.
      </footer>
    </main>
  );
}
