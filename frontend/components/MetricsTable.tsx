"use client";

import { CompareResponse, Metrics } from "@/lib/api";

type MetricKey = keyof Metrics;

// "target" means closest to nominal wins, not smallest: an infinitely wide band
// would score 100% coverage.
type Goal = "lower" | "target";

const METRICS: { key: MetricKey; label: string; goal: Goal; title: string }[] = [
  { key: "mae", label: "MAE", goal: "lower", title: "Mean absolute error (£)" },
  { key: "rmse", label: "RMSE", goal: "lower", title: "Root mean squared error (£) — punishes big misses" },
  { key: "mape", label: "MAPE %", goal: "lower", title: "Mean absolute % error, over non-zero days only" },
  { key: "smape", label: "sMAPE %", goal: "lower", title: "Symmetric MAPE — stable when actuals near zero" },
  { key: "coverage", label: "Cover %", goal: "target", title: "% of actuals inside the 80% band — closest to target wins" },
  { key: "pinball", label: "Pinball", goal: "lower", title: "Mean quantile loss — proper score for the whole band" },
  { key: "interval_width", label: "Width", goal: "lower", title: "Mean band width (£) — the cost of that coverage" },
];

const DEFAULT_NOMINAL = 80;

export default function MetricsTable({ data }: { data: CompareResponse }) {
  const nominal = data.nominal_coverage ?? DEFAULT_NOMINAL;
  const rows = Object.entries(data.models).map(([name, m]) => ({ name, ...m.metrics }));

  function bestValue(key: MetricKey, goal: Goal): number | null {
    const values = rows
      .map((r) => r[key])
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    if (values.length === 0) return null;
    return goal === "target"
      ? values.reduce((a, b) => (Math.abs(b - nominal) < Math.abs(a - nominal) ? b : a))
      : Math.min(...values);
  }

  return (
    <div className="table-scroll">
      <table className="metrics">
        <thead>
          <tr>
            <th>Model</th>
            {METRICS.map((m) => (
              <th key={m.key} title={m.title}>
                {m.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td className="model-name">{r.name}</td>
              {METRICS.map((m) => {
                const value = r[m.key];
                if (typeof value !== "number" || !Number.isFinite(value)) {
                  return (
                    <td key={m.key} className="muted">
                      —
                    </td>
                  );
                }
                const isBest = value === bestValue(m.key, m.goal);
                return (
                  <td key={m.key} className={isBest ? "best" : ""}>
                    {value.toLocaleString("en-GB", {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    })}
                    {isBest && <span className="badge">best</span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="table-note">
        Lower is better, except <strong>Coverage %</strong>, where the target is{" "}
        {nominal}% — the share of actuals that should fall inside the 80% band.
      </p>
    </div>
  );
}
