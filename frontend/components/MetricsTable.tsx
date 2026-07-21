"use client";

import { CompareResponse } from "@/lib/api";

const METRICS: { key: "mae" | "rmse" | "mape" | "smape"; label: string }[] = [
  { key: "mae", label: "MAE" },
  { key: "rmse", label: "RMSE" },
  { key: "mape", label: "MAPE %" },
  { key: "smape", label: "sMAPE %" },
];

export default function MetricsTable({ data }: { data: CompareResponse }) {
  const rows = Object.entries(data.models).map(([name, m]) => ({ name, ...m.metrics }));
  const best = (key: string) => Math.min(...rows.map((r) => (r as any)[key]));

  return (
    <table className="metrics">
      <thead>
        <tr>
          <th>Model</th>
          {METRICS.map((m) => (
            <th key={m.key}>{m.label}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.name}>
            <td className="model-name">{r.name}</td>
            {METRICS.map((m) => {
              const value = (r as any)[m.key] as number;
              const isBest = value === best(m.key);
              return (
                <td key={m.key} className={isBest ? "best" : ""}>
                  {value.toFixed(2)}
                  {isBest && <span className="badge">best</span>}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
