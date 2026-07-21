"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface ChartPoint {
  date: string;
  history?: number;
  actual?: number;
  chronos?: number;
  xgboost?: number;
  band?: [number, number]; // [lower, upper] prediction interval
}

export default function ForecastChart({ data }: { data: ChartPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={420}>
      <ComposedChart data={data} margin={{ top: 12, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#232936" />
        <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#8b94a7" }} minTickGap={48} />
        <YAxis tick={{ fontSize: 11, fill: "#8b94a7" }} width={48} />
        <Tooltip
          contentStyle={{
            background: "#0f1420",
            border: "1px solid #2a2f3a",
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: "#e2e8f0" }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Area
          dataKey="band"
          name="Prediction interval"
          stroke="none"
          fill="#6366f1"
          fillOpacity={0.15}
          isAnimationActive={false}
        />
        <Line dataKey="history" name="History" stroke="#64748b" dot={false} strokeWidth={1.5} />
        <Line dataKey="actual" name="Actual" stroke="#e2e8f0" dot={false} strokeWidth={2} />
        <Line
          dataKey="chronos"
          name="Chronos-Bolt (zero-shot)"
          stroke="#818cf8"
          dot={false}
          strokeWidth={2.5}
          connectNulls
        />
        <Line
          dataKey="xgboost"
          name="XGBoost (trained)"
          stroke="#34d399"
          dot={false}
          strokeWidth={2.5}
          connectNulls
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
