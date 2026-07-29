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
  ensemble?: number;
  band?: [number, number]; // [lower, upper] prediction interval
}

const INK = "#14130f";
const MUTED = "#8a8880";
const GRID = "#e8e6df";
const AXIS = "#d8d6cd";
const CHRONOS = "#2a78d6";
const XGBOOST = "#eb6834";
const ENSEMBLE = "#1baf7a";

const gbp = (v: number) => "£" + Math.round(v).toLocaleString("en-GB");
const axisGbp = (v: any) =>
  Math.abs(Number(v)) >= 1000 ? `£${Math.round(Number(v) / 1000)}k` : `£${Math.round(Number(v))}`;

export default function ForecastChart({ data }: { data: ChartPoint[] }) {
  // Recharts legends every <Line> it's given, even an empty one, so only render
  // the series this dataset actually contains.
  const has = (key: keyof ChartPoint) => data.some((d) => d[key] != null);
  const show = {
    band: has("band"),
    history: has("history"),
    actual: has("actual"),
    chronos: has("chronos"),
    xgboost: has("xgboost"),
    ensemble: has("ensemble"),
  };

  return (
    <ResponsiveContainer width="100%" height={380}>
      <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 4 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 12, fill: MUTED }}
          tickLine={false}
          axisLine={{ stroke: AXIS }}
          minTickGap={56}
        />
        <YAxis
          tick={{ fontSize: 12, fill: MUTED }}
          tickLine={false}
          axisLine={false}
          width={52}
          tickFormatter={axisGbp}
        />
        <Tooltip
          contentStyle={{
            background: "#ffffff",
            border: "1px solid #e8e6df",
            borderRadius: 10,
            boxShadow: "0 4px 16px rgba(20, 19, 15, 0.08)",
            fontSize: 12,
            color: INK,
          }}
          labelStyle={{ color: MUTED, marginBottom: 4 }}
          formatter={(value: any, name: any) =>
            Array.isArray(value)
              ? [`${gbp(value[0])} – ${gbp(value[1])}`, name]
              : [gbp(value as number), name]
          }
        />
        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} iconType="plainline" />
        {show.band && (
          <Area
            dataKey="band"
            name="Prediction interval"
            stroke="none"
            fill={CHRONOS}
            fillOpacity={0.1}
            isAnimationActive={false}
          />
        )}
        {show.history && (
          <Line dataKey="history" name="History" stroke={MUTED} dot={false} strokeWidth={1.5} isAnimationActive={false} />
        )}
        {show.actual && (
          <Line dataKey="actual" name="Actual" stroke={INK} dot={false} strokeWidth={2} isAnimationActive={false} />
        )}
        {show.chronos && (
          <Line
            dataKey="chronos"
            name="Chronos-Bolt"
            stroke={CHRONOS}
            dot={false}
            strokeWidth={2}
            connectNulls
            isAnimationActive={false}
          />
        )}
        {show.xgboost && (
          <Line
            dataKey="xgboost"
            name="XGBoost"
            stroke={XGBOOST}
            dot={false}
            strokeWidth={2}
            connectNulls
            isAnimationActive={false}
          />
        )}
        {show.ensemble && (
          <Line
            dataKey="ensemble"
            name="Ensemble"
            stroke={ENSEMBLE}
            dot={false}
            strokeWidth={2.5}
            connectNulls
            isAnimationActive={false}
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
