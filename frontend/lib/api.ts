// Typed client for the FastAPI forecasting backend.
// Default is same-origin ("") — the Next.js server proxies /api and /health to
// the backend via rewrites (see next.config.mjs). This makes the app work
// unchanged locally, in docker-compose, in Kubernetes, and in GitHub Codespaces.
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

export interface DemoSeries {
  dates: string[];
  values: number[];
}

export interface ModelForecast {
  model: string;
  horizon: number;
  median: number[];
  lower: number[];
  upper: number[];
}

export interface ForecastResponse {
  horizon: number;
  future_dates: string[];
  models: Record<string, ModelForecast>;
}

export interface Metrics {
  mae: number;
  rmse: number;
  mape: number;
  smape: number;
}

export interface CompareResponse {
  dates: string[];
  actual: number[];
  models: Record<string, { forecast: ModelForecast; metrics: Metrics }>;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return (await res.json()) as T;
}

export const getDemoSeries = (nDays = 730) =>
  req<DemoSeries>(`/api/demo-series?n_days=${nDays}`);

export const runForecast = (horizon: number, model: string) =>
  req<ForecastResponse>("/api/forecast", {
    method: "POST",
    body: JSON.stringify({ horizon, model }),
  });

export const runCompare = (horizon: number) =>
  req<CompareResponse>("/api/compare", {
    method: "POST",
    body: JSON.stringify({ horizon }),
  });
