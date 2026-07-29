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
  coverage: number;
  pinball: number;
  interval_width: number;
}

export interface CompareResponse {
  dates: string[];
  actual: number[];
  folds?: number;
  /** Target band coverage in percent, e.g. 80. */
  nominal_coverage?: number;
  models: Record<string, { forecast: ModelForecast; metrics: Metrics }>;
}

export interface ModelInfo {
  source: "artifact" | "fit-on-demand";
  model_path: string | null;
  trained_at: string | null;
  n_train_points: number | null;
  train_start: string | null;
  train_end: string | null;
  data_fingerprint: string | null;
  train_metrics: Record<string, number>;
  notes: string[];
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

export const runCompare = (horizon: number, folds?: number) =>
  req<CompareResponse>("/api/compare", {
    method: "POST",
    body: JSON.stringify(folds === undefined ? { horizon } : { horizon, folds }),
  });

export const getModelInfo = () => req<ModelInfo>("/api/model-info");
