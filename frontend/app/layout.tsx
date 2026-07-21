import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Time-Series Forecasting — Chronos vs XGBoost",
  description:
    "Zero-shot foundation model (Chronos-Bolt) vs trained XGBoost, served with FastAPI and deployed on Docker + Kubernetes.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
