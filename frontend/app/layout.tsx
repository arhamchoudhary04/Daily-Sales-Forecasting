import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Daily Sales Forecasting — Chronos vs XGBoost",
  description:
    "Forecasting daily online-retail sales revenue: zero-shot Chronos-Bolt vs a trained XGBoost baseline, served with FastAPI and deployed on Docker + Kubernetes.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
