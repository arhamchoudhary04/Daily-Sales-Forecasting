"""Rebuild online_retail_daily.csv from the source transactions.

Downloads the UCI Online Retail II dataset (~2 years of an online store's
invoices), keeps actual sales (drops returns / bad prices), and sums revenue
per day into a daily series. Run from anywhere:

    python backend/data/prepare_dataset.py
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

SOURCE = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
OUT = Path(__file__).with_name("online_retail_daily.csv")


def main() -> None:
    req = urllib.request.Request(SOURCE, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()

    sheets = pd.read_excel(io.BytesIO(blob), sheet_name=None, engine="openpyxl")
    raw = pd.concat(sheets.values(), ignore_index=True)

    raw = raw.rename(columns={"Quantity": "qty", "InvoiceDate": "ts", "Price": "price"})
    raw["ts"] = pd.to_datetime(raw["ts"], errors="coerce")
    raw = raw.dropna(subset=["ts", "qty", "price"])

    sales = raw[(raw["qty"] > 0) & (raw["price"] > 0)].copy()
    sales["revenue"] = sales["qty"] * sales["price"]

    daily = (
        sales.set_index("ts")["revenue"]
        .resample("D").sum()
        .round(2)
        .rename("value")
        .reset_index()
        .rename(columns={"ts": "date"})
    )
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily.to_csv(OUT, index=False)
    print(f"wrote {len(daily)} rows -> {OUT}")


if __name__ == "__main__":
    main()
