"""Generates the README figures as SVG — light and dark variants of each.

Hand-rolled SVG rather than matplotlib: no plotting dependency, vector output that
stays crisp at any zoom, and a light/dark pair so the README reads correctly in
either GitHub theme (`<picture>` picks one).

    python docs/make_figures.py

Series and weekday figures are computed from the bundled CSV. The scorecard
figures use the published backtest numbers (see PUBLISHED_* below), which are
reproduced by:

    cd backend && python -m training.backtest --horizon 21 --folds 6
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "backend" / "data" / "online_retail_daily.csv"
OUT_DIR = ROOT / "docs"

# Identity colours, shared with the app's chart (frontend/components/ForecastChart.tsx)
# so figures and screenshots read as one system. Validated for both surfaces:
# OKLCH lightness inside the light and dark bands, chroma above the gray floor,
# and worst adjacent CVD separation dE 8.6 (deuteranopia) / 25.3 (normal vision).
CHRONOS = "#2a78d6"
XGBOOST = "#e07038"
ENSEMBLE = "#1baf7a"

FONT = "ui-sans-serif, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

THEMES = {
    "light": {"ink": "#14130f", "ink2": "#55534d", "muted": "#8a8880", "grid": "#e8e6df"},
    "dark": {"ink": "#e6edf3", "ink2": "#c2cbd6", "muted": "#8b949e", "grid": "#272d36"},
}

# Rolling backtest, horizon 21, folds 6 — the numbers tabulated in the README.
PUBLISHED_POINT = {  # model -> (MAE, RMSE)
    "Chronos-Bolt": (12400, 18909),
    "XGBoost": (12030, 17277),
    "Ensemble": (11154, 16847),
}
PUBLISHED_COVERAGE = {"Chronos-Bolt": 81.0, "XGBoost": 22.2, "Ensemble": 61.9}
NOMINAL_COVERAGE = 80.0
MODEL_COLOURS = {"Chronos-Bolt": CHRONOS, "XGBoost": XGBOOST, "Ensemble": ENSEMBLE}

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --- data -------------------------------------------------------------------

def load_series() -> list[tuple[date, float]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        return [(date.fromisoformat(r["date"]), float(r["value"])) for r in csv.DictReader(fh)]


def rolling_mean(values: list[float], window: int) -> list[float | None]:
    """Trailing mean; None until the window is full, so nothing is implied early."""
    out: list[float | None] = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / window if i >= window - 1 else None)
    return out


def weekday_means(series: list[tuple[date, float]]) -> list[float]:
    sums = [0.0] * 7
    counts = [0] * 7
    for d, v in series:
        sums[d.weekday()] += v
        counts[d.weekday()] += 1
    return [s / c if c else 0.0 for s, c in zip(sums, counts)]


# --- svg primitives ---------------------------------------------------------

def gbp_k(v: float) -> str:
    return f"£{v / 1000:,.0f}k" if abs(v) >= 1000 else f"£{v:,.0f}"


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x: float, y: float, s: str, fill: str, size: float = 11.5,
         anchor: str = "start", weight: str = "400") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')


def bar_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """Bar with rounded top corners, square where it meets the baseline."""
    r = max(0.0, min(r, w / 2, h))
    return (f"M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
            f"V{y + h:.1f} Z")


def nice_ticks(top: float, count: int = 4) -> list[float]:
    """Round tick values at or below `top`, so gridlines land on readable numbers."""
    raw = top / count
    magnitude = 10 ** (len(f"{int(raw)}") - 1)
    for mult in (1, 2, 2.5, 5, 10):
        step = magnitude * mult
        if step >= raw:
            break
    return [i * step for i in range(count + 1) if i * step <= top * 1.0001]


def legend(items: list[tuple[str, str]], x: float, y: float, colours: dict[str, str] | None = None,
           ink: str = "#000") -> str:
    """Swatch + label pairs on one row. Identity is never colour alone."""
    parts, cursor = [], x
    for label, colour in items:
        parts.append(f'<rect x="{cursor:.1f}" y="{y - 7:.1f}" width="9" height="9" rx="2.5" fill="{colour}"/>')
        parts.append(text(cursor + 14, y + 1, label, ink, 11.5))
        cursor += 14 + 7.0 * len(label) + 20
    return "".join(parts)


def svg(width: int, height: int, body: str, title: str, desc: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
            f"<title>{esc(title)}</title><desc>{esc(desc)}</desc>{body}</svg>\n")


# --- figures ----------------------------------------------------------------

def fig_series(series: list[tuple[date, float]], t: dict[str, str]) -> str:
    W, H = 760, 300
    L, R, TOP, B = 52, 16, 44, 34
    plot_w, plot_h = W - L - R, H - TOP - B

    values = [v for _, v in series]
    trend = rolling_mean(values, 30)
    top = max(values) * 1.05
    xs = lambda i: L + plot_w * i / (len(values) - 1)
    ys = lambda v: TOP + plot_h * (1 - v / top)

    parts = [text(L, 18, "Daily sales revenue, Dec 2009 – Dec 2011", t["ink"], 13.5, weight="600"),
             text(L, 33, f"{len(values)} daily points, UCI Online Retail II aggregated to revenue", t["muted"], 11)]

    for tick in nice_ticks(top):
        y = ys(tick)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        parts.append(text(L - 8, y + 3.5, gbp_k(tick), t["muted"], 10.5, anchor="end"))

    first_year = series[0][0].year
    for i, (d, _) in enumerate(series):
        if d.day == 1 and d.month in (1, 4, 7, 10):
            label = f"{d:%b}" if d.year == first_year or d.month != 1 else f"{d:%b %Y}"
            parts.append(text(xs(i), H - B + 16, label, t["muted"], 10.5, anchor="middle"))

    daily = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(values))
    parts.append(f'<polyline points="{daily}" fill="none" stroke="{t["muted"]}" '
                 f'stroke-width="1" stroke-opacity="0.55"/>')

    smoothed = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(trend) if v is not None)
    parts.append(f'<polyline points="{smoothed}" fill="none" stroke="{t["ink"]}" '
                 f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')

    last = [v for v in trend if v is not None][-1]
    parts.append(f'<circle cx="{xs(len(values) - 1):.1f}" cy="{ys(last):.1f}" r="3.5" fill="{t["ink"]}"/>')
    parts.append(text(xs(len(values) - 1) - 8, ys(last) - 10, gbp_k(last), t["ink"], 11, anchor="end", weight="600"))

    parts.append(legend([("Daily revenue", t["muted"]), ("30-day mean", t["ink"])],
                        W - R - 220, 20, ink=t["ink2"]))
    return svg(W, H, "".join(parts), "Daily sales revenue with 30-day rolling mean",
               "Daily revenue is volatile with weekly zeros; the 30-day mean roughly doubles over two years.")


def fig_weekday(means: list[float], t: dict[str, str]) -> str:
    W, H = 760, 268
    L, R, TOP, B = 52, 16, 44, 40
    plot_w, plot_h = W - L - R, H - TOP - B
    top = max(means) * 1.18
    ys = lambda v: TOP + plot_h * (1 - v / top)

    parts = [text(L, 18, "Mean revenue by weekday", t["ink"], 13.5, weight="600"),
             text(L, 33, "Saturday is a structural zero — the store is closed, so MAPE is scored on non-zero days only",
                  t["muted"], 11)]

    for tick in nice_ticks(top):
        y = ys(tick)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        parts.append(text(L - 8, y + 3.5, gbp_k(tick), t["muted"], 10.5, anchor="end"))

    slot = plot_w / 7
    bar_w = slot - 22  # 2px+ surface gap between adjacent bars
    baseline = TOP + plot_h
    for i, (day, value) in enumerate(zip(WEEKDAYS, means)):
        x = L + i * slot + (slot - bar_w) / 2
        h = max(1.0, baseline - ys(value))
        parts.append(f'<path d="{bar_path(x, baseline - h, bar_w, h)}" fill="{CHRONOS}"/>')
        parts.append(text(x + bar_w / 2, baseline - h - 8, gbp_k(value), t["ink"], 11,
                          anchor="middle", weight="600"))
        parts.append(text(x + bar_w / 2, H - B + 18, day, t["ink2"], 11.5, anchor="middle"))

    sat_x = L + 5 * slot + slot / 2
    parts.append(text(sat_x, ys(0) - 26, "closed", t["muted"], 10.5, anchor="middle"))
    return svg(W, H, "".join(parts), "Mean revenue by weekday",
               "Monday to Thursday are highest, Sunday lower, and Saturday is near zero because the store is closed.")


def fig_point_error(t: dict[str, str]) -> str:
    W, H = 760, 288
    L, R, TOP, B = 56, 16, 62, 40
    plot_w, plot_h = W - L - R, H - TOP - B
    metrics = ["MAE", "RMSE"]
    top = max(v for pair in PUBLISHED_POINT.values() for v in pair) * 1.2
    ys = lambda v: TOP + plot_h * (1 - v / top)
    baseline = TOP + plot_h

    parts = [text(L, 18, "Point-forecast error, averaged over 6 rolling windows", t["ink"], 13.5, weight="600"),
             text(L, 33, "Horizon 21 days · lower is better · the ensemble wins both measures", t["muted"], 11),
             legend([(m, MODEL_COLOURS[m]) for m in PUBLISHED_POINT], L, 52, ink=t["ink2"])]

    for tick in nice_ticks(top):
        y = ys(tick)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        parts.append(text(L - 8, y + 3.5, gbp_k(tick), t["muted"], 10.5, anchor="end"))

    group_w = plot_w / len(metrics)
    bar_w = (group_w - 150) / 3
    for gi, metric in enumerate(metrics):
        group_x = L + gi * group_w + 75
        for mi, (model, pair) in enumerate(PUBLISHED_POINT.items()):
            value = pair[gi]
            x = group_x + mi * (bar_w + 6)  # 6px surface gap between adjacent bars
            h = max(1.0, baseline - ys(value))
            parts.append(f'<path d="{bar_path(x, baseline - h, bar_w, h)}" fill="{MODEL_COLOURS[model]}"/>')
            parts.append(text(x + bar_w / 2, baseline - h - 8, f"{value:,}", t["ink"], 10.5,
                              anchor="middle", weight="600"))
        parts.append(text(group_x + (3 * bar_w + 12) / 2, H - B + 18, f"{metric} (£)", t["ink2"], 12,
                          anchor="middle", weight="500"))
    return svg(W, H, "".join(parts), "Point-forecast error by model",
               "Ensemble has the lowest MAE and RMSE, ahead of XGBoost and Chronos-Bolt.")


def fig_coverage(t: dict[str, str]) -> str:
    W, H = 760, 288
    L, R, TOP, B = 52, 120, 62, 40
    plot_w, plot_h = W - L - R, H - TOP - B
    ys = lambda v: TOP + plot_h * (1 - v / 100.0)
    baseline = TOP + plot_h

    parts = [text(L, 18, "Does the 80% prediction interval actually contain 80% of actuals?", t["ink"], 13.5, weight="600"),
             text(L, 33, "Share of held-out actuals inside the band · closest to the target wins, not the highest",
                  t["muted"], 11),
             legend([(m, MODEL_COLOURS[m]) for m in PUBLISHED_COVERAGE], L, 52, ink=t["ink2"])]

    for tick in (0, 20, 40, 60, 80, 100):
        y = ys(tick)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        parts.append(text(L - 8, y + 3.5, f"{tick}%", t["muted"], 10.5, anchor="end"))

    slot = plot_w / 3
    bar_w = slot - 90
    for i, (model, value) in enumerate(PUBLISHED_COVERAGE.items()):
        x = L + i * slot + (slot - bar_w) / 2
        h = max(1.0, baseline - ys(value))
        parts.append(f'<path d="{bar_path(x, baseline - h, bar_w, h)}" fill="{MODEL_COLOURS[model]}"/>')
        parts.append(text(x + bar_w / 2, baseline - h - 10, f"{value:.1f}%", t["ink"], 12.5,
                          anchor="middle", weight="700"))
        gap = value - NOMINAL_COVERAGE
        parts.append(text(x + bar_w / 2, H - B + 18, model, t["ink2"], 11.5, anchor="middle", weight="500"))
        parts.append(text(x + bar_w / 2, H - B + 33, f"{gap:+.1f} vs target", t["muted"], 10.5, anchor="middle"))

    y_target = ys(NOMINAL_COVERAGE)
    parts.append(f'<line x1="{L}" y1="{y_target:.1f}" x2="{W - R + 6}" y2="{y_target:.1f}" '
                 f'stroke="{t["ink"]}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    parts.append(text(W - R + 12, y_target - 4, "80% target", t["ink"], 11, weight="600"))
    parts.append(text(W - R + 12, y_target + 12, "nominal coverage", t["muted"], 10.5))
    return svg(W, H, "".join(parts), "Interval coverage against the 80% target",
               "Chronos-Bolt covers 81 percent, the ensemble 61.9, and XGBoost only 22.2 percent.")


# --- entry point ------------------------------------------------------------

def main() -> None:
    series = load_series()
    values = [v for _, v in series]
    means = weekday_means(series)
    trend = [v for v in rolling_mean(values, 30) if v is not None]

    figures = [
        ("fig-series", lambda theme: fig_series(series, theme)),
        ("fig-weekday", lambda theme: fig_weekday(means, theme)),
        ("fig-point-error", fig_point_error),
        ("fig-coverage", fig_coverage),
    ]

    for theme_name, theme in THEMES.items():
        for name, render in figures:
            content = render(theme)
            path = OUT_DIR / f"{name}-{theme_name}.svg"
            path.write_text(content, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)} ({len(content):,} bytes)")

    print("\nfigures computed from the CSV (cross-check against the README's EDA table):")
    print(f"  points            {len(values)}  ({series[0][0]} -> {series[-1][0]})")
    print(f"  30-day mean       first {trend[0]:,.0f} -> last {trend[-1]:,.0f}")
    for day, mean in zip(WEEKDAYS, means):
        print(f"  {day:<17} {mean:>10,.0f}")


if __name__ == "__main__":
    main()
