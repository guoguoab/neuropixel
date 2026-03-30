#!/usr/bin/env python3
"""基于 sample_pair_bin_correlations.csv 绘制 true_data vs 随机 sample 的相关性统计图。"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

PAIR_ORDER = ["Gaba-Gaba", "Gaba-Glut", "Glut-Gaba", "Glut-Glut"]
PLOT_PAIR_ORDER = ["Gaba-Gaba", "Glut-Gaba", "Gaba-Glut", "Glut-Glut"]
PAIR_LABELS = {
    "Gaba-Gaba": "GABA-GABA",
    "Gaba-Glut": "GABA-Glut",
    "Glut-Gaba": "Glut-GABA",
    "Glut-Glut": "Glut-Glut",
}
BINS = ["0-20", "20-40", "40-60", "60-80", "80-100"]
BIN_LABELS = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
COLORS = ["#d9d2ad", "#cdd8d8", "#b7d4ca", "#aac8de", "#a4b4c5"]
SAMPLE_BAR_COLOR = "#7b4ab8"


def sign_flip_pvalue(diffs: List[float], n_perm: int = 10000, seed: int = 0) -> float:
    n = len(diffs)
    if n == 0:
        return float("nan")
    observed = abs(mean(diffs))

    if n <= 16:
        total = 1 << n
        extreme = 0
        for mask in range(total):
            s = 0.0
            for i, d in enumerate(diffs):
                sign = -1.0 if ((mask >> i) & 1) else 1.0
                s += sign * d
            if abs(s / n) >= observed - 1e-12:
                extreme += 1
        return (extreme + 1.0) / (total + 1.0)

    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_perm):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        if abs(s / n) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1.0) / (n_perm + 1.0)


def stars_for_p(p: float) -> str:
    if math.isnan(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def load_values(path: Path, metric: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]]:
    true_values = {p: {b: float("nan") for b in BINS} for p in PAIR_ORDER}
    sample_values: Dict[str, Dict[str, Dict[str, float]]] = {}

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sample = (row.get("sample") or "").strip()
            pair_type = (row.get("pair_type") or "").strip()
            overlap_bin = (row.get("overlap_bin") or "").strip()
            if pair_type not in PAIR_ORDER or overlap_bin not in BINS:
                continue

            v_raw = (row.get(metric) or "").strip()
            if not v_raw:
                continue
            try:
                value = float(v_raw)
            except ValueError:
                continue

            if sample == "true_data":
                true_values[pair_type][overlap_bin] = value
            elif sample:
                sample_values.setdefault(sample, {p: {b: float("nan") for b in BINS} for p in PAIR_ORDER})
                sample_values[sample][pair_type][overlap_bin] = value

    return true_values, sample_values


def compute_stats(true_values, sample_values):
    stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    for p in PAIR_ORDER:
        for b in BINS:
            tv = true_values[p][b]
            vals = [sample_values[s][p][b] for s in sorted(sample_values) if not math.isnan(sample_values[s][p][b])]
            if vals:
                m = mean(vals)
                sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
                pv = sign_flip_pvalue([x - tv for x in vals]) if not math.isnan(tv) else float("nan")
            else:
                m = float("nan")
                sd = float("nan")
                pv = float("nan")
            stats[(p, b)] = {"true": tv, "mean": m, "sd": sd, "p": pv, "sig": stars_for_p(pv), "n": len(vals)}
    return stats


def draw_plot(metric: str, metric_label: str, stats, out_svg: Path) -> None:
    width, height = 1500, 680
    left, right, top, bottom = 90, 40, 120, 110
    chart_w = width - left - right
    chart_h = height - top - bottom

    values = []
    for p in PAIR_ORDER:
        for b in BINS:
            st = stats[(p, b)]
            for k in ["true", "mean"]:
                if not math.isnan(st[k]):
                    values.append(st[k])
            if not math.isnan(st["mean"]) and not math.isnan(st["sd"]):
                values.extend([st["mean"] - st["sd"], st["mean"] + st["sd"]])

    vmin = min(values) if values else -1.0
    vmax = max(values) if values else 1.0
    max_abs = max(abs(vmin), abs(vmax), 0.25)
    y_lim = max_abs + 0.15

    def y_to_px(v: float) -> float:
        return top + chart_h - ((v + y_lim) / (2 * y_lim)) * chart_h

    group_w = chart_w / len(PLOT_PAIR_ORDER)
    bar_w = 18
    pair_gap = 3
    bin_gap = 8

    svg: List[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(f'<text x="750" y="40" text-anchor="middle" font-size="26" font-weight="700">Sample vs True {metric_label} (Permutation Test)</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#222" stroke-width="3"/>')
    svg.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#222" stroke-width="3"/>')

    zero_y = y_to_px(0.0)
    svg.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + chart_w}" y2="{zero_y:.2f}" stroke="#666" stroke-width="1.6" stroke-dasharray="5,4"/>')

    tick_step = 0.2
    t = -1.0
    while t <= 1.0001:
        y = y_to_px(t)
        svg.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#222" stroke-width="2"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="14">{t:.1f}</text>')
        t += tick_step

    lx, ly = 140, 72
    for i, b in enumerate(BIN_LABELS):
        x = lx + i * 145
        svg.append(f'<rect x="{x}" y="{ly - 12}" width="24" height="14" fill="{COLORS[i]}" stroke="#222" stroke-width="1.6"/>')
        svg.append(f'<text x="{x + 32}" y="{ly}" font-size="15">{b}</text>')
    x = lx + len(BIN_LABELS) * 145
    svg.append(f'<rect x="{x}" y="{ly - 12}" width="24" height="14" fill="{SAMPLE_BAR_COLOR}" stroke="#222" stroke-width="1.6"/>')
    svg.append(f'<text x="{x + 32}" y="{ly}" font-size="15">random sample</text>')

    for g, pair in enumerate(PLOT_PAIR_ORDER):
        center = left + group_w * (g + 0.5)
        cluster_w = len(BINS) * (2 * bar_w + pair_gap) + (len(BINS) - 1) * bin_gap
        start_x = center - cluster_w / 2

        for i, b in enumerate(BINS):
            bin_start = start_x + i * (2 * bar_w + pair_gap + bin_gap)
            true_x = bin_start
            mean_x = bin_start + bar_w + pair_gap

            st = stats[(pair, b)]
            tv, m, sd = st["true"], st["mean"], st["sd"]

            if not math.isnan(tv):
                y_true = y_to_px(tv)
                y_base = y_to_px(0.0)
                h_true = abs(y_base - y_true)
                y_true_top = min(y_true, y_base)
                svg.append(
                    f'<rect x="{true_x:.2f}" y="{y_true_top:.2f}" width="{bar_w}" height="{h_true:.2f}" fill="{COLORS[i]}" fill-opacity="0.35" stroke="#222" stroke-width="1.8"/>'
                )
                label_y = y_true - 4 if tv >= 0 else y_true + 14
                svg.append(f'<text x="{true_x + bar_w / 2:.2f}" y="{label_y:.2f}" text-anchor="middle" font-size="9" font-weight="600">{tv:.2f}</text>')

            if not math.isnan(m):
                y_mean = y_to_px(m)
                y_base = y_to_px(0.0)
                h_mean = abs(y_base - y_mean)
                y_mean_top = min(y_mean, y_base)
                svg.append(
                    f'<rect x="{mean_x:.2f}" y="{y_mean_top:.2f}" width="{bar_w}" height="{h_mean:.2f}" fill="{SAMPLE_BAR_COLOR}" fill-opacity="0.9" stroke="#222" stroke-width="1.8"/>'
                )
                label_y = y_mean - 4 if m >= 0 else y_mean + 14
                svg.append(f'<text x="{mean_x + bar_w / 2:.2f}" y="{label_y:.2f}" text-anchor="middle" font-size="9" font-weight="600" fill="#4b237a">{m:.2f}</text>')

            if not math.isnan(m) and not math.isnan(sd):
                ex = mean_x + bar_w / 2
                y1 = y_to_px(m - sd)
                y2 = y_to_px(m + sd)
                svg.append(f'<line x1="{ex:.2f}" y1="{y1:.2f}" x2="{ex:.2f}" y2="{y2:.2f}" stroke="#111" stroke-width="2.2"/>')
                svg.append(f'<line x1="{ex - 4:.2f}" y1="{y1:.2f}" x2="{ex + 4:.2f}" y2="{y1:.2f}" stroke="#111" stroke-width="2.2"/>')
                svg.append(f'<line x1="{ex - 4:.2f}" y1="{y2:.2f}" x2="{ex + 4:.2f}" y2="{y2:.2f}" stroke="#111" stroke-width="2.2"/>')

            if not math.isnan(tv) and not math.isnan(m):
                sig_anchor = max(tv, m + (0.0 if math.isnan(sd) else sd)) + 0.08
                sig_y = y_to_px(min(sig_anchor, y_lim - 0.02))
                sig_x = (true_x + mean_x + bar_w) / 2
                svg.append(f'<text x="{sig_x:.2f}" y="{max(16, sig_y):.2f}" text-anchor="middle" font-size="13" font-weight="700">{st["sig"]}</text>')

        svg.append(f'<text x="{center:.2f}" y="{top + chart_h + 45}" text-anchor="middle" font-size="20">{PAIR_LABELS[pair]}</text>')

    svg.append('</svg>')
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(svg), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="绘制相关性指标 true_data vs 随机 sample 对比图")
    p.add_argument("--input-csv", type=Path, default=Path("overlap_results/sample_pair_bin_correlations.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("overlap_results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Missing input: {args.input_csv}")

    metrics = [
        ("log2_cell_id_pearson_r", "log2(cell_id) Pearson r", "sample_vs_true_log2_cell_id_pearson_r.svg"),
        ("ei_pearson_r", "E-I Pearson r", "sample_vs_true_ei_pearson_r.svg"),
    ]

    for metric, label, out_name in metrics:
        true_values, sample_values = load_values(args.input_csv, metric)
        if not sample_values:
            raise SystemExit(f"No random sample rows loaded for {metric}")
        stats = compute_stats(true_values, sample_values)
        out_svg = args.out_dir / out_name
        draw_plot(metric, label, stats, out_svg)
        print(f"{metric}: samples={len(sample_values)} -> {out_svg}")


if __name__ == "__main__":
    main()
