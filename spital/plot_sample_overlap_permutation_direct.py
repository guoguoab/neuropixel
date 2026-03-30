#!/usr/bin/env python3
"""基于已计算的 CSV 直接统计并绘图（跳过逐 sample 原始表计算）。

输入：
1) sample_pair_overlap_percent.csv
2) true_data_pair_overlap_percent_nonzero_pair.csv（或同结构 true percent 表）
3) sample_vs_true_permutation_pvalues.csv（可选；用于直接读取显著性）

输出：
- sample_vs_true_overlap_plot.svg
- sample_vs_true_permutation_pvalues.recomputed.csv（仅当未提供 --pvalue-csv 时）
"""

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
BINS = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
COLORS = ["#d9d2ad", "#b7d4ca", "#57a9a5", "#3d80ad", "#294f72"]
SAMPLE_BAR_COLOR = "#7b4ab8"
TRUE_LABEL_COLOR = "#1f1f1f"
SAMPLE_LABEL_COLOR = "#4b237a"


def load_true_distribution(path: Path) -> Dict[str, Dict[str, float]]:
    out = {p: {b: 0.0 for b in BINS} for p in PAIR_ORDER}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            p = (row.get("pair_type") or "").strip()
            b = (row.get("overlap_bin") or "").strip()
            percent = row.get("percent") or row.get("true_percent") or "0"
            if p in out and b in out[p]:
                out[p][b] = float(percent)
    return out


def load_sample_distribution(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sample = (row.get("sample") or "").strip()
            p = (row.get("pair_type") or "").strip()
            b = (row.get("overlap_bin") or "").strip()
            if sample and p in PAIR_ORDER and b in BINS:
                out.setdefault(sample, {pp: {bb: 0.0 for bb in BINS} for pp in PAIR_ORDER})
                out[sample][p][b] = float(row.get("percent") or 0.0)
    return out


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


def compute_stats(
    true_dist: Dict[str, Dict[str, float]],
    sample_dists: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    for p in PAIR_ORDER:
        for b in BINS:
            vals = [sample_dists[s][p][b] for s in sorted(sample_dists)]
            tv = true_dist[p][b]
            m = mean(vals)
            sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
            pv = sign_flip_pvalue([x - tv for x in vals])
            stats[(p, b)] = {"true": tv, "mean": m, "sd": sd, "p": pv, "sig": stars_for_p(pv)}
    return stats


def load_stats_from_pvalue_csv(
    path: Path,
    true_dist: Dict[str, Dict[str, float]],
    sample_dists: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    stats = compute_stats(true_dist, sample_dists)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            p = (row.get("pair_type") or "").strip()
            b = (row.get("overlap_bin") or "").strip()
            if (p, b) not in stats:
                continue
            stats[(p, b)]["p"] = float(row.get("p_value") or "nan")
            stats[(p, b)]["sig"] = (row.get("significance") or "").strip() or stars_for_p(stats[(p, b)]["p"])
    return stats


def write_recomputed_pvalue_csv(stats: Dict[Tuple[str, str], Dict[str, float]], out_csv: Path, sample_n: int) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["pair_type", "overlap_bin", "true_percent", "sample_n", "sample_mean", "sample_sd", "p_value", "significance"],
        )
        writer.writeheader()
        for p in PAIR_ORDER:
            for b in BINS:
                st = stats[(p, b)]
                writer.writerow(
                    {
                        "pair_type": p,
                        "overlap_bin": b,
                        "true_percent": f"{st['true']:.6f}",
                        "sample_n": sample_n,
                        "sample_mean": f"{st['mean']:.6f}",
                        "sample_sd": f"{st['sd']:.6f}",
                        "p_value": "nan" if math.isnan(st["p"]) else f"{st['p']:.6g}",
                        "significance": st["sig"],
                    }
                )


def draw_plot(true_dist, sample_dists, stats, out_svg: Path) -> None:
    width, height = 1500, 680
    left, right, top, bottom = 90, 40, 120, 110
    chart_w = width - left - right
    chart_h = height - top - bottom

    max_true = max(true_dist[p][b] for p in PAIR_ORDER for b in BINS)
    max_sample = max(sample_dists[s][p][b] for s in sample_dists for p in PAIR_ORDER for b in BINS)
    max_mean_sd = max(stats[(p, b)]["mean"] + stats[(p, b)]["sd"] for p in PAIR_ORDER for b in BINS)
    y_max = max(60.0, max_true, max_sample, max_mean_sd) + 15.0

    def y_to_px(v: float) -> float:
        return top + chart_h - (v / y_max) * chart_h

    group_w = chart_w / len(PLOT_PAIR_ORDER)
    bar_w = 18
    pair_gap = 3
    bin_gap = 8

    svg: List[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append('<text x="690" y="40" text-anchor="middle" font-size="26" font-weight="700">Sample vs True Overlap Percent (Permutation Test)</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#222" stroke-width="3"/>')
    svg.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#222" stroke-width="3"/>')

    for tick in range(0, int(y_max) + 1, 10):
        y = y_to_px(float(tick))
        svg.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#222" stroke-width="2"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="14">{tick}%</text>')

    lx, ly = 160, 72
    for i, b in enumerate(BINS):
        x = lx + i * 145
        svg.append(f'<rect x="{x}" y="{ly - 12}" width="24" height="14" fill="{COLORS[i]}" stroke="#222" stroke-width="1.6"/>')
        svg.append(f'<text x="{x + 32}" y="{ly}" font-size="15">{b}</text>')

    for g, pair in enumerate(PLOT_PAIR_ORDER):
        center = left + group_w * (g + 0.5)
        cluster_w = len(BINS) * (2 * bar_w + pair_gap) + (len(BINS) - 1) * bin_gap
        start_x = center - cluster_w / 2

        for i, b in enumerate(BINS):
            bin_start = start_x + i * (2 * bar_w + pair_gap + bin_gap)
            true_x = bin_start
            mean_x = bin_start + bar_w + pair_gap
            tv = true_dist[pair][b]
            y_true = y_to_px(tv)
            h_true = top + chart_h - y_true
            svg.append(f'<rect x="{true_x:.2f}" y="{y_true:.2f}" width="{bar_w}" height="{h_true:.2f}" fill="{COLORS[i]}" fill-opacity="0.35" stroke="#222" stroke-width="1.8"/>')
            svg.append(f'<text x="{true_x + bar_w / 2:.2f}" y="{max(12, y_true - 4):.2f}" text-anchor="middle" font-size="9" font-weight="600" fill="{TRUE_LABEL_COLOR}">{tv:.1f}%</text>')

            st = stats[(pair, b)]
            m = st["mean"]
            sd = st["sd"]
            y_mean = y_to_px(m)
            h_mean = top + chart_h - y_mean
            svg.append(f'<rect x="{mean_x:.2f}" y="{y_mean:.2f}" width="{bar_w}" height="{h_mean:.2f}" fill="{SAMPLE_BAR_COLOR}" fill-opacity="0.9" stroke="#222" stroke-width="1.8"/>')
            svg.append(f'<text x="{mean_x + bar_w / 2:.2f}" y="{max(12, y_mean - 4):.2f}" text-anchor="middle" font-size="9" font-weight="600" fill="{SAMPLE_LABEL_COLOR}">{m:.1f}%</text>')

            ex = mean_x + bar_w / 2
            y1 = y_to_px(max(0.0, m - sd))
            y2 = y_to_px(m + sd)
            svg.append(f'<line x1="{ex:.2f}" y1="{y1:.2f}" x2="{ex:.2f}" y2="{y2:.2f}" stroke="#111" stroke-width="2.2"/>')
            svg.append(f'<line x1="{ex - 4:.2f}" y1="{y1:.2f}" x2="{ex + 4:.2f}" y2="{y1:.2f}" stroke="#111" stroke-width="2.2"/>')
            svg.append(f'<line x1="{ex - 4:.2f}" y1="{y2:.2f}" x2="{ex + 4:.2f}" y2="{y2:.2f}" stroke="#111" stroke-width="2.2"/>')

            sig_y = y_to_px(max(tv, m + sd) + 5)
            sig_x = (true_x + mean_x + bar_w) / 2
            svg.append(f'<text x="{sig_x:.2f}" y="{max(16, sig_y):.2f}" text-anchor="middle" font-size="13" font-weight="700">{st["sig"]}</text>')

        svg.append(f'<text x="{center:.2f}" y="{top + chart_h + 45}" text-anchor="middle" font-size="20">{PAIR_LABELS[pair]}</text>')

    svg.append('</svg>')
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(svg), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Directly plot from precomputed sample/true/pvalue CSV files.")
    p.add_argument("--sample-csv", type=Path, default=Path("overlap_results/sample_pair_overlap_percent.csv"))
    p.add_argument("--true-csv", type=Path, default=Path("true_data_pair_overlap_percent_nonzero_pair.csv"))
    p.add_argument("--pvalue-csv", type=Path, default=Path("overlap_results/sample_vs_true_permutation_pvalues.csv"))
    p.add_argument("--out-svg", type=Path, default=Path("overlap_results/sample_vs_true_overlap_plot.svg"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for p in [args.sample_csv, args.true_csv]:
        if not p.exists():
            raise SystemExit(f"Missing required input: {p}")

    sample_dists = load_sample_distribution(args.sample_csv)
    if not sample_dists:
        raise SystemExit(f"No sample data loaded from: {args.sample_csv}")

    true_dist = load_true_distribution(args.true_csv)

    if args.pvalue_csv.exists():
        stats = load_stats_from_pvalue_csv(args.pvalue_csv, true_dist, sample_dists)
    else:
        stats = compute_stats(true_dist, sample_dists)
        out_csv = args.out_svg.with_name("sample_vs_true_permutation_pvalues.recomputed.csv")
        write_recomputed_pvalue_csv(stats, out_csv, sample_n=len(sample_dists))
        print(f"Recomputed p-value table -> {out_csv}")

    draw_plot(true_dist, sample_dists, stats, args.out_svg)
    print(f"Samples: {len(sample_dists)}")
    print(f"Plot -> {args.out_svg}")


if __name__ == "__main__":
    main()
