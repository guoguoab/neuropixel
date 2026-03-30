#!/usr/bin/env python3
"""对 overlap_results 下所有 sample 总表进行百分比统计，与 true_data 做逐柱置换检验并绘图。

功能：
1) 读取 true_data_overlap_results/true_data_pair_overlap_percent.csv 作为真实值。
2) 读取 overlap_results/*_cellid_overlap_summary_filtered.csv，按 compute_true_data_overlap_plot.py 的逻辑
   计算每个 sample 在 4 个 pair_type x 5 个 overlap_bin 的百分比。
3) 对每个 (pair_type, overlap_bin) 做 one-sample sign-flip permutation test，比较 sample 百分比与 true 百分比。
4) 输出：
   - sample_pair_overlap_percent.csv
   - sample_vs_true_permutation_pvalues.csv
   - sample_vs_true_overlap_plot.svg（柱状图 + sample 点 + 误差线 + 显著性）
"""

from __future__ import annotations

import argparse
import csv
import math
from multiprocessing import Pool
import random
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


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


def load_type_mapping(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            cls = (row.get("class") or "").strip()
            neuron = (row.get("cell_Neuron_type") or "").strip()
            if cls and neuron:
                mapping[cls] = neuron
    return mapping


def normalize_type(value: str | None) -> str:
    token = (value or "").strip()
    if not token:
        return "Unknown"
    low = token.lower()
    if low == "gaba":
        return "Gaba"
    if low == "glut":
        return "Glut"
    if low == "nonneuron":
        return "NonNeuron"

    # 从 class 文本中兜底识别
    if "gaba" in low:
        return "Gaba"
    if "glut" in low:
        return "Glut"
    if "non" in low and "neuron" in low:
        return "NonNeuron"
    return token


def fill_type(raw_type: str | None, cls: str, mapping: Dict[str, str]) -> str:
    ntype = normalize_type(raw_type)
    if ntype != "Unknown":
        return ntype
    return normalize_type(mapping.get(cls, cls))


def overlap_bin(x: float) -> str:
    if x < 0.2:
        return "0–20%"
    if x < 0.4:
        return "20–40%"
    if x < 0.6:
        return "40–60%"
    if x < 0.8:
        return "60–80%"
    return "80–100%"


def pair_type_counts(total_rows: Iterable[Dict[str, object]]) -> Dict[str, int]:
    counts = {"Gaba-Gaba": 0, "Gaba-Glut": 0, "Glut-Gaba": 0, "Glut-Glut": 0}
    for row in total_rows:
        key = f"{row['a_type']}-{row['b_type']}"
        if key in counts:
            counts[key] += 1
    return counts


def collect_directional_values(total_rows: Iterable[Dict[str, object]], pair_name: str) -> List[float]:
    values: List[float] = []
    for row in total_rows:
        ab = f"{row['a_type']}-{row['b_type']}"
        a2b = float(row["overlap_a_in_b"])
        b2a = float(row["overlap_b_in_a"])

        if pair_name in {"Gaba-Gaba", "Glut-Glut"} and ab == pair_name:
            values.extend([a2b, b2a])
            continue

        if pair_name == "Gaba-Glut":
            if ab == "Gaba-Glut":
                values.append(a2b)
            elif ab == "Glut-Gaba":
                values.append(b2a)
        elif pair_name == "Glut-Gaba":
            if ab == "Glut-Gaba":
                values.append(a2b)
            elif ab == "Gaba-Glut":
                values.append(b2a)
    return values


def denominator_for_pair(pair_name: str, counts: Dict[str, int]) -> int:
    if pair_name == "Gaba-Gaba":
        return 2 * counts["Gaba-Gaba"]
    if pair_name == "Glut-Glut":
        return 2 * counts["Glut-Glut"]
    return counts["Gaba-Glut"] + counts["Glut-Gaba"]


def compute_distribution(values: List[float], denominator: int) -> Dict[str, float]:
    counts = {b: 0 for b in BINS}
    if denominator <= 0:
        return {b: 0.0 for b in BINS}

    for v in values:
        counts[overlap_bin(v)] += 1
    return {b: counts[b] * 100.0 / denominator for b in BINS}


def read_total_table(path: Path, mapping: Dict[str, str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            a_class = (row.get("a_class") or "").strip()
            b_class = (row.get("b_class") or "").strip()
            a_type = fill_type(row.get("a_cell_Neuron_type"), a_class, mapping)
            b_type = fill_type(row.get("b_cell_Neuron_type"), b_class, mapping)
            try:
                oa = float(row.get("overlap_a_in_b") or "")
                ob = float(row.get("overlap_b_in_a") or "")
            except ValueError:
                continue

            # 仅保留至少一个方向重叠比例大于 0 的记录
            if oa <= 0 and ob <= 0:
                continue

            rows.append({"a_type": a_type, "b_type": b_type, "overlap_a_in_b": oa, "overlap_b_in_a": ob})
    return rows


def compute_sample_distribution(path: Path, mapping: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    total_rows = read_total_table(path, mapping)
    counts = pair_type_counts(total_rows)
    dist_by_pair: Dict[str, Dict[str, float]] = {}
    for pair_name in PAIR_ORDER:
        vals = collect_directional_values(total_rows, pair_name)
        den = denominator_for_pair(pair_name, counts)
        dist_by_pair[pair_name] = compute_distribution(vals, den)
    return dist_by_pair


def compute_sample_distribution_task(task: Tuple[Path, Dict[str, str]]) -> Tuple[str, Dict[str, Dict[str, float]]]:
    path, mapping = task
    sample_name = path.stem.replace("_cellid_overlap_summary_filtered", "")
    return sample_name, compute_sample_distribution(path, mapping)


def load_true_distribution(path: Path) -> Dict[str, Dict[str, float]]:
    out = {p: {b: 0.0 for b in BINS} for p in PAIR_ORDER}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            p = row.get("pair_type", "")
            b = row.get("overlap_bin", "")
            if p in out and b in out[p]:
                out[p][b] = float(row.get("percent") or 0.0)
    return out


def sign_flip_pvalue(diffs: List[float], n_perm: int = 10000, seed: int = 0) -> float:
    n = len(diffs)
    if n == 0:
        return float("nan")
    observed = abs(mean(diffs))

    # 样本数较小时做精确置换（2^n）
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


def write_sample_table(sample_dists: Dict[str, Dict[str, Dict[str, float]]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample", "pair_type", "overlap_bin", "percent"])
        writer.writeheader()
        for sample, dist_by_pair in sorted(sample_dists.items()):
            for p in PAIR_ORDER:
                for b in BINS:
                    writer.writerow({"sample": sample, "pair_type": p, "overlap_bin": b, "percent": dist_by_pair[p][b]})


def write_pvalue_table(
    true_dist: Dict[str, Dict[str, float]],
    sample_dists: Dict[str, Dict[str, Dict[str, float]]],
    out_csv: Path,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    results: Dict[Tuple[str, str], Dict[str, float]] = {}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "pair_type",
            "overlap_bin",
            "true_percent",
            "sample_n",
            "sample_mean",
            "sample_sd",
            "p_value",
            "significance",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for p in PAIR_ORDER:
            for b in BINS:
                vals = [sample_dists[s][p][b] for s in sorted(sample_dists)]
                tv = true_dist[p][b]
                sd = math.sqrt(sum((x - mean(vals)) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
                pv = sign_flip_pvalue([x - tv for x in vals])
                sig = stars_for_p(pv)
                row = {
                    "pair_type": p,
                    "overlap_bin": b,
                    "true_percent": f"{tv:.6f}",
                    "sample_n": len(vals),
                    "sample_mean": f"{mean(vals):.6f}",
                    "sample_sd": f"{sd:.6f}",
                    "p_value": "nan" if math.isnan(pv) else f"{pv:.6g}",
                    "significance": sig,
                }
                writer.writerow(row)
                results[(p, b)] = {
                    "true": tv,
                    "mean": mean(vals),
                    "sd": sd,
                    "p": pv,
                    "sig": sig,
                }
    return results


def draw_plot(
    true_dist: Dict[str, Dict[str, float]],
    sample_dists: Dict[str, Dict[str, Dict[str, float]]],
    stats: Dict[Tuple[str, str], Dict[str, float]],
    out_svg: Path,
) -> None:
    width, height = 1500, 680
    left, right, top, bottom = 90, 40, 120, 110
    chart_w = width - left - right
    chart_h = height - top - bottom

    max_true = max(true_dist[p][b] for p in PAIR_ORDER for b in BINS)
    max_sample = max(sample_dists[s][p][b] for s in sample_dists for p in PAIR_ORDER for b in BINS)
    max_mean_sd = max(
        stats[(p, b)]["mean"] + stats[(p, b)]["sd"]
        for p in PAIR_ORDER
        for b in BINS
    )
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

    # axis
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#222" stroke-width="3"/>')
    svg.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#222" stroke-width="3"/>')

    for tick in range(0, int(y_max) + 1, 10):
        y = y_to_px(float(tick))
        svg.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#222" stroke-width="2"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="14">{tick}%</text>')

    # legend
    lx, ly = 160, 72
    for i, b in enumerate(BINS):
        x = lx + i * 145
        svg.append(f'<rect x="{x}" y="{ly - 12}" width="24" height="14" fill="{COLORS[i]}" stroke="#222" stroke-width="1.6"/>')
        svg.append(f'<text x="{x + 32}" y="{ly}" font-size="15">{b}</text>')
    svg.append('<rect x="1040" y="54" width="18" height="14" fill="#7f7f7f" fill-opacity="0.35" stroke="#222" stroke-width="1.6"/>')
    svg.append('<text x="1066" y="66" font-size="14">true</text>')
    svg.append(f'<rect x="1120" y="54" width="18" height="14" fill="{SAMPLE_BAR_COLOR}" fill-opacity="0.9" stroke="#222" stroke-width="1.6"/>')
    svg.append('<text x="1146" y="66" font-size="14">sample mean</text>')
    svg.append('<line x1="1248" y1="61" x2="1276" y2="61" stroke="#111" stroke-width="2.6"/>')
    svg.append('<line x1="1262" y1="53" x2="1262" y2="69" stroke="#111" stroke-width="2.6"/>')
    svg.append('<text x="1284" y="66" font-size="14">sample mean ± SD</text>')

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
            svg.append(
                f'<rect x="{true_x:.2f}" y="{y_true:.2f}" width="{bar_w}" height="{h_true:.2f}" '
                f'fill="{COLORS[i]}" fill-opacity="0.35" stroke="#222" stroke-width="1.8"/>'
            )
            true_label_y = max(12, y_true - 4)
            svg.append(
                f'<text x="{true_x + bar_w / 2:.2f}" y="{true_label_y:.2f}" text-anchor="middle" '
                f'font-size="9" font-weight="600" fill="{TRUE_LABEL_COLOR}">{tv:.1f}%</text>'
            )

            stat = stats[(pair, b)]
            m = stat["mean"]
            sd = stat["sd"]
            y_mean = y_to_px(m)
            h_mean = top + chart_h - y_mean
            svg.append(
                f'<rect x="{mean_x:.2f}" y="{y_mean:.2f}" width="{bar_w}" height="{h_mean:.2f}" '
                f'fill="{SAMPLE_BAR_COLOR}" fill-opacity="0.9" stroke="#222" stroke-width="1.8"/>'
            )
            mean_label_y = max(12, y_mean - 4)
            svg.append(
                f'<text x="{mean_x + bar_w / 2:.2f}" y="{mean_label_y:.2f}" text-anchor="middle" '
                f'font-size="9" font-weight="600" fill="{SAMPLE_LABEL_COLOR}">{m:.1f}%</text>'
            )

            ex = mean_x + bar_w / 2
            y1 = y_to_px(max(0.0, m - sd))
            y2 = y_to_px(m + sd)
            svg.append(f'<line x1="{ex:.2f}" y1="{y1:.2f}" x2="{ex:.2f}" y2="{y2:.2f}" stroke="#111" stroke-width="2.2"/>')
            svg.append(f'<line x1="{ex - 4:.2f}" y1="{y1:.2f}" x2="{ex + 4:.2f}" y2="{y1:.2f}" stroke="#111" stroke-width="2.2"/>')
            svg.append(f'<line x1="{ex - 4:.2f}" y1="{y2:.2f}" x2="{ex + 4:.2f}" y2="{y2:.2f}" stroke="#111" stroke-width="2.2"/>')

            sig_y = y_to_px(max(tv, m + sd) + 5)
            sig_x = (true_x + mean_x + bar_w) / 2
            svg.append(f'<text x="{sig_x:.2f}" y="{max(16, sig_y):.2f}" text-anchor="middle" font-size="13" font-weight="700">{stat["sig"]}</text>')

        svg.append(f'<text x="{center:.2f}" y="{top + chart_h + 45}" text-anchor="middle" font-size="20">{PAIR_LABELS[pair]}</text>')

    svg.append('</svg>')

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(svg), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute sample overlap percentages, permutation p-values, and plot.")
    parser.add_argument("--overlap-dir", type=Path, default=Path("overlap_results"))
    parser.add_argument("--true-percent", type=Path, default=Path("true_data_overlap_results/true_data_pair_overlap_percent.csv"))
    parser.add_argument("--mapping", type=Path, default=Path("Merfish_brain_cell_type_subclass.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("overlap_results"))
    parser.add_argument("--sample-pattern", type=str, default="*_cellid_overlap_summary_filtered.csv")
    parser.add_argument("--jobs", type=int, default=20, help="并行处理 sample 文件的进程数（默认 20）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.true_percent.exists():
        raise SystemExit(f"Missing true percent file: {args.true_percent}. Please run compute_true_data_overlap_plot.py first.")

    mapping = load_type_mapping(args.mapping)
    true_dist = load_true_distribution(args.true_percent)

    sample_files = sorted(args.overlap_dir.glob(args.sample_pattern))
    if not sample_files:
        raise SystemExit(f"No sample files matched: {args.overlap_dir / args.sample_pattern}")

    jobs = max(1, args.jobs)
    tasks = [(path, mapping) for path in sample_files]
    sample_dists: Dict[str, Dict[str, Dict[str, float]]] = {}
    total = len(tasks)
    with Pool(processes=jobs) as pool:
        for idx, (sample_name, dist) in enumerate(pool.imap_unordered(compute_sample_distribution_task, tasks), start=1):
            sample_dists[sample_name] = dist
            print(f"[{idx}/{total}] Processed sample: {sample_name}")

    sample_table = args.out_dir / "sample_pair_overlap_percent.csv"
    pvalue_table = args.out_dir / "sample_vs_true_permutation_pvalues.csv"
    plot_file = args.out_dir / "sample_vs_true_overlap_plot.svg"

    write_sample_table(sample_dists, sample_table)
    stats = write_pvalue_table(true_dist, sample_dists, pvalue_table)
    draw_plot(true_dist, sample_dists, stats, plot_file)

    print(f"Samples: {len(sample_dists)}")
    print(f"Sample percent table -> {sample_table}")
    print(f"Permutation p-value table -> {pvalue_table}")
    print(f"Plot -> {plot_file}")


if __name__ == "__main__":
    main()
