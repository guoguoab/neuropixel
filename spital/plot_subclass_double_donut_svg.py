#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def read_subclass_total(path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Read subclass totals from a semicolon-delimited file.

    Supports files with BOM and/or slightly different header spellings.
    """
    totals_by_name: Dict[str, float] = {}
    totals_by_code: Dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)
    if not rows:
        return totals_by_name, totals_by_code

    header = [h.strip().lower() for h in rows[0]]
    subclass_idx = 0
    number_idx = 1
    for i, h in enumerate(header):
        if "subclass" in h:
            subclass_idx = i
        if "number" in h or "total" in h:
            number_idx = i

    for row in rows[1:]:
        if len(row) <= max(subclass_idx, number_idx):
            continue
        name = row[subclass_idx].strip()
        number = row[number_idx].strip()
        try:
            value = float(number)
        except ValueError:
            continue
        totals_by_name[name] = value
        m = re.match(r"^(\d+)", name)
        if m:
            code = m.group(1).zfill(2)
            totals_by_code[code] = value
    return totals_by_name, totals_by_code


def read_class_subclass_code_map(data_dir: Path) -> Dict[str, str]:
    """Read mapping from class name to two-digit subclass code from data/*.txt files."""
    mapping: Dict[str, str] = {}
    if not data_dir.exists():
        return mapping
    for p in sorted(data_dir.glob("*.txt")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                cols = line.split("\t")
                if len(cols) < 14:
                    continue
                subclass = cols[12].strip()
                a_class = cols[13].strip()
                m = re.match(r"^(\d+)", subclass)
                if not m:
                    continue
                code = m.group(1).zfill(2)
                if a_class and a_class not in mapping:
                    mapping[a_class] = code
    return mapping


def read_partner_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_name = (row.get("sample_name") or "").strip()
            a_class = (row.get("a_class") or "").strip()
            a_merge_region = (row.get("a_merge_region") or "").strip()
            if not sample_name or not a_class:
                continue
            try:
                enrich = float(row.get("enrich_cell_num_sum") or 0)
            except ValueError:
                enrich = 0.0
            try:
                partner_n = float(row.get("partner_number") or 0)
            except ValueError:
                partner_n = 0.0
            rows.append(
                {
                    "sample_name": sample_name,
                    "a_class": a_class,
                    "a_merge_region": a_merge_region,
                    "enrich": enrich,
                    "partner_n": partner_n,
                }
            )
    return rows


def deduplicate_partner_rows(rows: List[dict]) -> List[dict]:
    """Deduplicate exact region rows caused by multiple B-side matches."""
    by_key: Dict[Tuple[str, str, str], dict] = {}
    for r in rows:
        key = (r["sample_name"], r["a_class"], r.get("a_merge_region", ""))
        if key not in by_key:
            by_key[key] = dict(r)
            continue
        by_key[key]["enrich"] = max(float(by_key[key]["enrich"]), float(r["enrich"]))
        by_key[key]["partner_n"] = max(float(by_key[key]["partner_n"]), float(r["partner_n"]))
    return list(by_key.values())


def normalize_to_class_level(rows: List[dict]) -> List[dict]:
    """Collapse to one row per (sample_name, a_class) to avoid repeated enrich inflation.

    Many inputs repeat the same class enrich value across a_merge_region. We therefore:
    - clustered = mean(enrich across unique regions for this class in this sample)
    - partner = clustered * (partner_region_count / region_count)
    - cluster_only = clustered - partner
    """
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["sample_name"], r["a_class"])].append(r)

    out: List[dict] = []
    for (sample_name, a_class), items in grouped.items():
        if not items:
            continue
        enrich_values = [float(x["enrich"]) for x in items]
        clustered = sum(enrich_values) / len(enrich_values)
        partner_regions = sum(1 for x in items if float(x["partner_n"]) > 0)
        frac = partner_regions / len(items)
        partner = clustered * frac
        cluster_only = clustered - partner
        out.append(
            {
                "sample_name": sample_name,
                "a_class": a_class,
                "clustered": clustered,
                "partner": partner,
                "cluster_only": cluster_only,
            }
        )
    return out


def fallback_class_code(a_class: str) -> str | None:
    m = re.match(r"^(\d+)", a_class)
    if not m:
        return None
    d = m.group(1)
    return d[:2].zfill(2)


def polar_to_xy(cx: float, cy: float, r: float, angle_deg: float) -> Tuple[float, float]:
    rad = math.radians(angle_deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def donut_segment_path(cx: float, cy: float, r_outer: float, r_inner: float, start_deg: float, end_deg: float) -> str:
    if abs(end_deg - start_deg) >= 360:
        end_deg = start_deg + 359.999
    x1o, y1o = polar_to_xy(cx, cy, r_outer, start_deg)
    x2o, y2o = polar_to_xy(cx, cy, r_outer, end_deg)
    x2i, y2i = polar_to_xy(cx, cy, r_inner, end_deg)
    x1i, y1i = polar_to_xy(cx, cy, r_inner, start_deg)
    large = 1 if (end_deg - start_deg) % 360 > 180 else 0
    return (
        f"M {x1o:.3f} {y1o:.3f} A {r_outer:.3f} {r_outer:.3f} 0 {large} 1 {x2o:.3f} {y2o:.3f} "
        f"L {x2i:.3f} {y2i:.3f} A {r_inner:.3f} {r_inner:.3f} 0 {large} 0 {x1i:.3f} {y1i:.3f} Z"
    )


def fmt(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def draw_double_donut(title: str, clustered: float, non_clustered: float, partner: float, cluster_only: float, cx: float, cy: float) -> str:
    total_outer = max(clustered + non_clustered, 1e-9)
    total_inner = max(partner + cluster_only, 1e-9)
    c_outer, nc_outer, p_inner, co_inner = "#4EA5D9", "#D99A17", "#2A9D75", "#8A8A8A"

    start = -90.0
    angle_clustered = 360.0 * (clustered / total_outer)
    angle_non = 360.0 - angle_clustered
    angle_partner = 360.0 * (partner / total_inner)
    angle_co = 360.0 - angle_partner

    pct_clustered = 100.0 * clustered / total_outer
    pct_non = 100.0 * non_clustered / total_outer
    pct_partner = 100.0 * partner / total_inner
    pct_co = 100.0 * cluster_only / total_inner

    return "\n".join([
        f'<text x="{cx}" y="45" text-anchor="middle" font-size="20" font-weight="700">{title}</text>',
        f'<path d="{donut_segment_path(cx, cy, 120, 80, start, start + angle_clustered)}" fill="{c_outer}" stroke="white" stroke-width="1.5"/>',
        f'<path d="{donut_segment_path(cx, cy, 120, 80, start + angle_clustered, start + angle_clustered + angle_non)}" fill="{nc_outer}" stroke="white" stroke-width="1.5"/>',
        f'<path d="{donut_segment_path(cx, cy, 74, 38, start, start + angle_partner)}" fill="{p_inner}" stroke="white" stroke-width="1.3"/>',
        f'<path d="{donut_segment_path(cx, cy, 74, 38, start + angle_partner, start + angle_partner + angle_co)}" fill="{co_inner}" stroke="white" stroke-width="1.3"/>',
        f'<text x="{cx}" y="215" text-anchor="middle" font-size="15">Clustered: {fmt(clustered)} ({fmt(pct_clustered)}%)</text>',
        f'<text x="{cx}" y="235" text-anchor="middle" font-size="15">Non-clustered: {fmt(non_clustered)} ({fmt(pct_non)}%)</text>',
        f'<text x="{cx}" y="255" text-anchor="middle" font-size="15">Partner: {fmt(partner)} ({fmt(pct_partner)}%)</text>',
        f'<text x="{cx}" y="275" text-anchor="middle" font-size="15">Cluster-only: {fmt(cluster_only)} ({fmt(pct_co)}%)</text>',
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("overlap_results/sample_true_partner_summary.csv"))
    parser.add_argument("--subclass-total", type=Path, default=Path("subclass_total_summary.txt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("overlap_results/a_class_double_donut"))
    parser.add_argument("--stats-out", type=Path, default=Path("overlap_results/a_class_double_donut_stats.csv"))
    args = parser.parse_args()

    totals_by_name, totals_by_code = read_subclass_total(args.subclass_total)
    class_to_subcode = read_class_subclass_code_map(args.data_dir)
    rows_raw = read_partner_rows(args.summary)
    rows_dedup = deduplicate_partner_rows(rows_raw)
    rows = normalize_to_class_level(rows_dedup)

    a_classes = sorted({r["a_class"] for r in rows})
    sample_names = sorted({r["sample_name"] for r in rows if r["sample_name"].startswith("sample_")})

    true_clustered = defaultdict(float)
    true_partner = defaultdict(float)
    true_cluster_only = defaultdict(float)
    sample_clustered = defaultdict(lambda: defaultdict(float))
    sample_partner = defaultdict(lambda: defaultdict(float))
    sample_cluster_only = defaultdict(lambda: defaultdict(float))

    for r in rows:
        sname, a_class = r["sample_name"], r["a_class"]
        clustered = float(r["clustered"])
        partner = float(r["partner"])
        cluster_only = float(r["cluster_only"])
        if sname == "true_data":
            true_clustered[a_class] += clustered
            true_partner[a_class] += partner
            true_cluster_only[a_class] += cluster_only
        elif sname.startswith("sample_"):
            sample_clustered[a_class][sname] += clustered
            sample_partner[a_class][sname] += partner
            sample_cluster_only[a_class][sname] += cluster_only

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_rows: List[dict] = []

    for a_class in a_classes:
        sub_code = class_to_subcode.get(a_class) or fallback_class_code(a_class) or ""
        total = totals_by_name.get(a_class, totals_by_code.get(sub_code, 0.0))

        t_clustered = true_clustered[a_class]
        t_non = max(total - t_clustered, 0.0)
        t_partner = true_partner[a_class]
        t_cluster_only = true_cluster_only[a_class]

        n = len(sample_names)
        s_clustered_mean = sum(sample_clustered[a_class].get(sn, 0.0) for sn in sample_names) / n if n else 0.0
        s_partner_mean = sum(sample_partner[a_class].get(sn, 0.0) for sn in sample_names) / n if n else 0.0
        s_cluster_only_mean = sum(sample_cluster_only[a_class].get(sn, 0.0) for sn in sample_names) / n if n else 0.0
        s_non_mean = max(total - s_clustered_mean, 0.0)

        width, height = 980, 330
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width/2}" y="25" text-anchor="middle" font-size="22" font-weight="700">{a_class} | subclass={sub_code} total={fmt(total)}</text>',
            draw_double_donut("true_data", t_clustered, t_non, t_partner, t_cluster_only, 250, 180),
            draw_double_donut(f"sample_mean (n={len(sample_names)})", s_clustered_mean, s_non_mean, s_partner_mean, s_cluster_only_mean, 730, 180),
            '<rect x="390" y="105" width="16" height="16" fill="#4EA5D9"/><text x="412" y="118" font-size="14">clustered</text>',
            '<rect x="390" y="130" width="16" height="16" fill="#D99A17"/><text x="412" y="143" font-size="14">non-clustered</text>',
            '<rect x="390" y="155" width="16" height="16" fill="#2A9D75"/><text x="412" y="168" font-size="14">partner</text>',
            '<rect x="390" y="180" width="16" height="16" fill="#8A8A8A"/><text x="412" y="193" font-size="14">cluster-only</text>',
            "</svg>",
        ]

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", a_class).strip("_")
        (args.output_dir / f"{safe_name}.svg").write_text("\n".join(parts), encoding="utf-8")

        stats_rows.append({
            "a_class": a_class,
            "subclass_code": sub_code,
            "total": total,
            "true_clustered": t_clustered,
            "true_non_clustered": t_non,
            "true_partner": t_partner,
            "true_cluster_only": t_cluster_only,
            "sample_clustered_mean": s_clustered_mean,
            "sample_non_clustered_mean": s_non_mean,
            "sample_partner_mean": s_partner_mean,
            "sample_cluster_only_mean": s_cluster_only_mean,
            "sample_count": len(sample_names),
        })

    with args.stats_out.open("w", encoding="utf-8", newline="") as f:
        fields = list(stats_rows[0].keys()) if stats_rows else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(stats_rows)

    print(
        f"Input rows: {len(rows_raw)} | Region-dedup rows: {len(rows_dedup)} | "
        f"Class-level rows: {len(rows)}"
    )
    print(f"Generated {len(a_classes)} SVG files in {args.output_dir}")
    print(f"Sample count used for mean: {len(sample_names)}")
    print(f"Mapped classes from data dir: {len(class_to_subcode)}")
    print(f"Wrote stats table: {args.stats_out}")


if __name__ == "__main__":
    main()
