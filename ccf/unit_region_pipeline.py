#!/usr/bin/env python3
"""从 Neuropixels nrrd 时序数据提取 unit 特征并训练脑区反推模型。

示例：
python ccf/unit_region_pipeline.py \
  --nrrd-path /path/to/raw_signal.nrrd \
  --units-csv ccf/test/units.csv \
  --channels-csv ccf/test/channels.csv \
  --out-dir ccf/output
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import nrrd  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pynrrd. Install with `pip install pynrrd` and rerun."
    ) from exc


@dataclass
class UnitRecord:
    unit_id: str
    channel_id: int
    quality: str


class NearestCentroidClassifier:
    """轻量级基线分类器：按脑区质心做最近邻分类。"""

    def __init__(self) -> None:
        self.labels_: List[str] = []
        self.centroids_: Dict[str, np.ndarray] = {}

    def fit(self, x: np.ndarray, y: Sequence[str]) -> "NearestCentroidClassifier":
        grouped: Dict[str, List[np.ndarray]] = defaultdict(list)
        for row, label in zip(x, y):
            grouped[label].append(row)

        self.labels_ = sorted(grouped.keys())
        self.centroids_ = {
            label: np.mean(np.stack(rows, axis=0), axis=0) for label, rows in grouped.items()
        }
        return self

    def _distance_matrix(self, x: np.ndarray) -> np.ndarray:
        ordered_centroids = np.stack([self.centroids_[l] for l in self.labels_], axis=0)
        diff = x[:, None, :] - ordered_centroids[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))

    def predict(self, x: np.ndarray) -> List[str]:
        dist = self._distance_matrix(x)
        idx = np.argmin(dist, axis=1)
        return [self.labels_[i] for i in idx]

    def topk(self, x: np.ndarray, k: int = 3) -> List[List[str]]:
        dist = self._distance_matrix(x)
        idx = np.argsort(dist, axis=1)[:, :k]
        return [[self.labels_[i] for i in row] for row in idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unit 脑区反推管线")
    p.add_argument("--nrrd-path", required=True, help="原始电信号 nrrd 文件路径")
    p.add_argument("--units-csv", required=True)
    p.add_argument("--channels-csv", required=True)
    p.add_argument("--out-dir", default="ccf/output")
    p.add_argument(
        "--channel-index-col",
        default="local_index",
        help="channels.csv 中用于映射 nrrd 通道索引的列名（默认 local_index）",
    )
    p.add_argument("--channel-axis", type=int, default=0, help="nrrd 中 channel 所在维度")
    p.add_argument("--quality", default="good", help="仅使用该质量 unit；空字符串表示不过滤")
    p.add_argument("--min-units-per-region", type=int, default=20)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_channels(path: str, channel_index_col: str) -> Dict[int, Tuple[str, int]]:
    """返回 channel_id -> (region, nrrd_channel_index)."""
    mapping: Dict[int, Tuple[str, int]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            ch_id_raw = (row.get("id") or "").strip()
            region = (row.get("ecephys_structure_acronym") or "").strip()
            if not ch_id_raw or not region:
                continue
            try:
                ch_id = int(float(ch_id_raw))
            except ValueError:
                continue

            idx_raw = (row.get(channel_index_col) or "").strip()
            if idx_raw == "":
                nrrd_idx = i
            else:
                try:
                    nrrd_idx = int(float(idx_raw))
                except ValueError:
                    nrrd_idx = i
            mapping[ch_id] = (region, nrrd_idx)
    return mapping


def read_units(path: str, quality: str) -> List[UnitRecord]:
    units: List[UnitRecord] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unit_id = (row.get("id") or "").strip()
            ch_raw = (row.get("ecephys_channel_id") or "").strip()
            q = (row.get("quality") or "").strip()
            if not unit_id or not ch_raw:
                continue
            if quality and q != quality:
                continue
            try:
                channel_id = int(float(ch_raw))
            except ValueError:
                continue
            units.append(UnitRecord(unit_id=unit_id, channel_id=channel_id, quality=q))
    return units


def extract_features(signal: np.ndarray) -> np.ndarray:
    """从一个 unit 对应的原始时序（可多维）提取紧凑特征。"""
    x = np.asarray(signal, dtype=float).reshape(-1)
    if x.size == 0:
        return np.zeros(10, dtype=float)

    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x * x)))
    ptp = float(np.max(x) - np.min(x))

    centered = x - mean
    eps = 1e-8
    skew = float(np.mean(centered ** 3) / (std ** 3 + eps))
    kurt = float(np.mean(centered ** 4) / (std ** 4 + eps))

    signs = np.signbit(x)
    zero_cross = float(np.mean(signs[1:] != signs[:-1])) if x.size > 1 else 0.0

    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2
    power_sum = float(np.sum(power)) + eps
    dom_bin = int(np.argmax(power))
    dom_freq_norm = dom_bin / max(1, len(power) - 1)

    p = power / power_sum
    spectral_entropy = float(-np.sum(p * np.log(p + eps)))

    return np.array(
        [
            mean,
            std,
            rms,
            ptp,
            skew,
            kurt,
            zero_cross,
            dom_freq_norm,
            spectral_entropy,
            float(x.size),
        ],
        dtype=float,
    )


def zscore_fit_transform(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    sd[sd < 1e-8] = 1.0
    return (x - mu) / sd, mu, sd


def zscore_transform(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def stratified_split(
    x: np.ndarray, y: Sequence[str], test_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    rnd = random.Random(seed)
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(y):
        by_label[label].append(i)

    train_idx: List[int] = []
    test_idx: List[int] = []
    for indices in by_label.values():
        rnd.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_ratio)))
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])

    return x[train_idx], x[test_idx], [y[i] for i in train_idx], [y[i] for i in test_idx]


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    return float(sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true))


def topk_accuracy(y_true: Sequence[str], y_topk: Sequence[Sequence[str]]) -> float:
    if not y_true:
        return 0.0
    hit = 0
    for gt, preds in zip(y_true, y_topk):
        if gt in preds:
            hit += 1
    return hit / len(y_true)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    signal, header = nrrd.read(args.nrrd_path)
    signal = np.asarray(signal)
    signal = np.moveaxis(signal, args.channel_axis, 0)

    channels = read_channels(args.channels_csv, args.channel_index_col)
    units = read_units(args.units_csv, args.quality)

    rows = []
    x_rows = []
    y_rows = []

    for unit in units:
        ch_info = channels.get(unit.channel_id)
        if ch_info is None:
            continue
        region, nrrd_idx = ch_info
        if nrrd_idx < 0 or nrrd_idx >= signal.shape[0]:
            continue

        feats = extract_features(signal[nrrd_idx])
        row = {
            "unit_id": unit.unit_id,
            "channel_id": unit.channel_id,
            "region": region,
            **{f"f{i}": float(v) for i, v in enumerate(feats)},
        }
        rows.append(row)
        x_rows.append(feats)
        y_rows.append(region)

    if not rows:
        raise SystemExit("No valid unit samples after joining units/channels/nrrd index.")

    # 脑区最小样本过滤
    counts = Counter(y_rows)
    kept_labels = {k for k, v in counts.items() if v >= args.min_units_per_region}
    filt = [i for i, y in enumerate(y_rows) if y in kept_labels]
    if not filt:
        raise SystemExit("No regions satisfy min-units-per-region. Try lowering threshold.")

    x = np.asarray([x_rows[i] for i in filt], dtype=float)
    y = [y_rows[i] for i in filt]

    x_train, x_test, y_train, y_test = stratified_split(x, y, args.test_ratio, args.seed)
    x_train, mu, sd = zscore_fit_transform(x_train)
    x_test = zscore_transform(x_test, mu, sd)

    model = NearestCentroidClassifier().fit(x_train, y_train)
    pred = model.predict(x_test)
    pred_top3 = model.topk(x_test, k=min(3, len(model.labels_)))

    metrics = {
        "n_total": int(len(y)),
        "n_regions": int(len(set(y))),
        "region_counts": dict(sorted(Counter(y).items(), key=lambda kv: kv[0])),
        "top1_acc": accuracy(y_test, pred),
        "top3_acc": topk_accuracy(y_test, pred_top3),
        "nrrd_shape": list(signal.shape),
        "nrrd_header_keys": sorted(list(header.keys())),
    }

    features_csv = os.path.join(args.out_dir, "unit_features_with_region.csv")
    with open(features_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    metrics_json = os.path.join(args.out_dir, "region_inference_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Done")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved: {features_csv}")
    print(f"Saved: {metrics_json}")


if __name__ == "__main__":
    main()
