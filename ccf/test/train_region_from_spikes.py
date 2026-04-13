"""Train a brain-region classifier from Neuropixels unit spikes.

This script is designed for Allen Neuropixels session files such as:
- session_715093703/session_715093703.nwb
- units.csv
- channels.csv

Workflow:
1) Read unit metadata from CSV files and build region labels per unit.
2) Extract spike times per unit from the NWB file.
3) Convert spike trains into fixed-length statistical features.
4) Train and evaluate a classification model to predict brain region.

Dependencies (install in your local environment):
    pip install numpy pandas scikit-learn pynwb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train model to infer unit brain region from spike trains."
    )
    parser.add_argument("--nwb", type=Path, required=True, help="Path to session_*.nwb")
    parser.add_argument("--units-csv", type=Path, required=True, help="Path to units.csv")
    parser.add_argument("--channels-csv", type=Path, required=True, help="Path to channels.csv")
    parser.add_argument(
        "--min-units-per-region",
        type=int,
        default=30,
        help="Keep only regions with at least this many units.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Proportion of held-out units."
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_region_model"),
        help="Directory to save metrics/artifacts.",
    )
    return parser.parse_args()


def load_labels(units_csv: Path, channels_csv: Path) -> pd.DataFrame:
    units_df = pd.read_csv(units_csv)
    channels_df = pd.read_csv(channels_csv)

    merged = units_df.merge(
        channels_df[["id", "ecephys_structure_acronym"]],
        left_on="ecephys_channel_id",
        right_on="id",
        how="left",
        suffixes=("_unit", "_channel"),
    )

    merged = merged.rename(columns={"id_unit": "unit_id", "ecephys_structure_acronym": "region"})
    merged = merged[["unit_id", "ecephys_channel_id", "quality", "region"]].copy()
    merged = merged.dropna(subset=["unit_id", "region"])
    return merged


def load_spike_times_from_nwb(nwb_path: Path) -> Dict[int, np.ndarray]:
    from pynwb import NWBHDF5IO

    spikes_by_unit: Dict[int, np.ndarray] = {}
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        units = nwbfile.units.to_dataframe()

    # Expected Allen columns: id, spike_times
    if "spike_times" not in units.columns:
        raise RuntimeError("NWB units table does not contain 'spike_times'.")

    # Allen Ecephys NWB commonly stores unit ids in index (not always as a column).
    if "id" in units.columns:
        unit_ids: Iterable[int] = units["id"].astype(int).tolist()
    else:
        unit_ids = units.index.astype(int).tolist()

    for unit_id, spike_entry in zip(unit_ids, units["spike_times"].tolist()):
        spike_times = np.asarray(spike_entry, dtype=float)
        if spike_times.size:
            spikes_by_unit[int(unit_id)] = spike_times
    return spikes_by_unit


def spike_features(spike_times: np.ndarray) -> np.ndarray:
    """Convert one spike train into robust numeric features."""
    duration = float(spike_times[-1] - spike_times[0]) if spike_times.size >= 2 else 0.0
    count = int(spike_times.size)
    fr = count / duration if duration > 0 else 0.0

    if spike_times.size >= 3:
        isi = np.diff(spike_times)
        isi_mean = float(np.mean(isi))
        isi_std = float(np.std(isi))
        isi_cv = isi_std / isi_mean if isi_mean > 0 else 0.0
        isi_median = float(np.median(isi))
        isi_p10 = float(np.percentile(isi, 10))
        isi_p90 = float(np.percentile(isi, 90))
        burst_ratio = float(np.mean(isi < 0.01))
    else:
        isi_mean = isi_std = isi_cv = isi_median = isi_p10 = isi_p90 = burst_ratio = 0.0

    bins = np.arange(0.0, min(duration, 120.0) + 1.0, 1.0)
    if bins.size >= 2:
        shifted = spike_times - spike_times[0]
        hist, _ = np.histogram(shifted, bins=bins)
        fano = float(np.var(hist) / np.mean(hist)) if np.mean(hist) > 0 else 0.0
    else:
        fano = 0.0

    return np.array(
        [
            count,
            duration,
            fr,
            isi_mean,
            isi_std,
            isi_cv,
            isi_median,
            isi_p10,
            isi_p90,
            burst_ratio,
            fano,
        ],
        dtype=float,
    )


def build_dataset(labels_df: pd.DataFrame, spikes_by_unit: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: List[np.ndarray] = []
    y: List[str] = []
    groups: List[int] = []

    for rec in labels_df.itertuples(index=False):
        unit_id = int(rec.unit_id)
        region = str(rec.region)
        if unit_id not in spikes_by_unit:
            continue
        feats = spike_features(spikes_by_unit[unit_id])
        rows.append(feats)
        y.append(region)
        groups.append(int(rec.ecephys_channel_id))

    if not rows:
        raise RuntimeError("No units could be matched between CSV labels and NWB spikes.")

    return np.vstack(rows), np.array(y), np.array(groups)


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[Pipeline, dict]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])

    report = classification_report(y[test_idx], pred, output_dict=True, zero_division=0)
    cm_labels = sorted(set(y[test_idx]))
    cm = confusion_matrix(y[test_idx], pred, labels=cm_labels)

    metrics = {
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "classes_test": cm_labels,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    return model, metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.units_csv, args.channels_csv)
    labels = labels[labels["quality"] == "good"].copy()

    counts = labels["region"].value_counts()
    keep_regions = counts[counts >= args.min_units_per_region].index
    labels = labels[labels["region"].isin(keep_regions)].copy()

    spikes_by_unit = load_spike_times_from_nwb(args.nwb)
    X, y, groups = build_dataset(labels, spikes_by_unit)

    model, metrics = train_and_evaluate(
        X=X,
        y=y,
        groups=groups,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    feature_names = [
        "spike_count",
        "duration_s",
        "firing_rate_hz",
        "isi_mean_s",
        "isi_std_s",
        "isi_cv",
        "isi_median_s",
        "isi_p10_s",
        "isi_p90_s",
        "burst_ratio_isi_lt10ms",
        "fano_1s",
    ]
    importances = model.named_steps["clf"].feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(args.output_dir / "feature_importance.csv", index=False)

    print("Done. Metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
