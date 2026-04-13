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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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

    channel_region_count = channels_df["ecephys_structure_acronym"].dropna().nunique()
    print(f"[labels] channels.csv 中共有 {channel_region_count} 个脑区(acronym)")

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
    merged_region_count = merged["region"].nunique()
    print(
        f"[labels] units.csv 与 channels.csv 合并后，实际有标签的脑区数: {merged_region_count}"
    )
    return merged


def load_spike_times_from_nwb(nwb_path: Path) -> Dict[int, np.ndarray]:
    from pynwb import NWBHDF5IO

    spikes_by_unit: Dict[int, np.ndarray] = {}
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        units = nwbfile.units.to_dataframe()
    print(f"[nwb] NWB units 表中共有 {len(units)} 个unit")

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


def build_model_candidates(random_state: int) -> Dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=600,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "svc_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=5.0,
                        gamma="scale",
                        kernel="rbf",
                        class_weight="balanced",
                        probability=False,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=2.0,
                        class_weight="balanced",
                        max_iter=3000,
                        multi_class="multinomial",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=11, weights="distance")),
            ]
        ),
    }


def train_and_select_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[Pipeline, str, dict, np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    candidates = build_model_candidates(random_state=random_state)
    model_results = []
    best_model = None
    best_model_name = ""
    best_pred = None
    best_score = -1.0
    best_report = {}

    for model_name, model in candidates.items():
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        report = classification_report(y[test_idx], pred, output_dict=True, zero_division=0)
        weighted_f1 = float(report["weighted avg"]["f1-score"])
        macro_f1 = float(report["macro avg"]["f1-score"])
        accuracy = float(report["accuracy"])

        model_results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "weighted_f1": weighted_f1,
                "macro_f1": macro_f1,
            }
        )

        if weighted_f1 > best_score:
            best_score = weighted_f1
            best_model = model
            best_model_name = model_name
            best_pred = pred
            best_report = report

    if best_model is None or best_pred is None:
        raise RuntimeError("No valid model could be trained.")

    model_results = sorted(model_results, key=lambda x: x["weighted_f1"], reverse=True)
    cm_labels = sorted(set(y[test_idx]))
    cm = confusion_matrix(y[test_idx], best_pred, labels=cm_labels)

    metrics = {
        "selected_model": best_model_name,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "candidate_models": model_results,
        "classes_test": cm_labels,
        "classification_report": best_report,
        "confusion_matrix": cm.tolist(),
    }
    return best_model, best_model_name, metrics, y[test_idx], best_pred


def plot_model_comparison(model_results: List[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    names = [r["model"] for r in model_results]
    weighted_f1 = [r["weighted_f1"] for r in model_results]
    accuracy = [r["accuracy"] for r in model_results]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, weighted_f1, width, label="weighted F1")
    ax.bar(x + width / 2, accuracy, width, label="accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on held-out test set")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], output_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", aspect="auto", cmap="Blues")
    ax.set_title("Normalized confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.units_csv, args.channels_csv)
    labels = labels[labels["quality"] == "good"].copy()

    counts = labels["region"].value_counts()
    keep_regions = counts[counts >= args.min_units_per_region].index
    labels = labels[labels["region"].isin(keep_regions)].copy()

    spikes_by_unit = load_spike_times_from_nwb(args.nwb)
    matched_labels = labels[labels["unit_id"].isin(spikes_by_unit.keys())]
    matched_region_count = matched_labels["region"].nunique()
    print(f"[nwb] 与NWB成功匹配的脑区数: {matched_region_count}")

    X, y, groups = build_dataset(labels, spikes_by_unit)

    model, model_name, metrics, y_test, pred_test = train_and_select_model(
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
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.mean(np.abs(clf.coef_), axis=0)
    else:
        importances = np.zeros(len(feature_names), dtype=float)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(args.output_dir / "feature_importance.csv", index=False)

    try:
        plot_model_comparison(metrics["candidate_models"], args.output_dir)
        plot_confusion_matrix(y_test, pred_test, metrics["classes_test"], args.output_dir)
    except ModuleNotFoundError as exc:
        print(f"[warn] plotting skipped because dependency is missing: {exc}")

    print("Done. Metrics:")
    print(f"Selected best model: {model_name}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
