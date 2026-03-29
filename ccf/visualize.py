import argparse
import csv
import os
import random
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pyvista as pv
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from pyvista import ImageData

# 下载路径
save_dir = "ignore"
os.makedirs(save_dir, exist_ok=True)

resolution = 25  # 25um
rsp = ReferenceSpaceApi()

annotation_path = os.path.join(save_dir, "annotation.nrrd")
template_path = os.path.join(save_dir, "template.nrrd")

if not os.path.exists(annotation_path):
    print("Downloading annotation volume...")
    rsp.download_annotation_volume("annotation/ccf_2017", resolution, annotation_path)

if not os.path.exists(template_path):
    print("Downloading template volume...")
    rsp.download_template_volume(resolution, template_path)

print("Download finished")

# -------------------------------
# 读取 annotation
# -------------------------------
import nrrd

annotation, header = nrrd.read(annotation_path)
print("Volume shape:", annotation.shape)

# 转换为 mesh
volume = pv.wrap(annotation.astype(np.uint16))

# 取脑组织区域（非0）
mask = annotation > 0

grid = ImageData()
grid.dimensions = annotation.shape
grid.point_data["values"] = mask.flatten(order="F")

surface = (
    grid.contour([0.5])
    .extract_surface(algorithm="dataset_surface")
    .triangulate()
    .smooth(n_iter=120, relaxation_factor=0.01)
)


def load_units_from_region(channels_csv, units_csv, structure_acronym, voxel_resolution_um):
    """
    根据 channels.csv 的脑区缩写筛选 channel，再在 unit.csv 中筛选对应神经元。
    返回:
    - points: N x 3 的坐标数组（单位: 25um voxel）
    - matched_units: 匹配到的 unit 行列表
    - region_channel_ids: 该脑区对应的 channel id 集合
    """
    region_channel_ids = set()
    channel_points = {}
    skipped_channels = 0

    def _safe_float(value):
        if value is None:
            return None
        value = value.strip()
        if value == "":
            return None
        return float(value)

    with open(channels_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ecephys_structure_acronym"] != structure_acronym:
                continue

            ap = _safe_float(row.get("anterior_posterior_ccf_coordinate"))
            dv = _safe_float(row.get("dorsal_ventral_ccf_coordinate"))
            lr = _safe_float(row.get("left_right_ccf_coordinate"))
            if ap is None or dv is None or lr is None:
                skipped_channels += 1
                continue

            channel_id = int(row["id"])
            region_channel_ids.add(channel_id)
            # Allen CCF 坐标是微米单位，25um 分辨率下需要缩放到 voxel 坐标
            channel_points[channel_id] = (
                ap / voxel_resolution_um,
                dv / voxel_resolution_um,
                lr / voxel_resolution_um,
            )

    matched_units = []
    points = []

    with open(units_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            channel_id_str = (row.get("ecephys_channel_id") or "").strip()
            if not channel_id_str:
                continue
            channel_id = int(channel_id_str)
            if channel_id not in region_channel_ids:
                continue

            matched_units.append(row)
            points.append(channel_points[channel_id])

    if points:
        points = np.asarray(points, dtype=float)
    else:
        points = np.empty((0, 3), dtype=float)

    if skipped_channels > 0:
        print(
            f"Skipped {skipped_channels} channels in region {structure_acronym} "
            "due to missing CCF coordinates."
        )

    return points, matched_units, region_channel_ids


def _id_key(raw_id: str) -> Optional[str]:
    if raw_id is None:
        return None
    value = str(raw_id).strip()
    if not value:
        return None
    try:
        return f"{float(value):.10f}"
    except ValueError:
        return None


def _split_id_list(raw_ids: str) -> Set[str]:
    out = set()
    if not raw_ids:
        return out
    for token in raw_ids.split(","):
        key = _id_key(token)
        if key is not None:
            out.add(key)
    return out


def _pick_merge_region_row(merge_region_data_dir: str, seed: Optional[int] = None):
    table_files = [
        name
        for name in os.listdir(merge_region_data_dir)
        if name.endswith("_merged_regions_table_cell_id.txt")
    ]
    if not table_files:
        raise FileNotFoundError(
            f"No *_merged_regions_table_cell_id.txt file found in {merge_region_data_dir}"
        )

    rng = random.Random(seed)
    selected_file = rng.choice(table_files)
    selected_path = os.path.join(merge_region_data_dir, selected_file)

    with open(selected_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        raise ValueError(f"No row found in {selected_path}")

    row = rng.choice(rows)
    row_index = rows.index(row)
    return selected_file, row, row_index, len(rows)


def _extract_slide_from_filename(filename: str) -> Optional[str]:
    m = re.search(r"(C57BL6J-\d+\.\d+)", filename)
    return m.group(1) if m else None


def _resolve_spatial_data_path(spital_data_dir: str, slide: str) -> str:
    direct = os.path.join(spital_data_dir, f"{slide}.txt")
    if os.path.exists(direct):
        return direct
    nested = os.path.join(spital_data_dir, "data", f"{slide}.txt")
    if os.path.exists(nested):
        return nested
    raise FileNotFoundError(f"Spatial data file not found for {slide} in {spital_data_dir}")


def _load_glut_gaba_points_from_merge_region(
    merge_region_data_dir: str,
    spital_data_dir: str,
    voxel_resolution_um: int,
    seed: Optional[int] = None,
):
    selected_file, row, row_index, total_rows = _pick_merge_region_row(
        merge_region_data_dir, seed=seed
    )

    slide = _extract_slide_from_filename(selected_file)
    if slide is None:
        raise ValueError(f"Cannot parse slide id from filename: {selected_file}")

    spatial_file = _resolve_spatial_data_path(spital_data_dir, slide)

    glut_ids = _split_id_list(row.get("Glut_Neruon_cell_ids", ""))
    gaba_ids = _split_id_list(row.get("GABA_Neruon_cell_ids", ""))

    glut_points: List[Tuple[float, float, float]] = []
    gaba_points: List[Tuple[float, float, float]] = []
    matched_cell_ids = 0
    missing_coord_rows = 0

    with open(spatial_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            cell_key = _id_key(r.get("cell_label", ""))
            if cell_key is None:
                continue

            if cell_key not in glut_ids and cell_key not in gaba_ids:
                continue

            ccf_x = r.get("CCF_x")
            ccf_y = r.get("CCF_y")
            ccf_z = r.get("CCF_z")
            if not ccf_x or not ccf_y or not ccf_z:
                missing_coord_rows += 1
                continue

            # spital 数据里的 CCF 坐标通常是毫米(mm)，转成 25um voxel: mm*1000/25
            x = float(ccf_x) * 1000.0 / voxel_resolution_um
            y = float(ccf_y) * 1000.0 / voxel_resolution_um
            z = float(ccf_z) * 1000.0 / voxel_resolution_um
            matched_cell_ids += 1

            if cell_key in glut_ids:
                glut_points.append((x, y, z))
            if cell_key in gaba_ids:
                gaba_points.append((x, y, z))

    return {
        "selected_file": selected_file,
        "selected_row_index": row_index,
        "total_rows_in_file": total_rows,
        "merge_regions": row.get("merge_regions", ""),
        "slide": slide,
        "spatial_file": spatial_file,
        "glut_points": np.asarray(glut_points, dtype=float)
        if glut_points
        else np.empty((0, 3), dtype=float),
        "gaba_points": np.asarray(gaba_points, dtype=float)
        if gaba_points
        else np.empty((0, 3), dtype=float),
        "glut_ids_total": len(glut_ids),
        "gaba_ids_total": len(gaba_ids),
        "matched_cell_ids": matched_cell_ids,
        "missing_coord_rows": missing_coord_rows,
    }


# -------------------------------
# 3D 可视化
# -------------------------------

parser = argparse.ArgumentParser(description="在 3D CCF 小鼠脑内可视化神经元点")
parser.add_argument("--channels-csv", default="test/channels.csv", help="channels.csv 路径")
parser.add_argument("--units-csv", default="test/units.csv", help="units.csv 路径")
parser.add_argument("--region", default="APN", help="目标脑区缩写（如 APN）")

parser.add_argument(
    "--use-merge-region-data",
    action="store_true",
    help="使用 merge_region_data + spital_data 匹配 Glut/GABA 神经元并可视化",
)
parser.add_argument(
    "--no-merge-region-data",
    dest="use_merge_region_data",
    action="store_false",
    help="关闭 merge_region_data 模式，改用 channels.csv + units.csv 模式",
)
parser.add_argument(
    "--merge-region-data-dir",
    default="test/merge_region_data",
    help="merge_region_data 文件夹路径",
)
parser.add_argument(
    "--spital-data-dir",
    default="test/spital_data",
    help="spital_data 文件夹路径",
)
parser.add_argument(
    "--random-seed",
    type=int,
    default=None,
    help="随机选取 merge_region_data 文件时可选随机种子",
)
parser.set_defaults(use_merge_region_data=True)
args = parser.parse_args()

plotter = pv.Plotter()
plotter.add_mesh(surface, color="purple", opacity=0.5, smooth_shading=True)

if args.use_merge_region_data:
    print("[DEBUG] merge_region_data mode enabled")
    print(f"[DEBUG] merge_region_data_dir = {args.merge_region_data_dir}")
    print(f"[DEBUG] spital_data_dir = {args.spital_data_dir}")
    print(f"[DEBUG] random_seed = {args.random_seed}")
    merge_result = _load_glut_gaba_points_from_merge_region(
        merge_region_data_dir=args.merge_region_data_dir,
        spital_data_dir=args.spital_data_dir,
        voxel_resolution_um=resolution,
        seed=args.random_seed,
    )

    print(f"Selected merge file: {merge_result['selected_file']}")
    print(
        f"Selected row index: {merge_result['selected_row_index']} / "
        f"{merge_result['total_rows_in_file'] - 1}"
    )
    print(f"Selected merge_regions: {merge_result['merge_regions']}")
    print(f"Slide: {merge_result['slide']}")
    print(f"Spatial file: {merge_result['spatial_file']}")
    print(
        f"Glut ids total/matched: {merge_result['glut_ids_total']}/{len(merge_result['glut_points'])}"
    )
    print(
        f"GABA ids total/matched: {merge_result['gaba_ids_total']}/{len(merge_result['gaba_points'])}"
    )
    print(f"Matched cell ids in spatial file: {merge_result['matched_cell_ids']}")
    print(f"Skipped rows due to missing CCF coords: {merge_result['missing_coord_rows']}")

    if len(merge_result["glut_points"]) > 0:
        plotter.add_points(
            merge_result["glut_points"],
            color="red",
            point_size=8,
            render_points_as_spheres=True,
        )

    if len(merge_result["gaba_points"]) > 0:
        plotter.add_points(
            merge_result["gaba_points"],
            color="blue",
            point_size=8,
            render_points_as_spheres=True,
        )
    if len(merge_result["glut_points"]) == 0 and len(merge_result["gaba_points"]) == 0:
        print(
            "[DEBUG] No Glut/GABA points matched. "
            "请检查 merge_region_data 与 spital_data 是否来自同一批次样本。"
        )
else:
    unit_points, matched_units, region_channel_ids = load_units_from_region(
        channels_csv=args.channels_csv,
        units_csv=args.units_csv,
        structure_acronym=args.region,
        voxel_resolution_um=resolution,
    )

    print(f"Region: {args.region}")
    print(f"Matched channels: {len(region_channel_ids)}")
    print(f"Matched units: {len(matched_units)}")

    if len(unit_points) > 0:
        plotter.add_points(
            unit_points,
            color="blue",
            point_size=8,
            render_points_as_spheres=True,
        )
    else:
        print(f"No units found for region {args.region}")

plotter.show_grid()
plotter.show()
