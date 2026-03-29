import argparse
import csv
import os

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

    rsp.download_annotation_volume(
        "annotation/ccf_2017",
        resolution,
        annotation_path
    )

if not os.path.exists(template_path):

    print("Downloading template volume...")

    rsp.download_template_volume(
        resolution,
        template_path
    )

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

    with open(channels_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ecephys_structure_acronym"] != structure_acronym:
                continue

            channel_id = int(row["id"])
            region_channel_ids.add(channel_id)
            # Allen CCF 坐标是微米单位，25um 分辨率下需要缩放到 voxel 坐标
            channel_points[channel_id] = (
                float(row["anterior_posterior_ccf_coordinate"]) / voxel_resolution_um,
                float(row["dorsal_ventral_ccf_coordinate"]) / voxel_resolution_um,
                float(row["left_right_ccf_coordinate"]) / voxel_resolution_um,
            )

    matched_units = []
    points = []

    with open(units_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            channel_id = int(row["ecephys_channel_id"])
            if channel_id not in region_channel_ids:
                continue

            matched_units.append(row)
            points.append(channel_points[channel_id])

    if points:
        points = np.asarray(points, dtype=float)
    else:
        points = np.empty((0, 3), dtype=float)

    return points, matched_units, region_channel_ids
# -------------------------------
# 3D 可视化
# -------------------------------

parser = argparse.ArgumentParser(
    description="在 3D CCF 小鼠脑内可视化指定脑区对应的神经元点（蓝色）"
)
parser.add_argument(
    "--channels-csv",
    default="test/channels.csv",
    help="channels.csv 路径",
)
parser.add_argument(
    "--units-csv",
    default="test/units.csv",
    help="units.csv 路径",
)
parser.add_argument(
    "--region",
    default="APN",
    help="目标脑区缩写（如 APN）",
)
args = parser.parse_args()

unit_points, matched_units, region_channel_ids = load_units_from_region(
    channels_csv=args.channels_csv,
    units_csv=args.units_csv,
    structure_acronym=args.region,
    voxel_resolution_um=resolution,
)

print(f"Region: {args.region}")
print(f"Matched channels: {len(region_channel_ids)}")
print(f"Matched units: {len(matched_units)}")

plotter = pv.Plotter()
plotter.add_mesh(surface, color="purple", opacity=0.5, smooth_shading=True)

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
