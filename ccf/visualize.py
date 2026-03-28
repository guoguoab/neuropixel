import os
import numpy as np
import pyvista as pv
from pyvista import ImageData
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi

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

surface = grid.contour([0.5])
# -------------------------------
# 3D 可视化
# -------------------------------

plotter = pv.Plotter()
plotter.add_mesh(surface, color="lightgray")
plotter.show_grid()
plotter.show()