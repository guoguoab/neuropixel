import pandas as pd

units = pd.read_csv("units.csv")
channels = pd.read_csv("channels.csv")

merged = units.merge(
    channels,
    left_on="ecephys_channel_id",
    right_on="id",
    how="left",
    suffixes=("_unit", "_channel")
)

print("merged columns:")
print(merged.columns.tolist())

# 质量筛选
good_units = merged[merged["quality"] == "good"].copy()

print("\nNumber of good units:", len(good_units))

# 每个脑区 unit 数
region_counts = good_units["ecephys_structure_acronym"].value_counts()
print("\nUnits per region:")
print(region_counts.head(30))

# 每个脑区平均 firing rate
region_fr = (
    good_units
    .groupby("ecephys_structure_acronym")["firing_rate"]
    .mean()
    .sort_values(ascending=False)
)

print("\nMean firing rate by region:")
print(region_fr.head(30))

# 深度分箱
good_units["depth_bin"] = pd.cut(good_units["probe_vertical_position"], bins=10)

depth_fr = good_units.groupby("depth_bin")["firing_rate"].mean()
print("\nMean firing rate by depth bin:")
print(depth_fr)

# 粗略波形分类
median_duration = good_units["duration"].median()
good_units["putative_class"] = "broad"
good_units.loc[good_units["duration"] < median_duration, "putative_class"] = "narrow"

type_region = (
    good_units
    .groupby(["ecephys_structure_acronym", "putative_class"])
    .size()
    .unstack(fill_value=0)
)

print("\nPutative class by region:")
print(type_region.head(30))