import os
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# =========================
# 1. 配置本地缓存目录
# =========================
# 这里改成你自己的路径
output_dir = r"D:\allen_ecephys_cache"
os.makedirs(output_dir, exist_ok=True)

manifest_path = os.path.join(output_dir, "manifest.json")

# =========================
# 2. 连接 AllenSDK 数据仓库
# =========================
print("Initializing cache...")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# =========================
# 3. 获取所有 session 的元信息
# =========================
print("Loading session table...")
sessions = cache.get_session_table()

print(f"Total number of sessions: {len(sessions)}")
print("\nSession table columns:")
print(sessions.columns.tolist())

# =========================
# 4. 按 notebook 里的条件筛选一个 session
#    条件：
#    - sex == 'M'
#    - full_genotype 包含 'Sst'
#    - session_type == 'brain_observatory_1.1'
#    - ecephys_structure_acronyms 包含 'VISl'
# =========================
filtered_sessions = sessions[
    (sessions.sex == "M") &
    (sessions.full_genotype.str.contains("Sst", na=False)) &
    (sessions.session_type == "brain_observatory_1.1") &
    (sessions.ecephys_structure_acronyms.apply(lambda x: "VISl" in x if isinstance(x, (list, tuple, np.ndarray)) else False))
]

print(f"\nFiltered sessions: {len(filtered_sessions)}")

if len(filtered_sessions) == 0:
    raise RuntimeError("No session matched the filter conditions.")

# 取第一条
session_id = filtered_sessions.index.values[0]
print(f"Selected session_id: {session_id}")

# =========================
# 5. 下载并读取该 session
#    这里和 notebook 一样，放宽 unit 质量过滤条件
# =========================
print("\nDownloading / loading session data...")
session = cache.get_session_data(
    session_id,
    isi_violations_maximum=np.inf,
    amplitude_cutoff_maximum=np.inf,
    presence_ratio_minimum=-np.inf
)

print("Session loaded successfully!")

# =========================
# 6. 查看 session 基本信息
# =========================
print("\nAvailable probes:")
print(session.probes.head())

print("\nAvailable channels:")
print(session.channels.head())

print("\nAvailable units:")
print(session.units.head())

print(f"\nNumber of probes: {len(session.probes)}")
print(f"Number of channels: {len(session.channels)}")
print(f"Number of units: {len(session.units)}")

# =========================
# 7. 可选：读取第一根 probe 的 LFP
# =========================
if len(session.probes) > 0:
    probe_id = session.probes.index.values[0]
    print(f"\nLoading LFP for probe_id: {probe_id}")
    lfp = session.get_lfp(probe_id)

    print("\nLFP loaded successfully!")
    print(lfp)
    print("\nLFP dims:", lfp.dims)
    print("LFP shape:", lfp.shape)
else:
    print("\nNo probes found in this session.")

print("\nDone.")