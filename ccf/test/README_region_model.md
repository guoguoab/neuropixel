# 从 Neuropixels spike 预测脑区（unit-level）

这个示例脚本将 `session_*.nwb` 中每个 unit 的 spike time 提取出来，并结合 `units.csv + channels.csv` 的脑区注释，训练一个分类器去预测该 unit 所在脑区。

## 1) 安装依赖

```bash
pip install numpy pandas scikit-learn pynwb
```

## 2) 运行

```bash
python ccf/test/train_region_from_spikes.py \
  --nwb ccf/session_715093703/session_715093703.nwb \
  --units-csv ccf/test/units.csv \
  --channels-csv ccf/test/channels.csv \
  --min-units-per-region 30 \
  --output-dir ccf/test/outputs_region_model
```

## 3) 输出

- `metrics.json`: 分类报告（precision/recall/F1）与混淆矩阵。
- `feature_importance.csv`: 基于随机森林的重要特征排序。

## 4) 方法说明（简要）

- 标签来源：`channels.csv` 的 `ecephys_structure_acronym`。
- spike 特征：spike 数、firing rate、ISI 统计量、burst 比例、1 秒 bin 的 Fano factor。
- 切分方式：`GroupShuffleSplit`，按 `ecephys_channel_id` 分组，降低同 channel 信息泄露。
- 基线模型：`RandomForestClassifier`。

## 5) 可扩展方向

- 用多 session 联合训练（将多个 `session_*.nwb` 的同构数据拼接）。
- 将波形特征（如 width、PT ratio）并入多模态特征。
- 尝试时序模型（RNN/Transformer）直接输入 spike train 序列。
