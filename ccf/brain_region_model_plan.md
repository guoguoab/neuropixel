# Unit 脑区反推方案（基于电信号时序）

## 1) 能否做到？
可以做。你现有的数据链路已经具备关键条件：

- `units.csv`：unit 与 `ecephys_channel_id` 的映射。
- `channels.csv`：channel 与脑区 `ecephys_structure_acronym` 的映射。
- 原始 `nrrd`：每个 channel 的电信号时序。

把三者 join 后即可得到监督学习样本：

`(unit时序特征) -> (脑区标签)`

## 2) 当前仓库中可直接运行的基线管线
新增脚本 `ccf/unit_region_pipeline.py`，完成：

1. 读取 nrrd 原始信号。
2. 读取 `units.csv` 与 `channels.csv`。
3. 按 `unit -> channel -> region` 关联。
4. 从每个 unit 对应通道时序提取统计 + 频域特征。
5. 训练轻量基线模型（Nearest Centroid）。
6. 输出：
   - `unit_features_with_region.csv`
   - `region_inference_metrics.json`

## 3) 建议的正式 AI 模型（比基线更强）

### 模型结构（推荐优先级）
1. **1D CNN + Transformer Encoder（首选）**
   - 输入：`[T]` 或 `[C_local, T]`（可加入邻近通道）
   - 输出：脑区分类概率
   - 优点：对局部波形形态 + 长程时序关系都敏感。

2. **TCN（Temporal Convolution Network）**
   - 对长时序稳定，训练速度快。

3. **XGBoost / LightGBM（工程强基线）**
   - 输入手工特征（峰谷、ISI统计、谱能量分段）
   - 在数据量不大时常常很强。

### 标签与任务定义建议
- 先做“层级分类”：
  - 级别1：大区（CTX/TH/HPF/STR/...）
  - 级别2：细分 acronym
- 这样可以缓解细粒度类别不平衡。

### 训练与评估建议
- 划分方式：按 **session / animal 分组切分**，避免泄漏。
- 指标：
  - Top-1 accuracy
  - Top-3 accuracy
  - Macro-F1（处理类不平衡）
- 分析：输出 confusion matrix，观察易混脑区。

## 4) 关键工程注意点
- `channels.csv` 中用于映射 nrrd 通道的列，默认是 `local_index`；若你的 nrrd 顺序不同，需改 `--channel-index-col` 或预先构造映射表。
- 若一个 unit 需要融合多个邻近通道，建议在脚本中扩展为 `k-neighbor channels` 拼接输入。
- 在真实数据上，建议先统一重采样长度、去工频噪声、带通滤波后再训练深度模型。

## 5) 快速运行示例

```bash
python ccf/unit_region_pipeline.py \
  --nrrd-path /your/local/real_raw_signal.nrrd \
  --units-csv ccf/test/units.csv \
  --channels-csv ccf/test/channels.csv \
  --out-dir ccf/output \
  --channel-axis 0 \
  --channel-index-col local_index \
  --quality good \
  --min-units-per-region 20
```

如果你愿意，我下一步可以直接把这个基线升级成 **PyTorch 的 1D CNN/Transformer 训练脚本**（含 early stopping、类别权重、混淆矩阵与可视化）。
