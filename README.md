# Piper 数据转换

该目录将 Piper 多相机采集的 `trajectory.hdf5` 转换为 OpenPI 当前依赖的 LeRobot v2.1 格式。LeRobot 被固定到 OpenPI `pyproject.toml` 中相同的提交 `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`，不依赖采集工程的 Python 模块。

## 初始化环境

```bash
cd /home/ubuntu/gcj/projects/data_collect/data_processed
uv sync --group dev
```

## 转换玉米放盘数据

先用一条轨迹检查图像、状态和动作字段：

```bash
uv run piper-hdf5-to-lerobot --config configs/corn_in_plate.yaml --max-episodes 1
```

确认后，将 `configs/corn_in_plate.yaml` 的 `output.overwrite` 保持为 `false`，直接运行全量转换：

```bash
uv run piper-hdf5-to-lerobot --config configs/corn_in_plate.yaml --quiet
```

若需要从头重新生成，先人工确认 `output.root` 仅包含可删除的产物，再将 `output.overwrite` 改为 `true`。

默认配置输出到 `datasets/taiyigong333/piper_corn_in_plate`，其中包含 LeRobot 所需的 `data/`、`meta/` 以及 `conversion_manifest.json` 和 `DATASET_DESCRIPTION.md`。默认图像模式会将图像嵌入 Parquet；启用视频模式后才生成 `videos/`。训练 OpenPI 前，将 `HF_LEROBOT_HOME` 设为本目录下的 `datasets`，标准加载器便会在 `HF_LEROBOT_HOME/<repo_id>` 找到该数据集。原始 HDF5、输出数据集和虚拟环境都不会被提交到 Git。

转换会先写入同目录的唯一暂存目录，只有全部轨迹成功后才替换目标数据集；中途失败不会破坏已有的完整产物。

## 数据语义

- 每个 RGB 帧使用同一时刻之前最近的原始机器人反馈对齐；超过采集 YAML 的 `max_alignment_age_ms` 会终止转换，避免悄悄引入错配。
- `state` 为当前关节位置（或 TCP）及可选夹爪开度；`actions` 是下一图像帧的目标。默认关节动作为相对增量，夹爪保持绝对开度，符合 OpenPI 常用动作约定。
- 图像通过等比例缩小和居中 padding 处理为 224x224；不拉伸场景内容。将 `target_width` 和 `target_height` 同时设为 `null` 可保留原尺寸。
- LeRobot 原始字段为 `image`、`wrist_image`、`state`、`actions`。OpenPI 的 `LeRobotLiberoDataConfig` 会将前两个字段映射为其内部的 `observation/image` 与 `observation/wrist_image`。

详细交接信息见 [docs/2026-07-24_项目交接.md](docs/2026-07-24_项目交接.md)。
