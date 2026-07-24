# OpenPI 训练说明

本文说明如何用本项目已转换的本地 LeRobot 数据集训练 OpenPI，并适用于项目固定的 OpenPI 子模块版本 `15a9616a00943ada6c20a0f158e3adb39df2ccac`。数据转换、媒体模式和数据集验证请参阅 [使用说明.md](使用说明.md)。

## 数据集契约

训练数据集为 `taiyigong333/piper_corn_in_plate`，本地路径为 `datasets/taiyigong333/piper_corn_in_plate`。当前版本包含 100 条轨迹、20,310 帧和一个任务文本 `pick up the corn and put it on the plate`，使用图片 Parquet 模式而非 MP4：

| LeRobot 字段 | 形状/类型 | OpenPI 中的用途 |
| --- | --- | --- |
| `image` | 224 x 224 x 3 RGB | 第三人称相机 |
| `wrist_image` | 224 x 224 x 3 RGB | 腕部相机 |
| `state` | `[J1, J2, J3, J4, J5, J6, gripper]`，7 维 `float32` | 当前 Piper 状态 |
| `actions` | 同上，7 维 `float32` | 下一时刻行为监督目标 |
| `task` | 单条自然语言任务 | 训练提示词 |

文件中的关节目标和夹爪开度均是绝对值。`LeRobotLiberoDataConfig(extra_delta_transform=True)` 在训练输入阶段把前六维转换为相对当前状态的关节差值，夹爪保持绝对值；策略输出阶段会自动还原前六维绝对关节目标。此逐元素变换只适用于当前六关节表示，不适用于四元数或一般 TCP/SE(3) 姿态。依据见 [2026-07-24_OpenPI绝对动作训练变换核查.md](2026-07-24_OpenPI绝对动作训练变换核查.md)。

保持 `configs/corn_in_plate.yaml` 的 `output.use_videos: false`。当前机器可以直接读取图片 Parquet；MP4 的生成已验证，但视频解码依赖损坏的系统动态 FFmpeg，不能用于当前机器的训练。

## 环境与数据加载

OpenPI 使用独立的 `uv` 环境，不复用本项目根目录 `.venv`。OpenPI 的当前要求是 Ubuntu 和 NVIDIA GPU；其文档估计 LoRA 微调至少需要 22.5 GB 显存，完整微调至少需要 70 GB 显存。

```bash
cd /home/ubuntu/gcj/projects/data_collect/data_processed/openpi
export UV_CACHE_DIR=/tmp/openpi-uv-cache
export HF_LEROBOT_HOME=/home/ubuntu/gcj/projects/data_collect/data_processed/datasets
export OPENPI_DATA_HOME=/home/ubuntu/gcj/projects/data_collect/data_processed/openpi-cache
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
nvidia-smi
uv run python -c "import jax; print(jax.devices())"
```

`HF_LEROBOT_HOME` 必须指向 `datasets` 的上一层，OpenPI 会据此从 `datasets/taiyigong333/piper_corn_in_plate` 加载本地数据集。`OPENPI_DATA_HOME` 用于缓存自动下载的基础模型权重。

在 `openpi` 目录确认数据可读取：

```bash
uv run python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; dataset = LeRobotDataset('taiyigong333/piper_corn_in_plate'); print(len(dataset), dataset[0]['state'].shape, dataset[0]['actions'].shape)"
```

预期样本数为 `20310`，两个向量的最后一维均为 `7`（不同 PyTorch/LeRobot 版本可能显示为 `torch.Size([7])` 或 `(7,)`）。

## 注册 Piper 配置

OpenPI 的训练和统计量命令都通过已注册的配置名工作。因此在 `openpi/src/openpi/training/config.py` 的 `_CONFIGS = [` 列表末尾、闭合 `]` 之前添加所需配置。现有的 `LeRobotLiberoDataConfig` 与 `LiberoInputs`/`LiberoOutputs` 可直接复用：它们重映射本项目的两路图像和状态字段，并在输出时只保留前 7 维动作。

### π₀-FAST LoRA

以下是推荐的低显存微调配置。它使用 π₀-FAST 基础权重、7 维动作和 10 步动作块；单张 24 GB 级别显卡可从 `batch_size=8` 开始。显存不足时减为 `4` 或 `2`，多 GPU 时批大小必须可被 GPU 数量整除。

```python
    TrainConfig(
        name="pi0_fast_piper_corn_lora",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="taiyigong333/piper_corn_in_plate",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=8,
        num_train_steps=30_000,
        save_interval=1_000,
        wandb_enabled=False,
    ),
```

显存充足并选择完整微调时，可参考同一文件的 `pi0_fast_libero`：去掉 `paligemma_variant`、`freeze_filter` 和 `ema_decay=None`。完整微调需要至少 70 GB 显存。

### π₀.₅ 完整微调

当前 OpenPI 只支持 π₀.₅ 的 flow-matching head 训练与推理，不存在可直接替换的 π₀.₅-FAST 配置。π₀.₅ 的基础权重为 `gs://openpi-assets/checkpoints/pi05_base/params`，推荐从基础模型训练本 Piper 数据，而不要直接采用 DROID 专家权重或其归一化统计：DROID 示例的动作语义是关节速度，与本项目的绝对关节位置目标不一致。

对于 Piper，应保留 π₀.₅ 内部的 `action_dim=32`，而不是改成 7。OpenPI 的 `PadStatesAndActions` 会把 7 维 `state`/`actions` 填充到基础模型所需的 32 维，`LiberoOutputs` 在推理时再截取前 7 维。`pi05_libero` 也使用 32 维内部动作空间。显式设置 `discrete_state_input=False` 与现有 LIBERO 路径一致，使 Piper 连续关节状态走连续状态输入；不要使用 π₀.₅ 默认的离散状态输入设置。

```python
    TrainConfig(
        name="pi05_piper_corn",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="taiyigong333/piper_corn_in_plate",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        wandb_enabled=False,
    ),
```

该配置是完整微调，`batch_size=8` 只是在显存允许时用于小数据集的起始值，不能把它理解为 24 GB 显卡可运行的保证。请先完成一小段训练验证显存，再确定批大小和训练步数。π₀.₅ 的公开示例没有针对本 Piper 数据已验证的 LoRA 配方，因此不将未经验证的 π₀.₅ LoRA 配置列为推荐路径。

### π₀.₅ 训练要点

- 仅有 100 条轨迹、单一任务文本时，模型很容易过拟合，且无法验证语言泛化。应在训练前按完整轨迹划分保留集，并用未参与训练的初始状态、光照和物体位置选择检查点。
- 当前 `meta/info.json` 的分割是 `train: 0:100`，没有验证集。不要以训练损失或训练轨迹成功率作为实体机器人部署依据。
- `action_horizon=10` 对应数据集 30 Hz 下约 0.33 秒动作块。实体 Piper 应采用滚动重规划、速度限制和关节限位，不能一次执行完整动作块后失去观测反馈。
- 必须每次使用当前数据集计算本地归一化统计量。只有状态和动作的机器人语义、量纲和顺序完全一致时，才能复用其他模型的统计量；Piper 与 DROID 不满足这一条件。
- `extra_delta_transform=True` 是 Piper 绝对关节数据所必需；现有 `pi05_libero` 设置为 `false` 是因为 LIBERO 动作原本就是相对量，不能直接照抄。
- 部署前确认关节序、弧度单位、夹爪单位和两个相机位置与训练数据一致。π₀.₅ 的预训练并不能修正这些运行时语义错误。

## 生成统计量与训练

保持前述环境变量，在 `openpi` 目录运行。以下命令以 `pi0_fast_piper_corn_lora` 为例；训练 π₀.₅ 时将配置名整体替换为 `pi05_piper_corn`。

先检查配置是否已经注册：

```bash
uv run python -c "from openpi.training import config; train_config = config.get_config('pi0_fast_piper_corn_lora'); print(train_config.data.repo_id, train_config.model.action_dim, train_config.model.action_horizon)"
```

计算当前 Piper 数据的归一化统计量。它会写入 `openpi/assets/<config_name>/taiyigong333/piper_corn_in_plate/`；每次更换数据、动作语义或配置后都要重新生成。

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_fast_piper_corn_lora
```

首次训练会自动下载相应的基础权重到 `OPENPI_DATA_HOME`，因此需要可访问 `gs://openpi-assets` 的网络。

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run scripts/train.py pi0_fast_piper_corn_lora --exp-name=corn_v1 --overwrite
```

检查点写入 `openpi/checkpoints/<config_name>/<exp_name>/`。中断恢复时移除 `--overwrite` 并使用相同实验名：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run scripts/train.py pi0_fast_piper_corn_lora --exp-name=corn_v1 --resume
```

## 策略服务与 Piper 接入

将 `<checkpoint_step>` 替换为检查点目录中的实际步数：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_piper_corn_lora \
    --policy.dir=checkpoints/pi0_fast_piper_corn_lora/corn_v1/<checkpoint_step>
```

Piper 控制端应向策略提供 `observation/image`、`observation/wrist_image`（均为 HWC、`uint8` RGB 图像）、`observation/state`（`[J1, J2, J3, J4, J5, J6, gripper]` 的 `float32` 向量）和训练时的 `prompt`。策略返回同一顺序的 7 维绝对关节目标及绝对夹爪开度。

首次连接实体机器人时，必须在限速、限位和急停有效的条件下逐步验证每个关节和夹爪动作。不要将未经离线和低速验证的策略输出直接下发到机器人。
