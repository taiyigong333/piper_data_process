"""Piper 原始 HDF5 到 LeRobot v2.1 的配置驱动转换实现。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
from typing import Any, Literal
from uuid import uuid4

import h5py
import numpy as np
from PIL import Image
import yaml


StateSource = Literal["joint_positions", "tcp_pose"]
ActionRepresentation = Literal["absolute", "relative"]
AlignmentStrategy = Literal["previous", "nearest"]


class ConversionError(RuntimeError):
    """输入轨迹或转换配置不满足数据契约时抛出。"""


@dataclass(frozen=True)
class CameraSpec:
    """一个原始相机到 LeRobot 特征名的映射。"""

    source_name: str
    feature_name: str


@dataclass(frozen=True)
class SourceConfig:
    hdf5_root: Path
    trajectory_glob: str
    collection_config: Path


@dataclass(frozen=True)
class OutputConfig:
    repo_id: str
    root: Path
    robot_type: str
    fps: int | None
    use_videos: bool
    video_codec: str
    video_preset: str
    video_crf: int
    video_keyframe_interval: int
    image_writer_processes: int
    image_writer_threads: int
    overwrite: bool


@dataclass(frozen=True)
class RepresentationConfig:
    state_source: StateSource
    include_gripper: bool
    action_representation: ActionRepresentation
    action_horizon_frames: int


@dataclass(frozen=True)
class AlignmentConfig:
    strategy: AlignmentStrategy
    max_state_age_ms: float | None


@dataclass(frozen=True)
class ImageConfig:
    target_width: int | None
    target_height: int | None
    pad_rgb: tuple[int, int, int]
    validate_source_size: bool
    cameras: tuple[CameraSpec, ...]


@dataclass(frozen=True)
class DescriptionConfig:
    task: str
    intended_model: str
    notes: str


@dataclass(frozen=True)
class ConversionConfig:
    """转换入口配置，所有路径在加载时规范为绝对路径。"""

    source: SourceConfig
    output: OutputConfig
    representation: RepresentationConfig
    alignment: AlignmentConfig
    images: ImageConfig
    description: DescriptionConfig
    source_path: Path


@dataclass(frozen=True)
class CollectionMetadata:
    """从采集 YAML 提取的、会影响数据解释的字段。"""

    task: str
    camera_fps: float
    max_alignment_age_ms: float
    pose_representation: str
    joint_count: int
    enabled_cameras: dict[str, tuple[int, int]]
    enabled_modalities: dict[str, bool]
    robot_name: str


@dataclass(frozen=True)
class EpisodeReport:
    source_path: str
    frames: int
    max_state_age_ms: float


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversionError(f"{name} 必须是 YAML 对象。")
    return value


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _required(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConversionError(f"缺少必填配置：{path}.{key}")
    return mapping[key]


def _positive_int(value: Any, path: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ConversionError(f"{path} 必须是正整数。") from error
    if number <= 0:
        raise ConversionError(f"{path} 必须是正整数。")
    return number


def load_conversion_config(path: str | Path) -> ConversionConfig:
    """加载转换 YAML；此处只解析配置，不接触原始数据。"""

    source_path = Path(path).expanduser().resolve()
    try:
        with source_path.open("r", encoding="utf-8") as file:
            raw = _as_mapping(yaml.safe_load(file), "根配置")
    except OSError as error:
        raise ConversionError(f"无法读取转换配置 {source_path}：{error}") from error

    source_raw = _as_mapping(_required(raw, "source", "根配置"), "source")
    output_raw = _as_mapping(_required(raw, "output", "根配置"), "output")
    representation_raw = _as_mapping(_required(raw, "representation", "根配置"), "representation")
    alignment_raw = _as_mapping(_required(raw, "alignment", "根配置"), "alignment")
    images_raw = _as_mapping(_required(raw, "images", "根配置"), "images")
    description_raw = _as_mapping(_required(raw, "description", "根配置"), "description")

    state_source = str(_required(representation_raw, "state_source", "representation"))
    if state_source not in {"joint_positions", "tcp_pose"}:
        raise ConversionError("representation.state_source 只支持 joint_positions 或 tcp_pose。")
    action_representation = str(_required(representation_raw, "action_representation", "representation"))
    if action_representation not in {"absolute", "relative"}:
        raise ConversionError("representation.action_representation 只支持 absolute 或 relative。")
    strategy = str(_required(alignment_raw, "strategy", "alignment"))
    if strategy not in {"previous", "nearest"}:
        raise ConversionError("alignment.strategy 只支持 previous 或 nearest。")

    target_width = _positive_int(images_raw.get("target_width"), "images.target_width", allow_none=True)
    target_height = _positive_int(images_raw.get("target_height"), "images.target_height", allow_none=True)
    if (target_width is None) != (target_height is None):
        raise ConversionError("images.target_width 和 images.target_height 必须同时设置或同时为 null。")
    pad_rgb_raw = images_raw.get("pad_rgb", [0, 0, 0])
    if not isinstance(pad_rgb_raw, list) or len(pad_rgb_raw) != 3:
        raise ConversionError("images.pad_rgb 必须是三个 0 到 255 的整数。")
    try:
        pad_rgb = tuple(int(value) for value in pad_rgb_raw)
    except (TypeError, ValueError) as error:
        raise ConversionError("images.pad_rgb 必须是三个 0 到 255 的整数。") from error
    if any(value < 0 or value > 255 for value in pad_rgb):
        raise ConversionError("images.pad_rgb 必须是三个 0 到 255 的整数。")

    cameras_raw = _required(images_raw, "cameras", "images")
    if not isinstance(cameras_raw, list) or not cameras_raw:
        raise ConversionError("images.cameras 必须是非空数组。")
    cameras = tuple(
        CameraSpec(
            source_name=str(_required(_as_mapping(camera, "images.cameras[]"), "source_name", "images.cameras[]")),
            feature_name=str(_required(_as_mapping(camera, "images.cameras[]"), "feature_name", "images.cameras[]")),
        )
        for camera in cameras_raw
    )
    if len({camera.source_name for camera in cameras}) != len(cameras):
        raise ConversionError("images.cameras.source_name 不能重复。")
    if len({camera.feature_name for camera in cameras}) != len(cameras):
        raise ConversionError("images.cameras.feature_name 不能重复。")
    if any("/" in camera.feature_name for camera in cameras):
        raise ConversionError("LeRobot v2.1 原始特征名不能包含 /；请使用 image、wrist_image 等名称。")

    max_state_age_raw = alignment_raw.get("max_state_age_ms")
    max_state_age_ms = None if max_state_age_raw is None else float(max_state_age_raw)
    if max_state_age_ms is not None and max_state_age_ms <= 0:
        raise ConversionError("alignment.max_state_age_ms 必须为正数或 null。")
    action_horizon_frames = _positive_int(
        representation_raw.get("action_horizon_frames", 1), "representation.action_horizon_frames"
    )
    fps_raw = output_raw.get("fps")
    fps = None if fps_raw is None else _positive_int(fps_raw, "output.fps")
    video_codec = str(output_raw.get("video_codec", "libx264"))
    if video_codec != "libx264":
        raise ConversionError("output.video_codec 当前只支持项目 FFmpeg 已验证的 libx264。")
    video_preset = str(output_raw.get("video_preset", "medium"))
    if video_preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}:
        raise ConversionError("output.video_preset 不是 libx264 支持的 preset。")
    video_crf = int(output_raw.get("video_crf", 23))
    if not 0 <= video_crf <= 51:
        raise ConversionError("output.video_crf 必须在 0 到 51 之间。")
    video_keyframe_interval = _positive_int(
        output_raw.get("video_keyframe_interval", 30), "output.video_keyframe_interval"
    )

    return ConversionConfig(
        source=SourceConfig(
            hdf5_root=_resolve_path(str(_required(source_raw, "hdf5_root", "source")), source_path.parent),
            trajectory_glob=str(source_raw.get("trajectory_glob", "**/trajectory.hdf5")),
            collection_config=_resolve_path(str(_required(source_raw, "collection_config", "source")), source_path.parent),
        ),
        output=OutputConfig(
            repo_id=str(_required(output_raw, "repo_id", "output")),
            root=_resolve_path(str(_required(output_raw, "root", "output")), source_path.parent),
            robot_type=str(output_raw.get("robot_type", "piper")),
            fps=fps,
            use_videos=bool(output_raw.get("use_videos", True)),
            video_codec=video_codec,
            video_preset=video_preset,
            video_crf=video_crf,
            video_keyframe_interval=video_keyframe_interval,
            image_writer_processes=int(output_raw.get("image_writer_processes", 0)),
            image_writer_threads=int(output_raw.get("image_writer_threads", 0)),
            overwrite=bool(output_raw.get("overwrite", False)),
        ),
        representation=RepresentationConfig(
            state_source=state_source,  # type: ignore[arg-type]
            include_gripper=bool(representation_raw.get("include_gripper", True)),
            action_representation=action_representation,  # type: ignore[arg-type]
            action_horizon_frames=action_horizon_frames,
        ),
        alignment=AlignmentConfig(
            strategy=strategy,  # type: ignore[arg-type]
            max_state_age_ms=max_state_age_ms,
        ),
        images=ImageConfig(
            target_width=target_width,
            target_height=target_height,
            pad_rgb=pad_rgb,  # type: ignore[arg-type]
            validate_source_size=bool(images_raw.get("validate_source_size", True)),
            cameras=cameras,
        ),
        description=DescriptionConfig(
            task=str(_required(description_raw, "task", "description")),
            intended_model=str(description_raw.get("intended_model", "")),
            notes=str(description_raw.get("notes", "")),
        ),
        source_path=source_path,
    )


def load_collection_metadata(path: Path) -> CollectionMetadata:
    """读取采集 YAML 的频率、相机尺寸、模态和姿态表示。"""

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = _as_mapping(yaml.safe_load(file), "采集配置")
    except OSError as error:
        raise ConversionError(f"无法读取采集配置 {path}：{error}") from error

    session = _as_mapping(_required(raw, "session", "采集配置"), "session")
    acquisition = _as_mapping(_required(raw, "acquisition", "采集配置"), "acquisition")
    robot = _as_mapping(_required(raw, "robot", "采集配置"), "robot")
    modalities = _as_mapping(_required(raw, "modalities", "采集配置"), "modalities")
    cameras_raw = _required(raw, "cameras", "采集配置")
    if not isinstance(cameras_raw, list):
        raise ConversionError("采集配置.cameras 必须是数组。")

    enabled_cameras: dict[str, tuple[int, int]] = {}
    for camera_value in cameras_raw:
        camera = _as_mapping(camera_value, "采集配置.cameras[]")
        if bool(camera.get("enabled", True)):
            name = str(_required(camera, "name", "采集配置.cameras[]"))
            enabled_cameras[name] = (
                int(_required(camera, "width", f"采集配置.cameras[{name}]")),
                int(_required(camera, "height", f"采集配置.cameras[{name}]")),
            )

    pose_representation = str(session.get("pose_representation", "xyz_xyzw"))
    if pose_representation not in {"xyz_rxryrz", "xyz_xyzw"}:
        raise ConversionError("采集配置.session.pose_representation 不受支持。")
    return CollectionMetadata(
        task=str(_required(session, "language_instruction", "采集配置.session")),
        camera_fps=float(_required(acquisition, "camera_rig_hz", "采集配置.acquisition")),
        max_alignment_age_ms=float(_required(acquisition, "max_alignment_age_ms", "采集配置.acquisition")),
        pose_representation=pose_representation,
        joint_count=int(robot.get("joint_count", 6)),
        enabled_cameras=enabled_cameras,
        enabled_modalities={name: bool(value) for name, value in modalities.items()},
        robot_name=str(robot.get("name", "piper")),
    )


def resize_with_padding(image: np.ndarray, target_width: int | None, target_height: int | None, pad_rgb: tuple[int, int, int]) -> np.ndarray:
    """按比例缩小并居中补边，避免把非方形场景拉伸为正方形。"""

    if target_width is None or target_height is None:
        return image
    source_height, source_width = image.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = np.asarray(
        Image.fromarray(image).resize((resized_width, resized_height), Image.Resampling.LANCZOS), dtype=np.uint8
    )
    canvas = np.full((target_height, target_width, 3), pad_rgb, dtype=np.uint8)
    y_offset = (target_height - resized_height) // 2
    x_offset = (target_width - resized_width) // 2
    canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
    return canvas


def align_timestamps(camera_timestamps: np.ndarray, state_timestamps: np.ndarray, strategy: AlignmentStrategy) -> tuple[np.ndarray, np.ndarray]:
    """将每个图像时刻匹配到机器人状态，并返回索引与绝对时间差。"""

    if len(camera_timestamps) == 0 or len(state_timestamps) == 0:
        raise ConversionError("相机和机器人状态序列均不能为空。")
    if not np.all(np.diff(camera_timestamps) > 0) or not np.all(np.diff(state_timestamps) > 0):
        raise ConversionError("相机和机器人时间戳必须严格递增。")

    if strategy == "previous":
        # side="right" 会让严格小于或等于图像时刻的最后一帧成为候选，保证不读取未来状态。
        indices = np.searchsorted(state_timestamps, camera_timestamps, side="right") - 1
        if np.any(indices < 0):
            raise ConversionError("存在早于第一条机器人状态的图像，无法做因果对齐。")
    else:
        right_indices = np.searchsorted(state_timestamps, camera_timestamps, side="left")
        left_indices = np.clip(right_indices - 1, 0, len(state_timestamps) - 1)
        right_indices = np.clip(right_indices, 0, len(state_timestamps) - 1)
        choose_left = np.abs(camera_timestamps - state_timestamps[left_indices]) <= np.abs(
            state_timestamps[right_indices] - camera_timestamps
        )
        indices = np.where(choose_left, left_indices, right_indices)
    ages_ms = np.abs(camera_timestamps - state_timestamps[indices]) * 1000.0
    return indices.astype(np.int64), ages_ms


def _read_string(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _read_series(file: h5py.File, group_name: str) -> tuple[np.ndarray, np.ndarray]:
    timestamp_path = f"puppet/{group_name}_raw/timestamp"
    data_path = f"puppet/{group_name}_raw/data"
    if timestamp_path not in file or data_path not in file:
        raise ConversionError(f"{file.filename} 缺少原始机器人序列 {group_name}_raw。")
    timestamps = np.asarray(file[timestamp_path][:], dtype=np.float64)
    data = np.asarray(file[data_path][:], dtype=np.float64)
    if timestamps.ndim != 1 or data.ndim != 2 or len(timestamps) != len(data):
        raise ConversionError(f"{file.filename} 的 {group_name}_raw 形状非法。")
    if not np.isfinite(timestamps).all() or not np.isfinite(data).all():
        raise ConversionError(f"{file.filename} 的 {group_name}_raw 含有 NaN 或 Inf。")
    return timestamps, data


def _state_group_name(state_source: StateSource) -> str:
    return "arm_single_position" if state_source == "joint_positions" else "end_effector_single_pose"


def _validate_collection_contract(config: ConversionConfig, metadata: CollectionMetadata) -> None:
    if not metadata.enabled_modalities.get("rgb", False):
        raise ConversionError("采集配置未启用 RGB，无法构建视觉 LeRobot 数据集。")
    required_modality = "arm_joint_positions" if config.representation.state_source == "joint_positions" else "tcp_pose"
    if not metadata.enabled_modalities.get(required_modality, False):
        raise ConversionError(f"采集配置未启用 {required_modality}，无法按当前 representation 转换。")
    if config.representation.include_gripper and not metadata.enabled_modalities.get("gripper_position", False):
        raise ConversionError("配置要求 include_gripper=true，但采集配置未启用 gripper_position。")
    for camera in config.images.cameras:
        if camera.source_name not in metadata.enabled_cameras:
            raise ConversionError(f"转换配置要求相机 {camera.source_name}，但采集配置中未启用该相机。")


def _read_images(file: h5py.File, camera: CameraSpec, index: int, expected_size: tuple[int, int] | None, config: ImageConfig) -> np.ndarray:
    path = f"camera_observations/color_images/{camera.source_name}"
    if path not in file:
        raise ConversionError(f"{file.filename} 缺少相机图像：{path}")
    encoded = np.asarray(file[path][index], dtype=np.uint8).tobytes()
    try:
        with Image.open(io.BytesIO(encoded)) as decoded:
            image = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    except Exception as error:
        raise ConversionError(f"{file.filename} 的 {camera.source_name} 第 {index} 帧无法解码。") from error
    if expected_size is not None and config.validate_source_size:
        expected_width, expected_height = expected_size
        if image.shape != (expected_height, expected_width, 3):
            raise ConversionError(
                f"{file.filename} 的 {camera.source_name} 尺寸为 {image.shape[1]}x{image.shape[0]}，"
                f"与采集配置 {expected_width}x{expected_height} 不一致。"
            )
    return resize_with_padding(image, config.target_width, config.target_height, config.pad_rgb)


def _state_names(state_source: StateSource, include_gripper: bool) -> list[str]:
    if state_source == "joint_positions":
        names = [f"joint_{index}" for index in range(1, 7)]
    else:
        names = ["tcp_x_m", "tcp_y_m", "tcp_z_m", "tcp_rx_rad", "tcp_ry_rad", "tcp_rz_rad"]
    return names + (["gripper_position_m"] if include_gripper else [])


def _build_action(current: np.ndarray, target: np.ndarray, config: RepresentationConfig, pose_representation: str) -> np.ndarray:
    """将相邻反馈状态变为监督动作；相对夹爪始终保持绝对开度。"""

    if config.action_representation == "absolute":
        return target.astype(np.float32, copy=False)
    if config.state_source == "tcp_pose" and pose_representation == "xyz_xyzw":
        raise ConversionError("TCP 四元数仅支持 absolute 动作；请改用 xyz_rxryrz 采集或 absolute。")

    action = target - current
    if config.state_source == "tcp_pose":
        # 欧拉角差需要回绕到 [-pi, pi]，否则跨越 pi 边界会产生伪大动作。
        action[3:6] = (action[3:6] + np.pi) % (2 * np.pi) - np.pi
    if config.include_gripper:
        # OpenPI 的常见约定是机器人主体为相对量，夹爪保留绝对目标开度。
        action[-1] = target[-1]
    return action.astype(np.float32, copy=False)


def _episode_vectors(
    file: h5py.File,
    config: ConversionConfig,
    metadata: CollectionMetadata,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """读取单条轨迹，按图像时刻对齐后返回状态、动作和相机帧索引。"""

    camera_timestamp_path = "camera_observations/timestamp"
    if camera_timestamp_path not in file:
        raise ConversionError(f"{file.filename} 缺少 {camera_timestamp_path}。")
    camera_timestamps = np.asarray(file[camera_timestamp_path][:], dtype=np.float64)
    state_timestamps, main_state = _read_series(file, _state_group_name(config.representation.state_source))
    if config.representation.state_source == "joint_positions" and main_state.shape[1] != metadata.joint_count:
        raise ConversionError(f"{file.filename} 的关节维度为 {main_state.shape[1]}，期望 {metadata.joint_count}。")
    if config.representation.state_source == "tcp_pose":
        expected_width = 6 if metadata.pose_representation == "xyz_rxryrz" else 7
        if main_state.shape[1] != expected_width:
            raise ConversionError(f"{file.filename} 的 TCP 维度为 {main_state.shape[1]}，期望 {expected_width}。")

    if config.representation.include_gripper:
        gripper_timestamps, gripper = _read_series(file, "end_effector_single_position")
        if gripper.shape[1] != 1 or not np.array_equal(state_timestamps, gripper_timestamps):
            raise ConversionError(f"{file.filename} 的夹爪与主体状态不是同一采样序列，拒绝混合对齐。")
        main_state = np.concatenate([main_state, gripper], axis=1)

    aligned_indices, ages_ms = align_timestamps(camera_timestamps, state_timestamps, config.alignment.strategy)
    max_state_age_ms = config.alignment.max_state_age_ms or metadata.max_alignment_age_ms
    if float(np.max(ages_ms)) > max_state_age_ms:
        raise ConversionError(
            f"{file.filename} 存在状态对齐年龄 {np.max(ages_ms):.2f}ms，超过允许值 {max_state_age_ms:.2f}ms。"
        )
    frame_count = len(camera_timestamps) - config.representation.action_horizon_frames
    if frame_count < 1:
        raise ConversionError(f"{file.filename} 图像帧数不足以构造动作监督。")
    source_indices = np.arange(frame_count, dtype=np.int64)
    current_states = main_state[aligned_indices[source_indices]]
    target_states = main_state[aligned_indices[source_indices + config.representation.action_horizon_frames]]
    actions = np.stack(
        [_build_action(current.copy(), target, config.representation, metadata.pose_representation) for current, target in zip(current_states, target_states)],
        axis=0,
    )
    task = _read_string(file["metadata/language_instruction"]) if "metadata/language_instruction" in file else metadata.task
    if not task.strip():
        raise ConversionError(f"{file.filename} 缺少非空语言任务。")
    return current_states.astype(np.float32), actions, source_indices, float(np.max(ages_ms)), task


def _create_features(config: ConversionConfig, metadata: CollectionMetadata) -> dict[str, dict[str, Any]]:
    names = _state_names(config.representation.state_source, config.representation.include_gripper)
    image_dtype = "video" if config.output.use_videos else "image"
    features: dict[str, dict[str, Any]] = {
        "state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "actions": {"dtype": "float32", "shape": (len(names),), "names": names},
    }
    for camera in config.images.cameras:
        if config.images.target_width is None or config.images.target_height is None:
            # 不缩放时仍需向 LeRobot 声明固定形状，直接采用采集 YAML 的相机尺寸。
            source_width, source_height = metadata.enabled_cameras[camera.source_name]
            image_height, image_width = source_height, source_width
        else:
            image_height, image_width = config.images.target_height, config.images.target_width
        features[camera.feature_name] = {
            "dtype": image_dtype,
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _prepare_output(config: ConversionConfig) -> Path:
    """分配暂存目录；仅在全量成功后才替换既有数据集。"""

    output_root = config.output.root
    if output_root.exists():
        if not config.output.overwrite:
            raise ConversionError(f"输出目录已存在：{output_root}。确认覆盖后将 output.overwrite 设为 true。")
    if output_root == output_root.parent or not output_root.name:
        raise ConversionError("输出根目录无效。")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    # 暂存目录与目标目录在同一文件系统，发布时 os.replace 才能保证原子性。
    return output_root.with_name(f".{output_root.name}.partial-{uuid4().hex}")


def _publish_output(staging_root: Path, output_root: Path, *, overwrite: bool) -> None:
    """将完整暂存数据集原子发布；异常前已有数据集保持不变。"""

    if output_root.exists():
        if not overwrite:
            raise ConversionError(f"输出目录已存在：{output_root}。")
        shutil.rmtree(output_root)
    os.replace(staging_root, output_root)


def _write_dataset_description(
    config: ConversionConfig,
    metadata: CollectionMetadata,
    reports: list[EpisodeReport],
    output_root: Path,
    fps: int,
) -> None:
    """写出训练数据的语义和可追溯来源，不把采集 YAML 的敏感字段复制进仓库。"""

    report_dict = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "converter_config": str(config.source_path),
        "collection_config": str(config.source.collection_config),
        "repo_id": config.output.repo_id,
        "fps": fps,
        "state_source": config.representation.state_source,
        "action_representation": config.representation.action_representation,
        "include_gripper": config.representation.include_gripper,
        "action_horizon_frames": config.representation.action_horizon_frames,
        "source_collection": {
            "camera_rig_hz": metadata.camera_fps,
            "max_alignment_age_ms": metadata.max_alignment_age_ms,
            "pose_representation": metadata.pose_representation,
            "robot_name": metadata.robot_name,
        },
        "alignment": {
            "strategy": config.alignment.strategy,
            "max_state_age_ms": config.alignment.max_state_age_ms or metadata.max_alignment_age_ms,
        },
        "image_size": [config.images.target_width, config.images.target_height],
        "media_storage": "mp4" if config.output.use_videos else "parquet_image",
        "video": {
            "codec": config.output.video_codec,
            "preset": config.output.video_preset,
            "crf": config.output.video_crf,
            "keyframe_interval": config.output.video_keyframe_interval,
        }
        if config.output.use_videos
        else None,
        "task": config.description.task,
        "intended_model": config.description.intended_model,
        "notes": config.description.notes,
        "episodes": [asdict(report) for report in reports],
    }
    (output_root / "conversion_manifest.json").write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    description = "\n".join(
        [
            "# Piper Corn In Plate LeRobot Dataset",
            "",
            f"- 任务：{config.description.task}",
            f"- 目标模型：{config.description.intended_model}",
            f"- 状态：{config.representation.state_source}，夹爪：{config.representation.include_gripper}",
            f"- 动作：{config.representation.action_representation}，目标为下一图像时刻的机器人反馈。",
            f"- 对齐：{config.alignment.strategy}；仅使用图像时刻之前的机器人状态。",
            f"- 图像：等比例缩放至 {config.images.target_width}x{config.images.target_height}，居中填充。",
            f"- 媒体：{'H.264 MP4' if config.output.use_videos else 'Parquet 内嵌图像'}。",
            f"- 轨迹：{len(reports)} 条；帧率：{fps} Hz。",
            "",
            config.description.notes,
            "",
            "OpenPI 的 `LeRobotLiberoDataConfig` 会将 LeRobot 的 `image`、`wrist_image`、"
            "`state` 和 `actions` 映射到训练内部字段。",
        ]
    )
    (output_root / "DATASET_DESCRIPTION.md").write_text(description + "\n", encoding="utf-8")


def convert_dataset(
    config: ConversionConfig, *, max_episodes: int | None = None, verbose: bool = True
) -> list[EpisodeReport]:
    """执行全量转换；只有所有成功写入的轨迹才会出现在最终 LeRobot 数据集中。"""

    metadata = load_collection_metadata(config.source.collection_config)
    _validate_collection_contract(config, metadata)
    if config.representation.state_source == "tcp_pose" and metadata.pose_representation == "xyz_xyzw":
        if config.representation.action_representation == "relative":
            raise ConversionError("四元数 TCP 不支持相对动作，请在配置中设为 absolute。")
    fps = config.output.fps or round(metadata.camera_fps)
    if fps <= 0:
        raise ConversionError("输出帧率必须为正整数。")
    trajectory_paths = sorted(config.source.hdf5_root.glob(config.source.trajectory_glob))
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ConversionError("max_episodes 必须为正整数。")
        trajectory_paths = trajectory_paths[:max_episodes]
    if not trajectory_paths:
        raise ConversionError(f"未在 {config.source.hdf5_root} 找到 {config.source.trajectory_glob}。")

    staging_root = _prepare_output(config)
    if not verbose:
        # datasets 在每个 episode 保存时都会输出进度条；批量转换时关闭以保持日志可读。
        import datasets

        datasets.disable_progress_bars()
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    if config.output.use_videos:
        from .ffmpeg import ProjectFFmpegError, install_lerobot_video_encoder

        try:
            install_lerobot_video_encoder(
                lerobot_dataset,
                codec=config.output.video_codec,
                preset=config.output.video_preset,
                crf=config.output.video_crf,
                keyframe_interval=config.output.video_keyframe_interval,
            )
        except ProjectFFmpegError as error:
            raise ConversionError(f"项目 FFmpeg 不可用，无法生成 MP4：{error}") from error

    dataset = lerobot_dataset.LeRobotDataset.create(
        repo_id=config.output.repo_id,
        root=staging_root,
        robot_type=config.output.robot_type or metadata.robot_name,
        fps=fps,
        features=_create_features(config, metadata),
        use_videos=config.output.use_videos,
        image_writer_processes=config.output.image_writer_processes,
        image_writer_threads=config.output.image_writer_threads,
    )
    reports: list[EpisodeReport] = []
    for episode_index, trajectory_path in enumerate(trajectory_paths):
        with h5py.File(trajectory_path, "r") as file:
            states, actions, image_indices, max_age_ms, task = _episode_vectors(file, config, metadata)
            for local_index, image_index in enumerate(image_indices):
                frame: dict[str, Any] = {
                    "state": states[local_index],
                    "actions": actions[local_index],
                    "task": task,
                }
                for camera in config.images.cameras:
                    frame[camera.feature_name] = _read_images(
                        file,
                        camera,
                        int(image_index),
                        metadata.enabled_cameras.get(camera.source_name),
                        config.images,
                    )
                dataset.add_frame(frame)
            dataset.save_episode()
        report = EpisodeReport(
            source_path=str(trajectory_path),
            frames=len(image_indices),
            max_state_age_ms=max_age_ms,
        )
        reports.append(report)
        if verbose:
            print(
                f"[{episode_index + 1}/{len(trajectory_paths)}] {trajectory_path.parent.name}: "
                f"{report.frames} 帧，最大状态年龄 {report.max_state_age_ms:.2f}ms"
            )
    _write_dataset_description(config, metadata, reports, staging_root, fps)
    _publish_output(staging_root, config.output.root, overwrite=config.output.overwrite)
    return reports
