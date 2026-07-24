from pathlib import Path

import numpy as np
import pytest

from piper_data_process.conversion import (
    CameraSpec,
    CollectionMetadata,
    ConversionError,
    ImageConfig,
    OutputConfig,
    RepresentationConfig,
    align_timestamps,
    resize_with_padding,
    _build_action,
    _create_features,
)


def test_previous_alignment_is_causal() -> None:
    camera = np.array([1.04, 1.11, 1.19])
    state = np.array([1.00, 1.10, 1.20])

    indices, ages_ms = align_timestamps(camera, state, "previous")

    assert indices.tolist() == [0, 1, 1]
    assert ages_ms == pytest.approx([40.0, 10.0, 90.0])


def test_previous_alignment_rejects_camera_before_first_state() -> None:
    with pytest.raises(ConversionError, match="因果对齐"):
        align_timestamps(np.array([0.9]), np.array([1.0, 1.1]), "previous")


def test_resize_preserves_aspect_ratio_and_pads() -> None:
    image = np.full((100, 200, 3), fill_value=(10, 20, 30), dtype=np.uint8)

    resized = resize_with_padding(image, target_width=100, target_height=100, pad_rgb=(0, 0, 0))

    assert resized.shape == (100, 100, 3)
    assert np.all(resized[:25] == 0)
    assert np.all(resized[25:75] == (10, 20, 30))


def test_original_resolution_uses_camera_size_from_collection_config() -> None:
    metadata = CollectionMetadata(
        task="task",
        camera_fps=30.0,
        max_alignment_age_ms=100.0,
        pose_representation="xyz_rxryrz",
        joint_count=6,
        enabled_cameras={"camera_front": (1280, 720)},
        enabled_modalities={},
        robot_name="piper",
    )
    image_config = ImageConfig(
        target_width=None,
        target_height=None,
        pad_rgb=(0, 0, 0),
        validate_source_size=True,
        cameras=(CameraSpec(source_name="camera_front", feature_name="image"),),
    )
    config = type("Config", (), {
        "representation": RepresentationConfig("joint_positions", True, "relative", 1),
        "images": image_config,
        "output": OutputConfig("repo", Path("/tmp/dataset"), "piper", 30, False, "libx264", "medium", 23, 30, 0, 0, False),
    })()

    features = _create_features(config, metadata)

    assert features["image"]["shape"] == (720, 1280, 3)


def test_video_mode_uses_lerobot_video_feature() -> None:
    metadata = CollectionMetadata(
        task="task",
        camera_fps=30.0,
        max_alignment_age_ms=100.0,
        pose_representation="xyz_rxryrz",
        joint_count=6,
        enabled_cameras={"camera_front": (1280, 720)},
        enabled_modalities={},
        robot_name="piper",
    )
    image_config = ImageConfig(
        target_width=224,
        target_height=224,
        pad_rgb=(0, 0, 0),
        validate_source_size=True,
        cameras=(CameraSpec(source_name="camera_front", feature_name="image"),),
    )
    config = type("Config", (), {
        "representation": RepresentationConfig("joint_positions", True, "absolute", 1),
        "images": image_config,
        "output": OutputConfig("repo", Path("/tmp/dataset"), "piper", 30, True, "libx264", "medium", 23, 30, 0, 0, False),
    })()

    features = _create_features(config, metadata)

    assert features["image"]["dtype"] == "video"


def test_relative_joint_action_keeps_gripper_absolute() -> None:
    config = RepresentationConfig(
        state_source="joint_positions",
        include_gripper=True,
        action_representation="relative",
        action_horizon_frames=1,
    )
    current = np.array([1, 2, 3, 4, 5, 6, 0.2], dtype=np.float64)
    target = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.8], dtype=np.float64)

    action = _build_action(current, target, config, "xyz_rxryrz")

    assert action == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])


def test_absolute_joint_action_keeps_next_target() -> None:
    config = RepresentationConfig(
        state_source="joint_positions",
        include_gripper=True,
        action_representation="absolute",
        action_horizon_frames=1,
    )
    current = np.array([1, 2, 3, 4, 5, 6, 0.2], dtype=np.float64)
    target = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.8], dtype=np.float64)

    action = _build_action(current, target, config, "xyz_rxryrz")

    assert action == pytest.approx(target)


def test_relative_tcp_wraps_euler_angle() -> None:
    config = RepresentationConfig(
        state_source="tcp_pose",
        include_gripper=False,
        action_representation="relative",
        action_horizon_frames=1,
    )
    current = np.array([0, 0, 0, 0, 0, 3.13], dtype=np.float64)
    target = np.array([0, 0, 0, 0, 0, -3.13], dtype=np.float64)

    action = _build_action(current, target, config, "xyz_rxryrz")

    assert action[-1] == pytest.approx(0.0231853, abs=1e-5)
