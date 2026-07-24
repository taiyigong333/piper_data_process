"""项目内 FFmpeg 管理与 LeRobot 视频编码适配。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from types import ModuleType


class ProjectFFmpegError(RuntimeError):
    """项目内 FFmpeg 缺失或视频编码失败时抛出。"""


def project_ffmpeg_path() -> Path:
    """返回 imageio-ffmpeg 随 uv 环境安装的静态可执行文件。"""

    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise ProjectFFmpegError("缺少 imageio-ffmpeg；请执行 uv sync。") from error
    path = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    if not path.is_file() or not path.stat().st_mode & 0o111:
        raise ProjectFFmpegError(f"项目 FFmpeg 不可执行：{path}")
    return path


def _run_ffmpeg(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(project_ffmpeg_path()), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def project_ffmpeg_info() -> dict[str, str]:
    """验证静态 FFmpeg 与 H.264 编码器，避免运行时回退到损坏的系统二进制。"""

    version = _run_ffmpeg(["-version"])
    if version.returncode != 0:
        raise ProjectFFmpegError(f"项目 FFmpeg 无法启动：{version.stderr.strip()}")
    encoders = _run_ffmpeg(["-hide_banner", "-encoders"])
    if encoders.returncode != 0 or re.search(r"\blibx264\b", encoders.stdout) is None:
        raise ProjectFFmpegError("项目 FFmpeg 未提供 libx264 编码器。")
    return {"path": str(project_ffmpeg_path()), "version": version.stdout.splitlines()[0]}


def encode_video_with_project_ffmpeg(
    images_dir: Path | str,
    video_path: Path | str,
    fps: int,
    *,
    codec: str,
    preset: str,
    crf: int,
    keyframe_interval: int,
) -> None:
    """将 LeRobot 临时 PNG 序列编码为 H.264 MP4。"""

    images_dir = Path(images_dir)
    video_path = Path(video_path)
    if not (images_dir / "frame_000000.png").is_file():
        raise ProjectFFmpegError(f"缺少视频首帧：{images_dir}")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(images_dir / "frame_%06d.png"),
        "-an",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-g",
        str(keyframe_interval),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    result = _run_ffmpeg(command)
    if result.returncode != 0 or not video_path.is_file() or video_path.stat().st_size == 0:
        raise ProjectFFmpegError(f"MP4 编码失败：{result.stderr.strip()}")


def install_lerobot_video_encoder(
    lerobot_dataset_module: ModuleType,
    *,
    codec: str,
    preset: str,
    crf: int,
    keyframe_interval: int,
) -> None:
    """以固定 LeRobot 提交的模块入口接入项目 FFmpeg，避免系统 FFmpeg 被调用。"""

    project_ffmpeg_info()

    def encode(images_dir: Path | str, video_path: Path | str, fps: int, **_: object) -> None:
        encode_video_with_project_ffmpeg(
            images_dir,
            video_path,
            fps,
            codec=codec,
            preset=preset,
            crf=crf,
            keyframe_interval=keyframe_interval,
        )

    lerobot_dataset_module.encode_video_frames = encode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查由 uv 环境管理的项目 FFmpeg。")
    parser.add_argument("--check", action="store_true", help="验证可执行文件与 libx264 编码器")
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(project_ffmpeg_info(), ensure_ascii=False))
    else:
        print(project_ffmpeg_path())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
