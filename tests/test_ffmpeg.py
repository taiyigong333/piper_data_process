from pathlib import Path
import subprocess

from PIL import Image

from piper_data_process.ffmpeg import encode_video_with_project_ffmpeg, project_ffmpeg_info, project_ffmpeg_path


def test_project_ffmpeg_encodes_h264_mp4(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for frame_index in range(3):
        Image.new("RGB", (32, 24), color=(frame_index * 40, 20, 120)).save(images_dir / f"frame_{frame_index:06d}.png")
    video_path = tmp_path / "episode.mp4"

    info = project_ffmpeg_info()
    encode_video_with_project_ffmpeg(
        images_dir,
        video_path,
        fps=10,
        codec="libx264",
        preset="ultrafast",
        crf=23,
        keyframe_interval=10,
    )
    probe = subprocess.run(
        [str(project_ffmpeg_path()), "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-f", "null", "-"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert "ffmpeg version" in info["version"]
    assert video_path.stat().st_size > 0
    assert probe.returncode == 0
