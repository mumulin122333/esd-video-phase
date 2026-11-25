from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .contracts import VideoWindow


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "video"


def probe_duration(video: Path, ffprobe: str = "ffprobe") -> float:
    if shutil.which(ffprobe) is None:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "Neither ffprobe nor PyAV is available. Install ffmpeg or `pip install av`."
            ) from exc
        with av.open(str(video)) as container:
            if container.duration is not None:
                duration = float(container.duration / av.time_base)
            else:
                stream = next((item for item in container.streams if item.type == "video"), None)
                if stream is None or stream.duration is None or stream.time_base is None:
                    raise ValueError(f"cannot determine video duration for {video}")
                duration = float(stream.duration * stream.time_base)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"invalid video duration for {video}: {duration}")
        return duration

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    duration = float(json.loads(completed.stdout)["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid video duration for {video}: {duration}")
    return duration


def window_starts(duration: float, window_seconds: float, stride_seconds: float) -> list[float]:
    if duration <= window_seconds:
        return [0.0]
    last = duration - window_seconds
    count = int(math.floor(last / stride_seconds))
    starts = [round(index * stride_seconds, 6) for index in range(count + 1)]
    if not math.isclose(starts[-1], last, abs_tol=1e-3):
        starts.append(round(last, 6))
    return starts


def extract_window(
    source: Path,
    destination: Path,
    start_seconds: float,
    window_seconds: float,
    *,
    ffmpeg: str = "ffmpeg",
    fps: int = 1,
    width: int = 512,
    height: int = 448,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    if shutil.which(ffmpeg) is None:
        _extract_window_pyav(
            source,
            temp,
            start_seconds,
            window_seconds,
            fps=fps,
            width=width,
            height=height,
        )
        temp.replace(destination)
        return
    video_filter = (
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=bicubic,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"tpad=stop_mode=clone:stop_duration={window_seconds}"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(source),
        "-t",
        f"{window_seconds:.6f}",
        "-an",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp),
    ]
    subprocess.run(command, check=True)
    temp.replace(destination)


def _extract_window_pyav(
    source: Path,
    destination: Path,
    start_seconds: float,
    window_seconds: float,
    *,
    fps: int,
    width: int,
    height: int,
) -> None:
    """PyAV fallback for A100 hosts without ffmpeg/ffprobe command-line tools."""

    try:
        import av
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg is unavailable and the PyAV fallback needs `av` and `Pillow`."
        ) from exc

    target_count = max(1, int(round(window_seconds * fps)))
    target_times = [start_seconds + (index + 0.5) / fps for index in range(target_count)]
    selected: list[Image.Image] = []
    previous_image: Image.Image | None = None
    previous_time: float | None = None

    with av.open(str(source)) as input_container:
        stream = next((item for item in input_container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"no video stream found in {source}")
        if stream.time_base is not None:
            seek_pts = max(0, int(start_seconds / float(stream.time_base)))
            input_container.seek(seek_pts, stream=stream, backward=True)

        target_index = 0
        for frame in input_container.decode(stream):
            if frame.pts is None or stream.time_base is None:
                continue
            frame_time = float(frame.pts * stream.time_base)
            if frame_time < start_seconds - 2:
                continue
            current_image = frame.to_image().convert("RGB")
            while target_index < target_count and frame_time >= target_times[target_index]:
                if (
                    previous_image is not None
                    and previous_time is not None
                    and abs(target_times[target_index] - previous_time)
                    <= abs(frame_time - target_times[target_index])
                ):
                    selected.append(previous_image.copy())
                else:
                    selected.append(current_image.copy())
                target_index += 1
            previous_image = current_image
            previous_time = frame_time
            if target_index >= target_count:
                break

    if previous_image is None and not selected:
        raise ValueError(f"no frames decoded from {source} at {start_seconds}s")
    last = selected[-1] if selected else previous_image
    assert last is not None
    while len(selected) < target_count:
        selected.append(last.copy())

    with av.open(str(destination), mode="w") as output_container:
        output_stream = output_container.add_stream("libx264", rate=fps)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"crf": "18", "preset": "veryfast"}
        for image in selected:
            image.thumbnail((width, height), Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (width, height), "black")
            canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
            for packet in output_stream.encode(av.VideoFrame.from_image(canvas)):
                output_container.mux(packet)
        for packet in output_stream.encode():
            output_container.mux(packet)


def prepare_video_windows(
    video: Path,
    case_id: str,
    output_dir: Path,
    *,
    window_seconds: float = 32.0,
    stride_seconds: float = 8.0,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    overwrite: bool = False,
) -> list[VideoWindow]:
    video = video.resolve()
    duration = probe_duration(video, ffprobe)
    video_id = video.stem
    safe_video_id = safe_name(video_id)
    windows: list[VideoWindow] = []
    pending: list[tuple[Path, float]] = []
    for index, start in enumerate(window_starts(duration, window_seconds, stride_seconds)):
        source_end = min(duration, start + window_seconds)
        clip = output_dir / safe_video_id / f"window_{index:05d}_{start:010.3f}.mp4"
        if overwrite or not clip.is_file():
            pending.append((clip, start))
        windows.append(
            VideoWindow(
                window_id=f"{safe_video_id}-{index:05d}",
                case_id=case_id,
                video_id=video_id,
                source_video=video,
                clip_path=clip.resolve(),
                start_seconds=start,
                source_end_seconds=source_end,
                model_duration_seconds=window_seconds,
            )
        )

    workers = max(1, int(os.environ.get("RENJI_QWEN_FFMPEG_WORKERS", "8")))

    def prepare(task: tuple[Path, float]) -> None:
        clip, start = task
        extract_window(video, clip, start, window_seconds, ffmpeg=ffmpeg)

    if workers == 1:
        for task in pending:
            prepare(task)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(prepare, pending))
    return windows


def discover_videos(paths: list[Path], video_dir: Path | None = None) -> list[Path]:
    found = [path.resolve() for path in paths]
    if video_dir is not None:
        found.extend(
            path.resolve()
            for path in sorted(video_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
    unique = list(dict.fromkeys(found))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"video files do not exist: {missing}")
    if not unique:
        raise ValueError("no input videos found")
    duplicate_ids = {path.stem for path in unique if sum(item.stem == path.stem for item in unique) > 1}
    if duplicate_ids:
        raise ValueError(f"video filename stems must be unique for the GUI: {sorted(duplicate_ids)}")
    return unique
