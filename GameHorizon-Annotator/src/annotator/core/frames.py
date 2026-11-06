from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .schema import NS_PER_S, atomic_write_json


def load_annotation_frames(path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        frames = payload.get("frames") or []
        meta = payload
    elif isinstance(payload, list):
        frames = payload
        meta = {}
    else:
        raise ValueError(f"invalid annotations JSON: {path}")
    clean = [dict(frame) for frame in frames if isinstance(frame, dict)]
    by_index = {int(frame.get("frame_index", index)): frame for index, frame in enumerate(clean)}
    return meta, clean, by_index


def infer_fps(frames: Sequence[Mapping[str, Any]], fallback: float) -> float:
    timestamps = []
    for frame in frames[: min(len(frames), 5000)]:
        try:
            timestamps.append(int(frame["timestamp_ns"]))
        except (KeyError, TypeError, ValueError):
            continue
    diffs = sorted(b - a for a, b in zip(timestamps, timestamps[1:]) if b > a)
    if not diffs:
        return fallback
    median = diffs[len(diffs) // 2]
    return NS_PER_S / median if median > 0 else fallback


def uniform_sample(values: Sequence[int] | Sequence[Path], maximum: int) -> list[Any]:
    if maximum <= 0 or len(values) <= maximum:
        return list(values)
    if maximum == 1:
        return [values[0]]
    selected: list[Any] = []
    seen: set[Any] = set()
    for position in range(maximum):
        index = round(position * (len(values) - 1) / (maximum - 1))
        value = values[index]
        if value not in seen:
            selected.append(value)
            seen.add(value)
    return selected


def frame_indices_for_segment(
    segment: Mapping[str, Any], frame_by_index: Mapping[int, Mapping[str, Any]]
) -> List[int]:
    try:
        source_start = int(segment["upstream_source_start_frame"])
        source_end = int(segment["upstream_source_end_frame"])
    except (KeyError, TypeError, ValueError):
        source_start = source_end = -1
    explicit_source_space = any(
        frame.get("source_frame_index") is not None
        and int(frame.get("source_frame_index")) != int(index)
        for index, frame in frame_by_index.items()
    )
    if explicit_source_space and source_start >= 0 and source_end >= source_start:
        matched = [
            index
            for index, frame in frame_by_index.items()
            if source_start <= int(frame.get("source_frame_index")) <= source_end
        ]
        if matched:
            return sorted(matched)

    # ``start_frame``/``end_frame`` are the canonical, gap-free bounds after
    # Stage 1 coverage normalization.  Prefer them over retained upstream
    # evidence spans; otherwise frames inserted to close a refiner boundary
    # gap would be omitted from Level 1 even though the manifest claims a
    # contiguous timeline.
    try:
        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
    except (KeyError, TypeError, ValueError):
        start_frame = end_frame = -1
    if start_frame >= 0 and end_frame >= start_frame:
        matched = [
            index for index in range(start_frame, end_frame + 1) if index in frame_by_index
        ]
        if matched:
            return matched

    if source_start >= 0 and source_end >= source_start:
        matched = [
            index
            for index, frame in frame_by_index.items()
            if source_start
            <= int(frame.get("source_frame_index", frame.get("frame_index", index)))
            <= source_end
        ]
        if matched:
            return sorted(matched)

    try:
        start_ts = int(segment["start_timestamp_ns"])
        end_ts = int(segment["end_timestamp_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"segment {segment.get('segment_index')} has no usable frame or timestamp span") from exc
    return [
        index
        for index, frame in frame_by_index.items()
        if start_ts <= int(frame.get("timestamp_ns", -1)) <= end_ts
    ]


def partition_segment_timeline(
    segments: Sequence[Mapping[str, Any]],
    frame_by_index: Mapping[int, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Turn inclusive/rounded 1b spans into one gap-free timeline partition.

    Game-specific refiners may round a semantic split to nearby frames. In the
    observed GTA and Cyberpunk outputs that left small holes, while other games
    shared a transition frame between adjacent segments. A canonical manifest
    needs each instant exactly once, so every non-final segment ends at the next
    segment's start boundary. The original source span is retained for audit.
    """

    result = [dict(segment) for segment in segments]

    def integer(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def timestamp_for_start(item: Mapping[str, Any]) -> int | None:
        value = integer(item.get("start_timestamp_ns"))
        if value is not None:
            return value
        frame_index = integer(item.get("start_frame"))
        frame = frame_by_index.get(frame_index) if frame_index is not None else None
        return integer(frame.get("timestamp_ns")) if frame else None

    for index, item in enumerate(result):
        upstream_source_start_frame = integer(item.get("source_start_frame"))
        upstream_source_end_frame = integer(item.get("source_end_frame"))
        source_start_frame = integer(item.get("start_frame"))
        source_end_frame = integer(item.get("end_frame"))
        source_start_ts = timestamp_for_start(item)
        source_end_ts = integer(item.get("end_timestamp_ns"))
        try:
            source_duration = float(item.get("duration_s"))
        except (TypeError, ValueError):
            source_duration = None

        item["source_start_frame"] = source_start_frame
        item["source_end_frame"] = source_end_frame
        if upstream_source_start_frame is not None:
            item["upstream_source_start_frame"] = upstream_source_start_frame
        if upstream_source_end_frame is not None:
            item["upstream_source_end_frame"] = upstream_source_end_frame
        item["source_start_timestamp_ns"] = source_start_ts
        item["source_end_timestamp_ns"] = source_end_ts
        item["source_duration_s"] = source_duration

        if source_start_ts is not None:
            item["start_timestamp_ns"] = source_start_ts

        if index + 1 < len(result):
            next_item = result[index + 1]
            boundary_ts = timestamp_for_start(next_item)
            next_start_frame = integer(next_item.get("start_frame"))
            if (
                source_start_frame is not None
                and next_start_frame is not None
                and next_start_frame > source_start_frame
            ):
                item["end_frame"] = next_start_frame - 1
        else:
            boundary_ts = source_end_ts
            if boundary_ts is None and source_end_frame is not None:
                final_frame = frame_by_index.get(source_end_frame)
                if final_frame:
                    boundary_ts = integer(
                        final_frame.get("next_frame_timestamp_ns")
                        or final_frame.get("timestamp_ns")
                    )

        if (
            source_start_ts is not None
            and boundary_ts is not None
            and boundary_ts > source_start_ts
        ):
            item["end_timestamp_ns"] = boundary_ts
            item["duration_s"] = round((boundary_ts - source_start_ts) / 1_000_000_000, 9)
        item["timeline_partitioned"] = True
    return result


def probe_video_start_ns(video: Path) -> int:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format_tags=creation_time",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    value = completed.stdout.strip()
    if not value:
        return 0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0
    return int(parsed.timestamp() * NS_PER_S)


def resolve_video_start_ns(
    video: Path | None,
    frames: Sequence[Mapping[str, Any]],
    explicit: int | None,
) -> int:
    if explicit is not None:
        return explicit
    if video is None:
        return 0
    origin = probe_video_start_ns(video)
    if not origin:
        return 0
    timestamps = [int(frame.get("timestamp_ns", 0)) for frame in frames[:100] if frame.get("timestamp_ns")]
    if not timestamps:
        return 0
    # Relative annotation timestamps are normally far below a Unix epoch timestamp.
    if max(timestamps) < 100_000_000_000_000_000:
        return 0
    return origin


def mouse_event_stream(frame: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Return the array carrying relative motion and raw button flags.

    Two capture schemas are in circulation. The newer one splits the pointer
    into a semantic ``mouse_events`` layer - absolute ``x``/``y`` moves, clicks
    as ``button``/``pressed``, wheel ticks - plus a raw ``raw_mouse_events``
    layer; the older one puts the raw stream in ``mouse_events`` and has no
    second array. Only the raw stream carries per-event ``dx``/``dy`` and the
    ``RI_MOUSE_*`` names the keybind tables are written against, so prefer it
    and fall back to the single-array layout.
    """

    raw = frame.get("raw_mouse_events")
    if isinstance(raw, list) and raw:
        return [event for event in raw if isinstance(event, dict)]
    events = frame.get("mouse_events")
    return [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []


def _mouse_button_names(event: Mapping[str, Any]) -> List[str]:
    names = event.get("button_flag_names")
    if isinstance(names, list):
        return [str(name) for name in names if name]
    button = event.get("button")
    if button is None:
        return []
    # Semantic layer: name the edge the way the raw flags and keybinds do.
    edge = "DOWN" if event.get("pressed", True) else "UP"
    return [f"RI_MOUSE_{str(button).upper()}_BUTTON_{edge}"]


def compact_actions(frame: Mapping[str, Any]) -> Dict[str, Any]:
    keyboard = []
    for event in frame.get("keyboard_events", []) or []:
        if not isinstance(event, dict):
            continue
        action = event.get("action")
        if action:
            keyboard.append(str(action))
        else:
            event_type = str(event.get("event_type", "")).strip()
            key_name = str(event.get("key_name", "")).strip()
            keyboard.append("_".join(part for part in (event_type, key_name) if part))
    mouse: List[str] = []
    movement: List[List[float]] = []
    for event in mouse_event_stream(frame):
        mouse.extend(_mouse_button_names(event))
        try:
            dx = float(event.get("dx", 0) or 0)
            dy = float(event.get("dy", 0) or 0)
        except (TypeError, ValueError):
            continue
        # Deltas are kept per event, in order, rather than summed: a frame can
        # hold a flick and its correction, and the sum of those is zero.
        if dx or dy:
            movement.append([dx, dy])
    return {
        "frame_index": frame.get("frame_index"),
        "keyboard": keyboard,
        "mouse_buttons": mouse,
        "mouse_movement": movement,
    }


@dataclass
class FrameSource:
    frames_dir: Path | None
    clips_dir: Path | None
    video: Path | None
    frame_by_index: Dict[int, Dict[str, Any]]
    video_start_ns: int
    fps: float | None = None
    image_ext: str = ".jpg"
    jpeg_quality: int = 90
    max_width: int = 1280

    def __post_init__(self) -> None:
        self._frame_files = self._index_frame_files(self.frames_dir)
        self._clip_files = self._index_clip_files(self.clips_dir)
        self.sparse_batch_audit: Dict[str, Any] | None = None
        self.sparse_batch_error: str | None = None

    @staticmethod
    def _index_frame_files(directory: Path | None) -> Dict[int, Path]:
        if directory is None or not directory.is_dir():
            return {}
        result: Dict[int, Path] = {}
        for path in directory.iterdir():
            if path.is_file() and path.stem.isdigit() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                result[int(path.stem)] = path
        return result

    @staticmethod
    def _index_clip_files(directory: Path | None) -> Dict[int, Path]:
        if directory is None or not directory.is_dir():
            return {}
        result: Dict[int, Path] = {}
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in {".mp4", ".mkv", ".mov", ".webm"}:
                continue
            prefix = path.stem.split("__", 1)[0].split("_", 1)[0]
            if prefix.isdigit():
                result.setdefault(int(prefix), path)
        return result

    def materialize(
        self,
        *,
        segment: Mapping[str, Any],
        frame_index: int,
        destination: Path,
    ) -> tuple[bool, str, str | None]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_frame = self._frame_files.get(frame_index)
        if source_frame is not None:
            try:
                self._materialize_image(source_frame, destination)
                return True, "frames_dir", str(source_frame)
            except RuntimeError:
                pass

        segment_index = int(segment.get("segment_index", -1))
        clip = self._clip_files.get(segment_index)
        frame = self.frame_by_index.get(frame_index, {})
        timestamp_ns = int(frame.get("timestamp_ns", segment.get("start_timestamp_ns", 0)) or 0)
        try:
            video_pts_ns = int(frame["video_pts_ns"])
        except (KeyError, TypeError, ValueError):
            video_pts_ns = None
        if clip is not None:
            if video_pts_ns is not None and segment.get("start_video_pts_ns") is not None:
                start_ns = int(segment["start_video_pts_ns"])
                relative_s = max(0.0, (video_pts_ns - start_ns) / NS_PER_S)
            else:
                start_ns = int(segment.get("start_timestamp_ns", timestamp_ns) or timestamp_ns)
                relative_s = max(0.0, (timestamp_ns - start_ns) / NS_PER_S)
            if self._extract_video_frame(clip, relative_s, destination):
                return True, "clip", str(clip)

        if self.video is not None:
            relative_s = (
                max(0.0, video_pts_ns / NS_PER_S)
                if video_pts_ns is not None
                else max(0.0, (timestamp_ns - self.video_start_ns) / NS_PER_S)
            )
            if self._extract_video_frame(self.video, relative_s, destination):
                return True, "video", str(self.video)
        return False, "missing", None

    def materialize_sparse_video_batch(
        self,
        *,
        frame_indices: Sequence[int],
        staging_dir: Path,
        chunk_size: int = 100,
        chunk_jobs: int = 1,
        ffmpeg_threads: int = 0,
        pts_cache_dir: Path | None = None,
    ) -> bool:
        """Decode scene-wide sampled video frames in bounded PTS-select chunks.

        Sampling previously launched one seeking FFmpeg process per retained
        frame. This method writes the normalized sparse target list once, then
        reuses the exact-PTS chunk extractor and its persistent manifest.
        """

        if (
            self.video is None
            or self.clips_dir is not None
            or self.image_ext.lower() not in {".jpg", ".jpeg"}
            or not frame_indices
        ):
            return False
        # Preserve frames_dir priority. Only video-decode sampled frames that
        # are not already available as materialized source images.
        indices = sorted(
            int(value)
            for value in {int(item) for item in frame_indices}
            if int(value) not in self._frame_files
        )
        if not indices:
            return False
        targets: List[Dict[str, int]] = []
        for frame_index in indices:
            frame = self.frame_by_index.get(frame_index)
            if frame is None:
                return False
            try:
                video_pts_ns = int(frame["video_pts_ns"])
            except (KeyError, TypeError, ValueError):
                return False
            targets.append(
                {
                    "frame_index": frame_index,
                    "video_pts_ns": video_pts_ns,
                }
            )
        target_document = {"frames": targets}
        targets_path = staging_dir.with_name(
            f"{staging_dir.name}_targets.json"
        )
        prior_targets: Any = None
        if targets_path.is_file():
            try:
                prior_targets = json.loads(
                    targets_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, json.JSONDecodeError):
                prior_targets = None
        if prior_targets != target_document:
            atomic_write_json(targets_path, target_document)
        try:
            from .ffmpeg_pts import extract_frames_by_pts_ffmpeg

            manifest = extract_frames_by_pts_ffmpeg(
                video_path=self.video,
                annotations_path=targets_path,
                output_dir=staging_dir,
                max_width=self.max_width,
                jpeg_quality=self.jpeg_quality,
                tolerance_ms=50.0,
                chunk_size=max(1, int(chunk_size)),
                chunk_jobs=max(1, int(chunk_jobs)),
                ffmpeg_threads=max(0, int(ffmpeg_threads)),
                pts_cache_dir=pts_cache_dir,
            )
        except (
            FileNotFoundError,
            ValueError,
            RuntimeError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            self.sparse_batch_error = (
                f"{type(exc).__name__}: {str(exc)[:500]}"
            )
            return False
        expected = [
            staging_dir / f"{frame_index:06d}.jpg"
            for frame_index in indices
        ]
        if not all(
            path.is_file() and path.stat().st_size > 0 for path in expected
        ):
            self.sparse_batch_error = "sparse batch output is incomplete"
            return False
        self.sparse_batch_audit = dict(manifest)
        self.sparse_batch_error = None
        return True

    @staticmethod
    def materialize_sparse_frame(
        *,
        frame_index: int,
        staging_dir: Path,
        destination: Path,
    ) -> bool:
        source = staging_dir / f"{int(frame_index):06d}.jpg"
        if not source.is_file() or source.stat().st_size <= 0:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.unlink(missing_ok=True)
            os.link(source, destination)
        except OSError:
            try:
                shutil.copy2(source, destination)
            except OSError:
                return False
        return destination.is_file() and destination.stat().st_size > 0

    def materialize_video_batch(
        self,
        *,
        frame_indices: Sequence[int],
        destination_dir: Path,
    ) -> bool:
        """Decode one contiguous annotation span with a single ffmpeg process.

        Full-frame E2E runs previously launched ffmpeg once per annotation
        frame.  Long scenes therefore spent most of their time repeatedly
        opening and seeking the same video.  When video is the active source
        and the requested annotation indices are contiguous, decode the span
        at the annotation FPS in one pass.  Any short/failed batch is removed
        and the caller can safely fall back to exact per-frame extraction.
        """

        if (
            self.video is None
            or self.clips_dir is not None
            or self.fps is None
            or self.fps <= 0
            or not frame_indices
        ):
            return False
        indices = [int(value) for value in frame_indices]
        first_index = indices[0]
        if indices != list(range(first_index, first_index + len(indices))):
            return False
        frame = self.frame_by_index.get(first_index, {})
        try:
            video_pts_ns = int(frame["video_pts_ns"])
        except (KeyError, TypeError, ValueError):
            timestamp_ns = int(frame.get("timestamp_ns", 0) or 0)
            video_pts_ns = timestamp_ns - int(self.video_start_ns)
        relative_s = max(0.0, video_pts_ns / NS_PER_S)
        destination_dir.mkdir(parents=True, exist_ok=True)
        pattern = destination_dir / f"%06d{self.image_ext}"
        filters = [f"fps=fps={self.fps:.12g}:start_time=0"]
        if self.max_width > 0:
            filters.append(f"scale={self.max_width}:-2:force_original_aspect_ratio=decrease")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{relative_s:.9f}",
            "-i",
            str(self.video),
            "-an",
            "-frames:v",
            str(len(indices)),
            "-vf",
            ",".join(filters),
            "-start_number",
            str(first_index),
        ]
        if self.image_ext.lower() in {".jpg", ".jpeg"}:
            qscale = max(2, min(31, round((100 - self.jpeg_quality) * 0.29) + 2))
            command.extend(["-q:v", str(qscale)])
        command.append(str(pattern))
        expected = [destination_dir / f"{index:06d}{self.image_ext}" for index in indices]
        timeout = max(120.0, (len(indices) / self.fps) * 4.0 + 120.0)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            completed = None
        ok = (
            completed is not None
            and completed.returncode == 0
            and all(path.is_file() and path.stat().st_size > 0 for path in expected)
        )
        if not ok:
            for path in expected:
                path.unlink(missing_ok=True)
        return ok

    def _materialize_image(self, source: Path, destination: Path) -> None:
        same_format = source.suffix.lower() == destination.suffix.lower()
        if same_format and self.max_width <= 0:
            shutil.copy2(source, destination)
            return
        if same_format and self.max_width > 0:
            try:
                from PIL import Image

                with Image.open(source) as image:
                    if image.width <= self.max_width:
                        shutil.copy2(source, destination)
                        return
            except (OSError, ValueError):
                # Let ffmpeg validate/convert malformed or unsupported images.
                pass
        if same_format and self.max_width > 0 and not self._convert_image(source, destination):
            raise RuntimeError(f"failed to resize {source}")
        if same_format:
            return
        if not self._convert_image(source, destination):
            raise RuntimeError(f"failed to convert {source}")

    def _convert_image(self, source: Path, destination: Path) -> bool:
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
        command.extend(self._output_options(destination))
        command.append(str(destination))
        return self._run_ffmpeg(command, destination)

    def _extract_video_frame(self, video: Path, seconds: float, destination: Path) -> bool:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{seconds:.9f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
        ]
        command.extend(self._output_options(destination))
        command.append(str(destination))
        return self._run_ffmpeg(command, destination)

    def _output_options(self, destination: Path) -> List[str]:
        options: List[str] = []
        if self.max_width > 0:
            options.extend(["-vf", f"scale={self.max_width}:-2:force_original_aspect_ratio=decrease"])
        if destination.suffix.lower() in {".jpg", ".jpeg"}:
            qscale = max(2, min(31, round((100 - self.jpeg_quality) * 0.29) + 2))
            options.extend(["-q:v", str(qscale)])
        return options

    @staticmethod
    def _run_ffmpeg(command: Sequence[str], destination: Path) -> bool:
        try:
            completed = subprocess.run(command, capture_output=True, timeout=120, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0 and destination.exists() and destination.stat().st_size > 0
