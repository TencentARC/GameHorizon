from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .local_cache import exclusive_path_lock
from .schema import atomic_write_json


def _fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _signature(value: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _probe_video_frame_pts(video_path: Path) -> List[Tuple[int, int]]:
    """Return presentation timestamps without decoding the full video.

    Video packet PTS values are sufficient for exact ``select=eq(pts, ...)``
    extraction.  Reading packets is orders of magnitude faster than
    ``-show_frames`` for long 4K HEVC recordings, which forces a full decode.
    B-frame packet order may differ from presentation order, so values are
    sorted and de-duplicated by PTS before nearest-neighbour lookup.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts,pts_time",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            "ffprobe could not enumerate video frame timestamps: "
            + completed.stderr.strip()[:500]
        )
    by_stream_pts: Dict[int, int] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            stream_pts = int(fields[0])
            pts_ns = round(float(fields[1]) * 1_000_000_000)
        except (TypeError, ValueError):
            continue
        by_stream_pts.setdefault(stream_pts, pts_ns)
    result = sorted(by_stream_pts.items(), key=lambda item: (item[1], item[0]))
    if not result:
        raise RuntimeError("ffprobe returned no timestamped video frames")
    return result


def _cached_video_frame_pts(
    video_path: Path,
    cache_dir: Path | None,
) -> tuple[List[Tuple[int, int]], Dict[str, Any]]:
    if cache_dir is None:
        result = _probe_video_frame_pts(video_path)
        return result, {
            "enabled": False,
            "status": "disabled",
            "frame_count": len(result),
        }

    cache_dir = cache_dir.expanduser().resolve()
    fingerprint = _fingerprint(video_path)
    key = _signature({"schema_version": 1, "video": fingerprint})
    cache_path = cache_dir / key[:2] / f"{key}.pts.json.gz"
    lock_path = cache_path.with_name(f"{cache_path.name}.lock")

    def read_cache() -> List[Tuple[int, int]] | None:
        if not cache_path.is_file():
            return None
        try:
            with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("video") != fingerprint:
                return None
            rows = payload.get("frames")
            if not isinstance(rows, list) or not rows:
                return None
            return [(int(row[0]), int(row[1])) for row in rows]
        except (OSError, ValueError, TypeError, json.JSONDecodeError, IndexError):
            return None

    cached = read_cache()
    if cached is not None:
        return cached, {
            "enabled": True,
            "status": "hit",
            "path": str(cache_path),
            "frame_count": len(cached),
        }

    with exclusive_path_lock(lock_path, ready=lambda: read_cache() is not None) as acquired:
        cached = read_cache()
        if not acquired or cached is not None:
            if cached is None:
                raise RuntimeError(f"PTS cache lock released without cache: {cache_path}")
            return cached, {
                "enabled": True,
                "status": "hit_after_wait",
                "path": str(cache_path),
                "frame_count": len(cached),
            }
        frames = _probe_video_frame_pts(video_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        try:
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as handle:
                json.dump(
                    {
                        "schema_version": 1,
                        "video": fingerprint,
                        "frames": frames,
                    },
                    handle,
                    separators=(",", ":"),
                )
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
    return frames, {
        "enabled": True,
        "status": "miss_built",
        "path": str(cache_path),
        "frame_count": len(frames),
    }


def _nearest_position(values: Sequence[int], target: int) -> int:
    right = bisect.bisect_left(values, target)
    if right <= 0:
        return 0
    if right >= len(values):
        return len(values) - 1
    left = right - 1
    return left if abs(values[left] - target) <= abs(values[right] - target) else right


def extract_frames_by_pts_ffmpeg(
    *,
    video_path: Path,
    annotations_path: Path,
    output_dir: Path,
    max_width: int = 1280,
    jpeg_quality: int = 90,
    tolerance_ms: float = 50.0,
    force: bool = False,
    chunk_size: int = 100,
    chunk_jobs: int = 1,
    ffmpeg_threads: int = 0,
    pts_cache_dir: Path | None = None,
) -> Dict[str, Any]:
    """Exact-PTS frame extraction using ffprobe timestamps and FFmpeg select.

    This is the dependency-free fallback for environments without PyAV.  It
    enumerates actual stream timestamps, maps every annotation target to its
    nearest source frame, then decodes selected PTS values in bounded command
    chunks.  Multiple annotation targets may safely reuse the same decoded
    frame near the beginning of a recording.
    """

    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_jobs <= 0:
        raise ValueError("chunk_jobs must be positive")
    if ffmpeg_threads < 0:
        raise ValueError("ffmpeg_threads cannot be negative")
    document = json.loads(annotations_path.read_text(encoding="utf-8"))
    frames = document.get("frames", []) if isinstance(document, dict) else document
    targets = [
        (int(frame.get("frame_index", index)), int(frame["video_pts_ns"]))
        for index, frame in enumerate(frames)
    ]
    if not targets:
        raise ValueError("annotations contain no video_pts_ns targets")
    signature_inputs = {
        # v3 retries a failed multi-frame chunk one PTS at a time when the
        # mjpeg encoder rejects non-monotonic timestamps on damaged captures.
        "backend": "ffmpeg_select_by_pts_v3",
        "video": _fingerprint(video_path),
        "annotations": _fingerprint(annotations_path),
        "target_count": len(targets),
        "first_pts_ns": targets[0][1],
        "last_pts_ns": targets[-1][1],
        "max_width": int(max_width),
        "jpeg_quality": int(jpeg_quality),
        "tolerance_ms": float(tolerance_ms),
        "chunk_size": int(chunk_size),
    }
    signature = _signature(signature_inputs)
    manifest_path = output_dir / "frames_manifest.json"
    if not force and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior.get("signature") == signature
            and int(prior.get("frames_written", 0)) == len(targets)
            and all((output_dir / f"{index:06d}.jpg").is_file() for index, _ in targets)
        ):
            return prior

    video_frames, pts_cache_audit = _cached_video_frame_pts(
        video_path,
        pts_cache_dir,
    )
    video_pts_ns = [value[1] for value in video_frames]
    tolerance_ns = round(float(tolerance_ms) * 1_000_000)
    selections: List[Dict[str, Any]] = []
    targets_by_stream_pts: Dict[int, List[int]] = defaultdict(list)
    selected_details: Dict[int, Tuple[int, int]] = {}
    leading_clamped = trailing_clamped = discontinuity_clamped = 0
    for frame_index, target_pts_ns in targets:
        position = _nearest_position(video_pts_ns, target_pts_ns)
        stream_pts, decoded_pts_ns = video_frames[position]
        error_ns = decoded_pts_ns - target_pts_ns
        if abs(error_ns) > tolerance_ns:
            if target_pts_ns < video_pts_ns[0]:
                leading_clamped += 1
            elif target_pts_ns > video_pts_ns[-1]:
                trailing_clamped += 1
            else:
                right = bisect.bisect_left(video_pts_ns, target_pts_ns)
                left = max(0, right - 1)
                right = min(len(video_pts_ns) - 1, right)
                source_gap_ns = video_pts_ns[right] - video_pts_ns[left]
                if source_gap_ns > 2 * tolerance_ns:
                    # The recording itself has no video packet near this
                    # annotation target because of a capture discontinuity.
                    # Reuse the nearest real boundary frame and retain the
                    # exact error in the audit.
                    discontinuity_clamped += 1
                else:
                    raise RuntimeError(
                        f"frame {frame_index}: nearest FFmpeg PTS differs by "
                        f"{error_ns / 1e6:.3f} ms"
                    )
        targets_by_stream_pts[stream_pts].append(frame_index)
        selected_details[stream_pts] = (decoded_pts_ns, error_ns)
        selections.append({
            "frame_index": frame_index,
            "requested_pts_ns": target_pts_ns,
            "decoded_pts_ns": decoded_pts_ns,
            "error_ns": error_ns,
            "file": f"{frame_index:06d}.jpg",
        })

    selected_pts = sorted(targets_by_stream_pts, key=lambda value: selected_details[value][0])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.ffmpeg_pts.", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        chunk_specs = [
            (
                chunk_number,
                selected_pts[start : start + chunk_size],
            )
            for chunk_number, start in enumerate(
                range(0, len(selected_pts), chunk_size)
            )
        ]

        def extract_pts_batch(
            chunk_dir: Path,
            chunk: Sequence[int],
        ) -> tuple[subprocess.CompletedProcess[bytes], List[Path]]:
            first_ns = selected_details[chunk[0]][0]
            last_ns = selected_details[chunk[-1]][0]
            seek_s = max(0.0, first_ns / 1_000_000_000 - 1.0)
            stop_s = last_ns / 1_000_000_000 + 1.0
            expression = "+".join(f"eq(pts\\,{value})" for value in chunk)
            filters = [f"select={expression}"]
            if max_width > 0:
                filters.append(
                    f"scale={max_width}:-2:force_original_aspect_ratio=decrease"
                )
            pattern = chunk_dir / "%06d.jpg"
            qscale = max(2, min(31, round((100 - jpeg_quality) * 0.29) + 2))
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
            ]
            if ffmpeg_threads > 0:
                command.extend(
                    [
                        "-filter_threads",
                        str(ffmpeg_threads),
                        "-threads",
                        str(ffmpeg_threads),
                    ]
                )
            command.extend(
                [
                    "-copyts",
                    "-ss",
                    f"{seek_s:.9f}",
                    "-i",
                    str(video_path),
                    "-to",
                    f"{stop_s:.9f}",
                    "-an",
                    "-vf",
                    ",".join(filters),
                    "-fps_mode",
                    "passthrough",
                    "-frames:v",
                    str(len(chunk)),
                    "-start_number",
                    "0",
                    "-q:v",
                    str(qscale),
                ]
            )
            if ffmpeg_threads > 0:
                command.extend(["-threads", str(ffmpeg_threads)])
            command.append(str(pattern))
            completed = subprocess.run(command, capture_output=True, check=False)
            extracted = [
                chunk_dir / f"{index:06d}.jpg" for index in range(len(chunk))
            ]
            return completed, extracted

        def extract_chunk(spec: tuple[int, List[int]]) -> tuple[int, List[int], Path]:
            chunk_number, chunk = spec
            chunk_dir = staging / f"chunk_{chunk_number:04d}"
            chunk_dir.mkdir(parents=True)
            completed, extracted = extract_pts_batch(chunk_dir, chunk)
            ok = (not completed.returncode) and all(
                path.is_file() and path.stat().st_size > 0 for path in extracted
            )
            if ok:
                return chunk_number, chunk, chunk_dir
            detail = completed.stderr.decode("utf-8", errors="replace")[:500]
            # Some captures trip the mjpeg encoder with "Invalid pts <= last"
            # inside a multi-frame select. Fall back to one seek per PTS so a
            # single bad packet cannot poison the chunk.
            if len(chunk) > 1 and "Invalid pts" in detail:
                for path in chunk_dir.glob("*.jpg"):
                    path.unlink(missing_ok=True)
                for index, stream_pts in enumerate(chunk):
                    single_dir = chunk_dir / f"single_{index:06d}"
                    single_dir.mkdir(parents=True, exist_ok=True)
                    single_completed, single_extracted = extract_pts_batch(
                        single_dir, [stream_pts]
                    )
                    if (
                        single_completed.returncode
                        or not single_extracted[0].is_file()
                        or single_extracted[0].stat().st_size <= 0
                    ):
                        raise RuntimeError(
                            f"FFmpeg PTS chunk {chunk_number} frame {index} "
                            f"(pts={stream_pts}) failed after Invalid-pts "
                            f"retry: "
                            + single_completed.stderr.decode(
                                "utf-8", errors="replace"
                            )[:500]
                        )
                    single_extracted[0].replace(chunk_dir / f"{index:06d}.jpg")
                    shutil.rmtree(single_dir, ignore_errors=True)
                return chunk_number, chunk, chunk_dir
            raise RuntimeError(
                f"FFmpeg PTS chunk {chunk_number} extracted fewer than "
                f"{len(chunk)} frames: "
                + detail
            )

        worker_count = min(max(1, int(chunk_jobs)), len(chunk_specs))
        if worker_count == 1:
            completed_chunks = [extract_chunk(spec) for spec in chunk_specs]
        else:
            completed_chunks = []
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="ffmpeg-pts",
            ) as executor:
                futures = [executor.submit(extract_chunk, spec) for spec in chunk_specs]
                for future in as_completed(futures):
                    completed_chunks.append(future.result())

        for chunk_number, chunk, chunk_dir in sorted(completed_chunks):
            extracted = [
                chunk_dir / f"{index:06d}.jpg" for index in range(len(chunk))
            ]
            for source, stream_pts in zip(extracted, chunk):
                frame_indices = targets_by_stream_pts[stream_pts]
                for frame_index in frame_indices[:-1]:
                    shutil.copy2(source, staging / f"{frame_index:06d}.jpg")
                source.replace(staging / f"{frame_indices[-1]:06d}.jpg")
            shutil.rmtree(chunk_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        expected_names = {f"{index:06d}.jpg" for index, _ in targets}
        for suffix in ("*.jpg", "*.jpeg", "*.png"):
            for existing in output_dir.glob(suffix):
                if existing.name not in expected_names or force:
                    existing.unlink()
        for frame_index, _ in targets:
            source = staging / f"{frame_index:06d}.jpg"
            if not source.is_file():
                raise RuntimeError(f"FFmpeg PTS staging frame missing: {source.name}")
            source.replace(output_dir / source.name)

    errors = [abs(int(item["error_ns"])) for item in selections]
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "signature": signature,
        "signature_inputs": signature_inputs,
        "source_video": str(video_path.resolve()),
        "source_annotations": str(annotations_path.resolve()),
        "extraction_backend": "ffmpeg_select_by_pts",
        "frames_written": len(selections),
        "unique_decoded_frames": len(selected_pts),
        "video_frames_probed": len(video_frames),
        "video_pts_cache": pts_cache_audit,
        "ffmpeg_chunk_count": len(chunk_specs),
        "ffmpeg_chunk_jobs": min(max(1, int(chunk_jobs)), len(chunk_specs)),
        "ffmpeg_threads": int(ffmpeg_threads),
        "leading_clamped_count": leading_clamped,
        "trailing_clamped_count": trailing_clamped,
        "discontinuity_clamped_count": discontinuity_clamped,
        "max_abs_error_ns": max(errors),
        "mean_abs_error_ns": sum(errors) / len(errors),
        "frames": selections,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
