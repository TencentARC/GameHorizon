"""Shared video, cache, and OpenAI-compatible API helpers for Elden Ring."""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar


NS_PER_S = 1_000_000_000
DEFAULT_GEMINI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
PUBLIC_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class FrameExtractionError(RuntimeError):
    pass


def extract_frame_jpeg(
    video_path: Path,
    t_seconds: float,
    max_width: int = 1280,
    quality: int = 6,
) -> bytes:
    if not video_path.exists():
        raise FrameExtractionError(f"video not found: {video_path}")
    command = [
        "ffmpeg",
        "-loglevel", "error",
        "-ss", f"{max(0.0, t_seconds):.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"scale='min({max_width},iw)':-2",
        "-q:v", str(quality),
        "-f", "mjpeg",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise FrameExtractionError("ffmpeg is not available on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise FrameExtractionError(f"ffmpeg timed out at {t_seconds:.3f}s") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace")[:500]
        raise FrameExtractionError(f"ffmpeg failed at {t_seconds:.3f}s: {detail}") from exc
    if not completed.stdout:
        raise FrameExtractionError(f"ffmpeg returned no frame at {t_seconds:.3f}s")
    return completed.stdout


def probe_video_start_ns(video_path: Path) -> Optional[int]:
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format_tags=creation_time",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(parsed.timestamp() * NS_PER_S)


def resolve_video_start_ns(
    video_path: Path,
    segments: List[Dict[str, Any]],
    explicit: Optional[int] = None,
) -> int:
    if explicit is not None:
        return int(explicit)
    origin = int(probe_video_start_ns(video_path) or 0)
    if origin and segments:
        max_end = max(int(item.get("end_timestamp_ns") or 0) for item in segments)
        if max_end < origin * 0.5:
            return 0
    return origin


def sample_segment_timestamps(
    start_ns: int,
    end_ns: int,
    k: int,
    inward_padding_s: float = 0.05,
) -> List[float]:
    start_s = start_ns / NS_PER_S
    end_s = end_ns / NS_PER_S
    if end_s <= start_s:
        return [start_s]
    padding = min(inward_padding_s, (end_s - start_s) / 4.0)
    left = start_s + padding
    right = end_s - padding
    if k <= 1:
        return [(left + right) / 2.0]
    if k == 2:
        return [left, right]
    step = (right - left) / (k - 1)
    return [round(left + index * step, 3) for index in range(k)]


def make_data_url(blob: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(blob).decode('ascii')}"


def build_image_content(
    prompt: str,
    jpeg_blobs: List[bytes],
    detail: str = "low",
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for blob in jpeg_blobs:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": make_data_url(blob), "detail": detail},
            }
        )
    return content


def gemini_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    import openai

    openai_key = os.environ.get("OPENAI_API_KEY")
    public_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key is None:
        if openai_key:
            api_key = openai_key
            inferred_url = DEFAULT_GEMINI_BASE_URL
        else:
            api_key = public_key
            inferred_url = PUBLIC_GEMINI_BASE_URL
    else:
        inferred_url = DEFAULT_GEMINI_BASE_URL
    if not api_key:
        raise RuntimeError(
            "No API key set; export OPENAI_API_KEY for an OpenAI-compatible API or "
            "GEMINI_API_KEY for the public Gemini endpoint."
        )
    # _call_model already owns bounded retries via with_retry. Leaving SDK
    # retries enabled multiplies the worst-case timeout for every stalled
    # request (SDK attempts x outer attempts), which can leave a refine worker
    # waiting for close to an hour. Keep one retry owner so the explicit
    # request timeout remains an actual bound per outer attempt.
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url or inferred_url,
        max_retries=0,
    )


def extract_json_object(raw: str) -> Dict[str, Any]:
    if raw is None:
        raise ValueError("empty model response")
    value = raw.strip()
    if value.startswith("```"):
        value = value.strip("`").strip()
        if value.lower().startswith("json"):
            value = value[4:].lstrip(":\n ")
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(value[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"no JSON object in model response: {raw[:200]!r}")


class JsonCache:
    def __init__(self, root: Optional[Path]):
        self.root = Path(root) if root else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash(key: Dict[str, Any]) -> str:
        canonical = json.dumps(key, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

    def _path(self, key: Dict[str, Any]) -> Optional[Path]:
        if self.root is None:
            return None
        digest = self._hash(key)
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: Dict[str, Any]) -> Any:
        path = self._path(key)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, key: Dict[str, Any], value: Any) -> None:
        path = self._path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


T = TypeVar("T")


def with_retry(
    max_attempts: int = 5,
    initial_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    retry_on: Tuple[type, ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorate(function: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay_s
            for attempt in range(1, max_attempts + 1):
                try:
                    return function(*args, **kwargs)
                except retry_on:
                    if attempt >= max_attempts:
                        raise
                    time.sleep(delay + random.uniform(0, delay * 0.2))
                    delay = min(max_delay_s, delay * 2)
            raise AssertionError("unreachable")
        return wrapped
    return decorate
