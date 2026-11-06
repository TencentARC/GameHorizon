"""Shared helpers for the Gemini-driven Apex pipeline.

Used by:

* ``1b_Apex_Legends_refine_with_gemini.py`` - vision-based fine-grained event refinement.
* :mod:`Apex_Legends_caption_segments` - first-person instruction generation.

Provides:

* ``extract_frame_jpeg(mkv, t_seconds, max_w)`` - call ``ffmpeg`` to grab one
  frame at a given timestamp, returning JPEG bytes (no temp file).
* ``sample_segment_timestamps(seg, k)`` - K equispaced seconds inside a
  segment, biased slightly inward to avoid black transition frames.
* ``make_data_url(jpeg_bytes)`` - format JPEG bytes as a chat-completions
  ``image_url`` data URL.
* ``gemini_client(...)`` - factory for an ``openai.OpenAI`` configured against
  Gemini's OpenAI-compatible endpoint.
* ``JsonCache`` - simple key->JSON disk cache so reruns are free.
* ``with_retry`` - exponential-backoff decorator for 429 / 5xx style errors.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar


NS_PER_S = 1_000_000_000
# Use OPENAI_BASE_URL when set; otherwise let the OpenAI client use its default endpoint.
DEFAULT_GEMINI_BASE_URL = os.environ.get("OPENAI_BASE_URL")


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


class FrameExtractionError(RuntimeError):
    pass


def extract_frame_jpeg(
    mkv_path: Path,
    t_seconds: float,
    max_width: int = 1280,
    quality: int = 6,
) -> bytes:
    """Return one JPEG frame at ``t_seconds`` from the video at ``mkv_path``.

    Streams ``ffmpeg`` output through a pipe; no intermediate file is written.
    ``max_width`` constrains the longer side via ``scale='min(W,iw)':-2``;
    ``quality`` is the JPEG quality scale (2 = best, 31 = worst, mjpeg's
    convention; default 6 ~ visually lossless and ~50 KB at 1280p).
    """
    if not mkv_path.exists():
        raise FrameExtractionError(f"video not found: {mkv_path}")
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-ss", f"{t_seconds:.3f}",
        "-i", str(mkv_path),
        "-frames:v", "1",
        "-vf", f"scale='min({max_width},iw)':-2",
        "-q:v", str(quality),
        "-f", "mjpeg",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except FileNotFoundError as e:
        raise FrameExtractionError("ffmpeg not found on PATH; brew install ffmpeg") from e
    except subprocess.CalledProcessError as e:
        raise FrameExtractionError(
            f"ffmpeg failed at t={t_seconds:.3f}: {e.stderr.decode(errors='replace')[:400]}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise FrameExtractionError(f"ffmpeg timed out at t={t_seconds:.3f}") from e

    if not result.stdout:
        raise FrameExtractionError(f"ffmpeg produced no bytes at t={t_seconds:.3f}")
    return result.stdout


def make_data_url(jpeg_bytes: bytes, mime: str = "image/jpeg") -> str:
    """Encode raw image bytes as a data: URL suitable for OpenAI chat.completions."""
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Segment timestamp sampling
# ---------------------------------------------------------------------------


def sample_segment_timestamps(
    start_ns: int,
    end_ns: int,
    k: int,
    inward_padding_s: float = 0.05,
) -> List[float]:
    """Return K (clip-relative) seconds inside ``[start_ns, end_ns]``.

    The endpoints are pulled inward by ``inward_padding_s`` to avoid landing
    exactly on a transition frame (which is often half-mid in the interpolated
    timeline). For K=1, the single sample is the midpoint.
    """
    start_s = start_ns / NS_PER_S
    end_s = end_ns / NS_PER_S
    if end_s <= start_s:
        return [start_s]
    pad = min(inward_padding_s, (end_s - start_s) / 4.0)
    a = start_s + pad
    b = end_s - pad
    if k <= 1:
        return [(a + b) / 2.0]
    if k == 2:
        return [a, b]
    step = (b - a) / (k - 1)
    return [round(a + i * step, 3) for i in range(k)]


# ---------------------------------------------------------------------------
# OpenAI-compat client for Gemini
# ---------------------------------------------------------------------------


def gemini_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """Create an ``openai.OpenAI`` client targeting an OpenAI-compatible Gemini endpoint.

    The base URL comes from an explicit argument or ``OPENAI_BASE_URL``;
    without either, the OpenAI client uses its default endpoint. The API key is
    read from ``OPENAI_API_KEY``. ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``
    are also honoured for the public Google endpoint.
    """
    import openai

    api_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "No API key set; export OPENAI_API_KEY for an OpenAI-compatible API or "
            "GEMINI_API_KEY (public Gemini), or pass --api-key on the CLI."
        )
    # The callers already own bounded retries via with_retry. Leaving SDK
    # retries enabled multiplies the worst-case timeout for every stalled
    # request (SDK attempts x outer attempts), which can leave a refine worker
    # waiting for close to an hour. Keep one retry owner so the explicit
    # request timeout remains an actual bound per outer attempt.
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url or DEFAULT_GEMINI_BASE_URL,
        max_retries=0,
    )


# ---------------------------------------------------------------------------
# Robust JSON extraction (providers may wrap output in ```json fences)
# ---------------------------------------------------------------------------


def extract_json_object(raw: str) -> Dict[str, Any]:
    """Best-effort: turn an LLM response string into a Python dict.

    Handles:
    * direct JSON
    * markdown ```json ... ``` fences
    * leading / trailing chatter outside the JSON object

    Raises ``ValueError`` if no JSON object can be recovered.
    """
    if raw is None:
        raise ValueError("empty response")
    s = raw.strip()

    # Strip ```json ... ``` (or plain ```) fences if present.
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip(":\n ")

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Fallback: locate the first balanced { ... }.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"no JSON object in response: {raw[:200]!r}")


# ---------------------------------------------------------------------------
# Disk JSON cache
# ---------------------------------------------------------------------------


class JsonCache:
    """Trivial sharded-by-prefix JSON disk cache.

    Keyed by an arbitrary dict (e.g. ``{"scene":..., "start":..., "end":..., "model":...}``).
    The hash is computed deterministically from the JSON-serialised key.
    """

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
        h = self._hash(key)
        return self.root / h[:2] / f"{h}.json"

    def get(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        p = self._path(key)
        if p is None or not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, key: Dict[str, Any], value: Dict[str, Any]) -> None:
        p = self._path(key)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

T = TypeVar("T")


def with_retry(
    max_attempts: int = 5,
    initial_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    retry_on: Tuple[type, ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Exponential-backoff retry decorator.

    Any exception in ``retry_on`` triggers retry; the final attempt re-raises.
    Adds jitter so concurrent workers don't sync up.
    """

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay_s
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retry_on as e:
                    last_exc = e
                    if attempt >= max_attempts:
                        break
                    sleep_s = min(delay, max_delay_s) * (1.0 + random.random() * 0.3)
                    time.sleep(sleep_s)
                    delay *= 2.0
            assert last_exc is not None
            raise last_exc
        return wrapper

    return deco


# ---------------------------------------------------------------------------
# Convenience: build chat content with multiple images
# ---------------------------------------------------------------------------


def build_image_content(
    text_prompt: str,
    jpeg_blobs: List[bytes],
    detail: str = "low",
) -> List[Dict[str, Any]]:
    """Build the ``content`` array for a multimodal user message."""
    content: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    for jpeg in jpeg_blobs:
        content.append({
            "type": "image_url",
            "image_url": {"url": make_data_url(jpeg), "detail": detail},
        })
    return content
