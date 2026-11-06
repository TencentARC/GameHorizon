"""Stage 1b: Gemini-vision refinement of action-segmented GTA5 clips.

Reads a segments JSON produced by ``1a_GTA5_scripts_segment_from_actions.py``,
selects segments where ``vision_needed=True``, samples frames from the source
video via ``ffmpeg``, and asks Gemini to choose the best GTA5 taxonomy leaf.

The result is written back into the segments JSON with new fields:

* ``refined_label`` - the chosen leaf label
* ``refined_tier1`` / ``refined_tier2`` - re-derived from taxonomy
* ``refined_confidence`` - 0..1 from the model
* ``refined_reason`` - short explanation
* ``refined_via`` - which model produced the refinement

Unknown segments can be subdivided into multiple refined child segments. This is
useful for GTA5 scenes where the action-only pass sees long stretches of mouse
movement but no keyboard events.

CLI::

    python 1b_GTA5_refine_with_gemini.py scene_segments.json \
        --video recording.mkv --out scene_segments_refined.json \
        --frames-per-segment 5 --model gemini-2.5-flash --concurrency 8
"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from GTA5_taxonomy import (
    LEAF_BY_LABEL,
    LEAVES,
    format_taxonomy_for_prompt,
    normalize_label,
    subtype_choices,
    vision_needed as taxonomy_vision_needed,
)


NS_PER_S = 1_000_000_000

# Annotation timestamps are absolute epoch nanoseconds, but ffmpeg ``-ss`` needs
# video-relative seconds. This offset (seconds of the video's frame 0 in epoch
# time) is detected in ``main`` and subtracted before every seek.
VIDEO_START_S = 0.0
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_S = 300.0
DEFAULT_GEMINI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
PROMPT_VERSION = 1


# ---------------------------------------------------------------------------
# Keybind reference
# ---------------------------------------------------------------------------

_KEYBINDS_JSON_PATH = Path(__file__).resolve().parent / "GTA5_keybinds.json"


def _load_keybinds() -> Dict[str, str]:
    try:
        raw = json.loads(_KEYBINDS_JSON_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, str)}


KEYBINDS: Dict[str, str] = _load_keybinds()


def format_keybinds_for_prompt() -> str:
    if not KEYBINDS:
        return ""
    lines = ["GTA5 key bindings (raw event name -> in-game action):"]
    for key, action in KEYBINDS.items():
        lines.append(f"  - {key}: {action}")
    return "\n".join(lines)


KEYBINDS_REFERENCE_TEXT = format_keybinds_for_prompt()


# ---------------------------------------------------------------------------
# Video / Gemini helpers
# ---------------------------------------------------------------------------


class FrameExtractionError(RuntimeError):
    pass


def probe_video_start_ns(video_path: Path) -> Optional[int]:
    """Best-effort container ``creation_time`` as epoch nanoseconds.

    Used to align absolute annotation timestamps to video-relative seconds so
    ``ffmpeg -ss`` seeks land inside the clip instead of billions of seconds
    past its end.
    """
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format_tags=creation_time",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.decode(errors="replace").strip()
    if not value:
        return None
    try:
        from datetime import datetime

        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(parsed.timestamp() * NS_PER_S)


def extract_frame_jpeg(
    video_path: Path,
    t_seconds: float,
    max_width: int = 1280,
    quality: int = 6,
) -> bytes:
    """Return one JPEG frame at absolute video timestamp ``t_seconds``."""
    if not video_path.exists():
        raise FrameExtractionError(f"video not found: {video_path}")
    seek_s = max(0.0, t_seconds - VIDEO_START_S)
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-ss", f"{seek_s:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"scale='min({max_width},iw)':-2",
        "-q:v", str(quality),
        "-pix_fmt", "yuvj420p",
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
        raise FrameExtractionError("ffmpeg not found on PATH") from e
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="replace")[:400]
        raise FrameExtractionError(f"ffmpeg failed at t={t_seconds:.3f}: {msg}") from e
    except subprocess.TimeoutExpired as e:
        raise FrameExtractionError(f"ffmpeg timed out at t={t_seconds:.3f}") from e
    if not result.stdout:
        raise FrameExtractionError(f"ffmpeg produced no frame at t={t_seconds:.3f}")
    return result.stdout


def sample_segment_timestamps(
    start_ns: int,
    end_ns: int,
    k: int,
    inward_padding_s: float = 0.05,
) -> List[float]:
    """Return K absolute seconds inside ``[start_ns, end_ns]``."""
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


def make_data_url(jpeg_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_image_content(
    text_prompt: str,
    jpeg_blobs: List[bytes],
    detail: str = "low",
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    for jpeg in jpeg_blobs:
        content.append({
            "type": "image_url",
            "image_url": {"url": make_data_url(jpeg), "detail": detail},
        })
    return content


def gemini_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    import openai

    api_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "No API key set; export OPENAI_API_KEY / GEMINI_API_KEY, "
            "or pass --api-key."
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


def extract_json_object(raw: str) -> Dict[str, Any]:
    if raw is None:
        raise ValueError("empty response")
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip(":\n ")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no JSON object in response: {raw[:200]!r}")


class JsonCache:
    """Small JSON-on-disk cache keyed by a deterministic hash."""

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
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


T = TypeVar("T")


def with_retry(
    max_attempts: int = 5,
    initial_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    retry_on: Tuple[type, ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
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
# Candidate sets and prompts
# ---------------------------------------------------------------------------


def all_non_idle_atomic_labels() -> List[str]:
    seen: set = set()
    out: List[str] = []
    for leaf in LEAVES:
        label = leaf.label
        if label.startswith("idle"):
            continue
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    if "unknown" not in seen:
        out.append("unknown")
    return out


UNKNOWN_FULL_CANDIDATES: List[str] = all_non_idle_atomic_labels()


def build_system_prompt() -> str:
    keybind_block = f"{KEYBINDS_REFERENCE_TEXT}\n\n" if KEYBINDS_REFERENCE_TEXT else ""
    return (
        "You are a fine-grained event classifier for short Grand Theft Auto V "
        "gameplay clips.\n"
        "You will see still frames sampled from a clip plus JSON metadata from "
        "a rule-based keyboard/mouse segmenter. Raw input names such as "
        "`KEY_W` and `RI_MOUSE_LEFT_BUTTON` should be interpreted using the "
        "GTA5 keybind reference below.\n"
        "Pick the SINGLE most accurate GTA5 taxonomy leaf label from the "
        "candidate list. If none of the candidates fit, return the given "
        "`parent_label`.\n\n"
        f"{keybind_block}"
        "GTA5 3-tier taxonomy (descriptions included):\n"
        f"{format_taxonomy_for_prompt(include_descriptions=True)}\n\n"
        "Important GTA5 visual context:\n"
        "- `idle_stationary` from the action pass can be visually wrong when a "
        "vehicle coasts without keyboard input; prefer vehicle labels if the "
        "frames show driving.\n"
        "- `aiming` means RMB/lock-on without clear firing; `firing_ads` means "
        "aiming while shots or muzzle flashes are visible.\n"
        "- Use `unknown` only when the clip cannot be classified reliably.\n\n"
        "Output STRICTLY one JSON object with keys:\n"
        '  "label": one of the candidates listed in the user message\n'
        '  "confidence": float in [0,1]\n'
        '  "reason": short clause (<= 20 words)\n'
        "Do not include any other text."
    )


def build_user_prompt(seg: Dict[str, Any], candidates: List[str]) -> str:
    parent = seg.get("label", "unknown")
    fields = {
        "parent_label": parent,
        "candidates": candidates,
        "duration_s": seg.get("duration_s"),
        "tier1": seg.get("tier1"),
        "tier2": seg.get("tier2"),
        "keys_pressed": seg.get("keys_pressed", []),
        "mouse_buttons_pressed": seg.get("mouse_buttons_pressed", []),
        "wheel_events": seg.get("wheel_events", 0),
        "mouse_motion_total": seg.get("mouse_motion_total", 0),
    }
    extra = ""
    if parent == "unknown":
        extra = (
            "\nDo NOT return any `idle_*` label for unknown clips. If the frames "
            "show no classifiable activity, return `unknown`."
        )
    return (
        "Pick the most appropriate label for this GTA5 clip from `candidates`. "
        "If none clearly fit, return `parent_label`."
        f"{extra}\n\n"
        f"Segment metadata:\n```json\n{json.dumps(fields, ensure_ascii=False, indent=2)}\n```\n"
        "Frames are attached below in temporal order."
    )


def build_subdivide_system_prompt() -> str:
    keybind_block = f"{KEYBINDS_REFERENCE_TEXT}\n\n" if KEYBINDS_REFERENCE_TEXT else ""
    return (
        "You are a fine-grained event SEGMENTER for short Grand Theft Auto V "
        "gameplay clips.\n"
        "You will see K still frames sampled in temporal order from a clip that "
        "the upstream action classifier labelled `unknown`.\n"
        "Group consecutive frames of the same event into `sub_events`. The clip "
        "may contain one event or multiple events in sequence.\n\n"
        "Rules:\n"
        "1. Sub-events must be contiguous and cover all frame indices 0..K-1.\n"
        "2. Labels must come from `candidates`; use `unknown` if none fit.\n"
        "3. Do NOT return any `idle_*` label.\n"
        "4. Only split when the visible event class actually changes.\n\n"
        f"{keybind_block}"
        "GTA5 taxonomy (descriptions included):\n"
        f"{format_taxonomy_for_prompt(include_descriptions=True)}\n\n"
        "Output STRICTLY one JSON object:\n"
        '{ "sub_events": ['
        '{ "start_frame_idx": int, "end_frame_idx": int, '
        '"label": string, "confidence": float, "reason": string }'
        "] }\n"
        "Do not include any other text."
    )


def build_subdivide_user_prompt(
    seg: Dict[str, Any],
    candidates: List[str],
    k: int,
) -> str:
    fields = {
        "K": k,
        "parent_label": "unknown",
        "candidates": candidates,
        "duration_s": seg.get("duration_s"),
        "tier1": seg.get("tier1"),
        "tier2": seg.get("tier2"),
        "keys_pressed": seg.get("keys_pressed", []),
        "mouse_buttons_pressed": seg.get("mouse_buttons_pressed", []),
        "wheel_events": seg.get("wheel_events", 0),
        "mouse_motion_total": seg.get("mouse_motion_total", 0),
    }
    return (
        f"This is an unknown GTA5 clip. Walk through the {k} attached frames "
        f"(indices 0..{k - 1}) and output sub_events covering all of them. "
        "Do NOT return idle labels.\n\n"
        "Segment metadata:\n"
        f"```json\n{json.dumps(fields, ensure_ascii=False, indent=2)}\n```\n"
        "Frames are attached below in temporal order."
    )


def _normalize_sub_events(raw: Any, candidates: List[str], k: int) -> List[Dict[str, Any]]:
    fallback = [{
        "start_frame_idx": 0,
        "end_frame_idx": k - 1,
        "label": "unknown",
        "confidence": None,
        "reason": "could not parse sub_events response",
    }]
    if not isinstance(raw, list) or not raw:
        return fallback

    candidates_set = set(candidates)
    parsed: List[Dict[str, Any]] = []
    for ev in raw:
        if not isinstance(ev, dict):
            continue
        try:
            start_i = int(ev.get("start_frame_idx"))
            end_i = int(ev.get("end_frame_idx"))
        except (TypeError, ValueError):
            continue
        start_i = max(0, min(start_i, k - 1))
        end_i = max(start_i, min(end_i, k - 1))
        raw_label = str(ev.get("label", "")).strip()
        label = raw_label if raw_label in candidates_set else normalize_label(raw_label)
        if label not in candidates_set:
            label = "unknown"
        try:
            confidence = float(ev["confidence"]) if "confidence" in ev else None
        except (TypeError, ValueError, KeyError):
            confidence = None
        parsed.append({
            "start_frame_idx": start_i,
            "end_frame_idx": end_i,
            "label": label,
            "confidence": confidence,
            "reason": str(ev.get("reason", ""))[:300],
        })
    if not parsed:
        return fallback

    parsed.sort(key=lambda e: (e["start_frame_idx"], e["end_frame_idx"]))
    fixed: List[Dict[str, Any]] = []
    cur_start = 0
    for i, ev in enumerate(parsed):
        start_i = max(cur_start, ev["start_frame_idx"])
        if i + 1 < len(parsed):
            next_start = max(parsed[i + 1]["start_frame_idx"], start_i + 1)
            end_i = min(ev["end_frame_idx"], next_start - 1)
        else:
            end_i = k - 1
        end_i = max(end_i, start_i)
        fixed.append({**ev, "start_frame_idx": start_i, "end_frame_idx": end_i})
        cur_start = end_i + 1
        if cur_start > k - 1:
            break
    if not fixed:
        return fallback
    fixed[0]["start_frame_idx"] = 0
    fixed[-1]["end_frame_idx"] = k - 1

    merged: List[Dict[str, Any]] = [fixed[0]]
    for ev in fixed[1:]:
        last = merged[-1]
        if ev["label"] == last["label"]:
            last["end_frame_idx"] = ev["end_frame_idx"]
            if (ev.get("confidence") or 0.0) > (last.get("confidence") or 0.0):
                last["confidence"] = ev.get("confidence")
                last["reason"] = ev.get("reason", "")
        else:
            merged.append(ev)
    return merged


# ---------------------------------------------------------------------------
# Gemini calls
# ---------------------------------------------------------------------------


@with_retry(max_attempts=4, initial_delay_s=1.5, max_delay_s=20.0)
def _call_gemini(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    jpeg_blobs: List[bytes],
    image_detail: str = "low",
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Dict[str, Any]:
    content = build_image_content(user_prompt, jpeg_blobs, detail=image_detail)
    content = [{"type": "text", "text": system_prompt}] + content
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        timeout=timeout_s,
    )
    raw = resp.choices[0].message.content or ""
    return extract_json_object(raw)


def refine_segment(
    seg: Dict[str, Any],
    *,
    video_path: Path,
    client,
    model: str,
    frames_per_segment: int,
    image_detail: str,
    cache: JsonCache,
    scene_name: str,
    system_prompt: str,
) -> Dict[str, Any]:
    parent = seg.get("label", "unknown")
    strict_candidates = False
    if parent == "unknown":
        candidates = list(UNKNOWN_FULL_CANDIDATES)
        strict_candidates = True
    else:
        candidates = list(subtype_choices(parent))

    cache_key = {
        "scene": scene_name,
        "start_frame": seg["start_frame"],
        "end_frame": seg["end_frame"],
        "parent": parent,
        "model": model,
        "k": frames_per_segment,
        "detail": image_detail,
        "stage": "gta5_refine",
        "prompt_v": PROMPT_VERSION,
    }
    cached = cache.get(cache_key)
    if cached:
        return cached

    timestamps = sample_segment_timestamps(
        seg["start_timestamp_ns"], seg["end_timestamp_ns"], frames_per_segment
    )
    jpeg_blobs = [extract_frame_jpeg(video_path, t, max_width=1024) for t in timestamps]
    parsed = _call_gemini(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=build_user_prompt(seg, candidates),
        jpeg_blobs=jpeg_blobs,
        image_detail=image_detail,
    )

    raw_label = str(parsed.get("label", "")).strip()
    label = raw_label if raw_label in candidates else normalize_label(raw_label)
    if strict_candidates and label not in candidates:
        label = parent
    elif label not in candidates and label not in LEAF_BY_LABEL:
        label = parent
    leaf = LEAF_BY_LABEL.get(label)
    try:
        confidence = float(parsed["confidence"]) if "confidence" in parsed else None
    except (TypeError, ValueError):
        confidence = None

    result = {
        "refined_label": label,
        "refined_tier1": leaf.tier1 if leaf else seg.get("tier1"),
        "refined_tier2": leaf.tier2 if leaf else seg.get("tier2"),
        "refined_confidence": confidence,
        "refined_reason": str(parsed.get("reason", ""))[:300],
        "refined_via": model,
        "refined_candidates": candidates,
        "refined_sampled_seconds": [round(t, 3) for t in timestamps],
    }
    cache.set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Unknown subdivision
# ---------------------------------------------------------------------------


def _interp_frame_index(seg: Dict[str, Any], t_ns: int) -> int:
    start_ns = int(seg["start_timestamp_ns"])
    end_ns = int(seg["end_timestamp_ns"])
    start_f = int(seg["start_frame"])
    end_f = int(seg["end_frame"])
    if end_ns <= start_ns or end_f <= start_f:
        return start_f
    t_ns = max(start_ns, min(t_ns, end_ns))
    frac = (t_ns - start_ns) / (end_ns - start_ns)
    return start_f + int(round(frac * (end_f - start_f)))


def _build_sub_segment(
    *,
    parent: Dict[str, Any],
    sub_event: Dict[str, Any],
    sample_timestamps_s: List[float],
    next_event_start_idx: Optional[int],
    is_first: bool,
    is_last: bool,
    sub_index: int,
    sub_total: int,
    candidates: List[str],
    model: str,
) -> Dict[str, Any]:
    start_i = int(sub_event["start_frame_idx"])
    end_i = int(sub_event["end_frame_idx"])
    label = str(sub_event.get("label") or "unknown")

    parent_start_ns = int(parent["start_timestamp_ns"])
    parent_end_ns = int(parent["end_timestamp_ns"])
    parent_dur_ns = max(parent_end_ns - parent_start_ns, 1)

    if is_first:
        start_ns = parent_start_ns
    else:
        start_ns = max(parent_start_ns, int(round(sample_timestamps_s[start_i] * NS_PER_S)))
    if is_last or next_event_start_idx is None:
        end_ns = parent_end_ns
    else:
        mid_s = (sample_timestamps_s[end_i] + sample_timestamps_s[next_event_start_idx]) / 2.0
        end_ns = max(start_ns + 1, min(parent_end_ns - 1, int(round(mid_s * NS_PER_S))))
    if end_ns <= start_ns:
        end_ns = start_ns + 1

    start_frame = _interp_frame_index(parent, start_ns)
    end_frame = max(start_frame, _interp_frame_index(parent, end_ns))
    duration_ns = end_ns - start_ns
    share = duration_ns / parent_dur_ns
    leaf = LEAF_BY_LABEL.get(label)

    new_seg = dict(parent)
    new_seg.update({
        "start_frame": start_frame,
        "end_frame": end_frame,
        "num_frames": max(end_frame - start_frame + 1, 1),
        "start_timestamp_ns": start_ns,
        "end_timestamp_ns": end_ns,
        "duration_s": round(duration_ns / NS_PER_S, 3),
        "label": "unknown",
        "tier1": parent.get("tier1"),
        "tier2": parent.get("tier2"),
        "refined_label": label,
        "refined_tier1": leaf.tier1 if leaf else parent.get("tier1"),
        "refined_tier2": leaf.tier2 if leaf else parent.get("tier2"),
        "refined_confidence": sub_event.get("confidence"),
        "refined_reason": str(sub_event.get("reason", ""))[:300],
        "refined_via": model,
        "refined_candidates": candidates,
        "refined_sampled_seconds": [
            round(sample_timestamps_s[i], 3) for i in range(start_i, end_i + 1)
        ],
        "subdivided_from_unknown": True,
        "sub_event_index": sub_index,
        "sub_event_total": sub_total,
        "sub_event_frame_range": [start_i, end_i],
        "mouse_motion_total": round(
            float(parent.get("mouse_motion_total", 0.0) or 0.0) * share, 1
        ),
        "wheel_events": int(round(float(parent.get("wheel_events", 0) or 0) * share)),
    })
    new_seg.pop("merged", None)
    new_seg.pop("sub_segments", None)
    return new_seg


def subdivide_unknown_segment(
    seg: Dict[str, Any],
    *,
    video_path: Path,
    client,
    model: str,
    frames_per_segment: int,
    image_detail: str,
    cache: JsonCache,
    scene_name: str,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    candidates = list(UNKNOWN_FULL_CANDIDATES)
    k = max(int(frames_per_segment), 2)
    cache_key = {
        "scene": scene_name,
        "start_frame": seg["start_frame"],
        "end_frame": seg["end_frame"],
        "parent": "unknown",
        "model": model,
        "k": k,
        "detail": image_detail,
        "stage": "gta5_subdivide_unknown",
        "prompt_v": PROMPT_VERSION,
    }
    cached = cache.get(cache_key)
    timestamps = sample_segment_timestamps(seg["start_timestamp_ns"], seg["end_timestamp_ns"], k)
    if cached is not None:
        sub_events = cached
    else:
        jpeg_blobs = [extract_frame_jpeg(video_path, t, max_width=1024) for t in timestamps]
        parsed = _call_gemini(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=build_subdivide_user_prompt(seg, candidates, k),
            jpeg_blobs=jpeg_blobs,
            image_detail=image_detail,
        )
        sub_events = _normalize_sub_events(parsed.get("sub_events"), candidates, k)
        cache.set(cache_key, sub_events)

    out: List[Dict[str, Any]] = []
    total = len(sub_events)
    for i, ev in enumerate(sub_events):
        next_start = sub_events[i + 1]["start_frame_idx"] if i + 1 < total else None
        out.append(_build_sub_segment(
            parent=seg,
            sub_event=ev,
            sample_timestamps_s=timestamps,
            next_event_start_idx=next_start,
            is_first=(i == 0),
            is_last=(i == total - 1),
            sub_index=i,
            sub_total=total,
            candidates=candidates,
            model=model,
        ))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def refine_segments(
    segments_payload: Dict[str, Any],
    *,
    video_path: Path,
    api_key: Optional[str],
    base_url: Optional[str],
    model: str,
    frames_per_segment: int,
    image_detail: str,
    concurrency: int,
    cache_dir: Optional[Path],
    limit: Optional[int],
    subdivide_unknown: bool = True,
    subdivide_frames_per_segment: Optional[int] = None,
) -> Dict[str, Any]:
    segments: List[Dict[str, Any]] = segments_payload["segments"]
    scene_name = segments_payload.get("scene_name") or video_path.stem

    targets = [i for i, seg in enumerate(segments) if seg.get("vision_needed")]
    if limit is not None:
        targets = targets[:limit]

    unknown_targets = sum(1 for i in targets if segments[i].get("label") == "unknown")
    other_targets = len(targets) - unknown_targets
    sub_k = (
        int(subdivide_frames_per_segment)
        if subdivide_frames_per_segment is not None
        else max(int(frames_per_segment), 10)
    )
    print(
        f"refining {len(targets)} / {len(segments)} GTA5 segments via {model} "
        f"(single-label: {other_targets}, "
        f"{'subdivide' if subdivide_unknown else 'single-label'} unknown: "
        f"{unknown_targets}, K={frames_per_segment}"
        f"{f' / sub_K={sub_k}' if subdivide_unknown else ''})"
    )

    client = gemini_client(api_key=api_key, base_url=base_url)
    cache = JsonCache(cache_dir)
    system_prompt = build_system_prompt()
    subdivide_system_prompt = build_subdivide_system_prompt()

    target_outputs: Dict[int, List[Dict[str, Any]]] = {}
    target_lock = threading.Lock()
    error_lock = threading.Lock()
    error_count = 0
    subdivided_count = 0
    sub_segments_emitted = 0

    def _worker(idx: int) -> None:
        nonlocal error_count, subdivided_count, sub_segments_emitted
        seg = segments[idx]
        try:
            if subdivide_unknown and seg.get("label") == "unknown":
                outs = subdivide_unknown_segment(
                    seg,
                    video_path=video_path,
                    client=client,
                    model=model,
                    frames_per_segment=sub_k,
                    image_detail=image_detail,
                    cache=cache,
                    scene_name=scene_name,
                    system_prompt=subdivide_system_prompt,
                )
                with target_lock:
                    target_outputs[idx] = outs
                with error_lock:
                    if len(outs) > 1:
                        subdivided_count += 1
                    sub_segments_emitted += len(outs)
            else:
                result = refine_segment(
                    seg,
                    video_path=video_path,
                    client=client,
                    model=model,
                    frames_per_segment=frames_per_segment,
                    image_detail=image_detail,
                    cache=cache,
                    scene_name=scene_name,
                    system_prompt=system_prompt,
                )
                new_seg = dict(seg)
                new_seg.update(result)
                with target_lock:
                    target_outputs[idx] = [new_seg]
        except Exception as e:  # noqa: BLE001
            with error_lock:
                error_count += 1
            sys.stderr.write(f"[seg {idx}] error: {e}\n")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_worker, idx): idx for idx in targets}
        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="refine", unit="seg")
        for future in iterator:
            future.result()

    out_segments: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        if i in target_outputs:
            out_segments.extend(target_outputs[i])
        else:
            out_segments.append(dict(seg))

    refined_count = sum(1 for seg in out_segments if "refined_label" in seg)
    label_changes: Dict[str, Dict[str, int]] = {}
    for seg in out_segments:
        refined_label = seg.get("refined_label")
        if refined_label and refined_label != seg.get("label"):
            label_changes.setdefault(seg["label"], {}).setdefault(refined_label, 0)
            label_changes[seg["label"]][refined_label] += 1

    payload = dict(segments_payload)
    payload["segments"] = out_segments
    payload["refine_meta"] = {
        "model": model,
        "frames_per_segment": frames_per_segment,
        "subdivide_frames_per_segment": sub_k if subdivide_unknown else None,
        "image_detail": image_detail,
        "vision_needed_total": sum(1 for seg in segments if seg.get("vision_needed")),
        "refined_total": refined_count,
        "errors": error_count,
        "label_changes": label_changes,
        "subdivide_unknown_enabled": subdivide_unknown,
        "unknown_inputs_total": unknown_targets,
        "unknown_inputs_subdivided": subdivided_count,
        "unknown_sub_segments_emitted": sub_segments_emitted,
        "segments_in": len(segments),
        "segments_out_after_refine": len(out_segments),
    }
    return payload


# ---------------------------------------------------------------------------
# Post-refine merge
# ---------------------------------------------------------------------------

_REFINED_FIELDS = (
    "refined_label",
    "refined_tier1",
    "refined_tier2",
    "refined_confidence",
    "refined_reason",
    "refined_via",
    "refined_candidates",
    "refined_sampled_seconds",
)

MOVEMENT_GROUP = frozenset({
    "walking",
    "sprinting",
    "stealth_movement",
    "sneaking",
    "jumping_climbing",
    "swimming",
    "swimming_fast",
})
COMBAT_GROUP = frozenset({"aiming", "firing_weapon", "firing_ads", "combat_in_cover"})
VEHICLE_GROUP = frozenset({
    "vehicle_driving",
    "vehicle_accelerating",
    "vehicle_braking_reversing",
    "vehicle_steering",
    "vehicle_cruising",
})
AIRCRAFT_GROUP = frozenset({
    "aircraft_flying",
    "aircraft_throttle_up",
    "aircraft_throttle_down",
    "aircraft_pitching",
    "aircraft_rolling",
})
MERGE_GROUPS: Tuple[Tuple[str, frozenset], ...] = (
    ("movement_group", MOVEMENT_GROUP),
    ("combat_group", COMBAT_GROUP),
    ("vehicle_group", VEHICLE_GROUP),
    ("aircraft_group", AIRCRAFT_GROUP),
)


def _effective_label(seg: Dict[str, Any]) -> str:
    refined_label = seg.get("refined_label")
    if isinstance(refined_label, str) and refined_label:
        return refined_label
    return seg.get("label", "") or ""


def _seg_components(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    existing = seg.get("sub_segments")
    if existing:
        return [dict(component) for component in existing]
    comp = {
        "label": seg.get("label"),
        "start_frame": seg["start_frame"],
        "end_frame": seg["end_frame"],
        "start_timestamp_ns": seg["start_timestamp_ns"],
        "end_timestamp_ns": seg["end_timestamp_ns"],
        "duration_s": seg.get("duration_s"),
    }
    if seg.get("refined_label"):
        comp["refined_label"] = seg["refined_label"]
    return [comp]


def _label_priority_for_group(kind: str, labels: List[Tuple[str, float]]) -> Optional[str]:
    present = {label for label, dur in labels if dur > 0}
    if kind == "combat_group":
        for label in ("firing_ads", "firing_weapon", "combat_in_cover", "aiming"):
            if label in present:
                return label
    return None


def _merge_label_for_group(kind: str, left: Dict[str, Any], right: Dict[str, Any]) -> str:
    labels = [
        (_effective_label(left), float(left.get("duration_s") or 0.0)),
        (_effective_label(right), float(right.get("duration_s") or 0.0)),
    ]
    priority = _label_priority_for_group(kind, labels)
    if priority:
        return priority
    left_label, left_dur = labels[0]
    right_label, right_dur = labels[1]
    return left_label if left_dur >= right_dur else right_label


def _merge_pair(left: Dict[str, Any], right: Dict[str, Any], label: str) -> Dict[str, Any]:
    leaf = LEAF_BY_LABEL.get(label)
    duration_ns = int(right["end_timestamp_ns"]) - int(left["start_timestamp_ns"])
    keys = sorted(set(left.get("keys_pressed", []) or []) | set(right.get("keys_pressed", []) or []))
    mouse = sorted(
        set(left.get("mouse_buttons_pressed", []) or [])
        | set(right.get("mouse_buttons_pressed", []) or [])
    )

    ref_src: Optional[Dict[str, Any]] = None
    if left.get("refined_label") == label:
        ref_src = left
    elif right.get("refined_label") == label:
        ref_src = right

    merged = {
        "start_frame": left["start_frame"],
        "end_frame": right["end_frame"],
        "num_frames": int(right["end_frame"]) - int(left["start_frame"]) + 1,
        "start_timestamp_ns": left["start_timestamp_ns"],
        "end_timestamp_ns": right["end_timestamp_ns"],
        "duration_s": round(duration_ns / NS_PER_S, 3),
        "label": label,
        "tier1": leaf.tier1 if leaf else left.get("tier1") or right.get("tier1"),
        "tier2": leaf.tier2 if leaf else left.get("tier2") or right.get("tier2"),
        "action_distinguishable": leaf.action_distinguishable if leaf else "n/a",
        "vision_needed": taxonomy_vision_needed(label),
        "vision_subtypes": list(leaf.vision_subtypes) if leaf else [],
        "keys_pressed": keys,
        "mouse_buttons_pressed": mouse,
        "mouse_motion_total": round(
            float(left.get("mouse_motion_total", 0.0) or 0.0)
            + float(right.get("mouse_motion_total", 0.0) or 0.0),
            1,
        ),
        "wheel_events": int(left.get("wheel_events", 0) or 0)
        + int(right.get("wheel_events", 0) or 0),
        "frame_file_start": left.get("frame_file_start"),
        "frame_file_end": right.get("frame_file_end"),
        "sub_segments": _seg_components(left) + _seg_components(right),
        "merged": True,
    }
    if (
        left.get("label") == "unknown"
        or right.get("label") == "unknown"
        or left.get("merged_from_unknown")
        or right.get("merged_from_unknown")
    ):
        merged["merged_from_unknown"] = True
    if ref_src is not None:
        for key in _REFINED_FIELDS:
            if key in ref_src:
                merged[key] = ref_src[key]
    return merged


def _resolve_merge(left: Dict[str, Any], right: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    left_label = _effective_label(left)
    right_label = _effective_label(right)
    if left_label == "unknown" and right_label == "unknown":
        return None
    if left_label and left_label == right_label:
        return {"label": left_label, "kind": "same_label"}
    for kind, group in MERGE_GROUPS:
        if left_label in group and right_label in group:
            return {"label": _merge_label_for_group(kind, left, right), "kind": kind}
    return None


def _recount_labels(segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, float]] = {}
    for seg in segments:
        label = seg.get("label", "unknown")
        bucket = counts.setdefault(label, {"segments": 0, "total_s": 0.0})
        bucket["segments"] += 1
        bucket["total_s"] += float(seg.get("duration_s") or 0.0)
    return {
        label: {"segments": int(values["segments"]), "total_s": round(values["total_s"], 2)}
        for label, values in counts.items()
    }


def merge_refined_unknowns(payload: Dict[str, Any]) -> Dict[str, Any]:
    segments = list(payload.get("segments", []))
    if not segments:
        return dict(payload)

    out: List[Dict[str, Any]] = []
    merges_performed = 0
    merges_by_label: Dict[str, int] = {}
    merges_by_kind: Dict[str, int] = {}
    for seg in segments:
        candidate = dict(seg)
        if not out:
            out.append(candidate)
            continue
        prev = out[-1]
        decision = _resolve_merge(prev, candidate)
        if decision is None:
            out.append(candidate)
            continue
        label = decision["label"]
        kind = decision["kind"]
        out[-1] = _merge_pair(prev, candidate, label)
        merges_performed += 1
        merges_by_label[label] = merges_by_label.get(label, 0) + 1
        merges_by_kind[kind] = merges_by_kind.get(kind, 0) + 1

    new_payload = dict(payload)
    new_payload["segments"] = out
    new_payload["label_counts"] = _recount_labels(out)
    refine_meta = dict(new_payload.get("refine_meta") or {})
    refine_meta["merge_unknown_meta"] = {
        "merges_performed": merges_performed,
        "segments_before": len(segments),
        "segments_after": len(out),
        "merges_by_label": dict(sorted(merges_by_label.items(), key=lambda kv: -kv[1])),
        "merges_by_kind": merges_by_kind,
        "merge_groups": {kind: sorted(group) for kind, group in MERGE_GROUPS},
    }
    new_payload["refine_meta"] = refine_meta
    return new_payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("segments_json", type=Path,
                        help="Input segments JSON from 1a_GTA5_scripts_segment_from_actions.py.")
    parser.add_argument("--video", type=Path, default=None,
                        help="Source MKV / MP4 video. Required unless --merge-only is set.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write the refined segments JSON.")
    parser.add_argument("--merge-unknowns", dest="merge_unknowns", action="store_true",
                        default=True,
                        help="Merge adjacent segments after refinement (default on).")
    parser.add_argument("--no-merge-unknowns", dest="merge_unknowns", action="store_false",
                        help="Disable the post-refine merge step.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip Gemini calls and only run post-refine merge.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Gemini model id (default {DEFAULT_MODEL}).")
    parser.add_argument("--frames-per-segment", type=int, default=5,
                        help="Frames sampled per non-unknown vision_needed segment.")
    parser.add_argument("--subdivide-unknown", dest="subdivide_unknown",
                        action="store_true", default=True,
                        help="Ask Gemini to subdivide unknown segments (default on).")
    parser.add_argument("--no-subdivide-unknown", dest="subdivide_unknown",
                        action="store_false",
                        help="Refine unknown segments as one label instead of subdividing.")
    parser.add_argument("--subdivide-frames-per-segment", type=int, default=10,
                        help="Frames sampled for unknown subdivision (default 10).")
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path(".gemini_cache_gta5"),
                        help="JSON cache directory. Use empty string to disable.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N vision_needed segments.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--video-start-ns", type=int, default=None,
                        help="Absolute epoch-ns of the video's first frame. If omitted, "
                             "it is auto-detected from the container creation_time so that "
                             "absolute annotation timestamps map to video-relative seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.segments_json.exists():
        sys.exit(f"segments JSON not found: {args.segments_json}")
    if not args.merge_only:
        if args.video is None:
            sys.exit("--video is required unless --merge-only is set")
        if not args.video.exists():
            sys.exit(f"video not found: {args.video}")

    with args.segments_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    cache_dir = args.cache_dir if str(args.cache_dir) else None
    if args.merge_only:
        print("--merge-only set: skipping Gemini refine, running merge step only")
        refined = payload
    else:
        global VIDEO_START_S
        segments = payload.get("segments") or []
        first_ts = next(
            (int(s["start_timestamp_ns"]) for s in segments
             if s.get("start_timestamp_ns") is not None),
            int(payload.get("interval_start_ns") or 0),
        )
        if args.video_start_ns is not None:
            origin_ns = int(args.video_start_ns)
        else:
            detected = probe_video_start_ns(args.video) or 0
            origin_ns = detected if detected and first_ts >= detected * 0.5 else 0
        VIDEO_START_S = origin_ns / NS_PER_S
        print(
            f"video_start_ns={origin_ns} "
            f"(seek offset {VIDEO_START_S:.3f}s; first_segment_ts={first_ts})"
        )
        refined = refine_segments(
            payload,
            video_path=args.video,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            frames_per_segment=args.frames_per_segment,
            image_detail=args.image_detail,
            concurrency=args.concurrency,
            cache_dir=cache_dir,
            limit=args.limit,
            subdivide_unknown=args.subdivide_unknown,
            subdivide_frames_per_segment=args.subdivide_frames_per_segment,
        )

    if args.merge_unknowns:
        refined = merge_refined_unknowns(refined)

    if args.out is not None:
        out_path = args.out
    elif args.merge_only:
        out_path = args.segments_json.with_name(args.segments_json.stem + "_merged.json")
    else:
        out_path = args.segments_json.with_name(args.segments_json.stem + "_refined.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")

    meta = refined.get("refine_meta") or {}
    if not args.merge_only:
        print(f"  refined: {meta.get('refined_total', 0)} / {meta.get('vision_needed_total', 0)}")
        print(f"  errors : {meta.get('errors', 0)}")
        if meta.get("subdivide_unknown_enabled"):
            print(
                "  subdivide unknown: "
                f"{meta.get('unknown_inputs_subdivided', 0)} / "
                f"{meta.get('unknown_inputs_total', 0)} inputs split into 2+ events, "
                f"{meta.get('unknown_sub_segments_emitted', 0)} sub-segments emitted "
                f"({meta.get('segments_in', 0)} -> "
                f"{meta.get('segments_out_after_refine', 0)} segments)"
            )
        if meta.get("label_changes"):
            print("  label changes (parent -> refined: count):")
            for parent, sub in sorted(meta["label_changes"].items()):
                for refined_label, count in sorted(sub.items(), key=lambda kv: -kv[1]):
                    print(f"    {parent:22s} -> {refined_label:24s} x {count}")

    merge_meta = meta.get("merge_unknown_meta")
    if merge_meta is not None:
        print(
            f"  merged unknowns: {merge_meta['merges_performed']} merges, "
            f"{merge_meta['segments_before']} -> {merge_meta['segments_after']} segments"
        )
        if merge_meta["merges_by_label"]:
            print("  merges by label (label: count):")
            for label, count in merge_meta["merges_by_label"].items():
                print(f"    {label:24s} x {count}")


if __name__ == "__main__":
    main()
