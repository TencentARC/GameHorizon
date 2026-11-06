"""Smoke tests: profiles resolve, bundled assets are intact, pipeline wires up."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from annotator.core.profiles import load_profile, supported_games
from annotator.core.schema import NS_PER_S

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "annotator"
ASSET_ROOT = PACKAGE_ROOT / "game_assets"
STAGES = ["segment", "refine", "level1_split", "level1", "merge12", "level2", "merge23", "level3"]


def test_profiles_load_and_reference_existing_assets() -> None:
    games = supported_games()
    assert games, "no game profiles were discovered"
    for game in games:
        profile = load_profile(game)
        for path in (
            profile.stage_1a_script,
            profile.stage_1b_script,
            profile.taxonomy_path,
            profile.keybinds_path,
        ):
            assert path.is_file(), f"{game}: missing asset {path}"
        assert profile.fallback_fps > 0, f"{game}: invalid fallback_fps"
        assert set(profile.prompt_focus) == {"level_1", "level_2", "level_3"}


def test_bundled_assets_match_integrity_manifest() -> None:
    manifest = json.loads((ASSET_ROOT / "public_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithm"] == "sha256"
    for entry in manifest["entries"]:
        path = ASSET_ROOT / entry["path"]
        assert path.is_file(), f"missing bundled asset: {entry['path']}"
        payload = path.read_bytes()
        assert len(payload) == entry["bytes"], f"size drift: {entry['path']}"
        digest = hashlib.sha256(payload).hexdigest()
        assert digest == entry["sha256"], f"digest drift: {entry['path']}"


def test_dry_run_builds_every_stage(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    annotations = tmp_path / "annotations.json"
    video.touch()
    annotations.write_text(json.dumps({"frames": []}), encoding="utf-8")

    for game in supported_games():
        completed = subprocess.run(
            [
                sys.executable, "-m", "annotator.cli.run_full_pipeline",
                "--game", game,
                "--video", str(video),
                "--annotations", str(annotations),
                "--workdir", str(tmp_path / game),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, f"{game} dry-run failed: {completed.stderr[-2000:]}"
        for stage in STAGES:
            assert f"=== {stage} ===" in completed.stdout, f"{game}: stage {stage} not planned"


def test_scenedetect_segmenter_is_opt_in(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    annotations = tmp_path / "annotations.json"
    video.touch()
    annotations.write_text(json.dumps({"frames": []}), encoding="utf-8")

    def plan(*extra: str) -> str:
        completed = subprocess.run(
            [
                sys.executable, "-m", "annotator.cli.run_full_pipeline",
                "--game", "minecraft",
                "--video", str(video),
                "--annotations", str(annotations),
                "--workdir", str(tmp_path / "work"),
                "--dry-run", *extra,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        return completed.stdout

    default = plan()
    assert "1a_Minecraft_scripts_segment_from_actions.py" in default
    assert "1a_scenedetect_segment" not in default

    opted_in = plan("--segmenter", "scenedetect")
    assert "1a_scenedetect_segment" in opted_in
    assert "1a_Minecraft_scripts_segment_from_actions.py" not in opted_in
    for stage in STAGES:
        assert f"=== {stage} ===" in opted_in, f"stage {stage} not planned"
    # Only Stage 1a changes; the refine/level stages must be untouched.
    tail = lambda text: text.split("=== refine ===", 1)[1]  # noqa: E731
    assert tail(default) == tail(opted_in)


def test_mouse_actions_survive_both_capture_schemas() -> None:
    """Motion and buttons must reach the prompt under either pointer schema."""

    from annotator.core.frames import compact_actions

    # Newer schema: semantic pointer layer plus a raw layer. Only the raw
    # layer carries dx/dy, so reading the semantic one loses all motion.
    split = {
        "frame_index": 7,
        "keyboard_events": [{"event_type": "press", "key_name": "KEY_W"}],
        "mouse_events": [
            {"event_type": "move", "x": 960, "y": 540},
            {"event_type": "click", "button": "right", "pressed": True, "x": 960, "y": 540},
        ],
        "raw_mouse_events": [
            {"dx": -1, "dy": -2, "button_flags": "0", "button_flag_names": []},
            {"dx": 0, "dy": 0, "button_flag_names": ["RI_MOUSE_RIGHT_BUTTON_DOWN"]},
        ],
    }
    compact = compact_actions(split)
    assert compact["mouse_movement"] == [[-1.0, -2.0]]
    assert compact["mouse_buttons"] == ["RI_MOUSE_RIGHT_BUTTON_DOWN"]
    assert compact["keyboard"] == ["press_KEY_W"]

    # Older schema: the raw stream *is* ``mouse_events`` and there is no
    # second array.
    legacy = {
        "frame_index": 7,
        "keyboard_events": [],
        "mouse_events": [
            {"dx": 3, "dy": -4, "button_flags": "0", "button_flag_names": []},
            {"dx": 2, "dy": 1, "button_flag_names": ["RI_MOUSE_LEFT_BUTTON_DOWN"]},
        ],
    }
    compact = compact_actions(legacy)
    assert compact["mouse_movement"] == [[3.0, -4.0], [2.0, 1.0]]
    assert compact["mouse_buttons"] == ["RI_MOUSE_LEFT_BUTTON_DOWN"]

    # A flick and its correction must not cancel out.
    swing = {
        "frame_index": 8,
        "keyboard_events": [],
        "raw_mouse_events": [{"dx": 40, "dy": 0}, {"dx": -40, "dy": 0}],
    }
    assert compact_actions(swing)["mouse_movement"] == [[40.0, 0.0], [-40.0, 0.0]]


def test_scenedetect_segments_tile_the_timeline() -> None:
    """Boundaries map onto frames and the spans stay a gap-free partition."""

    from annotator.core.scenedetect_segment import (
        _boundaries_from_cuts,
        _drop_short_spans,
        build_segments,
    )

    fps, video_start_ns = 20, 1_000_000_000
    step = NS_PER_S // fps
    frame_by_index = {
        index: {
            "frame_index": index,
            "timestamp_ns": video_start_ns + index * step,
            "next_frame_timestamp_ns": video_start_ns + (index + 1) * step,
            "keyboard_events": [{"key_name": "KEY_W", "event_type": "press"}],
            "mouse_events": [{"event_type": "scroll", "dx": 0, "dy": 1}],
            "raw_mouse_events": [{"dx": 3, "dy": 4}],
        }
        for index in range(200)  # 10 seconds
    }
    indices = sorted(frame_by_index)
    timestamps = [frame_by_index[index]["timestamp_ns"] for index in indices]

    # Cuts at 2s, 4s and 4.1s; the last is too close to survive the floor.
    starts = _boundaries_from_cuts(
        [2.0, 4.0, 4.1],
        video_start_ns=video_start_ns,
        timestamps=timestamps,
        indices=indices,
    )
    assert starts == [0, 40, 80, 82]

    kept = _drop_short_spans(starts, indices[-1], frame_by_index=frame_by_index, min_duration_s=1.0)
    assert kept == [0, 40, 80]

    segments = build_segments(kept, indices[-1], frame_by_index=frame_by_index, leaf=None)
    assert [segment["start_frame"] for segment in segments] == [0, 40, 80]
    assert [segment["end_frame"] for segment in segments] == [39, 79, 199]
    for left, right in zip(segments, segments[1:]):
        assert left["end_frame"] + 1 == right["start_frame"]
        assert left["end_timestamp_ns"] == right["start_timestamp_ns"]
    assert sum(segment["duration_s"] for segment in segments) == 10.0
    assert all(segment["label"] == "unknown" for segment in segments)
    assert all(segment["vision_needed"] for segment in segments)
    assert segments[0]["keys_pressed"] == ["KEY_W"]
    assert segments[0]["wheel_events"] == 40
    assert segments[0]["mouse_motion_total"] == 200.0  # 40 frames x hypot(3, 4)


if __name__ == "__main__":  # allow `python tests/test_smoke.py`
    import tempfile

    test_profiles_load_and_reference_existing_assets()
    test_bundled_assets_match_integrity_manifest()
    test_mouse_actions_survive_both_capture_schemas()
    test_scenedetect_segments_tile_the_timeline()
    with tempfile.TemporaryDirectory() as tmp:
        test_dry_run_builds_every_stage(Path(tmp))
        test_scenedetect_segmenter_is_opt_in(Path(tmp) / "optin")
    print("smoke tests passed")
