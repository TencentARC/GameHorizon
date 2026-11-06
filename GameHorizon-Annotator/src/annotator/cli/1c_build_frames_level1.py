#!/usr/bin/env python3
"""Build the canonical Level-1 frame manifest for any supported game."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from annotator.core.frames import (
    FrameSource,
    frame_indices_for_segment,
    infer_fps,
    load_annotation_frames,
    partition_segment_timeline,
    uniform_sample,
)
from annotator.core.profiles import load_profile, supported_games
from annotator.core.schema import (
    SCHEMA_VERSION,
    atomic_write_json,
    effective_label,
    make_manifest,
    manifest_path,
    normalize_segments,
    relative_ref,
)
from annotator.core.timebase import normalize_annotation_timebase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, choices=supported_games())
    parser.add_argument("--segments-json", type=Path, required=True)
    parser.add_argument("--annotations-json", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--clips-dir", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument(
        "--source-video",
        type=Path,
        help="Original source path retained in manifests when --video is an NVMe cache copy.",
    )
    parser.add_argument("--video-start-ns", type=int)
    parser.add_argument("--scene-start-offset-ns", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument(
        "--sample-max-frames",
        type=int,
        default=0,
        help="Frames retained per Level-1 scene; 0 keeps the complete legacy frame span (default).",
    )
    parser.add_argument(
        "--sparse-batch-size",
        type=int,
        default=100,
        help="Maximum sampled frames decoded per FFmpeg PTS-select invocation (default: 100).",
    )
    parser.add_argument(
        "--sparse-jobs",
        type=int,
        default=1,
        help="Concurrent FFmpeg PTS chunks (default: 1).",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=0,
        help="Decoder/filter/encoder threads per FFmpeg chunk; 0 uses FFmpeg defaults.",
    )
    parser.add_argument("--sparse-cache-dir", type=Path)
    parser.add_argument("--pts-cache-dir", type=Path)
    parser.add_argument("--skip-label", action="append", default=[])
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--image-ext", choices=(".jpg", ".jpeg", ".png"), default=".jpg")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def build_level1_manifest(
    *,
    game_id: str,
    segments_json: Path,
    annotations_json: Path,
    workdir: Path,
    frames_dir: Path | None = None,
    clips_dir: Path | None = None,
    video: Path | None = None,
    source_video: Path | None = None,
    video_start_ns: int | None = None,
    scene_start_offset_ns: int | None = None,
    fps: float | None = None,
    sample_max_frames: int = 0,
    sparse_batch_size: int = 100,
    sparse_jobs: int = 1,
    ffmpeg_threads: int = 0,
    sparse_cache_dir: Path | None = None,
    pts_cache_dir: Path | None = None,
    skip_labels: List[str] | None = None,
    min_frames: int = 1,
    image_ext: str = ".jpg",
    jpeg_quality: int = 90,
    max_width: int = 1280,
    clean: bool = True,
) -> Dict[str, Any]:
    if sparse_batch_size <= 0:
        raise ValueError("sparse_batch_size must be positive")
    if sparse_jobs <= 0:
        raise ValueError("sparse_jobs must be positive")
    if ffmpeg_threads < 0:
        raise ValueError("ffmpeg_threads cannot be negative")
    profile = load_profile(game_id)
    for path, label in ((segments_json, "segments"), (annotations_json, "annotations")):
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
    for path, label in (
        (frames_dir, "frames"),
        (clips_dir, "clips"),
        (video, "video"),
        (source_video, "source video"),
    ):
        if path is not None and not path.exists():
            raise FileNotFoundError(f"{label} source not found: {path}")
    if frames_dir is None and clips_dir is None and video is None:
        raise ValueError("one of --frames-dir, --clips-dir, or --video is required")

    provenance_video = source_video or video
    annotation_meta, raw_annotation_frames, _ = load_annotation_frames(annotations_json)
    annotation_document: Any
    if annotation_meta:
        annotation_document = annotation_meta
    else:
        annotation_document = raw_annotation_frames
    normalized_annotations, current_timebase_audit = normalize_annotation_timebase(
        annotation_document,
        video_start_ns=video_start_ns,
        scene_start_offset_ns=scene_start_offset_ns,
        fps=fps,
        fallback_fps=profile.fallback_fps,
        video=video,
        allow_unresolved_video_positions=video is None,
    )
    annotation_frames = normalized_annotations["frames"]
    frame_by_index = {
        int(frame.get("frame_index", index)): frame
        for index, frame in enumerate(annotation_frames)
    }
    embedded_audit = annotation_meta.get("timebase_normalization")
    timebase_audit = (
        dict(embedded_audit)
        if isinstance(embedded_audit, dict)
        else current_timebase_audit.as_dict()
    )
    resolved_fps = float(fps or timebase_audit.get("fps") or infer_fps(
        annotation_frames, profile.fallback_fps
    ))
    import json

    raw_segments = json.loads(segments_json.read_text(encoding="utf-8"))
    if clips_dir is not None and isinstance(raw_segments, dict):
        scene_name = raw_segments.get("scene_name")
        if scene_name and (clips_dir / str(scene_name)).is_dir():
            clips_dir = clips_dir / str(scene_name)
    segments = partition_segment_timeline(
        normalize_segments(raw_segments, fps=resolved_fps),
        frame_by_index,
    )
    level_dir = workdir / "frames_level_1"
    if clean and level_dir.exists():
        shutil.rmtree(level_dir)
    level_dir.mkdir(parents=True, exist_ok=True)

    resolved_start_ns = int(timebase_audit.get("video_start_ns") or 0)
    source = FrameSource(
        frames_dir=frames_dir,
        clips_dir=clips_dir,
        video=video,
        frame_by_index=frame_by_index,
        video_start_ns=resolved_start_ns,
        fps=resolved_fps,
        image_ext=image_ext,
        jpeg_quality=jpeg_quality,
        max_width=max_width,
    )
    skipped = set(skip_labels or [])
    items: List[Dict[str, Any]] = []
    skipped_by_label = skipped_short = missing_source = batch_video_segments = 0

    # Sampling is a scene-wide operation: collect every retained frame first,
    # then decode them in bounded exact-PTS chunks. This avoids launching one
    # seeking FFmpeg process for every sampled frame while preserving the same
    # uniform_sample result and the canonical per-segment frame layout.
    if sparse_cache_dir is not None:
        scene_cache_key = hashlib.sha256(
            str(workdir.resolve()).encode("utf-8")
        ).hexdigest()[:24]
        sparse_staging_dir = (
            sparse_cache_dir.expanduser().resolve()
            / scene_cache_key[:2]
            / scene_cache_key
            / ".level1_sparse_frames"
        )
    else:
        sparse_staging_dir = workdir / ".level1_sparse_frames"
    sparse_requested: List[int] = []
    sparse_sampling_active = False
    if sample_max_frames > 0:
        for segment in segments:
            if effective_label(segment) in skipped:
                continue
            all_indices = frame_indices_for_segment(segment, frame_by_index)
            sampled_indices = uniform_sample(all_indices, sample_max_frames)
            sparse_requested.extend(int(value) for value in sampled_indices)
            sparse_sampling_active = sparse_sampling_active or sampled_indices != all_indices
    sparse_requested = sorted(set(sparse_requested))
    sparse_video_batch = bool(
        sparse_sampling_active
        and source.materialize_sparse_video_batch(
            frame_indices=sparse_requested,
            staging_dir=sparse_staging_dir,
            chunk_size=sparse_batch_size,
            chunk_jobs=sparse_jobs,
            ffmpeg_threads=ffmpeg_threads,
            pts_cache_dir=pts_cache_dir,
        )
    )
    sparse_reference_dir = (
        sparse_staging_dir
        if sparse_video_batch and sparse_cache_dir is not None
        else None
    )
    if sparse_reference_dir is not None:
        print(f"[1c] direct frame-cache references: {sparse_reference_dir}")

    if sparse_sampling_active and not sparse_video_batch and source.sparse_batch_error:
        print(
            "[1c] sparse video batch unavailable; falling back to per-frame extraction: "
            f"{source.sparse_batch_error}"
        )

    for segment in segments:
        label = effective_label(segment)
        if label in skipped:
            skipped_by_label += 1
            continue
        all_indices = frame_indices_for_segment(segment, frame_by_index)
        sampled_indices = uniform_sample(all_indices, sample_max_frames)
        scene = f"scene_{len(items):06d}"
        scene_dir = level_dir / scene
        written_refs: List[str] = []
        source_kinds: List[str] = []
        source_paths: List[str] = []
        batch_video = sparse_video_batch or (
            sampled_indices == all_indices
            and source.materialize_video_batch(
                frame_indices=sampled_indices,
                destination_dir=scene_dir,
            )
        )
        if batch_video:
            batch_video_segments += 1
        sparse_materialized: Dict[int, bool] = {}
        if sparse_video_batch and sparse_reference_dir is None and sampled_indices:
            copy_workers = min(32, max(1, sparse_jobs * 4), len(sampled_indices))

            def materialize_sample(frame_value: int) -> tuple[int, bool]:
                frame_index = int(frame_value)
                destination = scene_dir / f"{frame_index:06d}{image_ext}"
                return frame_index, source.materialize_sparse_frame(
                    frame_index=frame_index,
                    staging_dir=sparse_staging_dir,
                    destination=destination,
                )

            with ThreadPoolExecutor(max_workers=copy_workers) as copy_pool:
                for frame_index, ok in copy_pool.map(materialize_sample, sampled_indices):
                    sparse_materialized[frame_index] = ok
        for frame_index in sampled_indices:
            destination_root = sparse_reference_dir or scene_dir
            destination = destination_root / f"{int(frame_index):06d}{image_ext}"
            if sparse_video_batch:
                ok = (
                    destination.is_file() and destination.stat().st_size > 0
                    if sparse_reference_dir is not None
                    else sparse_materialized.get(int(frame_index), False)
                )
                kind = "video"
                source_path = str(video) if video is not None else None
                if not ok:
                    destination = scene_dir / f"{int(frame_index):06d}{image_ext}"
                    ok, kind, source_path = source.materialize(
                        segment=segment,
                        frame_index=int(frame_index),
                        destination=destination,
                    )
            elif batch_video:
                ok = destination.is_file() and destination.stat().st_size > 0
                kind = "video"
                source_path = str(video) if video is not None else None
            else:
                ok, kind, source_path = source.materialize(
                    segment=segment,
                    frame_index=int(frame_index),
                    destination=destination,
                )
            if ok:
                written_refs.append(relative_ref(destination, workdir))
                source_kinds.append(kind)
                if source_path:
                    source_paths.append(source_path)
        if len(written_refs) < min_frames:
            shutil.rmtree(scene_dir, ignore_errors=True)
            if written_refs:
                skipped_short += 1
            else:
                missing_source += 1
            continue

        item = {
            "scene": scene,
            "order": len(items),
            "segment_index": segment["segment_index"],
            "start_frame": all_indices[0] if all_indices else segment.get("start_frame"),
            "end_frame": all_indices[-1] if all_indices else segment.get("end_frame"),
            "stage1_start_frame": segment.get("start_frame"),
            "stage1_end_frame": segment.get("end_frame"),
            "start_timestamp_ns": segment.get("start_timestamp_ns"),
            "end_timestamp_ns": segment.get("end_timestamp_ns"),
            "duration_s": segment["duration_s"],
            "label": segment.get("label"),
            "refined_label": segment.get("refined_label"),
            "effective_label": label,
            "tier1": segment.get("refined_tier1") or segment.get("tier1"),
            "tier2": segment.get("refined_tier2") or segment.get("tier2"),
            "refined_confidence": segment.get("refined_confidence"),
            "refined_reason": segment.get("refined_reason"),
            "frame_refs": written_refs,
            "sampled_frame_indices": [int(value) for value in sampled_indices],
            "num_source_frames": len(all_indices),
            "source_scenes": [],
            "source_instructions": [],
            "source_kind": source_kinds[0] if len(set(source_kinds)) == 1 else "mixed",
            "source_paths": sorted(set(source_paths)),
            "source_start_frame": segment.get("source_start_frame"),
            "source_end_frame": segment.get("source_end_frame"),
            "source_start_timestamp_ns": segment.get("source_start_timestamp_ns"),
            "source_end_timestamp_ns": segment.get("source_end_timestamp_ns"),
            "source_duration_s": segment.get("source_duration_s"),
            "upstream_source_start_frame": segment.get("upstream_source_start_frame"),
            "upstream_source_end_frame": segment.get("upstream_source_end_frame"),
            "timeline_partitioned": bool(segment.get("timeline_partitioned")),
        }
        for key in (
            "start_video_pts_ns",
            "end_video_pts_ns",
            "refined_label_en",
            "refined_label_zh",
            "name_en",
            "name_zh",
            "instruction_en",
            "instruction_zh",
            "refined_reason_en",
            "refined_reason_zh",
            "context_label",
            "context_confidence",
            "context_at_start",
            "context_at_end",
            "control_mode",
            "control_mode_at_start",
            "control_mode_at_end",
            "nethack_visible",
            "context_transition_class",
            "world_context",
            "locomotion_context",
            "combat_context",
            "concurrent_actions",
            "refinement_fallback",
            "quality_flags",
            "model_evidence_flags",
            "model_evidence_frame_indices",
            "refined_sampled_frames",
            "refined_sampled_video_pts_ns",
            "refined_sampled_roles",
            "refined_image_detail",
            "raw_input_label",
            "refined_via",
            "refined_candidates",
            "model_action_label",
            "model_action_confidence",
            "model_action_supported",
            "model_response_repaired",
            "acceptance_threshold",
            "domain_rule_applied",
            "visually_rejected_labels",
        ):
            if segment.get(key) is not None:
                item[key] = segment[key]
        items.append(item)

    stage1_refinement: Dict[str, Any] = {}
    if isinstance(raw_segments, dict):
        for key in (
            "refine_prompt_version",
            "refine_taxonomy_version",
            "refine_validation_version",
            "refine_sampling_version",
            "refine_model",
            "refine_wire_api",
            "refine_crop",
            "refine_image_detail_policy",
            "refine_meta",
            "domain_rule_counts",
        ):
            if raw_segments.get(key) is not None:
                stage1_refinement[key] = raw_segments[key]

    payload = make_manifest(
        game_id=game_id,
        level=1,
        source={
            "segments_json": str(segments_json.resolve()),
            "annotations_json": str(annotations_json.resolve()),
            "source_annotations_json": normalized_annotations.get("source_annotations"),
            "video": str(provenance_video.resolve()) if provenance_video else None,
            "decoded_video": str(video.resolve()) if video else None,
            "frames_dir": str(frames_dir.resolve()) if frames_dir else None,
            "clips_dir": str(clips_dir.resolve()) if clips_dir else None,
            "video_start_ns": resolved_start_ns,
            "scene_start_offset_ns": timebase_audit.get("scene_start_offset_ns"),
            "fps": resolved_fps,
            "video_position_source": timebase_audit.get("conversion_source"),
            "video_position_fallback_source": timebase_audit.get("fallback_source"),
            "video_pts_filled": timebase_audit.get("filled_video_pts_count", 0),
            "video_pts_unresolved": timebase_audit.get("unresolved_video_pts_count", 0),
            "timebase": timebase_audit,
            "sample_max_frames": sample_max_frames,
            "skipped_by_label": skipped_by_label,
            "skipped_short": skipped_short,
            "missing_source": missing_source,
            "batch_video_segments": batch_video_segments,
            "sparse_video_batch": sparse_video_batch,
            "sparse_video_requested_frames": len(sparse_requested),
            "sparse_video_batch_size": sparse_batch_size,
            "sparse_video_decoded_frames": (
                int(source.sparse_batch_audit.get("frames_written", 0))
                if source.sparse_batch_audit
                else 0
            ),
            "sparse_video_ffmpeg_chunks": (
                (
                    int(source.sparse_batch_audit.get("frames_written", 0))
                    + sparse_batch_size
                    - 1
                )
                // sparse_batch_size
                if sparse_video_batch and source.sparse_batch_audit
                else 0
            ),
            "sparse_video_batch_manifest": (
                str((sparse_staging_dir / "frames_manifest.json").resolve())
                if sparse_video_batch
                else None
            ),
            "sparse_video_max_abs_error_ns": (
                source.sparse_batch_audit.get("max_abs_error_ns")
                if source.sparse_batch_audit
                else None
            ),
            "sparse_video_batch_error": source.sparse_batch_error,
            "stage1_refinement": stage1_refinement or None,
        },
        items=items,
    )
    atomic_write_json(manifest_path(workdir, 1), payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = build_level1_manifest(
        game_id=args.game,
        segments_json=args.segments_json,
        annotations_json=args.annotations_json,
        workdir=args.workdir,
        frames_dir=args.frames_dir,
        clips_dir=args.clips_dir,
        video=args.video,
        source_video=args.source_video,
        video_start_ns=args.video_start_ns,
        scene_start_offset_ns=args.scene_start_offset_ns,
        fps=args.fps,
        sample_max_frames=args.sample_max_frames,
        sparse_batch_size=args.sparse_batch_size,
        sparse_jobs=args.sparse_jobs,
        ffmpeg_threads=args.ffmpeg_threads,
        sparse_cache_dir=args.sparse_cache_dir,
        pts_cache_dir=args.pts_cache_dir,
        skip_labels=args.skip_label,
        min_frames=args.min_frames,
        image_ext=args.image_ext,
        jpeg_quality=args.jpeg_quality,
        max_width=args.max_width,
        clean=not args.no_clean,
    )
    print(
        f"[1c] schema={SCHEMA_VERSION} game={args.game} "
        f"scenes={len(payload['items'])} -> {manifest_path(args.workdir, 1)}"
    )


if __name__ == "__main__":
    main()
