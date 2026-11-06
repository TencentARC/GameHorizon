<div align="center">

# 🏗️ GameHorizon-Annotator

**Automated multi-horizon gameplay instruction annotation**

<a href="https://gamehorizon-suite.github.io"><img src="https://img.shields.io/badge/🌐_Project_Page-Online-6366F1?style=flat-square&labelColor=2B2F3A"></a>
<a href="https://arxiv.org/abs/2609.25001"><img src="https://img.shields.io/badge/📄_Paper-arXiv-EF4444?style=flat-square&labelColor=2B2F3A"></a>
<a href="../LICENSE"><img src="https://img.shields.io/badge/⚖️_License-Apache_2.0-8B5CF6?style=flat-square&labelColor=2B2F3A"></a>
<a href="../README.md"><img src="https://img.shields.io/badge/⬅️_GameHorizon_Suite-Repo-14B8A6?style=flat-square&labelColor=2B2F3A"></a>

</div>

---

Turns **gameplay video + temporally aligned keyboard-and-mouse annotations** into a
**three-level instruction pyramid**:

| Level | Horizon | What it describes |
| --- | --- | --- |
| **Level 1** | short (seconds) | the immediate **operation** |
| **Level 2** | medium (1–2 min) | the **goal** those operations serve |
| **Level 3** | long (4–8 min) | the **strategy / mission** |

## 🎮 Supported games

`apex_legends` · `cyberpunk_2077` · `eldenring` · `eldenring_nightreign` · `gta5` · `minecraft`

Pass one of these to `--game`. Each is declared by a JSON profile in
`src/annotator/game_profiles/`, so adding a game means adding a profile plus its
Stage-1 assets (taxonomy, keybinds, segmentation script) — no core changes.

## 📦 Requirements

- Python **3.10+**
- **FFmpeg** and **ffprobe** on `PATH`
- An **OpenAI-compatible multimodal API** (chat completions with image input)

## 🔧 Install

```bash
cd GameHorizon-Annotator
uv sync             # or: pip install -e .
```

Optional, for the `scenedetect` segmentation backend:

```bash
pip install 'GameHorizon-Annotator[scenedetect]'
```

## 🔑 API configuration

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-provider/v1"
```

`OPENAI_API_KEY` falls back to `GEMINI_API_KEY`, then `GOOGLE_API_KEY`. You can also
pass `--api-key` / `--base-url` on the command line. See `.env.example`.

> **⚠️ Match the model names to your provider.** The defaults are Gemini names
> (`gemini-2.5-flash` for refinement, `gemini-3.5-flash` for instructions) served
> through an OpenAI-compatible gateway. If your endpoint does not expose those
> names, override both:
>
> ```bash
> --refine-model gpt-4o --instruction-model gpt-4o
> ```

## 🚀 Quick start

```bash
python -m annotator.cli.run_full_pipeline \
  --game minecraft \
  --video ./data/gameplay.mp4 \
  --annotations ./data/annotations.json \
  --workdir ./outputs/minecraft-scene
```

Add `--dry-run` to print every command without executing it.

A quick smoke test on a long recording:

```bash
python -m annotator.cli.run_full_pipeline ... \
  --refine-limit 2 --level1-limit 4 --level2-limit 1 --level3-limit 1
```

Over a directory of per-scene annotation files:

```bash
python -m annotator.cli.run_all_scenes --help
```

## 📥 Input format

**Video** — any FFmpeg-readable container (`.mp4`, `.mkv`, …).

**Annotations** — JSON shaped as `{"frames": [...]}` or a bare list of frames:

```json
{
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ns": 1717171717000000000,
      "next_frame_timestamp_ns": 1717171717062500000,
      "keyboard_events": [
        { "event_type": "press", "key_name": "KEY_W" }
      ],
      "raw_mouse_events": [
        { "dx": -6, "dy": 1, "button_flag_names": [] }
      ]
    }
  ]
}
```

`dx > 0` is rightward and `dy > 0` is downward (Windows RawInput convention).
Mouse deltas may instead live in `mouse_events` — both layouts are accepted.

FPS is inferred from `timestamp_ns` spacing. Use `--fps`, `--video-start-ns` or
`--scene-start-offset-ns` if the recording and annotation clocks differ.

## 📤 Outputs

```
<workdir>/
├── <video>_segments.json            # 1a  segmentation
├── <video>_segments_refined.json    # 1b  refined segmentation
├── frames_level_1/manifest.json     # 1c  + decoded frames
├── level_1_instructions.json        # 2a  ← Level-1 instructions
├── frames_level_2/manifest.json     # 2b
├── level_2_instructions.json        # 3a  ← Level-2 instructions
├── frames_level_3/manifest.json     # 3b
├── level_3_instructions.json        # 4a  ← Level-3 instructions
└── .gemini_cache/                   # refine-stage response cache
```

Render instructions back onto the gameplay as caption videos to eyeball quality:

```bash
python -m annotator.cli.build_instruction_videos --help
```

## 🔁 Pipeline stages

Eight stages run in order; each is addressable by name with `--only` / `--skip`.

| # | Stage | What it does |
| --- | --- | --- |
| 1a | `segment` | cut the timeline into labelled segments |
| 1b | `refine` | refine boundaries and labels with a multimodal model |
| 1c | `level1_split` | decode the frames for each segment |
| 2a | `level1` | generate **Level-1** instructions |
| 2b | `merge12` | merge Level-1 scenes into Level-2 windows |
| 3a | `level2` | generate **Level-2** instructions |
| 3b | `merge23` | merge Level-2 scenes into Level-3 windows |
| 4a | `level3` | generate **Level-3** instructions |

```bash
--only level2                      # re-run one stage
--skip segment --skip refine       # reuse existing refined segments
```

## ⚙️ Options

**Segmentation** — `--segmenter action` (default) cuts on the keybind taxonomy and
labels every segment. `--segmenter scenedetect` cuts on visual shot changes instead,
needs no per-game assets, and leaves labelling to Stage 1b; tune it with
`--scenedetect-threshold` (27.0, lower cuts more), `--scenedetect-min-scene-frames`,
`--scenedetect-frame-skip`. `--segment-min-duration-s` applies to both.

**Models** — `--refine-model` (Stage 1b), `--instruction-model` (Levels 1–3).

**Window lengths** — `--level2-min-seconds` / `--level2-max-seconds` (60/120),
`--level3-min-seconds` / `--level3-max-seconds` (240/480), `--merge-seed` (42).

**Sampling** — `--temperature` is unset by default, so wording varies between runs.
Pass a value to reduce the spread.

**Throughput** — `--instruction-concurrency` (8), `--refine-concurrency`,
`--api-max-frames` (30), `--timeout`, `--max-retries`, `--retry-delay`.

**Frames** — `--level1-sample-max-frames` (`0` keeps the full span),
`--frame-max-width` (1280), `--image-ext`, `--jpeg-quality`.

**Resuming** — `--skip-existing` reuses finished instructions;
`--level1-limit` / `--level2-limit` / `--level3-limit` cap work per level.

**Disk** — a run keeps every intermediate so any stage can be re-run. `--cleanup`
discards them at the end: `cache` drops the refine cache, `frames` also drops the
decoded frames (most of the footprint), `all` additionally drops the Stage-1 segment
files. Instruction files are never touched.

**Slow storage (env)** — `OWA_LOCAL_VIDEO_CACHE_DIR`, `OWA_LOCAL_FRAME_CACHE_DIR`,
`OWA_VIDEO_PTS_CACHE_DIR`, `OWA_LEVEL1_SPARSE_JOBS`, `OWA_LEVEL1_FFMPEG_THREADS`.

## 🗂️ Layout

```
GameHorizon-Annotator/
├── pyproject.toml
└── src/annotator/
    ├── cli/            # pipeline driver and per-stage entry points
    ├── core/           # segmentation, timebase, frames, instructions, merging
    ├── game_profiles/  # one JSON profile per game
    └── game_assets/    # bundled Stage-1 assets
```

## 📜 License

Apache-2.0 — see [LICENSE](../LICENSE) at the repository root.

## 📖 Citation

```bibtex
@article{GameHorizonSuite2026,
      title={GameHorizon Suite: Multi-Horizon Data and Evaluation in Gameplay},
      author={Yiran Wang and Xingyilang Yin and Junfu Pu and Guangzhi Wang and Kaifeng Li and Mingyu Ouyang and Huiqiang Sun and Lingen Li and Cheng Cheng and Wangbo Yu and Honghao Chen and Xiaodong Cun and Chi-Man Pun and Zhiguo Cao and Ying Shan},
      year={2026},
      journal={arXiv preprint arXiv:2609.25001},
      url={https://arxiv.org/abs/2609.25001},
}
```
