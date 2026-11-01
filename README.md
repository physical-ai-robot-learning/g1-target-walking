# G1Nav: Humanoid Locomotion RL with Hierarchical Target Navigation (Unitree G1, MuJoCo / MJX)

A PPO walking policy trained from scratch, driven by a hierarchical target-navigation layer:
a slow target-selection stage and a fast learned locomotion policy. A simulated Unitree G1 humanoid receives an English instruction such as
*"go to the red cube"*, a target is selected once, a Navigator turns the target position into
a velocity command, and a PPO policy trained from scratch outputs **29 lower-body PD joint
targets at 50 Hz** until the robot stops next to the object. Only the locomotion policy is
learned. **In the recorded demos the target is chosen by keyword matching against the
simulator's object labels and its position is read from the simulator (ground truth); the
GR00T N1.6 Eagle vision-language encoder runs on the ego frame, but its output is only
logged and does not affect the target.** The question this repository answers is therefore
the control half: can one velocity-conditioned walking policy, trained once with RL, be
driven by a simple Navigator to reach instructed targets in closed loop, physics only?

[![ci](https://github.com/physical-ai-vla/g1-target-walking/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial-orange.svg)](LICENSE)

**Engineering report:** [`docs/report.md`](docs/report.md) ([PDF](docs/report.pdf)) ·
**Claims check:** `python analysis/verify_claims.py` ([audit](docs/claims.md)) ·
**Third-party licences:** [`docs/third_party.md`](docs/third_party.md) · **Provenance:** [`docs/provenance.md`](docs/provenance.md) · status: finished prototype, 2026-11-01

![go to the orange cylinder: third-person view (left) and ego camera (right)](videos/seed0_go-to-the-orange-cylinder.gif)

## Overview

| step | what happens | learned? |
|---|---|:---:|
| 1 · scene | a seeded arena with three coloured objects and the G1 at a seeded spawn pose (`code/envs/scene_gen.py`) | no |
| 2 · target selection | the object label whose words overlap most with the instruction (`code/navigation/grounding.py`); position read from the generated scene. The GR00T N1.6 Eagle encoder embeds the start ego frame (logged only) | no |
| 3 · Navigator | turn-then-go controller: target world position + odometry → velocity command `(vx, vy, wz)` every 20 ms; stops within 0.55 m (`code/navigation/action_module.py`) | no |
| 4 · locomotion | PPO policy on MuJoCo Playground's `G1JoystickFlatTerrain` (MJX): 103-D proprioceptive observation + command → 29-D PD joint targets (`code/locomotion/train_walk.py`) | yes, RL from scratch |
| 5 · closed loop | MuJoCo physics at 500 Hz (10 substeps per control step), ego and third-person video (`code/navigation/navigate.py`) | — |

### Key findings

- **Walking from scratch.** The policy was trained for 154.8 M environment steps (77 min wall-clock,
  including compilation). In a CPU re-evaluation of all 16 saved checkpoints, the first one that walks
  without falling is at 41.3 M steps; the final policy moves at 0.52 m/s for a 0.6 m/s command
  (87 %) with a minimum torso height of 0.74 m over 5 s.
- **Closed-loop reaching.** In all 3 demo scenes the robot reaches the instructed object
  (Navigator stop radius 0.55 m) and stays upright: final distances 0.39 / 0.41 / 0.54 m,
  torso height 0.75–0.76 m. A CPU re-run of the three episodes reproduces these numbers.
- **What made it work.** Domain randomisation during training; evaluating in the raw
  environment, because the training wrapper's auto-reset made a falling policy look like it was
  stepping in place; and two deployment bugs: missing observation normalisation, and a gravity
  term computed in the wrong frame, which only showed when the robot turned
  ([engineering log](docs/report.md#5-what-went-wrong-and-the-fixes-engineering-log)).

### What this repository does not show

- **No learned or visual grounding.** Target selection is a keyword rule over simulator object
  labels, and the target position is simulator ground truth. The GR00T N1.6 Eagle embedding is
  computed and logged but never used. The colour/shape + ego-depth localisation code
  (`color_detect.py`, `Navigator.lock_goal_from_ego`) is not used in the demos and was not
  evaluated.
- **No vision in the control loop.** The walking policy observes proprioception and the
  velocity command only. This is not an end-to-end vision-language-action model.
- **No success rate.** Three seeded scenes, one run each, static objects, flat terrain, and each
  instruction names a unique colour + shape. Seed 2 stops at 0.54 m against a 0.55 m radius.
- **No new locomotion method.** The policy follows the MuJoCo Playground G1 Joystick recipe and
  reward; the work here is the coupling and the debugging.
- **No hardware.** Simulation only; no sim-to-real transfer was attempted.
- **No original training logs.** Per-step training console logs and the GPU runs' closed-loop
  logs are not included; the walking and closed-loop numbers are re-measured on CPU from the
  committed checkpoints (`results/`).

## Components

| component | role | where |
|---|---|---|
| scene generator | seed → 3 objects (colour, shape, pose), robot spawn, MJCF | `code/envs/scene_gen.py`, `code/envs/instructions.py` |
| integrated scene | Playground G1 model + objects + ego / third-person cameras | `code/navigation/integrated_scene.py` |
| target selection | keyword rule; optional Eagle embedding log | `code/navigation/grounding.py`, `code/navigation/cortex_eagle.py` |
| Navigator | world target + odometry → `(vx, vy, wz)`, slew-limited | `code/navigation/action_module.py` |
| closed-loop runner | policy + physics + video; one-shot pipeline | `code/navigation/navigate.py`, `code/navigation/g1nav_run.py` |
| walking policy training | brax PPO on `G1JoystickFlatTerrain`, domain randomisation, raw-env rollouts | `code/locomotion/train_walk.py` |
| walking evaluation | raw environment, no auto-reset; one checkpoint or a sweep | `code/locomotion/walk_eval.py` |
| dashboards | results replay (8501), interactive runs (8502), training monitor | `code/apps/`, `code/locomotion/app_train.py`, `run_dashboard.sh` |
| evidence and checks | readers for committed results, claims check, tests | `analysis/`, `tests/` |

## Installation

```bash
git clone https://github.com/physical-ai-vla/g1-target-walking.git
cd g1-target-walking
bash install.sh              # NVIDIA GPU -> full stack; otherwise dashboard-only
source .venv/bin/activate && bash run_dashboard.sh
```

The checks need only Python 3 with numpy and Pillow (`pip install numpy pillow pytest`). The
simulation stack is pinned to jax 0.5.3, brax 0.14.2, flax 0.10.6, mujoco / mujoco-mjx 3.4.0 and
playground 0.1.0. The GR00T N1.6 Eagle encoder is an **optional, logging-only** dependency kept in
[`requirements-eagle.txt`](requirements-eagle.txt) (old `transformers` pin for compatibility); it is not needed for
training, navigation or any reported result. Remote-GPU mode, the Eagle set-up and the CPU set-up used for the
re-measurements are described in [INSTALL.md](INSTALL.md).

## Pipeline

```bash
# 1. seeded scene (writes assets/scene_seed0.xml)
python code/envs/scene_gen.py --seed 0

# 2. train the walking policy (GPU; default Playground reward scales)
python code/locomotion/train_walk.py --timesteps 500000000 --num_envs 8192 \
    --termination -100 --track_lin 1.0 --stand_still -1.0 --num_evals 50 \
    --out checkpoint/walk_t4

# 3. evaluate every checkpoint in the raw environment (command vx = 0.6 m/s)
G1NAV_OUT_DIR=results/walk_sweep_cpu python code/locomotion/walk_eval.py \
    --sweep checkpoint 'walk_t4_step*.pkl'

# 4. one closed-loop episode: scene -> keyword target -> Navigator -> policy -> physics -> video
python code/navigation/g1nav_run.py --seed 0 --instruction "go to the orange cylinder"

# 5. re-measure the three demo episodes (CPU is enough)
python scripts/rerun_demo_episodes.py --cpu --outdir results/demo_rerun_cpu
```

## Codebase guide

```
code/
  envs/        scene_gen.py, instructions.py         seeded scenes and instruction templates
  locomotion/  train_walk.py, walk_eval.py,          PPO training, raw-env evaluation,
               rollout_walk.py, app_train.py         rollout video, training dashboard
  navigation/  grounding.py, cortex_eagle.py,        target selection (+ Eagle embedding log)
               action_module.py, navigate.py,        Navigator, closed loop
               integrated_scene.py, g1nav_run.py,    scene assembly, one-shot runner
               color_detect.py                       colour/shape detector (not used in the demos)
               *.sh                                  runners for a GPU machine with the repo at /workspace
  apps/        app_results.py, app_interactive.py    Streamlit dashboards (8501 / 8502)
analysis/      evidence.py, verify_claims.py         readers for committed results; claims check
tests/         pytest suite (no GPU, no MuJoCo)
scripts/       rerun_demo_episodes.py, smoke_test.py, build_report.sh, pod/ (machine set-up)
results/       CPU re-measurements (demo episodes, checkpoint sweep)
checkpoint/    walk_t4_latest.pkl + 16 walk_t4_step*.pkl, training metrics, final rollout
videos/        3 demo episodes (mp4, gif, start ego frame, cognition JSON), VIDEOS.md
assets/        vendored Menagerie unitree_g1 model, example generated scene
dataset/       NOTE.md: there is no collected dataset; scenes are regenerated from seeds
docs/          report, claims audit, third-party licences
legacy/        planner.py (unused behaviour-cloning teacher)
```

Conventions: the observation and action layouts are those of Playground's G1 Joystick
environment (103-D state, 216-D privileged state, 29 actions,
`motor_target = default_pose + 0.5 · action`); control at 50 Hz with physics at 500 Hz; the
velocity command is in the robot's heading frame.

## Results and how to verify

`analysis/verify_claims.py` recomputes every number in this README, in
[`docs/report.md`](docs/report.md) and in [`videos/VIDEOS.md`](videos/VIDEOS.md) from files in
the repository (checkpoint pickles read without JAX, metrics JSONL, rollout JSON, cognition
JSON, MP4/GIF frame counts, the CPU re-measurements) and exits non-zero on any mismatch:

```bash
python analysis/verify_claims.py          # prints "N claims checked, 0 failed."
python -m pytest -q tests                 # no GPU, no MuJoCo
```

**Walking policy** (`checkpoint/`, `results/walk_sweep_cpu/walk_sweep.json`):

| | value | source |
|---|---|---|
| training steps (final checkpoint) | 154,828,800 | `step` field of `checkpoint/walk_t4_latest.pkl` |
| training time / final evaluation reward | 77.1 min / 9.29 | `checkpoint/walk_t4_metrics.jsonl` |
| first checkpoint that walks (raw env) | 41.3 M steps (0.32 m/s, min torso 0.73 m) | `results/walk_sweep_cpu/walk_sweep.json` |
| final policy, 5 s re-evaluation | 0.52 m/s for a 0.6 m/s command, 2.59 m, min torso 0.74 m | `results/walk_sweep_cpu/walk_sweep.json` |
| final policy, 4 s training-time rollout | 0.519 m/s, 2.07 m, min torso 0.745 m | `checkpoint/walk_t4_rollout.json` |
| highest forward velocity in the sweep | 0.58 m/s at 144.5 M steps | `results/walk_sweep_cpu/walk_sweep.json` |

**Closed-loop demo episodes** (`videos/`, `results/demo_rerun_cpu/episodes.json`):

| instruction | chosen target | reached | final distance | torso height | frames |
|---|---|:---:|---:|---:|---:|
| "go to the orange cylinder" | orange cylinder | yes | 0.39 m | 0.75 m | 630 |
| "go to the red cube" | red cube | yes | 0.41 m | 0.76 m | 387 |
| "go to the purple cylinder" | purple cylinder | yes | 0.54 m | 0.75 m | 668 |

**"go to the red cube"**
![go to the red cube](videos/seed1_go-to-the-red-cube.gif)

**"go to the purple cylinder"**
![go to the purple cylinder](videos/seed2_go-to-the-purple-cylinder.gif)

Each episode also has its MP4 (50 fps), the start ego frame and the cognition JSON in
[`videos/`](videos/VIDEOS.md). The cognition JSON stores the keyword-selected label under the
historical key `vlm_target` and the Eagle embedding size (286 tokens × 2048-d); its stage-3 text
describes a CV + depth localisation step that was not used (see `videos/VIDEOS.md`).

## Current vs superseded

| current | superseded (kept) |
|---|---|
| keyword target selection with a ground-truth position (`grounding.py`, `navigate.py`) | the intended frozen vision-language grounding and CV + ego-depth localisation (`cortex_eagle.py` embedding, `color_detect.py`, `--use_vlm` flag declared but not implemented) |
| Navigator driving a velocity-conditioned PPO policy (`action_module.py`) | behaviour-cloning plan with a privileged teacher planner (`legacy/planner.py`) |
| default Playground reward scales, raw-env evaluation (`walk_eval.py`) | edited reward scales (`train_walk.py` argument defaults `--termination -3 --track_lin 2.0 --stand_still -3.0`); the softened termination penalty produced the marching-in-place policy described in the report |

Earlier versions of this README and of the report presented the Eagle encoder as the grounding
stage; the repository history before 2026-09-27 reflects that wording.

## Design intuition (optional)

The split mirrors a common pattern in motor control: a slow, deliberate decision about *what*
and *where*, and a fast reactive controller for balance and stepping. Here the slow part runs
once per episode for a static scene and the fast part every 20 ms. This analogy motivated the
structure; nothing in this repository tests it.

## Attribution

- **Robot model:** `assets/mujoco_menagerie/unitree_g1/` is vendored from
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Google DeepMind);
  the Unitree G1 model files are Copyright (c) 2016-2023 Unitree Robotics, BSD-3-Clause.
- **Environment and recipe:** the walking environment, reward and PPO hyperparameters come
  from [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground); training
  uses [Brax](https://github.com/google/brax) PPO and MuJoCo MJX.
- **Eagle encoder:** the vision-language backbone of
  [NVIDIA Isaac GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T); not redistributed here,
  obtained separately under NVIDIA's terms.

Details and licences: [`docs/third_party.md`](docs/third_party.md).

## License and citation

Original code, documentation, results and media in this repository are licensed under the
**PolyForm Noncommercial License 1.0.0** ([LICENSE](LICENSE)): free for research, teaching,
personal and other noncommercial use; commercial use requires a separate written agreement with
the author. Third-party parts keep their own licences (`assets/mujoco_menagerie/unitree_g1/`: BSD-3-Clause,
Unitree Robotics; see
[`docs/third_party.md`](docs/third_party.md)).

Citation: see [`CITATION.cff`](CITATION.cff).

```bibtex
@software{lee_g1nav_2026,
  author = {Lee, Jeong Hwan},
  title  = {G1Nav: Humanoid Locomotion RL with Hierarchical Target Navigation (Unitree G1, MuJoCo/MJX)},
  year   = {2026},
  url    = {https://github.com/physical-ai-vla/g1-target-walking}
}
```
