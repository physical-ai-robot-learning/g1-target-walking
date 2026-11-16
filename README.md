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

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial-orange.svg)](LICENSE)

Part of **[physical-ai-robot-learning](https://github.com/physical-ai-robot-learning)** — Physical AI / robot learning projects.

**Technical report:** [`report/report_en.pdf`](report/report_en.pdf) (earlier Markdown engineering report in [`report/archive/`](report/archive/report.md)) ·
**Results:** [`results/`](results/README.md) ·
**Claims audit:** [`docs/claims.md`](docs/claims.md) ·
**Code:** [available on request](#code-availability) ·
**Third-party licences:** [`docs/third_party.md`](docs/third_party.md) · **Provenance:** [`docs/provenance.md`](docs/provenance.md) · status: finished prototype, 2026-11-01

![go to the orange cylinder: third-person view (left) and ego camera (right)](media/seed0_go-to-the-orange-cylinder.gif)

*"go to the orange cylinder": third-person view (left) and ego camera (right), 2 s standing, walking, 2 s standing.*

## Result and status

**What works, measured how**

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
  ([technical report](report/report_en.pdf), Sec. VI; [engineering log](report/archive/report.md#5-what-went-wrong-and-the-fixes-engineering-log)).

**What this repository does not show**

- **No learned or visual grounding.** Target selection is a keyword rule over simulator object
  labels, and the target position is simulator ground truth. The GR00T N1.6 Eagle embedding is
  computed and logged but never used. The colour/shape + ego-depth localisation code
  (`archive/color_detect.py`, `Navigator.lockGoalFromEgo`) is not used in the demos and was not
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

## Code availability

The source code (PPO walking-policy training and raw-environment evaluation, the Navigator and
closed-loop episode runner, seeded scene generation, and the evaluation / claims-check scripts)
and the trained checkpoints are available on request: please open an issue in this repository.
Every number in the technical report was re-measured by a claims-check script against the
result files; that script is included with the code on request.

## What is in this repository

- `README.md`: this overview, with the results tables and limitations.
- [`report/report_en.pdf`](report/report_en.pdf): the technical report; the earlier Markdown
  engineering report is in [`report/archive/`](report/archive/report.md).
- [`media/`](media/README.md): the three demo episodes (MP4, GIF, start ego frame, cognition JSON).
- [`docs/`](docs/): claims audit, data note, provenance, third-party licences.
- [`results/`](results/README.md): training metrics, the final-checkpoint rollout (JSON + MP4) and
  the CPU re-measurements behind the numbers.

## Pipeline

| stage | what happens | learned? | module |
|---|---|:---:|---|
| 1 · scene | a seeded arena with three coloured objects and the G1 at a seeded spawn pose | no | `g1nav.envs.scene_gen` |
| 2 · target selection | the object label whose words overlap most with the instruction; position read from the generated scene. The GR00T N1.6 Eagle encoder embeds the start ego frame (logged only) | no | `g1nav.navigation.grounding`, `.cortex_eagle` |
| 3 · Navigator | turn-then-go controller: target world position + odometry → velocity command `(vx, vy, wz)` every 20 ms; stops within 0.55 m | no | `g1nav.navigation.action_module` |
| 4 · locomotion | PPO policy on MuJoCo Playground's `G1JoystickFlatTerrain` (MJX): 103-D proprioceptive observation + command → 29-D PD joint targets | yes, RL from scratch | `g1nav.locomotion.train_walk` |
| 5 · closed loop | MuJoCo physics at 500 Hz (10 substeps per control step), ego and third-person video | — | `g1nav.navigation.navigate` |

Conventions: the observation and action layouts are those of Playground's G1 Joystick
environment (103-D state, 216-D privileged state, 29 actions,
`motor_target = default_pose + 0.5 · action`); control at 50 Hz with physics at 500 Hz; the
velocity command is in the robot's heading frame.

## Results

The claims-check script (part of the code, available on request) recomputes every number in this
README, in [`report/archive/report.md`](report/archive/report.md) and in [`media/README.md`](media/README.md)
from the result files (checkpoint pickles read without JAX, metrics JSONL, rollout JSON, cognition
JSON, MP4/GIF frame counts, the CPU re-measurements) and exits non-zero on any mismatch. A second
check confirms that every number printed in the technical report appears in the file it was taken
from. The checkpoint pickles are part of the code release; the other files are in this repository.

**Walking policy** (`results/checkpoints/`, `results/walk_sweep_cpu/walk_sweep.json`):

| | value | source |
|---|---|---|
| training steps (final checkpoint) | 154,828,800 | `step` field of `results/checkpoints/walk_t4_latest.pkl` |
| training time / final evaluation reward | 77.1 min / 9.29 | `results/checkpoints/walk_t4_metrics.jsonl` |
| first checkpoint that walks (raw env) | 41.3 M steps (0.32 m/s, min torso 0.73 m) | `results/walk_sweep_cpu/walk_sweep.json` |
| final policy, 5 s re-evaluation | 0.52 m/s for a 0.6 m/s command, 2.59 m, min torso 0.74 m | `results/walk_sweep_cpu/walk_sweep.json` |
| final policy, 4 s training-time rollout | 0.519 m/s, 2.07 m, min torso 0.745 m | `results/checkpoints/walk_t4_rollout.json` |
| highest forward velocity in the sweep | 0.58 m/s at 144.5 M steps | `results/walk_sweep_cpu/walk_sweep.json` |

**Closed-loop demo episodes** (`media/`, `results/demo_rerun_cpu/episodes.json`):

| instruction | chosen target | reached | final distance | torso height | frames |
|---|---|:---:|---:|---:|---:|
| "go to the orange cylinder" | orange cylinder | yes | 0.39 m | 0.75 m | 630 |
| "go to the red cube" | red cube | yes | 0.41 m | 0.76 m | 387 |
| "go to the purple cylinder" | purple cylinder | yes | 0.54 m | 0.75 m | 668 |

**"go to the red cube"**
![go to the red cube](media/seed1_go-to-the-red-cube.gif)

**"go to the purple cylinder"**
![go to the purple cylinder](media/seed2_go-to-the-purple-cylinder.gif)

Each episode also has its MP4 (50 fps), the start ego frame and the cognition JSON in
[`media/`](media/README.md). The cognition JSON stores the keyword-selected label under the
historical key `vlm_target` and the Eagle embedding size (286 tokens × 2048-d); its stage-3 text
describes a CV + depth localisation step that was not used (see `media/README.md`).

## Current vs superseded

| current | superseded (kept) |
|---|---|
| keyword target selection with a ground-truth position (`grounding.py`, `navigate.py`) | the intended frozen vision-language grounding and CV + ego-depth localisation (`cortex_eagle.py` embedding, `archive/color_detect.py`, `--use_vlm` flag declared but not implemented) |
| Navigator driving a velocity-conditioned PPO policy (`action_module.py`) | behaviour-cloning plan with a privileged teacher planner (`archive/planner.py`) |
| default Playground reward scales, raw-env evaluation (`walk_eval.py`) | edited reward scales (`train_walk.py` argument defaults `--termination -3 --track_lin 2.0 --stand_still -3.0`); the softened termination penalty produced the marching-in-place policy described in the report |

Earlier versions of this README and of the report presented the Eagle encoder as the grounding
stage; the repository history before 2026-09-27 reflects that wording. The code moved from `code/` to the
`src/g1nav` package in 2026-11 without behaviour change. The code paths named in this README
refer to the code release, which is available on request.

## Design intuition (optional)

The split mirrors a common pattern in motor control: a slow, deliberate decision about *what*
and *where*, and a fast reactive controller for balance and stepping. Here the slow part runs
once per episode for a static scene and the fast part every 20 ms. This analogy motivated the
structure; nothing in this repository tests it.

## Attribution

- **Robot model:** the code release vendors `assets/mujoco_menagerie/unitree_g1/` from
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

The documentation, report, results and media in this repository (and the code, when shared on request) are licensed under the
**PolyForm Noncommercial License 1.0.0** ([LICENSE](LICENSE)): free for research, teaching,
personal and other noncommercial use; commercial use requires a separate written agreement with
the author. Third-party parts keep their own licences (the Unitree G1 model in the code release:
BSD-3-Clause, Unitree Robotics; see
[`docs/third_party.md`](docs/third_party.md)).

Citation: see [`CITATION.cff`](CITATION.cff).

```bibtex
@software{lee_g1nav_2026,
  author = {Lee, Jeong Hwan},
  title  = {G1Nav: Humanoid Locomotion RL with Hierarchical Target Navigation (Unitree G1, MuJoCo/MJX)},
  year   = {2026},
  url    = {https://github.com/physical-ai-robot-learning/g1-target-walking}
}
```
