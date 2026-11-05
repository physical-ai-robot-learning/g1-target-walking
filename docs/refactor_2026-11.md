# 2026-11 layout and code-style refactor

Branch `refactor/oa-layout`, started from `main` at `702a60a`. Goal: the organisation-wide layout (one installable
package under `src/`, `archive/` instead of `legacy/`, `media/`, `results/`, `configs/`, thin `scripts/`) and the
"OA" code style, **without changing behaviour or any number** in the README, report or `media/README.md`.

## Commits

| commit | content |
|---|---|
| characterization tests | golden values recorded on the untouched `code/` tree (see "Verification") |
| layout | `git mv` into the new tree + import / path rewrites only; `pyproject.toml`, `g1nav.paths` |
| OA style | module-by-module restyle, docstrings, `main(argv) -> int`, no import-time work, READMEs, docs |

## Old → new path map

| old | new |
|---|---|
| `code/envs/scene_gen.py`, `instructions.py` | `src/g1nav/envs/` (same names) |
| `scripts/smoke_test.py` | `src/g1nav/envs/smoke_test.py` (`python -m g1nav.envs.smoke_test`) |
| `code/navigation/{grounding,action_module,navigate,integrated_scene,cortex_eagle}.py` | `src/g1nav/navigation/` (same names) |
| `code/navigation/g1nav_run.py` | `src/g1nav/navigation/pipeline.py` |
| `scripts/rerun_demo_episodes.py` | `src/g1nav/navigation/rerun_demo_episodes.py` |
| `code/locomotion/{train_walk,walk_eval,rollout_walk}.py` | `src/g1nav/locomotion/` (same names) |
| `code/locomotion/app_train.py` | `src/g1nav/apps/training_dashboard.py` |
| `code/apps/app_results.py`, `app_interactive.py` | `src/g1nav/apps/results_dashboard.py`, `interactive_dashboard.py` |
| `code/navigation/{preview_scene,run_pipeline_n16,render_n16_videos}.sh` | `scripts/pod/` (same names) |
| `code/navigation/color_detect.py`, `run_cog_n16.sh` | `archive/` |
| `legacy/` | `archive/` |
| `videos/*`, `videos/VIDEOS.md` | `media/*`, `media/README.md` |
| `checkpoint/*` | `results/checkpoints/*` |
| `dataset/NOTE.md` | `docs/dataset.md` |
| `INSTALL.md` | `docs/install.md` |
| `docs/report.md`, `docs/report.pdf` | `report/report.md`, `report/report.pdf` |
| `install.sh`, `run_dashboard.sh` | `scripts/install.sh`, `scripts/run_dashboard.sh` |
| `requirements-{gpu,dashboard,eagle}.txt` | `configs/envs/` (kept as the record of what was run) + extras in `pyproject.toml` |
| `python code/<x>/<m>.py` | `python -m g1nav.<x>.<m>` (after `pip install -e .`) |
| `python analysis/verify_claims.py` | `python -m analysis.verify_claims` |
| Eagle default `checkpoint/eagle-n16` (one-shot runner) | `assets/eagle-n16` (git-ignored) or `$G1NAV_EAGLE` |

Renamed identifiers (OA style; the full list is in the diff): `sample_scene` → `sampleScene`, `build_mjcf` →
`buildMjcf`, `ObjSpec.color/shape/x/y/z` → `mColor/mShape/mX/mY/mZ` (`label` stays a property), `SceneSpec.*` →
`mSeed/mObjects/mArenaHalf/mRobotX/mRobotY/mRobotYaw` (`toDict()` keeps the old serialised keys),
`sample_instruction` → `sampleInstruction`, `Instruction.*` → `m*`, `keyword_target` / `label_score` →
`keywordTarget` / `labelScore`, `ego_to_world` / `world_to_ego` / `goal_to_command` / `robot_pose_from_qpos` →
camelCase, `navigate.quat_yaw` → `action_module.quatYaw`, `Navigator.goal_world/reached/lock_goal_from_ego/set_policy`
→ `mGoalWorld/mReached/lockGoalFromEgo/setPolicy`, `build_integrated_xml` → `buildIntegratedXml`,
`render_scene` / `ground_target` → `renderScene` / `selectTarget`, `walk_eval.N` → `N_STEPS`. Unchanged on purpose:
argparse flags and dests, JSON keys (`vlm_target`, `eagle_patches`, ...), environment variables, the `NAV` / `ROLLOUT` /
`CORTEX_DONE` / `STAGE=` output lines, brax callback signatures (`progress(step, metrics)`,
`saveCheckpoint(step, make_policy, params)`), Streamlit session-state keys.

Structural changes that keep behaviour: `walk_eval` did all its work at import time and parsed `sys.argv` by hand; it is
now a `WalkEvaluator` class with an argparse `main` that accepts the same two forms (`<ckpt>` and
`--sweep DIR PATTERN`). `navigate` printed and set `JAX_PLATFORMS` at import time; this now happens first in `main`.
The scene-XML pieces (ego camera, third-person camera, walls, objects, spawn keyframe) were duplicated in
`scene_gen` and `integrated_scene`; they are now shared helpers producing the identical strings. The three pod
scripts embedded a copy of the scene-render code; they now call `g1nav.navigation.pipeline.renderScene` (same
function as the one-shot runner) and take the workspace root from `G1NAV_WORKSPACE` (default `/workspace`).
The dashboards' module-level UI code is in `main()`; the interactive dashboard's remote machine is a `PodConfig`
(`G1NAV_POD_HOST/_PORT/_KEY/_WORKSPACE`, placeholder host only).

## What was archived

`archive/color_detect.py` (colour/shape detector, never used or evaluated) and `archive/run_cog_n16.sh` (pod script
that calls a non-existent `color_detect.detect`), plus the former `legacy/planner.py`. See
[archive/README.md](../archive/README.md). Nothing was deleted.

## Verification

Environment: macOS (Apple silicon). Checks venv: Python 3.12, numpy, Pillow, pytest, streamlit 1.66, pandas.
Simulation venv: Python 3.11, jax/jaxlib 0.5.3, brax 0.14.2, flax 0.10.6, mujoco / mujoco-mjx 3.4.0, playground 0.1.0
(CPU, `MUJOCO_GL=cgl`), the same recipe as `docs/install.md` "Re-measuring on CPU".

| check | before (`702a60a` tree) | after | result |
|---|---|---|---|
| claims check | 70 claims, 0 failed | 70 claims, 0 failed | same count; snippets that quote code (`CTRL_DT = 0.02`, `ACTION_SCALE = 0.5`, `tGoalWorld = (tTarget.mX, tTarget.mY)`, `N_STEPS`) and paths were updated |
| pytest (CI set) | 28 passed | 46 passed, 1 skipped (opt-in sim) | 18 characterization tests added before the refactor |
| characterization goldens (`tests/golden/characterization.json`) | recorded | identical | scenes of seeds 0–19, MJCF sha256 (seeds 0–2), instructions of seeds 0–19, success rules, Navigator command grid + 3 kinematic rollouts (sha256 of every command), geometry helpers, constants |
| CLI parsers (`tests/golden/cli_parsers.json`) | recorded | identical | 7 entry points: flags, dests, defaults, types, help strings; repository-path defaults mapped through the path table |
| dashboards' first render (`tests/golden/apps.json`) | recorded | identical except the intended changes below | results, interactive, training pages via streamlit AppTest |
| demo episodes re-run on CPU (`rerun_demo_episodes --cpu --keep_video`) | `NAV` lines = `results/demo_rerun_cpu` | same `NAV` lines | **rendered MP4s byte-identical** (sha1 c5647a2e…, 03d6a0e3…, af589450…; the pre-refactor code itself is byte-deterministic across two runs) |
| 16-checkpoint raw-env sweep on CPU (`walk_eval --sweep`) | `walk_sweep.json` byte-identical to the committed file | byte-identical | — |
| final checkpoint single eval (`tests/test_sim_rerun.py`) | passes | passes | opt-in (`G1NAV_SIM_TESTS=1`) |
| one-shot runner `pipeline --seed 1` (no Eagle) | — | cognition JSON, ego PNG and MP4 byte-identical to the old `g1nav_run.py` | — |
| `scene_gen --seed 4 --validate`, `instructions`, `action_module` self-test, `integrated_scene` self-test, `smoke_test` | — | same stdout; scene XML identical up to the relative `meshdir` (depends on the checkout location); smoke PNG identical | — |
| tiny CPU training (`train_walk --timesteps 1 --num_envs 8 --num_evals 1`: one PPO iteration, 2 checkpoint callbacks with raw-env rollouts, about 5 min) | recorded | same | all 5 pickles (final params, latest, 2 numbered, env config) leaf-for-leaf identical; rollout JSON and MP4 byte-identical; metrics identical except wall-clock |
| byte-compile, `bash -n` on all shell scripts | ok | ok | — |
| markdown link check, secret scan | — | 0 broken relative links; no tokens, keys, IPs, ports, private e-mail addresses or home paths | — |
| fresh clone of the branch, `pip install -e ".[test,dashboard]"`, CI steps | — | see the report of this refactor | — |

Intended visible changes (recorded in the tests, not hidden): the results dashboard's subtitle said "Language-conditioned
lower-body navigation"; it now reads "Humanoid locomotion RL with hierarchical target navigation" (the project's actual
scope). Module paths shown in dashboard captions changed with the move. The old results dashboard wrote
`st.video(x) if x else st.info(...)` as bare expressions, which made Streamlit "magic" print a help box of the returned
`DeltaGenerator` under each walking-emergence panel; written as if/else statements these boxes are gone.

## Not executable here

- Full-scale PPO training (`train_walk` with 8192 MJX environments) needs an NVIDIA GPU; only the one-iteration CPU run
  above was executed.
- `cortex_eagle` (GR00T N1.6 Eagle embedding) needs torch, transformers 4.51.3, the non-redistributed backbone and a
  CUDA GPU: only its argument parser was checked (torch / transformers stubbed). Its rewrite is mechanical (lazy imports,
  renamed locals, same JSON).
- `rollout_walk` (training-dashboard rollout) and the dashboards' Run / Roll out buttons, LOCAL GPU mode and REMOTE SSH
  mode: not run (no GPU machine, no remote hosts allowed). The pod shell scripts are syntax-checked only.
- `scripts/pod/pod_bootstrap.sh`, `setup_ada.sh` (apt / pip on a fresh Linux GPU machine): not run.

## Suspected bugs found (recorded, not fixed)

1. `rollout_walk.py` builds the PPO network **without** `preprocess_observations_fn=running_statistics.normalize`, so the
   training dashboard's rollout feeds the policy raw observations — the same deployment bug the report describes as
   fixed in `navigate.py` / `walk_eval.py` (§5.7a). The dashboard rollout of a trained checkpoint is therefore expected
   to fall; no reported number uses it.
2. `archive/run_cog_n16.sh` calls `color_detect.detect(path)`, which does not exist (archived, not fixed).
3. `assets/scene_seed0.xml` is not the current generator's output for seed 0: it has an older ego camera
   (`pos="0.10 0 0.45"`, `fovy="70"` instead of `0.08 0 0.55` / `95`) and the robot at the origin. It is labelled an
   "example output" and nothing reads it; regenerate with `python -m g1nav.envs.scene_gen --seed 0`.
4. `train_walk.py`'s reward-scale argument defaults are the superseded edited values (`--termination -3 --track_lin 2.0
   --stand_still -3.0`); the final run passed the Playground defaults explicitly. Documented in the README; kept so the
   CLI does not change.
5. `results/demo_rerun_cpu/episodes.json` records `"checkpoint": "checkpoint/walk_t4_latest.pkl"` (the path at the
   time of the run). A re-run now writes `results/checkpoints/walk_t4_latest.pkl`; the committed file was left as
   recorded.
6. `integrated_scene.buildIntegratedXml` writes its patched / integrated XML files into the installed
   `mujoco_playground` package folder (needed for its relative includes); it fails on a read-only site-packages.
