# Claims audit

Audit date 2026-11-01. Scope: every quantitative or framing statement in this repository's
README, report and `videos/VIDEOS.md`, in the earlier README / report wording (main branch up to
2026-09-27), and claims about this project that a CV or application might make. The CV drafts
reviewed at the time of the audit contain no statement about this project; the claims listed
under "CV-level claims" are the ones such a document would be tempted to make.

Status: **SUPPORTED** (evidence in the repository, re-checked), **PARTLY** (true only with the
stated rewording), **NOT SUPPORTED** (contradicted by, or absent from, the repository).
`python analysis/verify_claims.py` re-checks every number of the current README, report and
`videos/VIDEOS.md` automatically (70 checks).

## What the recorded demos actually used

Established from the code at the commit that added the demo videos (2026-06-01) and from the
recorded outputs; the target-selection and control logic is unchanged since then (later commits
changed only docstrings, captions and file locations, and moved the keyword rule into
`grounding.py` with identical behaviour, tested in `tests/test_grounding.py`).

| stage | what ran | evidence |
|---|---|---|
| ego frame | the start ego frame was rendered and saved | `videos/*.ego.png` |
| Eagle encoder | the GR00T N1.6 Eagle backbone **did run** on the ego frame; only the shape (286 × 2048) and norm of one hidden state were stored | `videos/*.cognition.json` keys `eagle_patches`, `eagle_embed_dim`, `eagle_norm` (a run without the encoder writes a `note` key instead, `code/navigation/g1nav_run.py:73-75`); `code/navigation/cortex_eagle.py:39-55` |
| target label | keyword overlap between the instruction and the scene's object labels; the embedding is not used | `code/navigation/cortex_eagle.py:73`, `code/navigation/grounding.py:21-27`; the recorded JSON's `vlm_target` equals the keyword rule's output for all three (`tests/test_grounding.py`) |
| target position | read from the generated scene (simulator ground truth) | `code/navigation/navigate.py:152`; `--use_vlm` is parsed (`:49`) but never read |
| motion | Navigator → PPO policy → `d.ctrl` → `mj_step`, physics only | `code/navigation/navigate.py:225-227`; `code/navigation/action_module.py:101-114` |
| control input | proprioception + velocity command; no image | `code/navigation/navigate.py:188-201` (`get_obs`) |

## NOT SUPPORTED

| claim (where it appeared) | finding | evidence | what to say instead |
|---|---|---|---|
| "0.55 m/s forward velocity" (CV-level) | The final policy moves at 0.519 m/s (4 s training-time rollout) / 0.52 m/s (5 s CPU re-evaluation) for a 0.6 m/s command. 0.55 m/s appears only for intermediate checkpoints in the CPU sweep (72.3 M, 113.5 M steps); the highest is 0.58 m/s at 144.5 M. 0.55 is also the Navigator's stop radius in metres, a likely source of the mix-up. | `checkpoint/walk_t4_rollout.json` (`fwd_vel` 0.5191); `results/walk_sweep_cpu/walk_sweep.json`; `code/navigation/action_module.py:46` | "0.52 m/s for a 0.6 m/s command (87 %)" |
| GR00T N1.6 Eagle does zero-shot perception / grounding of the target ("N1.6 grounded target", "Cognition — GR00T N1.6 Eagle (zero-shot)", earlier README/UI) | The encoder runs, but its output is only logged. The target is chosen by a keyword rule and its position is simulator ground truth. | table above | "the GR00T N1.6 Eagle encoder is run on the ego frame and its embedding logged; it does not influence the target" |
| "VLA" / "small VLA" framing; "VLA-0-style thin action module"; "data bottleneck sidestepped" (earlier README §Design rationale, report §2.4–2.5) | No vision or language model takes part in any decision; the policy sees no image. It is not a vision-language-action model. The data argument only holds because grounding is hand-written. | `code/navigation/navigate.py:188-201` (`get_obs`); `grounding.py` | "hierarchical control: hand-written target selection + Navigator + learned locomotion policy" |
| "Maps RGB-D frames + proprioception + histories + instruction → joint targets" (earlier report overview) | The policy input is the 103-D proprioceptive state with the command; no RGB-D and no history beyond the last action. | `code/navigation/navigate.py:198-200`; checkpoint first layer (103, 512) | "103-D proprioceptive observation + command → 29-D joint targets" |
| "real time on a single consumer GPU" (earlier report overview) | The training GPU (RTX 6000 Ada) is a workstation card and is not recorded in any committed file; no real-time measurement exists. | — | "trained on a single GPU in 77 min wall-clock" |
| Report §6 command `python -c "from code.envs.scene_gen import ..."` (dataset/NOTE.md) | Fails: the standard-library module `code` shadows the `code/` directory. | run it | fixed: `cd code/envs && python -c ...` |
| "8501 replays the recorded demos" (earlier README) | With default settings the results dashboard looked only in `~/Desktop/g1nav_dash` and `~/g1nav/videos`, so on a fresh clone it showed nothing. | `code/apps/app_results.py:23-24` (defaults), fallback added at `:26-29` | fixed: it now falls back to the repository's `videos/` and `checkpoint/`; the training-curve panel still needs a console log that is not included (stated in INSTALL.md) |
| Cognition JSON stage 3 "localize: object bearing + distance via color/shape CV + ego-depth" (recorded files) | A fixed string written by `cortex_eagle.py`; that path was not run. | `code/navigation/cortex_eagle.py:76-84` (stage strings; corrected text at `:80`); `navigate.py:152` | recorded files left unchanged and annotated in `videos/VIDEOS.md`; the string in the code is corrected |
| "Checkpoint = brax params + env config" (earlier report §4) | The pickle holds `{step, params}` only; the `_cfg.pkl` written by `train_walk.py` is not in the repository. | `analysis/evidence.py: checkpoint_summary` | "step counter + brax parameters incl. the observation normaliser" |

## PARTLY

| claim | finding | evidence | CV-safe rewording |
|---|---|---|---|
| "3/3 target reach" | True for three seeded scenes, one run each, with keyword target selection and a ground-truth target position. "Reached" = Navigator stop condition (planar distance ≤ 0.55 m) without a fall. Seed 2 ends at 0.54 m. A CPU re-run reproduces all three. | `videos/VIDEOS.md`; `results/demo_rerun_cpu/episodes.json`; `code/navigation/action_module.py:46-52`; `navigate.py:233` | "reached the instructed object (≤ 0.55 m) in all three recorded simulation scenes without falling" |
| "Grounding correct in 3 scenes" | The keyword rule picked the named object in all three. Each instruction names a unique colour + shape, so this tests a string match, not perception. | `videos/*.cognition.json`; `tests/test_grounding.py` | "a keyword rule selected the instructed object in all three scenes" |
| "Walking emerged at 41 M steps" | Supported by the CPU re-evaluation (first checkpoint that walks: 41.3 M, 0.32 m/s, min torso 0.73 m) and by the evaluation reward crossing zero there. The earlier report's intermediate velocities (e.g. −0.39 m/s at 31 M) came from training logs not in the repository; the re-evaluation gives −0.27 m/s. | `results/walk_sweep_cpu/walk_sweep.json`; `checkpoint/walk_t4_metrics.jsonl:5` | "the first checkpoint that walks in the raw environment is at 41.3 M steps" |
| "8192 parallel environments" | `train_walk.py` default and the documented command; the run's own configuration file was not kept. | `code/locomotion/train_walk.py:32` | keep, as "trained with 8192 parallel MJX environments (training command)" |
| "Trained on an RTX 6000 Ada" / engineering log items (cuSolver on A6000, 50× slow-down from rendering, +19 marching-in-place reward, falls at ~1.6 s) | Author's engineering record; no logs in the repository. | `docs/report.md` §5 | state as engineering notes, not results |
| "Trained from scratch with RL, no demonstrations" | The step-0 checkpoint exists (eval reward −6.30), `--restore` defaults to none, no dataset exists. Whether the final run used `--restore` is not recorded, but the checkpoint series starts at step 0. | `checkpoint/walk_t4_step0000000000.pkl`; `train_walk.py:34`; `dataset/NOTE.md` | keep |
| "Domain randomisation was the single most important ingredient" | Plausible engineering finding; no ablation run is in the repository. | — | "training without domain randomisation did not produce walking in our runs" |

## SUPPORTED

| claim | evidence |
|---|---|
| 154.8 M (154,828,800) training steps | `step` field of `checkpoint/walk_t4_latest.pkl` (read by `analysis/evidence.py`), `checkpoint/walk_t4_metrics.jsonl:16`; `walk_t4_latest.pkl` is byte-identical to `walk_t4_step0154828800.pkl` |
| 16 checkpoints every 10,321,920 steps | `checkpoint/walk_t4_step*.pkl`, `step` fields |
| 77 min of training (wall-clock incl. compilation), final evaluation reward 9.29 | `checkpoint/walk_t4_metrics.jsonl:16` (`elapsed_min` 77.12, `reward` 9.29) |
| ~2.3 M steps/min steady state, ~2.0 M steps/min overall | `walk_t4_metrics.jsonl:2,16` |
| final policy 0.519 m/s for a 0.6 m/s command over 4 s, net 2.07 m, min torso 0.745 m | `checkpoint/walk_t4_rollout.json` (written by `code/locomotion/train_walk.py:118-154` (`save_ckpt`, raw environment, 200 steps)) |
| final policy 0.52 m/s, 2.59 m, min torso 0.74 m over 5 s (CPU re-evaluation) | `results/walk_sweep_cpu/walk_sweep.json` |
| closed-loop final distances 0.39 / 0.41 / 0.54 m, torso 0.75 / 0.76 / 0.75 m | `videos/VIDEOS.md`; reproduced in `results/demo_rerun_cpu/episodes.json` |
| video frame counts 630 / 387 / 668 (2 s + navigation + 2 s at 50 fps) | MP4 `stsz` boxes (`analysis/evidence.py: mp4_frame_count`) |
| 103-D observation, 29-D action (PD targets `default_pose + 0.5·a`), 50 Hz control, 500 Hz physics | checkpoint layer shapes (103→…→58 = 2×29); `navigate.py:83-84,110,225` (`action_scale`, `ctrl_dt`, `SIM_DT`) |
| policy 103→512→256→128→58, value 216→512→256→128→1, observation normaliser in `params[0]` | `analysis/evidence.py: checkpoint_summary` |
| physics only, no kinematic playback | `navigate.py:225-233` (frames are rendered from `MjData` after `mj_step`) |
| Eagle embedding 286 tokens × 2048-d was computed for each demo | `videos/*.cognition.json` |
| no collected dataset | `dataset/NOTE.md`; no data files; PPO on-line in MJX |
| scene objects of seeds 0–2 are regenerated exactly from the seed | `tests/test_scene_gen.py` |
| checkpoints, videos (mp4 + gif), ego frames and cognition JSON are present | `checkpoint/` (16 + latest pickles, metrics, rollout json + mp4), `videos/` (3 × 4 files) |

## CV-safe sentences

1. "Trained a velocity-conditioned PPO locomotion policy for the Unitree G1 from scratch in
   MuJoCo Playground / MJX (154.8 M environment steps, 77 min on one GPU); it walks at 0.52 m/s
   for a 0.6 m/s command without falling."
2. "Coupled the policy to a hand-written Navigator that turns a target position into velocity
   commands; in three seeded simulation scenes the G1 reached the instructed object (within
   0.55 m) without falling, using a keyword rule for target selection and the simulator's
   object position."
3. "Diagnosed why the trained policy fell when deployed in a new scene: missing observation
   normalisation and a gravity term computed in the world frame instead of the pelvis-IMU frame
   (only visible when the robot turned); also found that the training wrapper's auto-reset hid
   falls during evaluation."
4. "Released code, checkpoints, videos and a claims check that recomputes every reported
   number from the committed files."
5. (If GR00T must be mentioned) "Ran the GR00T N1.6 Eagle vision-language encoder on the
   robot's ego frame as a placeholder for learned grounding; it was not used for the decision."
