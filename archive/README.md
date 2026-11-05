# archive/

Code that the current pipeline does not use, kept verbatim for the record (not run, not tested; it imports the old
`code/` layout and is only byte-compiled in CI). Nothing here backs a number in the README or the report.

| file | what it is | why it is not used | archived |
|---|---|---|---|
| `planner.py` | A privileged finite-state planner that drives the walking policy from ground-truth poses for all three instruction tiers (go-to / follow / turn-after-pass). It was meant to label joint-target trajectories for training a pixels + text → joints student by behaviour cloning. | The behaviour-cloning student was never built; the demos use `src/g1nav/navigation/action_module.py` (Navigator) instead. Its smoke test imports `scene_gen` / `instructions` from the old `code/envs/` and uses their pre-2026-11 attribute names. | before 2026-11 (was `legacy/`) |
| `color_detect.py` | HSV colour masking + shape heuristics on the ego RGB frame; with ego depth this would give the bearing/distance for `Navigator.lockGoalFromEgo`. | Never used in the recorded demos and never evaluated; target positions are simulator ground truth. | 2026-11 (was `code/navigation/color_detect.py`) |
| `run_cog_n16.sh` | Pod script: colour-detect the objects in an ego frame, then run the Eagle cognition on them. | Not part of any recorded run, and broken: it calls `color_detect.detect(path)`, which does not exist (the module has `detect_all(rgb)` / `locate_target(rgb, label)`). | 2026-11 (was `code/navigation/run_cog_n16.sh`) |

Superseded design choices that are still visible in the current code (the `--use_vlm` flag, the edited reward-scale
defaults of `train_walk.py`) are described in the README, "Current vs superseded".
