# videos/: the three recorded demo episodes

Each episode is one closed-loop run of the pipeline as it exists in the code:
**keyword target selection (target position from the generated scene) → Navigator → PPO
walking policy → MuJoCo physics**. Each clip shows the third-person view (left) and the ego
camera (right): 2 s standing, walking, then 2 s standing at the target. 50 frames per second.

Each episode has a matched set of files with the same `seed{N}_<instruction>` prefix:
`*.mp4` (video), `*.gif` (downsampled copy used in the README), `*.cognition.json` (the
keyword-selected target, stored under the historical key `vlm_target`, plus the Eagle
embedding statistics) and `*.ego.png` (the start ego frame).

| file prefix | instruction | scene objects (seed) | chosen target (keyword match) | reached | final dist | torso z | frames |
|---|---|---|---|:---:|---:|---:|---:|
| `seed0_go-to-the-orange-cylinder` | "go to the orange cylinder" | orange cylinder, yellow ball, purple cube | orange cylinder | yes | 0.39 m | 0.75 m | 630 |
| `seed1_go-to-the-red-cube` | "go to the red cube" | red cube, purple cube, yellow cone | red cube | yes | 0.41 m | 0.76 m | 387 |
| `seed2_go-to-the-purple-cylinder` | "go to the purple cylinder" | purple cylinder, purple ball, orange cone | purple cylinder | yes | 0.54 m | 0.75 m | 668 |

"Reached" = the Navigator's stop condition (planar distance ≤ 0.55 m) fired before the 12 s
budget without a fall; the final distance is measured after the 2 s post-roll. The distances
and torso heights come from the `NAV` line printed by `navigate.py`; the original GPU logs are
not in the repository, and a CPU re-run reproduces them (`results/demo_rerun_cpu/`).

Notes on the cognition JSON files:

- `stages["3"]` reads "localize: object bearing + distance via color/shape CV + ego-depth
  (downstream)". This was a fixed descriptive string written by `cortex_eagle.py` at the time;
  that localisation path was **not** used. The target position came from the generated scene.
  The string has since been corrected in the code; the recorded files are left unchanged.
- The Eagle embedding (286 tokens × 2048-d) is logged and does not affect the target.

Regenerate an episode (GPU machine with the pod layout, or the portable runner):

```bash
bash code/navigation/run_pipeline_n16.sh 0 "go to the orange cylinder"
python code/navigation/g1nav_run.py --seed 0 --instruction "go to the orange cylinder"
```
