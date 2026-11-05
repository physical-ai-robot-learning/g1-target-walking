# results/

**Training outputs** (`checkpoints/`, was `checkpoint/` before the 2026-11 refactor): `walk_t4_latest.pkl` + the 16
numbered `walk_t4_step*.pkl` brax parameter pickles (`{"step", "params"}`), `walk_t4_metrics.jsonl` (evaluation reward and
wall-clock per checkpoint), `walk_t4_rollout.{json,mp4}` (training-time raw-env rollout of the final checkpoint). Written by
`g1nav.locomotion.train_walk`; read without JAX by `analysis/evidence.py`.

**Re-measurements** made after the original runs, so that every closed-loop and walking number
in the README can be checked from files in the repository. Both were produced on a CPU-only
Mac (Apple silicon) with the pinned stack (jax 0.5.3, brax 0.14.2, mujoco 3.4.0,
playground 0.1.0) and the vendored G1 meshes; see docs/install.md, "Re-measuring on CPU".

| directory | produced by | contents |
|---|---|---|
| `demo_rerun_cpu/` | `python -m g1nav.navigation.rerun_demo_episodes --cpu` | `episodes.json` (reached, final distance, torso height, frame count for the three demo episodes) and the `navigate.py` log line of each run |
| `walk_sweep_cpu/` | `G1NAV_OUT_DIR=results/walk_sweep_cpu python -m g1nav.locomotion.walk_eval --sweep results/checkpoints 'walk_t4_step*.pkl'` | `walk_sweep.json` (net distance, mean local forward velocity, min torso height and verdict for each of the 16 committed checkpoints, command vx = 0.6 m/s, 250 steps) and the console output |

The original GPU runs printed the same quantities, but their logs are not in the repository.
The CPU re-run of the demo episodes matches the recorded final distances and torso heights to
the printed precision (seed 1: 386 frames vs 387 in the recorded video). The final checkpoint's
forward velocity in the sweep (0.52 m/s over 5 s) matches the training-time rollout in
`results/checkpoints/walk_t4_rollout.json` (0.519 m/s over 4 s).
