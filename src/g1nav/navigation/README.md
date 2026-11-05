# g1nav.navigation — target selection, Navigator, closed loop

**Status:** the code that produced the three demo episodes. Target selection is a keyword rule and the target
position is simulator ground truth; the Eagle embedding is logged only.

**Inputs → outputs:** instruction + scene labels → target label (`grounding.py`); target world (x, y) + odometry →
`(vx, vy, wz)` (`action_module.py`); checkpoint + seed + target → video + `NAV seed=.. reached=.. final_dist_to_goal=..`
line (`navigate.py`).

**Entry points**
```bash
python -m g1nav.navigation.pipeline --seed 0 --instruction "go to the orange cylinder"    # one episode, outputs/
python -m g1nav.navigation.navigate --ckpt results/checkpoints/walk_t4_latest.pkl --seed 0 --target "orange cylinder"
python -m g1nav.navigation.rerun_demo_episodes --cpu --outdir results/demo_rerun_cpu     # the 3 demo episodes
python -m g1nav.navigation.cortex_eagle --image ego.png --instruction "..." --labels "a,b,c"  # Eagle env + GPU
python -m g1nav.navigation.action_module                                                 # kinematic self-test
```

**Layout**

| module | role |
|---|---|
| `grounding.py` | keyword rule (`keywordTarget`) — the whole "grounding" of the demos |
| `cortex_eagle.py` | GR00T N1.6 Eagle embedding of the ego frame (logged) + keyword rule → cognition JSON |
| `action_module.py` | Navigator: turn-then-go controller, slew limits, stop radius 0.55 m |
| `integrated_scene.py` | MuJoCo Playground G1 + scene objects/walls/cameras |
| `navigate.py` | closed loop: 103-D observation, policy, physics with the training solver options, video |
| `pipeline.py` | seed + instruction → scene render → target → navigate (was `g1nav_run.py`) |
| `rerun_demo_episodes.py` | re-measures the demo episodes (`results/demo_rerun_cpu/`) |

**Verified here:** the three demo episodes re-run on CPU after the refactor give byte-identical videos and the same
`NAV` lines as before it (docs/refactor_2026-11.md); `cortex_eagle` needs torch + a CUDA GPU and was not run.

**Provenance:** author's code; observation layout and default pose follow Playground's G1 Joystick environment.
