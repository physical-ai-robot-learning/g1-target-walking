# dataset/ — there is no collected dataset, by design

This project **collects no training data**, because nothing in the pipeline is
trained on a dataset:

- **Grounding (what/where).** The instructed target is picked by keyword
  overlap between the instruction and the scene's object labels, and its
  position is taken from the generated scene in the recorded demos (see the
  README's "Scope and limitations"). The frozen GR00T N1.6 Eagle encoder is run
  on the ego frame but not trained. Nothing here is trained, so there is no
  grounding dataset.
- **Control (how to move).** The walking policy is trained by **on-policy RL
  (PPO) entirely inside the MuJoCo (MJX) simulator** — `G1JoystickFlatTerrain`,
  8192 parallel environments generating transitions on the fly each step. There
  is no offline/behavior-cloning dataset; the "data" exists only transiently
  inside the RL rollout buffers during training and is never stored.

So there is no training dataset to ship or regenerate; what matters is the
evaluation setup below.

## What *is* deterministic and regenerable: the evaluation suite

The only regenerable artifacts are the **evaluation scenes + instructions**,
produced from integer seeds (no large assets shipped):

```bash
# a scene (3 colored objects, deterministic from the seed)
python code/envs/scene_gen.py --seed 0 --validate

# an instruction for that scene (go-to / follow / turn-after-pass tiers)
cd code/envs && python -c "import numpy as np, instructions as I; \
from scene_gen import sample_scene; \
s = sample_scene(0); print(I.sample_instruction(s, np.random.default_rng(1000)).text)"
```

- `code/envs/scene_gen.py`  — seed → exact scene (objects, poses, cameras, MJCF).
- `code/envs/instructions.py` — seed → instruction text + success criterion.

The recorded demo episodes use seeds **0, 1 and 2** (extend the range to test
more). Because everything is seeded, anyone can regenerate the identical scenes
without image/trajectory files being shipped.

## Where the real results live

- `checkpoint/walk_t4_latest.pkl` — the trained controller (the only learned weights).
- `code/locomotion/walk_eval.py` — truthful raw-env evaluation (forward velocity,
  torso height, net distance; no AutoReset masking).
- `videos/` — closed-loop instruction-following episodes (ego ∥ third-person).
