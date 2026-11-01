# legacy/

Code from an earlier plan that the current pipeline does not use. Kept for the record.

| file | what it is | why it is not used |
|---|---|---|
| `planner.py` | A privileged finite-state planner that drives the walking policy from ground-truth poses for all three instruction tiers (go-to / follow / turn-after-pass). It was meant to label joint-target trajectories for training a pixels + text → joints student by behaviour cloning. | The behaviour-cloning student was never built; the demos use `code/navigation/action_module.py` (Navigator) instead. Its smoke test imports `scene_gen` and `instructions` from `code/envs/`. |
