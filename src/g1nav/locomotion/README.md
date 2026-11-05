# g1nav.locomotion — PPO walking policy

**Status:** produced `results/checkpoints/walk_t4_*` (final run: Playground default reward scales, see README).

**Inputs → outputs:** Playground `G1JoystickFlatTerrain` (MJX) → brax PPO → checkpoints `{"step", "params"}`, metrics
JSONL, raw-env rollout; checkpoint(s) → raw-environment evaluation rows (`walk_sweep.json`).

**Entry points**
```bash
python -m g1nav.locomotion.train_walk --timesteps 500000000 --num_envs 8192 \
    --termination -100 --track_lin 1.0 --stand_still -1.0 --num_evals 50 --out results/checkpoints/walk_t4   # GPU
G1NAV_OUT_DIR=results/walk_sweep_cpu python -m g1nav.locomotion.walk_eval --sweep results/checkpoints 'walk_t4_step*.pkl'
python -m g1nav.locomotion.rollout_walk --ckpt results/checkpoints/walk_t4_latest.pkl --out roll.mp4
```

**Layout:** `train_walk.py` (PPO with domain randomisation, checkpoint + raw-env rollout at every evaluation),
`walk_eval.py` (raw env, no auto-reset, command vx = 0.6 m/s, 250 steps), `rollout_walk.py` (video for the training
dashboard; builds the network without the observation normaliser — recorded as a suspected bug).

**Verified here:** the 16-checkpoint CPU sweep after the refactor is byte-identical to `results/walk_sweep_cpu/walk_sweep.json`;
a one-iteration CPU training run (`--timesteps 1 --num_envs 8 --num_evals 1`) gives the same rewards, rollout JSON and
parameters before and after. Full-scale training needs a GPU and was not run.

**Provenance:** environment, reward and PPO hyperparameters from MuJoCo Playground; training with brax PPO.
