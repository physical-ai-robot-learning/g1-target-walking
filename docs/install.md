# Installation and running

`scripts/install.sh` picks one of two set-ups:

| machine | what you get | where the pipeline runs |
|---|---|---|
| NVIDIA GPU (developed on an Ada-generation GPU) | the full stack | locally |
| no GPU | the dashboards | on a remote GPU machine over SSH |

The closed-loop demo (`navigate.py`) and the checkpoint sweep (`walk_eval.py`) also run on a
CPU-only machine with the pinned simulation stack (see [Re-measuring on CPU](#re-measuring-on-cpu)).

## 1. Install

```bash
bash scripts/install.sh           # auto-detects a GPU; force with: bash scripts/install.sh gpu|dashboard
source .venv/bin/activate
```

This creates `.venv`, installs the pinned `configs/envs/requirements-gpu.txt` or `configs/envs/requirements-dashboard.txt`
(the versions the results were produced with) and then the `g1nav` package itself (`pip install -e .`).
Without the script: `pip install -e ".[sim,dashboard]"` (extras defined in `pyproject.toml`); for the checks only,
`pip install -e ".[test]"`.
The GR00T N1.6 Eagle encoder is optional and logging-only (not needed for any reported result); to
reproduce the logged embeddings, also install `configs/envs/requirements-eagle.txt` (`WITH_EAGLE=1 bash scripts/install.sh gpu`).

## 2. Dashboards

```bash
bash scripts/run_dashboard.sh
```

- **http://localhost:8502, interactive.** Pick a scene seed and **Load scene** (ego view and the
  objects in that arena), type an instruction about one of them (e.g. `go to the red cube`) and
  **Run**: the target is chosen by keyword match against the scene labels (the GR00T N1.6 Eagle
  encoder, if present, runs on the ego frame and its embedding is logged), the PPO policy walks
  there in MuJoCo, and the ego ∥ third-person video comes back. The sidebar shows the mode
  (LOCAL / REMOTE).
- **http://localhost:8501, results.** Replays the recorded demo videos, the final-checkpoint
  rollout numbers and the cognition JSON from `media/` and `results/checkpoints/` (no GPU). The
  training-curve panel needs the training console log, which is not in the repository.

### Remote mode (laptop → GPU machine)

The remote machine needs this repository at `/workspace` (or `$G1NAV_POD_WORKSPACE`) and the checkpoint at
`/workspace/ckpt/walk_t4_latest.pkl` (see `scripts/pod/pod_bootstrap.sh` and
`scripts/pod/setup_ada.sh`).

```bash
export G1NAV_POD_HOST=root@<host>    # required in remote mode (no default host)
export G1NAV_POD_PORT=<port>
export G1NAV_POD_KEY=~/.ssh/id_ed25519
export G1NAV_POD_WORKSPACE=/workspace  # repository location on the remote machine (default)
bash scripts/run_dashboard.sh interactive
```

Force a mode with `export G1NAV_MODE=local` (or `remote`).

## 3. Command line

```bash
# scene seed + instruction → keyword target → walk → video in outputs/
python -m g1nav.navigation.pipeline --seed 0 --instruction "go to the orange cylinder"
```

## 4. Training and evaluation (GPU)

```bash
python -m g1nav.locomotion.train_walk --timesteps 500000000 --num_envs 8192 \
    --termination -100 --track_lin 1.0 --stand_still -1.0 --num_evals 50 \
    --out results/checkpoints/walk_t4

# raw-environment evaluation (no auto-reset); outputs go to $G1NAV_OUT_DIR (default /workspace/ckpt)
G1NAV_OUT_DIR=results python -m g1nav.locomotion.walk_eval results/checkpoints/walk_t4_latest.pkl
G1NAV_OUT_DIR=results python -m g1nav.locomotion.walk_eval --sweep results/checkpoints 'walk_t4_step*.pkl'
```

## Re-measuring on CPU

The numbers in `results/` were produced on a CPU-only Mac (Apple silicon, Python 3.11):

```bash
uv venv -p 3.11 simvenv
uv pip install -p simvenv/bin/python "jax==0.5.3" "jaxlib==0.5.3" "brax==0.14.2" "flax==0.10.6" \
    "mujoco==3.4.0" "mujoco-mjx==3.4.0" "playground==0.1.0" imageio imageio-ffmpeg pillow
uv pip install -p simvenv/bin/python -e .        # the g1nav package (python -m g1nav....)
# MuJoCo Playground loads the G1 meshes from <site-packages>/mujoco_menagerie; the vendored
# copy can be linked there:
mkdir -p simvenv/lib/python3.11/site-packages/mujoco_menagerie
ln -s "$PWD/assets/mujoco_menagerie/unitree_g1" simvenv/lib/python3.11/site-packages/mujoco_menagerie/unitree_g1
MUJOCO_GL=cgl simvenv/bin/python -m g1nav.navigation.rerun_demo_episodes --cpu --outdir results/demo_rerun_cpu
G1NAV_OUT_DIR=results/walk_sweep_cpu JAX_PLATFORMS=cpu MUJOCO_GL=cgl \
    simvenv/bin/python -m g1nav.locomotion.walk_eval --sweep results/checkpoints 'walk_t4_step*.pkl'
```

The opt-in test `G1NAV_SIM_TESTS=1 MUJOCO_GL=cgl simvenv/bin/python -m pytest -q tests/test_sim_rerun.py`
re-runs both and compares them with the committed files (about 2 minutes on an Apple-silicon CPU).

(`MUJOCO_GL=cgl` is the macOS renderer; use `egl` on a Linux machine with a GPU, `osmesa` without.)

## Requirements and notes

- **GPU:** developed on an RTX 6000 Ada (48 GB), CUDA 12.x. Avoid Ampere (cuSolver error with
  jax 0.5.3) and Blackwell / sm_120 (no kernels in the pinned stack). RAM 16 GB+.
- **Eagle encoder (optional):** the GR00T N1.6 Eagle vision-language backbone is the only reused
  pretrained model. It is not redistributed here; obtain it from NVIDIA under its licence (see
  [third_party.md](third_party.md)) and put the backbone directory at
  `assets/eagle-n16/` or set `G1NAV_EAGLE=/path/to/eagle-n16`. The step that extracts the
  backbone directory from the full checkpoint is not included. Without it, `g1nav.navigation.pipeline` uses
  the same keyword rule (`src/g1nav/navigation/grounding.py`), so the chosen target is identical.
- **No GPU and no remote machine:** use the results dashboard (8501).

See the technical report [report/report_en.pdf](../report/report_en.pdf) for the architecture and the engineering log.
