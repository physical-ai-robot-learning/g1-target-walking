"""Opt-in end-to-end checks with the simulation stack (CPU is enough, about 2 minutes).

Re-runs the three demo episodes (closed loop: Navigator + PPO policy + MuJoCo physics) and the raw-environment
evaluation of the final checkpoint, and compares them with the committed re-measurements in results/. Skipped
unless G1NAV_SIM_TESTS=1 and the pinned simulation stack (jax 0.5.3, brax 0.14.2, mujoco 3.4.0, playground 0.1.0,
see docs/install.md "Re-measuring on CPU") is importable. Not run in CI.

    G1NAV_SIM_TESTS=1 MUJOCO_GL=cgl python -m pytest -q tests/test_sim_rerun.py
"""
import json
import os
import subprocess
import sys

import pytest

from conftest import ROOT

if os.environ.get("G1NAV_SIM_TESTS") != "1":
    pytest.skip("set G1NAV_SIM_TESTS=1 to run the simulation checks", allow_module_level=True)
pytest.importorskip("mujoco_playground")

RERUN = [sys.executable, "-m", "g1nav.navigation.rerun_demo_episodes"]
WALK_EVAL = [sys.executable, "-m", "g1nav.locomotion.walk_eval"]
CKPT_LATEST = os.path.join(ROOT, "results", "checkpoints", "walk_t4_latest.pkl")
SWEEP_JSON = os.path.join(ROOT, "results", "walk_sweep_cpu", "walk_sweep.json")
EPISODES_JSON = os.path.join(ROOT, "results", "demo_rerun_cpu", "episodes.json")


def _env(**extra):
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    env.update(extra)
    return env


def test_demo_episodes_reproduce(tmp_path):
    r = subprocess.run(RERUN + ["--cpu", "--outdir", str(tmp_path)], cwd=ROOT, env=_env(),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with open(tmp_path / "episodes.json") as f:
        got = json.load(f)["episodes"]
    with open(EPISODES_JSON) as f:
        want = json.load(f)["episodes"]
    assert got == want


def test_final_checkpoint_walk_eval(tmp_path):
    r = subprocess.run(WALK_EVAL + [CKPT_LATEST], cwd=ROOT, capture_output=True, text=True,
                       env=_env(G1NAV_OUT_DIR=str(tmp_path), JAX_PLATFORMS="cpu"))
    assert r.returncode == 0, r.stdout + r.stderr
    text = r.stdout[r.stdout.index("{"):r.stdout.index("video ->")]
    got = json.loads(text)
    with open(SWEEP_JSON) as f:
        want = {row["step"]: row for row in json.load(f)}[got["step"]]
    assert {k: got[k] for k in ("step", "net_dist", "fwd_vel", "min_z", "verdict")} == \
        {k: want[k] for k in ("step", "net_dist", "fwd_vel", "min_z", "verdict")}
