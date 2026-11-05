"""Repository paths used by the g1nav packages, with environment-variable overrides.

Every machine-specific or repository-relative location lives here; no other module in src/ hard-codes a home or
pod path. The repository root is found from this file (src/g1nav/paths.py), which holds for an editable install
(`pip install -e .`) and for PYTHONPATH=src; set G1NAV_REPO_ROOT for any other layout.

    from g1nav.paths import CHECKPOINT_DIR, MEDIA_DIR
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("G1NAV_REPO_ROOT", Path(__file__).resolve().parents[2]))
ASSETS_DIR = REPO_ROOT / "assets"
# vendored MuJoCo Menagerie Unitree G1 model
G1_DIR = ASSETS_DIR / "mujoco_menagerie" / "unitree_g1"
# trained walking-policy checkpoints, training metrics and the training-time rollout (walk_t4_*)
CHECKPOINT_DIR = REPO_ROOT / "results" / "checkpoints"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "walk_t4_latest.pkl"
RESULTS_DIR = REPO_ROOT / "results"
# demo episode videos, start ego frames and cognition JSON
MEDIA_DIR = REPO_ROOT / "media"
# run outputs of the one-shot pipeline (git-ignored)
OUTPUTS_DIR = REPO_ROOT / "outputs"
# optional GR00T N1.6 Eagle backbone directory (third-party weights, not in the repository, git-ignored)
EAGLE_DIR = Path(os.environ.get("G1NAV_EAGLE", ASSETS_DIR / "eagle-n16"))
# checkpoint / output folder on the GPU machine used for training and the recorded runs ("pod layout")
WORKSPACE_CKPT_DIR = Path(os.environ.get("G1NAV_WORKSPACE_CKPT", "/workspace/ckpt"))
# where the results dashboard also looks for artifacts copied from the GPU machine
DASHBOARD_DATA_DIR = Path(os.environ.get("G1NAV_DATA", Path.home() / "Desktop" / "g1nav_dash"))
DASHBOARD_VIDEO_DIR = Path(os.environ.get("G1NAV_VIDEOS", Path.home() / "g1nav" / "videos"))
