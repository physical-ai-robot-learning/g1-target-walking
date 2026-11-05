"""Re-run the three recorded demo episodes and store what navigate.py reports.

Runs `code/navigation/navigate.py` (same arguments as `render_n16_videos.sh`:
the trained checkpoint, 12 s budget, keyword-selected target) for seeds 0-2 and
writes, per episode, the raw log and one JSON summary parsed from the `NAV ...`
line (reached, final planar distance to the target, torso height, frames).

The target label is taken from the recorded `videos/*.cognition.json`, so this
re-measures the control side (Navigator + PPO policy + MuJoCo physics) only.

    MUJOCO_GL=cgl python scripts/rerun_demo_episodes.py --cpu \
        --outdir results/demo_rerun_cpu          # macOS, CPU
    MUJOCO_GL=egl python scripts/rerun_demo_episodes.py --outdir results/demo_rerun_gpu

Needs the pinned simulation stack (jax 0.5.3, brax 0.14.2, mujoco 3.4.0,
playground 0.1.0) and the G1 assets that MuJoCo Playground expects.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

from g1nav.paths import LATEST_CHECKPOINT, MEDIA_DIR, REPO_ROOT, RESULTS_DIR

ROOT = str(REPO_ROOT)
EPISODES = [
    (0, "go to the orange cylinder"),
    (1, "go to the red cube"),
    (2, "go to the purple cylinder"),
]
NAV_RE = re.compile(
    r"NAV seed=(\d+) target='([^']*)' reached=(True|False) "
    r"final_dist_to_goal=([\d.]+)m base_z=([\d.]+) frames=(\d+)")


def slug(instruction: str) -> str:
    return instruction.replace(" ", "-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(str(RESULTS_DIR), "demo_rerun_cpu"))
    ap.add_argument("--ckpt", default=str(LATEST_CHECKPOINT))
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--cpu", action="store_true", help="run the policy with JAX on CPU")
    ap.add_argument("--keep_video", action="store_true", help="keep the rendered mp4s")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for seed, instruction in EPISODES:
        cog_path = os.path.join(str(MEDIA_DIR), f"seed{seed}_{slug(instruction)}.cognition.json")
        with open(cog_path) as f:
            target = json.load(f)["vlm_target"]   # keyword-matched label (see grounding.py)
        vid = os.path.join(args.outdir, f"nav_seed{seed}.mp4")
        cmd = [sys.executable, "-m", "g1nav.navigation.navigate",
               "--ckpt", args.ckpt, "--seed", str(seed), "--target", target,
               "--seconds", str(args.seconds), "--out", vid]
        if args.cpu:
            cmd.append("--cpu")
        env = dict(os.environ)
        if args.cpu:
            env["G1NAV_CPU"] = "1"
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
        log = r.stdout + r.stderr
        log_lines = [ln for ln in log.splitlines() if ln.startswith(("NAV ", "FELL", "[navigate]"))]
        with open(os.path.join(args.outdir, f"nav_seed{seed}.log"), "w") as f:
            f.write("\n".join(ln.replace(args.outdir, "<outdir>") for ln in log_lines) + "\n")
        m = NAV_RE.search(log)
        if not m:
            print(f"[rerun] seed {seed}: no NAV line (rc={r.returncode})\n{log[-2000:]}")
            return 1
        row = {"seed": seed, "instruction": instruction, "target": m.group(2),
               "reached": m.group(3) == "True", "final_dist_m": float(m.group(4)),
               "torso_z_m": float(m.group(5)), "frames": int(m.group(6)),
               "fell": "FELL" in log}
        rows.append(row)
        print(f"[rerun] {row}")
        if not args.keep_video and os.path.exists(vid):
            os.remove(vid)

    with open(os.path.join(args.outdir, "episodes.json"), "w") as f:
        json.dump({"checkpoint": os.path.relpath(args.ckpt, ROOT),
                   "seconds": args.seconds, "jax_cpu": args.cpu, "episodes": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
