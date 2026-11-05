"""Re-run the three recorded demo episodes and store what navigate reports (was scripts/rerun_demo_episodes.py).

Runs `python -m g1nav.navigation.navigate` with the same arguments as scripts/pod/render_n16_videos.sh (the
trained checkpoint, 12 s budget, keyword-selected target) for seeds 0-2 and writes, per episode, the navigate log
lines and one JSON summary parsed from the `NAV ...` line (reached, final planar distance to the target, torso
height, frames). The target label is taken from the recorded media/*.cognition.json, so this re-measures the
control side (Navigator + PPO policy + MuJoCo physics) only. Needs the pinned simulation stack (jax 0.5.3,
brax 0.14.2, mujoco 3.4.0, playground 0.1.0) and the G1 assets that MuJoCo Playground expects (docs/install.md).

Output: <outdir>/episodes.json and <outdir>/nav_seed<N>.log (results/demo_rerun_cpu/ holds the committed run).

    MUJOCO_GL=cgl python -m g1nav.navigation.rerun_demo_episodes --cpu --outdir results/demo_rerun_cpu   # macOS, CPU
    MUJOCO_GL=egl python -m g1nav.navigation.rerun_demo_episodes --outdir results/demo_rerun_gpu
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

from g1nav.paths import LATEST_CHECKPOINT, MEDIA_DIR, REPO_ROOT, RESULTS_DIR

EPISODES = [
    (0, "go to the orange cylinder"),
    (1, "go to the red cube"),
    (2, "go to the purple cylinder"),
]
NAV_RE = re.compile(
    r"NAV seed=(\d+) target='([^']*)' reached=(True|False) "
    r"final_dist_to_goal=([\d.]+)m base_z=([\d.]+) frames=(\d+)")
LOG_PREFIXES = ("NAV ", "FELL", "[navigate]")
LOG_TAIL_CHARS = 2000


def slug(instruction: str) -> str:
    return instruction.replace(" ", "-")


def runEpisode(seed: int, instruction: str, args) -> dict | None:
    """Run navigate for one demo episode; write its log; return the parsed summary (None if no NAV line)."""
    with open(os.path.join(str(MEDIA_DIR), f"seed{seed}_{slug(instruction)}.cognition.json")) as tFile:
        tTarget = json.load(tFile)["vlm_target"]   # keyword-matched label (see grounding.py)
    tVideo = os.path.join(args.outdir, f"nav_seed{seed}.mp4")
    tCmd = [sys.executable, "-m", "g1nav.navigation.navigate",
            "--ckpt", args.ckpt, "--seed", str(seed), "--target", tTarget,
            "--seconds", str(args.seconds), "--out", tVideo]
    if args.cpu:
        tCmd.append("--cpu")
    tEnv = dict(os.environ)
    if args.cpu:
        tEnv["G1NAV_CPU"] = "1"
    tRun = subprocess.run(tCmd, capture_output=True, text=True, env=tEnv, cwd=str(REPO_ROOT))
    tLog = tRun.stdout + tRun.stderr
    tLogLines = [tLine for tLine in tLog.splitlines() if tLine.startswith(LOG_PREFIXES)]
    with open(os.path.join(args.outdir, f"nav_seed{seed}.log"), "w") as tFile:
        tFile.write("\n".join(tLine.replace(args.outdir, "<outdir>") for tLine in tLogLines) + "\n")
    tMatch = NAV_RE.search(tLog)
    if not tMatch:
        print(f"[rerun] seed {seed}: no NAV line (rc={tRun.returncode})\n{tLog[-LOG_TAIL_CHARS:]}")
        return None
    tRow = {"seed": seed, "instruction": instruction, "target": tMatch.group(2),
            "reached": tMatch.group(3) == "True", "final_dist_m": float(tMatch.group(4)),
            "torso_z_m": float(tMatch.group(5)), "frames": int(tMatch.group(6)),
            "fell": "FELL" in tLog}
    print(f"[rerun] {tRow}")
    if not args.keep_video and os.path.exists(tVideo):
        os.remove(tVideo)
    return tRow


def main(argv: list[str] | None = None) -> int:
    tParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    tParser.add_argument("--outdir", default=os.path.join(str(RESULTS_DIR), "demo_rerun_cpu"))
    tParser.add_argument("--ckpt", default=str(LATEST_CHECKPOINT))
    tParser.add_argument("--seconds", type=float, default=12.0)
    tParser.add_argument("--cpu", action="store_true", help="run the policy with JAX on CPU")
    tParser.add_argument("--keep_video", action="store_true", help="keep the rendered mp4s")
    tArgs = tParser.parse_args(argv)
    os.makedirs(tArgs.outdir, exist_ok=True)

    tRows = []
    for tSeed, tInstruction in EPISODES:
        tRow = runEpisode(tSeed, tInstruction, tArgs)
        if tRow is None:
            return 1
        tRows.append(tRow)

    with open(os.path.join(tArgs.outdir, "episodes.json"), "w") as tFile:
        json.dump({"checkpoint": os.path.relpath(tArgs.ckpt, str(REPO_ROOT)),
                   "seconds": tArgs.seconds, "jax_cpu": tArgs.cpu, "episodes": tRows}, tFile, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
