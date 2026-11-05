"""One-shot G1Nav pipeline: seed + instruction -> keyword target -> closed-loop walk -> video (was g1nav_run.py).

Runs the full closed loop with repository-relative paths (g1nav.paths), in three stages:
    1) build the integrated scene for the seed and render the start ego frame
    2) target label picked by keyword match against the scene labels (cortex_eagle also logs the GR00T N1.6 Eagle
       embedding of the ego frame when the encoder is present; without it the same keyword rule runs here)
    3) navigate walks the trained PPO policy to that target (physics only), recording an ego || third-person video
       with 2 s pre/post-roll.

Prints machine-readable lines (STAGE=..., OBJECTS=..., TARGET=..., NAV ..., RESULT_JSON=..., VIDEO=..., DONE) that the
interactive dashboard parses. Needs the simulation stack; the Eagle stage is skipped gracefully when the encoder
directory is absent.

    python -m g1nav.navigation.pipeline --seed 0 --instruction "go to the orange cylinder"
    #   --ckpt   default results/checkpoints/walk_t4_latest.pkl
    #   --eagle  default $G1NAV_EAGLE or assets/eagle-n16 (GR00T N1.6 Eagle backbone, not in the repository)
    #   --outdir default outputs/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from g1nav.paths import EAGLE_DIR, LATEST_CHECKPOINT, OUTPUTS_DIR

RENDER_HEIGHT, RENDER_WIDTH = 240, 320
DEFAULT_SECONDS = 14.0


def renderScene(seed: int, egoPath: str) -> list[str]:
    """Build the integrated scene, save the start ego frame to egoPath and return the object labels."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image

    from g1nav.navigation import integrated_scene

    tPath, tScene = integrated_scene.buildIntegratedXml(seed)
    tModel = mujoco.MjModel.from_xml_path(tPath)
    tData = mujoco.MjData(tModel)
    mujoco.mj_resetDataKeyframe(tModel, tData, 0)
    mujoco.mj_forward(tModel, tData)
    tRenderer = mujoco.Renderer(tModel, RENDER_HEIGHT, RENDER_WIDTH)
    tRenderer.update_scene(tData, camera="ego")
    Image.fromarray(tRenderer.render()).save(egoPath)
    return [tObj.label for tObj in tScene.mObjects]


def selectTarget(egoPath: str, instruction: str, labels: list[str], eagleDir: str, resultPath: str):
    """Target label via cortex_eagle (Eagle embedding logged + keyword rule); keyword rule alone without Eagle.

    Returns (target_label, log_text) and writes the cognition JSON to resultPath.
    """
    if eagleDir and os.path.isdir(eagleDir):
        tEnv = dict(os.environ, G1NAV_EAGLE=eagleDir)
        tCmd = [sys.executable, "-m", "g1nav.navigation.cortex_eagle",
                "--image", egoPath, "--instruction", instruction,
                "--labels", ",".join(labels), "--result", resultPath]
        tRun = subprocess.run(tCmd, capture_output=True, text=True, env=tEnv)
        if os.path.exists(resultPath):
            try:
                with open(resultPath) as tFile:
                    return json.load(tFile).get("vlm_target"), tRun.stdout + tRun.stderr
            except json.JSONDecodeError:
                pass
    # fallback: colour + shape keyword overlap with the instruction (no extra checkpoint)
    from g1nav.navigation.grounding import keywordTarget
    tBest = keywordTarget(instruction, labels)
    with open(resultPath, "w") as tFile:
        json.dump({"vlm_target": tBest, "labels": labels,
                   "note": "Eagle encoder not found; used keyword grounding"}, tFile, indent=2)
    return tBest, "eagle encoder not found -> keyword grounding"


def main(argv: list[str] | None = None) -> int:
    tParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    tParser.add_argument("--seed", type=int, default=0)
    tParser.add_argument("--instruction", default="go to the orange cylinder")
    tParser.add_argument("--ckpt", default=str(LATEST_CHECKPOINT))
    tParser.add_argument("--eagle", default=str(EAGLE_DIR))
    tParser.add_argument("--outdir", default=str(OUTPUTS_DIR))
    tParser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    tArgs = tParser.parse_args(argv)
    os.makedirs(tArgs.outdir, exist_ok=True)

    tEgo = os.path.join(tArgs.outdir, f"ego_seed{tArgs.seed}.png")
    tResult = os.path.join(tArgs.outdir, f"cortex_seed{tArgs.seed}.json")
    tVideo = os.path.join(tArgs.outdir, f"nav_seed{tArgs.seed}.mp4")

    print("STAGE=scene", flush=True)
    tLabels = renderScene(tArgs.seed, tEgo)
    print("OBJECTS=" + ",".join(tLabels), flush=True)

    print("STAGE=cognition", flush=True)
    tTarget, _ = selectTarget(tEgo, tArgs.instruction, tLabels, tArgs.eagle, tResult)
    print("TARGET=" + (tTarget or "?"), flush=True)

    print("STAGE=navigate", flush=True)
    tCmd = [sys.executable, "-m", "g1nav.navigation.navigate",
            "--ckpt", tArgs.ckpt, "--seed", str(tArgs.seed),
            "--target", tTarget or tLabels[0], "--seconds", str(tArgs.seconds),
            "--out", tVideo]
    tRun = subprocess.run(tCmd, capture_output=True, text=True)
    for tLine in (tRun.stdout + tRun.stderr).splitlines():
        if tLine.startswith("NAV ") or tLine.startswith("FELL"):
            print(tLine, flush=True)

    print("RESULT_JSON=" + tResult, flush=True)
    print("VIDEO=" + tVideo, flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
