"""Instruction templates and success criteria for G1Nav scenes.

An instruction is a free-form English command tied to a scene. Each template knows how to render text (with
paraphrases) for a target object and how to decide success from a trajectory of robot poses. Everything is
deterministic given (scene, rng), so instructions are regenerable from seeds. The templates span three tiers:

  TIER 1 (single goal):    "go to the <obj>"   / "approach the <obj>"
  TIER 2 (tracking):       "follow the <obj>"  (static scene -> same success rule as go-to, kept distinct)
  TIER 3 (compositional):  "go straight and turn <dir> after passing the <obj>"

The recorded demos use only the go-to tier (seeds 0-2); the other tiers were written for the unused
behaviour-cloning plan (archive/planner.py).

Input: a SceneSpec and a numpy Generator. Output: an Instruction (text, target, success rule).

    python -m g1nav.envs.instructions          # print the instruction of seeds 0-5
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from g1nav.envs.scene_gen import SceneSpec, sampleScene

REACH_RADIUS = 0.6     # within this xy distance of the target counts as "reached"
PASS_RADIUS = 0.7      # within this the object counts as "passed"
TURN_MIN_DEG = 60      # heading change after passing that counts as a turn
TIERS = ["goto", "follow", "turn"]
TIER_PROBABILITIES = [0.5, 0.2, 0.3]
DEMO_SEEDS = 6
DEMO_RNG_OFFSET = 1000  # instruction rng seed = DEMO_RNG_OFFSET + scene seed (as in docs/dataset.md)


def _xy(p):
    return np.asarray(p[:2], dtype=float)


def distXy(a, b) -> float:
    """Planar distance between two poses / points (first two components)."""
    return float(np.linalg.norm(_xy(a) - _xy(b)))


@dataclass(frozen=True)
class Instruction:
    """A rendered instruction. mTargetXy is the ground-truth target position (privileged, for success only)."""

    mTemplateId: str
    mText: str
    mTargetLabel: str
    mTargetXy: tuple
    mParams: dict           # template-specific, e.g. {"turn": "right"}
    mSuccessFn: Callable    # (traj, objects, instruction) -> bool; traj is a list of robot (x, y, yaw)

    def succeeded(self, traj, objects) -> bool:
        return bool(self.mSuccessFn(traj, objects, self))


def _successGoto(traj, objects, instr: Instruction) -> bool:
    """Reached if any pose along the trajectory comes within REACH_RADIUS of the target."""
    tTarget = instr.mTargetXy
    return any(distXy(tPose, tTarget) <= REACH_RADIUS for tPose in traj)


def _successTurnAfterPass(traj, objects, instr: Instruction) -> bool:
    """(1) pass near the object, then (2) change heading by more than TURN_MIN_DEG in the commanded direction."""
    tTarget = instr.mTargetXy
    tTurn = instr.mParams["turn"]  # "left" or "right"
    tPassIndex = next((tI for tI, tPose in enumerate(traj) if distXy(tPose, tTarget) <= PASS_RADIUS), None)
    if tPassIndex is None or tPassIndex >= len(traj) - 1:
        return False
    tYaw0 = traj[tPassIndex][2]
    tYaw1 = traj[-1][2]
    tDeltaYaw = np.arctan2(np.sin(tYaw1 - tYaw0), np.cos(tYaw1 - tYaw0))  # wrapped
    if tTurn == "left":
        return tDeltaYaw > np.deg2rad(TURN_MIN_DEG)
    return tDeltaYaw < -np.deg2rad(TURN_MIN_DEG)


GOTO_PARAPHRASES = [
    "go to the {obj}",
    "walk to the {obj}",
    "approach the {obj}",
    "head toward the {obj}",
    "move to the {obj}",
]
FOLLOW_PARAPHRASES = [
    "follow the {obj}",
    "go follow the {obj}",
    "track the {obj}",
]
TURN_PARAPHRASES = [
    "go straight and turn {turn} after passing the {obj}",
    "walk forward, then turn {turn} once you pass the {obj}",
    "pass the {obj} and then turn {turn}",
]


def sampleInstruction(scene: SceneSpec, rng: np.random.Generator) -> Instruction:
    """Pick a template tier, a paraphrase and a target object for the scene, deterministically given rng."""
    tTarget = scene.mObjects[int(rng.integers(len(scene.mObjects)))]
    tLabel = tTarget.label
    tTargetXy = (tTarget.mX, tTarget.mY)

    tTier = rng.choice(TIERS, p=TIER_PROBABILITIES)
    if tTier == "goto":
        tText = rng.choice(GOTO_PARAPHRASES).format(obj=tLabel)
        return Instruction("goto", tText, tLabel, tTargetXy, {}, _successGoto)
    if tTier == "follow":
        tText = rng.choice(FOLLOW_PARAPHRASES).format(obj=tLabel)
        # static scene -> follow reduces to reach-and-stay; reuse the go-to rule
        return Instruction("follow", tText, tLabel, tTargetXy, {}, _successGoto)
    tTurn = rng.choice(["left", "right"])
    tText = rng.choice(TURN_PARAPHRASES).format(obj=tLabel, turn=tTurn)
    return Instruction("turn_after_pass", tText, tLabel, tTargetXy, {"turn": tTurn}, _successTurnAfterPass)


def main(argv: list[str] | None = None) -> int:
    for tSeed in range(DEMO_SEEDS):
        tScene = sampleScene(tSeed)
        tInstr = sampleInstruction(tScene, np.random.default_rng(DEMO_RNG_OFFSET + tSeed))
        print(f"seed={tSeed:2d} [{tInstr.mTemplateId:15s}] target={tInstr.mTargetLabel:16s} "
              f"@({tInstr.mTargetXy[0]:+.2f},{tInstr.mTargetXy[1]:+.2f}) :: \"{tInstr.mText}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
