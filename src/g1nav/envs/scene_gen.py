"""Seeded arena scenes for G1Nav: seed -> robot spawn + three coloured objects -> MuJoCo MJCF.

Stage 1 of the pipeline. `sampleScene(seed)` is pure numpy and deterministic: the robot starts at a wall of a
6 m x 6 m arena facing inward, and the objects are placed at random range/bearing inside its frontal field of view,
non-overlapping. `buildMjcf(scene)` wraps the vendored Menagerie Unitree G1 (assets/mujoco_menagerie/unitree_g1)
with a floor, low walls, the objects, an egocentric camera on the torso and a fixed third-person camera. The
closed-loop runs use the MuJoCo Playground G1 model instead (navigation/integrated_scene.py) with the same scene
layout, walls, objects and cameras (the XML helpers here are shared).

Input: an integer seed. Output: a SceneSpec and an MJCF XML string / file (no GPU; `mujoco` only for --validate).

    python -m g1nav.envs.scene_gen --seed 0 [--out scene.xml] [--validate]
"""
from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from g1nav.paths import ASSETS_DIR
from g1nav.paths import G1_DIR as G1_DIR_PATH

G1_DIR = str(G1_DIR_PATH)
G1_XML = os.path.join(G1_DIR, "g1.xml")

# ---- object palette -------------------------------------------------------
# colour name -> rgba
COLORS = {
    "red":    (0.85, 0.10, 0.10, 1.0),
    "yellow": (0.90, 0.85, 0.10, 1.0),
    "blue":   (0.10, 0.25, 0.85, 1.0),
    "green":  (0.10, 0.70, 0.20, 1.0),
    "orange": (0.95, 0.45, 0.05, 1.0),
    "purple": (0.55, 0.10, 0.75, 1.0),
}
SHAPES = {
    # shape_name -> (geom_type, size, z_offset_to_rest_on_floor)
    "ball":     ("sphere",   (0.09, 0, 0),        0.09),
    "cube":     ("box",      (0.08, 0.08, 0.08),  0.08),
    "cylinder": ("cylinder", (0.07, 0.12, 0),     0.12),
    "cone":     ("ellipsoid", (0.08, 0.08, 0.14),  0.14),  # MJCF has no cone; ellipsoid stand-in
}

ARENA_HALF_M = 3.0             # arena spans [-half, half] in x and y
N_OBJECTS = 3
MIN_SEPARATION_M = 0.7         # between object centres
VIEW_HALF_ANGLE_DEG = 36.0     # objects are placed inside this half-angle of the robot's heading
NEAR_M = 1.3                   # closest object range
FAR_FRACTION = 0.85            # farthest range as a fraction of the arena depth in front of the robot
MIN_RANGE_SPAN_M = 0.5         # the range interval is at least [near, near + this]
SPAWN_WALL_MARGIN_M = 0.35     # robot spawn distance from its wall
SPAWN_END_MARGIN_M = 0.8       # spawn point keeps this far from the arena corners (along the wall)
SPAWN_YAW_JITTER_RAD = 0.12
OBJECT_WALL_MARGIN_M = 0.4     # objects stay this far inside the arena
PLACEMENT_ATTEMPTS = 500       # rejection-sampling tries per object (the last try is kept if none succeeds)
WALL_THICKNESS_M = 0.05
WALL_HEIGHT_M = 0.3
DEFAULT_SIZE_TEXT = "0.05"     # size attribute when a shape has no non-zero size

# Egocentric RGBD camera on torso_link: roughly head height, wide FOV, tilted ~12 deg down toward the frontal fan.
EGO_CAMERA_XML = ('<camera name="ego" mode="fixed" pos="0.08 0 0.55" '
                  'xyaxes="0 -1 0 0.22 0 1" fovy="95"/>')


@dataclass(frozen=True)
class ObjSpec:
    """One static object of the scene; z is the height of its centre when resting on the floor."""

    mColor: str
    mShape: str
    mX: float
    mY: float
    mZ: float

    @property
    def label(self) -> str:
        return f"{self.mColor} {self.mShape}"


@dataclass(frozen=True)
class SceneSpec:
    """A generated scene: objects, arena size and the robot spawn pose (x, y, yaw) in the world frame."""

    mSeed: int
    mObjects: list = field(default_factory=list)
    mArenaHalf: float = ARENA_HALF_M
    mRobotYaw: float = 0.0
    mRobotX: float = 0.0
    mRobotY: float = 0.0

    def toDict(self) -> dict:
        """Plain dict with the original serialised keys (seed, objects[color, shape, x, y, z], arena_half, robot_*)."""
        return {"seed": self.mSeed,
                "objects": [{"color": tObj.mColor, "shape": tObj.mShape, "x": tObj.mX, "y": tObj.mY, "z": tObj.mZ}
                            for tObj in self.mObjects],
                "arena_half": self.mArenaHalf, "robot_yaw": self.mRobotYaw,
                "robot_x": self.mRobotX, "robot_y": self.mRobotY}


def sampleScene(seed: int, nMin: int = N_OBJECTS, nMax: int = N_OBJECTS, arenaHalf: float = ARENA_HALF_M,
                minSep: float = MIN_SEPARATION_M, maxViewHalfAngleDeg: float = VIEW_HALF_ANGLE_DEG,
                near: float = NEAR_M, farFrac: float = FAR_FRACTION) -> SceneSpec:
    """Deterministically sample a scene from a seed.

    1. pick a random wall and a random point along it -> robot spawn
    2. the robot faces the arena interior (toward the opposite wall), with a small random yaw jitter
    3. place the objects at random range/bearing in the frontal cone, inside the arena and non-overlapping,
       so all of them are visible at spawn and lie ahead to walk toward
    """
    tRng = np.random.default_rng(seed)
    tCount = int(tRng.integers(nMin, nMax + 1))

    tIdentities = [(tColor, tShape) for tColor in COLORS for tShape in SHAPES]
    tRng.shuffle(tIdentities)
    tIdentities = tIdentities[:tCount]

    # 1) robot on a random wall, 2) facing inward (+ jitter)
    tWall = int(tRng.integers(4))          # 0:-x, 1:+x, 2:-y, 3:+y
    tMargin = SPAWN_WALL_MARGIN_M
    tSpan = arenaHalf - SPAWN_END_MARGIN_M
    if tWall == 0:    # left wall (-x), face +x
        tRx, tRy, tBaseYaw = -arenaHalf + tMargin, float(tRng.uniform(-tSpan, tSpan)), 0.0
    elif tWall == 1:  # right wall (+x), face -x
        tRx, tRy, tBaseYaw = arenaHalf - tMargin, float(tRng.uniform(-tSpan, tSpan)), np.pi
    elif tWall == 2:  # bottom wall (-y), face +y
        tRx, tRy, tBaseYaw = float(tRng.uniform(-tSpan, tSpan)), -arenaHalf + tMargin, np.pi / 2
    else:             # top wall (+y), face -y
        tRx, tRy, tBaseYaw = float(tRng.uniform(-tSpan, tSpan)), arenaHalf - tMargin, -np.pi / 2
    tYaw = float(tBaseYaw + tRng.uniform(-SPAWN_YAW_JITTER_RAD, SPAWN_YAW_JITTER_RAD))

    # 3) objects in the frontal cone, inside the arena, non-overlapping
    tHalfView = np.deg2rad(maxViewHalfAngleDeg)
    # how far ahead objects can be placed before hitting the far wall
    tReach = (2 * arenaHalf - tMargin) * farFrac
    tObjects: list[ObjSpec] = []
    tPlaced: list[tuple[float, float]] = []
    for (tColor, tShape) in tIdentities:
        for _ in range(PLACEMENT_ATTEMPTS):
            tRange = float(tRng.uniform(near, max(near + MIN_RANGE_SPAN_M, tReach)))
            tBearing = float(tRng.uniform(-tHalfView, tHalfView))
            tX = tRx + tRange * np.cos(tYaw + tBearing)
            tY = tRy + tRange * np.sin(tYaw + tBearing)
            if abs(tX) > arenaHalf - OBJECT_WALL_MARGIN_M or abs(tY) > arenaHalf - OBJECT_WALL_MARGIN_M:
                continue
            if all(((tX - tPx) ** 2 + (tY - tPy) ** 2) ** 0.5 >= minSep for tPx, tPy in tPlaced):
                break
        tPlaced.append((tX, tY))
        tObjects.append(ObjSpec(mColor=tColor, mShape=tShape, mX=float(tX), mY=float(tY), mZ=SHAPES[tShape][2]))

    return SceneSpec(mSeed=seed, mObjects=tObjects, mArenaHalf=arenaHalf,
                     mRobotYaw=tYaw, mRobotX=float(tRx), mRobotY=float(tRy))


# ---- MJCF pieces shared with navigation/integrated_scene.py ------------------

def spawnQpos(qposText: str, scene: SceneSpec) -> str:
    """Keyframe qpos text with the free-joint base [x, y, z, qw, qx, qy, qz] set to the scene's spawn (z kept)."""
    tValues = qposText.split()
    tHalfYaw = scene.mRobotYaw / 2.0
    tValues[0] = f"{scene.mRobotX:.4f}"
    tValues[1] = f"{scene.mRobotY:.4f}"
    tValues[3] = f"{math.cos(tHalfYaw):.5f}"
    tValues[4] = "0"
    tValues[5] = "0"
    tValues[6] = f"{math.sin(tHalfYaw):.5f}"
    return " ".join(tValues)


def thirdPersonCameraXml(arenaHalf: float) -> str:
    """Fixed high overview camera behind/above the -y side that frames the whole arena."""
    return (f'<camera name="thirdperson" mode="fixed" '
            f'pos="0 {-(arenaHalf + 4.0):.2f} {arenaHalf + 4.5:.2f}" '
            f'xyaxes="1 0 0 0 0.66 0.75" fovy="55"/>')


def wallGeomXmls(arenaHalf: float) -> list[str]:
    """Four thin boxes (+y, -y, +x, -x) that keep the robot and the objects in the arena."""
    tWalls = [
        (0, arenaHalf, arenaHalf, WALL_THICKNESS_M),
        (0, -arenaHalf, arenaHalf, WALL_THICKNESS_M),
        (arenaHalf, 0, WALL_THICKNESS_M, arenaHalf),
        (-arenaHalf, 0, WALL_THICKNESS_M, arenaHalf),
    ]
    return [f'<geom name="wall_{tIndex}" type="box" pos="{tCx} {tCy} {WALL_HEIGHT_M / 2}" '
            f'size="{tSx} {tSy} {WALL_HEIGHT_M / 2}" rgba="0.5 0.5 0.55 1" contype="1" conaffinity="1"/>'
            for tIndex, (tCx, tCy, tSx, tSy) in enumerate(tWalls)]


def objectGeomXml(index: int, obj: ObjSpec) -> str:
    """Static geom (no freejoint: the scene is fully static) for one object."""
    tGeomType, tSize, _ = SHAPES[obj.mShape]
    tSizeText = " ".join(str(tS) for tS in tSize if tS > 0) or DEFAULT_SIZE_TEXT
    tRgba = " ".join(str(tC) for tC in COLORS[obj.mColor])
    return (f'<geom name="obj_{index}_{obj.mColor}_{obj.mShape}" type="{tGeomType}" '
            f'pos="{obj.mX} {obj.mY} {obj.mZ}" size="{tSizeText}" rgba="{tRgba}" '
            f'contype="1" conaffinity="1"/>')


def buildMjcf(scene: SceneSpec) -> str:
    """Full MJCF XML string for the scene, built around the vendored Menagerie g1.xml.

    Load g1.xml, inject the ego camera into torso_link, set the spawn pose in the first keyframe, then wrap it with
    a worldbody that adds floor, walls, third-person camera and objects. meshdir is absolute (see main()).
    """
    tTree = ET.parse(G1_XML)
    tRoot = tTree.getroot()

    # 1) inject the ego camera onto torso_link
    tTorso = next((tBody for tBody in tRoot.iter("body") if tBody.get("name") == "torso_link"), None)
    if tTorso is None:
        raise RuntimeError("torso_link not found in g1.xml")
    tTorso.append(ET.fromstring(EGO_CAMERA_XML))

    # 2) robot spawn pose (x, y, yaw): the first keyframe with a 7+-element qpos; z stays from the keyframe
    for tKey in tRoot.iter("key"):
        tQpos = tKey.get("qpos")
        if tQpos is None or len(tQpos.split()) < 7:
            continue
        tKey.set("qpos", spawnQpos(tQpos, scene))
        break

    # 3) additive worldbody pieces: third-person camera, walls, objects
    tExtra = ET.Element("worldbody")
    tExtra.append(ET.fromstring(thirdPersonCameraXml(scene.mArenaHalf)))
    for tWallXml in wallGeomXmls(scene.mArenaHalf):
        tExtra.append(ET.fromstring(tWallXml))
    for tIndex, tObj in enumerate(scene.mObjects):
        tExtra.append(ET.fromstring(objectGeomXml(tIndex, tObj)))

    # 4) assemble the final document: visual + scene assets + floor, then everything from g1.xml, then the extras
    tMujoco = ET.Element("mujoco", attrib={"model": f"g1nav_scene_{scene.mSeed}"})
    tMujoco.append(ET.fromstring(
        '<visual>'
        '<headlight diffuse="0.6 0.6 0.6" ambient="0.2 0.2 0.2" specular="0.3 0.3 0.3"/>'
        '<rgba haze="0.15 0.25 0.35 1"/>'
        '<global azimuth="140" elevation="-20" offwidth="640" offheight="480"/>'
        '</visual>'
    ))
    tMujoco.append(ET.fromstring(
        '<asset>'
        '<texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" '
        'width="512" height="3072"/>'
        '<texture type="2d" name="groundplane" builtin="checker" mark="edge" '
        'rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>'
        '<material name="groundplane" texture="groundplane" texuniform="true" '
        'texrepeat="5 5" reflectance="0.2"/>'
        '</asset>'
    ))
    tMujoco.append(ET.fromstring(
        '<worldbody><geom name="floor" size="0 0 0.05" type="plane" '
        'material="groundplane" contype="1" conaffinity="1"/></worldbody>'
    ))
    # everything from g1.xml (compiler, option, default, asset, worldbody, actuator, ...); meshdir made absolute so
    # the wrapped document finds the STL files
    for tChild in list(tRoot):
        if tChild.tag == "compiler":
            tChild.set("meshdir", os.path.join(G1_DIR, "assets"))
        tMujoco.append(tChild)
    tMujoco.append(tExtra)
    return ET.tostring(tMujoco, encoding="unicode")


def main(argv: list[str] | None = None) -> int:
    tParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    tParser.add_argument("--seed", type=int, default=0)
    tParser.add_argument("--out", type=str, default="")
    tParser.add_argument("--validate", action="store_true")
    tArgs = tParser.parse_args(argv)

    tScene = sampleScene(tArgs.seed)
    print(f"[scene_gen] seed={tScene.mSeed} n_objects={len(tScene.mObjects)} "
          f"robot_yaw={tScene.mRobotYaw:.2f}")
    for tObj in tScene.mObjects:
        print(f"    {tObj.label:18s} @ ({tObj.mX:+.2f},{tObj.mY:+.2f})")

    tXml = buildMjcf(tScene)
    tOut = tArgs.out or os.path.join(str(ASSETS_DIR), f"scene_seed{tArgs.seed}.xml")
    # The written file stores meshdir relative to its own directory (MuJoCo resolves a relative meshdir against the
    # model file when loaded with from_xml_path), so no machine-specific absolute path ends up in the scene file.
    # buildMjcf() itself keeps the absolute meshdir for callers that write the XML elsewhere.
    tAbsMeshdir = os.path.join(G1_DIR, "assets")
    tRelMeshdir = os.path.relpath(tAbsMeshdir, os.path.dirname(os.path.abspath(tOut)))
    tXml = tXml.replace(f'meshdir="{tAbsMeshdir}"', f'meshdir="{tRelMeshdir}"')
    with open(tOut, "w") as tFile:
        tFile.write(tXml)
    print(f"[scene_gen] wrote {tOut} ({len(tXml)} chars)")

    if tArgs.validate:
        import mujoco
        tModel = mujoco.MjModel.from_xml_path(tOut)
        tData = mujoco.MjData(tModel)
        mujoco.mj_forward(tModel, tData)
        tCameras = [mujoco.mj_id2name(tModel, mujoco.mjtObj.mjOBJ_CAMERA, tI) for tI in range(tModel.ncam)]
        print(f"[scene_gen] VALID: nq={tModel.nq} nu={tModel.nu} ncam={tModel.ncam} cameras={tCameras}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
