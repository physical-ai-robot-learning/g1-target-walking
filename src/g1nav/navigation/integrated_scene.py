"""Integrated MJCF for the closed loop: the MuJoCo Playground G1 + the seeded scene's objects, walls and cameras.

The Playground G1 model is the one the PPO policy was trained on, so its observation/action layout matches the
checkpoint exactly. Starting from Playground's scene_mjx_feetonly_flat_terrain.xml this injects an ego camera on
the torso (in a patched copy of the included g1_mjx_feetonly.xml), the third-person overview camera, the arena
walls and the coloured objects, and sets the robot spawn in the 'home' keyframe. The layout comes from
envs.scene_gen.sampleScene, so seeds match the dashboards and the recorded demos.

Input: a seed. Output: an MJCF file written into the installed Playground G1 xml folder (so its relative
meshdir/include paths resolve) and the SceneSpec. Needs `mujoco_playground` (imported when called).

    python -m g1nav.navigation.integrated_scene      # build and validate seeds 0 and 1
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from g1nav.envs.scene_gen import (EGO_CAMERA_XML, SceneSpec, objectGeomXml, sampleScene, spawnQpos,
                                  thirdPersonCameraXml, wallGeomXmls)

BASE_SCENE_FILE = "scene_mjx_feetonly_flat_terrain.xml"
G1_BODY_INCLUDE = "g1_mjx_feetonly"       # the included robot file that holds torso_link
SPAWN_KEYFRAME = "home"
VALIDATE_SEEDS = (0, 1)


def playgroundG1XmlDir() -> str:
    """Folder of the installed mujoco_playground G1 XML files."""
    import mujoco_playground
    return os.path.join(os.path.dirname(mujoco_playground.__file__), "_src", "locomotion", "g1", "xmls")


def buildIntegratedXml(seed: int, outPath: str | None = None) -> tuple[str, SceneSpec]:
    """Write the integrated MJCF for `seed`; return (xml_path, scene)."""
    tScene = sampleScene(seed)
    tXmlDir = playgroundG1XmlDir()
    tTree = ET.parse(os.path.join(tXmlDir, BASE_SCENE_FILE))
    tRoot = tTree.getroot()

    # The G1 body lives in an INCLUDED file (g1_mjx_feetonly.xml), not in this scene, so ElementTree cannot see
    # torso_link here. Write a patched copy of the body file with the ego camera on torso_link and point the
    # include at it.
    for tInclude in tRoot.iter("include"):
        tFile = tInclude.get("file")
        if tFile and G1_BODY_INCLUDE in tFile and "restricted" not in tFile:
            tSource = tFile if os.path.isabs(tFile) else os.path.join(tXmlDir, tFile)
            tBodyTree = ET.parse(tSource)
            for tBody in tBodyTree.iter("body"):
                if tBody.get("name") == "torso_link":
                    tBody.append(ET.fromstring(EGO_CAMERA_XML))
                    break
            tPatched = os.path.join(tXmlDir, f"_g1_ego_{seed}.xml")
            tBodyTree.write(tPatched)
            tInclude.set("file", tPatched)

    # robot spawn via the 'home' keyframe qpos[:7] = x, y, z, qw, qx, qy, qz
    for tKey in tRoot.iter("key"):
        if tKey.get("name") != SPAWN_KEYFRAME:
            continue
        tKey.set("qpos", spawnQpos(tKey.get("qpos"), tScene))
        break

    tWorldbody = tRoot.find("worldbody")
    tWorldbody.append(ET.fromstring(thirdPersonCameraXml(tScene.mArenaHalf)))
    for tWallXml in wallGeomXmls(tScene.mArenaHalf):
        tWorldbody.append(ET.fromstring(tWallXml))
    for tIndex, tObj in enumerate(tScene.mObjects):
        tWorldbody.append(ET.fromstring(objectGeomXml(tIndex, tObj)))

    # write INTO the Playground xml folder so all relative meshdir/include paths (assets/*.STL, sensor.xml,
    # g1_mjx_feetonly.xml) resolve
    tOutPath = outPath or os.path.join(tXmlDir, f"_integrated_{seed}.xml")
    tTree.write(tOutPath)
    return tOutPath, tScene


def main(argv: list[str] | None = None) -> int:
    import mujoco
    for tSeed in VALIDATE_SEEDS:
        tPath, tScene = buildIntegratedXml(tSeed)
        tModel = mujoco.MjModel.from_xml_path(tPath)
        tData = mujoco.MjData(tModel)
        mujoco.mj_resetDataKeyframe(tModel, tData, 0)
        mujoco.mj_forward(tModel, tData)
        tCameras = [mujoco.mj_id2name(tModel, mujoco.mjtObj.mjOBJ_CAMERA, tI) for tI in range(tModel.ncam)]
        print(f"seed{tSeed}: VALID nq={tModel.nq} nu={tModel.nu} ncam={tModel.ncam} {tCameras} "
              f"base=({tData.qpos[0]:.2f},{tData.qpos[1]:.2f},z={tData.qpos[2]:.2f}) "
              f"objs={[tObj.label for tObj in tScene.mObjects]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
