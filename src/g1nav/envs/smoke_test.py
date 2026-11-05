"""MuJoCo sanity check of the vendored Unitree G1 model: load, list actuators, step physics, render one frame.

Validates that the MJCF asset (assets/mujoco_menagerie/unitree_g1/scene.xml) and the MuJoCo install work before
any GPU work; runs on a CPU-only machine. Not part of training. Output: console report and, when offscreen
rendering works, assets/smoke_frame.png.

    python -m g1nav.envs.smoke_test
"""
from __future__ import annotations

import os

from g1nav.paths import ASSETS_DIR, G1_DIR

SCENE = os.path.join(str(G1_DIR), "scene.xml")
SMOKE_STEPS = 200
RENDER_HEIGHT, RENDER_WIDTH = 240, 320


def main(argv: list[str] | None = None) -> int:
    import mujoco

    print(f"[smoke] mujoco {mujoco.__version__}")
    print(f"[smoke] loading {SCENE}")
    tModel = mujoco.MjModel.from_xml_path(SCENE)
    tData = mujoco.MjData(tModel)

    print(f"[smoke] nq={tModel.nq} nv={tModel.nv} nu={tModel.nu} (actuators)")
    print(f"[smoke] nbody={tModel.nbody} ngeom={tModel.ngeom}")

    # actuator (joint) names -> these define the action space
    tNames = [mujoco.mj_id2name(tModel, mujoco.mjtObj.mjOBJ_ACTUATOR, tI) for tI in range(tModel.nu)]
    print(f"[smoke] actuators ({tModel.nu}):")
    for tName in tNames:
        print(f"         {tName}")

    # step physics from the default keyframe if present
    if tModel.nkey > 0:
        mujoco.mj_resetDataKeyframe(tModel, tData, 0)
        print("[smoke] reset to keyframe 0")
    mujoco.mj_forward(tModel, tData)
    for _ in range(SMOKE_STEPS):
        mujoco.mj_step(tModel, tData)
    print(f"[smoke] stepped {SMOKE_STEPS} steps OK; base height z={tData.qpos[2]:.3f}")

    # offscreen render
    try:
        tRenderer = mujoco.Renderer(tModel, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        tRenderer.update_scene(tData)
        tImage = tRenderer.render()
        tOut = os.path.join(str(ASSETS_DIR), "smoke_frame.png")
        try:
            import imageio.v2 as imageio
            imageio.imwrite(tOut, tImage)
            print(f"[smoke] wrote render {tOut} shape={tImage.shape}")
        except Exception as tError:
            print(f"[smoke] render OK shape={tImage.shape} but imageio save failed: {tError}")
    except Exception as tError:
        print(f"[smoke] OFFSCREEN RENDER FAILED: {tError}")
        print("[smoke] (physics is fine; rendering backend may need a GL context)")

    print("[smoke] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
