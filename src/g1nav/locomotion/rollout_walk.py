"""Roll out a trained (or mid-training) walking checkpoint with a fixed velocity command and render a video.

Used by the training dashboard to SEE how the current policy walks. Loads a train_walk.py checkpoint (<out>.pkl
params or <out>_latest.pkl {step, params}), runs the MuJoCo Playground G1 environment with a constant command and a
tracking camera, writes an mp4 and prints one `ROLLOUT step=.. survived=.. final_z=.. traveled=..m frames=..` line.
Note: unlike navigate.py and walk_eval.py, the network here is built WITHOUT the observation normaliser, so the
policy sees raw observations (the deployment bug described in the report, section 5.7a). Kept as it was; recorded
as a suspected bug in docs/refactor_2026-11.md.

    python -m g1nav.locomotion.rollout_walk --ckpt results/checkpoints/walk_t4_latest.pkl --vx 0.5 --out roll.mp4
"""
from __future__ import annotations

import argparse
import os
import pickle

from g1nav.paths import WORKSPACE_CKPT_DIR

RENDER_HEIGHT, RENDER_WIDTH = 320, 480
CAMERA_DISTANCE = 3.0
CAMERA_ELEVATION = -20.0
CAMERA_AZIMUTH = 90.0
LOOKAT_HEIGHT = 0.5
FALL_HEIGHT_M = 0.4


def loadParams(path):
    """(params, step) of a checkpoint; step is None for a bare params pickle."""
    with open(path, "rb") as tFile:
        tObj = pickle.load(tFile)
    if isinstance(tObj, dict) and "params" in tObj:
        return tObj["params"], tObj.get("step")
    return tObj, None


def main(argv: list[str] | None = None) -> int:
    tParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    tParser.add_argument("--ckpt", required=True, help="path to *_latest.pkl or *.pkl")
    tParser.add_argument("--env", default="G1JoystickFlatTerrain")
    tParser.add_argument("--vx", type=float, default=0.5)
    tParser.add_argument("--vy", type=float, default=0.0)
    tParser.add_argument("--wz", type=float, default=0.0)
    tParser.add_argument("--seconds", type=float, default=6.0)
    tParser.add_argument("--out", default=str(WORKSPACE_CKPT_DIR / "rollout.mp4"))
    tArgs = tParser.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")

    import jax
    import jax.numpy as jp
    import mujoco
    import numpy as np
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    tEnv = registry.load(tArgs.env)
    tConfig = registry.get_default_config(tArgs.env)
    tPpoConfig = locomotion_params.brax_ppo_config(tArgs.env)
    tNetworks = ppo_networks.make_ppo_networks(tEnv.observation_size, tEnv.action_size,
                                               **dict(tPpoConfig.network_factory))
    tMakePolicy = ppo_networks.make_inference_fn(tNetworks)
    tParams, tStepCount = loadParams(tArgs.ckpt)
    tPolicy = jax.jit(tMakePolicy(tParams, deterministic=True))
    tReset = jax.jit(tEnv.reset)
    tStep = jax.jit(tEnv.step)

    tRngKey = jax.random.PRNGKey(0)
    tState = tReset(tRngKey)
    tCommand = jp.array([tArgs.vx, tArgs.vy, tArgs.wz])   # fixed velocity command
    tState.info["command"] = tCommand

    tMjModel = tEnv.mj_model
    tMjData = mujoco.MjData(tMjModel)
    tRenderer = mujoco.Renderer(tMjModel, height=RENDER_HEIGHT, width=RENDER_WIDTH)
    tCamera = mujoco.MjvCamera()                           # follows the robot from behind-above
    tCamera.type = mujoco.mjtCamera.mjCAMERA_FREE
    tCamera.distance = CAMERA_DISTANCE
    tCamera.elevation = CAMERA_ELEVATION
    tCamera.azimuth = CAMERA_AZIMUTH

    tFrames = []
    tFell = False
    for _ in range(int(tArgs.seconds / tConfig.ctrl_dt)):
        tAct, _ = tPolicy(tState.obs, tRngKey)
        tState = tStep(tState, tAct)
        tState.info["command"] = tCommand
        # sync MjData from the MJX state for rendering
        tQpos = np.array(tState.data.qpos)
        tMjData.qpos[:] = tQpos
        tMjData.qvel[:] = np.array(tState.data.qvel)
        mujoco.mj_forward(tMjModel, tMjData)
        if tQpos[2] < FALL_HEIGHT_M:
            tFell = True
        tCamera.lookat[:] = [tQpos[0], tQpos[1], LOOKAT_HEIGHT]
        tRenderer.update_scene(tMjData, camera=tCamera)
        tFrames.append(tRenderer.render())

    try:
        import imageio
        imageio.mimsave(tArgs.out, tFrames, fps=int(1 / tConfig.ctrl_dt))
    except Exception as tError:
        print(f"video save failed: {tError}")

    tFinalZ = float(np.array(tState.data.qpos)[2])
    tDistance = float(np.linalg.norm(np.array(tState.data.qpos)[:2]))
    print(f"ROLLOUT step={tStepCount} survived={'NO' if tFell else 'YES'} "
          f"final_z={tFinalZ:.2f} traveled={tDistance:.2f}m frames={len(tFrames)} -> {tArgs.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
