"""Raw-environment evaluation of walking checkpoints (one checkpoint, or a sweep over all numbered ones).

Same setup as train_walk.py's in-training rollout: the RAW registry environment (no wrap_for_brax_training, so no
AutoResetWrapper: falls are real, never teleported back to the start), a constant forward command VX in the robot's
heading frame, N_STEPS control steps. Reported per checkpoint: net horizontal displacement, mean LOCAL forward
velocity (what tracking_lin_vel rewards), minimum base height and a verdict (WALKS / STEPS/STUCK / FALLS). The
network is built with the observation normaliser, as in training.

Output: a sweep writes $G1NAV_OUT_DIR/walk_sweep.json (results/walk_sweep_cpu/ holds the committed CPU run); a single
checkpoint prints its JSON row and renders $G1NAV_OUT_DIR/walk_eval.mp4. Needs the pinned simulation stack; CPU
works (JAX_PLATFORMS=cpu).

    G1NAV_OUT_DIR=results python -m g1nav.locomotion.walk_eval results/checkpoints/walk_t4_latest.pkl
    G1NAV_OUT_DIR=results/walk_sweep_cpu python -m g1nav.locomotion.walk_eval --sweep results/checkpoints 'walk_t4_step*.pkl'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

from g1nav.paths import WORKSPACE_CKPT_DIR

ENV = "G1JoystickFlatTerrain"
VX = 0.6
N_STEPS = 250  # 5 s; < 500 so the environment never resamples the command
WALK_MIN_FWD_VEL = 0.3   # m/s mean local forward velocity for "WALKS"
FALL_HEIGHT_M = 0.4      # minimum base height below this = "FALLS"
RENDER_FPS = 50
RENDER_HEIGHT, RENDER_WIDTH = 288, 384
CAMERA_DISTANCE = 4.0
CAMERA_ELEVATION = -12
CAMERA_AZIMUTH = 135
LOOKAT_HEIGHT = 0.5
LINVEL_SENSOR = "local_linvel_pelvis"


def outDir() -> str:
    """Where the sweep JSON / eval video are written: $G1NAV_OUT_DIR (default: the pod checkpoint folder)."""
    return os.environ.get("G1NAV_OUT_DIR", str(WORKSPACE_CKPT_DIR))


def verdict(meanForward: float, minZ: float) -> str:
    if meanForward > WALK_MIN_FWD_VEL and minZ > FALL_HEIGHT_M:
        return "WALKS"
    return "STEPS/STUCK" if minZ > FALL_HEIGHT_M else "FALLS"


class WalkEvaluator:
    """Raw environment + policy factory, built once and reused for every checkpoint."""

    def __init__(self):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import jax
        import jax.numpy as jp
        import mujoco
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
        from mujoco_playground import registry
        from mujoco_playground.config import locomotion_params

        self.mJax = jax
        self.mMujoco = mujoco
        # The reward config only affects the environment build, not the policy weights; the final run used the
        # official defaults, so nothing is overridden.
        tConfig = registry.get_default_config(ENV)
        self.mEnv = registry.load(ENV, config=tConfig)          # RAW env — no wrappers, no auto-reset
        tPpoConfig = locomotion_params.brax_ppo_config(ENV)
        tNetworks = ppo_networks.make_ppo_networks(self.mEnv.observation_size, self.mEnv.action_size,
                                                   preprocess_observations_fn=running_statistics.normalize,
                                                   **dict(tPpoConfig.network_factory))
        self.mMakePolicy = ppo_networks.make_inference_fn(tNetworks)
        self.mReset = jax.jit(self.mEnv.reset)
        self.mStep = jax.jit(self.mEnv.step)
        self.mMjModel = self.mEnv.mj_model
        self.mLinvelAdr = self.mMjModel.sensor_adr[
            mujoco.mj_name2id(self.mMjModel, mujoco.mjtObj.mjOBJ_SENSOR, LINVEL_SENSOR)]
        self.mCommand = jp.array([VX, 0.0, 0.0])

    def evaluate(self, ckpt: str, renderPath: str | None = None) -> dict:
        """Roll out one checkpoint for N_STEPS; return its row (and render a video when renderPath is set)."""
        import numpy as np

        tJax, tMujoco = self.mJax, self.mMujoco
        with open(ckpt, "rb") as tFile:
            tCheckpoint = pickle.load(tFile)
        tParams = tCheckpoint["params"] if isinstance(tCheckpoint, dict) and "params" in tCheckpoint else tCheckpoint
        tStepCount = tCheckpoint.get("step") if isinstance(tCheckpoint, dict) else None
        tPolicy = tJax.jit(self.mMakePolicy(tParams, deterministic=True))
        tState = self.mReset(tJax.random.PRNGKey(0))
        tState.info["command"] = self.mCommand
        tXs, tYs, tZs, tForward, tFrames = [], [], [], [], []
        if renderPath:
            tRenderer = tMujoco.Renderer(self.mMjModel, RENDER_HEIGHT, RENDER_WIDTH)
            tMjData = tMujoco.MjData(self.mMjModel)
            tCamera = tMujoco.MjvCamera()
            tCamera.type = tMujoco.mjtCamera.mjCAMERA_FREE
            tCamera.distance = CAMERA_DISTANCE
            tCamera.elevation = CAMERA_ELEVATION
            tCamera.azimuth = CAMERA_AZIMUTH
        for _ in range(N_STEPS):
            tState.info["command"] = self.mCommand
            tAct, _ = tPolicy(tState.obs, tJax.random.PRNGKey(0))
            tState = self.mStep(tState, tAct)
            tQpos = np.array(tState.data.qpos).reshape(-1)
            tSensorData = np.array(tState.data.sensordata).reshape(-1)
            tXs.append(float(tQpos[0]))
            tYs.append(float(tQpos[1]))
            tZs.append(float(tQpos[2]))
            tForward.append(float(tSensorData[self.mLinvelAdr]))
            if renderPath:
                tMjData.qpos[:] = tQpos
                tMjData.qvel[:] = np.array(tState.data.qvel).reshape(-1)
                tMujoco.mj_forward(self.mMjModel, tMjData)
                tCamera.lookat[:] = [tQpos[0], tQpos[1], LOOKAT_HEIGHT]
                tRenderer.update_scene(tMjData, camera=tCamera)
                tFrames.append(tRenderer.render())
        if renderPath:
            import imageio
            imageio.mimsave(renderPath, tFrames, fps=RENDER_FPS)
            tRenderer.close()
        tDx, tDy = tXs[-1] - tXs[0], tYs[-1] - tYs[0]
        tNet = float(np.hypot(tDx, tDy))
        tMeanForward = float(np.mean(tForward))
        tMinZ = float(min(tZs))
        return {"ckpt": os.path.basename(ckpt), "step": tStepCount, "net_dist": round(tNet, 2),
                "fwd_vel": round(tMeanForward, 3), "min_z": round(tMinZ, 2), "verdict": verdict(tMeanForward, tMinZ)}


def main(argv: list[str] | None = None) -> int:
    tParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    tParser.add_argument("ckpt", nargs="?", default=str(WORKSPACE_CKPT_DIR / "walk_t4_latest.pkl"),
                         help="checkpoint to evaluate (and render)")
    tParser.add_argument("--sweep", nargs=2, metavar=("DIR", "PATTERN"),
                         help="evaluate every checkpoint matching DIR/PATTERN and write walk_sweep.json")
    tArgs = tParser.parse_args(argv)

    tEvaluator = WalkEvaluator()
    if tArgs.sweep:
        tDir, tPattern = tArgs.sweep
        tCheckpoints = sorted(glob.glob(os.path.join(tDir, tPattern)))
        print(f"sweeping {len(tCheckpoints)} checkpoints (cmd_vx={VX}, raw env, {N_STEPS} steps)")
        tRows = []
        for tCkpt in tCheckpoints:
            tRow = tEvaluator.evaluate(tCkpt)
            tRows.append(tRow)
            print(f"  {tRow['step'] if tRow['step'] is not None else '?':>11}  net={tRow['net_dist']:5.2f}m  "
                  f"fwd_vel={tRow['fwd_vel']:+.2f}  min_z={tRow['min_z']:5.2f}  {tRow['verdict']}")
        with open(os.path.join(outDir(), "walk_sweep.json"), "w") as tFile:
            json.dump(tRows, tFile, indent=2)
        tBest = max(tRows, key=lambda tR: tR["fwd_vel"])
        print(f"BEST fwd_vel: step={tBest['step']} fwd_vel={tBest['fwd_vel']} ({tBest['verdict']})")
        return 0
    tVideo = os.path.join(outDir(), "walk_eval.mp4")
    tRow = tEvaluator.evaluate(tArgs.ckpt, renderPath=tVideo)
    print(json.dumps(tRow, indent=2))
    print(f"video -> {tVideo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
