"""Closed loop: the trained PPO walking policy, driven by the Navigator toward the selected target, in MuJoCo physics.

Stages 3-5 of the pipeline. The target is fixed once (static scene) and tracked by odometry:

  target world (x, y), once -> read from the generated scene (ground truth); --use_vlm is reserved, not implemented
  every CTRL_DT:
    odometry (qpos) -> Navigator -> velocity command (vx, vy, wz)
    103-D Playground-compatible observation with that command
    PPO policy(obs) -> action(29) -> motor targets = DEFAULT_POSE + action * ACTION_SCALE
    mj_step x N_SUBSTEPS (physics), until reached / fallen / timeout
  render ego + third-person every step -> side-by-side mp4 (2 s standing pre-roll, 2 s post-roll when reached)

Observation layout (must match mujoco_playground G1 Joystick _get_obs 'state'):
  [linvel(3), gyro(3), gravity(3), command(3), qpos[7:]-default(29), qvel[6:](29), last_act(29), cos/sin phase(2)]
  = 103

Input: a train_walk.py checkpoint, a scene seed and a target label. Output: an mp4 and one machine-readable line
`NAV seed=.. target='..' reached=.. final_dist_to_goal=..m base_z=.. frames=.. -> <out>` (parsed by pipeline.py and
rerun_demo_episodes.py). Needs the pinned simulation stack; CPU is enough (--cpu or G1NAV_CPU=1).

    python -m g1nav.navigation.navigate --ckpt results/checkpoints/walk_t4_latest.pkl --seed 0 \
        --target "orange cylinder" --seconds 12 --out outputs/nav_seed0.mp4
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

from g1nav.navigation.action_module import Navigator, quatYaw
from g1nav.paths import WORKSPACE_CKPT_DIR

ENV_NAME = "G1JoystickFlatTerrain"
# Network input sizes of the Playground G1 Joystick environment (state, privileged critic state) and action size.
OBS_SIZE = {"state": 103, "privileged_state": 216}
ACT_SIZE = 29
# Registry config scalars, hard-coded so that registry.load (which puts an MJX model on the GPU) is not needed.
ACTION_SCALE = 0.5
CTRL_DT = 0.02
# Physics dt of the training environment: Playground sets opt.timestep = sim_dt = 0.002 in code (not in the XML)
# with ctrl_dt = 0.02, i.e. 10 substeps; the XML loaded here may have a different default timestep.
SIM_DT = 0.002
# Solver options of g1_mjx_feetonly.xml as used in MJX training (<option iterations="3" ls_iterations="5"
# integrator="Euler"><flag eulerdamp="disable"/>). Plain MuJoCo defaults (100 iterations, eulerdamp on) give
# different contact dynamics and the policy drifts and falls after ~1.5 s.
SOLVER_ITERATIONS = 3
SOLVER_LS_ITERATIONS = 5
# The training environment's hard-coded _default_pose (knees ~0.67, elbows ~0.6), NOT the keyframe qpos[7:]
# (knees 0.3, elbows 1.28). Using the keyframe pose made both obs (qpos - default) and control (default + action)
# wrong -> instant fall.
DEFAULT_POSE = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,      # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,      # right leg
    0.0, 0.0, 0.0,                              # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,          # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,         # right arm
], dtype=float)
GAIT_FREQ_HZ = 1.4
INITIAL_PHASE = (0.0, np.pi)
# ramp the command up over the first 0.5 s so the standing policy is not hit by a large velocity target at step 0
WARMUP_STEPS = 25
HOLD_S = 2.0                     # standing pre-roll and post-roll
FALL_HEIGHT_M = 0.4              # base height below this = fallen
RENDER_HEIGHT, RENDER_WIDTH = 240, 320
LINVEL_SENSOR = "local_linvel_pelvis"
GYRO_SENSOR = "gyro_pelvis"
IMU_SITE = "imu_in_pelvis"
WORLD_DOWN = np.array([0.0, 0.0, -1.0])


class ClosedLoop:
    """MuJoCo model/data of the integrated scene + the policy, stepping one control step at a time."""

    def __init__(self, mujocoModule, model, data, jitPolicy, rngKey, jnpModule):
        self.mMujoco = mujocoModule
        self.mModel = model
        self.mData = data
        self.mPolicy = jitPolicy
        self.mRngKey = rngKey
        self.mJnp = jnpModule
        self.mSubsteps = int(round(CTRL_DT / SIM_DT))   # = 10 substeps per control step
        self.mPhase = np.array(INITIAL_PHASE)
        self.mPhaseDt = 2 * np.pi * CTRL_DT * GAIT_FREQ_HZ
        self.mLastAct = np.zeros(model.nu)
        self.mFrames = []
        self.mLinvelAdr, self.mLinvelDim = self._sensorAddress(LINVEL_SENSOR)   # velocimeter (3)
        self.mGyroAdr, self.mGyroDim = self._sensorAddress(GYRO_SENSOR)         # gyro (3)
        # The training environment computes the projected-gravity term as site_xmat[imu_in_pelvis].T @ [0, 0, -1],
        # i.e. the world down vector in the pelvis-IMU LOCAL frame. The `upvector_pelvis` sensor returns the
        # world-frame +z axis, whose negation only coincides with it while the robot walks straight; on every yaw
        # the two diverge and the policy misreads its tilt (fell at ~1.6 s). Use the site matrix as in training.
        self.mImuSiteId = mujocoModule.mj_name2id(model, mujocoModule.mjtObj.mjOBJ_SITE, IMU_SITE)
        if self.mImuSiteId < 0:
            raise RuntimeError(f"site '{IMU_SITE}' not found in integrated model")
        self.mRendererEgo = mujocoModule.Renderer(model, RENDER_HEIGHT, RENDER_WIDTH)
        self.mRendererTp = mujocoModule.Renderer(model, RENDER_HEIGHT, RENDER_WIDTH)

    def _sensorAddress(self, name):
        """Read the same named sensors the training environment uses, so the observation matches _get_obs."""
        tSensorId = self.mMujoco.mj_name2id(self.mModel, self.mMujoco.mjtObj.mjOBJ_SENSOR, name)
        if tSensorId < 0:
            raise RuntimeError(f"sensor '{name}' not found in integrated model")
        return self.mModel.sensor_adr[tSensorId], self.mModel.sensor_dim[tSensorId]

    def observation(self, command):
        """103-D policy observation, exactly as the training environment's 'state'."""
        tSensorData = self.mData.sensordata
        tLinvel = tSensorData[self.mLinvelAdr:self.mLinvelAdr + self.mLinvelDim]   # local linear velocity
        tGyro = tSensorData[self.mGyroAdr:self.mGyroAdr + self.mGyroDim]           # local angular velocity
        tImuRotation = self.mData.site_xmat[self.mImuSiteId].reshape(3, 3)
        tGravity = tImuRotation.T @ WORLD_DOWN                                      # world down in the IMU frame
        tJointAngles = self.mData.qpos[7:7 + self.mModel.nu] - DEFAULT_POSE
        tJointVel = self.mData.qvel[6:6 + self.mModel.nu]
        tCos = np.cos(self.mPhase)
        tSin = np.sin(self.mPhase)
        return np.concatenate([tLinvel, tGyro, tGravity, command, tJointAngles, tJointVel, self.mLastAct,
                               np.concatenate([tCos, tSin])]).astype(np.float32)

    def stepOnce(self, command):
        """One control step with `command`; appends a frame. Returns False if the robot fell."""
        tObs = self.observation(command)
        tAct, _ = self.mPolicy({"state": self.mJnp.array(tObs),
                                "privileged_state": self.mJnp.zeros(OBS_SIZE["privileged_state"])},
                               self.mRngKey)
        tAct = np.array(tAct)
        self.mData.ctrl[:] = DEFAULT_POSE + tAct * ACTION_SCALE
        for _ in range(self.mSubsteps):
            self.mMujoco.mj_step(self.mModel, self.mData)
        self.mLastAct = tAct
        self.mPhase[:] = (self.mPhase + self.mPhaseDt + np.pi) % (2 * np.pi) - np.pi
        self.mRendererEgo.update_scene(self.mData, camera="ego")
        tEgo = self.mRendererEgo.render()
        self.mRendererTp.update_scene(self.mData, camera="thirdperson")
        tThird = self.mRendererTp.render()
        self.mFrames.append(np.concatenate([tThird, tEgo], axis=1))  # side by side
        return self.mData.qpos[2] >= FALL_HEIGHT_M


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    tParser.add_argument("--ckpt", required=True)
    tParser.add_argument("--seed", type=int, default=0)
    tParser.add_argument("--target", default=None,
                         help="object label e.g. 'orange cylinder'; default=first object")
    tParser.add_argument("--seconds", type=float, default=12.0)
    tParser.add_argument("--out", default=str(WORKSPACE_CKPT_DIR / "nav.mp4"))
    tParser.add_argument("--use_vlm", action="store_true",
                         help="reserved, not implemented: the target position is always read "
                              "from the generated scene (ground truth)")
    tParser.add_argument("--cpu", action="store_true",
                         help="force JAX onto CPU (use when a GPU training job owns the device)")
    tParser.add_argument("--fixed_cmd", type=float, nargs=3, default=None,
                         metavar=("VX", "VY", "WZ"),
                         help="ignore Navigator; drive a constant (vx,vy,wz) command")
    return tParser


def loadPolicy(ckptPath, jaxModule):
    """Deterministic jitted inference function of a train_walk.py checkpoint.

    Training used normalize_observations=True, so the checkpoint stores the running-statistics normaliser as
    params[0]. The network MUST be built with the same normalising preprocessor, otherwise make_policy ignores
    those statistics and feeds the policy raw (out-of-distribution) observations -> it topples (this was the bug).
    """
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground.config import locomotion_params

    tPpoConfig = locomotion_params.brax_ppo_config(ENV_NAME)
    tNetKwargs = dict(tPpoConfig.network_factory)
    tNetworks = ppo_networks.make_ppo_networks(
        OBS_SIZE, ACT_SIZE,
        preprocess_observations_fn=running_statistics.normalize, **tNetKwargs)
    tMakePolicy = ppo_networks.make_inference_fn(tNetworks)
    with open(ckptPath, "rb") as tFile:
        tCheckpoint = pickle.load(tFile)
    tParams = tCheckpoint["params"] if isinstance(tCheckpoint, dict) and "params" in tCheckpoint else tCheckpoint
    return jaxModule.jit(tMakePolicy(tParams, deterministic=True))


def loadModel(mujocoModule, seed):
    """Integrated scene for `seed` with the training environment's physics options; robot at DEFAULT_POSE."""
    from g1nav.navigation import integrated_scene

    tPath, tScene = integrated_scene.buildIntegratedXml(seed)
    tModel = mujocoModule.MjModel.from_xml_path(tPath)
    tModel.opt.timestep = SIM_DT
    tModel.opt.iterations = SOLVER_ITERATIONS
    tModel.opt.ls_iterations = SOLVER_LS_ITERATIONS
    tModel.opt.integrator = mujocoModule.mjtIntegrator.mjINT_EULER
    tModel.opt.disableflags |= mujocoModule.mjtDisableBit.mjDSBL_EULERDAMP
    tData = mujocoModule.MjData(tModel)
    mujocoModule.mj_resetDataKeyframe(tModel, tData, 0)
    mujocoModule.mj_forward(tModel, tData)
    assert DEFAULT_POSE.shape[0] == tModel.nu, (DEFAULT_POSE.shape, tModel.nu)
    # start the robot AT the training default pose, so step 0 matches
    tData.qpos[7:7 + tModel.nu] = DEFAULT_POSE
    mujocoModule.mj_forward(tModel, tData)
    return tModel, tData, tScene


def main(argv: list[str] | None = None) -> int:
    # Force CPU before anything imports jax when G1NAV_CPU=1 (a GPU training job may own the device).
    if os.environ.get("G1NAV_CPU") == "1":
        os.environ["JAX_PLATFORMS"] = "cpu"
    print(f"[navigate] G1NAV_CPU={os.environ.get('G1NAV_CPU')} "
          f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')}", flush=True)
    tArgs = buildParser().parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    if tArgs.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"   # must be set BEFORE importing jax

    import jax
    import jax.numpy as jp
    import mujoco

    tPolicy = loadPolicy(tArgs.ckpt, jax)
    tModel, tData, tScene = loadModel(mujoco, tArgs.seed)
    tLoop = ClosedLoop(mujoco, tModel, tData, tPolicy, jax.random.PRNGKey(0), jp)
    print(f"[navigate] sim_dt={SIM_DT} ctrl_dt={CTRL_DT} n_substeps={tLoop.mSubsteps}",
          flush=True)

    # --- target world position ---
    tTargetName = tArgs.target or tScene.mObjects[0].label
    tTarget = next((tObj for tObj in tScene.mObjects if tObj.label == tTargetName), tScene.mObjects[0])
    # Ground truth from the generated scene. A perception-based estimate would plug in here
    # (Navigator.lockGoalFromEgo); it is not implemented.
    tGoalWorld = (tTarget.mX, tTarget.mY)
    tNavigator = Navigator()
    tNavigator.mGoalWorld = tGoalWorld

    tSteps = int(tArgs.seconds / CTRL_DT)
    tHoldSteps = int(round(HOLD_S / CTRL_DT))   # 2 s at 50 Hz = 100 control steps
    tZero = np.zeros(3, dtype=np.float32)
    # optional fixed command instead of the Navigator, to isolate walking from navigation,
    # e.g. --fixed_cmd 0 0 0 (stand), 0.5 0 0 (forward)
    tFixed = tArgs.fixed_cmd
    tReached = False

    for _ in range(tHoldSteps):                 # pre-roll: standing still
        tLoop.stepOnce(tZero)
    for tStep in range(tSteps):                 # navigate to the target
        tX, tY, tYaw = float(tData.qpos[0]), float(tData.qpos[1]), quatYaw(tData.qpos[3:7])
        if tFixed is not None:
            tVx, tVy, tWz, tDone = tFixed[0], tFixed[1], tFixed[2], False
        else:
            tVx, tVy, tWz, tDone = tNavigator.command(tX, tY, tYaw)
        tRamp = min(1.0, (tStep + 1) / WARMUP_STEPS)
        tCommand = np.array([tVx * tRamp, tVy * tRamp, tWz * tRamp], dtype=np.float32)
        if not tLoop.stepOnce(tCommand):
            print(f"FELL at step {tStep}")
            break
        if tDone:
            tReached = True
            break
    if tReached:                                # post-roll: standing at the reached target
        for _ in range(tHoldSteps):
            tLoop.stepOnce(tZero)

    try:
        import imageio
        imageio.mimsave(tArgs.out, tLoop.mFrames, fps=int(1 / CTRL_DT))
    except Exception as tError:
        print("video save failed:", tError)
    tFinalX, tFinalY = float(tData.qpos[0]), float(tData.qpos[1])
    tDistance = float(np.hypot(tGoalWorld[0] - tFinalX, tGoalWorld[1] - tFinalY))
    print(f"NAV seed={tArgs.seed} target='{tTargetName}' reached={tReached} "
          f"final_dist_to_goal={tDistance:.2f}m base_z={tData.qpos[2]:.2f} "
          f"frames={len(tLoop.mFrames)} -> {tArgs.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
