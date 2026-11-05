"""Train the velocity-conditioned PPO walking policy for the Unitree G1 (MuJoCo Playground G1JoystickFlatTerrain, MJX).

Stage 4 of the pipeline (the only learned part). The environment's action is a PD joint-position target
(motor_targets = default_pose + action * action_scale) and its command is (vx, vy, yaw_rate) in the robot's heading
frame — the interface the Navigator drives. Training is brax PPO with Playground's tuned hyperparameters, the
official domain randomiser and a separate eval_env (without domain randomisation the reward rose but the policy did
not walk). At every evaluation a checkpoint is saved and a 4 s rollout is run in the RAW environment (no
auto-reset wrapper, so falls are real; walk_eval.py uses the identical setup).

The reward-scale flags default to the edited values of an earlier run (--termination -3 --track_lin 2.0
--stand_still -3.0); the final run passed the Playground defaults explicitly (-100 / 1.0 / -1.0), see the README.

Outputs (prefix --out): <out>_metrics.jsonl (step, eval reward, wall-clock), <out>_latest.pkl and
<out>_step<N>.pkl ({"step", "params"}), <out>_rollout.{mp4,json}, and at the end <out>.pkl + <out>_cfg.pkl.
Needs a GPU with the pinned stack (configs/envs/requirements-gpu.txt).

    python -m g1nav.locomotion.train_walk --timesteps 500000000 --num_envs 8192 \
        --termination -100 --track_lin 1.0 --stand_still -1.0 --num_evals 50 --out results/checkpoints/walk_t4
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import time

from g1nav.paths import WORKSPACE_CKPT_DIR

ROLLOUT_STEPS = 200                  # 4 s at 50 Hz
ROLLOUT_COMMAND = (0.6, 0.0, 0.0)    # forward command of the in-training rollout (m/s, m/s, rad/s)
ROLLOUT_FPS = 50
ROLLOUT_RENDER_HEIGHT, ROLLOUT_RENDER_WIDTH = 288, 384
ROLLOUT_CAMERA_DISTANCE = 4.0
ROLLOUT_CAMERA_ELEVATION = -12
ROLLOUT_CAMERA_AZIMUTH = 135
ROLLOUT_LOOKAT_HEIGHT = 0.5
LINVEL_SENSOR = "local_linvel_pelvis"


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    tParser.add_argument("--env", default="G1JoystickFlatTerrain")
    tParser.add_argument("--timesteps", type=int, default=60_000_000,
                         help="PPO env steps. Full recipe is 200M; 40-80M usually walks.")
    tParser.add_argument("--out", default=str(WORKSPACE_CKPT_DIR / "walk"))
    tParser.add_argument("--num_envs", type=int, default=8192)
    tParser.add_argument("--seed", type=int, default=0)
    tParser.add_argument("--restore", default=None,
                         help="path to a *_latest.pkl / *.pkl to resume training from")
    tParser.add_argument("--termination", type=float, default=-3.0,
                         help="fall penalty scale; default -3 (env default is -100, "
                              "which makes the policy too scared to attempt walking)")
    tParser.add_argument("--num_evals", type=int, default=100,
                         help="number of evals across training; each saves a checkpoint")
    tParser.add_argument("--track_lin", type=float, default=2.0,
                         help="tracking_lin_vel reward scale (default env=1.0; raise to force forward motion)")
    tParser.add_argument("--stand_still", type=float, default=-3.0,
                         help="stand_still penalty scale (default env=-1.0; more negative discourages standing)")
    return tParser


class TrainingCallbacks:
    """brax PPO callbacks: metrics log (progress_fn) and checkpoint + raw-env rollout (policy_params_fn)."""

    def __init__(self, outPrefix, renderEnvFactory, jaxModule):
        self.mOut = outPrefix
        self.mJax = jaxModule
        self.mMetricsPath = outPrefix + "_metrics.jsonl"
        open(self.mMetricsPath, "w").close()  # truncate
        self.mTimes = [time.monotonic()]      # wall-clock starts here (before the render env is built)
        self.mRenderEnv = renderEnvFactory()  # raw registry env (obs/dynamics), no brax wrappers

    def elapsedMin(self) -> float:
        return (time.monotonic() - self.mTimes[0]) / 60

    def progress(self, step, metrics):
        """brax progress_fn(step, metrics): print and append one JSON line per evaluation."""
        self.mTimes.append(time.monotonic())
        tReward = float(metrics.get("eval/episode_reward", float("nan")))
        tRewardStd = float(metrics.get("eval/episode_reward_std", float("nan")))
        tElapsed = self.mTimes[-1] - self.mTimes[0]
        print(f"[walk] step={step:>12,}  reward={tReward:8.3f} +/-{tRewardStd:6.3f}  "
              f"elapsed={tElapsed/60:5.1f}m", flush=True)
        with open(self.mMetricsPath, "a") as tFile:
            tFile.write(json.dumps({"step": int(step), "reward": tReward,
                                    "reward_std": tRewardStd, "elapsed_min": tElapsed / 60}) + "\n")

    def saveCheckpoint(self, step, make_policy, params):
        """brax policy_params_fn(step, make_policy, params): latest + numbered checkpoint, then a raw-env rollout."""
        try:
            tBlob = {"step": int(step), "params": params}
            with open(self.mOut + "_latest.pkl", "wb") as tFile:
                pickle.dump(tBlob, tFile)
            # also a NUMBERED checkpoint so every evaluation can be swept later (~2 MB each)
            with open(self.mOut + f"_step{int(step):010d}.pkl", "wb") as tFile:
                pickle.dump(tBlob, tFile)
        except Exception as tError:
            print(f"[walk] ckpt save failed: {tError}", flush=True)
            return
        try:
            self.rollout(step, make_policy, params)
        except Exception as tError:
            print(f"[walk] rollout failed: {tError}", flush=True)

    def rollout(self, step, makePolicy, params):
        """4 s forward-command rollout in the RAW env (no AutoResetWrapper): video + local forward velocity.

        The brax training wrapper teleports the robot back to the start pose on every fall, which masks falls as
        "stepping in place" (min_z ~0.74, x ~0). The raw env shows reality. The command is in the robot's heading
        frame, so the LOCAL forward velocity (what tracking_lin_vel rewards) and the net horizontal displacement
        are reported, not world-X.
        """
        import imageio
        import jax.numpy as jp
        import mujoco
        import numpy as np

        tJax = self.mJax
        tMjModel = self.mRenderEnv.mj_model
        tPolicy = tJax.jit(makePolicy(params, deterministic=True))
        tReset = tJax.jit(self.mRenderEnv.reset)
        tStep = tJax.jit(self.mRenderEnv.step)
        tState = tReset(tJax.random.PRNGKey(0))            # RAW, unbatched, no auto-reset
        tCommand = jp.array(list(ROLLOUT_COMMAND))
        tLinvelId = mujoco.mj_name2id(tMjModel, mujoco.mjtObj.mjOBJ_SENSOR, LINVEL_SENSOR)
        tLinvelAdr = tMjModel.sensor_adr[tLinvelId]
        tMjData = mujoco.MjData(tMjModel)
        tRenderer = mujoco.Renderer(tMjModel, ROLLOUT_RENDER_HEIGHT, ROLLOUT_RENDER_WIDTH)
        tCamera = mujoco.MjvCamera()
        tCamera.type = mujoco.mjtCamera.mjCAMERA_FREE
        tCamera.distance = ROLLOUT_CAMERA_DISTANCE
        tCamera.elevation = ROLLOUT_CAMERA_ELEVATION
        tCamera.azimuth = ROLLOUT_CAMERA_AZIMUTH
        tFrames, tXs, tYs, tZs, tForward = [], [], [], [], []
        for _ in range(ROLLOUT_STEPS):
            tState.info["command"] = tCommand
            tAct, _ = tPolicy(tState.obs, tJax.random.PRNGKey(0))
            tState = tStep(tState, tAct)
            tQpos = np.array(tState.data.qpos).reshape(-1)
            tSensorData = np.array(tState.data.sensordata).reshape(-1)
            tMjData.qpos[:] = tQpos
            tMjData.qvel[:] = np.array(tState.data.qvel).reshape(-1)
            mujoco.mj_forward(tMjModel, tMjData)
            tCamera.lookat[:] = [tQpos[0], tQpos[1], ROLLOUT_LOOKAT_HEIGHT]
            tRenderer.update_scene(tMjData, camera=tCamera)
            tFrames.append(tRenderer.render())
            tXs.append(float(tQpos[0]))
            tYs.append(float(tQpos[1]))
            tZs.append(float(tQpos[2]))
            tForward.append(float(tSensorData[tLinvelAdr]))
        imageio.mimsave(self.mOut + "_rollout.mp4", tFrames, fps=ROLLOUT_FPS)
        tRenderer.close()
        tDx = tXs[-1] - tXs[0]
        tDy = tYs[-1] - tYs[0]
        tNet = float((tDx * tDx + tDy * tDy) ** 0.5)        # net horizontal distance
        tMeanForward = float(np.mean(tForward))              # mean local forward velocity (cmd 0.6)
        with open(self.mOut + "_rollout.json", "w") as tFile:
            json.dump({"step": int(step), "x_travel": tNet, "dx": float(tDx),
                       "dy": float(tDy), "fwd_vel": tMeanForward,
                       "min_z": float(min(tZs))}, tFile)
        print(f"[walk]   rollout @ {int(step)}: net_dist={tNet:.2f}m "
              f"fwd_vel={tMeanForward:+.2f}(cmd0.6) min_z={min(tZs):.2f}", flush=True)


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.makedirs(os.path.dirname(tArgs.out), exist_ok=True)

    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import locomotion_params

    print(f"[walk] jax devices: {jax.devices()}")
    # Stack pinned to mujoco 3.4 + playground 0.1.0 -> MJX uses the classic JAX backend (no warp).
    tEnvConfig = registry.get_default_config(tArgs.env)
    tEnvConfig.reward_config.scales.termination = tArgs.termination
    tEnvConfig.reward_config.scales.tracking_lin_vel = tArgs.track_lin
    tEnvConfig.reward_config.scales.stand_still = tArgs.stand_still
    tEnv = registry.load(tArgs.env, config=tEnvConfig)
    print(f"[walk] termination={tEnvConfig.reward_config.scales.termination} "
          f"tracking_lin_vel={tEnvConfig.reward_config.scales.tracking_lin_vel} "
          f"stand_still={tEnvConfig.reward_config.scales.stand_still}")
    tPpoConfig = locomotion_params.brax_ppo_config(tArgs.env)
    tPpoConfig.num_timesteps = tArgs.timesteps
    tPpoConfig.num_envs = tArgs.num_envs
    tPpoConfig.num_evals = tArgs.num_evals   # more evals -> checkpoint saved more often
    print(f"[walk] env={tArgs.env} obs={tEnv.observation_size} act={tEnv.action_size}")
    print(f"[walk] timesteps={tPpoConfig.num_timesteps:,} num_envs={tPpoConfig.num_envs}")

    tMakeNetworks = functools.partial(ppo_networks.make_ppo_networks, **dict(tPpoConfig.network_factory))
    tCallbacks = TrainingCallbacks(tArgs.out, lambda: registry.load(tArgs.env, config=tEnvConfig), jax)

    # optionally resume from a previous checkpoint (continue training)
    tExtra = {}
    if tArgs.restore:
        with open(tArgs.restore, "rb") as tFile:
            tRestored = pickle.load(tFile)
        tExtra["restore_params"] = (tRestored["params"] if isinstance(tRestored, dict) and "params" in tRestored
                                    else tRestored)
        print(f"[walk] RESUMING from {tArgs.restore}", flush=True)
    # Domain randomisation, as in the official locomotion notebook. Without it the G1 policy overfits and does
    # not actually walk (the reward rises anyway). This was the missing piece.
    try:
        tRandomizationFn = registry.get_domain_randomizer(tArgs.env)
        print(f"[walk] domain randomizer: {tRandomizationFn is not None}", flush=True)
    except Exception as tError:
        tRandomizationFn = None
        print(f"[walk] no domain randomizer: {tError}", flush=True)
    if tRandomizationFn is not None:
        tExtra["randomization_fn"] = tRandomizationFn

    tTrainFn = functools.partial(
        ppo.train,
        **{tKey: tValue for tKey, tValue in tPpoConfig.items() if tKey not in ("network_factory",)},
        network_factory=tMakeNetworks,
        # mujoco_playground's brax wrapper (the MJX State is not a brax pipeline_state)
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=tCallbacks.progress,
        policy_params_fn=tCallbacks.saveCheckpoint,
        seed=tArgs.seed,
        **tExtra,
    )
    # separate eval_env (official recipe), with the same reward overrides
    tEvalEnv = registry.load(tArgs.env, config=tEnvConfig)
    _, tParams, _ = tTrainFn(environment=tEnv, eval_env=tEvalEnv)

    with open(tArgs.out + ".pkl", "wb") as tFile:
        pickle.dump(tParams, tFile)
    # also the env config, for reproducible rollouts
    with open(tArgs.out + "_cfg.pkl", "wb") as tFile:
        pickle.dump({"env": tArgs.env, "env_cfg": tEnvConfig.to_dict()}, tFile)
    print(f"[walk] saved params -> {tArgs.out}.pkl")
    print(f"[walk] total wall-clock: {tCallbacks.elapsedMin():.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
