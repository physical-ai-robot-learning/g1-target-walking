"""Navigator: closed-loop velocity commands toward a fixed world target (static scene).

Stage 3 of the pipeline, between target selection and the walking policy.
  1. ONCE: fix the target's world coordinate. In the recorded demos navigate.py sets `mGoalWorld` directly from the
     generated scene (ground truth). The intended alternative, not used in the demos, is `lockGoalFromEgo`: an
     egocentric (bearing, distance) estimate converted with the robot's spawn pose.
  2. EVERY CONTROL STEP: from the robot's current pose (odometry from MuJoCo qpos) recompute bearing/distance to
     that fixed point -> turn-then-go velocity command (vx, vy, wz), slew-rate limited -> the PPO walking policy
     turns it into 29 joint PD targets. Stop inside REACH_RADIUS.
Target selection is not repeated every step (the scene is static), which keeps the control loop cheap.

Input: robot pose (x, y, yaw) per step. Output: (vx, vy, wz, done). Pure numpy.

    python -m g1nav.navigation.action_module      # kinematic self-test: a unicycle reaches a locked goal
"""
from __future__ import annotations

import numpy as np

# Kept well inside the walking policy's stable command regime (constant forward 0.4 m/s was verified not to fall;
# a saturated 0.8 rad/s step yaw rate did topple it).
VX_MAX = 0.45            # m/s
VY_MAX = 0.3             # m/s (the Navigator never commands lateral motion)
WZ_MAX = 0.5             # rad/s
REACH_RADIUS = 0.55
TURN_GAIN = 1.0          # wz = TURN_GAIN * bearing (rad), clipped to WZ_MAX
FACING_DEG = 40.0        # |bearing| below this -> full forward speed
TURNING_VX_FRACTION = 0.25  # forward speed fraction while turning (keep moving while turning)
APPROACH_GAIN = 1.2      # vx <= APPROACH_GAIN * distance (slow down near the target)
VX_SLEW_PER_STEP = 0.04  # m/s per control step: the walking policy topples on a step change in command
WZ_SLEW_PER_STEP = 0.05  # rad/s per control step
SELF_TEST_DT = 0.05
SELF_TEST_STEPS = 400


# ---- geometry: egocentric (bearing, distance) <-> world --------------------

def egoToWorld(robotX, robotY, robotYaw, bearingDeg, distanceM):
    """Target world (x, y) from the robot pose + egocentric bearing (+left, deg) / distance (m)."""
    tHeading = robotYaw + np.deg2rad(bearingDeg)
    return (robotX + distanceM * np.cos(tHeading),
            robotY + distanceM * np.sin(tHeading))


def worldToEgo(robotX, robotY, robotYaw, targetX, targetY):
    """Egocentric (bearing_deg +left, distance_m) of the world point (targetX, targetY)."""
    tDx, tDy = targetX - robotX, targetY - robotY
    tDistance = float(np.hypot(tDx, tDy))
    tBearing = np.arctan2(tDy, tDx) - robotYaw
    tBearing = (tBearing + np.pi) % (2 * np.pi) - np.pi
    return float(np.rad2deg(tBearing)), tDistance


# ---- command policy: (bearing, distance) -> (vx, vy, wz) --------------------

def goalToCommand(bearingDeg, distanceM):
    """Turn-then-go controller. Returns (vx, vy, wz, done)."""
    if distanceM <= REACH_RADIUS:
        return 0.0, 0.0, 0.0, True
    tBearingRad = np.deg2rad(bearingDeg)
    tWz = float(np.clip(TURN_GAIN * tBearingRad, -WZ_MAX, WZ_MAX))
    tFacing = abs(bearingDeg) < FACING_DEG
    tVx = VX_MAX if tFacing else TURNING_VX_FRACTION * VX_MAX
    tVx = float(np.clip(min(tVx, APPROACH_GAIN * distanceM), 0.0, VX_MAX))
    return tVx, 0.0, tWz, False


# ---- odometry: robot base pose from MuJoCo qpos -----------------------------

def quatYaw(quat):
    """Yaw (rad) of a (qw, qx, qy, qz) quaternion."""
    tQw, tQx, tQy, tQz = quat
    return np.arctan2(2 * (tQw * tQz + tQx * tQy), 1 - 2 * (tQy * tQy + tQz * tQz))


def robotPoseFromQpos(qpos):
    """qpos[:7] = [x, y, z, qw, qx, qy, qz] (free-joint base). Returns (x, y, yaw)."""
    return float(qpos[0]), float(qpos[1]), float(quatYaw(qpos[3:7]))


class Navigator:
    """Drives the G1 to a fixed world goal through a command-conditioned walking policy.

    The policy itself (103-D Playground 'state' observation with the command injected -> 29 PD joint targets) is
    wired by the caller, because the observation construction depends on the environment used at deployment
    (navigate.py); `setPolicy` only stores it.
    """

    def __init__(self):
        self.mGoalWorld = None      # (x, y), fixed once
        self.mPolicyFn = None
        self.mReached = False
        self.mVx = 0.0              # slew-limited command state
        self.mWz = 0.0

    def setPolicy(self, policyFn):
        self.mPolicyFn = policyFn
        return self

    def lockGoalFromEgo(self, robotX, robotY, robotYaw, bearingDeg, distanceM):
        """Fix the goal from an egocentric estimate (not used in the recorded demos)."""
        self.mGoalWorld = egoToWorld(robotX, robotY, robotYaw, bearingDeg, distanceM)
        self.mReached = False
        return self.mGoalWorld

    def command(self, robotX, robotY, robotYaw):
        """Current velocity command (vx, vy, wz, done) toward the locked goal (odometry-based)."""
        if self.mGoalWorld is None:
            return 0.0, 0.0, 0.0, False
        tBearing, tDistance = worldToEgo(robotX, robotY, robotYaw, *self.mGoalWorld)
        tVx, tVy, tWz, tDone = goalToCommand(tBearing, tDistance)
        # Slew-rate limit both channels: the walking policy topples on a step change in command. (This, together
        # with the observation-normalisation fix, is what stops the robot from falling mid-turn.)
        self.mVx += float(np.clip(tVx - self.mVx, -VX_SLEW_PER_STEP, VX_SLEW_PER_STEP))
        self.mWz += float(np.clip(tWz - self.mWz, -WZ_SLEW_PER_STEP, WZ_SLEW_PER_STEP))
        if tDone:
            self.mReached = True
        return self.mVx, tVy, self.mWz, tDone


def main(argv: list[str] | None = None) -> int:
    """Kinematic sanity test: a unicycle driven by the Navigator reaches a goal seen at +25 deg, 2.5 m."""
    tNavigator = Navigator()
    tGoalX, tGoalY = tNavigator.lockGoalFromEgo(0, 0, 0.0, 25.0, 2.5)
    print(f"locked goal world=({tGoalX:.2f},{tGoalY:.2f})")
    tPose = np.array([0.0, 0.0, 0.0])
    for tStep in range(SELF_TEST_STEPS):
        tVx, _, tWz, tDone = tNavigator.command(*tPose)
        tPose[0] += tVx * np.cos(tPose[2]) * SELF_TEST_DT
        tPose[1] += tVx * np.sin(tPose[2]) * SELF_TEST_DT
        tPose[2] += tWz * SELF_TEST_DT
        if tDone:
            print(f"reached in {tStep} steps at ({tPose[0]:.2f},{tPose[1]:.2f})")
            return 0
    print(f"did not reach; ended at ({tPose[0]:.2f},{tPose[1]:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
