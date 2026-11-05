import numpy as np

from g1nav.navigation.action_module import REACH_RADIUS, Navigator, egoToWorld, goalToCommand, worldToEgo


def test_ego_world_roundtrip():
    gx, gy = egoToWorld(1.0, -2.0, 0.7, 25.0, 2.5)
    bearing, dist = worldToEgo(1.0, -2.0, 0.7, gx, gy)
    assert abs(bearing - 25.0) < 1e-9 and abs(dist - 2.5) < 1e-9


def test_stop_inside_radius():
    assert goalToCommand(10.0, REACH_RADIUS)[3] is True
    assert goalToCommand(10.0, REACH_RADIUS + 0.01)[3] is False


def test_unicycle_reaches_goal():
    nav = Navigator()
    nav.lockGoalFromEgo(0, 0, 0.0, 120.0, 2.5)    # target behind-left: must turn first
    pose = np.zeros(3)
    for _ in range(2000):
        vx, vy, wz, done = nav.command(*pose)
        if done:
            break
        pose += np.array([vx * np.cos(pose[2]), vx * np.sin(pose[2]), wz]) * 0.02
    assert done
    assert np.hypot(nav.mGoalWorld[0] - pose[0], nav.mGoalWorld[1] - pose[1]) <= REACH_RADIUS


def test_command_limits():
    nav = Navigator()
    nav.mGoalWorld = (5.0, 5.0)
    for _ in range(200):
        vx, vy, wz, _ = nav.command(0.0, 0.0, 0.0)
        assert 0.0 <= vx <= 0.45 and abs(wz) <= 0.5 and vy == 0.0
