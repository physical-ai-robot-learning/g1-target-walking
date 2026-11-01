import numpy as np

from action_module import (REACH_RADIUS, Navigator, ego_to_world, goal_to_command,
                           world_to_ego)


def test_ego_world_roundtrip():
    gx, gy = ego_to_world(1.0, -2.0, 0.7, 25.0, 2.5)
    bearing, dist = world_to_ego(1.0, -2.0, 0.7, gx, gy)
    assert abs(bearing - 25.0) < 1e-9 and abs(dist - 2.5) < 1e-9


def test_stop_inside_radius():
    assert goal_to_command(10.0, REACH_RADIUS)[3] is True
    assert goal_to_command(10.0, REACH_RADIUS + 0.01)[3] is False


def test_unicycle_reaches_goal():
    nav = Navigator()
    nav.lock_goal_from_ego(0, 0, 0.0, 120.0, 2.5)    # target behind-left: must turn first
    pose = np.zeros(3)
    for _ in range(2000):
        vx, vy, wz, done = nav.command(*pose)
        if done:
            break
        pose += np.array([vx * np.cos(pose[2]), vx * np.sin(pose[2]), wz]) * 0.02
    assert done
    assert np.hypot(nav.goal_world[0] - pose[0], nav.goal_world[1] - pose[1]) <= REACH_RADIUS


def test_command_limits():
    nav = Navigator()
    nav.goal_world = (5.0, 5.0)
    for _ in range(200):
        vx, vy, wz, _ = nav.command(0.0, 0.0, 0.0)
        assert 0.0 <= vx <= 0.45 and abs(wz) <= 0.5 and vy == 0.0
