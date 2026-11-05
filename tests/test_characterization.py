"""Characterization (golden) tests for the pure-Python parts of the pipeline.

The expected values in tests/golden/characterization.json were recorded from the code as it was before the
2026-11 layout refactor (commit 702a60a, `code/` tree) and must not change: the refactor renamed modules and
identifiers, it did not change behaviour. Regenerate only for an intended behaviour change:

    G1NAV_REGEN_GOLDEN=1 python -m pytest -q tests/test_characterization.py
"""
import hashlib
import json
import os

import numpy as np
import pytest

from conftest import ROOT

import action_module as am
import instructions as ins
import navigate as nav
import scene_gen as sg

GOLDEN = os.path.join(ROOT, "tests", "golden", "characterization.json")
REGEN = os.environ.get("G1NAV_REGEN_GOLDEN") == "1"
SEEDS = range(20)


def _scene(seed):
    s = sg.sample_scene(seed)
    return {"seed": s.seed, "arena_half": s.arena_half, "robot": [s.robot_x, s.robot_y, s.robot_yaw],
            "objects": [[o.color, o.shape, o.x, o.y, o.z, o.label] for o in s.objects]}


def _mjcf_sha(seed):
    xml = sg.build_mjcf(sg.sample_scene(seed))
    xml = xml.replace(os.path.join(sg.G1_DIR, "assets"), "<G1_DIR>/assets")
    return hashlib.sha256(xml.encode()).hexdigest()


def _instruction(seed):
    i = ins.sample_instruction(sg.sample_scene(seed), np.random.default_rng(1000 + seed))
    return [i.template_id, i.text, i.target_label, list(i.target_xy), i.params]


def _success_cases():
    scene = sg.sample_scene(3)
    rows = []
    for k in range(12):
        i = ins.sample_instruction(scene, np.random.default_rng(k))
        tx, ty = i.target_xy
        straight = [(tx - 2 + 0.25 * j, ty, 0.0) for j in range(9)]          # passes through the target
        turn_l = straight + [(tx + 0.1, ty, 0.0), (tx + 0.2, ty + 0.1, 1.3)]   # +74 deg after passing
        turn_r = straight + [(tx + 0.1, ty, 0.0), (tx + 0.2, ty - 0.1, -1.3)]
        far = [(tx + 5, ty + 5, 0.0), (tx + 4, ty + 5, 0.0)]
        rows.append([i.template_id, i.params, [i.succeeded(t, scene.objects) for t in (straight, turn_l, turn_r, far)]])
    return rows


def _commands():
    out = []
    for b in (-170.0, -90.0, -40.0, -39.9, -10.0, 0.0, 5.0, 39.9, 40.0, 120.0, 180.0):
        for d in (0.0, 0.3, 0.55, 0.56, 1.0, 3.0):
            out.append([b, d, list(am.goal_to_command(b, d))])
    return out


def _geometry():
    rows = []
    for rx, ry, ryaw, b, d in [(0, 0, 0.0, 25.0, 2.5), (1.0, -2.0, 0.7, 25.0, 2.5), (-2.65, 1.1, 3.0, -170.0, 0.4)]:
        gx, gy = am.ego_to_world(rx, ry, ryaw, b, d)
        rows.append([float(gx), float(gy), list(am.world_to_ego(rx, ry, ryaw, gx, gy))])
    q = [[0.1, -0.2, 0.75, 0.9, 0.1, -0.05, 0.42], [1, 2, 0.7, 1, 0, 0, 0], [0, 0, 0, 0.0, 0.0, 0.0, 1.0]]
    rows.append([list(am.robot_pose_from_qpos(np.array(x, dtype=float))) for x in q])
    rows.append([float(nav.quat_yaw(np.array(x[3:7], dtype=float))) for x in q])
    return rows


def _navigator_rollouts():
    """Kinematic unicycle driven by the Navigator (as in action_module's self-test and tests/test_navigator.py)."""
    rows = []
    for bearing, dist, dt in [(25.0, 2.5, 0.05), (120.0, 2.5, 0.02), (-150.0, 4.0, 0.02)]:
        n = am.Navigator()
        goal = n.lock_goal_from_ego(0, 0, 0.0, bearing, dist)
        pose = np.zeros(3)
        cmds = []
        for step in range(2000):
            vx, vy, wz, done = n.command(*pose)
            cmds.append((vx, vy, wz, done))
            pose[0] += vx * np.cos(pose[2]) * dt
            pose[1] += vx * np.sin(pose[2]) * dt
            pose[2] += wz * dt
            if done:
                break
        sha = hashlib.sha256(repr(cmds).encode()).hexdigest()
        rows.append([list(map(float, goal)), step, pose.tolist(), n.reached, sha])
    return rows


def _compute():
    return {
        "scenes": [_scene(s) for s in SEEDS],
        "mjcf_sha256": [_mjcf_sha(s) for s in range(3)],
        "instructions": [_instruction(s) for s in SEEDS],
        "success": _success_cases(),
        "commands": _commands(),
        "geometry": _geometry(),
        "navigator": _navigator_rollouts(),
        "constants": [am.VX_MAX, am.VY_MAX, am.WZ_MAX, am.REACH_RADIUS, ins.REACH_RADIUS, ins.PASS_RADIUS,
                      sorted(sg.COLORS.items()), sorted((k, list(v[1]), v[0], v[2]) for k, v in sg.SHAPES.items())],
    }


def _normalise(x):
    return json.loads(json.dumps(x))


@pytest.fixture(scope="module")
def golden():
    if REGEN:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            json.dump(_normalise(_compute()), f, indent=1)
    with open(GOLDEN) as f:
        return json.load(f)


@pytest.mark.parametrize("key", ["scenes", "mjcf_sha256", "instructions", "success", "commands", "geometry",
                                 "navigator", "constants"])
def test_matches_golden(golden, key):
    assert _normalise(_compute()[key]) == golden[key]

