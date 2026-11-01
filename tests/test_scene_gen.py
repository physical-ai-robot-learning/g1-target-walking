import pytest

from scene_gen import sample_scene
from analysis import evidence as ev


@pytest.mark.parametrize("seed,instruction", ev.DEMO_EPISODES)
def test_demo_scenes_match_recorded_labels(seed, instruction):
    assert [o.label for o in sample_scene(seed).objects] == ev.cognition(seed, instruction)["labels"]


def test_deterministic():
    a, b = sample_scene(7), sample_scene(7)
    assert [(o.label, o.x, o.y) for o in a.objects] == [(o.label, o.x, o.y) for o in b.objects]
    assert (a.robot_x, a.robot_y, a.robot_yaw) == (b.robot_x, b.robot_y, b.robot_yaw)


def test_objects_inside_arena():
    for seed in range(20):
        s = sample_scene(seed)
        for o in s.objects:
            assert abs(o.x) <= s.arena_half and abs(o.y) <= s.arena_half
