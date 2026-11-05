import pytest

from g1nav.envs.scene_gen import sampleScene
from analysis import evidence as ev


@pytest.mark.parametrize("seed,instruction", ev.DEMO_EPISODES)
def test_demo_scenes_match_recorded_labels(seed, instruction):
    assert [o.label for o in sampleScene(seed).mObjects] == ev.cognition(seed, instruction)["labels"]


def test_deterministic():
    a, b = sampleScene(7), sampleScene(7)
    assert [(o.label, o.mX, o.mY) for o in a.mObjects] == [(o.label, o.mX, o.mY) for o in b.mObjects]
    assert (a.mRobotX, a.mRobotY, a.mRobotYaw) == (b.mRobotX, b.mRobotY, b.mRobotYaw)


def test_objects_inside_arena():
    for seed in range(20):
        s = sampleScene(seed)
        for o in s.mObjects:
            assert abs(o.mX) <= s.mArenaHalf and abs(o.mY) <= s.mArenaHalf
