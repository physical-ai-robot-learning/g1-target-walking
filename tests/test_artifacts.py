import json
import os

import pytest

from analysis import evidence as ev


def test_checkpoint_metadata():
    s = ev.checkpoint_summary(os.path.join(ev.CKPT_DIR, "walk_t4_latest.pkl"))
    assert s["step"] == 154_828_800
    assert s["policy_layers"][0] == (103, 512) and s["policy_layers"][-1] == (128, 58)
    assert s["value_layers"][0] == (216, 512) and s["value_layers"][-1] == (128, 1)


def test_checkpoint_steps_match_metrics():
    steps = ev.checkpoint_steps()
    assert steps == [m["step"] for m in ev.metrics()]
    assert len(steps) == 16 and steps[-1] == 154_828_800
    for p in steps:
        assert ev.load_checkpoint(os.path.join(ev.CKPT_DIR, f"walk_t4_step{p:010d}.pkl"))["step"] == p


def test_rollout_is_final_checkpoint():
    r = ev.rollout()
    assert r["step"] == 154_828_800 and r["min_z"] > 0.7 and 0.5 < r["fwd_vel"] < 0.6


@pytest.mark.parametrize("seed,instruction", ev.DEMO_EPISODES)
def test_videos_and_rerun(seed, instruction):
    pre = os.path.join(ev.VIDEO_DIR, ev.demo_prefix(seed, instruction))
    for ext in (".mp4", ".gif", ".ego.png", ".cognition.json"):
        assert os.path.exists(pre + ext)
    frames = ev.mp4_frame_count(pre + ".mp4")
    run = {r["seed"]: r for r in ev.rerun_episodes()}[seed]
    assert run["reached"] and not run["fell"] and run["final_dist_m"] <= ev.REACH_RADIUS_M
    assert run["target"] == ev.cognition(seed, instruction)["vlm_target"]
    # 2 s pre-roll + navigation (<= 12 s) + 2 s post-roll at 50 Hz
    assert 2 * ev.HOLD_FRAMES < frames <= 2 * ev.HOLD_FRAMES + 12 * ev.CTRL_HZ
    assert abs(frames - run["frames"]) <= 1


def test_sweep_covers_all_checkpoints():
    with open(ev.SWEEP_JSON) as f:
        rows = json.load(f)
    assert [r["step"] for r in rows] == ev.checkpoint_steps()
    walks = [r["verdict"] == "WALKS" for r in rows]
    first = walks.index(True)
    assert all(walks[first:]) and rows[first]["step"] == 41_287_680
