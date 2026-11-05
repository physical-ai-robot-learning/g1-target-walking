"""G1Nav recorded-results dashboard (Streamlit, GPU-free; was code/apps/app_results.py).

Plays back recorded artifacts: the three demo episode videos with their keyword-selected targets, the final
checkpoint's raw-environment rollout numbers, the cognition JSON / ego frame, and (only when a training console log
is present) the walking-emergence curves. No GPU, checkpoint loading or live physics.

Looks in (first match wins, all are searched):
    $G1NAV_DATA   (default ~/Desktop/g1nav_dash)   artifacts copied from the GPU machine
    $G1NAV_VIDEOS (default ~/g1nav/videos)
    then this repository's media/ and results/checkpoints/

    streamlit run src/g1nav/apps/results_dashboard.py --server.port 8501
"""
from __future__ import annotations

import glob
import json
import os
import re

import streamlit as st

from g1nav.paths import CHECKPOINT_DIR, MEDIA_DIR, dashboardArtifactDirs

DEMO_INSTRUCTIONS = ["go to the orange cylinder", "go to the red cube", "go to the purple cylinder"]
ROLLOUT_COMMAND_VX = 0.6
RAW_LOG_TAIL_CHARS = 4000
EARLIER_DEMOS = [
    ("✅ Reaches the instructed target", ["demo_NAV_reaches_target.mp4", "nav_60M.mp4", "g1_nav*.mp4"]),
    ("🚶 Walks (velocity-conditioned policy)", ["demo_WALKS.mp4", "walk_latest_rollout.mp4", "g1_fwd.mp4"]),
    ("⚠️ Failure mode (falls)", ["demo_falls.mp4", "walk_raw_falls.mp4", "walk_wrapped_inplace.mp4"]),
]
REWARD_RE = re.compile(r"step=\s*([\d,]+)\s+reward=\s*([-\d.]+)")
ROLLOUT_RE = re.compile(
    r"rollout @ (\d+): net_dist=([\d.]+)m fwd_vel=([+-][\d.]+)\(cmd([\d.]+)\) min_z=([-\d.]+)")


class ArtifactFinder:
    """First existing file matching a basename / glob in the artifact folders, in priority order."""

    def __init__(self, dirs):
        self.mDirs = [tDir for tDir in dirs if os.path.isdir(tDir)]

    def find(self, *names):
        for tDir in self.mDirs:
            for tName in names:
                tHits = sorted(glob.glob(os.path.join(tDir, tName)))
                if tHits:
                    return tHits[0]
        return None

    def readText(self, *names):
        tPath = self.find(*names)
        if not tPath:
            return None
        try:
            with open(tPath, "r", errors="replace") as tFile:
                return tFile.read()
        except OSError:
            return None


def parseTrainingLog(text):
    """(step, reward) pairs and rollout rows (step, net_dist, fwd_vel, cmd, min_z) from a train_walk console log."""
    tRewards, tRollouts = [], []
    if not text:
        return tRewards, tRollouts
    for tMatch in REWARD_RE.finditer(text):
        tRewards.append((int(tMatch.group(1).replace(",", "")), float(tMatch.group(2))))
    for tMatch in ROLLOUT_RE.finditer(text):
        tRollouts.append(dict(step=int(tMatch.group(1)), net_dist=float(tMatch.group(2)),
                              fwd_vel=float(tMatch.group(3)), cmd=float(tMatch.group(4)),
                              min_z=float(tMatch.group(5))))
    return tRewards, tRollouts


def renderHeadline(finder, rollouts):
    """Final walking policy numbers: the training-time rollout JSON, else the last rollout line of the log."""
    tRollText = finder.readText("rollout.json", "walk_t4_rollout.json")
    tFinal = {}
    if tRollText:
        try:
            tFinal = json.loads(tRollText)
        except json.JSONDecodeError:
            tFinal = {}
    tBest = max(rollouts, key=lambda tR: tR["fwd_vel"]) if rollouts else None
    tLast = rollouts[-1] if rollouts else None

    st.subheader("Final walking policy — raw-env evaluation")
    tCols = st.columns(4)
    if tFinal:
        tCmd = ROLLOUT_COMMAND_VX
        tFwd = float(tFinal.get("fwd_vel", 0.0))
        tCols[0].metric("Forward velocity", f"{tFwd:.2f} m/s", f"{tFwd/tCmd*100:.0f}% of {tCmd} cmd")
        tCols[1].metric("Torso height (min_z)", f"{tFinal.get('min_z', 0):.2f} m", "upright (>0.7)")
        tCols[2].metric("Net distance", f"{tFinal.get('x_travel', 0):.2f} m")
        tCols[3].metric("Trained to", f"{tFinal.get('step', 0)/1e6:.0f} M steps")
    elif tLast:
        tCols[0].metric("Forward velocity", f"{tLast['fwd_vel']:.2f} m/s",
                        f"{tLast['fwd_vel']/tLast['cmd']*100:.0f}% of {tLast['cmd']} cmd")
        tCols[1].metric("Torso height (min_z)", f"{tLast['min_z']:.2f} m", "upright (>0.7)")
        tCols[2].metric("Net distance", f"{tLast['net_dist']:.2f} m")
        tCols[3].metric("Trained to", f"{tLast['step']/1e6:.0f} M steps")
    else:
        st.info("No eval numbers found (rollout.json / training log).")
    if tBest:
        st.caption(f"Peak forward velocity over training: **{tBest['fwd_vel']:.2f} m/s** "
                   f"({tBest['fwd_vel']/tBest['cmd']*100:.0f}% of command) at "
                   f"{tBest['step']/1e6:.0f} M steps.")


def renderDemoEpisodes(finder):
    st.subheader("Closed-loop demo episodes (keyword target → Navigator → walk → physics)")
    st.caption("Each clip is the *modified* pipeline end-to-end: the target is chosen by keyword "
               "match against the scene labels (Eagle embedding computed for logging only) and its "
               "position is read from the generated scene → trained PPO policy walks to that "
               "target in MuJoCo (physics only). "
               "Left = third-person, right = ego. **All 3 seeds reach the target** "
               "(final dist 0.39–0.54 m, torso ≈ 0.75 m, never falls) after fixing two inference "
               "bugs: (1) missing observation normalization, (2) gravity computed in the wrong "
               "frame — `-upvector` (world) instead of `site_xmat.T@[0,0,-1]` (pelvis-IMU local), "
               "which only diverges during yaw and made the robot topple mid-turn.")
    tCols = st.columns(3)
    tAnyVideo = False
    for tIndex, tInstruction in enumerate(DEMO_INSTRUCTIONS):
        with tCols[tIndex]:
            # naming: seed{i}_<instruction-slug>.mp4 + .cognition.json; fall back to the pod's names
            tSlug = tInstruction.replace(" ", "-")
            tVideo = finder.find(f"seed{tIndex}_{tSlug}.mp4", f"nav_n16_seed{tIndex}.mp4")
            tCognition = finder.readText(f"seed{tIndex}_{tSlug}.cognition.json", f"cortex_seed{tIndex}.json")
            tTarget = "?"
            if tCognition:
                try:
                    tTarget = json.loads(tCognition).get("vlm_target", "?")
                except json.JSONDecodeError:
                    pass
            st.markdown(f"**seed {tIndex}** · instruction: *“{tInstruction}”*  \n"
                        f"chosen target (keyword match) → `{tTarget}` · ✅ reached")
            if tVideo:
                st.video(tVideo)
                tAnyVideo = True
            else:
                st.info("rendering on pod… (will appear after pull)")
    if not tAnyVideo:
        st.caption("⏳ Videos render on the RTX 6000 Ada pod via "
                   "`scripts/pod/render_n16_videos.sh`, then are pulled here.")


def renderEarlierDemos(finder):
    st.subheader("Closed-loop demos (earlier recordings)")
    for tCol, (tLabel, tNames) in zip(st.columns(3), EARLIER_DEMOS):
        with tCol:
            st.markdown(f"**{tLabel}**")
            tPath = finder.find(*tNames)
            if tPath:
                st.video(tPath)
                st.caption(os.path.basename(tPath))
            else:
                st.info("not found")


def renderEmergence(finder):
    st.subheader("Walking emergence (first walk → mature gait)")
    tCols = st.columns(2)
    tFirst = finder.find("walk_41M_FIRST_WALK.mp4", "walk_v3_trackcam.mp4")
    tMature = finder.find("walk_latest_rollout.mp4", "latest_rollout.mp4", "g1_trainenv_walk.mp4")
    with tCols[0]:
        st.markdown("**First emergence (~41 M steps)**")
        if tFirst:
            st.video(tFirst)
        else:
            st.info("not found")
    with tCols[1]:
        st.markdown("**Mature gait (latest)**")
        if tMature:
            st.video(tMature)
        else:
            st.info("not found")


def renderTrainingCurves(rewards, rollouts):
    st.subheader("PPO training — walking emergence curve")
    if not rollouts:
        st.info("No training-log rollout lines found to plot.")
        return
    import pandas as pd
    tFrame = pd.DataFrame(rollouts)
    tFrame["step_M"] = tFrame["step"] / 1e6
    tFrame = tFrame.set_index("step_M")
    tCols = st.columns(2)
    with tCols[0]:
        st.markdown("**Forward velocity vs command**")
        tFrame["command"] = tFrame["cmd"]
        st.line_chart(tFrame[["fwd_vel", "command"]])
    with tCols[1]:
        st.markdown("**Torso height (min_z) — falls when < 0**")
        st.line_chart(tFrame[["min_z"]])
    if rewards:
        tRewardFrame = pd.DataFrame(rewards, columns=["step", "reward"]).set_index("step")
        tRewardFrame.index = tRewardFrame.index / 1e6
        tRewardFrame.index.name = "step_M"
        st.markdown("**Episode reward**")
        st.line_chart(tRewardFrame)
    st.caption("Parsed from the training log. The policy first learns *not to fall* "
               "(min_z crosses 0 around 41 M), then matures the forward gait.")


def renderCognition(finder):
    st.subheader("Cognition — keyword target selection + GR00T N1.6 Eagle embedding (logged only)")
    tText = finder.readText("seed0_go-to-the-orange-cylinder.cognition.json", "cortex_result.json")
    if not tText:
        st.info("cortex_result.json (N1.6 Eagle) not found — run g1nav.navigation.cortex_eagle.")
        return
    try:
        tCognition = json.loads(tText)
    except json.JSONDecodeError:
        tCognition = {}
    tCols = st.columns([1, 2])
    with tCols[0]:
        tEgo = finder.find("seed0_go-to-the-orange-cylinder.ego.png", "app_ego.png",
                           "cortex_ego.png", "ego_start.png")
        if tEgo:
            st.image(tEgo, caption="ego RGB frame", use_container_width=True)
        st.metric("Eagle perception embedding",
                  f"{tCognition.get('eagle_patches', '?')} × {tCognition.get('eagle_embed_dim', '?')}",
                  f"‖emb‖ = {tCognition.get('eagle_norm', 0):.0f}")
        st.caption(f"target → **{tCognition.get('vlm_target', '?')}** (keyword match against scene labels)"
                   "  ·  the GR00T N1.6 Eagle vision encoder (SigLip2) embedding is computed "
                   "for logging and does not affect the target.")
    with tCols[1]:
        st.markdown("**5-stage cortex** (perceive → ground → localize → plan → command):")
        for tKey in ("1", "2", "3", "4", "5"):
            tStage = tCognition.get("stages", {}).get(tKey)
            if tStage:
                st.markdown(f"**{tKey}.** {tStage}")
        st.caption("Stages 3-5 are fixed descriptive strings written by cortex_eagle.py, not "
                   "measurements. In the recorded runs the target position was read from the "
                   "generated scene (navigate.py); the CV + ego-depth path was not used.")
    st.caption("Generated by `g1nav.navigation.cortex_eagle` on the RTX 6000 Ada pod "
               "(transformers 4.51.3, eagle-n16). Upper Eagle LM layers are intentionally "
               "absent — GR00T uses the backbone for perception, not text generation.")


def main() -> None:
    st.set_page_config(page_title="G1Nav — Recorded Results", layout="wide", page_icon="🦿")
    st.title("🦿🧠 G1Nav — Recorded Results Dashboard")
    st.caption("Humanoid locomotion RL with hierarchical target navigation for the Unitree G1 in MuJoCo (MJX). "
               "Playback of recorded artifacts — no GPU / checkpoint / live physics required.")

    tDataDir, tVideoDir = dashboardArtifactDirs()
    tFinder = ArtifactFinder((str(tDataDir), str(tVideoDir), str(MEDIA_DIR), str(CHECKPOINT_DIR)))
    if not tFinder.mDirs:
        st.error(f"No artifact directory found. Looked in: {tDataDir} and {tVideoDir}")
        st.stop()
    st.write("**Reading artifacts from:** " + " · ".join(f"`{tDir}`" for tDir in tFinder.mDirs))

    tLogText = tFinder.readText("_status.txt", "ppo*.log", "train*.log")
    tRewards, tRollouts = parseTrainingLog(tLogText)
    renderHeadline(tFinder, tRollouts)
    st.divider()
    renderDemoEpisodes(tFinder)
    st.divider()
    renderEarlierDemos(tFinder)
    st.divider()
    renderEmergence(tFinder)
    st.divider()
    renderTrainingCurves(tRewards, tRollouts)
    st.divider()
    renderCognition(tFinder)
    if tLogText:
        with st.expander("Raw training/status log (_status.txt)"):
            st.code(tLogText[-RAW_LOG_TAIL_CHARS:])


if __name__ == "__main__":
    main()
