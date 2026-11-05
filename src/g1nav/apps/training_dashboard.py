"""Streamlit monitor for the PPO walking training (was code/locomotion/app_train.py).

Shows the training curve (eval reward vs step) parsed from <WALK_OUT>_metrics.jsonl, the latest checkpoint's step,
and a button that rolls out the CURRENT policy (g1nav.locomotion.rollout_walk, on CPU so it does not fight the
GPU-bound trainer) into a short video. $WALK_OUT is the --out prefix of train_walk (default
<pod checkpoint folder>/walk_full).

Run on the training machine (separate port from the other dashboards), e.g.:
    streamlit run src/g1nav/apps/training_dashboard.py --server.port 8080 \
        --server.address 0.0.0.0 --server.headless true --server.fileWatcherType none
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys

import streamlit as st

from g1nav.paths import WORKSPACE_CKPT_DIR

ROLLOUT_TIMEOUT_S = 600
ROLLOUT_OUTPUT_TAIL_CHARS = 1500


def outPrefix() -> str:
    return os.environ.get("WALK_OUT", str(WORKSPACE_CKPT_DIR / "walk_full"))


def loadMetrics(metricsPath):
    """Rows of the metrics JSONL; unparsable lines are skipped."""
    tRows = []
    if os.path.exists(metricsPath):
        with open(metricsPath) as tFile:
            for tLine in tFile:
                tLine = tLine.strip()
                if tLine:
                    try:
                        tRows.append(json.loads(tLine))
                    except Exception:
                        pass
    return tRows


def renderCurve(rows):
    tCol1, tCol2, tCol3 = st.columns(3)
    if rows:
        tLast = rows[-1]
        tCol1.metric("Latest step", f"{tLast['step']:,}")
        tCol2.metric("Latest reward", f"{tLast['reward']:.2f}")
        tCol3.metric("Elapsed", f"{tLast['elapsed_min']:.1f} min")
    else:
        st.info("No metrics yet — training is still compiling / starting.")
    if rows:
        st.subheader("Training curve")
        tSteps = [tRow["step"] for tRow in rows]
        tRewards = [tRow["reward"] for tRow in rows]
        try:
            import pandas as pd
            st.line_chart(pd.DataFrame({"step": tSteps, "reward": tRewards}).set_index("step"))
        except Exception:
            st.line_chart({"reward": tRewards})


def renderRollout(latestPath, rolloutMp4):
    st.subheader("Watch the current policy walk")
    if not os.path.exists(latestPath):
        st.info("No checkpoint saved yet — appears after the first eval "
                "(give it a few minutes after step 0).")
        return
    try:
        with open(latestPath, "rb") as tFile:
            tStep = pickle.load(tFile).get("step")
        st.write(f"Latest checkpoint at **step {tStep:,}**")
    except Exception:
        st.write("Latest checkpoint present.")
    tCols = st.columns(4)
    tVx = tCols[0].number_input("vx (m/s)", 0.0, 1.0, 0.5, 0.1)
    tWz = tCols[1].number_input("yaw rate", -1.0, 1.0, 0.0, 0.1)
    tSeconds = tCols[2].number_input("seconds", 2.0, 10.0, 6.0, 1.0)
    if tCols[3].button("▶ Roll out", type="primary"):
        with st.spinner("Rolling out current policy (renders video)…"):
            tCmd = [sys.executable, "-m", "g1nav.locomotion.rollout_walk",
                    "--ckpt", latestPath, "--vx", str(tVx), "--wz", str(tWz),
                    "--seconds", str(tSeconds), "--out", rolloutMp4]
            # roll out on CPU so it does not fight the GPU-bound PPO trainer
            tEnv = dict(os.environ, MUJOCO_GL="egl", JAX_PLATFORMS="cpu")
            tRun = subprocess.run(tCmd, capture_output=True, text=True, env=tEnv, timeout=ROLLOUT_TIMEOUT_S)
        st.code((tRun.stdout + "\n" + tRun.stderr)[-ROLLOUT_OUTPUT_TAIL_CHARS:])
    if os.path.exists(rolloutMp4):
        st.video(rolloutMp4)


def main() -> None:
    st.set_page_config(page_title="G1Nav · PPO Training", layout="wide")
    tPrefix = outPrefix()
    st.title("🦿 G1Nav — PPO Walking Training")
    st.caption("Low-level walking policy (the 'spinal cord'). Trained ONCE, reused "
               "for every instruction. Target selection is separate (keyword match; no training).")
    renderCurve(loadMetrics(tPrefix + "_metrics.jsonl"))
    st.divider()
    renderRollout(tPrefix + "_latest.pkl", str(WORKSPACE_CKPT_DIR / "rollout_dash.mp4"))
    st.caption("Refresh the page to update the curve / metrics.")


if __name__ == "__main__":
    main()
