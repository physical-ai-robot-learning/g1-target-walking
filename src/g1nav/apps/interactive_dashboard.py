"""G1Nav interactive dashboard (Streamlit; was code/apps/app_interactive.py): see a scene, then instruct the robot.

Step 1: pick a scene seed and **Load scene** -> renders the start ego frame and lists the objects in that arena.
Step 2: type an instruction referring to one of those objects and **Run** -> the full pipeline runs:
            instruction + ego frame
              -> target picked by keyword match against the scene labels
                 (GR00T N1.6 Eagle embedding computed for logging only)
              -> Navigator -> trained PPO walking policy -> MuJoCo physics
              -> ego || third-person video (physics only)

Two execution modes, auto-detected (override with G1NAV_MODE=local|remote):
  LOCAL  — a GPU + the simulation stack + the checkpoint are present: runs g1nav.navigation.pipeline here.
  REMOTE — otherwise runs scripts/pod/*.sh on an SSH-reachable GPU machine that has this repository at
           $G1NAV_POD_WORKSPACE (default /workspace) and pulls the video back. Configure the machine with
           G1NAV_POD_HOST / G1NAV_POD_PORT / G1NAV_POD_KEY (no real host is built in).

    streamlit run src/g1nav/apps/interactive_dashboard.py --server.port 8502
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass

import streamlit as st

from g1nav.paths import LATEST_CHECKPOINT, OUTPUTS_DIR, REPO_ROOT

PREVIEW_SCRIPT = os.path.join("scripts", "pod", "preview_scene.sh")
RUN_SCRIPT = os.path.join("scripts", "pod", "run_pipeline_n16.sh")
SSH_CONNECT_TIMEOUT_S = 20
RUN_TIMEOUT_S = 900
PREVIEW_TIMEOUT_S = 120
CHECK_TIMEOUT_S = 25
RAW_OUTPUT_TAIL_CHARS = 6000
MAX_SEED = 9999


@dataclass(frozen=True)
class PodConfig:
    """SSH target for REMOTE mode (from G1NAV_POD_HOST / _PORT / _KEY / _WORKSPACE)."""

    mHost: str
    mPort: str
    mKey: str
    mWorkspace: str

    @staticmethod
    def fromEnv() -> "PodConfig":
        return PodConfig(mHost=os.environ.get("G1NAV_POD_HOST", "root@your-gpu-host"),
                         mPort=os.environ.get("G1NAV_POD_PORT", "22"),
                         mKey=os.path.expanduser(os.environ.get("G1NAV_POD_KEY", "~/.ssh/id_ed25519")),
                         mWorkspace=os.environ.get("G1NAV_POD_WORKSPACE", "/workspace"))

    def sshCommand(self) -> list[str]:
        return ["ssh", "-p", self.mPort, "-i", self.mKey, "-o", "StrictHostKeyChecking=no",
                "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}", "-o", "BatchMode=yes"]

    def scpCommand(self) -> list[str]:
        return ["scp", "-P", self.mPort, "-i", self.mKey, "-o", "StrictHostKeyChecking=no",
                "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}"]

    def remotePath(self, relPath: str) -> str:
        return f"{self.mWorkspace}/{relPath}"


def detectMode() -> str:
    """'local' if the simulation stack imports, a GPU is visible and the checkpoint exists; else 'remote'."""
    tForced = os.environ.get("G1NAV_MODE")
    if tForced in ("local", "remote"):
        return tForced
    try:
        import jax
        import mujoco_playground  # noqa: F401
        tHasGpu = any(tDevice.platform == "gpu" for tDevice in jax.devices())
    except Exception:  # noqa: BLE001
        tHasGpu = False
    tHasCheckpoint = os.path.exists(LATEST_CHECKPOINT)
    return "local" if (tHasGpu and tHasCheckpoint) else "remote"


def grab(output: str, key: str):
    """Value of the first `KEY=value` line in a runner's output."""
    tMatch = re.search(rf"{key}=(.+)", output)
    return tMatch.group(1).strip() if tMatch else None


# ───────────────────────────── LOCAL backend ──────────────────────────────
def localPreview(seed):
    from g1nav.navigation import pipeline

    tOutDir = str(OUTPUTS_DIR)
    os.makedirs(tOutDir, exist_ok=True)
    tEgo = os.path.join(tOutDir, f"ego_seed{seed}.png")
    tLabels = pipeline.renderScene(int(seed), tEgo)
    return ",".join(tLabels), (tEgo if os.path.exists(tEgo) else None)


def localRun(seed, instruction):
    tCmd = ["python", "-m", "g1nav.navigation.pipeline", "--seed", str(int(seed)), "--instruction", instruction]
    tRun = subprocess.run(tCmd, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=RUN_TIMEOUT_S)
    tOut = tRun.stdout + "\n" + tRun.stderr
    tVideo = grab(tOut, "VIDEO")
    tResultJson = grab(tOut, "RESULT_JSON")
    return tOut, (tVideo if tVideo and os.path.exists(tVideo) else None), \
        (tResultJson if tResultJson and os.path.exists(tResultJson) else None)


# ───────────────────────────── REMOTE backend ─────────────────────────────
class RemoteBackend:
    """Runs the pod scripts over SSH and copies the results back."""

    def __init__(self, pod: PodConfig):
        self.mPod = pod

    def sshRun(self, cmd, timeout=RUN_TIMEOUT_S):
        return subprocess.run(self.mPod.sshCommand() + [self.mPod.mHost, cmd], capture_output=True, text=True,
                              timeout=timeout)

    def push(self, relPath):
        tRemote = self.mPod.remotePath(relPath)
        subprocess.run(self.mPod.sshCommand() + [self.mPod.mHost, f"mkdir -p {os.path.dirname(tRemote)}"],
                       capture_output=True, text=True, timeout=30)
        subprocess.run(self.mPod.scpCommand() + [os.path.join(str(REPO_ROOT), relPath), f"{self.mPod.mHost}:{tRemote}"],
                       capture_output=True, text=True, timeout=60)
        return tRemote

    def pull(self, remote, local, timeout=120):
        subprocess.run(self.mPod.scpCommand() + [f"{self.mPod.mHost}:{remote}", local], capture_output=True,
                       text=True, timeout=timeout)

    def preview(self, seed):
        tRemoteScript = self.push(PREVIEW_SCRIPT)
        tRun = self.sshRun(f"bash {tRemoteScript} {int(seed)}", timeout=PREVIEW_TIMEOUT_S)
        tOut = tRun.stdout + "\n" + tRun.stderr
        tObjects = grab(tOut, "OBJECTS")
        tRemoteEgo = grab(tOut, "EGO")
        tLocalEgo = None
        if tRemoteEgo:
            tLocalEgo = os.path.join(tempfile.mkdtemp(prefix="g1nav_prev_"), os.path.basename(tRemoteEgo))
            self.pull(tRemoteEgo, tLocalEgo, 60)
            if not os.path.exists(tLocalEgo):
                tLocalEgo = None
        return tObjects, tLocalEgo

    def run(self, seed, instruction):
        tRemoteScript = self.push(RUN_SCRIPT)
        tRun = self.sshRun(f"bash {tRemoteScript} {int(seed)} {shlex.quote(instruction)}", timeout=RUN_TIMEOUT_S)
        tOut = tRun.stdout + "\n" + tRun.stderr
        tRemoteVideo = grab(tOut, "VIDEO")
        tRemoteJson = grab(tOut, "RESULT_JSON")
        tLocalVideo = tLocalJson = None
        if tRemoteVideo:
            tTmp = tempfile.mkdtemp(prefix="g1nav_ui_")
            tLocalVideo = os.path.join(tTmp, os.path.basename(tRemoteVideo))
            self.pull(tRemoteVideo, tLocalVideo, 120)
            if tRemoteJson:
                tLocalJson = os.path.join(tTmp, os.path.basename(tRemoteJson))
                self.pull(tRemoteJson, tLocalJson, 60)
        return tOut, (tLocalVideo if tLocalVideo and os.path.exists(tLocalVideo) else None), \
            (tLocalJson if tLocalJson and os.path.exists(tLocalJson) else None)


# ───────────────────────────────── UI ─────────────────────────────────────
def renderSidebar(mode, pod, remote):
    with st.sidebar:
        st.header("Execution")
        st.metric("Mode", mode.upper())
        if mode == "local":
            st.caption("Running the pipeline on THIS machine's GPU.")
        else:
            st.caption(f"Running on remote GPU pod `{pod.mHost}` (this machine has no usable GPU).")
            if st.button("Check pod"):
                try:
                    tRun = remote.sshRun("echo OK; nvidia-smi --query-gpu=memory.free --format=csv,noheader",
                                         CHECK_TIMEOUT_S)
                    tOk = "OK" in tRun.stdout
                    (st.success if tOk else st.error)((tRun.stdout + tRun.stderr).strip())
                except Exception as tError:  # noqa: BLE001
                    st.error(str(tError))
        st.caption("Override with env var G1NAV_MODE=local|remote.")


def renderSceneStep(state, previewFn):
    st.subheader("Step 1 — choose a scene")
    tCol1, tCol2 = st.columns([1, 1])
    with tCol1:
        tSeed = st.number_input("Scene seed", min_value=0, max_value=MAX_SEED, value=0, step=1,
                                help="Each seed is a deterministic 3-object arena.")
    with tCol2:
        st.write("")
        st.write("")
        tLoad = st.button("🔄 Load scene", type="secondary")

    if tLoad:
        with st.status("Rendering the scene…", expanded=False) as tStatus:
            try:
                tObjects, tEgo = previewFn(int(tSeed))
                state.scene_seed = int(tSeed)
                state.objects = tObjects
                state.ego_path = tEgo
                tStatus.update(label="Scene loaded", state="complete")
                if not tObjects:
                    st.error("Could not read scene objects.")
            except Exception as tError:  # noqa: BLE001
                tStatus.update(label="Error", state="error")
                st.exception(tError)

    if state.objects:
        st.success(f"**Scene seed {state.scene_seed}** — objects in this arena:")
        tCols = st.columns([1, 1])
        with tCols[0]:
            if state.ego_path:
                st.image(state.ego_path, caption="ego view (robot's start view)", use_container_width=True)
        with tCols[1]:
            for tObject in [tX.strip() for tX in state.objects.split(",") if tX.strip()]:
                st.markdown(f"- **{tObject}**")
            st.caption("Type an instruction below referring to one of these.")


def renderResult(instruction, output, video, resultJson):
    tTarget = grab(output, "TARGET")
    tNavLine = next((tLine for tLine in output.splitlines() if tLine.startswith("NAV ")), "")
    tFell = "FELL" in output
    st.subheader("Result")
    tCol1, tCol2 = st.columns([3, 2])
    with tCol1:
        if video and os.path.getsize(video) > 0:
            st.video(video)
            st.caption("Left = third-person · Right = ego.  2s stand → walk → 2s stand.")
        else:
            st.error("No video came back — see raw output.")
    with tCol2:
        (st.error if tFell else st.success)(
            "⚠️ The robot FELL before reaching the target." if tFell
            else "✅ Reached the target (stayed upright).")
        st.metric("You asked", instruction)
        st.metric("chosen target (keyword match)", tTarget or "?")
        if tNavLine:
            st.code(tNavLine, language="text")
        if resultJson:
            with st.expander("Cognition (cortex_result.json)"):
                with open(resultJson) as tFile:
                    st.json(json.load(tFile))
    with st.expander("Raw output"):
        st.code(output[-RAW_OUTPUT_TAIL_CHARS:], language="text")


def renderInstructionStep(state, runFn):
    st.subheader("Step 2 — give an instruction")
    tReady = state.objects is not None
    tDefault = ""
    if tReady:
        tFirst = [tX.strip() for tX in state.objects.split(",") if tX.strip()]
        tDefault = f"go to the {tFirst[0]}" if tFirst else ""
    tInstruction = st.text_input("Instruction", value=tDefault, disabled=not tReady,
                                 placeholder="Load a scene first (Step 1)…",
                                 help="e.g. 'go to the red cube', 'walk to the purple cylinder'")
    tRun = st.button("▶ Run pipeline", type="primary", disabled=not tReady)

    if tRun and tReady:
        with st.status("Running the full pipeline… (~1–2 min: model load + rollout)",
                       expanded=True) as tStatus:
            try:
                tStart = time.time()
                tOut, tVideo, tResultJson = runFn(int(state.scene_seed), tInstruction)
                tSeconds = time.time() - tStart
                tStatus.update(label=f"Done in {tSeconds:.0f}s", state="complete", expanded=False)
                renderResult(tInstruction, tOut, tVideo, tResultJson)
            except subprocess.TimeoutExpired:
                tStatus.update(label="Timed out", state="error")
                st.error("The run timed out. Try again, or check the pod (sidebar).")
            except Exception as tError:  # noqa: BLE001
                tStatus.update(label="Error", state="error")
                st.exception(tError)
    elif not tReady:
        st.info("⬆️ Load a scene in Step 1 first — then you'll see its objects and can instruct.")


def main() -> None:
    tMode = detectMode()
    tPod = PodConfig.fromEnv()
    tRemote = RemoteBackend(tPod)
    st.set_page_config(page_title="G1Nav — Interactive", layout="wide", page_icon="🦿")
    st.title("🦿🧠 G1Nav — Interactive (run it yourself)")
    st.caption("1) Load a scene to see its objects.  2) Tell the robot which one to go to.  "
               "The target is chosen by keyword match against the scene labels (the GR00T N1.6 "
               "Eagle embedding is computed for logging only) → the trained PPO policy walks "
               "there in MuJoCo (physics only).")

    # session-state keys are the original ones (scene_seed, objects, ego_path)
    tState = st.session_state
    tState.setdefault("scene_seed", None)
    tState.setdefault("objects", None)
    tState.setdefault("ego_path", None)

    renderSidebar(tMode, tPod, tRemote)
    renderSceneStep(tState, localPreview if tMode == "local" else tRemote.preview)
    st.divider()
    renderInstructionStep(tState, localRun if tMode == "local" else tRemote.run)


if __name__ == "__main__":
    main()
