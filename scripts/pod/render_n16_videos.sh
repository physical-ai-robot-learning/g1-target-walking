#!/usr/bin/env bash
# Render the closed-loop demo videos (ego || third-person) — the script that produced media/seed*.mp4.
# For each seed:
#   1) build the integrated scene, render the ego frame at the home pose
#   2) g1nav.navigation.cortex_eagle picks the target label by keyword match (Eagle embedding logged)
#   3) g1nav.navigation.navigate walks the trained PPO policy to that target, recording
#      ego || third-person side-by-side video (physics only).
# Runs on the GPU machine: repository at $G1NAV_WORKSPACE (default /workspace), simulation venv
# $G1NAV_WORKSPACE/venv, Eagle venv $G1NAV_WORKSPACE/venv_groot. Outputs $G1NAV_WORKSPACE/ckpt/nav_n16_seed*.mp4.
# (The instruction is "go to the <first object label>", as in the recorded runs.)
set -uo pipefail
WS="${G1NAV_WORKSPACE:-/workspace}"
cd "$WS"
EGV="$WS/venv_groot"         # transformers 4.51.3  (Eagle cognition)
RLV="$WS/venv"               # jax/brax/mujoco_playground (policy + render)
export MUJOCO_GL=egl PYTHONPATH="$WS/src${PYTHONPATH:+:$PYTHONPATH}"
CKPT="$WS/ckpt/walk_t4_latest.pkl"
export G1NAV_EAGLE="$WS/ckpt/eagle-n16"
LABELS_TXT="$WS/ckpt/_labels.txt"

SEEDS=(0 1 2)
for S in "${SEEDS[@]}"; do
  echo "=================== seed $S ==================="
  EGO="$WS/ckpt/ego_seed${S}.png"
  # 1) scene objects + ego frame at home pose
  "$RLV/bin/python" - "$S" "$EGO" "$LABELS_TXT" <<'PY' 2>/dev/null
import sys
from g1nav.navigation.pipeline import renderScene
seed, out, labelsTxt = int(sys.argv[1]), sys.argv[2], sys.argv[3]
labels = renderScene(seed, out)
open(labelsTxt, "w").write(",".join(labels))
PY
  LABELS=$(cat "$LABELS_TXT")
  echo "[seed $S] objects: $LABELS"
  TGT=$(echo "$LABELS" | cut -d, -f1)
  INSTRUCTION="go to the $TGT"
  echo "[seed $S] instruction: $INSTRUCTION"
  # 2) N1.6 Eagle cognition -> keyword-selected target (embedding logged)
  "$EGV/bin/python" -m g1nav.navigation.cortex_eagle \
    --image "$EGO" --instruction "$INSTRUCTION" --labels "$LABELS" \
    --result "$WS/ckpt/cortex_seed${S}.json" 2>&1 | grep -E "CORTEX_DONE|Error|Traceback" | head
  NTGT=$("$RLV/bin/python" -c "import json,sys;print(json.load(open(sys.argv[1]))['vlm_target'])" "$WS/ckpt/cortex_seed${S}.json")
  echo "[seed $S] chosen target (keyword match) = $NTGT"
  # 3) closed-loop walk to that target -> ego||3rd video
  "$RLV/bin/python" -m g1nav.navigation.navigate \
    --ckpt "$CKPT" --seed "$S" --target "$NTGT" --seconds 12 \
    --out "$WS/ckpt/nav_n16_seed${S}.mp4" 2>&1 | grep -E "^NAV|FELL|sim_dt|Error|Traceback" | head
done
echo "##### ALL DONE #####"
ls -la "$WS"/ckpt/nav_n16_seed*.mp4 2>/dev/null
