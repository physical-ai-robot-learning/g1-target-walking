#!/usr/bin/env bash
# Parameterized single run of the full G1Nav pipeline, for the interactive dashboard (remote mode).
# Args:  $1 = seed (int)   $2 = instruction (free text)
#   1) build the integrated scene for the seed, render the ego frame
#   2) target label picked by keyword match against scene labels
#      (GR00T N1.6 Eagle embedding computed for logging only)
#   3) g1nav.navigation.navigate walks the trained PPO policy to that target (physics only),
#      recording ego||third-person side-by-side video with 2 s pre/post-roll.
# Runs on the GPU machine: repository at $G1NAV_WORKSPACE (default /workspace), simulation venv
# $G1NAV_WORKSPACE/venv, Eagle venv $G1NAV_WORKSPACE/venv_groot (transformers 4.51.3), checkpoint and
# Eagle backbone under $G1NAV_WORKSPACE/ckpt.
# Prints machine-readable result lines:  RESULT_JSON=...  VIDEO=...
set -uo pipefail
SEED="${1:-0}"
INSTR="${2:-go to the orange cylinder}"
WS="${G1NAV_WORKSPACE:-/workspace}"
cd "$WS"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$WS/src${PYTHONPATH:+:$PYTHONPATH}"
EGV="$WS/venv_groot"   # transformers 4.51.3 (Eagle)
RLV="$WS/venv"         # jax/brax/mujoco_playground (policy + render)
export G1NAV_EAGLE="$WS/ckpt/eagle-n16"
CKPT="$WS/ckpt/walk_t4_latest.pkl"
EGO="$WS/ckpt/ui_ego_seed${SEED}.png"
RES="$WS/ckpt/ui_cortex_seed${SEED}.json"
VID="$WS/ckpt/ui_nav_seed${SEED}.mp4"
LABELS_TXT="$WS/ckpt/ui_labels.txt"

echo "STAGE=scene"
"$RLV/bin/python" - "$SEED" "$EGO" "$LABELS_TXT" <<'PY' 2>/dev/null
import sys
from g1nav.navigation.pipeline import renderScene
seed, out, labelsTxt = int(sys.argv[1]), sys.argv[2], sys.argv[3]
labels = renderScene(seed, out)
open(labelsTxt, "w").write(",".join(labels))
PY
LABELS=$(cat "$LABELS_TXT" 2>/dev/null)
echo "OBJECTS=$LABELS"

echo "STAGE=cognition"
"$EGV/bin/python" -m g1nav.navigation.cortex_eagle \
  --image "$EGO" --instruction "$INSTR" --labels "$LABELS" --result "$RES" \
  2>&1 | grep -E "CORTEX_DONE" | head
TARGET=$("$RLV/bin/python" -c "import json,sys;print(json.load(open(sys.argv[1]))['vlm_target'])" "$RES" 2>/dev/null)
echo "TARGET=$TARGET"

echo "STAGE=navigate"
"$RLV/bin/python" -m g1nav.navigation.navigate \
  --ckpt "$CKPT" --seed "$SEED" --target "$TARGET" --seconds 14 --out "$VID" \
  2>&1 | grep -E "^NAV|FELL" | head

echo "RESULT_JSON=$RES"
echo "VIDEO=$VID"
echo "DONE"
