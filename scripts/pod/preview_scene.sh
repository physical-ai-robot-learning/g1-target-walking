#!/usr/bin/env bash
# Fast scene preview for the interactive dashboard (remote mode): render the ego frame and list
# the objects for a seed. No GPU cognition, just a MuJoCo render — returns quickly so the user can
# SEE the scene before typing an instruction. Runs on the GPU machine, with this repository at
# $G1NAV_WORKSPACE (default /workspace) and the simulation venv at $G1NAV_WORKSPACE/venv.
# Arg: $1 = seed (int). Prints:  OBJECTS=...  EGO=<path>  DONE
set -uo pipefail
SEED="${1:-0}"
WS="${G1NAV_WORKSPACE:-/workspace}"
cd "$WS"
export MUJOCO_GL=egl PYTHONPATH="$WS/src${PYTHONPATH:+:$PYTHONPATH}"
RLV="$WS/venv"
EGO="$WS/ckpt/ui_ego_seed${SEED}.png"
LABELS_TXT="$WS/ckpt/ui_labels.txt"

"$RLV/bin/python" - "$SEED" "$EGO" "$LABELS_TXT" <<'PY' 2>/dev/null
import sys
from g1nav.navigation.pipeline import render_scene
seed, out, labelsTxt = int(sys.argv[1]), sys.argv[2], sys.argv[3]
labels = render_scene(seed, out)
open(labelsTxt, "w").write(",".join(labels))
PY
echo "OBJECTS=$(cat "$LABELS_TXT" 2>/dev/null)"
echo "EGO=$EGO"
echo "DONE"
