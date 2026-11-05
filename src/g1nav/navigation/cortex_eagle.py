"""GR00T N1.6 Eagle embedding of the start ego frame (logged only) + keyword target selection; writes the cognition JSON.

Runs in a separate environment with torch + transformers 4.51.3 (configs/envs/requirements-eagle.txt) and a CUDA
GPU. The Eagle vision-language backbone that is part of the Isaac GR00T N1.6 checkpoint perceives the ego frame
the way GR00T N1.6 uses it (as a perception embedding, not for text generation; GR00T trims the Eagle LM to ~16
layers). The target is chosen by keyword overlap between the instruction and the scene labels (grounding.py); the
embedding's shape and norm are recorded but do not affect the chosen target. The JSON keeps the historical key
`vlm_target` for the keyword-selected label; stages 3-5 are fixed descriptive strings, not measurements.

Input: ego image, instruction, comma-separated scene labels; the backbone directory from $G1NAV_EAGLE.
Output: JSON {stages{1..5}, vlm_target, eagle_patches, eagle_embed_dim, eagle_norm, labels} and a CORTEX_DONE line.

    python -m g1nav.navigation.cortex_eagle --image ego.png --instruction "go to the orange cylinder" \
        --labels "orange cylinder,yellow ball,purple cube" --result out.json
"""
from __future__ import annotations

import argparse
import json
import os

from g1nav.navigation.grounding import keywordTarget
from g1nav.paths import WORKSPACE_CKPT_DIR

EAGLE_IMAGE_SIZE = (448, 448)   # do_resize=false in the processor, patch size 14
PROMPT = "Describe the scene."


def eagleDir() -> str:
    """Backbone directory: $G1NAV_EAGLE, else <pod checkpoint folder>/eagle-n16 (as on the GPU machine)."""
    return os.environ.get("G1NAV_EAGLE", str(WORKSPACE_CKPT_DIR / "eagle-n16"))


def loadEagle():
    """(model, processor) of the Eagle backbone in bfloat16 on CUDA, SDPA attention."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor

    tDir = eagleDir()
    tConfig = AutoConfig.from_pretrained(tDir, trust_remote_code=True)
    for tSubConfig in [tConfig] + [getattr(tConfig, tName) for tName in ("vision_config", "text_config")
                                   if hasattr(tConfig, tName)]:
        try:
            tSubConfig._attn_implementation = "sdpa"
        except Exception:
            pass
    tModel = AutoModel.from_pretrained(tDir, config=tConfig, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).cuda().eval()
    tProcessor = AutoProcessor.from_pretrained(tDir, trust_remote_code=True)
    return tModel, tProcessor


def eagleVisionEmbed(model, processor, image):
    """Early (vision-fused) hidden state of the Eagle backbone for the ego frame: ((seq, dim), norm).

    The upper LM layers are trimmed in GR00T N1.6, so an early layer is taken; it is evidence that the N1.6
    backbone perceived the scene, nothing downstream uses it.
    """
    import torch

    tMessages = [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": PROMPT}]}]
    tText = processor.apply_chat_template(tMessages, add_generation_prompt=True, tokenize=False)
    tInputs = processor(images=[image], text=[tText], return_tensors="pt").to("cuda")
    for tKey in tInputs:
        if torch.is_tensor(tInputs[tKey]) and tInputs[tKey].dtype == torch.float32:
            tInputs[tKey] = tInputs[tKey].to(torch.bfloat16)
    with torch.no_grad():
        tOut = model.generate(**tInputs, max_new_tokens=1, do_sample=False,
                              output_hidden_states=True, return_dict_in_generate=True)
    tHidden = tOut.hidden_states[0][1]       # first step, early vision-fused layer
    return tuple(tHidden.shape[-2:]), float(tHidden.float().norm())


def main(argv: list[str] | None = None) -> int:
    tParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    tParser.add_argument("--image", required=True)
    tParser.add_argument("--instruction", required=True)
    tParser.add_argument("--labels", default="", help="comma-separated scene object labels")
    tParser.add_argument("--result", default=str(WORKSPACE_CKPT_DIR / "cortex_result.json"))
    tArgs = tParser.parse_args(argv)

    from PIL import Image

    tImage = Image.open(tArgs.image).convert("RGB").resize(EAGLE_IMAGE_SIZE)
    tModel, tProcessor = loadEagle()
    (tPatches, tHiddenDim), tNorm = eagleVisionEmbed(tModel, tProcessor, tImage)

    tLabels = [tLabel.strip() for tLabel in tArgs.labels.split(",") if tLabel.strip()]
    tInstruction = tArgs.instruction.lower()
    # IDENTIFY: pick the scene object by colour + shape keyword overlap (grounding.py)
    tTarget = keywordTarget(tInstruction, tLabels) if tLabels else tInstruction

    tStages = {
        1: (f"GR00T N1.6 Eagle vision encoder (SigLip2) perceived the ego frame: "
            f"{tPatches} patch tokens x {tHiddenDim}-d (‖emb‖={tNorm:.0f}). Candidate objects: "
            + (", ".join(tLabels) if tLabels else "n/a")),
        2: f"target = {tTarget}   (keyword match: instruction words vs scene object labels)",
        3: "localize: target position read from the generated scene by navigate.py (ground truth)",
        4: "plan: head toward the grounded target",
        5: "motor intent: forward / turn -> velocity command (Navigator)",
    }
    with open(tArgs.result, "w") as tFile:
        json.dump({"stages": tStages, "vlm_target": tTarget, "eagle_patches": tPatches,
                   "eagle_embed_dim": tHiddenDim, "eagle_norm": tNorm, "labels": tLabels}, tFile, indent=2)
    print("CORTEX_DONE", json.dumps({"vlm_target": tTarget, "eagle": f"{tPatches}x{tHiddenDim}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
