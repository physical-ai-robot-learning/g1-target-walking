"""Keyword target selection used by the recorded demos.

This is the whole "grounding" step of the pipeline as it was run: the instructed target is the scene object label
whose words occur most often in the instruction (substring test on the lower-cased instruction, ties broken by label
order). No image and no learned model is involved. `cortex_eagle` (with the GR00T N1.6 Eagle encoder available) and
the fallback in `pipeline` both call this.

Input: instruction text + scene object labels. Output: the selected label (or None for no labels).

    from g1nav.navigation.grounding import keywordTarget
    keywordTarget("go to the red cube", ["red cube", "purple cube"])   # -> "red cube"
"""
from __future__ import annotations

from typing import Optional, Sequence


def labelScore(label: str, instruction: str) -> int:
    """Number of words of `label` that occur in the lower-cased instruction."""
    tInstruction = instruction.lower()
    return sum(1 for tWord in label.lower().split() if tWord in tInstruction)


def keywordTarget(instruction: str, labels: Sequence[str]) -> Optional[str]:
    """Label with the highest word overlap; first label on ties or no overlap. None when `labels` is empty."""
    if not labels:
        return None
    return max(labels, key=lambda tLabel: labelScore(tLabel, instruction))
