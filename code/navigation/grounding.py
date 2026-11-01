"""Keyword target selection used by the recorded demos.

This is the whole "grounding" step of the pipeline as it was run: the instructed
target is the scene object label whose words occur most often in the instruction
(substring test on the lower-cased instruction, ties broken by label order). No
image and no learned model is involved. `cortex_eagle.py` (with the GR00T N1.6
Eagle encoder available) and the fallback in `g1nav_run.py` both call this.
"""
from __future__ import annotations

from typing import Optional, Sequence


def label_score(label: str, instruction: str) -> int:
    """Number of words of `label` that occur in the lower-cased instruction."""
    instr = instruction.lower()
    return sum(1 for w in label.lower().split() if w in instr)


def keyword_target(instruction: str, labels: Sequence[str]) -> Optional[str]:
    """Label with the highest word overlap; first label on ties or no overlap.

    Returns None when `labels` is empty.
    """
    if not labels:
        return None
    return max(labels, key=lambda lbl: label_score(lbl, instruction))
