import json
import os

import pytest

from grounding import keyword_target, label_score
from analysis import evidence as ev


def _original_cortex_rule(instruction, labels):
    """The inline rule cortex_eagle.py used when the demos were recorded."""
    instr = instruction.lower()

    def score(lbl):
        return sum(1 for w in lbl.lower().split() if w in instr)
    target = max(labels, key=score) if labels else instr
    if labels and score(target) <= 0:
        target = labels[0]
    return target


def _original_fallback_rule(instruction, labels):
    """The inline fallback g1nav_run.py used before grounding.py existed."""
    instr = instruction.lower()
    return max(labels, key=lambda l: sum(w in instr for w in l.lower().split()),
               default=labels[0] if labels else None)


CASES = [
    ("go to the red cube", ["red cube", "purple cube", "yellow cone"]),
    ("Go to the PURPLE cylinder", ["purple cylinder", "purple ball", "orange cone"]),
    ("walk to the ball", ["red cube", "green ball", "blue ball"]),
    ("dance", ["red cube", "green ball"]),
    ("approach the orange cone", ["orange cylinder", "yellow ball", "orange cone"]),
]


@pytest.mark.parametrize("instruction,labels", CASES)
def test_same_as_recorded_rules(instruction, labels):
    assert keyword_target(instruction, labels) == _original_cortex_rule(instruction, labels)
    assert keyword_target(instruction, labels) == _original_fallback_rule(instruction, labels)


def test_empty_labels():
    assert keyword_target("go to the red cube", []) is None


def test_scores():
    assert label_score("red cube", "go to the red cube") == 2
    assert label_score("purple cube", "go to the red cube") == 1


@pytest.mark.parametrize("seed,instruction", ev.DEMO_EPISODES)
def test_recorded_cognition_is_the_keyword_rule(seed, instruction):
    cog = ev.cognition(seed, instruction)
    assert keyword_target(instruction, cog["labels"]) == cog["vlm_target"]
    assert cog["eagle_patches"] == 286 and cog["eagle_embed_dim"] == 2048
