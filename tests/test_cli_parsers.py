"""Characterization of every command-line interface that can be parsed without the simulation stack.

Each entry point is run as `__main__` with `ArgumentParser.parse_args` patched to stop right after the parser is
built; the recorded options (flags, dest, default, type, nargs, required, help) are compared with
tests/golden/cli_parsers.json, recorded before the 2026-11 layout refactor. Defaults that are repository paths are
stored relative to the repository root; the paths that moved in the refactor are mapped by PATH_MOVES.
cortex_eagle imports torch / transformers, which are replaced by empty stubs here (only the parser is read).

    G1NAV_REGEN_GOLDEN=1 python -m pytest -q tests/test_cli_parsers.py      # re-record (intended change only)
"""
import argparse
import json
import os
import runpy
import sys
import types

import pytest

from conftest import ROOT

GOLDEN = os.path.join(ROOT, "tests", "golden", "cli_parsers.json")
REGEN = os.environ.get("G1NAV_REGEN_GOLDEN") == "1"

# name (as recorded in the golden file) -> module, run as `python -m <module>`
ENTRY_POINTS = {
    "scene_gen": "g1nav.envs.scene_gen",
    "navigate": "g1nav.navigation.navigate",
    "g1nav_run": "g1nav.navigation.pipeline",
    "cortex_eagle": "g1nav.navigation.cortex_eagle",
    "train_walk": "g1nav.locomotion.train_walk",
    "rollout_walk": "g1nav.locomotion.rollout_walk",
    "rerun_demo_episodes": "g1nav.navigation.rerun_demo_episodes",
}
# repository paths that moved in the 2026-11 refactor (old prefix -> new prefix)
PATH_MOVES = {"<repo>/checkpoint/eagle-n16": "<repo>/assets/eagle-n16",
              "<repo>/checkpoint/": "<repo>/results/checkpoints/"}


class _Captured(Exception):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser


def _stub_heavy(monkeypatch):
    for name in ("torch", "transformers"):
        mod = types.ModuleType(name)
        for attr in ("AutoModel", "AutoProcessor", "AutoConfig"):
            setattr(mod, attr, object)
        monkeypatch.setitem(sys.modules, name, mod)


def _norm_default(value):
    if isinstance(value, str) and value.startswith(ROOT + os.sep):
        rel = "<repo>/" + os.path.relpath(value, ROOT)
        for old, new in PATH_MOVES.items():
            if rel.startswith(new):
                rel = old + rel[len(new):]
        return rel
    return value


def _spec(parser):
    rows = []
    for a in parser._actions:
        if isinstance(a, argparse._HelpAction):
            continue
        rows.append({"flags": list(a.option_strings), "dest": a.dest, "default": _norm_default(a.default),
                     "type": getattr(a.type, "__name__", None), "nargs": a.nargs, "required": a.required,
                     "help": a.help, "action": type(a).__name__, "metavar": a.metavar})
    return rows


def _capture(name, monkeypatch):
    _stub_heavy(monkeypatch)

    def fake_parse(self, *args, **kwargs):
        raise _Captured(self)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse)
    monkeypatch.setattr(sys, "argv", [name])
    monkeypatch.delitem(sys.modules, ENTRY_POINTS[name], raising=False)
    with pytest.raises(_Captured) as exc:
        runpy.run_module(ENTRY_POINTS[name], run_name="__main__", alter_sys=True)
    return json.loads(json.dumps(_spec(exc.value.parser)))


@pytest.fixture(scope="module")
def golden():
    if REGEN or not os.path.exists(GOLDEN):
        return None
    with open(GOLDEN) as f:
        return json.load(f)


@pytest.mark.parametrize("name", sorted(ENTRY_POINTS))
def test_parser_unchanged(name, golden, monkeypatch):
    spec = _capture(name, monkeypatch)
    if golden is None:
        data = {}
        if os.path.exists(GOLDEN):
            with open(GOLDEN) as f:
                data = json.load(f)
        data[name] = spec
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        return
    assert spec == golden[name]
