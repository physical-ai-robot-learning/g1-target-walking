"""Characterization of the Streamlit dashboards (first render, no GPU, no simulator).

Each app is rendered once with streamlit's AppTest and the visible text of every element is compared with
tests/golden/apps.json, recorded before the 2026-11 layout refactor. Repository paths shown in the page are written
relative to the repository root; the folders that moved in the refactor are mapped by PATH_MOVES. Skipped when
streamlit is not installed (it is in the `dashboard` extra).

    G1NAV_REGEN_GOLDEN=1 python -m pytest -q tests/test_apps.py      # re-record (intended change only)
"""
import json
import os

import pytest

from conftest import ROOT

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

GOLDEN = os.path.join(ROOT, "tests", "golden", "apps.json")
REGEN = os.environ.get("G1NAV_REGEN_GOLDEN") == "1"

APPS = {
    "results": "code/apps/app_results.py",
    "interactive": "code/apps/app_interactive.py",
    "train": "code/locomotion/app_train.py",
}
# environment for a deterministic first render: no external artifact folders, the committed training run
ENV = {"G1NAV_DATA": "/nonexistent/g1nav_dash", "G1NAV_VIDEOS": "/nonexistent/g1nav_videos",
       "G1NAV_MODE": "remote", "G1NAV_POD_HOST": "root@example-host", "G1NAV_POD_PORT": "22",
       "WALK_OUT": os.path.join(ROOT, "checkpoint", "walk_t4")}
# repository folders that moved in the 2026-11 refactor (old -> new), applied to the rendered text
PATH_MOVES = {}


def _texts(at):
    rows = []
    for el in at.main:
        rows.extend(_walk(el))
    for el in at.sidebar:
        rows.extend(_walk(el))
    return rows


def _walk(el):
    out = []
    kind = type(el).__name__
    for attr in ("label", "value", "body", "help", "placeholder", "caption"):
        v = getattr(el, attr, None)
        if isinstance(v, (str, int, float, bool)) and v != "":
            out.append(f"{kind}.{attr}={v}")
    for child in getattr(el, "children", {}).values() if isinstance(getattr(el, "children", None), dict) else []:
        out.extend(_walk(child))
    return out


def _normalise(text):
    for old, new in PATH_MOVES.items():
        text = text.replace(os.path.join(ROOT, new), os.path.join(ROOT, old))
    return text.replace(ROOT, "<repo>")


def _render(name, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    at = AppTest.from_file(os.path.join(ROOT, APPS[name]), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    return [_normalise(t) for t in _texts(at)]


@pytest.fixture(scope="module")
def golden():
    if REGEN or not os.path.exists(GOLDEN):
        return None
    with open(GOLDEN) as f:
        return json.load(f)


@pytest.mark.parametrize("name", sorted(APPS))
def test_first_render_unchanged(name, golden, monkeypatch):
    texts = _render(name, monkeypatch)
    if golden is None:
        data = {}
        if os.path.exists(GOLDEN):
            with open(GOLDEN) as f:
                data = json.load(f)
        data[name] = texts
        with open(GOLDEN, "w") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        return
    assert texts == golden[name]
