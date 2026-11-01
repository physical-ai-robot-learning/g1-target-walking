"""Repository hygiene: tracked text files are English-only (no Hangul)."""
import os
import re
import subprocess

import pytest

from conftest import ROOT

HANGUL = re.compile("[\uac00-\ud7a3]")  # Hangul syllables


def _tracked():
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [f for f in out.stdout.decode().split("\0") if f]


def test_no_hangul_in_tracked_text():
    hits = []
    for rel in _tracked():
        with open(os.path.join(ROOT, rel), "rb") as f:
            data = f.read()
        if b"\0" in data[:8000]:
            continue
        for i, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if HANGUL.search(line):
                hits.append(f"{rel}:{i}")
    assert not hits, hits
