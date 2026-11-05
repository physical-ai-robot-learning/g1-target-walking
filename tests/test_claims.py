import subprocess
import sys

from conftest import ROOT


def test_verify_claims_passes():
    r = subprocess.run([sys.executable, "-m", "analysis.verify_claims"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert ", 0 failed." in r.stdout
