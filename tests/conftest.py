import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("", os.path.join("code", "envs"), os.path.join("code", "navigation")):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)
