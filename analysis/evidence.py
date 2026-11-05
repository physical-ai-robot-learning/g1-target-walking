"""Readers for the result files committed in this repository.

Everything here runs with Python + numpy (+ Pillow for GIFs): no JAX, brax,
MuJoCo or GPU. Checkpoints are opened with a restricted unpickler that only
materialises numpy/builtin objects and replaces every other class (brax, jax)
by an inert stub, so the step counter and the array shapes can be read without
importing or executing third-party code.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import struct
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR = os.path.join(ROOT, "results", "checkpoints")
VIDEO_DIR = os.path.join(ROOT, "media")
RERUN_JSON = os.path.join(ROOT, "results", "demo_rerun_cpu", "episodes.json")
SWEEP_JSON = os.path.join(ROOT, "results", "walk_sweep_cpu", "walk_sweep.json")

# (seed, instruction) of the three recorded demo episodes
DEMO_EPISODES = [
    (0, "go to the orange cylinder"),
    (1, "go to the red cube"),
    (2, "go to the purple cylinder"),
]
REACH_RADIUS_M = 0.55      # src/g1nav/navigation/action_module.py
CTRL_HZ = 50               # CTRL_DT = 0.02 s (src/g1nav/navigation/navigate.py)
HOLD_FRAMES = 100          # 2 s standing before and after (navigate.py HOLD)


def demo_prefix(seed: int, instruction: str) -> str:
    return f"seed{seed}_{instruction.replace(' ', '-')}"


# ---------------------------------------------------------------- checkpoints
_SAFE_MODULES = ("numpy", "builtins", "collections", "_codecs", "copyreg")


class _Stub:
    """Inert stand-in for a non-numpy class found in a pickle."""

    def __setstate__(self, state):
        self.__dict__["_state"] = state


def _stub_type(module: str, name: str):
    def _new(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._args = args
        return obj
    return type(name, (_Stub,), {"__new__": _new, "__module__": module})


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in _SAFE_MODULES:
            return super().find_class(module, name)
        return _stub_type(module, name)


def load_checkpoint(path: str) -> Any:
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


def array_shape(obj: Any, depth: int = 0) -> Optional[tuple]:
    """Shape of a pickled numpy/jax array without rebuilding it.

    A pickled ndarray carries its state as (version, shape, dtype, is_fortran,
    raw bytes); jax arrays wrap that inside their own reconstruct call.
    """
    if isinstance(obj, np.ndarray):
        return tuple(obj.shape)
    if depth > 8:
        return None
    if isinstance(obj, _Stub):
        return array_shape(getattr(obj, "_args", ()), depth + 1)
    if isinstance(obj, tuple) and len(obj) == 5 and isinstance(obj[2], np.dtype) \
            and isinstance(obj[1], tuple):
        return tuple(int(x) for x in obj[1])
    if isinstance(obj, (tuple, list)):
        for x in obj:
            r = array_shape(x, depth + 1)
            if r is not None:
                return r
    return None


def checkpoint_summary(path: str) -> Dict[str, Any]:
    """Step counter and network layer shapes of a train_walk.py checkpoint."""
    obj = load_checkpoint(path)
    params = obj["params"]
    policy = params[1]["params"]
    value = params[2]["params"]
    shapes = lambda p: [array_shape(p[f"hidden_{i}"]["kernel"]) for i in range(len(p))]
    norm_state = params[0].__dict__.get("_state", {})
    return {
        "step": int(obj["step"]),
        "policy_layers": shapes(policy),
        "value_layers": shapes(value),
        "normalizer_keys": sorted(norm_state.keys()) if isinstance(norm_state, dict) else [],
    }


def checkpoint_steps() -> List[int]:
    files = sorted(glob.glob(os.path.join(CKPT_DIR, "walk_t4_step*.pkl")))
    return [int(os.path.basename(f)[len("walk_t4_step"):-4]) for f in files]


def metrics() -> List[Dict[str, float]]:
    with open(os.path.join(CKPT_DIR, "walk_t4_metrics.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def rollout() -> Dict[str, float]:
    with open(os.path.join(CKPT_DIR, "walk_t4_rollout.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------- videos
def mp4_frame_count(path: str) -> int:
    """Number of samples in the first video track (reads the 'stsz' box)."""
    with open(path, "rb") as f:
        data = f.read()

    def boxes(buf, start, end):
        i = start
        while i + 8 <= end:
            size, kind = struct.unpack(">I4s", buf[i:i + 8])
            hdr = 8
            if size == 1:
                size = struct.unpack(">Q", buf[i + 8:i + 16])[0]
                hdr = 16
            elif size == 0:
                size = end - i
            yield kind.decode("latin1"), i + hdr, i + size
            i += size

    def find(buf, start, end, path_):
        for kind, s, e in boxes(buf, start, end):
            if kind == path_[0]:
                if len(path_) == 1:
                    return s, e
                r = find(buf, s, e, path_[1:])
                if r:
                    return r
        return None

    for kind, s, e in boxes(data, 0, len(data)):
        if kind != "moov":
            continue
        for k2, s2, e2 in boxes(data, s, e):
            if k2 != "trak":
                continue
            hdlr = find(data, s2, e2, ["mdia", "hdlr"])
            if not hdlr or data[hdlr[0] + 8:hdlr[0] + 12] != b"vide":
                continue
            stsz = find(data, s2, e2, ["mdia", "minf", "stbl", "stsz"])
            if stsz:
                return struct.unpack(">I", data[stsz[0] + 8:stsz[0] + 12])[0]
    raise ValueError(f"no video track in {path}")


def gif_frame_count(path: str) -> int:
    from PIL import Image
    with Image.open(path) as im:
        return im.n_frames


def cognition(seed: int, instruction: str) -> Dict[str, Any]:
    with open(os.path.join(VIDEO_DIR, demo_prefix(seed, instruction) + ".cognition.json")) as f:
        return json.load(f)


def rerun_episodes() -> List[Dict[str, Any]]:
    with open(RERUN_JSON) as f:
        return json.load(f)["episodes"]
