"""Check every number stated in README.md, report/archive/report.md (the Markdown engineering report) and media/README.md.

Each claim recomputes a value from files committed in this repository and checks
that the documents state exactly that value (as a literal text snippet). Needs
only numpy and Pillow; no JAX, brax, MuJoCo or GPU.

    python -m analysis.verify_claims                 # exit 1 on any failure (needs `pip install -e .`)
    python -m analysis.verify_claims --md out.md     # also write a per-claim table
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from typing import Callable, List, Sequence, Tuple

from analysis import evidence as ev

ROOT = ev.ROOT
README = "README.md"
REPORT = "report/archive/report.md"
VIDEOS = "media/README.md"


def _read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _const(rel: str, name: str):
    """Value of a module-level constant in a Python file (parsed, not imported)."""
    tree = ast.parse(_read(rel))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
                if isinstance(t, ast.Tuple):
                    for i, el in enumerate(t.elts):
                        if isinstance(el, ast.Name) and el.id == name:
                            return ast.literal_eval(node.value.elts[i])
    raise KeyError(f"{name} not found in {rel}")


def _argparse_default(rel: str, flag: str):
    m = re.search(rf'add_argument\("{re.escape(flag)}"[^)]*?default=([^,)\n]+)', _read(rel), re.S)
    if not m:
        raise KeyError(f"{flag} not found in {rel}")
    return ast.literal_eval(m.group(1).strip())


# ------------------------------------------------------------------ evidence
CK = ev.checkpoint_summary(os.path.join(ev.CKPT_DIR, "walk_t4_latest.pkl"))
METRICS = ev.metrics()
ROLL = ev.rollout()
with open(ev.SWEEP_JSON) as _f:
    SWEEP = {r["step"]: r for r in json.load(_f)}
RERUN = {r["seed"]: r for r in ev.rerun_episodes()}
FINAL = METRICS[-1]
FIRST_WALK = min(s for s, r in SWEEP.items() if r["verdict"] == "WALKS")
BEST = max(SWEEP.values(), key=lambda r: r["fwd_vel"])
COG = {s: ev.cognition(s, i) for s, i in ev.DEMO_EPISODES}
FRAMES = {s: ev.mp4_frame_count(os.path.join(ev.VIDEO_DIR, ev.demo_prefix(s, i) + ".mp4"))
          for s, i in ev.DEMO_EPISODES}
REACH = _const("src/g1nav/navigation/action_module.py", "REACH_RADIUS")


def m(step: int) -> str:
    return f"{step / 1e6:.1f} M"


def _sha(path: str) -> str:
    with open(os.path.join(ROOT, path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _scene_labels(seed: int) -> List[str]:
    from g1nav.envs.scene_gen import sampleScene  # pure numpy
    return [o.label for o in sampleScene(seed).mObjects]


def _keyword_target(instruction: str, labels: Sequence[str]) -> str:
    from g1nav.navigation.grounding import keywordTarget
    return keywordTarget(instruction, labels)


# ------------------------------------------------------------------ claims
# (id, files that must contain the snippet, snippet, check) -- the snippet is
# built from recomputed values, so "present in the file" == "value matches".
Claim = Tuple[str, Sequence[str], str, Callable[[], bool]]
CLAIMS: List[Claim] = []


def claim(cid: str, files: Sequence[str], snippet: str, check: Callable[[], bool] = lambda: True):
    CLAIMS.append((cid, files, snippet, check))


# training
claim("steps_final_README", [README], "154,828,800",
      lambda: CK["step"] == 154_828_800 == FINAL["step"])
claim("steps_final_M", [README, REPORT], f"{CK['step'] / 1e6:.1f} M")
claim("ckpt_latest_equals_last", [REPORT], "byte-identical to `walk_t4_step0154828800.pkl`",
      lambda: _sha("results/checkpoints/walk_t4_latest.pkl") == _sha("results/checkpoints/walk_t4_step0154828800.pkl"))
claim("n_checkpoints", [README], f"all {len(ev.checkpoint_steps())} saved checkpoints",
      lambda: len(ev.checkpoint_steps()) == len(SWEEP) == len(METRICS))
claim("ckpt_spacing", [REPORT], f"{ev.checkpoint_steps()[1]:,} steps",
      lambda: all(b - a == ev.checkpoint_steps()[1] for a, b in
                  zip(ev.checkpoint_steps(), ev.checkpoint_steps()[1:])))
claim("train_minutes", [README], f"{FINAL['elapsed_min']:.1f} min / {FINAL['reward']:.2f}")
claim("train_minutes_round", [README], f"{FINAL['elapsed_min']:.0f} min wall-clock")
claim("train_minutes_report", [REPORT], f"whole {FINAL['elapsed_min']:.0f}-minute run")
claim("final_reward", [REPORT], f"+{FINAL['reward']:.2f}")
_ss = (FINAL["step"] - METRICS[1]["step"]) / (FINAL["elapsed_min"] - METRICS[1]["elapsed_min"]) / 1e6
claim("throughput_steady", [REPORT], f"about {_ss:.1f} M steps/min")
claim("throughput_steady_window", [REPORT],
      f"{METRICS[1]['step'] / 1e6:.1f} M → {FINAL['step'] / 1e6:.1f} M steps in "
      f"{FINAL['elapsed_min'] - METRICS[1]['elapsed_min']:.1f} min")
claim("throughput_total", [REPORT], f"about {FINAL['step'] / FINAL['elapsed_min'] / 1e6:.1f} M steps/min")
claim("num_envs_default", [README, REPORT], "--num_envs 8192",
      lambda: _argparse_default("src/g1nav/locomotion/train_walk.py", "--num_envs") == 8192)
for row_step in (0, 10321920, 30965760, 41287680, 61931520, 144506880):
    r = SWEEP[row_step]
    rew = next(x["reward"] for x in METRICS if x["step"] == row_step)
    label = "0" if row_step == 0 else m(row_step)
    claim(f"emergence_row_{row_step}", [REPORT],
          f"| {'**' + label + '**' if row_step == FIRST_WALK else label} | "
          f"{'**' if row_step == FIRST_WALK else ''}{rew:+.2f}".replace("-", "−") +
          f"{'**' if row_step == FIRST_WALK else ''} | "
          f"{'**' if row_step == FIRST_WALK else ''}{r['fwd_vel']:+.2f}".replace("-", "−") +
          f"{'**' if row_step == FIRST_WALK else ''} | "
          f"{'**' if row_step == FIRST_WALK else ''}{r['net_dist']:.2f} m"
          f"{'**' if row_step == FIRST_WALK else ''} | "
          f"{'**' if row_step == FIRST_WALK else ''}{r['min_z']:.2f}".replace("-", "−") +
          f"{'**' if row_step == FIRST_WALK else ''} |")
_fr = SWEEP[FINAL["step"]]
claim("emergence_row_final", [REPORT],
      f"| {m(FINAL['step'])} (final) | +{FINAL['reward']:.2f} | +{_fr['fwd_vel']:.2f} | "
      f"{_fr['net_dist']:.2f} m | {_fr['min_z']:.2f} | walks |")

# walking evaluation
claim("first_walk", [README], f"first checkpoint that walks (raw env) | {m(FIRST_WALK)} steps "
      f"({SWEEP[FIRST_WALK]['fwd_vel']:.2f} m/s, min torso {SWEEP[FIRST_WALK]['min_z']:.2f} m)",
      lambda: all(SWEEP[s]["verdict"] == "WALKS" for s in SWEEP if s >= FIRST_WALK))
claim("first_walk_text", [README], f"is at {m(FIRST_WALK)} steps")
claim("first_walk_report", [REPORT], f"Walking first appears at the {m(FIRST_WALK)} checkpoint")
claim("final_fwd_sweep", [README],
      f"0.52 m/s for a 0.6 m/s command, {_fr['net_dist']:.2f} m, min torso {_fr['min_z']:.2f} m",
      lambda: f"{_fr['fwd_vel']:.2f}" == "0.52")
claim("final_fwd_pct", [README], f"({_fr['fwd_vel'] / 0.6 * 100:.0f} %)",
      lambda: f"{ROLL['fwd_vel'] / 0.6 * 100:.0f}" == f"{_fr['fwd_vel'] / 0.6 * 100:.0f}")
claim("final_min_torso_5s", [README], f"minimum torso height of {_fr['min_z']:.2f} m over 5 s",
      lambda: _const("src/g1nav/locomotion/walk_eval.py", "N_STEPS") == 250 and _const("src/g1nav/locomotion/walk_eval.py", "VX") == 0.6)
claim("rollout_4s", [README], f"{ROLL['fwd_vel']:.3f} m/s, {ROLL['x_travel']:.2f} m, min torso {ROLL['min_z']:.3f} m",
      lambda: ROLL["step"] == CK["step"])
claim("rollout_4s_report", [REPORT],
      f"{ROLL['fwd_vel']:.3f} m/s mean local\nforward velocity, {ROLL['x_travel']:.2f} m net displacement and "
      f"{ROLL['min_z']:.3f} m minimum torso height")
claim("best_sweep", [README], f"{BEST['fwd_vel']:.2f} m/s at {m(BEST['step'])} steps")
claim("best_sweep_report", [REPORT], f"| {m(BEST['step'])} | +")

# network / interfaces (read from the checkpoint)
pol = CK["policy_layers"]
claim("obs_dim", [README, REPORT], f"{pol[0][0]}-D", lambda: pol[0][0] == 103)
claim("act_dim", [README, REPORT], "29", lambda: pol[-1][1] == 2 * 29)
claim("policy_mlp", [REPORT], " → ".join(str(x) for x in [pol[0][0]] + [p[1] for p in pol]))
val = CK["value_layers"]
claim("value_mlp", [REPORT], " → ".join(str(x) for x in [val[0][0]] + [p[1] for p in val]))
claim("privileged_dim", [README], f"{val[0][0]}-D privileged state")
claim("normaliser", [REPORT], "running\nobservation normaliser is stored as `params[0]`",
      lambda: "mean" in CK["normalizer_keys"] and "std" in CK["normalizer_keys"])
claim("ctrl_50hz", [README], "50 Hz",
      lambda: abs(1 / 0.02 - 50) < 1e-9 and "CTRL_DT = 0.02" in _read("src/g1nav/navigation/navigate.py"))
claim("physics_500hz", [README], "500 Hz (10 substeps per control step)",
      lambda: "SIM_DT = 0.002" in _read("src/g1nav/navigation/navigate.py"))
claim("action_scale", [README], "`motor_target = default_pose + 0.5 · action`",
      lambda: "ACTION_SCALE = 0.5" in _read("src/g1nav/navigation/navigate.py"))
claim("reach_radius", [README, REPORT, VIDEOS], f"{REACH:.2f} m")

# closed loop
for seed, instr in ev.DEMO_EPISODES:
    r, cog = RERUN[seed], COG[seed]
    tgt = cog["vlm_target"]
    claim(f"labels_seed{seed}", [VIDEOS], ", ".join(cog["labels"]),
          lambda s=seed, c=cog: _scene_labels(s) == c["labels"])
    claim(f"keyword_seed{seed}", [README, REPORT, VIDEOS], f"\"{instr}\" | {tgt}",
          lambda c=cog, i=instr: _keyword_target(i, c["labels"]) == c["vlm_target"]
          and c["vlm_target"] in i)
    dist, z = f"{r['final_dist_m']:.2f} m", f"{r['torso_z_m']:.2f} m"
    claim(f"episode_seed{seed}_readme", [README],
          f"| \"{instr}\" | {tgt} | yes | {dist} | {z} | {FRAMES[seed]} |",
          lambda r=r: r["reached"] and not r["fell"] and r["final_dist_m"] <= REACH)
    claim(f"episode_seed{seed}_report", [REPORT],
          f"| {seed} | \"{instr}\" | {tgt} | yes | {dist} | {z} | {FRAMES[seed]} |")
    claim(f"episode_seed{seed}_videos", [VIDEOS], f"| {tgt} | yes | {dist} | {z} | {FRAMES[seed]} |")
    claim(f"frames_seed{seed}", [README], f"{FRAMES[seed]}",
          lambda s=seed, r=r: abs(FRAMES[s] - r["frames"]) <= 1
          and FRAMES[s] - 2 * ev.HOLD_FRAMES <= 12 * ev.CTRL_HZ)
    claim(f"gif_seed{seed}", [README], f"media/{ev.demo_prefix(seed, instr)}.gif",
          lambda s=seed, i=instr: ev.gif_frame_count(
              os.path.join(ev.VIDEO_DIR, ev.demo_prefix(s, i) + ".gif")) > 1)
dists = " / ".join(f"{RERUN[s]['final_dist_m']:.2f}" for s, _ in ev.DEMO_EPISODES)
claim("dists_summary", [README], f"final distances {dists} m")
zs = sorted(RERUN[s]["torso_z_m"] for s, _ in ev.DEMO_EPISODES)
claim("torso_summary", [README], f"torso height {zs[0]:.2f}–{zs[-1]:.2f} m")
claim("n_reached", [README], f"In all {sum(r['reached'] for r in RERUN.values())} demo scenes",
      lambda: len(RERUN) == 3 and all(r["reached"] for r in RERUN.values()))
claim("seed1_frame_diff", [REPORT], f"({RERUN[1]['frames']} vs {FRAMES[1]})")
claim("seed2_margin", [README], f"Seed 2 stops at {RERUN[2]['final_dist_m']:.2f} m against a {REACH:.2f} m radius")
claim("eagle_dims", [README, REPORT, VIDEOS],
      f"{COG[0]['eagle_patches']} tokens × {COG[0]['eagle_embed_dim']}-d",
      lambda: all(c["eagle_patches"] == 286 and c["eagle_embed_dim"] == 2048 for c in COG.values()))
claim("stage3_note", [VIDEOS], "localize: object bearing + distance via color/shape CV + ego-depth",
      lambda: all("CV + ego-depth" in c["stages"]["3"] for c in COG.values()))
claim("use_vlm_unused", [README, REPORT], "--use_vlm",
      lambda: ".use_vlm" not in _read("src/g1nav/navigation/navigate.py"))
claim("gt_target_position", [README, REPORT], "ground truth",
      lambda: "tGoalWorld = (tTarget.mX, tTarget.mY)" in _read("src/g1nav/navigation/navigate.py"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", help="write a per-claim markdown table here")
    args = ap.parse_args()
    docs = {f: _read(f) for f in (README, REPORT, VIDEOS)}
    rows, failed = [], 0
    for cid, files, snippet, check in CLAIMS:
        problems = [f"not in {f}" for f in files if snippet not in docs[f]]
        try:
            if not check():
                problems.append("recomputation check failed")
        except Exception as e:  # noqa: BLE001
            problems.append(f"check raised {type(e).__name__}: {e}")
        ok = not problems
        failed += not ok
        rows.append((cid, snippet.replace("\n", " "), ", ".join(files), "PASS" if ok else "FAIL: " + "; ".join(problems)))
        if not ok:
            print(f"FAIL {cid}: {snippet!r}: {'; '.join(problems)}")
    print(f"{len(CLAIMS)} claims checked, {failed} failed.")
    if args.md:
        with open(args.md, "w") as f:
            f.write("| id | stated text (recomputed) | documents | status |\n|---|---|---|---|\n")
            for r in rows:
                f.write("| " + " | ".join(x.replace("|", "\\|") for x in r) + " |\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
