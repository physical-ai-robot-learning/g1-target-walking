"""Build the figures of report_en.tex from committed files (media/, results/) into report/figures/.

Inputs: media/seed{0,1,2}_*.{mp4,ego.png}, results/checkpoints/walk_t4_metrics.jsonl,
results/walk_sweep_cpu/walk_sweep.json. Outputs: figures/fig_episode.jpg (third-person | ego frames of the seed-0
episode), figures/fig_ego_starts.jpg (start ego frames of the three demo scenes), figures/fig_emergence.pdf (evaluation
reward, forward velocity and minimum torso height per checkpoint). Needs numpy, matplotlib, imageio(-ffmpeg), Pillow.

    python report/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "report" / "figures"
EPISODES = ["seed0_go-to-the-orange-cylinder", "seed1_go-to-the-red-cube", "seed2_go-to-the-purple-cylinder"]
# frames of the seed-0 video (50 fps): standing pre-roll end, turning, approaching, standing at the target
EPISODE_FRAMES = (99, 250, 400, 629)
JPEG_QUALITY = 88


def episodeFigure() -> None:
    tVideo = iio.imread(REPO / "media" / f"{EPISODES[0]}.mp4", plugin="pyav")
    tFrames = [tVideo[tIndex] for tIndex in EPISODE_FRAMES]
    tGrid = np.concatenate([np.concatenate(tFrames[:2], axis=1), np.concatenate(tFrames[2:], axis=1)], axis=0)
    Image.fromarray(tGrid).save(FIGURES / "fig_episode.jpg", quality=JPEG_QUALITY)


def egoStartsFigure() -> None:
    tImages = [np.asarray(Image.open(REPO / "media" / f"{tName}.ego.png").convert("RGB")) for tName in EPISODES]
    Image.fromarray(np.concatenate(tImages, axis=1)).save(FIGURES / "fig_ego_starts.jpg", quality=JPEG_QUALITY)


def emergenceFigure() -> None:
    with open(REPO / "results" / "checkpoints" / "walk_t4_metrics.jsonl") as tFile:
        tMetrics = [json.loads(tLine) for tLine in tFile if tLine.strip()]
    with open(REPO / "results" / "walk_sweep_cpu" / "walk_sweep.json") as tFile:
        tSweep = json.load(tFile)
    tSteps = np.array([tRow["step"] for tRow in tSweep]) / 1e6
    tFig, tAxes = plt.subplots(3, 1, figsize=(3.4, 3.5), sharex=True)
    tAxes[0].plot(np.array([tRow["step"] for tRow in tMetrics]) / 1e6, [tRow["reward"] for tRow in tMetrics], "o-", ms=3)
    tAxes[0].axhline(0, color="0.6", lw=0.6)
    tAxes[0].set_ylabel("eval reward")
    tAxes[1].plot(tSteps, [tRow["fwd_vel"] for tRow in tSweep], "o-", ms=3, color="C1")
    tAxes[1].axhline(0.6, color="0.6", lw=0.6, ls="--")
    tAxes[1].set_ylabel("fwd vel (m/s)")
    tAxes[2].plot(tSteps, [tRow["min_z"] for tRow in tSweep], "o-", ms=3, color="C2")
    tAxes[2].axhline(0.4, color="0.6", lw=0.6, ls="--")
    tAxes[2].set_ylabel("min torso z (m)")
    tAxes[2].set_xlabel("environment steps (M)")
    for tAx in tAxes:
        tAx.grid(alpha=0.3)
        tAx.tick_params(labelsize=7)
        tAx.yaxis.label.set_size(8)
    tAxes[2].xaxis.label.set_size(8)
    tFig.tight_layout(h_pad=0.4)
    tFig.savefig(FIGURES / "fig_emergence.pdf")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    episodeFigure()
    egoStartsFigure()
    emergenceFigure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
