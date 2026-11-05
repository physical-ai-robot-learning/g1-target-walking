# report/

The technical report of the project: [`report_en.tex`](report_en.tex) → [`report_en.pdf`](report_en.pdf) (IEEEtran,
two-column, English; status of 5 November 2026). It covers the system (G1, simulator, observation / action spaces), the
method (PPO locomotion policy, keyword target selection, Navigator, deployment), the results (walking emergence,
closed-loop episodes), the failure analysis, limitations and reproducibility. Earlier report in [`archive/`](archive/).

**Status:** current (`report_en`); `archive/` kept for reference.

## Inputs → Outputs
`report_en.tex` + `figures/` → `report_en.pdf`. Build from this folder:

```bash
cd report && tectonic -X compile report_en.tex
```

`figures/` is built from committed files by [`make_figures.py`](make_figures.py) (`python report/make_figures.py`;
needs numpy, matplotlib, imageio + PyAV, Pillow): `fig_ego_starts.jpg` (start ego frames, `media/*.ego.png`),
`fig_emergence.pdf` (evaluation reward per checkpoint from `results/checkpoints/walk_t4_metrics.jsonl`; forward velocity
and minimum torso height from `results/walk_sweep_cpu/walk_sweep.json`), `fig_episode.jpg` (four frames of
`media/seed0_go-to-the-orange-cylinder.mp4`). The architecture figure is TikZ in the source.

## Checking the numbers
Every number printed in `report_en.tex` is mapped to the repository file it was taken from (README, the archived
report, `media/README.md`, the code constants of the Navigator / episode protocol / evaluation) and checked by
[`../analysis/verify_report.py`](../analysis/verify_report.py) (run from the repository root as
`python -m analysis.verify_report`; prints `N claims checked, 0 failed`; also run in CI and by `tests/test_report.py`).
An unmapped number or a stale mapping fails. The report also has to state that no success rate is reported, that the
Eagle embedding does not influence the target and that the system is not a vision-language-action model. The numbers
in the README, the archived report and `media/README.md` are recomputed from the result files by
[`../analysis/verify_claims.py`](../analysis/verify_claims.py).

## Layout
- `report_en.{tex,pdf}` — the current report; `figures/`, `make_figures.py` — its figures.
- `archive/report.{md,pdf}` — the Markdown engineering report (2026-11-01, was `docs/report.md`); the source of most of
  the current report's text and numbers, still checked by `verify_claims.py`. Its PDF is rebuilt with
  `bash scripts/build_report.sh` (pandoc + tectonic). Its repository paths are those of the 2026-11 layout.
