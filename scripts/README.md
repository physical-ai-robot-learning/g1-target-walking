# scripts/

**Status:** thin operational launchers only; all Python lives in `src/g1nav/` (run as `python -m g1nav....`).

| script | what | where it runs |
|---|---|---|
| `install.sh` | creates `.venv`, installs the pinned `configs/envs/requirements-*.txt` and the package (`pip install -e .`) | any machine (`gpu` / `dashboard` mode) |
| `run_dashboard.sh` | launches the results (8501) and interactive (8502) dashboards | any machine |
| `build_report.sh` | rebuilds the archived Markdown report's PDF (pandoc + tectonic) | author's Mac (STIX / Menlo fonts) |
| `pod/pod_bootstrap.sh`, `pod/setup_ada.sh` | set up a fresh GPU machine (apt, venv, pinned JAX / MuJoCo / Playground stack) | GPU machine, repository at `$G1NAV_WORKSPACE` (default `/workspace`) |
| `pod/render_n16_videos.sh` | renders the three demo episodes (produced `media/seed*.mp4`) | GPU machine |
| `pod/preview_scene.sh`, `pod/run_pipeline_n16.sh` | scene preview / one run for the interactive dashboard's remote mode | GPU machine (pushed over SSH) |

**Inputs → outputs:** see each script's header. No host names, IPs, ports or keys are stored; the remote machine is
given by `G1NAV_POD_HOST` / `G1NAV_POD_PORT` / `G1NAV_POD_KEY` (dashboard) and its folder by `G1NAV_WORKSPACE`.

**Not executable here:** the `pod/` scripts need a Linux GPU machine with the pod layout (`$G1NAV_WORKSPACE/venv`,
`venv_groot`, `ckpt/`); they are syntax-checked (`bash -n`) only.

**Provenance:** author's scripts; moved from the repository root and `code/navigation/` in 2026-11.
