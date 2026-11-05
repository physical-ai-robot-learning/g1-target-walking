# g1nav.apps — Streamlit dashboards

**Status:** convenience tools; no reported number depends on them.

**Inputs → outputs:** committed `media/` and `results/checkpoints/` (or artifacts copied from the GPU machine) →
pages; the interactive page runs `g1nav.navigation.pipeline` locally or `scripts/pod/*.sh` over SSH.

**Entry points**
```bash
bash scripts/run_dashboard.sh                                        # 8501 results + 8502 interactive
streamlit run src/g1nav/apps/results_dashboard.py --server.port 8501   # GPU-free replay
streamlit run src/g1nav/apps/interactive_dashboard.py --server.port 8502
WALK_OUT=results/checkpoints/walk_t4 streamlit run src/g1nav/apps/training_dashboard.py
```
Remote mode: `G1NAV_POD_HOST`, `G1NAV_POD_PORT`, `G1NAV_POD_KEY`, `G1NAV_POD_WORKSPACE` (no host is built in).

**Layout:** `results_dashboard.py` (was `app_results.py`), `interactive_dashboard.py` (was `app_interactive.py`),
`training_dashboard.py` (was `code/locomotion/app_train.py`).

**Verified here:** first render of each page compared with the pre-refactor render (`tests/test_apps.py`).
