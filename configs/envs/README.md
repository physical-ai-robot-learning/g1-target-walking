# configs/envs/ — recorded environments

Pinned requirement files that record what the results were produced with. `pyproject.toml` defines the same
dependencies as extras (`sim`, `dashboard`, `eagle`, `test`); `scripts/install.sh` installs these files and then the
package.

| file | environment |
|---|---|
| `requirements-gpu.txt` | GPU training / closed-loop stack (jax[cuda12] 0.5.3, brax 0.14.2, flax 0.10.6, mujoco / mujoco-mjx 3.4.0, playground 0.1.0) + dashboards |
| `requirements-dashboard.txt` | laptop dashboards only |
| `requirements-eagle.txt` | optional, logging-only GR00T N1.6 Eagle encoder (old `transformers` pin; separate environment, see `docs/third_party.md`) |

The CPU re-measurements in `results/` used the same simulation pins on Python 3.11 (docs/install.md, "Re-measuring on CPU").
