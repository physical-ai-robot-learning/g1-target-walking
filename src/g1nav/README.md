# src/g1nav — the G1Nav package

**Status:** finished prototype (2026-11-01); restyled and moved here from `code/` in 2026-11 without behaviour change
(see [docs/refactor_2026-11.md](../../docs/refactor_2026-11.md)).

**Inputs → outputs:** scene seed + English instruction + trained walking checkpoint → keyword-selected target →
Navigator velocity commands → PPO joint targets → MuJoCo physics → ego ∥ third-person video and a `NAV ...` result line.

| subpackage | stage | read this if… |
|---|---|---|
| [`envs/`](envs/README.md) | 1 · seeded scene, instruction templates | you want to know what a "scene" or "instruction" is |
| [`navigation/`](navigation/README.md) | 2 · keyword target, 3 · Navigator, 5 · closed loop | you want the control loop and the deployment fixes |
| [`locomotion/`](locomotion/README.md) | 4 · PPO walking policy, raw-env evaluation | you want how the policy was trained and measured |
| [`apps/`](apps/README.md) | dashboards | you want to replay results or drive a run interactively |
| `paths.py` | — | all repository / machine paths and their environment-variable overrides |

**Entry points:** every runnable module has `main(argv) -> int` and is run as `python -m g1nav.<subpackage>.<module>`
(the dashboards with `streamlit run src/g1nav/apps/<name>.py`).

**Install:** `pip install -e .` (base: numpy only) plus extras: `.[sim]` for MuJoCo / MJX / brax, `.[dashboard]`,
`.[eagle]` (optional, separate environment), `.[test]`.

**Provenance:** author's code; the walking environment, reward and PPO hyperparameters are MuJoCo Playground's
(`G1JoystickFlatTerrain`), see [docs/third_party.md](../../docs/third_party.md).
