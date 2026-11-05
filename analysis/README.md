# analysis/ — evidence and claim checks

**Status:** run in CI on every push.

**Inputs → outputs:** committed files (checkpoint pickles read without JAX by a restricted unpickler, metrics JSONL,
rollout JSON, cognition JSON, MP4/GIF frame counts, the CPU re-measurements in `results/`) → recomputed numbers →
checked against the literal text of `README.md`, `report/archive/report.md` and `media/README.md`.

**Entry points**
```bash
python -m analysis.verify_claims            # prints "70 claims checked, 0 failed."; exit 1 on any mismatch
python -m analysis.verify_claims --md out.md
```
Needs `pip install -e .` (it imports `g1nav` for the scene generator and the keyword rule) plus numpy and Pillow.

**Layout:** `evidence.py` (readers for the committed result files), `verify_claims.py` (the claim list and checker).

**Provenance:** author's code; the claim-by-claim audit behind it is [docs/claims.md](../docs/claims.md).
