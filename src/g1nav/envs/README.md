# g1nav.envs — seeded scenes and instructions

**Status:** used by every run (scene layout); `instructions.py` is used only for the go-to tier in the demos.

**Inputs → outputs:** integer seed → `SceneSpec` (robot spawn at a wall facing inward, 3 coloured objects in its field
of view) → MJCF; (scene, rng) → `Instruction` (text, target label, success rule). Pure numpy; MuJoCo only to validate.

**Entry points**
```bash
python -m g1nav.envs.scene_gen --seed 0 --validate     # writes assets/scene_seed0.xml
python -m g1nav.envs.instructions                      # instructions of seeds 0-5
python -m g1nav.envs.smoke_test                        # load / step / render the vendored G1 (CPU)
```

**Layout:** `scene_gen.py` (sampleScene, buildMjcf, MJCF pieces shared with navigation/integrated_scene.py),
`instructions.py` (three template tiers + success rules), `smoke_test.py`.

**Provenance:** author's code; the G1 model is vendored from MuJoCo Menagerie (`assets/mujoco_menagerie/`).
Characterized by `tests/test_characterization.py` (scenes of seeds 0-19, MJCF hashes, instructions, success rules).
