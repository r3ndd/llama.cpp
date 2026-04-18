# Notes for `scripts/molr`

- Phase 0 tooling assumes `svd_report.json` schema `1.1` and requires `run.fidelity_mode == "full_svd"`; reject mismatches unless model mismatch is explicitly allowed.
- Keep Phase 0 check/archive CLIs separate: validation-only checks should not depend on archive-specific args like `--run-dir`.
- Coverage plausibility is intentionally lightweight (`candidates > 0`, `analyzed > 0`, analyzed/candidates ratio heuristic, layer/expert presence) and should stay stable for operator checklist gating.
- `--strict-coverage` is a policy gate: any non-`pass` plausibility status (including `warn`) must fail archive/check commands.
- In `archive_phase0_baseline.py`, metadata command/model fields must reflect `svd_report.run.model_spec` when `--allow-model-mismatch` is used; otherwise archived reproduction metadata becomes misleading.
- `check_phase0_svd_report.py` treats quantization-caveat visibility as string presence in `assumptions_and_caveats` (`"Q4_K_M"`), so report wording changes can silently affect that signal.
