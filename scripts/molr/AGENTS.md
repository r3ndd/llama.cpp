# Notes for `scripts/molr`

- Phase 0 tooling assumes `svd_report.json` schema `1.1` and requires `run.fidelity_mode == "full_svd"`; reject mismatches unless model mismatch is explicitly allowed.
- Keep Phase 0 check/archive CLIs separate: validation-only checks should not depend on archive-specific args like `--run-dir`.
- Coverage plausibility is intentionally lightweight (`candidates > 0`, `analyzed > 0`, analyzed/candidates ratio heuristic, layer/expert presence) and should stay stable for operator checklist gating.
- `--strict-coverage` is a policy gate: any non-`pass` plausibility status (including `warn`) must fail archive/check commands.
- In `archive_phase0_baseline.py`, metadata command/model fields must reflect `svd_report.run.model_spec` when `--allow-model-mismatch` is used; otherwise archived reproduction metadata becomes misleading.
- `check_phase0_svd_report.py` treats quantization-caveat visibility as string presence in `assumptions_and_caveats` (`"Q4_K_M"`), so report wording changes can silently affect that signal.
- Phase 1 plan generation (`plan_from_svd.py`) intentionally consumes the fixed 19-point `SPECTRAL_ENERGY_RANK_FRACTIONS` grid from `moe_svd.svd_metrics`; if target energy is not met on that grid, it escalates to full rank and records this explicitly.
- Keep artifact schema constants centralized in `scripts/molr/types.py` (`molr_plan.v1`, covariance schema versions, SVD schema coupling) so Phase 0/1 tools stay contract-aligned.
- `capture_expert_covariance.py` is contract-driven in Phase 1: it consumes pre-captured routed inputs NPZ (`inputs/layers/experts`) and supports explicit `--allow-empty` scaffold outputs until runtime routed capture integration exists.
- In covariance summaries, `observed.experts_observed_total` must count all unique `(layer, expert)` pairs in the routed-input contract before `--max-experts` truncation; `experts_processed_total` reflects the post-cap subset.
