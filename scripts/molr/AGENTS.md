# Notes for `scripts/molr`

## Phase 0/1 contracts

- Phase 0 accepts only `svd_report.json` schema `1.1` with `run.fidelity_mode == "full_svd"`; model mismatch is reject-by-default and only allowed via explicit override.
- Coverage plausibility is intentionally checklist-light (`candidates/analyzed > 0`, analyzed/candidates heuristic, layer/expert presence); `--strict-coverage` upgrades any non-`pass` (including `warn`) to hard failure.
- Quantization-caveat checks are string-coupled (`"Q4_K_M"` in `assumptions_and_caveats`), so caveat wording edits can silently change pass/fail signals.
- When `--allow-model-mismatch` is used for archiving, run metadata should be anchored to `svd_report.run.model_spec` so replay metadata matches the analyzed artifact.
- Plan generation is contract-coupled to the fixed 19-point `SPECTRAL_ENERGY_RANK_FRACTIONS` grid from `scripts/moe_svd/svd_metrics.py`; unmet target energy on that grid escalates to full rank with explicit annotation.
- Keep schema/version constants centralized in `scripts/molr/types.py` so Phase 0/1/2 tools stay artifact-compatible.
- Covariance capture consumes routed-input NPZ contract arrays (`inputs`, `layers`, `experts`) and permits explicit scaffold artifacts via `--allow-empty`.
- Covariance summary counters are split by intent: `experts_observed_total` counts unique routed pairs before any `--max-experts` cap; `experts_processed_total` reflects post-cap execution.

## Phase 2 training/orchestration

- `train_expert_molr.py` requires per-expert full weights for gate/up/down (aliases `w1/w3/w2` accepted); matrix orientation is inferred from covariance `d_model` and validated with down-projection intermediate size.
- Phase 2 objective keeps true-error targets detached in the error-head term; validation pass/fail is gated only by cosine and error-correlation thresholds.
- `train_all_experts.py` normalizes execution order by sorting experts `(layer, expert)` before optional `--max-experts` truncation; per-expert seeds are `base_seed + sorted_index`.
- Orchestration records pre-subprocess skips (`train_skipped_missing_cov`, `train_skipped_missing_weights`) and always emits both merged validation report and failure ledger, even when zero experts complete successfully.
- Model/schema checks remain strict across Phase 2: `--model` must match `molr_plan.json:model_spec` when present, and covariance schema must match `molr_covariance_npz.v1`.
