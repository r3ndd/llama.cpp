# Notes for `scripts/molr`

## Cross-phase contracts

- Keep artifact schema/version constants in `scripts/molr/types.py`; Phase 0-4 tools are intentionally wired through those shared constants.
- Planning is coupled to `scripts/moe_svd/svd_metrics.py:SPECTRAL_ENERGY_RANK_FRACTIONS` (fixed 19-point grid). Grid/order changes are artifact compatibility changes, not local tuning.

## Phase 0/1 validation + planning

- Phase 0 accepts only `svd_report.json` schema `1.1` and `run.fidelity_mode == "full_svd"`; model mismatch is reject-by-default unless explicitly overridden.
- Coverage checks are intentionally heuristic (non-zero + plausibility), and `--strict-coverage` upgrades any non-`pass` (including `warn`) to hard failure.
- Quantization caveat gating is string-coupled (`"Q4_K_M"` in `assumptions_and_caveats`), so caveat text edits can silently change pass/fail behavior.
- With `--allow-model-mismatch` during archiving, metadata should still anchor to `svd_report.run.model_spec` to preserve replay provenance.
- Covariance capture expects routed-input NPZ arrays (`inputs`, `layers`, `experts`); `--allow-empty` is scaffold-only and should not be treated as normal missing-data recovery.
- Covariance counters intentionally separate intent: `experts_observed_total` (before cap) vs `experts_processed_total` (after `--max-experts`).

## Phase 2 training/orchestration

- `train_expert_molr.py` requires full expert weights for gate/up/down (`w1/w3/w2` aliases accepted); matrix orientation is inferred from covariance `d_model` and then validated against down-projection intermediate size.
- Error-head training uses detached true-error targets; validation pass/fail is driven by cosine + error-correlation thresholds (not full objective value).
- `train_all_experts.py` deterministically sorts experts `(layer, expert)` before optional truncation, and derives per-expert seeds as `base_seed + sorted_index`.
- Orchestration records pre-subprocess skips (`train_skipped_missing_cov`, `train_skipped_missing_weights`) and still emits merged validation + failure-ledger artifacts even when zero experts train successfully.
- Phase 2 keeps strict compatibility gates: `--model` must match `molr_plan.json:model_spec` (when present) and covariance schema must be `molr_covariance_npz.v1`.

## Phase 3 calibration/packaging

- `calibrate_fallback.py` enforces strict validation-row contracts (`molr_expert_validation.v1`): non-numeric error means, invalid `layer`/`expert`, or non-list `failure_reasons` are hard failures.
- Threshold sweep is contract-defined as `pred_error_mean > threshold`, with candidate thresholds built from `{0.0} ∪ unique(pred_error_mean)`.
- Profile selection is asymmetric by design: pick the lowest fallback-rate threshold meeting `quality_proxy_min`; if none meet target, fall back to the max-quality row.
- Cache-candidate ranking intentionally front-loads non-`pass` experts, then highest predicted/true error, to prioritize full-weight retention where MoLR is least reliable.
- `package_molr_bundle.py` validates plan/threshold/checkpoint schemas and hard-fails on filename-vs-NPZ `(layer, expert)` mismatches or duplicate expert checkpoints.
- Bundle coverage is explicit in `molr_bundle_manifest.v1`; missing plan experts are always listed, and `--require-all-plan-experts` upgrades that gap to hard failure.

## Phase 4 guarded runtime scaffold

- `package_molr_bundle.py` stamps runtime compatibility schema versions in the manifest and can emit `runtime_config.template.json` with `enabled=false` as rollback-safe default.
- `runtime_bundle.py` enforces strict manifest compatibility (plan/threshold/checkpoint/runtime schemas) and requires exactly one threshold selector: `fallback_threshold` xor `quality_profile`.
- Manifest checkpoint paths can be absolute or bundle-relative; runtime loading normalizes relative paths against `bundle_dir` before NPZ schema/model checks.
- `runtime_shadow.py` is validation-only and emits recommendations only; explicit-enable enforcement is opt-in (`--require-explicit-enable`), not default.
- `runtime_telemetry.py` is strict on event typing (`used_fallback` must be boolean) and computes `avg_fallback_latency_ms` over fallback calls only.
