# Notes for `scripts/molr`

## Cross-phase contracts

- Keep artifact/schema version constants centralized in `scripts/molr/types.py`; phases intentionally share these constants rather than redefining them.
- Planning is contract-coupled to `scripts/moe_svd/svd_metrics.py:SPECTRAL_ENERGY_RANK_FRACTIONS` (fixed 19-point grid). Grid/order edits are compatibility changes.

## Phase 0 validation/planning

- Phase 0 accepts only `svd_report.json` schema `1.1` with `run.fidelity_mode == "full_svd"`; mismatched model is reject-by-default unless explicitly overridden.
- Coverage status is heuristic; `--strict-coverage` upgrades any non-`pass` status (including `warn`) to hard failure.
- Quantization caveat handling is string-coupled (`"Q4_K_M"` searched in `assumptions_and_caveats`), so caveat wording changes can alter gating.

## Phase 1 prompt-driven capture bridge (`capture_expert_covariance.py`)

- Input modes are exclusive: contract mode (`--routed-inputs-npz`) vs capture mode (`--capture-layer-traces --capture-prompts-jsonl`, with `--capture-routed-traces` as deprecated alias).
- Layer granularity is the default (`--input-granularity auto -> layer`): layer-mode contract NPZ only requires `inputs/layers`, while expert mode still requires `inputs/layers/experts`.
- Prompt source accepts JSON object (`records`), single-record JSON object, JSON array, or JSONL; each record requires `prompt`, optional `inference_params`.
- Capture bridge runs one `llama-cli` call per prompt, sets `LLAMA_MOE_TRACE_ENABLE=1`, `LLAMA_MOE_TRACE_FORMAT=jsonl`, `LLAMA_MOE_TRACE_GRANULARITY=<resolved granularity>`, and forces `--no-display-prompt` for machine-readable traces.
- Trace sink path is mandatory in capture mode (`--capture-trace-jsonl` or `LLAMA_MOE_TRACE_JSONL`) and is reset before execution to avoid stale-row contamination.
- Record-level inference params override `--capture-common-inference-params`; default seed becomes `capture_seed + record_index` only when record seed is absent.
- Routed trace loader accepts routed-input events and requires `layer/expert/inputs`; vector width (`d_model`) is locked by first valid row and later rows must match.
- `--allow-empty` is scaffold-only for missing/empty routed rows (including zero-row capture), not a bypass for malformed inputs.
- Summary intentionally distinguishes observed vs processed experts (`experts_observed_total` before `--max-experts` cap, `experts_processed_total` after).

## Phase 2+ runtime pipeline contracts

- `train_all_experts.py` sorts `(layer, expert)` deterministically before truncation and derives per-expert seed as `base_seed + sorted_index`.
- `calibrate_fallback.py` threshold semantics are contract-defined: fallback when `pred_error_mean > threshold`, candidates from `{0.0} ∪ unique(pred_error_mean)`.
- `runtime_bundle.py` requires exactly one threshold selector (`fallback_threshold` xor `quality_profile`) and normalizes bundle-relative checkpoint paths before schema/model checks.
- `runtime_shadow.py` recommendation mode is non-enforcing by default; explicit-enable checks are opt-in via `--require-explicit-enable`.
- `runtime_telemetry.py` treats `used_fallback` as strict boolean input and computes `avg_fallback_latency_ms` over fallback calls only.
