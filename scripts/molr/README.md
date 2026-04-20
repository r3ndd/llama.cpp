# MoLR scripts (Phase 0 + Phase 4)

This directory contains operator scaffolding for the MoLR pilot workflow.

- **Phase 0**: spectral baseline capture/archive around `scripts/analyze_moe_svd.py`
- **Phase 1**: plan generation from SVD report and covariance artifact contracts
- **Phase 2**: per-expert MoLR training + merged validation/failure reporting
- **Phase 3**: fallback calibration + bundle packaging manifest
- **Phase 4**: guarded runtime integration scaffold (opt-in config + telemetry + shadow harness)

Phase 0 goal (from design): produce and archive a reproducible spectral baseline using existing
`scripts/analyze_moe_svd.py` for pilot model:

- `unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M`

## What is included in Phase 0

- A repeatable baseline command template around `scripts/analyze_moe_svd.py`
- A checklist-oriented report check for:
  - non-zero / plausible expert-matrix coverage
  - required full-SVD fidelity mode
  - quantization caveat visibility
- Artifact archiving helpers for:
  - `svd_report.json`
  - `run_metadata.json`
  - `phase0_manifest.json` (with hashes)

## What is included in Phase 1

- `plan_from_svd.py`: consumes `svd_report.json` and emits `molr_plan.json` with:
  - per-expert/per-matrix rank decisions,
  - nearest-above fraction-grid energy policy,
  - strided rank-index assignment map for K components,
  - explicit accounting for skipped matrix records.
- `capture_expert_covariance.py`: layer-input-first covariance contracts:
  - consumes pre-captured routed inputs contract NPZ (`inputs`, `layers`, `experts`),
  - supports integrated capture mode behind explicit `--capture-layer-traces` (`--capture-routed-traces` kept as deprecated alias),
  - defaults to layer-granularity covariance fitting (`--input-granularity auto|layer`),
  - computes per-layer `mu` and Cholesky factors with jitter retries,
  - emits `covariance_stats.npz` + `covariance_summary.json` in v2 schema for layer mode,
  - supports compatibility expert mode (`--input-granularity expert`) with v1 output,
  - optionally emits layer trace artifact NPZ (`molr_layer_traces_npz.v1`) in layer mode,
  - tracks explicit layer/expert failure reasons,
  - supports `--allow-empty` for scaffold-only runs when routed-input capture integration is not yet wired.

## What is included in Phase 2

- `train_expert_molr.py`: trains one expert MoLR from:
  - `molr_plan.json` (rank/K/init partition contract),
  - `covariance_stats.npz` (`mu`, `chol` for the expert),
  - full expert weights NPZ (`gate/up/down` or `w1/w3/w2` aliases).
- Training objective contract:
  - `L_total = L_mse + λ_lb * L_load_balance + λ_err * L_error_head`
  - `L_mse`: output MSE to frozen full expert targets,
  - `L_load_balance`: sum of squared batch-mean router weights,
  - `L_error_head`: MSE(predicted_error, true_error_detached), with detached true-error target.
- Validation metrics emitted per expert:
  - cosine similarity mean,
  - relative output norm error mean,
  - router entropy mean,
  - error-head Pearson correlation.
- `train_all_experts.py`: orchestrates expert-by-expert training and emits:
  - merged validation report (`molr_validation_report.json`),
  - failure ledger (`molr_failure_ledger.json`),
  - per-expert checkpoints and validation JSON files.

## What is included in Phase 3

- `calibrate_fallback.py`: sweeps fallback thresholds using per-expert validation summaries and emits
  `molr_thresholds.json` with:
  - explicit schema version (`molr_thresholds.v1`),
  - quality-vs-fallback lookup table,
  - quality profile threshold selections,
  - full-expert cache candidate recommendations.
- `package_molr_bundle.py`: packages plan + thresholds + checkpoints into bundle layout and emits
  `molr_bundle_manifest.json` with:
  - explicit schema version (`molr_bundle_manifest.v1`),
  - SHA-256 checksums for packaged artifacts,
  - compatibility metadata for plan/threshold/checkpoint schema contracts,
  - plan-vs-checkpoint coverage accounting.

## 1) Produce the baseline SVD report

```bash
python scripts/analyze_moe_svd.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --out-json "<run>/svd_report.json" \
  --full-svd \
  --workers 16 \
  --blas-threads 1
```

Notes:
- `--full-svd` is required.
- For reproducibility, keep worker and BLAS settings in metadata (captured below).

## 2) Validate checklist intent on `svd_report.json`

```bash
python scripts/molr/check_phase0_svd_report.py \
  --svd-report "<run>/svd_report.json" \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --strict-coverage \
  --out-json "<run>/phase0_check.json"
```

This check verifies:
- report schema and `run.model_spec`
- `run.fidelity_mode == full_svd`
- candidate/analyzed coverage is non-zero and plausible
- whether quantization caveat text references `Q4_K_M` in report caveats

Use `--allow-model-mismatch` only when intentionally validating a non-pilot run.

## 3) Archive baseline artifacts + metadata

```bash
python scripts/molr/archive_phase0_baseline.py \
  --svd-report "<run>/svd_report.json" \
  --run-dir "<archive-run-dir>" \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --workers 16 \
  --blas-threads 1 \
  --strict-coverage
```

Outputs in `--run-dir`:
- `svd_report.json` (archived copy)
- `run_metadata.json`
- `phase0_manifest.json`

`run_metadata.json` includes:
- command template + argv used for baseline reproduction
- coverage plausibility summary
- environment capture (`hostname`, `git_commit`, `git_branch`, python path)
- quantization caveat annotation for Q4_K_M runs

## Quantization caveat (operator-facing)

For pilot model `Q4_K_M`, singular spectra are derived from quantized GGUF weights.
Interpret energy/compressibility conclusions with this caveat:

- quantization can shift singular value distribution compared with FP16/BF16 checkpoints
- rank heuristics based on this baseline are pilot-oriented and should be cross-checked later

## Phase 1 usage

### 4) Build planning artifact from SVD report

```bash
python scripts/molr/plan_from_svd.py \
  --svd-json "<run>/svd_report.json" \
  --target-energy 0.90 \
  --k-components 4 \
  --out-json "<run>/molr_plan.json"
```

Plan schema version:
- `molr_plan.v1`

SVD compatibility requirement:
- `svd_report.schema_version == "1.1"`
- `svd_report.run.fidelity_mode == "full_svd"`

### 5) Capture covariance artifacts from routed-input contract

```bash
python scripts/molr/capture_expert_covariance.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --tokens 50000 \
  --routed-inputs-npz "<run>/routed_inputs_contract.npz" \
  --min-samples-per-expert 16 \
  --out-npz "<run>/covariance_stats.npz" \
  --out-json "<run>/covariance_summary.json"
```

Optional compatibility expert-granularity mode:

```bash
python scripts/molr/capture_expert_covariance.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --tokens 50000 \
  --input-granularity expert \
  --routed-inputs-npz "<run>/routed_inputs_contract.npz" \
  --min-samples-per-expert 16 \
  --out-npz "<run>/covariance_stats.npz" \
  --out-json "<run>/covariance_summary.json"
```

Required routed-input NPZ arrays:
- layer mode: `inputs` (`[N, d_model]`), `layers` (`[N]`)
- expert mode: `inputs` (`[N, d_model]`), `layers` (`[N]`), `experts` (`[N]`)

Covariance artifact schema versions:
- layer mode (default):
  - `covariance_summary.json`: `molr_covariance_summary.v2`
  - `covariance_stats.npz`: `molr_covariance_npz.v2` with `granularity="layer"`
- expert mode (compatibility):
  - `covariance_summary.json`: `molr_covariance_summary.v1`
  - `covariance_stats.npz`: `molr_covariance_npz.v1`

If routed capture data is unavailable yet, you can produce explicit empty scaffold artifacts:

```bash
python scripts/molr/capture_expert_covariance.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --tokens 0 \
  --allow-empty \
  --out-npz "<run>/covariance_stats.npz" \
  --out-json "<run>/covariance_summary.json"
```

Integrated capture mode (explicit trigger):

```bash
python scripts/molr/capture_expert_covariance.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --tokens 50000 \
  --capture-layer-traces \
  --capture-prompts-jsonl "scripts/molr/layer_capture_prompts.json" \
  --capture-trace-jsonl "<run>/moe_routed_trace.jsonl" \
  --capture-seed 42 \
  --min-samples-per-layer 16 \
  --out-layer-traces-npz "<run>/layer_traces.npz" \
  --out-npz "<run>/covariance_stats.npz" \
  --out-json "<run>/covariance_summary.json"
```

Starter prompt set (safe, non-sensitive, multi-domain) is provided at:
- `scripts/molr/layer_capture_prompts.json`

Capture mode `--capture-prompts-jsonl` input contract:

- **JSONL format**: one JSON record per line.
- **JSON format**: top-level JSON object with `records` array (or a top-level array).

Each record schema:
- `prompt`: string (**required**)
- `inference_params`: object (optional)

`inference_params` supports these bridge keys directly:
- `n_predict`, `seed`, `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `repeat_last_n`,
  `n_ctx`, `n_batch`, `n_ubatch`, `no_display_prompt`, `extra_cli_args`

Runtime capture bridge behavior:
- For each record, the script runs `llama-cli` (or `--capture-llama-cli`) with MoE routed tracing enabled.
- The script requires a routed-trace sink path via `--capture-trace-jsonl` or `LLAMA_MOE_TRACE_JSONL`.
- The bridge accepts trace JSONL rows with `layer` + `inputs` (layer events) and legacy routed rows (`layer`, `expert`, `inputs`).

Runtime MoE trace controls (default off, additive):
- `--moe-trace` / `--no-moe-trace`
- `--moe-trace-granularity {layer,expert}`
- `--moe-trace-path PATH` (env: `LLAMA_MOE_TRACE_JSONL`)
- `--moe-trace-format jsonl` (env: `LLAMA_MOE_TRACE_FORMAT`)
- `--moe-trace-precision {f16,f32}`
- `--moe-trace-sample-rate FLOAT`
- `--moe-trace-max-rows-total N`
- `--moe-trace-max-rows-per-layer N`
- `--moe-trace-max-rows-per-expert N`
- `--moe-trace-buffer-rows N`
- `--moe-trace-flush-interval-ms N`
- `--moe-trace-strict`

Privacy/safety note:
- Routed MoE input vectors may leak prompt semantics; keep trace files on restricted local storage.

Flag compatibility highlights:
- `--capture-layer-traces` is mutually exclusive with `--routed-inputs-npz`
- `--capture-prompts-jsonl` is required when capture mode is enabled
- `--capture-trace-jsonl` (or env `LLAMA_MOE_TRACE_JSONL`) is required in capture mode
- `--out-layer-traces-npz` is valid only in capture mode
- `--capture-routed-traces` is accepted as a deprecated alias during transition

Summary accounting adds per-expert sample provenance fields:
- `sample_source` (`routed` or `layer_fallback`)
- `routed_sample_count`
- `layer_sample_count`
- `effective_sample_count`
- `fallback_applied`

## Phase 2 usage

### 6) Train a single expert

```bash
python scripts/molr/train_expert_molr.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --plan-json "<run>/molr_plan.json" \
  --cov-npz "<run>/covariance_stats.npz" \
  --weights-npz "<run>/expert_weights_12_34.npz" \
  --layer 12 \
  --expert 34 \
  --steps 20000 \
  --batch-size 512 \
  --lr 1e-4 \
  --out-checkpoint "<run>/checkpoints/molr_expert_12_34.npz" \
  --out-validation "<run>/validation/molr_validation_12_34.json"
```

### 7) Train all experts (orchestrated)

```bash
python scripts/molr/train_all_experts.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --plan-json "<run>/molr_plan.json" \
  --cov-npz "<run>/covariance_stats.npz" \
  --weights-dir "<run>/expert_weights" \
  --weights-pattern "expert_weights_{layer}_{expert}.npz" \
  --steps 20000 \
  --batch-size 512 \
  --lr 1e-4 \
  --out-dir "<run>/phase2"
```

Key outputs under `--out-dir`:
- `checkpoints/molr_expert_<layer>_<expert>.npz`
- `validation/molr_validation_<layer>_<expert>.json`
- `molr_validation_report.json`
- `molr_failure_ledger.json`

## Phase 3 usage

### 8) Calibrate fallback thresholds from Phase-2 validation outputs

```bash
python scripts/molr/calibrate_fallback.py \
  --checkpoints "<run>/phase2/checkpoints" \
  --validation-dir "<run>/phase2/validation" \
  --quality-profiles "balanced:0.90,quality:0.95,strict:0.98" \
  --top-cache-candidates 32 \
  --out-json "<run>/molr_thresholds.json"
```

Output schema version:
- `molr_thresholds.json`: `molr_thresholds.v1`

### 9) Package runtime-consumable Phase-3 bundle

```bash
python scripts/molr/package_molr_bundle.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --plan-json "<run>/molr_plan.json" \
  --checkpoints "<run>/phase2/checkpoints" \
  --thresholds-json "<run>/molr_thresholds.json" \
  --out-dir "<run>/bundle"
```

Bundle outputs:
- `bundle/molr_plan.json`
- `bundle/molr_thresholds.json`
- `bundle/checkpoints/molr_expert_<layer>_<expert>.npz`
- `bundle/molr_bundle_manifest.json` (`molr_bundle_manifest.v1`)

Use `--emit-runtime-config-template` to also emit:
- `bundle/runtime_config.template.json` (disabled by default; explicit opt-in required)

Use `--require-all-plan-experts` to hard-fail packaging if any expert from `molr_plan.json`
does not have a packaged checkpoint.

## Phase 4 usage (guarded runtime scaffold)

Phase 4 in this directory intentionally keeps default inference behavior unchanged and provides
opt-in runtime contracts + shadow validation scaffolding.

### 10) Emit telemetry snapshot from runtime event logs

```bash
python scripts/molr/runtime_telemetry.py \
  --events-jsonl "<run>/runtime_events.jsonl" \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --out-json "<run>/molr_runtime_telemetry.json"
```

Telemetry schema version:
- `molr_runtime_telemetry.v1`

Expected per-line event fields (JSONL):
- `layer` (int)
- `expert` (int)
- `used_fallback` (bool)
- `predicted_error` (float)
- `molr_latency_ms` (float)
- `fallback_latency_ms` (float)

### 11) Validate bundle + runtime config and summarize shadow telemetry

```bash
python scripts/molr/runtime_shadow.py \
  --bundle-dir "<run>/bundle" \
  --runtime-config-json "<run>/runtime_config.json" \
  --telemetry-json "<run>/molr_runtime_telemetry.json" \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --require-explicit-enable \
  --out-json "<run>/molr_runtime_shadow_report.json"
```

Shadow report schema version:
- `molr_runtime_shadow_report.v1`

What this validates:
- bundle manifest/plan/threshold/checkpoint compatibility contracts,
- runtime config opt-in state and threshold resolution,
- telemetry counters for fallback rate and MoLR-vs-fallback latency comparisons,
- auto-disable recommendations for high fallback-rate / high latency-ratio experts.

### Runtime config contract (Phase 4 scaffold)

`runtime_config.json` fields:
- `schema_version`: `molr_runtime_config.v1`
- `enabled`: bool (must be `true` when `--require-explicit-enable` is used)
- one of:
  - `quality_profile`: profile name from `molr_thresholds.json`, or
  - `fallback_threshold`: explicit numeric threshold
- `telemetry_enabled`: bool

Rollback-safe default:
- Keep `enabled=false` (default) to preserve baseline full-expert behavior.
