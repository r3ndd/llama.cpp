# MoLR scripts (Phase 0 + Phase 2)

This directory contains operator scaffolding for the MoLR pilot workflow.

- **Phase 0**: spectral baseline capture/archive around `scripts/analyze_moe_svd.py`
- **Phase 1**: plan generation from SVD report and covariance artifact contracts
- **Phase 2**: per-expert MoLR training + merged validation/failure reporting

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
- `capture_expert_covariance.py`: scaffold for routed-input covariance contracts:
  - consumes pre-captured routed inputs contract NPZ (`inputs`, `layers`, `experts`),
  - computes per-expert `mu` and Cholesky factors with jitter retries,
  - emits `covariance_stats.npz` + `covariance_summary.json`,
  - tracks explicit per-expert failure reasons,
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

Required routed-input NPZ arrays:
- `inputs`: float array shaped `[N, d_model]`
- `layers`: int array shaped `[N]`
- `experts`: int array shaped `[N]`

Covariance artifact schema versions:
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
