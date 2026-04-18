# MoLR scripts (Phase 0 baseline)

This directory contains **Phase 0** operator scaffolding for the MoLR pilot workflow.

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

No Phase 1+ MoLR planning/training/runtime integration is implemented here.

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
