# Notes for `scripts/moe_svd`

## Tensor layout and metadata

- In `gguf-py`, `ReaderTensor.shape` is the logical GGUF shape (file order), while `ReaderTensor.data.shape` may be quantized byte layout; derive matrix dims from `reversed(tensor.shape.tolist())`.
- Always dequantize quantized tensors via `gguf.dequantize(tensor.data, tensor.tensor_type)` before SVD; raw `tensor.data` rows are byte-packed and invalid for linear algebra.
- MoE metadata is architecture-scoped (`<architecture>.expert_count`, `<architecture>.expert_used_count`, `<architecture>.block_count`), not always `llama.*`.
- Qwen3.5 MoE stores routed experts in packed 3D `ffn_*_exps.weight` tensors; shared experts remain 2D (`ffn_*_shexp.weight`), so candidate counts look low unless 3D tensors are unpacked.
- For packed 3D experts, detect expert axis by matching `<architecture>.expert_count` against logical dimensions; if ambiguous, skip with an explicit unknown-layout reason instead of guessing.
- Keep `MatrixRef.shape` aligned with the actual dequantized 2D slice in `load_matrix_from_reader` and validate shape immediately to catch axis/order mistakes early.

## Metrics and spectral energy

- Compute `explained_spectral_energy_rank_r` from squared singular values: `sum(s[:r]^2) / sum(s^2)`.
- Treat zero-total spectral energy as a handled edge case (`0.0` + warning) and reject non-finite results.
- Use the fixed 19-point rank-fraction grid (`0.05..0.95`) for spectral-energy reporting: per-matrix in `explained_spectral_energy_rank_fractions`, summary in `spectral_energy_rank_fractions` with aligned `explained_spectral_energy_rank_fractions_mean`.

## Performance and parallelization

- In packed-expert paths, cache each dequantized 3D source tensor once and slice experts from it; re-dequantizing per expert causes large avoidable overhead.
- `analyze_matrix` only needs singular values, so `np.linalg.svd(..., compute_uv=False)` preserves metrics while reducing memory and runtime.
- Phase split for parallel execution: keep tensor discovery/dequantization-caching setup deterministic (phase 0), then parallelize matrix SVD/metrics work (phase 1).
- For multiprocessing SVD, prevent BLAS oversubscription (`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`) and partition by `source_tensor_name` to preserve packed-tensor cache reuse per worker.
- Apply BLAS thread limits in the parent process too; sequential paths (`workers=1`, `--fail-fast`) bypass worker initialization.

## Dev/test ergonomics

- llama.cpp HF cache lookup precedence: `LLAMA_CACHE` -> `HF_HUB_CACHE`/`HUGGINGFACE_HUB_CACHE` -> `HF_HOME/hub` -> `XDG_CACHE_HOME/huggingface/hub` -> `~/.cache/huggingface/hub`.
- Run `scripts/tests/test_moe_svd_*` with `PYTHONPATH=<repo>/scripts` so `from moe_svd...` imports resolve during pytest collection.
