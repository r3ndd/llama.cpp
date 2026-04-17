# Notes for `scripts/moe_svd`

- In `gguf-py`, `ReaderTensor.shape` stores logical GGUF dimensions (in file order), while `ReaderTensor.data.shape` can be quantized byte-layout; use `reversed(tensor.shape.tolist())` for true matrix dimensions.
- For quantized tensors, dequantization should go through `gguf.dequantize(tensor.data, tensor.tensor_type)`; relying on raw `tensor.data` gives byte-packed rows and invalid SVD input.
- llama.cpp HF cache compatibility follows env precedence: `LLAMA_CACHE` -> `HF_HUB_CACHE`/`HUGGINGFACE_HUB_CACHE` -> `HF_HOME/hub` -> `XDG_CACHE_HOME/huggingface/hub` -> `~/.cache/huggingface/hub`.
- Run `scripts/tests/test_moe_svd_*` with `PYTHONPATH=<repo>/scripts` so `from moe_svd...` imports resolve during pytest collection.
- For non-llama architectures (e.g. `qwen35moe`), GGUF MoE metadata is stored under `<architecture>.*` keys (`qwen35moe.expert_count`, `qwen35moe.expert_used_count`, `qwen35moe.block_count`) rather than `llama.*`.
- Qwen3.5 MoE GGUF packs routed experts into 3D tensors (`ffn_*_exps.weight`, shape `[experts, ...]`) and exposes only shared expert matrices as 2D (`ffn_*_shexp.weight`), so low candidate counts can be expected unless 3D tensors are explicitly unpacked.
- For packed 3D expert tensors, infer the expert axis from `<architecture>.expert_count` against logical `reversed(tensor.shape.tolist())`; if no unique axis matches, skip with an explicit unknown-layout reason instead of assuming an axis.
- Keep `MatrixRef.shape` consistent with dequantized matrix slices in `load_matrix_from_reader`; validate the produced 2D shape against the discovered shape to catch axis/order mismatches early.
- `explained_spectral_energy_rank_r` should be computed from squared singular values (`sum(s[:r]^2)/sum(s^2)`), not Frobenius-on-reconstruction shortcuts; this keeps rank-retention comparable across matrices.
- Treat zero-total spectral energy as a handled edge case (`0.0` plus warning) and reject non-finite spectral-energy results to surface invalid decompositions early.
