# Notes for `scripts/moe_svd`

- In `gguf-py`, `ReaderTensor.shape` stores logical GGUF dimensions (in file order), while `ReaderTensor.data.shape` can be quantized byte-layout; use `reversed(tensor.shape.tolist())` for true matrix dimensions.
- For quantized tensors, dequantization should go through `gguf.dequantize(tensor.data, tensor.tensor_type)`; relying on raw `tensor.data` gives byte-packed rows and invalid SVD input.
- llama.cpp HF cache compatibility follows env precedence: `LLAMA_CACHE` -> `HF_HUB_CACHE`/`HUGGINGFACE_HUB_CACHE` -> `HF_HOME/hub` -> `XDG_CACHE_HOME/huggingface/hub` -> `~/.cache/huggingface/hub`.
- Run `scripts/tests/test_moe_svd_*` with `PYTHONPATH=/root/llama.cpp/scripts` so `from moe_svd...` imports resolve during pytest collection.
