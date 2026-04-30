# imatrix agent notes

## Build and wiring

- MoE covariance flag wiring spans `common/common.h` (state), `common/arg.cpp` (parse + cross-flag validation), and `tools/imatrix/imatrix.cpp` (runtime write path); partial edits leave flags parsed but ineffective.
- `llama-imatrix` sources are explicitly listed in `tools/imatrix/CMakeLists.txt`; new helper `.cpp` files must be added there or they will not build.
- `tools/imatrix/moe-cov-io.cpp` uses llama symbols (`llama_model_desc`) even with `model=nullptr`; standalone tests reusing it must link `llama` (not only `common`).

## Covariance format, merge, and IO

- Append compatibility is enforced at load-time (`general.type`, `moe_cov.version`, `moe_cov.convention`, `moe_cov.precision`, model fingerprint); parse-time still requires `--moe-trace-cov-out`.
- Append merge identity is `(layer, expert, role, role_variant, tensor_name)`; overlaps merge `n/sum/outer` in precision domain, then recompute `cov_pop`.
- Covariance writes use temp-file + replace semantics; append preserves existing `moe_cov.created_at` and unions `moe_cov.sources` with the current prompt source.
- `std::filesystem::rename(tmp, out)` may fail when `out` exists; keep the `copy_file(..., overwrite_existing)` + temp cleanup fallback for portability.
- Propagate `gguf_write_to_file(...)` boolean failures to user-facing errors with specific context.
- `--moe-trace-cov-targets` parsing should reject empty CSV elements (for example trailing commas) to match strict layer/expert SPEC validation.

## Expert-role classification

- Role mapping for `GGML_OP_MUL_MAT_ID` must use routed weight tensor name from `src0` and routed activation vector from `src1`; both are needed for correct per-`(layer,expert,role)` stats.
- In alias matching, test merged `gate_up`/`up_gate` patterns before generic `ffn_gate*`; otherwise merged SwiGLU tensors are mislabeled `gate` instead of `up` (`role_variant=gate_up_merged`).
- GGML has no dedicated float8 tensor enum; `--moe-trace-cov-precision f8` payloads are serialized via `GGML_TYPE_I8`, with lossy integer-domain accumulation.
- Even when accumulation precision is `f8`, write `moe_cov.*.cov_pop` as `GGML_TYPE_F32`; storing covariance in int8 rounds sub-unit population covariance to zero (notably `down` paths with sparse/small activations).
