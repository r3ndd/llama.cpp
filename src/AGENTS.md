## src core integration learnings

- Keep feature gating and graph-eval callback wiring in `llama_context` (where `llama_context_params` becomes `llama_cparams`); on graph rebuild, re-install via `ggml_backend_sched_set_eval_callback` so user callbacks still fan out.
- Treat `llama_context_params` edits as a 3-file contract change: `include/llama.h`, `llama_context_default_params()`, and `common_context_params_to_llama()`.
- Do not use `LLAMA_COMMIT` from `src/`; it is defined in `common/` and absent in standalone core `llama` builds.
- New graph-time feature handles must be threaded through all reuse legs: `llm_graph_params`, `allow_reuse()`, and `llm_graph_context`; missing one causes stale graph reuse.

## src MoE runtime learnings

- MoE trace parity should validate `ffn_moe_topk` against the `ffn_moe_argsort` prefix per token, and check top-k IDs/weights for range, uniqueness, and finiteness before emitting trace rows.
- Keep MoE trace validation fail-safe: on per-layer batch parity failure, drop that batch and warn once instead of aborting inference.
- For revised Algorithm-2 traces, capture per-selected-expert outputs immediately after `ffn_moe_down(_scaled)` and before `ffn_moe_weighted` to preserve alignment with `topk_ids`/`topk_weights`.
- A MoE lookup layer is active only when `replaced_mask` contains at least one replaced expert; sidecar presence alone must not enable lookup.
- `llama_moe_lookup_table::load()` currently requires a readable `--moe-lookup-replaced-experts` sidecar even when ELT1 `replaced_ids` exists; treat the JSON sidecar as a required runtime artifact.
