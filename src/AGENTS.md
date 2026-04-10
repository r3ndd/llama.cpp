# src core integration learnings

- Use `llama_context` as the feature-gating choke point: architecture-specific toggles/warnings belong where `llama_context_params` is materialized into `llama_cparams`.
- For graph-time observability, fan out eval notifications from a wrapped scheduler callback in `llama_context`; install via `ggml_backend_sched_set_eval_callback` on graph rebuild to preserve existing user callbacks.
- Treat `llama_context_params` changes as a 3-file contract update: `include/llama.h`, `llama_context_default_params()`, and `common_context_params_to_llama()`.
- Avoid `LLAMA_COMMIT` in `src/`; it is exported by `common/` and unavailable when building the standalone core `llama` target.
