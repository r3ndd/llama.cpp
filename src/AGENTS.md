## src core integration learnings

- Keep feature gating and graph-eval callback wiring in `llama_context` (where `llama_context_params` becomes `llama_cparams`); when rebuilding graphs, re-install via `ggml_backend_sched_set_eval_callback` so user callbacks still fan out.
- Treat `llama_context_params` edits as a 3-file contract change: `include/llama.h`, `llama_context_default_params()`, and `common_context_params_to_llama()`.
- Do not use `LLAMA_COMMIT` from `src/`; it is provided by `common/` and missing in standalone core `llama` builds.
