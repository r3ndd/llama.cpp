# Notes for `src/`

- MoE routed-input tracing is a two-stage callback flow: `graph_get_cb()` only names tensors / snapshots ubatch metadata, while scheduler `cb_eval` is the only place with computed tensor values.
- Keep MoE trace tensor names stable: `graph_get_cb()` receives base names (`ffn_moe_topk`, `ffn_moe_expert_in`) and appends `-<layer>`; eval-side parsing depends on those exact prefixes.
- In `llama_context::graph_eval_cb`, preserve callback composition semantics: ask-phase uses OR (`trace_wants || user_wants`), eval-phase uses AND (`trace_ok && user_ok`) so user callbacks still receive tensors and can fail evaluation.
- Keep `graph_eval_trace_wants()` aligned with `llama_moe_trace::wants_tensor()` granularity filtering; otherwise ask-phase may force unnecessary tensor materialization (e.g. topk/expert tensors in layer mode).
- In `build_moe_ffn()`, tag `ffn_moe_expert_in` at the tensor immediately feeding routed expert matmul; later tensors are no longer the canonical routed input vector.
- For new string fields in `llama_context_params` / `llama_cparams`, move lifetime ownership into `llama_context` (`std::string` members + rebind `const char *`) because params are copied by value.
