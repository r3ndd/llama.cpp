## MoE trace learnings

- MoE trace parity should use both `ffn_moe_topk` and `ffn_moe_argsort`: compare top-k IDs against the argsort prefix per token, and validate top-k IDs/weights (range, uniqueness, finite) before appending trace rows.
- Keep trace validation fail-safe: if a layer batch fails parity/consistency checks, drop that batch and emit a warn-once signal instead of aborting inference.
- For revised Algorithm-2 traces, capture per-selected-expert outputs from `experts` immediately after `ffn_moe_down(_scaled)` and before `ffn_moe_weighted`; this preserves per-expert vectors aligned to `topk_ids`/`topk_weights`.
- When writing NPZ `metadata.json`, materialize `meta.str()` to a local `std::string` before building a byte vector; constructing a vector from iterators over temporary `meta.str()` causes corrupted metadata bytes.
