## MoE trace learnings

- MoE trace parity should use both `ffn_moe_topk` and `ffn_moe_argsort`: compare top-k IDs against the argsort prefix per token, and validate top-k IDs/weights (range, uniqueness, finite) before appending trace rows.
- Keep trace validation fail-safe: if a layer batch fails parity/consistency checks, drop that batch and emit a warn-once signal instead of aborting inference.
