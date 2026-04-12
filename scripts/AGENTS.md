## Scripts learnings

- MoE trace NPZ v1 stores arrays as `.npy` members plus a raw `metadata.json` ZIP entry; parse metadata via `zipfile` (or equivalent ZIP reader) rather than relying on `np.load()` keys alone.
- Some trace NPZs include malformed/non-JSON `metadata.json`; treat metadata as optional and continue with array-derived dimensions after warning.
- ELT1 conversion must enforce runtime-loader invariants pre-write: `scaling_mode == s_missing` (with alias normalization), finite centroid/contribution values, and `replaced_count <= n_expert - n_topk` per layer.
- When NPZ includes `replaced_expert_mask`, keep it in lockstep with replaced-experts JSON during conversion; mismatch should hard-fail to avoid silent sidecar/JSON drift.
