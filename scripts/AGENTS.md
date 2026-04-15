## Scripts learnings

- MoE trace NPZ v1 stores arrays as `.npy` members plus a raw `metadata.json` ZIP entry; parse metadata via `zipfile` (or equivalent ZIP reader) rather than relying on `np.load()` keys alone.
- Some trace NPZs include malformed/non-JSON `metadata.json`; treat metadata as optional and continue with array-derived dimensions after warning.
- ELT1 conversion must enforce runtime-loader invariants pre-write: `scaling_mode == s_missing` (with alias normalization), finite centroid/contribution values, and `replaced_count <= n_expert - n_topk` per layer.
- When NPZ includes `replaced_expert_mask`, keep it in lockstep with replaced-experts JSON during conversion; mismatch should hard-fail to avoid silent sidecar/JSON drift.
- Evaluation-matrix tooling should be artifact-driven: discover lookup dimensions from existing lookup NPZ (`layers`, per-layer centroid counts, scaling mode) and pair each lookup with `<lookup>.replaced-experts.json` to auto-add matching remove-only rows.
- Lookup NPZs may yield non-uniform per-layer centroid counts (e.g. `< requested k`) after k-means; matrix discovery will emit `clusters_per_layer` as a list of observed counts rather than a single scalar.
- If baseline PPL is unavailable in existing artifacts, eval-matrix summarization still reports throughput deltas and marks quality gate as `N/A` instead of forcing a false fail.
- For Qwen3.5-MoE sidecars, `convert-moe-lookup-to-elt1.py --model-id` must match runtime `model.arch_name()` exactly (`qwen35moe` for current GGUFs); using legacy `qwen3moe` hard-disables lookup at load with a model_id mismatch warning.
- `build-moe-lookup.py` keeps rows with zero replaced mass in clustering/aggregation (targets stay zero and are averaged into centroid contributions); they are not filtered out before table construction.
