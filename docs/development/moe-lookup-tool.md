# MoE Lookup Table Builder (Algorithm 2 PoC)

This document describes the offline tool for building per-layer shared lookup/contribution tables from MoE trace NPZ v1 artifacts.

- Script: `scripts/build-moe-lookup.py`
- Scope: Qwen3.5-MoE Algorithm 2 PoC (shared table per layer)
- Runtime integration: sidecar generation only (no inference-path wiring required in this stage)

## Input trace requirements

Each input trace NPZ must include these arrays:

- `layer_ids` (`[N]`)
- `token_ids` (`[N]`)
- `h_pre_moe` (`[N, n_embd]`)
- `topk_ids` (`[N, k]`)
- `topk_weights` (`[N, k]`)
- `topk_expert_outputs` (`[N, k, n_embd]`) — separate output for each selected top-k expert

Optional arrays:

- `y_full` (`[N, n_embd]`) may be present for trace analysis; it is not required by this builder.

## CLI usage

Example:

```bash
python3 scripts/build-moe-lookup.py \
  --input trace-run-a.npz \
  --input trace-run-b.npz \
  --output out/qwen35moe.lookup.npz \
  --output-replaced-experts out/qwen35moe.replaced-experts.json \
  --clusters-per-layer 1024 \
  --replace-ratio 0.10 \
  --scaling-mode s_missing
```

Key options:

- `--input` (repeatable): trace NPZ v1 input files
- `--output`: sidecar output NPZ
- `--output-replaced-experts`: replaced experts JSON output (defaults to `<output>.replaced-experts.json`)
- `--replaced-experts-json`: optional pre-defined replaced sets
- `--layers`: optional layer subset (`0,1,3-5`)
- `--clusters-per-layer`: k-means cluster count per layer
- `--kmeans-iters`: k-means iteration count
- `--kmeans-max-samples-per-layer`: cap on rows used to train centroids
- `--replace-ratio`: fraction of experts replaced per layer when deriving sets
- `--scaling-mode`: `s_missing|router_mass_replaced` (`router_mass_replaced` is accepted as a deprecated alias of `s_missing`)

## Target semantics (revised Algorithm 2)

For each token/sample at layer `L`, let `M` be the selected top-k experts that are marked replaced.

- If `M` is empty: token target is the zero vector.
- Else: token target is a removed-only relative mixture:
  - `u = sum_{e in M} (w_e / sum_{j in M} w_j) * y_e`
  - where `w_e` is the router score in selected top-k, and `y_e` is from `topk_expert_outputs`.

The table stores cluster means `U_L[key] = mean(u)`.
At inference, runtime applies mandatory scale by current missing mass `s_missing`.

## Output artifacts

### 1) Lookup sidecar NPZ (`--output`)

Global arrays/fields:

- `format_version` (`int32`, value `1`)
- `algorithm` (`"algorithm2_shared_table"`)
- `layers` (`int32[]`)
- `n_layer_total`, `n_embd`, `n_expert`, `n_topk` (`int32` scalars)
- `replaced_expert_mask` (`bool[n_layer_total, n_expert]`)
- `metadata_json` (JSON string with build parameters)
- `target_semantics` (`"removed_expert_relative_weighted_contribution"`)
- `runtime_scaling`, `scaling_mode` (`"s_missing"`)

Per selected layer `L`:

- `layer_<L>_centroids` (`float16[n_keys_L, n_embd]`)
- `layer_<L>_contributions` (`float16[n_keys_L, n_embd]`)
- `layer_<L>_counts` (`int32[n_keys_L]`)
- `layer_<L>_mean_replaced_mass` (`float32` scalar)

### 2) Replaced experts JSON (`--output-replaced-experts`)

```json
{
  "format_version": 1,
  "n_layer_total": 64,
  "n_expert": 128,
  "layers": {
    "0": [3, 7, 19],
    "1": [4, 9, 20]
  }
}
```

## Validation/error handling

The tool validates:

- required arrays exist
- array rank/shape compatibility (`N`, `k`, `n_embd`)
- finite router weights and non-negative expert IDs
- cross-file consistency (`n_embd`, top-k width)
- metadata mismatches when metadata fields are present (`format_version`, `n_embd`, `n_expert_used`)
- replaced expert IDs are in range when loading from JSON
- argument sanity (`--clusters-per-layer > 0`, `--kmeans-iters > 0`, `--distance-batch-size != 0`)

Trace metadata is treated as optional: if `metadata.json` is missing or malformed, the tool continues using array-derived dimensions.

On validation failure, the tool exits with code `2` and a clear error message.
