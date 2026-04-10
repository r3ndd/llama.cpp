# MoE Lookup Table Builder (Algorithm 2 PoC)

This document describes the offline tool for building per-layer shared lookup/residual tables from MoE trace NPZ v1 artifacts.

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
- `y_full` (`[N, n_embd]`)

Optional arrays used when available:

- `residual_target` (`[N, n_embd]`) – preferred when present
- `y_kept` (`[N, n_embd]`) – fallback source via `y_full - y_kept`

If neither optional residual array is present, the tool can still build a PoC table using a proxy target (`router_mass_replaced * y_full`) for experimentation.

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
  --scaling-mode router_mass_replaced
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
- `--residual-source`: `auto|residual_target|y_full_minus_y_kept|proxy_replaced_mass_y_full`
- `--scaling-mode`: `none|router_mass_replaced`

## Output artifacts

### 1) Lookup sidecar NPZ (`--output`)

Global arrays/fields:

- `format_version` (`int32`, value `1`)
- `algorithm` (`"algorithm2_shared_table"`)
- `layers` (`int32[]`)
- `n_layer_total`, `n_embd`, `n_expert`, `n_topk` (`int32` scalars)
- `replaced_expert_mask` (`bool[n_layer_total, n_expert]`)
- `metadata_json` (JSON string with build parameters)
- `scaling_mode`, `residual_source` (string scalars)

Per selected layer `L`:

- `layer_<L>_centroids` (`float16[n_keys_L, n_embd]`)
- `layer_<L>_residuals` (`float16[n_keys_L, n_embd]`)
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
