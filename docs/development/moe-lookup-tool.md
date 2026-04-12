# MoE Lookup Table Builder (Algorithm 2 PoC)

This document describes the offline tool for building per-layer shared lookup/contribution tables from MoE trace NPZ v1 artifacts.

- Scripts:
  - `scripts/build-moe-lookup.py` (trace -> lookup NPZ + replaced JSON)
  - `scripts/convert-moe-lookup-to-elt1.py` (lookup NPZ + replaced JSON -> runtime ELT1 binary)
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

Example (build lookup tables):

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

Example (convert builder output to ELT1 runtime sidecar):

```bash
python3 scripts/convert-moe-lookup-to-elt1.py \
  --input out/qwen35moe.lookup.npz \
  --replaced-experts out/qwen35moe.replaced-experts.json \
  --output out/qwen35moe.lookup.elt1 \
  --model-id qwen3moe
```

Example (plot-only heuristic mode):

```bash
python3 scripts/build-moe-lookup.py \
  --input trace-run-a.npz \
  --input trace-run-b.npz \
  --plot-heuristic \
  --plot-output out/qwen35moe.heuristic.png
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
- `--plot-heuristic`: plot-only mode; computes routing-usage heuristic and writes a histogram instead of generating lookup/replaced-expert artifacts
- `--plot-output`: image output path for `--plot-heuristic` (defaults to `<output>.heuristic.png` when `--output` is set, else `./moe-heuristic.png`)

## Heuristic plot mode

When `--plot-heuristic` is enabled, the script:

1. Loads and validates traces as usual.
2. Computes per-layer per-expert heuristic scores (currently usage counts from `topk_ids`).
3. Builds one histogram over all selected layer/expert scores with a log-scale x-axis.
4. Draws percentile markers at 10%, 20%, …, 90% as vertical lines.
5. Saves the plot image and exits successfully.

In this mode, lookup sidecar and replaced-experts JSON outputs are not produced.
The plotting path uses a non-interactive backend (`Agg`) for headless environments.
Zero/near-zero heuristic scores are clamped to a small positive floor for plotting so log scaling is safe.

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

### 3) Runtime ELT1 binary sidecar (from converter)

`convert-moe-lookup-to-elt1.py` emits the runtime binary sidecar schema loaded by `src/moe-lookup.cpp`:

- Header (`llama_moe_lookup_header_v1`):
  - `magic = "ELT1"`
  - `format_version = 1`
  - `model_id`
  - `n_layer`, `n_embd`, `n_expert`, `n_expert_used`
  - `vector_dtype = fp16`
  - `scaling_mode = s_missing`
  - `n_layers_payload`
- Per-layer payload (`llama_moe_lookup_layer_header_v1` + raw arrays):
  - `layer_id`, `n_keys`, `replaced_count`
  - `centroids` (`fp16[n_keys, n_embd]`)
  - `contributions` (`fp16[n_keys, n_embd]`)
  - `replaced_ids` (`u32[replaced_count]`)

The converter performs strict validation to match runtime loader expectations (shape/type/scaling/layer/expert constraints).

## Canonical end-to-end flow

1. Generate trace NPZ with `--moe-trace-enable`.
2. Build lookup NPZ + replaced JSON:

   ```bash
   python3 scripts/build-moe-lookup.py ...
   ```

3. Convert to runtime ELT1 binary:

   ```bash
   python3 scripts/convert-moe-lookup-to-elt1.py \
     --input out/qwen35moe.lookup.npz \
     --replaced-experts out/qwen35moe.replaced-experts.json \
     --output out/qwen35moe.lookup.elt1 \
     --model-id qwen3moe
   ```

4. Run inference with lookup enabled:

   ```bash
   ./bin/llama-cli \
     -m /path/to/qwen35moe.gguf \
     --moe-lookup-enable \
     --moe-lookup-file out/qwen35moe.lookup.elt1 \
     --moe-lookup-replaced-experts out/qwen35moe.replaced-experts.json \
     -p "Hello"
   ```

Note: `--moe-lookup-replaced-experts` remains required by current runtime config validation even though ELT1 payload includes per-layer replaced IDs.

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
