# Brainstorm Plan: SVD Compressibility of MoE Expert Weights

Date: 2026-04-17

## 1) Brainstorm objective

Create a standalone Python analysis script (outside llama.cpp core) that:

1. Ensures the target MoE GGUF model is available via llama.cpp-compatible HF download flow.
2. Inspects model architecture metadata to identify expert weight matrices.
3. Runs SVD per expert matrix to estimate intrinsic dimensionality via participation ratio.
4. Reconstructs each matrix at low rank with `r = max(1, round(0.05 * min(m, n)))`.
5. Computes cosine similarity between original and low-rank reconstructed matrices.
6. Reports per-expert results and aggregate statistics (CLI + JSON).

Target model (first iteration):

- `unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M`

## 2) Confirmed decisions from user

- Scope: **single MoE model pilot**
- Source: **Hugging Face via llama.cpp workflow**
- Priority: **maximum fidelity**
- Outputs: **CLI summary + JSON artifact**
- Rank policy: **per-matrix 5% of `min(m,n)`**

## 3) Constraints and implications

- This remains a standalone analysis script (not integrated into llama.cpp production paths).
- Shared model files with llama.cpp are required (reuse the same local cache/storage layout).
- Quantized GGUF (`Q4_K_M`) may reduce SVD fidelity relative to FP16/BF16 checkpoints; this is acceptable for pilot but should be documented in output metadata.
- Full-fidelity SVD on large matrices can be expensive; for this pilot we favor exactness over runtime.

## 4) Proposed technical flow

### Stage A — Model acquisition and local reuse

1. Accept model spec in `repo:filename` form.
2. Check whether the GGUF already exists in llama.cpp-expected local location.
3. If missing, invoke llama.cpp-compatible HF download path to fetch it.
4. Resolve absolute model file path and log it in results metadata.

### Stage B — Tensor and architecture discovery

1. Parse GGUF metadata and tensor index.
2. Detect MoE expert tensors by name patterns and shape signatures.
3. Group tensors by layer/expert and matrix role.
4. Restrict analysis to 2D expert matrices for direct SVD.
5. Emit a discovery summary before heavy compute starts.

### Stage C — Per-matrix SVD analysis

For each discovered expert matrix `W`:

1. Convert/dequantize to dense floating representation (highest practical precision).
2. Compute full SVD: `W = U S V^T`.
3. Compute participation ratio using singular values:
   - `PR = (sum_i s_i^2)^2 / sum_i s_i^4`
4. Set low rank:
   - `r = max(1, round(0.05 * min(m,n)))`
5. Build rank-`r` reconstruction:
   - `W_r = U[:, :r] @ diag(S[:r]) @ V[:r, :]`
6. Compute cosine similarity between flattened matrices:
   - `cos = <vec(W), vec(W_r)> / (||W|| ||W_r||)`
7. Store matrix-level metrics and metadata (layer, expert id, tensor name, shape, rank).

### Stage D — Aggregate statistics and reporting

Compute summary stats across all analyzed expert matrices:

- Participation ratio: mean, median, std, min, max, p10/p25/p75/p90
- Cosine similarity: mean, median, std, min, max, p10/p25/p75/p90
- Counts: total tensors discovered, analyzed, skipped (+ reasons)

Outputs:

1. CLI summary table (human-readable)
2. JSON report (full per-matrix + aggregate + run metadata)

## 5) Script interface sketch

Potential CLI:

```bash
python analyze_moe_svd.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --out-json "results/qwen35-a3b-q4km-svd.json" \
  --rank-frac 0.05 \
  --full-svd
```

Suggested optional flags:

- `--cache-dir` (optional explicit cache location)
- `--include-pattern` / `--exclude-pattern` for tensor filtering
- `--max-matrices` (debug-only)
- `--dtype` for dequantized compute precision (`float32` default)

## 6) JSON schema sketch

```json
{
  "run": {
    "timestamp": "...",
    "model_spec": "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
    "resolved_path": "...",
    "rank_fraction": 0.05,
    "fidelity_mode": "full_svd"
  },
  "discovery": {
    "total_tensors": 0,
    "expert_candidate_tensors": 0,
    "analyzed_matrices": 0,
    "skipped": [{"name": "...", "reason": "..."}]
  },
  "per_matrix": [
    {
      "tensor": "...",
      "layer": 0,
      "expert": 0,
      "shape": [0, 0],
      "rank_used": 0,
      "participation_ratio": 0.0,
      "cosine_similarity_lowrank": 0.0
    }
  ],
  "summary": {
    "participation_ratio": {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0},
    "cosine_similarity": {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
  }
}
```

## 7) Risks to track

1. **Tensor naming variability** across GGUF exports may require robust pattern matching.
2. **Memory pressure** for full SVD of large expert matrices.
3. **Quantization effects** can bias intrinsic dimensionality estimates.
4. **MoE structure differences** across future models may need model-specific adapters.

## 8) Next-step implementation plan

1. Build model resolution/download helper that reuses llama.cpp-compatible local files.
2. Implement GGUF tensor discovery and expert matrix grouping.
3. Implement SVD + PR + low-rank cosine loop.
4. Add aggregate statistics utility.
5. Add CLI + JSON writer + concise terminal summary.
6. Run pilot on `unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M` and validate outputs.
