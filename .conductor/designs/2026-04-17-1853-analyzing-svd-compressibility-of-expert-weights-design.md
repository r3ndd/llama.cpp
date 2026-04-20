# Technical Design: Standalone MoE Expert SVD Compressibility Analyzer

Date: 2026-04-17  
Status: Proposed (implementation-ready)  
Scope: Standalone analysis script outside llama.cpp core

## 1. Context and objectives

This design defines a **standalone Python analysis script** that evaluates SVD compressibility of MoE expert weight matrices in GGUF models, starting with:

- `unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M`

The script must:

1. Resolve/download model files using a **llama.cpp-compatible Hugging Face workflow and cache layout**.
2. Discover MoE expert matrices from GGUF metadata/tensor names.
3. For each eligible 2D expert matrix, compute:
   - full SVD,
   - participation ratio,
   - low-rank reconstruction cosine similarity at `r = max(1, round(0.05 * min(m, n)))`.
4. Emit both:
   - human-readable CLI summary,
   - machine-readable JSON report (per-matrix + aggregate + metadata).

Non-goal: integrating functionality into llama.cpp runtime paths.

---

## 2. Key assumptions and constraints

1. **Pilot model format** is GGUF and may be quantized (`Q4_K_M`), requiring dequantization prior to numeric analysis.
2. **Fidelity-first**: use full SVD (exact algorithm in NumPy/SciPy backend), even with high runtime/memory cost.
3. **Standalone script** should avoid touching llama.cpp core C/C++ code; reuse existing download/cache conventions.
4. **Tensor naming may vary** across MoE families; discovery must be pattern-driven with clear logging of skipped tensors.
5. **Memory limits** may be hit on very large tensors; design includes predictable failure handling and optional limits.

---

## 3. Architecture overview

## 3.1 Module boundaries

Proposed files (all standalone tooling area, e.g., `scripts/` or `.conductor/tools/`):

1. `analyze_moe_svd.py` (entrypoint CLI)
2. `moe_svd/model_resolver.py`
3. `moe_svd/gguf_discovery.py`
4. `moe_svd/svd_metrics.py`
5. `moe_svd/stats.py`
6. `moe_svd/reporting.py`
7. `moe_svd/types.py`

Design intent:

- Keep parsing/discovery independent from linear algebra.
- Keep report schema centralized in typed structures for future expansion.
- Make model-specific discovery extensible without rewriting compute code.

## 3.2 End-to-end data flow

1. Parse CLI args + validate values.
2. Resolve `repo:filename` model spec to local GGUF path:
   - if exists in cache, reuse;
   - if missing, download via llama.cpp-compatible HF workflow.
3. Load GGUF metadata + tensor index.
4. Discover candidate MoE expert matrices (2D only), apply include/exclude filters.
5. For each matrix (sequentially):
   - materialize dense float matrix,
   - compute full SVD,
   - compute PR,
   - compute rank `r` and low-rank reconstruction,
   - compute cosine similarity,
   - append per-matrix record.
6. Aggregate distribution stats.
7. Print concise CLI summary.
8. Write JSON artifact atomically.

---

## 4. CLI interface specification

```bash
python analyze_moe_svd.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --out-json "/path/to/results.json" \
  --rank-frac 0.05 \
  --dtype float32 \
  --full-svd
```

Required args:

- `--model <repo:filename>`
- `--out-json <path>`

Optional args:

- `--cache-dir <path>`: explicit cache root override.
- `--rank-frac <float>`: default `0.05`, must satisfy `(0, 1]`.
- `--dtype <float32|float64>`: dequantized analysis dtype (default `float32`).
- `--include-pattern <regex>` (repeatable).
- `--exclude-pattern <regex>` (repeatable).
- `--max-matrices <int>`: debug throttle.
- `--fail-fast` (bool): stop on first tensor analysis failure.
- `--quiet` (bool): minimal terminal output.

Exit codes:

- `0`: success with JSON emitted.
- `2`: CLI usage/validation error.
- `3`: model resolution/download error.
- `4`: GGUF parsing/discovery error.
- `5`: analysis failed for all tensors.
- `6`: output write error.

---

## 5. Module/interface design

## 5.1 `model_resolver.py`

Responsibility:

- Parse model spec (`repo:filename`).
- Resolve local path using llama.cpp-compatible cache conventions.
- Download missing file using llama.cpp-compatible HF path.

Interface:

- `resolve_model_path(model_spec: str, cache_dir: str | None) -> ResolvedModel`

`ResolvedModel` fields:

- `model_spec: str`
- `repo_id: str`
- `filename: str`
- `local_path: str` (absolute)
- `downloaded: bool`
- `cache_dir_used: str`

Decision:

- Prefer reusing existing llama.cpp scripts/commands for HF GGUF fetch (invocation wrapper) rather than introducing a custom download stack.

Tradeoff:

- Pros: exact cache compatibility.
- Cons: tighter coupling to local llama.cpp tooling assumptions.

## 5.2 `gguf_discovery.py`

Responsibility:

- Open GGUF, read metadata/tensor descriptors.
- Identify MoE expert candidate tensors.
- Return analyzable 2D matrices plus skipped entries with reasons.

Interface:

- `discover_expert_matrices(gguf_path: str, include: list[str], exclude: list[str]) -> DiscoveryResult`

`DiscoveryResult`:

- `total_tensors: int`
- `candidates: list[MatrixRef]`
- `skipped: list[SkippedTensor]`
- `metadata: dict[str, Any]` (model architecture summary)

`MatrixRef`:

- `tensor_name: str`
- `shape: tuple[int, int]`
- `layer: int | None`
- `expert: int | None`
- `role: str | None` (e.g., `w1`, `w2`, `w3`, unknown)

Detection strategy (incremental):

1. Primary regex over tensor names for known MoE patterns (`expert`, `experts`, layer/expert indices).
2. Secondary shape gate: 2D tensors only.
3. Optional model-family adapter map (future): pattern bundles keyed by architecture metadata.

Failure modes:

- GGUF unreadable/corrupt -> hard error.
- No expert matrices found -> soft failure with diagnostics, exits code `4`.

## 5.3 `svd_metrics.py`

Responsibility:

- For one matrix, compute SVD-based metrics.

Interface:

- `analyze_matrix(matrix: np.ndarray, rank_frac: float) -> MatrixMetrics`

`MatrixMetrics` fields:

- `m: int`, `n: int`
- `rank_used: int`
- `singular_value_count: int`
- `participation_ratio: float`
- `cosine_similarity_lowrank: float`
- `fro_norm: float`
- `analysis_warnings: list[str]`

Algorithms:

1. Full SVD:
   - `U, S, VT = np.linalg.svd(W, full_matrices=False)`
2. Participation ratio:
   - `PR = (sum(S**2)**2) / sum(S**4)`
   - numerical guard: if denominator is zero (all-zero tensor), set `PR = 0.0` and warning.
3. Rank:
   - `r = max(1, round(rank_frac * min(m, n)))`
4. Low-rank reconstruction:
   - `Wr = (U[:, :r] * S[:r]) @ VT[:r, :]`
5. Cosine similarity (flattened):
   - `cos = dot(W.ravel(), Wr.ravel()) / (||W|| * ||Wr||)`
   - guard zero norms -> `cos = 0.0` with warning.

Decision:

- Use exact SVD, no randomized/truncated approximations for pilot.

## 5.4 `stats.py`

Responsibility:

- Aggregate per-matrix metrics into summary distributions.

Interface:

- `compute_summary(per_matrix: list[PerMatrixRecord]) -> SummaryStats`

Required summary fields:

- For `participation_ratio` and `cosine_similarity_lowrank`:
  - mean, median, std, min, max, p10, p25, p75, p90.
- Counts:
  - total tensors,
  - candidates,
  - analyzed,
  - skipped by reason,
  - failed by reason.

## 5.5 `reporting.py`

Responsibility:

- Render CLI summary.
- Serialize JSON with versioned schema.
- Atomic write to output path.

Interface:

- `print_cli_summary(report: Report, quiet: bool) -> None`
- `write_json_report(report: Report, out_path: str) -> None`

JSON should include:

- `schema_version` (e.g., `1.0`)
- `run` metadata (timestamp, script version/commit hash if available, model spec/path, dtype, rank frac)
- `discovery`
- `per_matrix`
- `summary`
- `assumptions_and_caveats` (explicit quantization caveat)

---

## 6. Algorithm definitions (normative)

For each analyzed matrix `W ∈ R^(m×n)`:

1. `W` is converted to dense `dtype` (`float32` default) before decomposition.
2. Full SVD (`full_matrices=False`) yields singular values `s_i`.
3. Participation ratio:
   - `PR(W) = (Σ s_i^2)^2 / (Σ s_i^4)`.
4. Rank policy:
   - `r = max(1, round(rank_frac * min(m,n)))`.
5. Low-rank approximation:
   - `W_r = U_r diag(S_r) V_r^T`.
6. Similarity:
   - cosine between flattened `W` and `W_r`.

Numerical stability rules:

- If matrix contains non-finite values after dequantization, mark failure and skip.
- Use `np.isfinite` checks on outputs.
- Clamp final cosine to `[-1, 1]` for tiny rounding excursions.

---

## 7. Performance and memory strategy

Given fidelity-first constraint, primary strategy is controlled sequential processing:

1. **One matrix at a time** to bound peak memory.
2. Release intermediate arrays (`U`, `S`, `VT`, `Wr`) before moving to next tensor.
3. Optional `--max-matrices` for smoke tests.
4. Log elapsed time per tensor and total runtime.

Memory expectations:

- Full SVD and reconstruction can require multiple matrix-sized buffers; practical peak can exceed ~4–8x matrix bytes depending on LAPACK backend.

If OOM occurs:

- capture and record failure reason (`MemoryError`),
- continue to next tensor unless `--fail-fast`.

---

## 8. Error handling and failure modes

Hard-fail conditions (terminate run):

1. Invalid CLI/model spec.
2. Cannot resolve/download GGUF.
3. GGUF file cannot be parsed.
4. JSON cannot be written.

Soft-fail conditions (record and continue):

1. Tensor is non-2D.
2. Tensor not matched as expert matrix.
3. Dequantization failure for one tensor.
4. SVD numerical failure for one tensor.
5. MemoryError for one tensor.

If all candidate tensors fail analysis:

- exit code `5`, still attempt writing JSON with failure diagnostics.

---

## 9. Extensibility for additional MoE models

Design extension points:

1. **Discovery adapters**:
   - `DiscoveryAdapter` interface keyed by GGUF architecture metadata.
   - Default regex adapter + model-specific adapters as needed.
2. **Metric registry**:
   - future metrics (e.g., explained variance at fixed rank, nuclear norm ratios) can be added without changing discovery logic.
3. **Output schema versioning**:
   - additive fields under `summary`/`per_matrix` with `schema_version` bump only for breaking changes.

Why this approach:

- Keeps pilot implementation simple while preventing hard-coded dependence on one naming convention.

---

## 10. Validation plan

## 10.1 Unit-level validation

1. **Model spec parser tests**:
   - valid/invalid `repo:filename` cases.
2. **Rank policy tests**:
   - edge shapes and rounding behavior.
3. **Metric math tests** using synthetic matrices:
   - diagonal matrix with known singular values -> exact PR expected.
   - low-rank synthetic matrix -> cosine near 1.0 at sufficient `r`.
   - zero matrix -> PR/cosine guards.
4. **Summary stats tests**:
   - quantile correctness and empty-list handling.

## 10.2 Integration validation

1. Dry run on pilot model with `--max-matrices 3` to verify discovery and report schema.
2. Full run on pilot model; verify:
   - non-empty `per_matrix`,
   - aggregate stats present,
   - skipped reasons populated sensibly.
3. Re-run with same inputs; ensure deterministic matrix ordering and stable outputs (allowing tiny numeric tolerance).

## 10.3 Manual sanity checks

1. Confirm all analyzed tensors are plausible expert matrices by name/layer/expert structure.
2. Spot-check a handful of matrices for `rank_used == round(0.05*min_dim)`.
3. Verify CLI summary matches JSON aggregates.

---

## 11. Rollout strategy (incremental)

Phase 1 (pilot-ready):

- Implement default regex-based discovery + full metric pipeline + JSON/CLI output.

Phase 2 (robustness):

- Add architecture-specific discovery adapters for additional MoE families.
- Add richer skipped/failure taxonomy.

Phase 3 (scalability, optional):

- Optional approximate SVD mode behind explicit flag (not default), preserving full-SVD baseline for comparability.

---

## 12. Risks and mitigations

1. **Quantization bias (Q4_K_M)**
   - Risk: singular spectrum and PR differ from FP/BF checkpoints.
   - Mitigation: include explicit caveat in JSON + CLI; capture quantization/type metadata.

2. **Naming variability across GGUF exports**
   - Risk: missed expert tensors or false positives.
   - Mitigation: record skipped/matched traces; adapter-based discovery extension.

3. **High memory/time cost of full SVD**
   - Risk: long runs or per-tensor failure.
   - Mitigation: sequential processing, fail-soft behavior, debug throttle.

4. **Tooling coupling for download path**
   - Risk: local script path differences across environments.
   - Mitigation: resolver supports explicit `--cache-dir`; clear error messages with remediation steps.

---

## 13. Concrete implementation checklist

1. Create script/module scaffold and typed report structures.
2. Implement model resolution/downloading wrapper with llama.cpp-compatible cache behavior.
3. Implement GGUF loader + discovery logic + reasoned skip tracking.
4. Implement SVD metric engine with numeric guards.
5. Implement aggregate statistics and deterministic ordering.
6. Implement CLI rendering and atomic JSON writer.
7. Add tests for parser, math, stats, and small integration fixture.
8. Execute pilot run on `unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M` and archive JSON artifact.

---

## 14. Primary design decisions summary

1. **Exact full SVD is the default and only pilot mode** to maximize fidelity.
2. **Sequential matrix processing** is chosen over parallelism to control memory pressure.
3. **Discovery is pattern-based with structured skip/failure accounting** for transparency.
4. **JSON schema is first-class, versioned, and reproducibility-oriented** (run metadata + caveats).
5. **Download/cache reuse follows llama.cpp-compatible HF workflow** rather than bespoke model fetching.
