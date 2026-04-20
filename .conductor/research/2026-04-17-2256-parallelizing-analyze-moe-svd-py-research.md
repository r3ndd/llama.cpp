# Research Brief: Parallelizing `scripts/analyze_moe_svd.py` for Maximum Throughput

Date: 2026-04-17
Scope: standalone script performance (not llama.cpp runtime)

## Context

Goal: speed up `scripts/analyze_moe_svd.py` as much as practical via CPU/GPU parallelization while preserving report correctness.

Primary code paths reviewed:
- Entrypoint loop: `scripts/analyze_moe_svd.py:257-310`
- Matrix load/dequantize: `scripts/moe_svd/gguf_discovery.py:293-377`
- SVD metrics: `scripts/moe_svd/svd_metrics.py:12-59`
- Summary/reporting: `scripts/moe_svd/stats.py:41-66`, `scripts/moe_svd/reporting.py:21-60`

## Evidence (observed facts)

### 1) Current pipeline is strictly sequential
- `main()` processes candidates in a single `for` loop (`scripts/analyze_moe_svd.py:257-310`) with no concurrency controls.
- Caveat text explicitly states sequential execution (`scripts/analyze_moe_svd.py:180-181`).

**Implication:** all candidate matrices are independent work items but are currently serialized.

---

### 2) Dominant compute cost is SVD, with avoidable extra work
- `analyze_matrix()` calls `np.linalg.svd(matrix, full_matrices=False)` and computes `u, s, vt` (`scripts/moe_svd/svd_metrics.py:24`).
- Only `s` is used for current metrics (`participation_ratio`, `explained_spectral_energy_rank_r`, and `fro_norm`) (`scripts/moe_svd/svd_metrics.py:26-43`). `u` and `vt` are not consumed.

Microbenchmarks run in this environment:
- Command: `python -c "... np.linalg.svd(A, full_matrices=False) ... compute_uv=False ..."`
- Result on 2048x2048 FP32: `svd_full=4.000s`, `svdvals=1.304s`, speedup `3.07x`.
- End-to-end metric path benchmark (current vs singular-values-only formulation): `2.027s -> 1.131s` (`1.79x`), with near-identical metric output.

**Implication:** largest immediate win is switching to singular-values-only path before any multiprocessing/GPU work.

---

### 3) Dequantization path is CPU and per-source-tensor cached only within process
- Dequantization uses `gguf.dequantize(tensor.data, tensor.tensor_type)` (`scripts/moe_svd/gguf_discovery.py:322-328`).
- Cache is one-entry (`cache["source_tensor_name"]`) and local to the process/run (`scripts/moe_svd/gguf_discovery.py:312-334`, `scripts/analyze_moe_svd.py:241`).
- GGUF reader uses memory mapping (`gguf-py/gguf/gguf_reader.py:132-133`), so opening in each process avoids duplicating full file into RAM.

**Implication:** work partitioning should preserve locality by `source_tensor_name` so one dequantized 3D packed tensor can serve many expert slices per worker.

---

### 4) BLAS backend and thread behavior matter
- `np.show_config()` reports OpenBLAS backend (`openblas64`) with OpenMP support and `MAX_THREADS=2` in this environment.
- Benchmark with `OPENBLAS_NUM_THREADS=1` vs `2` for 1536x1536 `compute_uv=False`: `1.258s -> 0.849s` (`1.48x`).

**Implication:** if we add process-level parallelism without controlling BLAS threads, oversubscription risk is high; thread/process co-tuning is mandatory.

---

### 5) Discovery/reporting overhead is minor relative to SVD
- Discovery does regex/pattern/filtering over tensor metadata (`scripts/moe_svd/gguf_discovery.py:168-238`), no heavy math.
- Reporting is JSON serialization and summary printing (`scripts/moe_svd/reporting.py:51-60`).

**Implication:** optimize dequant+SVD path first; reporting parallelization is not impactful.

## External API references (authoritative)

- NumPy SVD docs: `numpy.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)` supports `compute_uv=False`; decomposition uses LAPACK `_gesdd`; stacked mode supported for higher-rank inputs.
  Source: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
- PyTorch SVD docs: `torch.linalg.svd` supports batched inputs and CUDA `driver` choices (`gesvdj`, `gesvda`, `gesvd`), default fallback behavior documented.
  Source: https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html
- CuPy SVD docs: `cupy.linalg.svd` supports `compute_uv=False`; on CUDA, algorithm path differs by shape/batch (`gesvdj` fast path for some small batched cases, else `gesvd`).
  Source: https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.svd.html
- SciPy partial SVD docs: `scipy.sparse.linalg.svds` computes only top/bottom-k singular values (truncated/approximate use cases).
  Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html
- sklearn randomized SVD docs: `randomized_svd` is approximate truncated SVD with tunable quality/perf tradeoff.
  Source: https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html

## Bottleneck analysis by stage

### A) Discovery
- **Observed:** mostly metadata parsing + regex; O(number of tensors), low arithmetic intensity.
- **Parallelization value:** low.
- **Recommendation:** keep single-threaded; measure but do not optimize first.

### B) Dequantization
- **Observed:** every source tensor must be dequantized to dense float; expensive for large packed-expert tensors.
- **Parallelization value:** medium-high with source-aware partitioning.
- **Risks:** duplicated dequant work if experts from same packed tensor are processed by different workers.

### C) SVD + metric computation
- **Observed:** major hotspot; currently computes full vectors unnecessarily.
- **Parallelization value:** highest.
- **Risks:** CPU oversubscription; GPU transfer overhead; numerical drift if mixed precision.

### D) Reporting
- **Observed:** negligible in comparison.
- **Parallelization value:** low.

## Recommended CPU strategy (realistic for this repo)

### Phase 0 (no parallelism yet, highest ROI)
1. Replace `np.linalg.svd(..., full_matrices=False)` with singular-values-only path:
   - `s = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)` (or `np.linalg.svdvals` when available).
2. Compute `fro_norm` from singular values: `sqrt(sum(s^2))`.
3. Keep metric formulas unchanged.

**Expected gain:** ~1.7x–3.0x in SVD section (observed locally); low complexity/risk.

---

### Phase 1 (single-host CPU parallelism)
1. Add `--workers` (default 1). Use `ProcessPoolExecutor` or `multiprocessing.Pool` (processes, not threads).
2. Partition tasks by `source_tensor_name` (key from `MatrixRef.source_tensor_name`):
   - each worker gets groups of matrices sharing source tensor to maximize reuse of per-process dequant cache.
3. In worker init, set BLAS env vars before NumPy import if possible:
   - `OPENBLAS_NUM_THREADS=1` (or configurable), `OMP_NUM_THREADS=1`.
4. Main process merges per-worker records; keep deterministic global sort (already done in main).

**Expected gain:**
- If CPU has cores available: near-linear until memory bandwidth / BLAS saturation.
- Practical range: `1.5x–4x` on mainstream 8–16 core hosts when combined with Phase 0.

**Complexity:** medium.
**Risk:** medium (oversubscription, worker startup cost, exception plumbing).

---

### Phase 2 (affinity and batching tuning)
1. Add optional `--blas-threads` and enforce per-process BLAS thread cap.
2. Add chunked scheduling by matrix size bucket (largest first) to reduce tail latency.
3. Optional CPU affinity pinning (Linux): pin worker processes to distinct cores/sockets.

**Expected gain:** incremental (`10–30%` beyond Phase 1 on some hosts).
**Complexity:** medium-high.
**Risk:** platform-dependent behavior.

## Recommended GPU strategy (feasible paths)

### Path G1: PyTorch CUDA backend (preferred if GPU path is added)
Implementation idea:
1. Add optional `--backend numpy|torch|cupy` and `--device cpu|cuda` flags.
2. Dequantize on CPU (existing gguf path), then transfer matrix to GPU.
3. Use `torch.linalg.svdvals` (or `torch.linalg.svd` only when vectors needed).
4. Compute metrics on GPU and transfer only scalar outputs to host.

Why PyTorch first:
- Strong docs and mature CUDA driver selection (`gesvdj/gesvd/gesvda`) for tradeoff control.
- Batched input support can reduce kernel-launch overhead when grouping same-shape matrices.

Constraints:
- Host->device copy may dominate for smaller matrices.
- `gguf.dequantize` remains CPU-side; no direct quantized GGUF decode on GPU in current code.

Expected gain:
- Large matrices + sufficient VRAM: potential `2x–6x` versus single-process CPU.
- Small matrices or transfer-bound workloads: low/no gain.

---

### Path G2: CuPy CUDA backend
- Similar flow to PyTorch, minimal API changes if using NumPy-like code.
- `cupy.linalg.svd` supports `compute_uv=False` and batched input semantics.
- Docs warn about cuSOLVER invalid-result conditions; must enable error checking (`cupyx.errstate`).

Expected gain: similar class to G1 when transfer is amortized.
Risk: medium (error handling and ecosystem familiarity).

---

### Path G3: JAX (least practical here)
- Possible but higher integration overhead for a standalone utility and fewer direct benefits vs PyTorch/CuPy.

## Approximation options (optional, non-default)

For “as much as possible” speed, add explicit approximation mode later (not default):
- `scipy.sparse.linalg.svds` or `sklearn.utils.extmath.randomized_svd` for top-k metrics only.
- This changes fidelity semantics and should require opt-in flag + clear caveat in JSON.

**Expected gain:** can be very large when only low-rank spectra are needed.
**Risk:** metric drift; reduced comparability with current full-spectrum analysis.

## Proposed phased implementation plan

| Phase | Scope | Complexity | Risk | Expected Speed Impact |
|---|---|---:|---:|---:|
| P0 | Singular-values-only (`compute_uv=False`) + `fro_norm` from `s` | Low | Low | High (1.7x–3x in hotspot) |
| P1 | Process pool + source-tensor partition + BLAS thread capping | Medium | Medium | High (additional 1.5x–4x host-dependent) |
| P2 | Scheduling/affinity/size bucketing tuning | Med-High | Medium | Medium (10–30%) |
| P3 | Optional GPU backend (PyTorch first, then CuPy) | Medium | Medium | High on large matrices (2x–6x vs baseline) |
| P4 | Optional approximate/truncated mode | Medium | High (fidelity) | Very high (workload-dependent) |

## Instrumentation and benchmark recommendations

Add timing breakdown in JSON (new fields under `run` or `summary`):
- `time_discovery_s`
- `time_load_dequant_s_total`
- `time_svd_s_total`
- `time_reporting_s`
- per-matrix: keep existing `elapsed_seconds` and add `load_s`, `svd_s`.

Benchmark matrix:
1. Baseline current script (workers=1, full vectors).
2. P0 only (workers=1, svdvals path).
3. P0+P1 grid search:
   - workers in `{1,2,4,8}`
   - BLAS threads in `{1,2,4}` (if host supports)
4. GPU path:
   - compare total wall time and isolated SVD time with transfer accounted.

Suggested command templates:
- `OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/llama.cpp/scripts python /root/llama.cpp/scripts/analyze_moe_svd.py ... --full-svd`
- Repeat with `--workers N` once added.

Acceptance criteria:
1. **Correctness:** summary metrics and per-matrix metrics match baseline within tolerance (`abs <= 1e-6` float32 path) for non-approx mode.
2. **Stability:** no increase in per-matrix failure rate; deterministic sorted output order preserved.
3. **Performance:**
   - P0: >=30% total wall-time reduction on reference model.
   - P0+P1: >=2x total speedup on >=8-core host.
   - GPU path (if enabled): >=1.5x end-to-end speedup on large-matrix workloads, else auto-fallback to CPU.

## Open questions / assumptions to validate

1. **Observed constraint:** current env OpenBLAS advertises low max threads (`MAX_THREADS=2`) — host-dependent; re-measure on target deployment machines.
2. **Hypothesis:** source-tensor grouped scheduling will materially reduce redundant dequant in multi-process mode; needs profiling on real packed-expert models.
3. **Hypothesis:** GPU path wins only when matrix sizes are sufficiently large and grouped; requires transfer-vs-compute crossover measurement.
4. **Decision needed:** whether to keep `--full-svd` naming if implementation moves to singular-values-only but still mathematically exact for current metrics.

## Recommended next steps (short)

1. Implement P0 first (`compute_uv=False`, `fro_norm` from singular values) and benchmark.
2. Add P1 process pool with `--workers` + BLAS thread controls and source-tensor partitioning.
3. Prototype PyTorch CUDA backend behind optional flag; ship only if end-to-end wins on reference model.
4. Add instrumentation fields and lock in acceptance thresholds before further tuning.
