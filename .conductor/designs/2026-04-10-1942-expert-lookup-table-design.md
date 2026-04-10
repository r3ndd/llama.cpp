# Expert Lookup Table (ELT) Design — Qwen3.5-MoE, Algorithm 2 PoC

Date: 2026-04-10 19:42
Status: Proposed implementation design (Phase 1 PoC)
Scope: `Qwen/Qwen3.5-35B-A3B` in llama.cpp runtime, mixed CPU+GPU offload, **Algorithm 2 only**

---

## 1) Executive Summary

This design introduces an **experimental, default-off Expert Lookup Table (ELT)** path for Qwen3.5-MoE in llama.cpp. The ELT path replaces selected routed experts with a **shared per-layer residual lookup** (Algorithm 2), using precomputed sidecar artifacts derived from trace data.

Phase 1 is intentionally narrow:
- Qwen3.5-MoE only.
- Shared table per layer (Algorithm 2).
- Sidecar artifacts external to GGUF.
- FP16 residual storage.
- NPZ traces for offline training/build pipeline.

Primary acceptance gate: **<= 2.0 perplexity delta** on a fixed WikiText-103 slice relative to baseline under matched runtime settings.

---

## 2) Explicit Decisions Already Made (Locked for v1)

1. **Algorithm choice:** Implement **Algorithm 2 (shared table per layer)** first.
2. **Model scope:** Stay **Qwen3.5-MoE only** until stability criteria are met.
3. **Trace format v1:** Use **NPZ per run** for trace output.
4. **Runtime config surface:** Use **CLI flags + file paths** (no env/config metadata dependency in v1).
5. **Residual representation:** Store residual vectors as **FP16** in v1.
6. **Packaging:** Keep lookup data in **sidecar artifact(s)**, not GGUF, for PoC iteration speed.

These decisions are treated as non-goals for this phase; alternatives are deferred to follow-up phases.

---

## 3) Goals, Non-Goals, and Success Criteria

### Goals
- Enable end-to-end ELT experimentation for Qwen3.5-MoE on mixed CPU+GPU offload.
- Capture routing + hidden-state traces with exact routing correctness.
- Execute inference-time replacement of selected experts via layer shared residual lookups.
- Provide robust fallback to baseline MoE behavior when ELT assets are missing/invalid.

### Non-Goals (Phase 1)
- Multi-model support beyond Qwen3.5-MoE.
- GGUF embedding of lookup tables.
- INT8/Hadamard/binary residual compression.
- Algorithm 1 (per-expert unique tables).
- Autotuning replacement heuristics in runtime.

### Phase-1 Success Criteria
1. ELT pipeline runs end-to-end without crashes.
2. Replacement of at least 5–10% experts is functional.
3. At least one tested setting meets **<= 2.0 ppl delta** (WikiText-103 slice).
4. Trace parity checks show exact top-k IDs/weights correspondence with non-tracing path.
5. Mixed CPU+GPU runs show no regressions in correctness/stability.

---

## 4) Architecture Overview

## 4.1 High-Level Components

1. **Runtime Trace Capture (llama.cpp)**
   - Emits NPZ traces containing per-token, per-layer data needed for offline ELT training.

2. **Offline ELT Builder (external Python/tooling)**
   - Consumes NPZ traces, runs k-means over `h_pre_moe` per layer, computes residual table values, and emits runtime sidecar artifact.

3. **Runtime ELT Inference (llama.cpp)**
   - Loads sidecar + replacement policy.
   - During MoE FFN, computes baseline from non-replaced experts and adds looked-up residual correction when applicable.

4. **Evaluation Harness**
   - Baseline vs ELT on fixed corpus slice with matched settings; compares ppl, throughput, and memory.

## 4.2 Data Flow

1. Inference with trace mode:
   - For each MoE layer token pass, capture:
     - `h_pre_moe`
     - top-k IDs and weights
     - `y_full`
     - optionally `y_kept` or enough data to reconstruct residual offline
2. Save run trace to NPZ.
3. Offline builder:
   - Rank/select replaced experts per layer using chosen heuristic.
   - Build shared cluster key space per layer.
   - Compute residual target `r = y_full - y_kept`.
   - Aggregate `R_L[key] = mean(r)` and emit sidecar.
4. Inference with ELT enabled:
   - Determine routed experts.
   - Route only non-replaced experts (plus next-best non-replaced to maintain k).
   - Lookup key from `h_pre_moe`, fetch residual, apply scaling strategy, add to layer output.

---

## 5) Component Boundaries and Responsibilities

## 5.1 Runtime MoE Graph Integration

**Primary insertion:** `src/llama-graph.cpp` in `llm_graph_context::build_moe_ffn(...)`

Responsibilities:
- Preserve existing baseline routing and aggregation semantics.
- Inject optional trace hooks (capture-only, no math changes).
- Inject optional ELT substitution branch only when feature enabled and layer configured.
- Guarantee deterministic fallback to baseline path on any ELT precondition failure.

## 5.2 Model-Specific Feature Gating

**Initial scope gate:** `src/models/qwen35moe.cpp` in `build_layer_ffn(...)`

Responsibilities:
- Restrict ELT activation to Qwen3.5-MoE model family.
- Reject/ignore ELT flags for other models with explicit warning.

## 5.3 Runtime Configuration and CLI Parsing

Likely touchpoints:
- `common/arg.cpp` (or equivalent CLI argument registration)
- runtime parameter structs used by `llama-cli` / server path

Responsibilities:
- Parse ELT and trace flags.
- Validate paths.
- Materialize config object passed into context/model runtime.

## 5.4 Sidecar Loader and Validation

Likely new module(s):
- `src/moe_lookup.*` (or similarly isolated ELT runtime helper)

Responsibilities:
- Load sidecar header + per-layer tables.
- Validate shape compatibility with live model (`n_layer`, `n_embd`, router top-k assumptions where relevant).
- Expose immutable lookup interface to graph builder.

## 5.5 Trace Writer

Likely new module(s):
- `src/moe_trace.*` + optional helper in `common/` for buffering/flush.

Responsibilities:
- Accumulate trace rows with bounded memory.
- Flush to NPZ with metadata.
- Minimize overhead, and be disabled by default.

---

## 6) Runtime Configuration / Flags (v1)

Tentative flags (as planned):

- `--moe-lookup-enable`
- `--moe-lookup-file <path>`
- `--moe-lookup-replaced-experts <path>`
- `--moe-trace-enable`
- `--moe-trace-out <path>`

### Semantics
- All ELT behavior is **default-off**.
- If `--moe-lookup-enable` is set but assets are missing/invalid/incompatible:
  - Emit warning log with reason.
  - Continue using baseline MoE path (no hard failure in v1).
- Trace and lookup can be enabled independently.
  - Trace-only mode for data generation.
  - Lookup-only mode for inference experiments.

---

## 7) Trace Schema v1 (NPZ)

Per-run NPZ artifact containing flattened token-layer rows.

Required arrays (v1):
- `layer_ids` (`int16` or `int32`)
- `token_ids` (`int32`)
- `h_pre_moe` (`float16`) — shape `[N, n_embd]`
- `topk_ids` (`int16`/`int32`) — shape `[N, k]`
- `topk_weights` (`float16`) — shape `[N, k]`
- `y_full` (`float16`) — shape `[N, n_embd]`
- optional `y_kept` (`float16`) or precomputed `residual_target` (`float16`)
- optional `replaced_mask` (`bool`) for slice analysis

Metadata JSON entry (embedded or adjacent):
- `format_version: 1`
- `model_id`
- `commit_hash`
- `n_layer`, `n_embd`, `n_expert`, `n_expert_used`
- `routing_mode`
- `prompt_source`
- `trace_sampling_policy`

### Correctness Requirements
- top-k IDs/weights must match inference routing exactly for same token/layer event.
- token/layer ordering must be deterministic within a run artifact.

### Overhead Strategy
- Buffer rows in memory up to configurable bound, then flush.
- Optional sampling mode (token stride or probabilistic) for scale tests.
- Keep trace path isolated from core math path to reduce accidental perturbation.

---

## 8) Sidecar Format Strategy (Inference Artifact)

v1 direction: start with **versioned sidecar header** and layer payload, while retaining ability to generate from NPZ builder output.

## 8.1 Header (required)
- `format_version` (u32)
- `model_id` (string)
- `n_layer` (u32)
- `n_embd` (u32)
- `k_per_layer` or cluster count map
- `residual_dtype` (`fp16` in v1)
- `scaling_mode` enum

## 8.2 Per-layer payload
- `layer_id`
- key centroids: `C_L` shape `[n_keys_L, n_embd]` (`fp16` or `fp32` for distance stability; see unresolved items)
- residual table: `R_L` shape `[n_keys_L, n_embd]` (`fp16`)
- replaced expert set/list for layer (or external JSON if split retained in v1)

## 8.3 Strategy Choice for v1
- **Preferred:** compiled binary sidecar for inference loading speed + explicit header checks.
- **Interim compatible path:** allow NPZ direct load in very early bring-up if that reduces implementation lead time.

Rationale:
- Sidecar remains decoupled from GGUF.
- Binary loader simplifies runtime and reduces parse overhead.
- NPZ remains canonical trace/training format.

---

## 9) Inference Integration Path (Algorithm 2)

For each token at MoE layer `L`:

1. Compute router outputs and top-k candidates as baseline.
2. Split selected experts into:
   - `kept` (not replaced)
   - `replaced` (configured replaced set)
3. Fill compute set to maintain `k` using next-best non-replaced experts.
4. Compute `y_kept` from actual expert MLP outputs + routing weights.
5. If `replaced` is non-empty and ELT table exists for layer:
   - derive key index from `h_pre_moe` via nearest centroid.
   - load residual vector `R_L[key]`.
   - compute scale factor `s` per scaling mode.
   - output `y = y_kept + s * R_L[key]`.
6. Else fallback: `y = y_kept` (baseline behavior with kept+filled experts).

### Scaling modes (v1 experiments)
- `none`: `s = 1.0`
- `router_mass_replaced`: `s = sum(weights of replaced routed experts)`
- `normalized_router_mass`: optional calibrated variant (deferred unless needed)

Default mode for first implementation: `router_mass_replaced` (matches algorithm intuition and safer magnitude control).

---

## 10) Fallback and Failure Modes

Fallback principle: **never block inference for ELT issues in v1**.

Failure modes and behavior:

1. **Lookup file missing/unreadable**
   - Warn once; disable ELT globally; continue baseline.

2. **Header/model mismatch (`model_id`, dims, layer count)**
   - Warn with mismatch details; disable ELT globally; continue baseline.

3. **Layer payload missing/corrupt**
   - Warn; disable ELT only for affected layer; continue elsewhere if valid.

4. **No replaced experts active in layer/token**
   - Fast path baseline computations.

5. **Key lookup numeric issue (NaN distance, invalid index)**
   - Warn in debug; skip residual add for that token/layer.

6. **Trace writer backpressure/flush failure**
   - Warn and drop trace rows (bounded-loss policy) rather than stall inference.

---

## 11) Code Touchpoints in llama.cpp (Implementation Map)

Primary:
- `src/llama-graph.cpp`
  - `llm_graph_context::build_moe_ffn(...)`
  - Add trace capture and ELT substitution branches.

Model scope gate:
- `src/models/qwen35moe.cpp`
  - Restrict feature to Qwen3.5-MoE path in `build_layer_ffn(...)` flow.

CLI/config plumbing:
- `common/arg.cpp` (and associated runtime params structs)
  - Register flags and parse into config.

New modules (proposed):
- `src/moe_lookup.h/.cpp` (sidecar schema structs, loader, validation, key lookup API)
- `src/moe_trace.h/.cpp` (trace buffering/writing, metadata capture)

Optional integration points (depending on build layering):
- `src/llama-context.*` / runtime init path to attach ELT and trace handles.

---

## 12) Validation Plan

## 12.1 Correctness Validation

1. **Routing parity tests (trace mode):**
   - Assert traced `topk_ids/topk_weights` match actual runtime values for sampled events.

2. **ELT disabled equivalence:**
   - With flags off, outputs and perplexity identical to baseline within numerical tolerance.

3. **Fallback coverage tests:**
   - Broken file, mismatched header, missing layer payload all gracefully fall back.

4. **Determinism checks:**
   - Same seed/settings produce stable trace row counts and ordering.

## 12.2 Performance Validation

Measure on mixed CPU+GPU offload for baseline vs ELT variants:
- Tokens/sec
- Peak host memory / VRAM usage
- Trace overhead when enabled (target: bounded and acceptable; no hard threshold yet)

## 12.3 Quality Validation

Dataset: fixed WikiText-103 slice.

Matrix:
- replacement ratio: 5%, 10%, 20%
- keys/layer: 1k, 4k, 10k
- scaling mode: `none`, `router_mass_replaced`

Acceptance gate:
- At least one configuration achieves **<= 2.0 ppl delta** vs baseline.

---

## 13) Phased Rollout Plan with Acceptance Gates

### Stage A — Config and plumbing
Deliverables:
- CLI flags parsed, runtime config object available.
- Qwen-only gate wired.
- No-op by default.

Gate A:
- Baseline behavior unchanged when flags absent.

### Stage B — Trace capture (accuracy first)
Deliverables:
- NPZ trace writer v1 with required schema.
- Routing parity checks.

Gate B:
- 100% sampled parity pass for IDs/weights.
- Trace mode does not crash under long runs.

### Stage C — Offline builder
Deliverables:
- Tool/script producing sidecar + replacement-set artifact from NPZ.
- Header versioning + compatibility checks implemented.

Gate C:
- Sidecar validates against target model metadata.

### Stage D — Runtime ELT substitution
Deliverables:
- Inference residual lookup integration in `build_moe_ffn(...)`.
- Fallback logic for all known error modes.

Gate D:
- End-to-end run with 5–10% replacement completes without regressions/crashes.

### Stage E — Evaluation + stabilization
Deliverables:
- Baseline vs ELT comparison report (ppl/tok/s/memory).
- Best-known config documented.

Gate E (promotion gate):
- **<=2.0 ppl delta** achieved on WikiText-103 slice.
- Mixed CPU+GPU stability validated.
- Trace correctness checks remain green.

---

## 14) Tradeoffs and Rationale

1. **Qwen-only first vs generalized MoE abstraction now**
   - Chosen: Qwen-only first.
   - Why: minimizes architectural risk, shortens feedback loop, avoids premature abstraction.

2. **Sidecar vs GGUF embedding**
   - Chosen: sidecar.
   - Why: no GGUF schema changes in PoC; faster A/B iteration.

3. **FP16 residuals vs aggressive compression**
   - Chosen: FP16.
   - Why: better quality preservation and simpler implementation for initial quality gate.

4. **NPZ traces vs custom binary traces**
   - Chosen: NPZ.
   - Why: immediate interoperability with Python ecosystem for clustering experiments.

5. **Soft fallback vs hard fail**
   - Chosen: soft fallback.
   - Why: protects inference availability and simplifies experimentation in mixed environments.

---

## 15) Unresolved Items and Decision Criteria

1. **Trace write policy (buffer/flush/async model)**
   - Criteria:
     - overhead impact on tok/s,
     - implementation complexity,
     - data-loss tolerance under long runs.
   - Decision trigger: profiling data from Stage B.

2. **Inference sidecar wire format (NPZ direct vs compiled binary default)**
   - Criteria:
     - load latency,
     - runtime memory overhead,
     - ease of schema evolution and validation safety.
   - Decision trigger: Stage C prototype benchmark and implementation effort.

3. **Residual scaling formula finalization**
   - Criteria:
     - ppl delta impact across replacement ratios,
     - stability across prompt domains,
     - robustness to router mass extremes.
   - Decision trigger: Stage E matrix outcomes.

4. **Centroid precision for key search (`fp16` vs `fp32`)**
   - Criteria:
     - quality impact (nearest-key fidelity),
     - memory footprint,
     - lookup speed.
   - Decision trigger: microbench + ppl spot checks.

5. **Replacement heuristic in v1 (least-routed vs redundancy-informed)**
   - Criteria:
     - ppl retention at same replacement ratio,
     - reproducibility,
     - offline tool complexity.
   - Decision trigger: compare first two heuristics after baseline pipeline stable.

---

## 16) Risk Register (Top Risks)

1. **Trace overhead too high**
   - Impact: slow data generation, impractical workflow.
   - Mitigation: bounded buffers, optional sampling, optional async flush.

2. **Cluster keying underfits residual behavior**
   - Impact: perplexity regression beyond gate.
   - Mitigation: larger key count, improved scaling mode, heuristic refinement.

3. **Router-mass scaling mismatch**
   - Impact: unstable residual magnitude and quality variance.
   - Mitigation: test multiple scaling modes with held-out validation.

4. **Mixed CPU+GPU integration complexity**
   - Impact: runtime regressions/crashes due to tensor placement/path assumptions.
   - Mitigation: keep ELT insertion minimal and fallback-first; incremental testing on offload matrices.

5. **Schema churn between trace and sidecar**
   - Impact: brittle tooling/runtime compatibility.
   - Mitigation: versioned headers from v1 + compatibility checks + explicit migration notes.

---

## 17) Promotion / Generalization Criteria (Post-PoC)

Do **not** expand beyond Qwen until all are true:
- Reproducible <=2 ppl delta on target evaluation slice.
- Stable mixed CPU+GPU runs.
- Trace correctness parity passing consistently.
- Acceptable overhead profile for trace and lookup modes.

After this gate, extract reusable MoE lookup interfaces around `build_moe_ffn(...)` and evaluate additional MoE model families.

---

## 18) Implementation Checklist (Execution-Oriented)

- [x] Add CLI/runtime params (default-off).
- [x] Add Qwen-only activation guard.
- [x] Implement trace hooks + NPZ writer v1.
- [x] Implement parity assertions/tests for trace correctness.
- [ ] Build offline clustering/residual table tool.
- [ ] Define sidecar header v1 and loader validation.
- [ ] Integrate runtime lookup substitution in MoE path.
- [ ] Implement fallback handling and warning logs.
- [ ] Run quality/perf evaluation matrix.
- [ ] Publish result summary vs acceptance gates.

---

This document is the implementation handoff baseline for Phase 1 ELT PoC.
