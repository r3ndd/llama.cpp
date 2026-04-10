# Brainstorm Plan: Expert Lookup Table (Algorithm 2, Qwen3.5-MoE)

Date: 2026-04-10 19:17  
Model target: `Qwen/Qwen3.5-35B-A3B` (GGUF in llama.cpp runtime)  
Primary objective: **Proof of concept**  
Acceptance gate: **<= 2.0 perplexity delta** on a WikiText-103 slice  
Hardware target: **mixed CPU+GPU offload**

## 1) Brainstorm Outcome Summary

We agreed to start with a narrow but high-signal prototype:

- Implement **Algorithm 2 (shared table per MoE layer)** first.
- Scope code changes to **Qwen3.5-MoE path first** for faster iteration.
- Use **input hidden-state clustering** as the lookup key strategy.
- Keep trace-capture implementation details for a later implementation pass, but enforce two non-negotiables:
  - **100% routing-trace correctness** (exact top-k expert IDs/weights seen by inference path)
  - **minimal runtime overhead** during trace collection.

---

## 2) Where to Modify llama.cpp (initial map)

### Primary insertion point
- `src/llama-graph.cpp` → `llm_graph_context::build_moe_ffn(...)`
  - This is the central MoE assembly point:
    - computes router logits/probs
    - selects top-k experts (`ggml_argsort_top_k`)
    - applies expert MLP paths
    - applies expert weights and aggregates output.

### Initial model-specific scope
- `src/models/qwen35moe.cpp`
  - Qwen3.5-MoE calls `build_moe_ffn(...)` in `build_layer_ffn(...)`.
  - Best place to gate feature enablement for phase 1 (e.g., runtime flag/config check) without broad architectural risk.

---

## 3) Hypotheses to Test

H1. Replacing a subset of routed experts with a **shared per-layer lookup residual** (Algorithm 2) can preserve quality within <=2 ppl delta at modest replacement ratios.

H2. Layer-input hidden-state clustering is a sufficient keying mechanism to approximate missing expert contribution for a first PoC.

H3. For mixed CPU+GPU offload setups, lookup substitution can reduce effective MoE compute/memory pressure enough to improve practical deployability without catastrophic quality loss.

---

## 4) Experiment Design (PoC)

### Independent variables
- Replacement ratio per MoE layer (start small): e.g. 5%, 10%, 20% of experts replaced.
- Lookup table size per layer (cluster count): e.g. 1k, 4k, 10k keys.
- Expert ranking heuristic for replacement (start simple):
  - least-routed experts,
  - then compare vs redundancy-oriented heuristics later.

### Dependent metrics
- Perplexity on WikiText-103 slice (primary gate, <=2 delta).
- Throughput (tok/s) and memory footprint in mixed CPU+GPU offload.
- Trace correctness checks (selected experts/weights parity in tracing mode).

### Control
- Unmodified Qwen3.5-MoE baseline with identical runtime settings and eval corpus.

---

## 5) Algorithm 2 PoC Mechanics (concrete)

For each MoE layer `L`:

1. Collect training tuples over prompt tokens:
   - input hidden state `h_L` (pre-MoE input),
   - top-k selected experts + weights,
   - full MoE output `y_full`,
   - reduced output `y_kept` (with replaced experts removed and next-best non-replaced experts filling top-k count),
   - residual target `r = y_full - y_kept`.

2. Build shared key space:
   - k-means clusters over `h_L` for that layer.

3. Build lookup table:
   - For each cluster key `c`, store average residual `R_L[c] = mean(r)`.

4. In inference (when replaced expert selected):
   - compute regular MoE from non-replaced experts (with fallback next-best not-replaced experts to maintain k count),
   - find key from `h_L`, retrieve `R_L[key]`,
   - add scaled residual to combined MoE output.

5. Residual scaling option:
   - weight residual by aggregate router mass that would have gone to replaced experts.
   - Keep as configurable strategy in experiments.

---

## 6) Proposed Implementation Stages

### Stage A — Instrumentation and toggle plumbing
- Add experimental runtime toggles (off by default) for:
  - enable tracing,
  - enable lookup substitution,
  - path to per-layer lookup artifacts.
- Keep all behavior no-op unless enabled.

### Stage B — Trace capture (accuracy-first)
- Capture exact tensors required for Algorithm 2 training data at MoE boundary.
- Validate parity: traced selected experts/weights must match actual inference routing exactly.
- Add low-overhead mode and bounded buffering to reduce slowdown risk.

### Stage C — Offline table builder (external tool/script)
- Consume trace files.
- Perform layer-wise clustering + residual averaging.
- Emit compact per-layer table artifact format.

### Stage D — Inference-time lookup substitution
- In Qwen3.5-MoE path, enable replacement list + table lookup apply.
- Fallback to normal MoE if key/table/layer missing.

### Stage E — Evaluation harness
- Run baseline + modified on WikiText-103 slice.
- Produce ppl delta + speed/memory comparisons.

---

## 7) Risks and Mitigations

1. **Trace overhead too high**
   - Mitigate with sampling mode, binary format, and optional asynchronous flush.

2. **Keying quality too weak** (hidden-state clusters underfit residuals)
   - Mitigate by increasing table size or testing hybrid key later.

3. **Router-weight interaction mismatch**
   - Mitigate by testing multiple residual scaling formulas and calibrating on held-out traces.

4. **Generalization across prompt domains**
   - Mitigate by expanding prompt diversity once PoC passes initial gate.

---

## 8) Minimal Success Criteria (Phase 1)

- Feature runs end-to-end on Qwen3.5-MoE in llama.cpp with mixed CPU+GPU offload.
- Lookup substitution can replace at least a small expert subset (e.g., 5-10%) without crashes/regressions.
- WikiText-103 slice perplexity delta is <=2.0 for at least one replacement/table-size setting.

---

## 9) Open Questions Resolution (continued brainstorm)

### 9.1 Trace artifact schema (v1 decision)

**Decision:** Use **NPZ per run** for phase 1.

Rationale:
- Fastest path for offline Python tooling (k-means + aggregation).
- Easy inspectability/debugging while algorithm is still evolving.
- Avoids premature format lock-in.

Proposed NPZ payload (per run, per layer arrays):
- `layer_ids` (int16/int32)
- `token_ids` (int32) or token index within run
- `h_pre_moe` (float16) — pre-MoE hidden state used for keying
- `topk_ids` (int16/int32)
- `topk_weights` (float16)
- `y_full` (float16) and/or `residual_target` (float16)
- `replaced_mask` (bool) for analysis slices
- metadata JSON side entry: model name, commit hash, n_expert_used, routing mode, prompt source

Note: For strict overhead control, we can start by storing only the minimum required to reconstruct residuals offline (`h_pre_moe`, routing info, and outputs), then prune fields after profiling.

### 9.2 Runtime config surface (v1 decision)

**Decision:** Use **CLI flags + file path** (no env/metadata dependency in v1).

Proposed flags (names tentative):
- `--moe-lookup-enable`
- `--moe-lookup-file /path/to/lookup.npz|bin`
- `--moe-lookup-replaced-experts /path/to/replaced.json`
- `--moe-trace-enable`
- `--moe-trace-out /path/to/traces/`

Design rule:
- Default-off experimental behavior.
- If flags are absent or artifacts invalid, silently fall back to baseline MoE path (with clear warning log).

### 9.3 Residual vector representation (v1 decision)

**Decision:** Store residual vectors as **FP16** in v1.

Rationale:
- Best early tradeoff between quality retention and artifact size.
- Minimal implementation complexity versus INT8/Hadamard binary.
- Keeps numeric behavior closer to baseline during early validation.

Deferred to v2:
- INT8 residual quantization and/or transform-compressed variants after ppl gate is stable.

### 9.4 Lookup table packaging (v1 decision)

**Decision:** Keep lookup tables in a **separate sidecar artifact** for phase 1.

Rationale:
- No GGUF format changes required for PoC.
- Faster iteration on schema and training pipeline.
- Easier A/B swapping of table variants without model repack.

Compatibility note:
- Define a simple versioned sidecar header early (`format_version`, `model_id`, `n_layer`, `n_embd`) to reduce migration pain.

### 9.5 Promotion path after Qwen-only PoC (v1 decision)

**Decision:** **Stay Qwen-only until stable**.

Stability exit criteria before generalization:
- Reproducible <=2 ppl delta on WikiText-103 slice.
- No crash/regression in mixed CPU+GPU offload runs.
- Trace correctness checks passing consistently.
- Basic profiling showing acceptable overhead.

Only then:
- Factor reusable abstractions around `build_moe_ffn(...)` and expand model coverage.

### 9.6 Remaining open items (next pass)

1. Exact trace write policy for low-overhead mode (buffer sizing, flush cadence, async/threading model).
2. Final sidecar file format for inference runtime (NPZ direct load vs compiled binary generated from NPZ).
3. Residual scaling formula choice for Algorithm 2 in inference (raw add vs router-mass scaled vs calibrated).
