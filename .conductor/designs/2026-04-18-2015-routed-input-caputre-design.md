# Technical Design: Routed-Input Trace Capture Integration for `capture_expert_covariance.py`

Date: 2026-04-18  
Status: Proposed (implementation-ready handoff)  
Scope: Phase-1 covariance capture enhancement for MoLR pipeline

Related required artifacts reviewed:
- `/root/llama.cpp/.conductor/plans/2026-04-17-0000-molr-plan.md`
- `/root/llama.cpp/.conductor/designs/2026-04-18-0058-molr-llm-compresson-design.md`

---

## 1) Goal and problem statement

`scripts/molr/capture_expert_covariance.py` currently supports only a **contract-consumer scaffold** mode:
- reads pre-captured routed inputs NPZ (`inputs`, `layers`, `experts`),
- computes per-(layer,expert) `(mu, chol)` if `sample_count >= min_samples_per_expert`,
- emits `covariance_stats.npz` + summary JSON,
- allows empty placeholder output via `--allow-empty`.

This design adds two capabilities while keeping default behavior backward-compatible:

1. **Flag-triggered routed input trace capture** integrated into this script (no separate pre-capture step required).
2. **Layer-wide fallback flag**: if a routed expert has too few samples, compute its covariance from all layer inputs (for that layer) instead of failing that expert.

---

## 2) Key design decisions

### D1) Keep existing default path unchanged; add opt-in capture mode

**Decision:** Keep current contract-only behavior as default. New capture path is only activated by an explicit flag.

**Why preferred:**
- preserves current tests/contracts and operator workflows,
- no behavior change for existing callers,
- lower migration risk.

---

### D2) Introduce dual sample pools: per-expert routed + per-layer global

**Decision:** During capture, collect both:
- routed samples keyed by `(layer, expert)` and
- layer-wide samples keyed by `layer`.

This enables deterministic substitution for under-sampled experts without a second model pass.

**Tradeoff:** more memory and bookkeeping during capture; mitigated via bounded sampling (`--max-trace-samples-*`) and optional compression.

---

### D3) Keep covariance output NPZ schema v1 stable; extend summary JSON for provenance

**Decision:** Do not break `molr_covariance_npz.v1` to avoid cascading changes in `train_all_experts.py` schema checks.

**How:**
- retain current NPZ required arrays,
- encode substitution provenance in summary JSON (`sample_source`: `routed` | `layer_fallback`),
- optionally add NPZ arrays only if additive and ignored by current consumers.

**Why preferred:** phase-2 orchestration already hard-checks `molr_covariance_npz.v1`; stability minimizes risk.

---

### D4) Capture should emit optional raw trace artifact for audit/replay

**Decision:** Add optional raw trace NPZ output (off by default) for debugging/forensics.

**Tradeoff:** potentially large files; managed by hard caps and optional `float16` trace storage.

---

## 3) Proposed CLI interface (normative)

## 3.1 Existing flags (unchanged)
- `--model`
- `--tokens`
- `--routed-inputs-npz`
- `--min-samples-per-expert`
- `--max-experts`
- `--allow-empty`
- `--out-npz`
- `--out-json`

## 3.2 New flags

### A) Capture trigger flag

- `--capture-routed-traces`
  - Type: boolean flag
  - Default: `False`
  - Semantics:
    - `False`: current behavior (consume `--routed-inputs-npz` contract).
    - `True`: run integrated model-token pass to generate routed traces in-process.

### B) Capture input source flags (required when capture enabled)

- `--capture-prompts-jsonl <path>`
  - JSONL source for prompts/text chunks used for tokenization + forward pass.
  - Minimum required for deterministic reproducibility.

- `--capture-max-sequences <int>` (default `0` = no explicit sequence cap)
- `--capture-seed <int>` (default `0`)

### C) Layer fallback flag (required by this request)

- `--fallback-to-layer-inputs-on-low-samples`
  - Type: boolean flag
  - Default: `False`
  - Semantics:
    - If expert routed `sample_count < min_samples_per_expert`:
      - when `False`: current behavior (`insufficient_samples` failure for that expert),
      - when `True`: use all captured inputs for that expert’s layer to estimate covariance for that expert.

### D) Fallback floor guard

- `--min-layer-samples-for-fallback <int>`
  - Type: int
  - Default: `0` (interpreted as `min_samples_per_expert`)
  - Semantics: fallback is only allowed if layer sample count meets this threshold.

### E) Trace retention / limits

- `--out-routed-traces-npz <path>` (optional)
- `--trace-dtype {float16,float32}` (default `float16`)
- `--max-trace-samples-total <int>` (default `200000`)
- `--max-trace-samples-per-expert <int>` (default `0` = uncapped)
- `--max-trace-samples-per-layer <int>` (default `0` = uncapped)

## 3.3 Flag compatibility rules

- `--capture-routed-traces` and `--routed-inputs-npz` are mutually exclusive.
- If capture is enabled, `--capture-prompts-jsonl` is required.
- `--allow-empty` remains valid only for non-capture mode (or capture mode with explicit `--allow-empty` + no captured rows).
- New fallback flag applies in both modes (contract NPZ mode and capture mode).

---

## 4) Data flow and module boundaries

## 4.1 Updated high-level flow

1. Parse CLI and mode-select (`contract` vs `capture`).
2. Produce unified in-memory trace tables:
   - routed table: `inputs_routed`, `layers`, `experts`
   - layer table: `inputs_layer`, `layers_layer`
3. Aggregate samples by `(layer, expert)` and by `layer`.
4. For each target expert:
   - if routed count >= `min_samples_per_expert`: use routed pool,
   - else if fallback flag on and layer count threshold met: use layer pool,
   - else: fail expert.
5. Compute `mu`, covariance, Cholesky with existing jitter schedule.
6. Emit outputs + detailed provenance in summary JSON.

## 4.2 Internal boundaries (within same script, no architecture rewrite)

Recommended internal interfaces (function-level contracts):

- `capture_routed_traces(config) -> TraceCaptureResult`
  - Responsible for model run + trace extraction.
- `load_routed_contract(path) -> TraceCaptureResult`
  - Existing NPZ loader path.
- `build_sample_pools(trace_result) -> (expert_pool, layer_pool)`
  - deterministic grouping.
- `select_samples_for_expert(layer, expert, pools, cfg) -> SampleSelection`
  - encapsulates fallback decision logic.
- `compute_covariance(selection) -> CovarianceResult`
  - existing mean/cov/cholesky logic.
- `emit_artifacts(...)`
  - NPZ + summary + optional trace NPZ.

This keeps behavior readable/testable while remaining incremental.

---

## 5) Schemas and storage strategy

## 5.1 Existing output (retain)

`covariance_stats.npz` (`molr_covariance_npz.v1`) remains required-compatible:
- `schema_version`
- `model_spec`
- `d_model`
- `layers`
- `experts`
- `sample_count`
- `jitter_used`
- `mu`
- `chol`

`covariance_summary.json` (`molr_covariance_summary.v1`) gains additive metadata only.

## 5.2 New optional raw trace artifact

`routed_traces.npz` (`molr_routed_traces_npz.v1`) when `--out-routed-traces-npz` is set:

Required arrays:
- `schema_version` (scalar string)
- `model_spec` (scalar string)
- `d_model` (scalar int)
- `inputs` (`float16|float32`, shape `[N, D]`) routed expert inputs
- `layers` (`int64`, shape `[N]`)
- `experts` (`int64`, shape `[N]`)

Optional arrays (if available from capture backend):
- `seq_id` (`int64`, `[N]`)
- `token_pos` (`int64`, `[N]`)
- `router_score` (`float32`, `[N]`)

## 5.3 Summary JSON additive fields

Add under each expert record:
- `sample_source`: `"routed" | "layer_fallback"`
- `routed_sample_count`
- `layer_sample_count`
- `effective_sample_count`
- `fallback_applied` (bool)

Add top-level accounting:
- `fallback_policy` block (enabled flag + thresholds)
- `experts_fallback_used_total`
- `experts_fallback_failed_total`

---

## 6) Selection semantics (normative)

Given expert key `(L, E)`:

1. `routed_count = |X_routed[L,E]|`
2. If `routed_count >= min_samples_per_expert`:
   - `X_effective = X_routed[L,E]`
   - `sample_source = routed`
3. Else if `fallback_to_layer_inputs_on_low_samples = true`:
   - `layer_count = |X_layer[L]|`
   - `layer_threshold = min_layer_samples_for_fallback if >0 else min_samples_per_expert`
   - If `layer_count >= layer_threshold`:
     - `X_effective = X_layer[L]`
     - `sample_source = layer_fallback`
   - Else fail with `insufficient_layer_samples_for_fallback`
4. Else fail with `insufficient_samples(<min>)`

Important invariant: fallback substitution happens **only** for experts below threshold; experts with enough routed samples always use routed-only data.

---

## 7) Performance, limits, and risks

## 7.1 Main risks
- **Memory blow-up** while storing dense traces (`N x D`) for large token budgets.
- **I/O pressure** if raw trace NPZ is written unbounded.
- **Sampling bias** if cap policies unintentionally skew expert/layer distributions.

## 7.2 Mitigations
- default `float16` for optional trace dumps,
- hard cap defaults (`max-trace-samples-total`),
- deterministic reservoir sampling per `(layer,expert)` and per `layer`,
- clear summary counters for dropped samples due to caps.

## 7.3 Complexity expectation
- Covariance compute remains O(D²) per successful expert.
- New capture pass cost dominated by model forward; bounded by `--tokens` and input caps.

---

## 8) Failure modes and handling

## 8.1 Validation failures (hard fail)
- mutually incompatible flags,
- missing capture source when capture enabled,
- invalid trace schema in contract mode,
- non-finite inputs.

## 8.2 Expert-level soft failures (continue)
- insufficient routed samples and fallback disabled,
- fallback enabled but insufficient layer samples,
- Cholesky failure after jitter schedule.

These remain accounted in `experts_failed` and `failure_accounting.by_reason`.

## 8.3 Capture runtime failures
- model load / tokenizer / generation failure in capture mode:
  - global script failure (exit 2) unless `--allow-empty` explicitly requested,
  - summary should include global failure reason.

---

## 9) Backward compatibility guarantees

1. Existing invocations using `--routed-inputs-npz` continue to work unchanged.
2. Existing `--allow-empty` scaffold behavior is preserved.
3. `covariance_stats.npz` schema remains `molr_covariance_npz.v1` for Phase-2 compatibility.
4. New behavior requires explicit flags; defaults are safe and conservative.

---

## 10) Files to modify

Primary:
- `scripts/molr/capture_expert_covariance.py`

Schema/constants:
- `scripts/molr/types.py`
  - add optional new constant: `MOLR_ROUTED_TRACES_NPZ_SCHEMA_VERSION = "molr_routed_traces_npz.v1"`

Tests:
- `scripts/tests/test_molr_phase1_artifacts.py`
  - add fallback substitution coverage and new flag validation tests.
- (optional) new focused file:
  - `scripts/tests/test_molr_capture_routed_traces.py`

Docs:
- `scripts/molr/README.md` (if present in branch by implementation time)

---

## 11) Phased rollout and checklist

## Phase A — CLI and selection-logic extension (no runtime capture yet)
- [ ] Add new fallback flags and selection semantics in contract mode.
- [ ] Add summary provenance fields.
- [ ] Keep existing tests green.
- [ ] Add tests for low-sample fallback path.

## Phase B — Integrated capture mode (trigger flag)
- [ ] Add `--capture-routed-traces` mode and capture source flags.
- [ ] Implement in-script trace collection + pool building.
- [ ] Add optional raw trace NPZ output.
- [ ] Add capture-mode validation and smoke tests.

## Phase C — Scale/perf hardening
- [ ] Add cap-based sampling and dropped-sample accounting.
- [ ] Validate memory footprint under 10k/50k token runs.
- [ ] Confirm output compatibility with `train_all_experts.py`.

## Phase D — Operationalization
- [ ] Update runbook commands in design/README.
- [ ] Record recommended defaults for token budget + cap settings.

---

## 12) Validation strategy

## 12.1 Unit-level
- Flag parsing/compatibility matrix tests.
- Selection semantics tests:
  - routed sufficient (no fallback),
  - routed insufficient + fallback enabled + layer sufficient,
  - routed insufficient + fallback disabled,
  - routed insufficient + layer insufficient.
- Provenance field correctness in summary.

## 12.2 Integration-level
- Contract mode end-to-end against synthetic NPZ fixtures.
- Capture mode smoke test (small prompt set, tiny token budget).
- Optional trace NPZ schema/shape test.

## 12.3 Pipeline compatibility
- Run `train_all_experts.py` against produced covariance NPZ and ensure no schema rejection.
- Verify expected skip/pass behavior remains deterministic.

## 12.4 Performance gates
- Validate capture run memory within predefined envelope.
- Validate wall-clock increase vs contract mode is expected and documented.

---

## 13) Example command set (post-implementation)

### 13.1 Existing contract mode (unchanged)
```bash
python scripts/molr/capture_expert_covariance.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --tokens 50000 \
  --routed-inputs-npz "<run>/routed_inputs.npz" \
  --min-samples-per-expert 16 \
  --out-npz "<run>/covariance_stats.npz" \
  --out-json "<run>/covariance_summary.json"
```

### 13.2 Contract mode + layer fallback
```bash
python scripts/molr/capture_expert_covariance.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --tokens 50000 \
  --routed-inputs-npz "<run>/routed_inputs.npz" \
  --min-samples-per-expert 16 \
  --fallback-to-layer-inputs-on-low-samples \
  --min-layer-samples-for-fallback 32 \
  --out-npz "<run>/covariance_stats.npz" \
  --out-json "<run>/covariance_summary.json"
```

### 13.3 Full integrated capture mode (new)
```bash
python scripts/molr/capture_expert_covariance.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --tokens 50000 \
  --capture-routed-traces \
  --capture-prompts-jsonl "<data>/prompts.jsonl" \
  --capture-seed 42 \
  --min-samples-per-expert 16 \
  --fallback-to-layer-inputs-on-low-samples \
  --out-routed-traces-npz "<run>/routed_traces.npz" \
  --out-npz "<run>/covariance_stats.npz" \
  --out-json "<run>/covariance_summary.json"
```

---

## 14) Compact handoff decisions

1. Add explicit capture trigger `--capture-routed-traces`; keep current contract mode default.
2. Add explicit fallback control `--fallback-to-layer-inputs-on-low-samples` for under-sampled experts.
3. Preserve `molr_covariance_npz.v1` to avoid breaking Phase-2 tooling.
4. Track provenance and fallback usage in summary JSON for auditability.
5. Roll out incrementally: fallback logic first, capture integration second, then scale hardening.
