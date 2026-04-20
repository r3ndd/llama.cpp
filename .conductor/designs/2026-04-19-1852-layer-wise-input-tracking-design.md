# Technical Design: Layer-wise Input Tracking (replace routed expert input traces)

Date: 2026-04-19  
Status: Proposed (implementation-ready handoff)

Related artifacts reviewed:
- `/root/llama.cpp/.conductor/plans/2026-04-17-0000-molr-plan.md`
- `/root/llama.cpp/.conductor/designs/2026-04-18-0058-molr-llm-compresson-design.md`
- `/root/llama.cpp/.conductor/designs/2026-04-18-2015-routed-input-caputre-design.md`
- `/root/llama.cpp/.conductor/designs/2026-04-19-0540-llama-cli-core-trace-design.md`

---

## 1) Goal

Convert the current MoE trace/covariance capture path from **per-routed-expert input vectors** to **layer-wise input vectors only** (one vector per token per MoE layer), and update `scripts/molr/capture_expert_covariance.py` to support this mode with no per-expert trace dependency.

Primary outcomes:
1. Major storage/compute reduction in trace collection and covariance fitting.
2. Simpler capture pipeline (no top-k join requirement for core tracing, no per-expert fallback policy complexity).
3. Backward-safe migration path for existing v1 routed/expert artifacts.

---

## 2) Key decisions

### D1) Make layer-wise tracing the canonical runtime trace format

**Decision:** Runtime tracing captures `x_layer` at MoE FFN layer entry and emits rows keyed by `layer` (no `expert` field required).

**Why preferred:**
- eliminates per-expert row multiplication (`~k`x row explosion for top-k routing),
- decouples trace collection from router internals,
- keeps capture cost predictable and bounded.

### D2) Add covariance schema v2 with explicit granularity

**Decision:** Introduce `molr_covariance_npz.v2` and `molr_covariance_summary.v2` with `granularity: "layer" | "expert"`.

**Why preferred:**
- avoids overloading v1 semantics,
- allows native layer-only outputs without fake per-expert duplication,
- enables consumers to choose lookup behavior deterministically.

### D3) Keep compatibility readers for v1 expert artifacts during transition

**Decision:** `train_expert_molr.py` and `train_all_experts.py` accept both v1 expert covariance and v2 layer covariance.

**Why preferred:**
- safe incremental rollout,
- existing runs/artifacts remain usable,
- rollback does not require data regeneration.

### D4) Simplify low-sample fallback semantics for layer mode

**Decision:** In layer mode, remove expert-vs-layer fallback logic from decision path. Sampling adequacy is validated per-layer only.

**Why preferred:**
- fallback concept is redundant when capture source is already layer-level,
- fewer branches and fewer failure categories,
- easier operator expectations.

Tradeoff: layer covariance is less specialized than per-expert covariance. We accept this for Phase-1/2 because objective is storage+compute reduction and pipeline simplification; quality risk is mitigated with validation gates and optional expert-mode fallback compatibility.

---

## 3) Architecture changes

## 3.1 Runtime/core trace path (`src/` + `common/`)

### Current (from prior design)
- Captures `ffn_moe_topk` and `ffn_moe_expert_in`.
- Reconstructs routed rows with `(layer, expert, inputs)`.

### New
- Capture only a new callback tag: `ffn_moe_layer_in` (tensor entering MoE expert block).
- Emit one JSONL event per `(layer, token)`.
- No top-k dependency, no routed row join.

### Touchpoints
- `src/llama-graph.cpp`
  - Add/retarget callback at canonical layer-input tensor:
    - `cb(cur, "ffn_moe_layer_in", il)` before expert routing/projection.
- `src/llama-moe-trace.h/.cpp`
  - Add layer-mode row encoder.
  - Keep routed/expert mode parser behind compatibility flag for one transition phase.
- `src/llama-context.cpp`
  - Wire mode selection + counters.
- `common/common.h`, `common/arg.cpp`, `common/common.cpp`
  - Add trace granularity CLI/config fields.

## 3.2 Offline capture script (`scripts/molr/capture_expert_covariance.py`)

### New responsibility split
1. **Trace ingest** (contract NPZ or JSONL capture bridge): parse layer rows.
2. **Layer grouping**: aggregate by `layer` only.
3. **Covariance fitting**: compute `(mu, chol)` per layer.
4. **Artifact emit**: write covariance v2 + summary v2 + optional layer-trace NPZ.

### Legacy support
- Keep parser support for old routed rows (`layer, expert, inputs`) but collapse to layer grouping when `--input-granularity layer`.
- Legacy expert-mode remains available only as compatibility path in transition phases.

## 3.3 Downstream training consumers

- `train_expert_molr.py`: covariance lookup order:
  1) exact `(layer, expert)` if covariance granularity/expert data exists,
  2) else layer-only `(layer)` row if v2 layer mode,
  3) else hard fail.
- `train_all_experts.py`: coverage check treats layer covariance row as satisfying all experts in that layer.

---

## 4) Normative interface changes

## 4.1 Runtime trace JSONL schema

### New schema: `moe_layer_input_jsonl.v1`

Required fields:
- `schema_version`: `"moe_layer_input_jsonl.v1"`
- `event`: `"moe_layer_input"`
- `layer`: int
- `inputs`: float[D]

Optional fields:
- `seq_id`, `token_pos`, `ubatch_index`, `model`, `ts_unix_ms`

Compatibility parser behavior (capture script):
- accepts both `moe_layer_input` and legacy routed events,
- for routed events in layer mode: ignore `expert` field.

## 4.2 Optional trace NPZ side artifact

Add `MOLR_LAYER_TRACES_NPZ_SCHEMA_VERSION = "molr_layer_traces_npz.v1"` in `scripts/molr/types.py`.

`layer_traces.npz` arrays:
- `schema_version` (scalar string)
- `model_spec` (scalar string)
- `d_model` (scalar int)
- `inputs` (`float16|float32`, `[N, D]`)
- `layers` (`int64`, `[N]`)
- optional: `seq_id`, `token_pos`

## 4.3 Covariance NPZ schema v2

Add `MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2 = "molr_covariance_npz.v2"` and summary v2 constant.

Required arrays (v2):
- `schema_version`
- `model_spec`
- `granularity` (`"layer"` or `"expert"`)
- `d_model`
- `layers` (`[M]`)
- `sample_count` (`[M]`)
- `jitter_used` (`[M]`)
- `mu` (`[M, D]`)
- `chol` (`[M, D, D]`)

Conditional arrays:
- for `granularity="expert"`: `experts` (`[M]`) required
- for `granularity="layer"`: `experts` omitted (or optional sentinel-only, not required)

## 4.4 `capture_expert_covariance.py` CLI changes

### Add
- `--input-granularity {layer,expert,auto}` (default `auto` during transition; later `layer`)
- `--capture-layer-traces` (alias of capture mode intent; keep `--capture-routed-traces` as deprecated alias)
- `--out-layer-traces-npz <path>` (new optional artifact)

### Deprecate (layer mode)
- `--fallback-to-layer-inputs-on-low-samples`
- `--min-layer-samples-for-fallback`

Layer mode validation:
- if these flags are set with `--input-granularity layer`, warn + ignore in phase 1, then hard error in phase 2.

### Keep
- existing capture controls: sample caps, dtype, allow-empty, trace path selection.

---

## 5) Data flow (layer-only mode)

1. Parse args/config and select `input_granularity=layer`.
2. Acquire traces:
   - contract mode: NPZ/JSONL ingestion,
   - capture mode: invoke `llama-cli` with layer trace enabled.
3. Validate vectors (`finite`, fixed `d_model`, non-empty).
4. Group rows by `layer`.
5. For each layer with `count >= min_samples_per_layer` (reuse current `--min-samples-per-expert` as compatibility alias in phase 1):
   - compute `mu`, covariance, Cholesky (+ jitter schedule).
6. Emit `covariance_stats.npz` v2 + summary v2.
7. Optionally emit `layer_traces.npz`.

Failure handling:
- layer-level failures are soft-fail (continue),
- global malformed input/contract mismatch remains hard-fail.

---

## 6) Storage + compute impact (and practical controls)

## 6.1 Qualitative impact

1. **Trace rows:** from per-routed row to per-layer row (remove factor `k = n_expert_used`).
2. **Covariance units:** from per-(layer,expert) to per-layer (remove factor `n_expert`).
3. **Cholesky storage:** major reduction because `chol[D,D]` dominates artifact size.
4. **Runtime tracing CPU:** lower (no top-k join/reconstruction).

## 6.2 Practical magnitude

If model has `L_moe` MoE layers, average routed experts `k`, experts/layer `E`:
- trace rows reduce by ~`k`x,
- covariance rows reduce by ~`E`x,
- covariance compute (O(rows * D^2) factorizations) reduces by ~`E`x.

For common MoE settings (`k=8`, `E=64..128`), this is substantial.

## 6.3 Recommended controls

Runtime trace defaults (safe):
- `sample_rate`: `0.1` initial,
- `max_rows_total`: 50k,
- `max_rows_per_layer`: 10k,
- `precision`: `f16`,
- `buffer_rows`: 2048,
- `flush_interval_ms`: 1000.

Capture script defaults:
- keep hard caps and dropped-row counters,
- enforce deterministic truncation/sampling in summary,
- retain `allow-empty` only for explicit scaffold workflows.

---

## 7) Migration and compatibility strategy

## Phase A (introduce, no break)
- Add new layer trace event/schema in runtime.
- `capture_expert_covariance.py` reads both legacy routed and new layer events.
- Emit covariance v2 when `--input-granularity layer`, else v1/v2 expert compatibility.
- Consumers (`train_*`) accept v1 and v2.

## Phase B (default transition)
- Default `--input-granularity` to `layer`.
- Keep deprecated expert fallback flags as no-op warning in layer mode.
- Update docs/examples to layer mode only.

## Phase C (cleanup)
- Remove routed/expert trace emission from runtime path.
- Remove expert-specific fallback semantics from capture script.
- Keep v1 read-only support in training utilities for old artifacts.

Rollback safety:
- one switch to re-enable legacy expert trace mode during transition (`--moe-trace-granularity expert` + script `--input-granularity expert`),
- no need to revert training code because dual-schema readers remain.

---

## 8) `capture_expert_covariance.py` concrete implementation plan

1. **Argument layer**
   - Add `--input-granularity`.
   - Add `--capture-layer-traces` + deprecated alias mapping from `--capture-routed-traces`.
   - Add `--out-layer-traces-npz`.
   - Introduce `--min-samples-per-layer` (compat alias fallback to existing min flag in phase 1).

2. **Validation matrix updates**
   - enforce mode-specific required fields (`layer+inputs` for layer mode),
   - reject/ignore fallback flags in layer mode per phase policy,
   - keep mutual exclusivity checks for capture vs contract modes.

3. **Trace extraction**
   - parser accepts:
     - layer event with required layer fields,
     - legacy routed events (maps to layer).
   - remove dependency on `expert` in layer mode path.

4. **Aggregation + covariance**
   - new `_build_layer_pool()` and `_compute_covariance_for_layer()`.
   - summary objects become `layers_succeeded`/`layers_failed` in v2; optionally keep compatibility counters.

5. **Outputs**
   - emit `covariance_stats.npz` v2 (granularity layer).
   - emit summary v2 with:
     - input mode/granularity,
     - rows observed/dropped,
     - layer success/failure accounting,
     - deprecation warnings encountered.
   - optional `layer_traces.npz`.

6. **Downstream compatibility hooks**
   - add adapter helpers in training scripts to resolve covariance source:
     - exact expert if available,
     - else layer row.

---

## 9) Testing strategy

## 9.1 Unit tests
- Trace parser:
  - accepts `moe_layer_input`,
  - accepts legacy routed rows in layer mode,
  - rejects missing `layer` or malformed `inputs`.
- Covariance selection:
  - layer mode ignores expert fallback flags,
  - min-sample enforcement per layer.
- Schema emit:
  - v2 arrays/fields present and consistent.

## 9.2 Integration tests
- `capture_expert_covariance.py` capture mode using mocked `llama-cli` writes layer traces and emits v2 covariance.
- `train_expert_molr.py` can train with v2 layer covariance.
- `train_all_experts.py` recognizes layer coverage and does not mark missing covariance per expert.

## 9.3 Performance/observability checks
- Compare row counts and output size vs routed baseline on same prompt set.
- Verify dropped-row counters and cap reasons in summary.
- Ensure disabled trace mode overhead is unchanged.

---

## 10) Observability and rollout

Add/retain counters:
- runtime: `rows_emitted`, `rows_dropped_*`, `effective_sample_rate`, `trace_mode`.
- capture script: `rows_total`, `rows_kept`, `rows_dropped_by_reason`, `layers_succeeded_total`, `layers_failed_total`.

Rollout:
1. Ship dual-mode runtime+script support (default old behavior).
2. Run A/B offline capture with same token budget; compare artifact size and training quality metrics.
3. Flip defaults to layer mode after acceptance thresholds.
4. Remove expert-trace emission path after one stable release cycle.

---

## 11) Files to modify

Core/runtime:
- `src/llama-graph.cpp`
- `src/llama-moe-trace.h`
- `src/llama-moe-trace.cpp`
- `src/llama-context.cpp`
- `common/common.h`
- `common/arg.cpp`
- `common/common.cpp`

MoLR scripts:
- `scripts/molr/types.py`
- `scripts/molr/capture_expert_covariance.py`
- `scripts/molr/train_expert_molr.py`
- `scripts/molr/train_all_experts.py`
- `scripts/molr/README.md` (if present)

Tests:
- `scripts/tests/test_molr_phase1_artifacts.py`
- `tests/test-arg-parser.cpp`
- optional new focused test: `scripts/tests/test_molr_layer_trace_capture.py`

---

## 12) Compact handoff decisions

1. Move canonical trace capture to layer-wise events only (`layer + inputs`).
2. Introduce covariance schema v2 with explicit granularity and native layer outputs.
3. Update `capture_expert_covariance.py` for layer-mode-first processing; simplify away expert fallback logic in that mode.
4. Keep v1/v2 compatibility readers in training scripts for safe migration/rollback.
5. Enforce caps/sampling/buffering defaults tuned for storage and compute reduction, with explicit counters for dropped data.
