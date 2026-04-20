# Technical Design: llama-cli Core MoE Trace for Routed Expert Covariance Capture

Date: 2026-04-19  
Status: Proposed (implementation-ready handoff)  
Scope: `llama-cli` + runtime/core trace path for MoE routed-input capture compatible with `scripts/molr/capture_expert_covariance.py`

Related artifacts reviewed:
- `/root/llama.cpp/.conductor/plans/2026-04-17-0000-molr-plan.md`
- `/root/llama.cpp/.conductor/designs/2026-04-18-0058-molr-llm-compresson-design.md`
- `/root/llama.cpp/.conductor/designs/2026-04-18-1913-pass-that-checklist-along-with-the-molr-plan-md--design.md`
- `/root/llama.cpp/.conductor/designs/2026-04-18-2015-routed-input-caputre-design.md`

---

## 1) Goal and constraints

Implement a **default-off**, low-risk runtime tracing capability that records routed MoE expert input vectors (`x`) from `llama-cli`/runtime, so Phase-1 covariance tooling can consume them directly.

Primary requirement: produce trace rows that `capture_expert_covariance.py` can parse today (required fields: `layer`, `expert`, `inputs`; optional fields allowed).

Constraints:
- preserve baseline behavior when tracing is disabled,
- avoid model-specific logic in CLI interfaces,
- keep runtime overhead bounded and controllable,
- keep failure mode rollback-safe (disable tracing, never break inference by default).

---

## 2) Key decisions and tradeoffs

### D1) Capture at the MoE expert-input boundary in core graph build

**Decision:** add a dedicated callback tag in `build_moe_ffn()` for the tensor that is actually fed into routed expert matmul (pre-first expert projection for non-Llama4, weighted expert input for Llama4 path where applicable).

**Why:** this is the canonical vector for expert covariance estimation; avoids reconstructing from later tensors.

**Tradeoff:** one additional callback point in hot graph build path; cost is negligible when callback is null.

### D2) Route reconstruction from existing top-k tensor + expert-input tensor

**Decision:** runtime tracer reconstructs per-routed-sample rows using:
- `ffn_moe_topk` (selected expert ids),
- new `ffn_moe_expert_in` tensor (input vectors per token/per selected expert),
- `llama_ubatch` metadata (`pos`, `seq_id`, batch index).

**Why:** minimal graph changes; no extra ggml ops.

**Tradeoff:** requires small stateful join per layer within one eval pass.

### D3) JSONL v1 event schema with strict compatibility fields

**Decision:** emit JSONL rows with `event=moe_routed_input` and fields `layer`, `expert`, `inputs`; optional metadata is additive.

**Why:** immediately compatible with `capture_expert_covariance.py` parser contract and existing `LLAMA_MOE_TRACE_*` env usage.

**Tradeoff:** JSONL is larger/slower than binary, but dramatically easier to debug and already integrated.

### D4) Default-off + soft-fail tracing

**Decision:** if tracing cannot initialize (bad path, I/O failure), inference continues unless explicit strict mode is enabled.

**Why:** tracing is observability/capture, not correctness-critical inference logic.

**Tradeoff:** possible silent data gaps if operator ignores warnings; mitigated with summary counters/log warnings.

---

## 3) Module boundaries and data flow

## 3.1 Runtime/core boundaries

1. **Config surface (common):** parse CLI/env flags into `common_params`.
2. **Runtime tracer service (`src/`):** owns buffering, row assembly, JSONL writer, counters.
3. **Graph callback hook (`llama_context::graph_get_cb`):** forwards selected MoE tensors into tracer service.
4. **Graph build tagging (`llama-graph.cpp`):** ensures expert-input tensor is named and callback-visible.
5. **CLI wiring (`tools/cli`):** no bespoke logic beyond common config usage.

## 3.2 Data flow

1. User enables trace via CLI/env.
2. `common_init_from_params()` initializes context with trace runtime state.
3. During eval callback:
   - tracer observes `ffn_moe_topk-<layer>` and stores expert ids by token,
   - tracer observes `ffn_moe_expert_in-<layer>` and materializes vectors,
   - tracer emits one row per `(token, selected_expert)`.
4. Writer buffers rows and flushes to JSONL based on configured policy.
5. `capture_expert_covariance.py --capture-routed-traces` consumes resulting file.

---

## 4) What vectors to capture, when, and exact schema

## 4.1 Vector definition (normative)

For each MoE layer invocation and each routed expert selection:
- capture vector `x_routed` of shape `[d_model]` from tensor **immediately consumed by expert projection path**,
- emit one row per `(layer, token_index_in_ubatch, selected_expert_rank)`.

If `n_expert_used = k`, each token contributes `k` rows.

## 4.2 Capture timing (normative)

Within `llm_graph_context::build_moe_ffn()`:
- existing capture point: `ffn_moe_topk` (already tagged),
- **new capture point:** `cb(cur, "ffn_moe_expert_in", il)` right before first expert `build_lora_mm_id(..., selected_experts)` call.

This guarantees layer alignment with `selected_experts` and avoids post-FFN transformations.

## 4.3 Output schema (normative, JSONL)

Each line is a JSON object:

```json
{
  "schema_version": "moe_routed_input_jsonl.v1",
  "event": "moe_routed_input",
  "layer": 12,
  "expert": 34,
  "inputs": [0.0123, -0.9932, ...],
  "seq_id": 0,
  "token_pos": 153,
  "ubatch_index": 7,
  "expert_rank": 1,
  "router_weight": 0.7321,
  "model": "<model path/spec>",
  "ts_unix_ms": 1776500000123
}
```

Compatibility-critical fields (required):
- `layer` (int)
- `expert` (int)
- `inputs` (float array, length `d_model`)

Optional/additive fields:
- `schema_version`, `event`, `seq_id`, `token_pos`, `ubatch_index`, `expert_rank`, `router_weight`, `model`, `ts_unix_ms`.

`capture_expert_covariance.py` compatibility:
- accepts `event=moe_routed_input`,
- accepts flat fields or `data.{layer,expert,inputs}`; implementation SHOULD emit flat fields for simplicity.

---

## 5) CLI/env/config surfaces (default-off)

Additive `common_params` fields (names may be bikeshedded, semantics fixed):

- `moe_trace_enable` (bool, default `false`)
- `moe_trace_path` (string, default empty)
- `moe_trace_format` (`jsonl`, default `jsonl`)
- `moe_trace_precision` (`f16|f32`, default `f16`)
- `moe_trace_sample_rate` (`0..1`, default `1.0`)
- `moe_trace_max_rows_total` (default `200000`)
- `moe_trace_max_rows_per_layer` (default `0` uncapped)
- `moe_trace_max_rows_per_expert` (default `0` uncapped)
- `moe_trace_buffer_rows` (default `2048`)
- `moe_trace_flush_interval_ms` (default `1000`)
- `moe_trace_strict` (default `false`)

CLI flags (new):
- `--moe-trace`
- `--moe-trace-path PATH`
- `--moe-trace-format jsonl`
- `--moe-trace-precision {f16,f32}`
- `--moe-trace-sample-rate FLOAT`
- `--moe-trace-max-rows-total N`
- `--moe-trace-max-rows-per-layer N`
- `--moe-trace-max-rows-per-expert N`
- `--moe-trace-buffer-rows N`
- `--moe-trace-flush-interval-ms N`
- `--moe-trace-strict`

Env compatibility (required):
- `LLAMA_MOE_TRACE_ENABLE` (`1/0`)
- `LLAMA_MOE_TRACE_JSONL` (path)
- `LLAMA_MOE_TRACE_FORMAT` (`jsonl`)

Backward-safe behavior:
- tracing disabled by default,
- flags ignored with warning for non-MoE models,
- if enabled but misconfigured and `strict=false`: trace disabled + warning, inference continues,
- if `strict=true`: startup/config failure.

---

## 6) Performance, memory, I/O, and privacy controls

## 6.1 Performance/memory/I/O controls

1. **Sampling:** Bernoulli sampling at routed-row level via `sample_rate`.
2. **Caps:** enforce total/per-layer/per-expert row caps with drop counters.
3. **Precision:** store vectors as `f16` (default) or `f32` in JSON serialization (rounded for `f16` mode).
4. **Buffering:** in-memory row buffer; flush on `buffer_rows` or `flush_interval_ms`.
5. **Writer policy:** append mode by default; caller (capture script) may pre-delete file.
6. **Backpressure:** if queue full, drop rows (do not block decode) and increment `dropped_queue_full`.

## 6.2 Observability counters

Expose in log summary (and optionally server metrics later):
- `rows_emitted`, `rows_dropped_total`, `rows_dropped_by_reason{cap_total,cap_layer,cap_expert,sample,queue_full,io_error}`,
- `flush_count`, `flush_error_count`,
- `effective_sample_rate`.

## 6.3 Security/privacy

Routed vectors can leak prompt semantics. Controls:
- explicit opt-in only,
- operator warning in help text and startup logs,
- recommend writing to restricted path (`0600`) and non-shared storage,
- no automatic network upload,
- optional future toggle to suppress metadata fields (`token_pos`, `seq_id`).

---

## 7) Failure modes and handling

1. **Path/file open failure:**
   - strict=false: disable tracer + warn,
   - strict=true: fail initialization.
2. **Write/flush error mid-run:**
   - mark tracer degraded, stop further writes, continue inference.
3. **Malformed tensor shape/type at capture point:**
   - increment `rows_dropped_shape_mismatch`, skip affected layer step.
4. **Callback collision with existing `cb_eval`:**
   - runtime callback wrapper calls tracer first, then existing callback (preserve existing behavior).
5. **Non-MoE model:**
   - no rows emitted; log once (`moe_trace_no_moe_layers`).

Safety invariant: tracing must never be required for inference correctness.

---

## 8) Probable repo touchpoints

## 8.1 Core/runtime (`src/`)
- `src/llama-graph.cpp`
  - add `cb(..., "ffn_moe_expert_in", il)` capture tag.
- `src/llama-context.cpp`
  - instantiate and own tracer runtime state,
  - call tracer from `graph_get_cb()` after tensor naming.
- `src/llama-cparams.h` (optional if runtime state pointer/flags propagated here).
- **new** `src/llama-moe-trace.h/.cpp`
  - tracer config, per-layer join state, row serialization, buffered writer, counters.

## 8.2 Common config plumbing (`common/`)
- `common/common.h`
  - add trace config fields to `common_params`.
- `common/arg.cpp`
  - add CLI flags + env bindings + validation.
- `common/common.cpp`
  - pass config into context/runtime initialization.

## 8.3 CLI/server surface (`tools/`)
- `tools/cli/README.md`
  - document trace flags and privacy warning.
- `tools/server/...` (phase-gated)
  - optionally mirror same flags in server help/props for parity.

## 8.4 Scripts/tests/docs
- `scripts/molr/README.md`
  - confirm runtime trace contract and env usage.
- `scripts/tests/test_molr_phase1_artifacts.py`
  - add integration smoke using produced JSONL field names.
- `tests/test-arg-parser.cpp`
  - add flag/env parse coverage.

---

## 9) Phased implementation plan

## Phase 1 — Core trace scaffolding (default-off)

Deliverables:
- config flags/env in `common_params`,
- new tracer module with no-op when disabled,
- `ffn_moe_expert_in` callback tag added.

Acceptance criteria:
- zero behavior change when flags absent,
- `llama-cli` starts with/without trace flags,
- no rows for non-MoE model with clear warning.

## Phase 2 — Row assembly + JSONL emission compatibility

Deliverables:
- join logic for topk + expert input tensors,
- JSONL writer with required fields,
- compatibility test feeding output into `capture_expert_covariance.py` parser path.

Acceptance criteria:
- generated JSONL rows are parseable by `_extract_trace_fields`,
- `capture_expert_covariance.py --capture-routed-traces` succeeds against produced file,
- row counts match expected `tokens * n_expert_used` under uncapped/no-sampling settings.

## Phase 3 — Controls, counters, and hardening

Deliverables:
- caps/sampling/precision/buffering controls,
- drop-reason accounting and summary logs,
- strict/non-strict error handling.

Acceptance criteria:
- cap/sampling determinism test with fixed seed,
- no decode crash on induced write failures,
- bounded overhead in disabled mode (near-zero regression).

## Phase 4 — Rollout and operationalization

Deliverables:
- docs/runbook updates,
- optional server parity for mode/status,
- staged enablement guidance for MoLR Phase-1 capture runs.

Acceptance criteria:
- documented one-command capture flow works end-to-end,
- rollback path is one switch (`--no-moe-trace` / unset env).

---

## 10) Validation strategy

## 10.1 Unit/component
- tensor shape mapping tests (`ffn_moe_topk` + `ffn_moe_expert_in` -> rows),
- schema conformance test (required fields present),
- cap/sampling/drop accounting tests,
- strict/non-strict error policy tests.

## 10.2 Integration
- `llama-cli` MoE model smoke run with trace enabled -> non-empty JSONL,
- `capture_expert_covariance.py` capture mode consumes emitted trace file and produces covariance artifacts,
- non-MoE model run yields zero rows and non-fatal warning.

## 10.3 Performance gates
- disabled mode latency delta within noise budget,
- enabled mode overhead characterized for sample rates (1.0, 0.25, 0.1),
- memory overhead bounded by configured buffer.

---

## 11) Rollout and rollback

Rollout:
1. ship default-off trace plumbing,
2. enable in internal MoLR capture runs only,
3. tune caps/sampling for large token budgets,
4. (optional) expose telemetry in server metrics.

Rollback:
- immediate: disable with CLI/env (`--no-moe-trace` or unset `LLAMA_MOE_TRACE_ENABLE`),
- if issues persist: revert tracer initialization path while leaving inert flags.

---

## 12) Acceptance criteria (final)

1. **Compatibility:** trace JSONL is accepted by `capture_expert_covariance.py` without parser changes.
2. **Correctness:** each routed `(layer, expert)` row has vector length `d_model` and finite values.
3. **Safety:** tracing remains off by default and inference succeeds when tracing fails (non-strict mode).
4. **Control:** sampling/caps/precision/buffering are configurable and reflected in summary counters.
5. **Maintainability:** changes are additive, localized to common config + runtime trace module + one graph tag.

---

## 13) Compact handoff summary

- Add a dedicated MoE expert-input callback tag in `build_moe_ffn` and reconstruct routed rows using existing `ffn_moe_topk`.
- Emit default-off JSONL trace rows with required fields (`layer`, `expert`, `inputs`) for direct compatibility with covariance capture tooling.
- Provide robust controls (sampling/caps/precision/buffering), soft-fail behavior, and explicit privacy warnings.
- Implement incrementally across common config, runtime tracer module, and minimal graph/context touchpoints with clear tests and rollback.
