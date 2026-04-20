# Technical Design: MoLR Runtime Checklist Execution Plan (Config → Validation → Guarded Runtime)

Date: 2026-04-18
Status: Proposed (implementation handoff)
Scope: Runtime integration + rollout controls for MoLR in llama.cpp
Related artifacts:
- `/root/llama.cpp/.conductor/plans/2026-04-17-0000-molr-plan.md`
- `/root/llama.cpp/.conductor/designs/2026-04-18-0058-molr-llm-compresson-design.md`

---

## 1. Objective and constraints

This plan operationalizes the checklist into a sequence of small, reviewable changes that make MoLR runtime support:

1. **Explicit opt-in only** (never on by default).
2. **Safe by construction** via startup compatibility checks and full-expert fallback.
3. **Observable** via telemetry counters for fallback/latency/error.
4. **Rollout-safe** via shadow mode + A/B gates.
5. **Reversible quickly** via single-switch disable and manifest rollback.

### Guardrails

- Preserve existing inference behavior when MoLR is disabled.
- Avoid model-specific hard-coding in server APIs; keep runtime controls generic.
- Keep graph/runtime hooks minimal and behind gating predicates.
- Prefer additive schema evolution with `schema_version` and strict validation.

---

## 2. Key decisions and tradeoffs

## D1) Centralize runtime controls in `common_params` and reuse existing arg plumbing

**Decision:** Add MoLR settings to `common_params` (used by both `llama-server` and `llama-cli`) and parse in `common/arg.cpp`.

**Why preferred:**
- One source of truth for CLI/server config and env vars.
- Fits current architecture (`common_params_parse` for both server and CLI).

**Tradeoff:** Slightly larger global parameter struct; acceptable for consistency and maintainability.

---

## D2) Bundle compatibility is a startup gate, not best-effort per-token logic

**Decision:** Validate MoLR bundle manifest at model load (`server_context::load_model` path and underlying loader helper), then set runtime mode:
- `DISABLED` (feature off)
- `SHADOW` (compute/record, do not affect served output)
- `ACTIVE` (eligible for serving when quality gate passes)

**Why preferred:**
- Moves schema/layout drift failures to startup time.
- Avoids latent crashes/undefined behavior mid-generation.

**Tradeoff:** Slight startup overhead for manifest + checksums; negligible relative to model load.

---

## D3) Keep MoE forward hook non-invasive and predicate-guarded

**Decision:** Add a MoLR branch in MoE FFN build/eval path only when all conditions hold:

`molr_enabled && bundle_compatible && expert_has_entry && pred_error <= threshold`

Else: full expert path unchanged.

**Why preferred:**
- Keeps baseline path untouched and easy to reason about.
- Limits regression blast radius.

**Tradeoff:** Additional branch complexity in hot path; mitigated by strict gating and shadow-first rollout.

---

## D4) Telemetry must be first-class before active rollout

**Decision:** Add counters/histograms into server metrics export before enabling ACTIVE mode.

**Why preferred:**
- Enables data-driven rollback and threshold tuning.
- Required for quality and latency confidence.

**Tradeoff:** Some metrics overhead; bounded by coarse aggregation (no per-token heavy logs by default).

---

## 3. Checklist-to-touchpoint mapping

| Checklist item | Primary touchpoints | Implementation intent |
|---|---|---|
| 1) Runtime config plumbing (server + CLI), explicit opt-in | `common/common.h`, `common/arg.cpp`, `tools/server/README.md`, `tools/cli/README.md` (generated) | Add MoLR flags/env vars with default disabled; expose identical semantics to CLI/server. |
| 2) Bundle load + compatibility checks in startup with safe fallback | `tools/server/server-context.cpp` (`load_model`), new MoLR runtime helper under `src/` (manifest parser + compat check), optional `src/llama-model-loader.*` adapter | Validate manifest schema/model fingerprint/layout before marking MoLR usable; on failure log warning + disable MoLR (no crash). |
| 3) Expert forward-path hook with MoLR enable + threshold | `src/llama-graph.h`, `src/llama-graph.cpp` (MoE FFN path), associated model structs in `src/llama-model.h` as needed | Add guarded branch for MoLR expert computation and threshold fallback decision; preserve current full-expert path as default branch. |
| 4) Telemetry for fallback/latency/error | `tools/server/server-context.cpp` (`server_metrics`, `/metrics` exporter, `/props`), `tools/server/server-task.h/.cpp` if needed | Add aggregate counters/gauges for MoLR attempts, fallback counts/rates, predicted error summaries, latency split (MoLR/full/shadow). |
| 5) Shadow mode / A-B safety rollout | `common/common.h`, `common/arg.cpp`, `tools/server/server-context.cpp`, runtime MoLR state manager under `src/` | Introduce `off|shadow|active` mode and sampled shadow execution; return baseline outputs until gate passes. |
| 6) Failure and rollback operations | server startup + runtime guard rails, docs/runbook under `.conductor` + `tools/server/README.md` | Single-switch disable and auto-demotion conditions; manifest pinning and rollback recipe. |
| 7) Validation required before prod enablement | CI + benchmark scripts (new lightweight validation commands), server tests under `tools/server/tests` | Define mandatory pass/fail gates for schema, quality, latency, and stability before active rollout. |

---

## 4. Proposed interfaces (no code, normative behavior)

## 4.1 New runtime config surface (explicit opt-in)

Add to `common_params` (names indicative; implementers may bikeshed naming while preserving semantics):

- `molr_mode`: enum/string `{off, shadow, active}`; default `off`.
- `molr_bundle`: string path/URI to packaged bundle manifest root.
- `molr_threshold`: optional float override.
- `molr_quality_profile`: optional named profile mapping to calibrated threshold.
- `molr_shadow_sample_rate`: float `[0,1]`, default conservative (e.g. `0.05`) when shadow enabled.
- `molr_auto_disable_fallback_rate`: float threshold for safety demotion.
- `molr_auto_disable_error_rate`: float threshold for runtime errors in MoLR branch.

CLI/env requirements:
- All MoLR flags disabled by default.
- Flags are ignored (with warning) when no MoE experts exist.
- Incompatible combinations fail fast (e.g., `active` without bundle).

## 4.2 Bundle manifest contract (runtime-critical subset)

Runtime-required fields in `molr_bundle_manifest.json`:
- `schema_version`
- `model_id` (canonical identifier)
- `model_fingerprint` (architecture + tensor/layout hash)
- `llama_commit_compat` (range or exact tested baseline)
- `experts[]` mapping keyed by `(layer, expert)` with artifact refs
- `threshold_profiles` map
- `checksums` for all referenced artifact files

Compatibility gate outcomes:
- **PASS:** MoLR runtime can enter SHADOW/ACTIVE (depending on config).
- **SOFT FAIL:** log warning, set effective mode `off`, continue full expert.
- **HARD FAIL (optional strict mode only):** startup failure if operator explicitly requested strict compatibility.

## 4.3 Runtime decision state machine

For each routed expert invocation:

1. If `effective_mode == off` → full expert.
2. If `effective_mode == shadow`:
   - compute baseline full expert result (served output),
   - optionally run MoLR on sampled subset,
   - record deltas/latency/fallback telemetry only.
3. If `effective_mode == active`:
   - run error-head estimate,
   - if `pred_error > threshold` → fallback full expert,
   - else run MoLR output.
4. Any MoLR runtime exception/load miss → increment error counter + immediate per-expert demotion for session.

Safety invariant: **served output must always be available from full-expert path.**

---

## 5. Detailed phased implementation plan

Each phase is intended as one or more small PRs.

## Phase 1 — Config plumbing and startup validation scaffold

**Goal:** Add explicit opt-in controls and non-invasive startup checks with no change to serving outputs.

### Tasks

1. Add MoLR config fields in `common/common.h`.
2. Add arg parsing + env wiring in `common/arg.cpp`:
   - server + CLI examples.
   - usage/help text includes explicit “disabled by default” warning.
3. Extend server `/props` payload (`tools/server/server-context.cpp`) with effective MoLR mode + bundle status.
4. Introduce runtime manifest parser/validator module under `src/` (e.g., `src/llama-molr-runtime.*`).
5. Integrate validator call into `server_context::load_model` startup sequence.

### Dependencies

- None (foundation phase).

### Acceptance criteria

- `llama-server` and `llama-cli` accept MoLR flags.
- Default behavior remains identical when flags absent.
- Invalid/absent bundle in non-strict mode downgrades to `off` with warning (no crash).
- `/props` includes MoLR mode/status for observability.

### Rollback controls

- Remove/disable MoLR flags at launch (`--molr-mode off`) instantly restores baseline.

---

## Phase 2 — Telemetry first (before functional MoLR serving)

**Goal:** Add metrics plumbing required for safe rollout and operations.

### Tasks

1. Extend server metrics structures in `tools/server/server-context.cpp` (and result structs if needed):
   - `molr_attempt_total`
   - `molr_fallback_total`
   - `molr_runtime_error_total`
   - `molr_shadow_eval_total`
   - `molr_pred_error_sum` / count
   - latency accumulators for `molr_path_ms`, `fallback_path_ms`, `shadow_overhead_ms`
2. Export new Prometheus metrics from `/metrics`.
3. Add lightweight logs with bounded cardinality (avoid per-token noisy labels).
4. Add server test coverage for metrics endpoint fields and disabled-mode invariants.

### Dependencies

- Phase 1 complete (mode + status available).

### Acceptance criteria

- New metrics visible only when server metrics endpoint enabled (`--metrics`).
- Overhead benchmark in disabled mode shows no meaningful regression vs baseline budget.
- Telemetry remains valid with mode transitions (`off` ↔ `shadow` ↔ `active`).

### Rollback controls

- Metrics can be ignored operationally without changing inference behavior.

---

## Phase 3 — Guarded MoE forward hook (functional path, default-off)

**Goal:** Introduce MoLR branch in MoE expert execution with strict guard predicates and fallback.

### Tasks

1. Add MoLR runtime state accessor to graph build/eval context.
2. Introduce guarded branch in MoE FFN path (`src/llama-graph.cpp`, declarations in `src/llama-graph.h`):
   - preserve existing full-expert branch exactly.
   - execute MoLR branch only when mode active and expert mapping exists.
3. Implement threshold decision logic using preloaded error-head parameters.
4. On any MoLR branch failure, route to full expert and emit telemetry counter.
5. Add per-expert session demotion after repeated failures.

### Dependencies

- Phase 1 (manifest/runtime state)
- Phase 2 (telemetry instrumentation)

### Acceptance criteria

- In `off` mode, bitwise/near-bitwise parity with baseline path (as applicable to existing nondeterminism).
- In `active`, fallback always succeeds when MoLR path rejects/fails.
- No crashes from missing expert entries in bundle.

### Rollback controls

- Runtime switch `--molr-mode off` bypasses MoLR branch globally.
- Per-expert auto-demotion toggles expert back to full path for current session.

---

## Phase 4 — Shadow mode and A/B safety rollout

**Goal:** Validate quality/performance before serving MoLR outputs.

### Tasks

1. Implement shadow executor in server inference loop:
   - serve baseline full-expert output,
   - run MoLR side-path on sampled traffic,
   - collect divergence and latency telemetry.
2. Add A/B control hooks (config-based split) at process level:
   - Group A: `off`
   - Group B: `shadow`
   - Group C (later): `active` pilot
3. Define and document promotion gates from shadow to active.

### Dependencies

- Phase 3 (functional hook exists)

### Acceptance criteria

- Shadow mode does not change output payloads.
- Statistical reports available: fallback-rate projections, latency deltas, error-head calibration drift.
- Promotion decision can be made from telemetry without code changes.

### Rollback controls

- Downgrade any group to `off` via config/env only.

---

## Phase 5 — Limited active pilot and expansion gates

**Goal:** Controlled enablement with explicit stop/rollback conditions.

### Tasks

1. Enable ACTIVE for a constrained pilot (single model, low traffic slice).
2. Enforce auto-disable conditions:
   - fallback rate > configured threshold for sustained window,
   - MoLR runtime error rate above threshold,
   - latency budget breach.
3. Add operator runbook for rollback:
   - config kill switch,
   - manifest pin/rollback,
   - restart requirements and verification steps.
4. After stable pilot, increase scope incrementally by model/traffic tier.

### Dependencies

- Phase 4 gates pass.

### Acceptance criteria

- Pilot remains within agreed quality/latency/error budgets.
- Rollback drill executed successfully in staging.

### Rollback controls

- Immediate: set `molr_mode=off`.
- Secondary: revert bundle manifest pointer to last known-good.
- Last resort: binary rollback to prior release.

---

## 6. Milestone checklist with deliverables

## M1: Config + bundle validation complete
- Deliverables: arg/docs updates, startup validator, `/props` MoLR status.
- Exit gate: zero behavior change when disabled.

## M2: Telemetry complete
- Deliverables: `/metrics` MoLR counters/gauges, tests.
- Exit gate: metrics verified in staging with negligible disabled-mode overhead.

## M3: Guarded forward hook complete
- Deliverables: MoE hook, threshold/fallback branch, error handling.
- Exit gate: fallback reliability + off-mode parity.

## M4: Shadow/A-B complete
- Deliverables: shadow sampling path + comparison reporting.
- Exit gate: quality and latency targets meet promotion thresholds.

## M5: Pilot active and expansion-ready
- Deliverables: runbook, auto-disable guardrails, pilot report.
- Exit gate: production enablement approval based on validation criteria.

---

## 7. Validation strategy (required before production)

## 7.1 Unit / component validation

- Manifest parser schema/version tests.
- Compatibility matcher tests (model fingerprint, layer/expert coverage).
- Threshold decision tests (`<=`, `>`, NaN/Inf handling).
- Metrics accumulation tests for off/shadow/active modes.

## 7.2 Integration validation

- `llama-server` startup matrix:
  - off + no bundle
  - shadow + valid bundle
  - active + invalid bundle (soft-disable path)
- End-to-end inference sanity with MoE models and non-MoE models.
- `/props` and `/metrics` contract tests.

## 7.3 Quality/performance validation gates

Must pass before ACTIVE production:
- Output quality delta within agreed budget (task-level benchmarks/perplexity).
- Fallback rate in target band for chosen threshold profile.
- p50/p95 latency regression within approved envelope.
- Runtime error rate below threshold with no crash incidents.

## 7.4 Operational readiness

- On-call runbook reviewed.
- Rollback drill performed and timed.
- Alert rules configured for new counters.

---

## 8. Failure modes and mitigation matrix

| Risk | Detection | Mitigation | Rollback |
|---|---|---|---|
| Model/layout drift | Startup compatibility check fails | Soft-disable MoLR, log explicit mismatch | Keep running full experts |
| Schema drift in bundle | Manifest parser/version error | Versioned schema + backward-compatible parser where possible | Soft-disable or strict fail per config |
| Telemetry overhead | Throughput/latency regression in off mode | Aggregate counters only, no high-cardinality labels | Disable metrics endpoint if needed |
| Quality regression from thresholding | Shadow A/B divergence + pilot quality alarms | Recalibrate thresholds, adjust quality profile defaults | Switch mode to off |
| Import/config ambiguity | Conflicting flags at startup | Deterministic precedence + explicit warnings/errors | Fall back to `off` effective mode |

---

## 9. Rollout gates and promotion criteria

Promotion path follows the suggested order from handoff:

1. **Config + bundle validation** (M1)
2. **Telemetry + shadow** (M2 + M4)
3. **Guarded forward hook** (M3, still default-off)
4. **Limited pilot** (M5 initial)
5. **Gradual expansion** (M5 extended)

Hard stop conditions (auto-disable to `off`):
- sustained fallback rate above configured cap,
- sustained MoLR runtime error rate above cap,
- severe latency regression breach,
- any correctness incident attributable to MoLR path.

---

## 10. Small-PR execution plan (suggested sequence)

1. **PR-1:** `common_params` + arg/env flags + docs skeleton (no behavior change).
2. **PR-2:** manifest validator module + startup integration + `/props` status.
3. **PR-3:** telemetry structs + `/metrics` export + tests.
4. **PR-4:** guarded MoE hook with fallback-only dry path (active not serving yet).
5. **PR-5:** shadow mode sampling + A/B controls + comparison metrics.
6. **PR-6:** active-mode serving enable + auto-disable guardrails + runbook.

Each PR should include:
- clear invariant checks,
- explicit default-off proof,
- targeted test updates,
- operational notes for rollout/rollback.

---

## 11. Compact operator runbook (initial)

Enable (shadow):
1. Set `--molr-mode shadow --molr-bundle <path> --metrics`.
2. Verify `/props` shows effective mode `shadow` and bundle `compatible`.
3. Monitor `molr_shadow_eval_total`, `molr_runtime_error_total`, latency metrics.

Promote (active pilot):
1. Set `--molr-mode active --molr-quality-profile <profile>`.
2. Confirm fallback/error/latency within gate windows.

Rollback:
1. Immediate: set `--molr-mode off` and restart process.
2. If needed: repoint to last-known-good manifest and retest in shadow.
3. If incident persists: roll binary back to previous release.

---

## 12. Final handoff decisions summary

- **Opt-in only** MoLR runtime controls are centralized in shared arg/config plumbing.
- **Startup compatibility gating** is mandatory; invalid bundles never break serving.
- **Forward hook is strictly guarded** and always has a full-expert fallback path.
- **Telemetry precedes active rollout** to de-risk performance/quality regressions.
- **Shadow → pilot → expand** is the required promotion path with explicit kill switches.
