# Research Handoff: Best Approach to Generating ELT1 Sidecars

Date: 2026-04-12
Scope: Bridge current NPZ offline builder outputs to runtime-required ELT1 sidecar for end-to-end MoE lookup usage.

## Context

- Goal from design docs: keep NPZ as trace/training format, but use a versioned binary sidecar at inference for speed + strict validation.
- Runtime now already loads **ELT1** binary sidecars (`magic="ELT1"`, `format_version=1`) via `src/moe-lookup.cpp`.
- Current offline builder (`scripts/build-moe-lookup.py`) still emits lookup **NPZ** + replaced-experts JSON.

## Observed Evidence (repo facts)

1. **Runtime expects ELT1 binary, not NPZ**
   - `src/moe-lookup.cpp:17` defines magic `0x31544c45` (`"ELT1"`).
   - `src/moe-lookup.h:22-36` defines binary v1 header schema.
   - `src/moe-lookup.cpp:93-331` reads binary header/payload, validates dimensions/model identity, and rejects unsupported format/dtype/scaling.

2. **Builder emits NPZ lookup artifact today**
   - `scripts/build-moe-lookup.py:5-8` says outputs are NPZ + JSON.
   - `scripts/build-moe-lookup.py:740-741` writes `np.savez_compressed(...)`.
   - `docs/development/moe-lookup-tool.md:92` documents lookup sidecar NPZ.

3. **Runtime fallback behavior is soft-disable (robust)**
   - `src/llama-context.cpp:225-245` loads lookup table; on failure logs warning and disables MoE lookup, continues inference.
   - `src/llama-context.cpp:236-239` disables if no active layers.

4. **Model gating is explicit (Qwen3.5-MoE only)**
   - `src/llama-context.cpp:210-214` disables `--moe-lookup-*` and `--moe-trace-*` for non-Qwen3.5-MoE.

5. **Validation contract is already strict in loader**
   - Header checks include `n_layer`, `n_embd`, `n_expert`, `n_expert_used`, `model_id`, vector dtype fp16, scaling mode `s_missing` (`src/moe-lookup.cpp:145-178,135-142`).
   - Per-layer safety checks include duplicate/invalid layers, invalid replaced IDs, non-finite centroid values, trailing bytes (`src/moe-lookup.cpp:188-307`).

6. **Testing exists for builder and basic layer validity, but no end-to-end NPZ→ELT1 path yet**
   - Builder tests: `tests/test-build-moe-lookup.py`.
   - Loader-layer validity tests only: `tests/test-moe-lookup.cpp`.
   - No test currently validates converting builder NPZ output into ELT1 and loading it in runtime.

7. **Non-obvious mismatch: replaced-experts file is required by CLI/runtime load path but not consumed by loader data path**
   - `src/moe-lookup.cpp:72-89` requires and opens `--moe-lookup-replaced-experts` file.
   - Actual replaced masks are loaded from sidecar layer payload (`src/moe-lookup.cpp:230-261`), not from parsed JSON.

## Viable Approaches

### Approach A — Extend builder to emit ELT1 natively (single-stage output)

**What**
- Add binary ELT1 writer directly in `scripts/build-moe-lookup.py` (or split into a builder module + writer module).
- Keep NPZ optional for diagnostics only.

**Pros**
- Fewer moving parts at user level (single command, direct runtime-ready artifact).
- Can enforce schema at creation time.

**Cons / Risks**
- Large mixed responsibility in one script (training + clustering + binary serialization).
- Harder to iterate schema independently from training logic.
- Higher chance of regressions in current tested NPZ flow.

**Implication**
- Good long-term simplification candidate, but higher immediate integration risk.

---

### Approach B — Add explicit NPZ→ELT1 converter (two-stage pipeline)

**What**
- Keep current builder unchanged as canonical training output.
- Add a new converter tool (e.g., `scripts/convert-moe-lookup-to-elt1.py`) that:
  1) reads builder NPZ + replaced-experts JSON,
  2) validates required fields against ELT1 constraints,
  3) writes binary ELT1 payload matching `llama_moe_lookup_header_v1` and layer payload ordering.

**Pros**
- Minimal disruption to existing builder/tests.
- Clean separation: training math vs runtime packaging.
- Enables deterministic format migrations (v1 NPZ can be re-packed repeatedly).
- Fastest practical bridge to end-to-end because runtime loader already exists.

**Cons / Risks**
- Extra artifact step in workflow.
- Must keep converter schema rules in sync with runtime loader.

**Implication**
- Best balance of robustness and maintainability for current repo state.

---

### Approach C — Runtime NPZ support (load NPZ directly in C++)

**What**
- Add NPZ reader into runtime loader (parallel to ELT1 path).

**Pros**
- No converter step.
- Immediate compatibility with builder outputs.

**Cons / Risks**
- Pulls Python-side experimental schema into C++ runtime contract.
- Increases runtime complexity and attack surface (ZIP/NPY parsing, shape handling).
- Undermines current strict binary validation intent and future format discipline.

**Implication**
- Useful only as very short-lived bring-up fallback; poor maintainability.

## Recommendation

### Primary approach: **Approach B (NPZ→ELT1 converter), then optionally absorb into builder later**

Rationale:
- Aligns with current architecture decisions in design docs (NPZ trace/training; binary inference sidecar).
- Uses already-implemented strict loader validation (`src/moe-lookup.cpp`) rather than bypassing it.
- De-risks by isolating format conversion from clustering logic.

### Transition fallback: **Temporary runtime NPZ support only behind experimental flag (if unblock is urgent)**

- Only if converter blocks progress.
- Must be explicitly temporary and removed once converter is stable.

## Schema / Versioning Guidance

1. **Treat ELT1 as authoritative runtime schema v1**
   - Match `llama_moe_lookup_header_v1` and `llama_moe_lookup_layer_header_v1` exactly (`src/moe-lookup.h`).

2. **Version separately by layer**
   - Keep converter input schema pinned to builder `format_version=1` (`scripts/build-moe-lookup.py:675,720`).
   - Emit ELT1 `format_version=1` for runtime.

3. **Compatibility rules**
   - Converter should hard-fail on NPZ missing keys: `layers`, `n_embd`, `n_expert`, layer centroids/contributions.
   - Converter should preserve model constraints via side metadata inputs (model id + dims), not guess.

4. **Model ID source-of-truth**
   - Runtime checks `model.arch_name()` against sidecar `model_id` (`src/moe-lookup.cpp:173-177`).
   - Converter must require explicit `--model-id` (or derive from trace metadata with strict validation) to avoid mismatches.

## Validation Guarantees to Enforce

Converter must validate before writing ELT1:
- Every emitted layer has matching centroid/contribution shapes `[n_keys, n_embd]`.
- `replaced_count <= n_expert - n_expert_used` to satisfy fill-safe constraint mirrored by loader (`src/moe-lookup.cpp:263-269`).
- All centroid values finite (avoid runtime skip).
- No duplicate or out-of-range replaced IDs.
- Layer ids unique and in range.

Runtime already guarantees:
- Header/schema mismatch triggers lookup disable + baseline fallback.
- Partial sidecar loads with warnings, preserving inference continuity.

## Testing Strategy

1. **Unit tests (Python converter)**
- Golden NPZ + JSON fixture -> byte-level ELT1 parse assertions.
- Negative tests: bad dims, missing arrays, out-of-range expert IDs, non-finite centroids.

2. **C++ loader tests (expand `tests/test-moe-lookup.cpp`)**
- Add tests that load small real ELT1 fixtures and verify:
  - valid load,
  - rejected magic/version,
  - rejected mismatch dims/model,
  - partial-layer warning behavior.

3. **Pipeline integration test (new test target)**
- Trace fixture NPZ -> builder NPZ -> converter ELT1 -> loader success in a minimal context.
- Assert fallback path when ELT1 intentionally corrupted.

4. **Smoke runtime test**
- `llama-cli` with Qwen3.5-MoE flags and tiny synthetic sidecar to confirm enable/disable logs and no crash.

## Migration / Rollout Sequence (concrete in-repo)

1. **Stage 1: Converter introduction (no runtime change)**
- Add `scripts/convert-moe-lookup-to-elt1.py`.
- Add doc section in `docs/development/moe-lookup-tool.md` for two-stage flow.
- Add Python tests in `tests/test-build-moe-lookup.py` (or split new `test-convert-moe-lookup.py`).

2. **Stage 2: Fixture-backed loader tests**
- Add ELT1 test fixtures + new loader tests in `tests/test-moe-lookup.cpp`.

3. **Stage 3: End-to-end command recipe + CI wiring**
- Add a reproducible mini pipeline command in docs:
  - trace NPZ generation
  - builder NPZ generation
  - converter ELT1 generation
  - runtime invocation with `--moe-lookup-enable --moe-lookup-file ...`

4. **Stage 4: Optional consolidation**
- After stability, optionally add `--output-format {npz,elt1,both}` to builder and internally call converter logic.

## Open Questions

1. Should replaced-experts JSON remain a required runtime flag if replaced IDs are fully encoded in ELT1 payload? (Current behavior suggests possible redundancy.)
2. Should converter output include deterministic layer ordering and checksum metadata for reproducibility audits?
3. Is `model_id` in converter best sourced from trace metadata or explicit CLI arg only?
4. Do we want an ELT1 schema doc in `docs/development/` mirroring `moe-lookup.h` to prevent drift?

---

## Compact Handoff Summary (Decision + Rationale + Next Actions)

**Decision:** Implement a dedicated **NPZ→ELT1 converter** as the primary bridge, keep runtime ELT1 loader as-is, and avoid adding long-term runtime NPZ parsing.

**Rationale:** Current repo already has strict ELT1 loader validation/fallback and a working NPZ builder. A converter is the lowest-risk path to end-to-end operation while preserving maintainable separation between training outputs and runtime packaging.

**Recommended next actions:**
1. Add `scripts/convert-moe-lookup-to-elt1.py` with strict pre-write validation mapped to `src/moe-lookup.h`.
2. Add converter + loader fixture tests (positive and negative cases) to close end-to-end coverage gap.
3. Update `docs/development/moe-lookup-tool.md` with canonical two-step generation flow and runtime invocation examples.
