# Research Handoff: MoE Lookup Vector Mixing (2026-04-14)

## Context
Scope investigated: current repository implementation for MoE lookup table build + ELT1 conversion + runtime graph path (Qwen3.5-MoE).

Primary files inspected:
- `scripts/build-moe-lookup.py`
- `scripts/convert-moe-lookup-to-elt1.py`
- `src/moe-lookup.cpp`
- `src/llama-graph.cpp`
- `src/llama-context.cpp`
- `docs/development/moe-lookup-tool.md`
- tests: `tests/test-build-moe-lookup.py`, `tests/test-moe-lookup.cpp`, `tests/test-moe-trace.cpp`

---

## Direct answers to requested questions

### 1) Are retrieved vectors normalized when saved or before substitution? Can very small vectors be loaded/mixed?

**Answer:**
- **No rank/L2 normalization is applied to stored lookup contribution vectors** at build time or immediately before runtime substitution.
- **Yes, very small-magnitude vectors can be loaded and mixed** (subject only to finite/range checks and fp16 conversion).

**Evidence (observed):**
- Build-time targets are formed as removed-only weighted mixtures and then averaged per centroid:
  - `scripts/build-moe-lookup.py:615-621` computes normalized weights only **across replaced experts** (`w_e / sum_removed_w`) to form per-row target `u`.
  - `scripts/build-moe-lookup.py:570-582` averages target vectors into table entries; no vector norm normalization.
  - `scripts/build-moe-lookup.py:714-715` writes centroids/contributions as `float16`.
- Runtime uses looked-up contribution directly, then scales only by missing mass:
  - `src/llama-graph.cpp:1794-1799` retrieves nearest contribution vector and computes `lookup_scaled = lookup * s_missing`.
  - `src/llama-graph.cpp:1801` adds scaled lookup to MoE output.
  - No post-retrieval norm/clamp is applied to contribution magnitude.
- Loader validates finiteness but does not normalize vectors:
  - `src/moe-lookup.cpp:271-296` computes centroid L2 only for distance; it does not normalize centroids/contributions.
  - `scripts/convert-moe-lookup-to-elt1.py:235-244` validates finite/fp16-range only.

**Implication:** contribution magnitude quality is data-dependent; low-energy lookup vectors remain low-energy at inference (and may become negligible when `s_missing` is small).

---

### 2) How are no-replaced-expert instances (`replaced output vector = 0`) saved into table?

**Answer:**
- Rows with no replaced experts are explicitly encoded as **zero target vectors** and **included** in clustering/aggregation.
- They are **not filtered out** before table construction.

**Evidence (observed):**
- `scripts/build-moe-lookup.py:595-621`
  - Initializes `targets` to zeros.
  - Computes `mass` of replaced experts per row.
  - Only rows with `mass > 0` get nonzero normalized mixture targets.
- `scripts/build-moe-lookup.py:710-717`
  - All rows are assigned to centroids (`assignments` from full `layer_h`).
  - `aggregate_contribution_table()` averages contributions over all assigned rows, including zero-target rows.
- Unit test confirms zero output behavior for no-removed case:
  - `tests/test-build-moe-lookup.py:79-98` expects `targets == [0,0]` and `mass == 0`.

**Implication:** if many rows in a cluster have no replaced experts, cluster contribution means are diluted toward zero.

---

### 3) During inference, if some experts are replaced, is effective top-k reduced or refilled to keep total k?

**Answer:**
- Runtime computes a **new routed top-k of the same width `n_expert_used`** after masking replaced experts with large negative bias.
- So effective compute top-k is **refilled** with next-best non-replaced experts (not reduced to `k - replaced`).

**Evidence (observed):**
- Baseline top-k first: `src/llama-graph.cpp:1403-1406` (`selected_experts`).
- Replaced experts masked by bias `-1e9`: `src/llama-graph.cpp:1431-1433`.
- Routed top-k recomputed with same `n_expert_used`: `src/llama-graph.cpp:1462-1463` (`selected_experts_routed = argsort_top_k(..., n_expert_used)`).
- Expert MLP path uses routed set: `src/llama-graph.cpp:1562`, `1694`.
- Missing-mass scaling still uses baseline selected set (`selected_experts`, `weights_base`): `src/llama-graph.cpp:1751-1759`.
- Fill-safety is enforced in conversion/load to guarantee enough non-replaced experts:
  - `scripts/convert-moe-lookup-to-elt1.py:208-250` requires `replaced_count <= n_expert - n_topk`.
  - `src/moe-lookup.cpp:263-269` skips invalid layer payload with same condition.

**Implication:** compute fan-out remains k when lookup is active; replacement changes **which** experts are computed, not the compute count.

---

## Caveats / doc-vs-behavior mismatches

1. **Stale doc statement:** `docs/development/moe-lookup-tool.md:10` says runtime integration is not required in this stage, but runtime integration is present in current `src/llama-graph.cpp` / `src/llama-context.cpp`.
2. Runtime still requires `--moe-lookup-replaced-experts` readable sidecar even though ELT1 carries `replaced_ids` (`src/moe-lookup.cpp:72-89`, doc note at `docs/development/moe-lookup-tool.md:189`).
3. Lookup path is currently gated to Qwen3.5-MoE (`src/llama-graph.cpp:1412`, `src/llama-context.cpp:211-214`).

---

## Practical implications for quality/stability

- **Magnitude sensitivity:** No explicit contribution normalization means lookup correction amplitude is driven by trace-derived table statistics and current `s_missing`; this can under-correct when contributions are tiny.
- **Zero-target dilution risk:** Including many `mass==0` rows in centroid means can push lookup vectors toward zero for frequent regions, lowering correction power.
- **Stable compute budget:** Routed top-k refill keeps MoE compute width stable (`k`), avoiding accidental quality/perf shifts from reduced expert count.
- **Safety checks are mostly structural, not semantic:** converter/loader enforce shape/finite/fill-safe invariants but do not constrain contribution energy distribution.

---

## Open questions

1. Is zero-target-row inclusion intentional for production-quality tables, or should clustering/aggregation be conditioned on `replaced_mass > 0`?
2. Should there be optional contribution post-processing (e.g., per-layer norm clipping or shrinkage) to reduce variance and tiny-vector collapse?
3. Should docs be updated to reflect current runtime integration status and exact top-k refill behavior?

---

## Compact final handoff summary

- **Q1:** No rank/vector normalization of lookup contributions at save/load/inference; only removed-only weight normalization during target construction and runtime `s_missing` scaling. Small vectors can be loaded/mixed.
- **Q2:** No-replaced rows become zero targets and are included in centroid averaging (not dropped).
- **Q3:** Inference keeps top-k width `k` by recomputing routed top-k from non-replaced experts (refill behavior), while `s_missing` is computed from baseline top-k.
