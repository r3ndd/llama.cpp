# Algorithm 2 Revision Notes — Residual Table → Removed-Expert Contribution Table

Date: 2026-04-10 16:07  
Applies to:
- Plan: `.conductor/plans/2026-04-10-1917-expert-lookup-table-plan.md`
- Design: `.conductor/designs/2026-04-10-1942-expert-lookup-table-design.md`

---

## 1) Purpose

This note documents the semantic change from the prior Algorithm 2 definition (residual-table based) to the new Algorithm 2 definition requested for the Qwen3.5-MoE PoC.

---

## 2) Superseded vs Current Algorithm 2 Definition

## 2.1 Superseded (Old)

- Per key, table stored average residual `r = y_full - y_kept`.
- In inference, runtime looked up residual and added it (often with router-mass scaling options).

## 2.2 Current (New)

- Per key, table stores average **router score-weighted output of removed top-k experts**, where weights are normalized **only among removed experts**.
- Token-level training target for layer `L`:
  - Let `M` be removed experts that are also in token's selected top-k.
  - If `M` is empty: target is zero vector.
  - Else:
    - `u = sum_{e in M} (w_e / sum_{j in M} w_j) * y_e`
    - with router scores `w_e` from selected top-k, and per-expert outputs `y_e` captured separately.
- Per key table entry becomes `U_L[key] = mean(u)`.
- Inference adds `s_missing * U_L[key]`, where `s_missing = sum_{e in M_current} w_e` for current token routing.

---

## 3) Why This Revision Matters

1. **Target semantics are more direct**: table models removed experts’ own contribution prototype, not a residual that entangles kept-expert behavior.
2. **Cleaner decomposition**: relative mixture (`u`) captures shape; runtime net mass (`s_missing`) captures magnitude under current routing.
3. **Zero-missing correctness**: no removed selected experts naturally yields zero table contribution.

---

## 4) Documentation Changes Applied

## 4.1 Plan updates

- Updated Algorithm 2 mechanics section to mark old residual definition as superseded and new definition as current.
- Updated Stage B trace requirements to explicitly require per-top-k expert outputs.
- Updated Stage C builder text from residual averaging to removed-expert contribution averaging.
- Updated NPZ schema to require:
  - `h_pre_moe`
  - `topk_ids`
  - `topk_weights`
  - `topk_expert_outputs` (shape `[N, k, n_embd]`)
- Updated risk and deferred-item language from residual-centric to contribution-centric semantics.

## 4.2 Design updates

- Replaced residual language across architecture/data-flow/inference sections with contribution-prototype semantics.
- Trace schema updated to require `topk_expert_outputs` and remove residual-specific required fields.
- Sidecar payload renamed from residual table `R_L` to contribution table `U_L`.
- Inference path updated to mandatory net-mass scaling with `s_missing`.
- Stage/checklist text aligned with revised builder/tracing requirements.

---

## 5) Revised Core Equations (Canonical)

For sample `i` at layer `L`:

- Removed-selected set: `M_i = topk_i ∩ replaced_L`
- Missing mass: `s_i = sum_{e in M_i} w_{i,e}`
- Contribution prototype target:
  - if `|M_i| = 0`, `u_i = 0`
  - else `u_i = sum_{e in M_i} (w_{i,e}/s_i) * y_{i,e}`

For key/cluster `c`:
- `U_L[c] = mean_{i: key(i)=c}(u_i)`

Inference output at layer/token:
- `y = y_kept + s_current * U_L[key(h_pre_moe)]`

---

## 6) Remaining Ambiguities (Post-Revision)

1. **Per-expert output capture point**
   - Precisely define whether `y_e` is captured before/after any expert-local scaling/normalization details in implementation.

2. **Scale calibration policy**
   - Base scaling by `s_missing` is mandated; decide if optional global/per-layer calibrated multiplier is needed.

3. **Table behavior for sparse removed-hit clusters**
   - Decide min-sample thresholds and smoothing/backoff when many cluster assignments have low removed-hit support.

4. **Centroid precision choice (`fp16` vs `fp32`)**
   - Finalize based on key-match fidelity vs memory/speed.

5. **Sidecar composition of replacement map**
   - Keep replacement map separate JSON vs embed into binary sidecar header/payload for single-file deployment.

---

## 7) Consistency Statement

The plan and design documents are now aligned to the **new Algorithm 2 variant** and explicitly mark the old residual-based definition as superseded.
