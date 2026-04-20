# Technical Design: MoLR LLM Compresson (Pilot on Qwen3.5-35B-A3B GGUF)

Date: 2026-04-18
Status: Proposed (implementation-ready handoff)
Target pilot model: `unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M`

## 1) Scope, goals, and non-goals

## Goals
- Implement a maintainable, incremental MoLR pipeline that replaces routed MoE experts with a Mixture-of-Low-Rank (MoLR) approximator.
- Reuse existing spectral analysis tooling (`scripts/analyze_moe_svd.py`) as the canonical entry for expert discovery and SVD diagnostics.
- Deliver a first end-to-end pilot for Qwen3.5-35B-A3B GGUF Q4_K_M, including:
  - offline preparation,
  - per-expert MoLR training,
  - calibration for optional fallback,
  - runtime integration plan with staged rollout.

## Non-goals (pilot)
- No immediate default-on replacement in llama.cpp production inference path.
- No new quantization type additions in ggml.
- No multi-model generalization work beyond architecture hooks needed for Qwen3.5.

---

## 2) Design principles and key decisions

1. **Incremental by phase**: deliver offline pipeline first, then shadow inference validation, then guarded runtime use.
2. **Reuse before rewrite**: `analyze_moe_svd.py` remains the discovery + spectral baseline tool; MoLR planning consumes its output.
3. **Artifact-first pipeline**: each stage emits versioned artifacts (JSON/NPZ/checkpoints) to keep runs reproducible and debuggable.
4. **Per-expert isolation**: experts train independently to simplify failure containment and parallelism.
5. **Quality-gated rollout**: fallback and threshold calibration are mandatory before any runtime enablement.

Tradeoff: this adds pipeline complexity (more artifacts/scripts), but sharply improves correctness, observability, and rollback safety.

---

## 3) End-to-end architecture and data flow

## 3.1 High-level flow
1. **Spectral profiling** (existing `analyze_moe_svd.py`)
2. **MoLR planning** (rank/K assignments, strided init plan)
3. **Covariance capture** (real-token routed inputs per expert)
4. **Synthetic training set generation** (on-demand Gaussian sampling from `(mu, L)`)
5. **Per-expert MoLR training** (components + router + error head)
6. **Per-expert validation + calibration** (quality metrics + threshold sweep)
7. **Packaging/export** (MoLR expert bundle + calibration table)
8. **Inference integration (gated)** with optional fallback to full expert

## 3.2 Artifacts and contracts
- `svd_report.json` (existing schema from `analyze_moe_svd.py`)
- `molr_plan.json` (new): per-layer/expert/matrix `rank`, `K`, initialization partition metadata
- `covariance_stats.npz` (new): per expert `{mu, L, sample_count}`
- `molr_expert_{layer}_{expert}.pt` (new): trained component/router/error-head params
- `molr_validation.json` (new): cosine/rel-norm/router-entropy/error-corr + pass/fail
- `molr_thresholds.json` (new): threshold -> {fallback_rate, quality}
- `molr_bundle_manifest.json` (new): final packaged model metadata, versions, compatibility flags

All artifacts are immutable per run-id (`<timestamp>-<gitsha>`), enabling rollback by manifest switch.

---

## 4) Module boundaries and file layout

## 4.1 Reused existing modules
- `scripts/analyze_moe_svd.py` (existing): discovery + SVD metrics + report.
- `scripts/moe_svd/*` (existing): GGUF discovery, matrix dequantization, metrics, reporting.

## 4.2 New modules (offline MoLR pipeline)
Proposed under `scripts/molr/`:

1. `scripts/molr/plan_from_svd.py`
   - Input: `svd_report.json`
   - Output: `molr_plan.json`
   - Responsibility: choose per-matrix rank to meet target energy (default 0.90), assign K, build strided SVD partition map.

2. `scripts/molr/capture_expert_covariance.py`
   - Input: model spec, token sample source, routing config
   - Output: `covariance_stats.npz`
   - Responsibility: run full model on 10k–50k tokens, capture routed inputs per expert, compute `(mu, L)`.

3. `scripts/molr/train_expert_molr.py`
   - Input: one expert full weights + `molr_plan.json` + `covariance_stats.npz`
   - Output: expert checkpoint + expert validation summary
   - Responsibility: train one expert MoLR with synthetic batches.

4. `scripts/molr/train_all_experts.py`
   - Input: same as above
   - Output: all expert checkpoints + merged validation report
   - Responsibility: orchestrate embarrassingly parallel training with retry/failure accounting.

5. `scripts/molr/calibrate_fallback.py`
   - Input: trained experts + validation sets
   - Output: `molr_thresholds.json`
   - Responsibility: threshold sweep and quality/fallback tradeoff table.

6. `scripts/molr/package_molr_bundle.py`
   - Input: checkpoints + plan + calibration
   - Output: `molr_bundle_manifest.json` (+ parameter shards)
   - Responsibility: normalize naming, checksum all files, produce runtime-consumable bundle.

## 4.3 Runtime integration modules (gated)
Initial integration should be behind an explicit opt-in flag in inference path:

- `src/llama-model-loader.*` (or equivalent model loading path): load optional MoLR bundle metadata.
- `src/llama-moe-expert.*` (or equivalent expert forward path): branch full-expert vs MoLR-expert execution.
- `tools/server` config path: expose quality mode / threshold selection and fallback monitoring counters.

Exact C/C++ file names can vary with current branch layout; implementers should map to existing MoE expert dispatch points.

---

## 5) Interface/CLI specifications

## 5.1 Existing command (must be reused)
```bash
python scripts/analyze_moe_svd.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --out-json "<run>/svd_report.json" \
  --full-svd \
  --workers 4 \
  --blas-threads 1
```

## 5.2 New planning command
```bash
python scripts/molr/plan_from_svd.py \
  --svd-json "<run>/svd_report.json" \
  --target-energy 0.90 \
  --k-components 4 \
  --out-json "<run>/molr_plan.json"
```

`molr_plan.json` (normative fields):
- `schema_version`
- `model_spec`
- `target_energy`
- `default_k`
- `experts[]`:
  - `layer`, `expert`
  - `matrices[]`: `{role, shape, rank, fro_norm, energy_curve_ref}`
  - `init_partition`: `{strategy: "strided", component_assignments}`

## 5.3 Covariance capture command
```bash
python scripts/molr/capture_expert_covariance.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --tokens 50000 \
  --out-npz "<run>/covariance_stats.npz" \
  --out-json "<run>/covariance_summary.json"
```

## 5.4 Training commands
```bash
python scripts/molr/train_all_experts.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --plan-json "<run>/molr_plan.json" \
  --cov-npz "<run>/covariance_stats.npz" \
  --steps 20000 \
  --batch-size 512 \
  --lr 1e-4 \
  --out-dir "<run>/checkpoints"
```

## 5.5 Calibration and packaging
```bash
python scripts/molr/calibrate_fallback.py \
  --checkpoints "<run>/checkpoints" \
  --out-json "<run>/molr_thresholds.json"

python scripts/molr/package_molr_bundle.py \
  --model "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M" \
  --plan-json "<run>/molr_plan.json" \
  --checkpoints "<run>/checkpoints" \
  --thresholds-json "<run>/molr_thresholds.json" \
  --out-dir "<run>/bundle"
```

---

## 6) Detailed technical behavior

## 6.1 Rank selection and SVD reuse
- Primary source of matrix inventory and compressibility baseline: `analyze_moe_svd.py` output.
- Rank policy: smallest `r` such that explained spectral energy >= target (default 0.90).
- Since existing report currently stores energy at fixed rank fractions, **pilot uses nearest-above fraction**.
- Optional enhancement (recommended in Phase 1): add sidecar spectrum output in `analyze_moe_svd.py` to enable exact per-rank cutoff without re-running SVD elsewhere.

Rationale: avoids redundant tensor discovery/dequantization logic and leverages already validated MoE packed-tensor handling.

## 6.2 Initialization
- Build low-rank factors from SVD-derived components.
- Apply **strided singular vector assignment** across `K` components.
- Equalize per-component Frobenius norm after assignment.

## 6.3 Training objective
`L_total = L_mse + λ_lb * L_load_balance + λ_err * L_error_head`
- `L_mse`: MSE(MoLR(x), Expert(x))
- `L_load_balance`: sum of squared batch-mean router weights
- `L_error_head`: MSE(pred_error(x), ||MoLR(x)-Expert(x)||2_detached)
- Defaults: `λ_lb=0.01`, `λ_err=0.05`

## 6.4 Inference decision logic (runtime)
Per routed expert call:
1. Compute error-head scalar from input `x`.
2. If `pred_error > threshold`: execute full expert (RAM cache first, then SSD-backed load).
3. Else execute all K components and weighted sum.

Telemetry counters (must exist):
- per-expert fallback count/rate
- per-expert average predicted error
- per-expert runtime latency (MoLR vs full fallback)

---

## 7) Failure modes and handling

## Offline pipeline
- Missing/invalid SVD report -> hard fail in planning stage.
- Missing covariance for expert -> expert marked `train_skipped_missing_cov`, continue others.
- Cholesky failure (non-PD covariance) -> add jitter schedule (`1e-6` to `1e-3`) then retry; if still fail, mark expert failed.
- Divergent training (NaN/Inf loss) -> checkpoint rollback to last good state, reduce LR, retry once.

## Runtime
- Missing MoLR bundle for expert -> fallback to full expert (no crash).
- MoLR parameter load failure -> disable MoLR for that expert and emit warning.
- High fallback rate above alert threshold -> auto-demote to full expert for session.

---

## 8) Implementation phases, milestones, and checklist

## Phase 0 — Baseline spectral artifact and pilot envelope
**Milestone:** reproducible SVD baseline for Qwen3.5 pilot model.

Checklist:
- [ ] Run `scripts/analyze_moe_svd.py` on `unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M` with `--full-svd` and `--workers 16`.
- [ ] Archive `svd_report.json` and run metadata.
- [ ] Confirm candidate expert matrix coverage is non-zero and plausible.
- [ ] Document quantization caveat (Q4_K_M spectral bias).

## Phase 1 — MoLR planning and covariance capture
**Milestone:** produce `molr_plan.json` + `covariance_stats.npz`.

Checklist:
- [ ] Implement `plan_from_svd.py` consuming existing SVD report schema.
- [ ] Generate per-expert/matrix rank decisions and component partition map.
- [ ] Implement covariance capture from routed real-token passes (10k–50k tokens).
- [ ] Validate Cholesky factors for all experts or mark explicit failures.
- [ ] Freeze schema versions for plan/covariance artifacts.

## Phase 2 — Per-expert MoLR training pipeline
**Milestone:** train MoLR checkpoints for all experts with validation metrics.

Checklist:
- [ ] Implement single-expert training and multi-expert orchestrator.
- [ ] Enforce frozen full-expert weights and detached true-error supervision.
- [ ] Emit per-expert metrics: cosine, relative norm error, router entropy, error-corr.
- [ ] Flag weak experts via thresholds (`cos<0.95`, `error-corr<0.7`).
- [ ] Produce merged validation report and failure ledger.

## Phase 3 — Fallback calibration and bundle packaging
**Milestone:** runtime-consumable MoLR bundle with calibrated thresholds.

Checklist:
- [ ] Sweep thresholds and build quality-vs-fallback lookup table.
- [ ] Identify poor experts for full-weight cache candidates.
- [ ] Package checkpoints + manifests + checksums.
- [ ] Add compatibility/version checks in manifest.

## Phase 4 — Runtime integration (opt-in) and shadow rollout
**Milestone:** guarded inference path with monitoring and rollback.

Checklist:
- [ ] Add opt-in config (`--molr-bundle`, `--molr-quality-profile` or explicit threshold).
- [ ] Integrate pre-check error head and fallback branch in expert forward.
- [ ] Add per-expert fallback + latency counters to logs/metrics.
- [ ] Run A/B shadow evaluation against full expert baseline.
- [ ] Define auto-disable conditions and manifest-based rollback path.

---

## 9) Validation strategy

## 9.1 Unit validation
- Rank-selection logic from spectral curves/fractions.
- Strided component assignment correctness and norm equalization.
- Covariance-to-Cholesky sampling correctness.
- Loss decomposition and detached error-target behavior.

## 9.2 Integration validation
- End-to-end dry run with limited experts (`--max-experts`, short steps).
- Full pilot run producing all artifacts and complete manifest.
- Determinism checks on planning outputs and stable ordering.

## 9.3 Quality and performance gates
- Per-expert quality gates:
  - cosine similarity mean >= 0.95,
  - relative norm error below configured threshold,
  - error-head Pearson r >= 0.7.
- System gates (pilot quality profile):
  - fallback rate within preselected target range,
  - perplexity delta vs baseline within agreed budget,
  - latency impact acceptable at chosen threshold.

---

## 10) Risks, tradeoffs, and mitigations

1. **Quantized source weights (Q4_K_M) distort spectra**
   Mitigation: keep caveat in every artifact; later compare against higher-precision checkpoint where possible.

2. **Coarse rank-fraction grid in current SVD report**
   Mitigation: nearest-above fraction in pilot; optional sidecar spectrum export enhancement.

3. **Covariance mismatch to production traffic**
   Mitigation: monitor runtime fallback rates; retrain/recalibrate from observed data.

4. **Runtime complexity from fallback/cache paths**
   Mitigation: start opt-in only; full-expert path remains default-safe fallback.

5. **Expert collapse in router**
   Mitigation: load-balance term + entropy monitoring + initialization norm equalization.

---

## 11) Rollout and rollback plan (pilot model)

1. **Offline-only completion**: all artifacts generated for Qwen3.5 pilot.
2. **Shadow mode**: execute MoLR path and full path side-by-side on sampled traffic; compare outputs/metrics without serving MoLR output.
3. **Limited opt-in**: enable MoLR for explicit test workloads with conservative threshold.
4. **Quality profile exposure**: map user quality levels to calibrated thresholds.
5. **Rollback**: single config switch disables MoLR bundle usage; inference reverts to full experts only.

---

## 12) Files to modify for implementation

Existing files:
- `scripts/analyze_moe_svd.py` (reuse as baseline; optional enhancement for richer spectral export)
- `scripts/moe_svd/types.py` (only if adding optional spectra sidecar metadata)
- `scripts/moe_svd/reporting.py` (only if new report fields are added)

New files:
- `scripts/molr/plan_from_svd.py`
- `scripts/molr/capture_expert_covariance.py`
- `scripts/molr/train_expert_molr.py`
- `scripts/molr/train_all_experts.py`
- `scripts/molr/calibrate_fallback.py`
- `scripts/molr/package_molr_bundle.py`
- `scripts/molr/types.py` (artifact schemas)
- `scripts/molr/README.md` (operator guide)

Runtime integration touchpoints (phase-gated):
- MoE expert forward path and model loader path in `src/`.
- Optional server CLI/config under `tools/server/`.

---

## 13) Handoff summary (implementation intent)

- Keep `analyze_moe_svd.py` as the authoritative first-stage pipeline input.
- Build MoLR as an artifact-driven offline workflow with explicit schemas and per-phase checkpoints.
- Validate each expert independently with hard quality gates before packaging.
- Introduce runtime support only behind opt-in flags, with telemetry and immediate rollback safety.
- Start with `unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M`, acknowledging quantization-related spectrum bias in evaluation.
