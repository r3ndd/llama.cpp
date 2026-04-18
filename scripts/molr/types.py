from __future__ import annotations

SVD_REPORT_SCHEMA_VERSION = "1.1"

MOLR_PLAN_SCHEMA_VERSION = "molr_plan.v1"
MOLR_COVARIANCE_SUMMARY_SCHEMA_VERSION = "molr_covariance_summary.v1"
MOLR_COVARIANCE_NPZ_SCHEMA_VERSION = "molr_covariance_npz.v1"
MOLR_ROUTED_TRACES_NPZ_SCHEMA_VERSION = "molr_routed_traces_npz.v1"
MOLR_EXPERT_WEIGHTS_NPZ_SCHEMA_VERSION = "molr_expert_weights_npz.v1"
MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION = "molr_expert_checkpoint_npz.v1"
MOLR_EXPERT_VALIDATION_SCHEMA_VERSION = "molr_expert_validation.v1"
MOLR_VALIDATION_REPORT_SCHEMA_VERSION = "molr_validation_report.v1"
MOLR_FAILURE_LEDGER_SCHEMA_VERSION = "molr_failure_ledger.v1"
MOLR_THRESHOLDS_SCHEMA_VERSION = "molr_thresholds.v1"
MOLR_BUNDLE_MANIFEST_SCHEMA_VERSION = "molr_bundle_manifest.v1"
MOLR_RUNTIME_CONFIG_SCHEMA_VERSION = "molr_runtime_config.v1"
MOLR_RUNTIME_TELEMETRY_SCHEMA_VERSION = "molr_runtime_telemetry.v1"
MOLR_RUNTIME_SHADOW_REPORT_SCHEMA_VERSION = "molr_runtime_shadow_report.v1"

# Retry schedule for Cholesky factorization when covariance is near-singular.
CHOLESKY_JITTER_SCHEDULE = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
