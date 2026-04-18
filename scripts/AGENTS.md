# Notes for `scripts/`

- `scripts` Python packages (`molr`, `moe_svd`) are imported via `PYTHONPATH=scripts`; run script-focused pytest from repo root with that env so collection/imports resolve consistently.
- `scripts/molr` planning is coupled to `scripts/moe_svd/svd_metrics.py` rank-fraction constants; treat cross-package metric-grid changes as contract changes, not local refactors.
