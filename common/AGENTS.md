# Notes for `common/`

- In `common/arg.cpp`, `common_arg::get_env()` auto-generates a negated env var only for names starting with `LLAMA_ARG_`; if a boolean option with `args_neg` uses a custom env name (e.g. `LLAMA_MOE_TRACE_ENABLE`), it will duplicate the same env key and break `test-arg-parser` duplicate-env checks.
- MoE trace CLI is intentionally fail-soft unless `--moe-trace-strict` is enabled: invalid format/precision/range/path should warn and continue, with strict mode upgrading those warnings to init errors.
