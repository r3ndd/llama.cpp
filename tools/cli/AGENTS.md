# Notes for `tools/cli/`

- `llama-cli` MoE trace behavior should stay plumbing-only: flag parsing/lifetime/validation belongs in `common/`, while runtime capture/join logic belongs in `src/`.
- `--moe-trace-path` implies trace enablement (same as env `LLAMA_MOE_TRACE_JSONL`), so CLI UX/tests should treat path-only config as an active trace request.
