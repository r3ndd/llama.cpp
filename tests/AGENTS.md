# tests agent notes

- In Release builds, test targets may compile with `NDEBUG`; tests that rely on `assert(...)` must explicitly `#undef NDEBUG` before including `<cassert>`.
- `tests/test-arg-parser.cpp` reuses a single `common_params`; reset it (`params = common_params();`) between sections or earlier parse state can invalidate later expectations.
