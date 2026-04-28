# tests agent notes

- Release test binaries may compile with `NDEBUG`; tests that must exercise `assert(...)` need `#undef NDEBUG` before including `<cassert>`.
- `tests/test-arg-parser.cpp` reuses a single `common_params`; reset with `params = common_params();` between sections to avoid state bleed between parse expectations.
