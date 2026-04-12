## Tests learnings

- After editing `tests/CMakeLists.txt` (e.g., adding a new `llama_test_cmd`), re-run CMake configure for existing build dirs before running `ctest`; otherwise the new test target/name is not registered.
