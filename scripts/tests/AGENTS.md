# scripts/tests agent notes

- Run MoE-SVD script tests with `PYTHONPATH=<repo>/scripts` so imports like `from moe_svd...` resolve during pytest collection.
- Keep a regression case for the `imatrix_calibration_generate` fallback where `message.content` is empty and `message.reasoning_content` carries output.
