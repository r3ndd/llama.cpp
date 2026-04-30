# scripts agent notes

## imatrix calibration generation

- `imatrix_calibration_generate.py` must handle OpenAI-compatible chat responses where `choices[0].message.content` is empty but `choices[0].message.reasoning_content` contains the generated text (seen with some reasoning-capable models).
- Prompt JSONL sampling fields should use `repetition_penalty`; keep it in `PASS_THROUGH_FIELDS` (alongside any legacy aliases) so generation payloads preserve the intended penalty settings.

## MoE covariance analysis (`analyze_moe_covariance.py`)

- Enumerate analysis targets from metadata (`moe_cov.target_count` + `moe_cov.target.t<idx>.*`) and then read `moe_cov.t<idx>.n` / `cov_pop`; do not infer target IDs by scanning tensor names.
- When turning GGUF tensors into matrices, derive logical dimensions from `reversed(tensor.shape.tolist())` before reshaping numeric data.
- Treat missing `cov_pop` as unavailable (`NaN`), not `0.0`, so aggregate means are not biased by absent tensors.
- For coverage accounting, derive expected layer/expert/target universe from the source model architecture (`discover_expert_matrices` + arch metadata), not covariance filter metadata alone.
- Normalize discovery roles (`w1/w2/w3`) to covariance roles (`gate/down/up`) before comparing coverage.
