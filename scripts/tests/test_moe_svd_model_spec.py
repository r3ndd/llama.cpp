from __future__ import annotations

import pytest

from moe_svd.model_resolver import ModelResolutionError, parse_model_spec


def test_parse_model_spec_valid() -> None:
    repo, filename = parse_model_spec("unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M")
    assert repo == "unsloth/Qwen3.5-35B-A3B-GGUF"
    assert filename == "Q4_K_M"


@pytest.mark.parametrize(
    "model_spec",
    [
        "missing-colon",
        "bad/repo/format:Q4_K_M",
        "repoonly:",
        ":file.gguf",
    ],
)
def test_parse_model_spec_invalid(model_spec: str) -> None:
    with pytest.raises(ModelResolutionError):
        parse_model_spec(model_spec)
