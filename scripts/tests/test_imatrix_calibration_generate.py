from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imatrix_calibration_generate import parse_generated_text


def test_parse_generated_text_prefers_content_when_present() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": "final answer",
                    "reasoning_content": "internal reasoning",
                }
            }
        ]
    }

    assert parse_generated_text(response) == "final answer"


def test_parse_generated_text_falls_back_to_reasoning_content() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "answer emitted in reasoning field",
                }
            }
        ]
    }

    assert parse_generated_text(response) == "answer emitted in reasoning field"
