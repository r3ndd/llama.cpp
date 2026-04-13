#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval-moe-lookup-matrix.py"


def load_module():
    spec = importlib.util.spec_from_file_location("eval_moe_lookup_matrix", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load eval-moe-lookup-matrix.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestEvalMoeLookupMatrix(unittest.TestCase):
    def test_discover_matrix_adds_baseline_lookup_and_remove_only_rows(self):
        mod = load_module()

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            lookup_npz = tmp / "lookup.npz"
            replaced_json = tmp / "lookup.replaced-experts.json"

            np.savez_compressed(
                lookup_npz,
                n_layer_total=np.asarray(2, dtype=np.int32),
                n_expert=np.asarray(8, dtype=np.int32),
                layers=np.asarray([0], dtype=np.int32),
                scaling_mode=np.asarray("s_missing"),
                metadata_json=np.asarray(json.dumps({"replace_ratio": 0.25})),
                layer_0_centroids=np.asarray([[1.0, 2.0]], dtype=np.float16),
            )

            replaced_payload = {
                "format_version": 1,
                "n_layer_total": 2,
                "n_expert": 8,
                "layers": {"0": [1, 2]},
            }
            replaced_json.write_text(json.dumps(replaced_payload), encoding="utf-8")

            matrix = mod.discover_matrix([lookup_npz], baseline_id="baseline")
            rows = matrix["rows"]
            modes = [r["mode"] for r in rows]

            self.assertIn("baseline", modes)
            self.assertIn("lookup", modes)
            self.assertIn("remove-only", modes)

            lookup_row = next(r for r in rows if r["mode"] == "lookup")
            self.assertEqual(lookup_row["replace_ratio"], 0.25)
            self.assertEqual(lookup_row["clusters_per_layer"], 1)

    def test_parse_bench_tok_s_prefers_generation_rows(self):
        mod = load_module()

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bench = tmp / "bench.jsonl"
            bench.write_text(
                "\n".join(
                    [
                        json.dumps({"n_prompt": 32, "n_gen": 0, "avg_ts": 10.0}),
                        json.dumps({"n_prompt": 0, "n_gen": 128, "avg_ts": 20.0}),
                        json.dumps({"n_prompt": 0, "n_gen": 64, "avg_ts": 30.0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertAlmostEqual(mod.parse_bench_tok_s(bench), 25.0)

    def test_summarize_matrix_parses_result_paths_and_evaluates_gate(self):
        mod = load_module()

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ppl_txt = tmp / "ppl.txt"
            bench_jsonl = tmp / "bench.jsonl"

            ppl_txt.write_text("perplexity: 7.5\n", encoding="utf-8")
            bench_jsonl.write_text(
                json.dumps({"n_prompt": 0, "n_gen": 128, "avg_ts": 120.0}) + "\n",
                encoding="utf-8",
            )

            matrix = {
                "acceptance_gates": {"quality_max_ppl_delta": 2.0},
                "rows": [
                    {
                        "id": "baseline",
                        "mode": "baseline",
                        "ppl": 6.0,
                        "tok_s": 100.0,
                        "stable": True,
                    },
                    {
                        "id": "lookup-a",
                        "mode": "lookup",
                        "ppl": None,
                        "tok_s": None,
                        "ppl_result_path": str(ppl_txt),
                        "bench_result_path": str(bench_jsonl),
                        "stable": True,
                    },
                ],
            }

            summary = mod.summarize_matrix(matrix, require_complete=True)
            self.assertTrue(summary["acceptance"]["quality_gate_pass"])
            self.assertTrue(summary["acceptance"]["promotion_gate_pass"])

            lookup_row = next(r for r in summary["rows"] if r["id"] == "lookup-a")
            self.assertAlmostEqual(lookup_row["ppl"], 7.5)
            self.assertAlmostEqual(lookup_row["tok_s"], 120.0)
            self.assertAlmostEqual(lookup_row["ppl_delta"], 1.5)

    def test_summarize_matrix_fails_quality_when_all_lookup_rows_exceed_delta(self):
        mod = load_module()
        matrix = {
            "acceptance_gates": {"quality_max_ppl_delta": 2.0},
            "rows": [
                {"id": "baseline", "mode": "baseline", "ppl": 5.0, "tok_s": 100.0},
                {"id": "lookup-bad", "mode": "lookup", "ppl": 7.6, "tok_s": 98.0, "stable": True},
            ],
        }

        summary = mod.summarize_matrix(matrix, require_complete=False)
        self.assertFalse(summary["acceptance"]["quality_gate_pass"])
        self.assertFalse(summary["acceptance"]["promotion_gate_pass"])

    def test_summarize_matrix_stable_string_false_is_false(self):
        mod = load_module()
        matrix = {
            "acceptance_gates": {"quality_max_ppl_delta": 2.0},
            "rows": [
                {"id": "baseline", "mode": "baseline", "ppl": 5.0, "tok_s": 100.0},
                {"id": "lookup-unstable", "mode": "lookup", "ppl": 6.5, "tok_s": 98.0, "stable": "false"},
            ],
        }

        summary = mod.summarize_matrix(matrix, require_complete=False)
        self.assertTrue(summary["acceptance"]["quality_gate_pass"])
        self.assertFalse(summary["acceptance"]["promotion_gate_pass"])

    def test_summarize_matrix_rejects_invalid_stable_value(self):
        mod = load_module()
        matrix = {
            "acceptance_gates": {"quality_max_ppl_delta": 2.0},
            "rows": [
                {"id": "baseline", "mode": "baseline", "ppl": 5.0, "tok_s": 100.0},
                {"id": "lookup-ambiguous", "mode": "lookup", "ppl": 6.5, "tok_s": 98.0, "stable": "maybe"},
            ],
        }

        with self.assertRaises(ValueError):
            mod.summarize_matrix(matrix, require_complete=True)

    def test_summarize_matrix_supports_partial_baseline_tok_s_only(self):
        mod = load_module()
        matrix = {
            "acceptance_gates": {"quality_max_ppl_delta": 2.0},
            "rows": [
                {"id": "baseline", "mode": "baseline", "ppl": None, "tok_s": 100.0},
                {"id": "lookup-a", "mode": "lookup", "ppl": None, "tok_s": 110.0, "stable": True},
            ],
        }

        summary = mod.summarize_matrix(matrix, require_complete=False)
        self.assertIsNone(summary["acceptance"]["quality_gate_pass"])
        lookup_row = next(r for r in summary["rows"] if r["id"] == "lookup-a")
        self.assertIsNone(lookup_row["ppl_delta"])
        self.assertAlmostEqual(lookup_row["tok_s_delta_pct"], 10.0)

    def test_summarize_matrix_requires_at_least_one_baseline_metric(self):
        mod = load_module()
        matrix = {
            "acceptance_gates": {"quality_max_ppl_delta": 2.0},
            "rows": [
                {"id": "baseline", "mode": "baseline", "ppl": None, "tok_s": None},
                {"id": "lookup-a", "mode": "lookup", "ppl": 6.0, "tok_s": 90.0, "stable": True},
            ],
        }

        with self.assertRaises(ValueError):
            mod.summarize_matrix(matrix, require_complete=True)


if __name__ == "__main__":
    unittest.main()
