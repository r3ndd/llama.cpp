#!/usr/bin/env python3

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build-moe-lookup.py"


def load_builder_module():
    spec = importlib.util.spec_from_file_location("build_moe_lookup", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load build-moe-lookup.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestBuildMoeLookup(unittest.TestCase):
    def test_removed_relative_contribution_zero_when_no_removed(self):
        mod = load_builder_module()

        layer_ids = np.asarray([0], dtype=np.int32)
        topk_ids = np.asarray([[1, 2]], dtype=np.int32)
        topk_weights = np.asarray([[0.6, 0.4]], dtype=np.float32)
        topk_expert_outputs = np.asarray([[[10.0, -1.0], [20.0, -2.0]]], dtype=np.float32)
        replaced_mask = np.zeros((1, 8), dtype=bool)

        targets, mass = mod.compute_removed_relative_contributions(
            layer_ids=layer_ids,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            topk_expert_outputs=topk_expert_outputs,
            replaced_mask=replaced_mask,
        )

        np.testing.assert_allclose(mass, np.asarray([0.0], dtype=np.float32))
        np.testing.assert_allclose(targets, np.asarray([[0.0, 0.0]], dtype=np.float32))

    def test_removed_relative_contribution_uses_removed_only_normalization(self):
        mod = load_builder_module()

        layer_ids = np.asarray([0], dtype=np.int32)
        topk_ids = np.asarray([[1, 2]], dtype=np.int32)
        topk_weights = np.asarray([[0.6, 0.4]], dtype=np.float32)
        topk_expert_outputs = np.asarray([[[10.0, 2.0], [30.0, 8.0]]], dtype=np.float32)
        replaced_mask = np.zeros((1, 8), dtype=bool)
        replaced_mask[0, 2] = True

        targets, mass = mod.compute_removed_relative_contributions(
            layer_ids=layer_ids,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            topk_expert_outputs=topk_expert_outputs,
            replaced_mask=replaced_mask,
        )

        np.testing.assert_allclose(mass, np.asarray([0.4], dtype=np.float32))
        np.testing.assert_allclose(targets, np.asarray([[30.0, 8.0]], dtype=np.float32))

    def test_cli_writes_contribution_table_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            trace_path = tmp / "trace.npz"
            out_path = tmp / "lookup.npz"
            replaced_path = tmp / "replaced.json"

            layer_ids = np.asarray([0, 0], dtype=np.int32)
            token_ids = np.asarray([0, 1], dtype=np.int32)
            h_pre_moe = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
            topk_ids = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
            topk_weights = np.asarray([[0.7, 0.3], [0.4, 0.6]], dtype=np.float16)
            topk_expert_outputs = np.asarray(
                [
                    [[1.0, 10.0], [2.0, 20.0]],
                    [[3.0, 30.0], [4.0, 40.0]],
                ],
                dtype=np.float16,
            )

            np.savez_compressed(
                trace_path,
                layer_ids=layer_ids,
                token_ids=token_ids,
                h_pre_moe=h_pre_moe,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                topk_expert_outputs=topk_expert_outputs,
            )

            replaced_payload = {
                "format_version": 1,
                "layers": {"0": [2]},
            }
            replaced_path.write_text(json.dumps(replaced_payload), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--input",
                    str(trace_path),
                    "--output",
                    str(out_path),
                    "--replaced-experts-json",
                    str(replaced_path),
                    "--clusters-per-layer",
                    "1",
                    "--kmeans-iters",
                    "1",
                ],
                check=True,
            )

            with np.load(out_path, allow_pickle=False) as npz:
                self.assertIn("layer_0_contributions", npz.files)
                self.assertNotIn("layer_0_residuals", npz.files)
                self.assertEqual(str(npz["scaling_mode"]), "s_missing")


if __name__ == "__main__":
    unittest.main()
