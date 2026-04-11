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
    def test_compute_usage_scores_counts_per_layer(self):
        mod = load_builder_module()

        layer_ids = np.asarray([0, 0, 1], dtype=np.int32)
        topk_ids = np.asarray([[0, 2], [2, 3], [1, 3]], dtype=np.int32)

        usage = mod.compute_usage_scores(
            topk_ids=topk_ids,
            layer_ids=layer_ids,
            layers=[0, 1],
            n_expert=4,
        )

        np.testing.assert_array_equal(usage[0], np.asarray([1, 0, 2, 1], dtype=np.int64))
        np.testing.assert_array_equal(usage[1], np.asarray([0, 1, 0, 1], dtype=np.int64))

    def test_prepare_scores_for_log_x_axis_clamps_zeros(self):
        mod = load_builder_module()

        raw = np.asarray([0.0, 0.0, 2.0, 5.0], dtype=np.float64)
        safe, floor = mod.prepare_scores_for_log_x_axis(raw)

        self.assertGreater(floor, 0.0)
        self.assertTrue(np.all(safe > 0.0))
        self.assertEqual(float(safe[2]), 2.0)
        self.assertEqual(float(safe[3]), 5.0)

    def test_validate_args_allows_plot_mode_without_output(self):
        mod = load_builder_module()
        args = mod.argparse.Namespace(
            output=None,
            plot_heuristic=True,
            clusters_per_layer=1,
            kmeans_iters=1,
            distance_batch_size=1,
        )
        mod.validate_args(args)

    def test_validate_args_requires_output_without_plot_mode(self):
        mod = load_builder_module()
        args = mod.argparse.Namespace(
            output=None,
            plot_heuristic=False,
            clusters_per_layer=1,
            kmeans_iters=1,
            distance_batch_size=1,
        )
        with self.assertRaisesRegex(ValueError, "--output is required"):
            mod.validate_args(args)

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

    def test_cli_plot_heuristic_writes_image_without_lookup_outputs(self):
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not available")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            trace_path = tmp / "trace.npz"
            plot_path = tmp / "heuristic.png"

            np.savez_compressed(
                trace_path,
                layer_ids=np.asarray([0, 0, 1], dtype=np.int32),
                token_ids=np.asarray([0, 1, 2], dtype=np.int32),
                h_pre_moe=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float16),
                topk_ids=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
                topk_weights=np.asarray([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]], dtype=np.float16),
                topk_expert_outputs=np.asarray(
                    [
                        [[1.0, 10.0], [2.0, 20.0]],
                        [[3.0, 30.0], [4.0, 40.0]],
                        [[5.0, 50.0], [6.0, 60.0]],
                    ],
                    dtype=np.float16,
                ),
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--input",
                    str(trace_path),
                    "--plot-heuristic",
                    "--plot-output",
                    str(plot_path),
                ],
                check=True,
            )

            self.assertTrue(plot_path.exists())

    def test_prepare_matplotlib_import_path_evicts_mpl_toolkits_modules(self):
        mod = load_builder_module()

        sys.modules["mpl_toolkits"] = object()  # type: ignore[assignment]
        sys.modules["mpl_toolkits.mplot3d"] = object()  # type: ignore[assignment]

        try:
            mod._prepare_matplotlib_import_path()
            self.assertNotIn("mpl_toolkits", sys.modules)
            self.assertNotIn("mpl_toolkits.mplot3d", sys.modules)
        finally:
            sys.modules.pop("mpl_toolkits", None)
            sys.modules.pop("mpl_toolkits.mplot3d", None)


if __name__ == "__main__":
    unittest.main()
