#!/usr/bin/env python3

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "convert-moe-lookup-to-elt1.py"


def load_converter_module():
    spec = importlib.util.spec_from_file_location("convert_moe_lookup_to_elt1", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load convert-moe-lookup-to-elt1.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_elt1(path: Path):
    data = path.read_bytes()
    off = 0

    header = struct.unpack_from("<10I", data, off)
    off += struct.calcsize("<10I")

    (
        magic,
        fmt,
        model_id_len,
        n_layer,
        n_embd,
        n_expert,
        n_expert_used,
        vector_dtype,
        scaling_mode,
        n_layers_payload,
    ) = header

    model_id = data[off : off + model_id_len].decode("utf-8")
    off += model_id_len

    layers = []
    for _ in range(n_layers_payload):
        layer_id, n_keys, replaced_count = struct.unpack_from("<3I", data, off)
        off += struct.calcsize("<3I")

        vec_size = n_keys * n_embd
        centroids = np.frombuffer(data, dtype=np.dtype("<f2"), count=vec_size, offset=off).reshape((n_keys, n_embd))
        off += vec_size * np.dtype("<f2").itemsize

        contributions = np.frombuffer(data, dtype=np.dtype("<f2"), count=vec_size, offset=off).reshape((n_keys, n_embd))
        off += vec_size * np.dtype("<f2").itemsize

        replaced_ids = np.frombuffer(data, dtype=np.dtype("<u4"), count=replaced_count, offset=off)
        off += replaced_count * np.dtype("<u4").itemsize

        layers.append((layer_id, centroids, contributions, replaced_ids))

    return {
        "magic": magic,
        "format_version": fmt,
        "model_id": model_id,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_expert": n_expert,
        "n_expert_used": n_expert_used,
        "vector_dtype": vector_dtype,
        "scaling_mode": scaling_mode,
        "n_layers_payload": n_layers_payload,
        "layers": layers,
        "end_offset": off,
        "file_size": len(data),
    }


class TestConvertMoeLookupToElt1(unittest.TestCase):
    def test_convert_writes_expected_elt1_binary(self):
        mod = load_converter_module()

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            npz_path = tmp / "lookup.npz"
            replaced_path = tmp / "replaced.json"
            elt1_path = tmp / "lookup.elt1"

            np.savez_compressed(
                npz_path,
                format_version=np.asarray(1, dtype=np.int32),
                layers=np.asarray([0], dtype=np.int32),
                n_layer_total=np.asarray(2, dtype=np.int32),
                n_embd=np.asarray(3, dtype=np.int32),
                n_expert=np.asarray(4, dtype=np.int32),
                n_topk=np.asarray(2, dtype=np.int32),
                scaling_mode=np.asarray("s_missing"),
                replaced_expert_mask=np.asarray(
                    [[False, False, True, False], [False, False, False, False]],
                    dtype=bool,
                ),
                layer_0_centroids=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float16),
                layer_0_contributions=np.asarray([[4.0, 5.0, 6.0]], dtype=np.float16),
            )

            replaced_payload = {
                "format_version": 1,
                "n_layer_total": 2,
                "n_expert": 4,
                "layers": {"0": [2]},
            }
            replaced_path.write_text(json.dumps(replaced_payload), encoding="utf-8")

            mod.convert(npz_path, replaced_path, elt1_path, model_id="qwen3moe")

            parsed = parse_elt1(elt1_path)
            self.assertEqual(parsed["magic"], mod.ELT1_MAGIC)
            self.assertEqual(parsed["format_version"], mod.ELT1_FORMAT_VERSION)
            self.assertEqual(parsed["model_id"], "qwen3moe")
            self.assertEqual(parsed["n_layer"], 2)
            self.assertEqual(parsed["n_embd"], 3)
            self.assertEqual(parsed["n_expert"], 4)
            self.assertEqual(parsed["n_expert_used"], 2)
            self.assertEqual(parsed["vector_dtype"], mod.ELT1_VECTOR_DTYPE_FP16)
            self.assertEqual(parsed["scaling_mode"], mod.ELT1_SCALING_S_MISSING)
            self.assertEqual(parsed["n_layers_payload"], 1)

            layer_id, centroids, contributions, replaced_ids = parsed["layers"][0]
            self.assertEqual(layer_id, 0)
            np.testing.assert_allclose(centroids.astype(np.float32), np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
            np.testing.assert_allclose(
                contributions.astype(np.float32),
                np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32),
            )
            np.testing.assert_array_equal(replaced_ids, np.asarray([2], dtype=np.uint32))
            self.assertEqual(parsed["end_offset"], parsed["file_size"])

    def test_convert_rejects_nonfinite_centroids(self):
        mod = load_converter_module()

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            npz_path = tmp / "lookup.npz"
            replaced_path = tmp / "replaced.json"
            elt1_path = tmp / "lookup.elt1"

            np.savez_compressed(
                npz_path,
                format_version=np.asarray(1, dtype=np.int32),
                layers=np.asarray([0], dtype=np.int32),
                n_layer_total=np.asarray(1, dtype=np.int32),
                n_embd=np.asarray(2, dtype=np.int32),
                n_expert=np.asarray(4, dtype=np.int32),
                n_topk=np.asarray(2, dtype=np.int32),
                scaling_mode=np.asarray("s_missing"),
                layer_0_centroids=np.asarray([[np.nan, 0.0]], dtype=np.float32),
                layer_0_contributions=np.asarray([[1.0, 2.0]], dtype=np.float32),
            )
            replaced_path.write_text(json.dumps({"format_version": 1, "layers": {"0": []}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "centroid contains non-finite value"):
                mod.convert(npz_path, replaced_path, elt1_path, model_id="qwen3moe")

    def test_convert_rejects_fp16_overflow_in_centroids(self):
        mod = load_converter_module()

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            npz_path = tmp / "lookup.npz"
            replaced_path = tmp / "replaced.json"
            elt1_path = tmp / "lookup.elt1"

            np.savez_compressed(
                npz_path,
                format_version=np.asarray(1, dtype=np.int32),
                layers=np.asarray([0], dtype=np.int32),
                n_layer_total=np.asarray(1, dtype=np.int32),
                n_embd=np.asarray(2, dtype=np.int32),
                n_expert=np.asarray(4, dtype=np.int32),
                n_topk=np.asarray(2, dtype=np.int32),
                scaling_mode=np.asarray("s_missing"),
                layer_0_centroids=np.asarray([[70000.0, 0.0]], dtype=np.float32),
                layer_0_contributions=np.asarray([[1.0, 2.0]], dtype=np.float32),
            )
            replaced_path.write_text(json.dumps({"format_version": 1, "layers": {"0": []}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "centroid overflows fp16"):
                mod.convert(npz_path, replaced_path, elt1_path, model_id="qwen3moe")

    def test_convert_rejects_fp16_overflow_in_contributions(self):
        mod = load_converter_module()

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            npz_path = tmp / "lookup.npz"
            replaced_path = tmp / "replaced.json"
            elt1_path = tmp / "lookup.elt1"

            np.savez_compressed(
                npz_path,
                format_version=np.asarray(1, dtype=np.int32),
                layers=np.asarray([0], dtype=np.int32),
                n_layer_total=np.asarray(1, dtype=np.int32),
                n_embd=np.asarray(2, dtype=np.int32),
                n_expert=np.asarray(4, dtype=np.int32),
                n_topk=np.asarray(2, dtype=np.int32),
                scaling_mode=np.asarray("s_missing"),
                layer_0_centroids=np.asarray([[0.0, 1.0]], dtype=np.float32),
                layer_0_contributions=np.asarray([[1.0, 70000.0]], dtype=np.float32),
            )
            replaced_path.write_text(json.dumps({"format_version": 1, "layers": {"0": []}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "contribution overflows fp16"):
                mod.convert(npz_path, replaced_path, elt1_path, model_id="qwen3moe")

    def test_convert_rejects_replaced_count_above_fill_safe_limit(self):
        mod = load_converter_module()

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            npz_path = tmp / "lookup.npz"
            replaced_path = tmp / "replaced.json"
            elt1_path = tmp / "lookup.elt1"

            np.savez_compressed(
                npz_path,
                format_version=np.asarray(1, dtype=np.int32),
                layers=np.asarray([0], dtype=np.int32),
                n_layer_total=np.asarray(1, dtype=np.int32),
                n_embd=np.asarray(2, dtype=np.int32),
                n_expert=np.asarray(4, dtype=np.int32),
                n_topk=np.asarray(3, dtype=np.int32),
                scaling_mode=np.asarray("s_missing"),
                layer_0_centroids=np.asarray([[0.0, 1.0]], dtype=np.float32),
                layer_0_contributions=np.asarray([[1.0, 2.0]], dtype=np.float32),
            )
            replaced_path.write_text(json.dumps({"format_version": 1, "layers": {"0": [0, 1]}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exceeds fill-safe limit"):
                mod.convert(npz_path, replaced_path, elt1_path, model_id="qwen3moe")

    def test_convert_rejects_duplicate_replaced_ids(self):
        mod = load_converter_module()

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            npz_path = tmp / "lookup.npz"
            replaced_path = tmp / "replaced.json"
            elt1_path = tmp / "lookup.elt1"

            np.savez_compressed(
                npz_path,
                format_version=np.asarray(1, dtype=np.int32),
                layers=np.asarray([0], dtype=np.int32),
                n_layer_total=np.asarray(1, dtype=np.int32),
                n_embd=np.asarray(2, dtype=np.int32),
                n_expert=np.asarray(8, dtype=np.int32),
                n_topk=np.asarray(2, dtype=np.int32),
                scaling_mode=np.asarray("s_missing"),
                layer_0_centroids=np.asarray([[0.0, 1.0]], dtype=np.float32),
                layer_0_contributions=np.asarray([[1.0, 2.0]], dtype=np.float32),
            )
            replaced_path.write_text(json.dumps({"format_version": 1, "layers": {"0": [3, 3]}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate replaced expert IDs"):
                mod.convert(npz_path, replaced_path, elt1_path, model_id="qwen3moe")

    def test_cli_converts_router_mass_replaced_alias(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            npz_path = tmp / "lookup.npz"
            replaced_path = tmp / "replaced.json"
            elt1_path = tmp / "lookup.elt1"

            np.savez_compressed(
                npz_path,
                format_version=np.asarray(1, dtype=np.int32),
                layers=np.asarray([0], dtype=np.int32),
                n_layer_total=np.asarray(1, dtype=np.int32),
                n_embd=np.asarray(2, dtype=np.int32),
                n_expert=np.asarray(4, dtype=np.int32),
                n_topk=np.asarray(2, dtype=np.int32),
                scaling_mode=np.asarray("router_mass_replaced"),
                layer_0_centroids=np.asarray([[0.0, 1.0]], dtype=np.float32),
                layer_0_contributions=np.asarray([[1.0, 2.0]], dtype=np.float32),
            )
            replaced_path.write_text(json.dumps({"format_version": 1, "layers": {"0": [3]}}), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--input",
                    str(npz_path),
                    "--replaced-experts",
                    str(replaced_path),
                    "--output",
                    str(elt1_path),
                    "--model-id",
                    "qwen3moe",
                ],
                check=True,
            )

            parsed = parse_elt1(elt1_path)
            self.assertEqual(parsed["scaling_mode"], 1)


if __name__ == "__main__":
    unittest.main()
