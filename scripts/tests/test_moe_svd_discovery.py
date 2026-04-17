from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import moe_svd.gguf_discovery as discovery
from moe_svd.types import MatrixRef


class _FakeField:
    def __init__(self, value):
        self._value = value

    def contents(self):
        return self._value


class _FakeTensorType:
    def __init__(self, name: str):
        self.name = name


class _FakeShape:
    def __init__(self, dims: list[int]):
        self._dims = dims

    def tolist(self):
        return self._dims


def _tensor(name: str, dims: list[int], tensor_type: str = "Q4_K"):
    return SimpleNamespace(name=name, shape=_FakeShape(dims), tensor_type=_FakeTensorType(tensor_type))


def test_discovery_records_packed_expert_tensor_and_arch_metadata(monkeypatch, tmp_path) -> None:
    gguf_path = tmp_path / "dummy.gguf"
    gguf_path.write_bytes(b"x")

    reader = SimpleNamespace(
        tensors=[
            _tensor("blk.0.ffn_gate_exps.weight", [2048, 512, 256]),
            _tensor("blk.0.ffn_gate_shexp.weight", [2048, 512]),
            _tensor("blk.0.attn_qkv.weight", [6144, 2048]),
        ],
        get_field=lambda key: {
            "general.architecture": _FakeField("qwen35moe"),
            "general.name": _FakeField("Qwen3.5-35B-A3B"),
            "general.file_type": _FakeField(15),
            "qwen35moe.block_count": _FakeField(40),
            "qwen35moe.expert_count": _FakeField(256),
            "qwen35moe.expert_used_count": _FakeField(9),
        }.get(key),
    )

    monkeypatch.setattr(discovery, "open_gguf_reader", lambda _: reader)

    result = discovery.discover_expert_matrices(str(gguf_path), include=[], exclude=[])

    assert result.total_tensors == 3
    routed = [c for c in result.candidates if c.source_tensor_name == "blk.0.ffn_gate_exps.weight"]
    assert len(routed) == 256
    assert routed[0].tensor_name == "blk.0.ffn_gate_exps.weight::expert[0]"
    assert routed[0].shape == (512, 2048)
    assert routed[0].expert == 0
    assert routed[0].packed_expert_axis == 0
    assert routed[-1].tensor_name == "blk.0.ffn_gate_exps.weight::expert[255]"
    assert all(s.name != "blk.0.ffn_gate_exps.weight" for s in result.skipped)

    shared = [c for c in result.candidates if c.tensor_name == "blk.0.ffn_gate_shexp.weight"]
    assert len(shared) == 1
    assert result.metadata["block_count"] == 40
    assert result.metadata["expert_count"] == 256
    assert result.metadata["expert_used_count"] == 9


def test_discovery_skips_unknown_packed_layout(monkeypatch, tmp_path) -> None:
    gguf_path = tmp_path / "dummy.gguf"
    gguf_path.write_bytes(b"x")

    reader = SimpleNamespace(
        tensors=[
            _tensor("blk.0.custom_exps.weight", [11, 13, 17]),
        ],
        get_field=lambda key: {
            "general.architecture": _FakeField("mysterymoe"),
            "mysterymoe.expert_count": _FakeField(9),
        }.get(key),
    )

    monkeypatch.setattr(discovery, "open_gguf_reader", lambda _: reader)

    result = discovery.discover_expert_matrices(str(gguf_path), include=[], exclude=[])

    assert result.candidates == []
    assert any(
        s.name == "blk.0.custom_exps.weight" and s.reason == "packed_expert_tensor_3d_unknown_layout"
        for s in result.skipped
    )


def test_load_matrix_from_reader_unpacks_packed_3d_tensor(monkeypatch) -> None:
    packed = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    fake_tensor = SimpleNamespace(
        name="blk.0.ffn_gate_exps.weight",
        data=packed,
        tensor_type=_FakeTensorType("F32"),
    )
    reader = SimpleNamespace(tensors=[fake_tensor])

    fake_gguf = SimpleNamespace(dequantize=lambda data, _tensor_type: data)
    monkeypatch.setattr(discovery, "_load_gguf_module", lambda: fake_gguf)

    ref = MatrixRef(
        tensor_name="blk.0.ffn_gate_exps.weight::expert[1]",
        source_tensor_name="blk.0.ffn_gate_exps.weight",
        shape=(2, 3),
        layer=0,
        expert=1,
        role="gate",
        tensor_type="F32",
        packed_expert_index=1,
        packed_expert_axis=2,
    )

    matrix = discovery.load_matrix_from_reader(reader=reader, matrix_ref=ref, dtype="float32")
    assert matrix.shape == (2, 3)
    assert np.array_equal(matrix, packed[:, :, 1])


def test_load_matrix_from_reader_validates_expected_shape(monkeypatch) -> None:
    matrix_2d = np.zeros((2, 3), dtype=np.float32)
    fake_tensor = SimpleNamespace(
        name="blk.0.ffn_gate_shexp.weight",
        data=matrix_2d,
        tensor_type=_FakeTensorType("F32"),
    )
    reader = SimpleNamespace(tensors=[fake_tensor])

    fake_gguf = SimpleNamespace(dequantize=lambda data, _tensor_type: data)
    monkeypatch.setattr(discovery, "_load_gguf_module", lambda: fake_gguf)

    ref = MatrixRef(
        tensor_name="blk.0.ffn_gate_shexp.weight",
        source_tensor_name="blk.0.ffn_gate_shexp.weight",
        shape=(3, 2),
        layer=0,
        expert=None,
        role="gate",
        tensor_type="F32",
    )

    with pytest.raises(discovery.DiscoveryError, match="produced shape"):
        discovery.load_matrix_from_reader(reader=reader, matrix_ref=ref, dtype="float32")
