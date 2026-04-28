from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ResolvedModel:
    model_spec: str
    repo_id: str
    filename: str
    local_path: str
    downloaded: bool
    cache_dir_used: str


@dataclass(slots=True)
class MatrixRef:
    tensor_name: str
    source_tensor_name: str
    shape: tuple[int, int]
    layer: int | None
    expert: int | None
    role: str | None
    tensor_type: str
    packed_expert_index: int | None = None
    packed_expert_axis: int | None = None


@dataclass(slots=True)
class SkippedTensor:
    name: str
    reason: str


@dataclass(slots=True)
class DiscoveryResult:
    total_tensors: int
    candidates: list[MatrixRef]
    skipped: list[SkippedTensor]
    metadata: dict[str, Any]


@dataclass(slots=True)
class MatrixMetrics:
    m: int
    n: int
    rank_used: int
    singular_value_count: int
    participation_ratio: float
    explained_spectral_energy_rank_fractions: list[float]
    fro_norm: float
    analysis_warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PerMatrixRecord:
    tensor: str
    source_tensor: str
    layer: int | None
    expert: int | None
    role: str | None
    tensor_type: str
    packed_expert_index: int | None
    packed_expert_axis: int | None
    shape: tuple[int, int]
    rank_used: int
    singular_value_count: int
    participation_ratio: float
    explained_spectral_energy_rank_fractions: list[float]
    fro_norm: float
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FailedMatrix:
    tensor: str
    source_tensor: str
    layer: int | None
    expert: int | None
    role: str | None
    packed_expert_index: int | None
    packed_expert_axis: int | None
    reason: str


@dataclass(slots=True)
class SummaryDistribution:
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    p10: float
    p25: float
    p75: float
    p90: float


@dataclass(slots=True)
class SummaryStats:
    participation_ratio: SummaryDistribution
    spectral_energy_rank_fractions: list[float]
    explained_spectral_energy_rank_fractions_mean: list[float]
    counts: dict[str, Any]


@dataclass(slots=True)
class Report:
    schema_version: str
    run: dict[str, Any]
    discovery: dict[str, Any]
    per_matrix: list[PerMatrixRecord]
    failed_matrices: list[FailedMatrix]
    summary: SummaryStats
    assumptions_and_caveats: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
