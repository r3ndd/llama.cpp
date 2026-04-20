#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molr.types import (
    MOLR_COVARIANCE_NPZ_SCHEMA_VERSION,
    MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2,
    MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION,
    MOLR_EXPERT_VALIDATION_SCHEMA_VERSION,
    MOLR_EXPERT_WEIGHTS_NPZ_SCHEMA_VERSION,
    MOLR_PLAN_SCHEMA_VERSION,
)


EXIT_OK = 0
EXIT_VALIDATION_ERROR = 2


ROLE_ALIASES = {
    "gate": "gate",
    "w1": "gate",
    "up": "up",
    "w3": "up",
    "down": "down",
    "w2": "down",
}


class MolrTrainError(RuntimeError):
    """Raised when per-expert MoLR training cannot proceed."""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        raise MolrTrainError(f"Failed writing JSON '{path}': {exc}") from exc


def _save_npz(path: Path, **arrays: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("wb") as handle:
            np.savez(handle, **arrays)
        tmp_path.replace(path)
    except Exception as exc:
        raise MolrTrainError(f"Failed writing NPZ '{path}': {exc}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MolrTrainError(f"Failed reading JSON '{path}': {exc}") from exc


def _load_npz(path: Path) -> Any:
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:
        raise MolrTrainError(f"Failed loading NPZ '{path}': {exc}") from exc


def _npz_scalar_string(payload: Any, key: str) -> str | None:
    if key not in payload:
        return None
    value = np.asarray(payload[key])
    if value.ndim == 0:
        return str(value.item())
    return str(value)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a single expert MoLR checkpoint from molr_plan.json + covariance_stats.npz + full expert weights NPZ."
        ),
    )
    parser.add_argument("--model", required=True, help="Model spec for contract validation and metadata.")
    parser.add_argument("--plan-json", required=True, help="Path to molr_plan.json.")
    parser.add_argument("--cov-npz", required=True, help="Path to covariance_stats.npz.")
    parser.add_argument(
        "--weights-npz",
        required=True,
        help=(
            "Path to full expert weights NPZ contract for one expert. "
            "Required matrix arrays: gate/up/down (or w1/w3/w2 aliases)."
        ),
    )
    parser.add_argument("--layer", type=int, required=True, help="Target expert layer index.")
    parser.add_argument("--expert", type=int, required=True, help="Target expert index within layer.")
    parser.add_argument("--steps", type=int, default=20000, help="Training steps. Default: 20000.")
    parser.add_argument("--batch-size", type=int, default=512, help="Synthetic batch size. Default: 512.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate. Default: 1e-4.")
    parser.add_argument("--lambda-lb", type=float, default=0.01, help="Load-balance coefficient. Default: 0.01.")
    parser.add_argument("--lambda-err", type=float, default=0.05, help="Error-head coefficient. Default: 0.05.")
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=2048,
        help="Synthetic held-out validation sample count. Default: 2048.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic sampling. Default: 0.")
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.95,
        help="Validation gate threshold for cosine similarity. Default: 0.95.",
    )
    parser.add_argument(
        "--error-corr-threshold",
        type=float,
        default=0.70,
        help="Validation gate threshold for error-head Pearson correlation. Default: 0.70.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=250,
        help="Progress print interval in steps. Set <=0 to disable. Default: 250.",
    )
    parser.add_argument("--out-checkpoint", required=True, help="Output path for molr_expert_<layer>_<expert>.npz")
    parser.add_argument("--out-validation", required=True, help="Output path for per-expert validation JSON.")

    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be > 0.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0.")
    if args.lr <= 0.0:
        parser.error("--lr must be > 0.")
    if args.lambda_lb < 0.0:
        parser.error("--lambda-lb must be >= 0.")
    if args.lambda_err < 0.0:
        parser.error("--lambda-err must be >= 0.")
    if args.validation_samples <= 0:
        parser.error("--validation-samples must be > 0.")
    return args


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(shifted)
    return ex / np.sum(ex, axis=1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _silu_grad(x: np.ndarray) -> np.ndarray:
    sig = _sigmoid(x)
    return sig + x * sig * (1.0 - sig)


def _infer_oriented_matrix(raw: np.ndarray, *, d_model: int, role: str, d_intermediate: int | None = None) -> np.ndarray:
    if raw.ndim != 2:
        raise MolrTrainError(f"weights matrix for role='{role}' must be 2D, got shape={raw.shape}")

    a, b = int(raw.shape[0]), int(raw.shape[1])
    if role in ("gate", "up"):
        if a == d_model:
            return raw.astype(np.float64, copy=False)
        if b == d_model:
            return raw.T.astype(np.float64, copy=False)
        raise MolrTrainError(
            f"Cannot orient role='{role}' matrix shape={raw.shape} with d_model={d_model}",
        )

    if role != "down":
        raise MolrTrainError(f"Unexpected role '{role}'")

    if d_intermediate is None:
        raise MolrTrainError("Internal error: d_intermediate required for role='down' orientation.")
    if a == d_intermediate and b == d_model:
        return raw.astype(np.float64, copy=False)
    if a == d_model and b == d_intermediate:
        return raw.T.astype(np.float64, copy=False)
    raise MolrTrainError(
        f"Cannot orient role='down' matrix shape={raw.shape} with d_model={d_model}, d_intermediate={d_intermediate}",
    )


def _lookup_plan_expert(plan: dict[str, Any], *, layer: int, expert: int) -> dict[str, Any]:
    if str(plan.get("schema_version") or "") != MOLR_PLAN_SCHEMA_VERSION:
        raise MolrTrainError(
            f"Unexpected plan schema_version='{plan.get('schema_version')}', expected '{MOLR_PLAN_SCHEMA_VERSION}'.",
        )
    for entry in plan.get("experts", []):
        if int(entry.get("layer")) == layer and int(entry.get("expert")) == expert:
            return entry
    raise MolrTrainError(f"Expert (layer={layer}, expert={expert}) not found in molr_plan.json")


def _plan_matrix_by_role(expert_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_role: dict[str, dict[str, Any]] = {}
    for matrix in expert_plan.get("matrices", []):
        role_raw = str(matrix.get("role") or "").strip().lower()
        role = ROLE_ALIASES.get(role_raw)
        if role is None:
            continue
        by_role.setdefault(role, matrix)

    missing = [role for role in ("gate", "up", "down") if role not in by_role]
    if missing:
        raise MolrTrainError(f"Plan for expert missing matrix roles: {', '.join(missing)}")
    return by_role


def _load_weights_by_role(path: Path) -> dict[str, np.ndarray]:
    payload = _load_npz(path)
    role_keys = {
        "gate": ("gate", "w_gate", "gate_weight", "w1"),
        "up": ("up", "w_up", "up_weight", "w3"),
        "down": ("down", "w_down", "down_weight", "w2"),
    }

    resolved: dict[str, np.ndarray] = {}
    for role, keys in role_keys.items():
        found = None
        for key in keys:
            if key in payload:
                found = np.asarray(payload[key])
                break
        if found is None:
            raise MolrTrainError(
                f"weights NPZ '{path}' missing matrix for role='{role}'. Expected one of keys: {keys}",
            )
        resolved[role] = found

    schema_value = _npz_scalar_string(payload, "schema_version")

    if schema_value is not None and schema_value != MOLR_EXPERT_WEIGHTS_NPZ_SCHEMA_VERSION:
        raise MolrTrainError(
            f"weights NPZ schema mismatch: got '{schema_value}', expected '{MOLR_EXPERT_WEIGHTS_NPZ_SCHEMA_VERSION}'",
        )

    return resolved


def _lookup_covariance(path: Path, *, layer: int, expert: int) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_npz(path)

    schema_version = _npz_scalar_string(payload, "schema_version")
    accepted_versions = {
        MOLR_COVARIANCE_NPZ_SCHEMA_VERSION,
        MOLR_COVARIANCE_NPZ_SCHEMA_VERSION_V2,
    }
    if schema_version is not None and schema_version not in accepted_versions:
        raise MolrTrainError(
            f"covariance NPZ schema mismatch: got '{schema_version}', expected one of {sorted(accepted_versions)}",
        )

    for key in ("layers", "mu", "chol"):
        if key not in payload:
            raise MolrTrainError(f"covariance NPZ '{path}' missing required array '{key}'")

    granularity = _npz_scalar_string(payload, "granularity")
    if granularity is None:
        granularity = "expert" if "experts" in payload else "layer"

    layers = np.asarray(payload["layers"]).astype(np.int64, copy=False)
    mu = np.asarray(payload["mu"]).astype(np.float64, copy=False)
    chol = np.asarray(payload["chol"]).astype(np.float64, copy=False)

    if mu.ndim != 2 or chol.ndim != 3:
        raise MolrTrainError(f"Invalid covariance shapes: mu={mu.shape}, chol={chol.shape}")
    if layers.ndim != 1:
        raise MolrTrainError(f"Invalid covariance key shape: layers={layers.shape}")
    if not (layers.shape[0] == mu.shape[0] == chol.shape[0]):
        raise MolrTrainError("Covariance arrays are not aligned on expert axis.")

    matches = np.array([], dtype=np.int64)
    used_layer_fallback = False
    if "experts" in payload and granularity != "layer":
        experts = np.asarray(payload["experts"]).astype(np.int64, copy=False)
        if experts.ndim != 1 or experts.shape[0] != layers.shape[0]:
            raise MolrTrainError(f"Invalid covariance expert key shape: experts={experts.shape}")
        matches = np.where((layers == layer) & (experts == expert))[0]

    if matches.size == 0:
        # fallback to layer-level covariance when expert row is unavailable
        used_layer_fallback = True
        matches = np.where(layers == layer)[0]

    if matches.size == 0:
        raise MolrTrainError(f"No covariance entry for expert/layer (layer={layer}, expert={expert})")
    if matches.size > 1:
        if used_layer_fallback:
            raise MolrTrainError(f"Duplicate layer covariance entries for layer={layer}")
        raise MolrTrainError(f"Duplicate covariance entries for resolved key (layer={layer}, expert={expert})")

    idx = int(matches[0])
    return mu[idx], chol[idx]


def _sample_synthetic_batch(rng: np.random.Generator, mu: np.ndarray, chol: np.ndarray, batch_size: int) -> np.ndarray:
    z = rng.standard_normal(size=(batch_size, mu.shape[0]))
    return z @ chol.T + mu


def _full_expert_forward(x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray) -> np.ndarray:
    gate = x @ w_gate
    up = x @ w_up
    hidden = _silu(gate) * up
    return hidden @ w_down


def _component_forward(x: np.ndarray, comp: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x_gate = x @ comp["gate_A"]
    gate = x_gate @ comp["gate_B"]
    x_up = x @ comp["up_A"]
    up = x_up @ comp["up_B"]
    silu_gate = _silu(gate)
    hidden = silu_gate * up
    x_down = hidden @ comp["down_A"]
    out = x_down @ comp["down_B"]
    cache = {
        "x_gate": x_gate,
        "gate": gate,
        "x_up": x_up,
        "up": up,
        "silu_gate": silu_gate,
        "hidden": hidden,
        "x_down": x_down,
    }
    return out, cache


def _init_component_from_svd(
    *,
    matrix: np.ndarray,
    rank: int,
    rank_indices: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rank <= 0:
        raise MolrTrainError(f"Invalid rank={rank}; expected > 0")

    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, int(s.shape[0]))
    if rank <= 0:
        raise MolrTrainError("SVD rank collapsed to zero during initialization.")

    u_r = u[:, :rank]
    s_r = s[:rank]
    vt_r = vt[:rank, :]

    a = np.zeros((matrix.shape[0], rank), dtype=np.float64)
    b = np.zeros((rank, matrix.shape[1]), dtype=np.float64)

    for idx in rank_indices:
        if idx < 0 or idx >= rank:
            continue
        scale = math.sqrt(max(float(s_r[idx]), 0.0))
        a[:, idx] = u_r[:, idx] * scale
        b[idx, :] = vt_r[idx, :] * scale

    if not np.any(a):
        a += rng.normal(loc=0.0, scale=1e-6, size=a.shape)
    if not np.any(b):
        b += rng.normal(loc=0.0, scale=1e-6, size=b.shape)

    return a, b


def _equalize_component_norms(
    components: list[dict[str, np.ndarray]],
    role_a: str,
    role_b: str,
) -> list[float]:
    norms: list[float] = []
    for comp in components:
        approx = comp[role_a] @ comp[role_b]
        norms.append(float(np.linalg.norm(approx, ord="fro")))

    non_zero = [n for n in norms if n > 0.0 and math.isfinite(n)]
    if not non_zero:
        return norms
    target = float(sum(non_zero) / len(non_zero))

    for idx, comp in enumerate(components):
        n = norms[idx]
        if n <= 0.0 or not math.isfinite(n):
            continue
        scale = math.sqrt(target / n)
        comp[role_a] *= scale
        comp[role_b] *= scale

    return norms


def build_initial_params(
    *,
    x_dim: int,
    d_intermediate: int,
    k_components: int,
    weights: dict[str, np.ndarray],
    plan_by_role: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    components: list[dict[str, np.ndarray]] = []

    role_rank: dict[str, int] = {}
    role_assignments: dict[str, list[list[int]]] = {"gate": [], "up": [], "down": []}
    for role in ("gate", "up", "down"):
        matrix_plan = plan_by_role[role]
        role_rank[role] = int(matrix_plan.get("rank", 0))
        init_partition = matrix_plan.get("init_partition", {})
        component_assignments = init_partition.get("component_assignments", [])
        if len(component_assignments) != k_components:
            raise MolrTrainError(
                f"Plan role='{role}' has {len(component_assignments)} component assignments, expected {k_components}",
            )
        for assignment in component_assignments:
            role_assignments[role].append([int(x) for x in assignment.get("rank_indices", [])])

    for comp_idx in range(k_components):
        gate_a, gate_b = _init_component_from_svd(
            matrix=weights["gate"],
            rank=role_rank["gate"],
            rank_indices=role_assignments["gate"][comp_idx],
            rng=rng,
        )
        up_a, up_b = _init_component_from_svd(
            matrix=weights["up"],
            rank=role_rank["up"],
            rank_indices=role_assignments["up"][comp_idx],
            rng=rng,
        )
        down_a, down_b = _init_component_from_svd(
            matrix=weights["down"],
            rank=role_rank["down"],
            rank_indices=role_assignments["down"][comp_idx],
            rng=rng,
        )
        components.append(
            {
                "gate_A": gate_a,
                "gate_B": gate_b,
                "up_A": up_a,
                "up_B": up_b,
                "down_A": down_a,
                "down_B": down_b,
            }
        )

    pre_norms = {
        "gate": _equalize_component_norms(components, "gate_A", "gate_B"),
        "up": _equalize_component_norms(components, "up_A", "up_B"),
        "down": _equalize_component_norms(components, "down_A", "down_B"),
    }

    params = {
        "router_w": np.zeros((x_dim, k_components), dtype=np.float64),
        "router_b": np.zeros((k_components,), dtype=np.float64),
        "error_w": np.zeros((x_dim,), dtype=np.float64),
        "error_b": np.array(0.0, dtype=np.float64),
        "components": components,
    }

    init_meta = {
        "k_components": k_components,
        "d_model": x_dim,
        "d_intermediate": d_intermediate,
        "ranks": role_rank,
        "component_init_fro_norms_before_equalization": pre_norms,
    }
    return params, init_meta


def _zero_like_params(params: dict[str, Any]) -> dict[str, Any]:
    grads = {
        "router_w": np.zeros_like(params["router_w"]),
        "router_b": np.zeros_like(params["router_b"]),
        "error_w": np.zeros_like(params["error_w"]),
        "error_b": np.zeros_like(params["error_b"]),
        "components": [],
    }
    for comp in params["components"]:
        grads["components"].append({name: np.zeros_like(value) for name, value in comp.items()})
    return grads


def compute_objective_and_gradients(
    *,
    x: np.ndarray,
    y_true: np.ndarray,
    params: dict[str, Any],
    lambda_lb: float,
    lambda_err: float,
    detach_true_error_target: bool = True,
    epsilon_norm: float = 1e-12,
) -> tuple[dict[str, float], dict[str, Any], dict[str, np.ndarray]]:
    batch_size = int(x.shape[0])
    d_model = int(y_true.shape[1])
    k_components = int(len(params["components"]))

    router_logits = x @ params["router_w"] + params["router_b"]
    router = _softmax(router_logits)

    outputs = np.zeros((batch_size, k_components, d_model), dtype=np.float64)
    caches: list[dict[str, np.ndarray]] = []
    for k in range(k_components):
        out_k, cache_k = _component_forward(x, params["components"][k])
        outputs[:, k, :] = out_k
        caches.append(cache_k)

    y_hat = np.sum(outputs * router[:, :, None], axis=1)
    diff = y_hat - y_true

    err_logits = x @ params["error_w"] + params["error_b"]
    pred_error = _softplus(err_logits)
    true_error = np.sqrt(np.sum(diff * diff, axis=1) + epsilon_norm)

    loss_mse = float(np.mean(diff * diff))
    batch_mean_router = np.mean(router, axis=0)
    loss_lb = float(np.sum(batch_mean_router * batch_mean_router))
    loss_err = float(np.mean((pred_error - true_error) ** 2))
    loss_total = loss_mse + lambda_lb * loss_lb + lambda_err * loss_err

    grads = _zero_like_params(params)

    grad_y = (2.0 / float(batch_size * d_model)) * diff
    grad_router = np.sum(grad_y[:, None, :] * outputs, axis=2)

    grad_outputs = np.zeros_like(outputs)
    for k in range(k_components):
        grad_outputs[:, k, :] = grad_y * router[:, k : k + 1]

    if lambda_lb > 0.0:
        grad_router += lambda_lb * (2.0 / float(batch_size)) * batch_mean_router[None, :]

    grad_router_logits = router * (grad_router - np.sum(grad_router * router, axis=1, keepdims=True))
    grads["router_w"] = x.T @ grad_router_logits
    grads["router_b"] = np.sum(grad_router_logits, axis=0)

    grad_true_error = np.zeros_like(true_error)
    if not detach_true_error_target and lambda_err > 0.0:
        grad_true_error = lambda_err * (2.0 / float(batch_size)) * (true_error - pred_error)

    if np.any(grad_true_error != 0.0):
        grad_y += (grad_true_error[:, None] * diff) / true_error[:, None]
        grad_outputs = np.zeros_like(outputs)
        grad_router = np.sum(grad_y[:, None, :] * outputs, axis=2)
        for k in range(k_components):
            grad_outputs[:, k, :] = grad_y * router[:, k : k + 1]
        if lambda_lb > 0.0:
            grad_router += lambda_lb * (2.0 / float(batch_size)) * batch_mean_router[None, :]
        grad_router_logits = router * (grad_router - np.sum(grad_router * router, axis=1, keepdims=True))
        grads["router_w"] = x.T @ grad_router_logits
        grads["router_b"] = np.sum(grad_router_logits, axis=0)

    grad_pred_error = lambda_err * (2.0 / float(batch_size)) * (pred_error - true_error)
    grad_err_logits = grad_pred_error * _sigmoid(err_logits)
    grads["error_w"] = x.T @ grad_err_logits
    grads["error_b"] = np.sum(grad_err_logits, axis=0)

    for k in range(k_components):
        comp = params["components"][k]
        cache = caches[k]
        grad_o = grad_outputs[:, k, :]

        grad_x_down = grad_o @ comp["down_B"].T
        grads["components"][k]["down_B"] = cache["x_down"].T @ grad_o
        grad_hidden = grad_x_down @ comp["down_A"].T
        grads["components"][k]["down_A"] = cache["hidden"].T @ grad_x_down

        grad_silu_gate = grad_hidden * cache["up"]
        grad_up = grad_hidden * cache["silu_gate"]

        grad_x_up = grad_up @ comp["up_B"].T
        grads["components"][k]["up_B"] = cache["x_up"].T @ grad_up
        grads["components"][k]["up_A"] = x.T @ grad_x_up

        grad_gate = grad_silu_gate * _silu_grad(cache["gate"])
        grad_x_gate = grad_gate @ comp["gate_B"].T
        grads["components"][k]["gate_B"] = cache["x_gate"].T @ grad_gate
        grads["components"][k]["gate_A"] = x.T @ grad_x_gate

    forward = {
        "router": router,
        "y_hat": y_hat,
        "pred_error": pred_error,
        "true_error": true_error,
    }
    losses = {
        "loss_total": loss_total,
        "loss_mse": loss_mse,
        "loss_load_balance": loss_lb,
        "loss_error_head": loss_err,
    }
    return losses, grads, forward


def _adam_init(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": 0,
        "m": _zero_like_params(params),
        "v": _zero_like_params(params),
    }


def _adam_update(
    *,
    params: dict[str, Any],
    grads: dict[str, Any],
    state: dict[str, Any],
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    state["step"] += 1
    t = int(state["step"])
    bias1 = 1.0 - (beta1**t)
    bias2 = 1.0 - (beta2**t)

    for key in ("router_w", "router_b", "error_w", "error_b"):
        m = state["m"][key]
        v = state["v"][key]
        g = grads[key]
        m *= beta1
        m += (1.0 - beta1) * g
        v *= beta2
        v += (1.0 - beta2) * (g * g)
        m_hat = m / bias1
        v_hat = v / bias2
        params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    for idx, comp in enumerate(params["components"]):
        for name in ("gate_A", "gate_B", "up_A", "up_B", "down_A", "down_B"):
            m = state["m"]["components"][idx][name]
            v = state["v"]["components"][idx][name]
            g = grads["components"][idx][name]
            m *= beta1
            m += (1.0 - beta1) * g
            v *= beta2
            v += (1.0 - beta2) * (g * g)
            m_hat = m / bias1
            v_hat = v / bias2
            comp[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def _pearson_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    if a.shape != b.shape:
        raise MolrTrainError(f"Pearson inputs shape mismatch: {a.shape} vs {b.shape}")
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    denom = math.sqrt(float(np.sum(a0 * a0))) * math.sqrt(float(np.sum(b0 * b0)))
    if denom <= eps:
        return 0.0
    return float(np.sum(a0 * b0) / denom)


def _evaluate_validation(
    *,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    lambda_lb: float,
    lambda_err: float,
) -> dict[str, Any]:
    losses, _, forward = compute_objective_and_gradients(
        x=x_val,
        y_true=y_val,
        params=params,
        lambda_lb=lambda_lb,
        lambda_err=lambda_err,
        detach_true_error_target=True,
    )

    y_hat = forward["y_hat"]
    router = forward["router"]
    pred_error = forward["pred_error"]
    true_error = forward["true_error"]

    dot = np.sum(y_hat * y_val, axis=1)
    norm_hat = np.sqrt(np.sum(y_hat * y_hat, axis=1) + 1e-12)
    norm_true = np.sqrt(np.sum(y_val * y_val, axis=1) + 1e-12)
    cosine = dot / (norm_hat * norm_true)
    rel_norm_error = np.sqrt(np.sum((y_hat - y_val) ** 2, axis=1)) / norm_true

    router_entropy = -np.sum(router * np.log(np.clip(router, 1e-12, 1.0)), axis=1)
    corr = _pearson_corr(pred_error, true_error)

    return {
        **losses,
        "cosine_similarity_mean": float(np.mean(cosine)),
        "relative_output_norm_error_mean": float(np.mean(rel_norm_error)),
        "router_entropy_mean": float(np.mean(router_entropy)),
        "error_head_pearson_r": float(corr),
        "pred_error_mean": float(np.mean(pred_error)),
        "true_error_mean": float(np.mean(true_error)),
    }


def _build_training_weights(
    *,
    raw_weights: dict[str, np.ndarray],
    d_model: int,
) -> tuple[dict[str, np.ndarray], int]:
    gate = _infer_oriented_matrix(raw_weights["gate"], d_model=d_model, role="gate")
    up = _infer_oriented_matrix(raw_weights["up"], d_model=d_model, role="up")
    if gate.shape[1] != up.shape[1]:
        raise MolrTrainError(
            f"gate/up intermediate dims differ after orientation: gate={gate.shape}, up={up.shape}",
        )
    d_intermediate = int(gate.shape[1])
    down = _infer_oriented_matrix(raw_weights["down"], d_model=d_model, role="down", d_intermediate=d_intermediate)
    return {"gate": gate, "up": up, "down": down}, d_intermediate


def _extract_k_components(plan_by_role: dict[str, dict[str, Any]], default_k: int) -> int:
    k_values: set[int] = {default_k}
    for role in ("gate", "up", "down"):
        k_values.add(int(plan_by_role[role].get("k_components", default_k)))
    if len(k_values) != 1:
        raise MolrTrainError(f"Inconsistent k_components across roles in plan: {sorted(k_values)}")
    k = int(next(iter(k_values)))
    if k <= 0:
        raise MolrTrainError(f"Invalid k_components={k}")
    return k


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        plan_path = Path(args.plan_json).expanduser().resolve()
        cov_path = Path(args.cov_npz).expanduser().resolve()
        weights_path = Path(args.weights_npz).expanduser().resolve()
        out_ckpt = Path(args.out_checkpoint).expanduser().resolve()
        out_val = Path(args.out_validation).expanduser().resolve()

        if not plan_path.is_file():
            raise MolrTrainError(f"--plan-json path is not a file: '{plan_path}'")
        if not cov_path.is_file():
            raise MolrTrainError(f"--cov-npz path is not a file: '{cov_path}'")
        if not weights_path.is_file():
            raise MolrTrainError(f"--weights-npz path is not a file: '{weights_path}'")

        plan = _load_json(plan_path)
        plan_model_spec = str(plan.get("model_spec") or "")
        if plan_model_spec and plan_model_spec != args.model:
            raise MolrTrainError(
                f"Model spec mismatch: --model='{args.model}' does not match plan model_spec='{plan_model_spec}'",
            )
        expert_plan = _lookup_plan_expert(plan, layer=int(args.layer), expert=int(args.expert))
        plan_by_role = _plan_matrix_by_role(expert_plan)
        k_components = _extract_k_components(plan_by_role, default_k=int(plan.get("default_k", 0)))

        mu, chol = _lookup_covariance(cov_path, layer=int(args.layer), expert=int(args.expert))
        d_model = int(mu.shape[0])
        if chol.shape != (d_model, d_model):
            raise MolrTrainError(
                f"Covariance cholesky shape mismatch for expert: expected ({d_model},{d_model}), got {chol.shape}",
            )

        raw_weights = _load_weights_by_role(weights_path)
        full_weights, d_intermediate = _build_training_weights(raw_weights=raw_weights, d_model=d_model)

        params, init_meta = build_initial_params(
            x_dim=d_model,
            d_intermediate=d_intermediate,
            k_components=k_components,
            weights=full_weights,
            plan_by_role=plan_by_role,
            seed=int(args.seed),
        )
        optim = _adam_init(params)
        rng_train = np.random.default_rng(int(args.seed))
        rng_val = np.random.default_rng(int(args.seed) + 1)

        loss_history_tail: list[dict[str, float]] = []
        for step in range(1, int(args.steps) + 1):
            x_batch = _sample_synthetic_batch(rng_train, mu, chol, int(args.batch_size))
            y_batch = _full_expert_forward(
                x_batch,
                full_weights["gate"],
                full_weights["up"],
                full_weights["down"],
            )

            losses, grads, _ = compute_objective_and_gradients(
                x=x_batch,
                y_true=y_batch,
                params=params,
                lambda_lb=float(args.lambda_lb),
                lambda_err=float(args.lambda_err),
                detach_true_error_target=True,
            )

            if not np.isfinite(losses["loss_total"]):
                raise MolrTrainError(
                    f"Encountered non-finite loss at step={step}: {losses['loss_total']}",
                )

            _adam_update(
                params=params,
                grads=grads,
                state=optim,
                lr=float(args.lr),
            )

            if step > int(args.steps) - 1000 or step == int(args.steps):
                loss_history_tail.append({"step": float(step), **losses})
            if len(loss_history_tail) > 1000:
                loss_history_tail = loss_history_tail[-1000:]

            if int(args.log_interval) > 0 and (step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps)):
                print(
                    "[molr-train-expert] "
                    f"layer={args.layer} expert={args.expert} "
                    f"step={step}/{args.steps} "
                    f"loss_total={losses['loss_total']:.6f} "
                    f"mse={losses['loss_mse']:.6f} "
                    f"lb={losses['loss_load_balance']:.6f} "
                    f"err={losses['loss_error_head']:.6f}",
                )

        x_val = _sample_synthetic_batch(rng_val, mu, chol, int(args.validation_samples))
        y_val = _full_expert_forward(
            x_val,
            full_weights["gate"],
            full_weights["up"],
            full_weights["down"],
        )
        val = _evaluate_validation(
            x_val=x_val,
            y_val=y_val,
            params=params,
            lambda_lb=float(args.lambda_lb),
            lambda_err=float(args.lambda_err),
        )

        failed_reasons: list[str] = []
        if val["cosine_similarity_mean"] < float(args.cosine_threshold):
            failed_reasons.append(
                f"cosine_below_threshold({val['cosine_similarity_mean']:.6f}<{args.cosine_threshold:.6f})",
            )
        if val["error_head_pearson_r"] < float(args.error_corr_threshold):
            failed_reasons.append(
                f"error_corr_below_threshold({val['error_head_pearson_r']:.6f}<{args.error_corr_threshold:.6f})",
            )

        comp = params["components"]
        ckpt_arrays: dict[str, Any] = {
            "schema_version": np.array(MOLR_EXPERT_CHECKPOINT_SCHEMA_VERSION),
            "model_spec": np.array(args.model),
            "layer": np.array(int(args.layer), dtype=np.int64),
            "expert": np.array(int(args.expert), dtype=np.int64),
            "k_components": np.array(int(k_components), dtype=np.int64),
            "d_model": np.array(int(d_model), dtype=np.int64),
            "d_intermediate": np.array(int(d_intermediate), dtype=np.int64),
            "router_w": params["router_w"].astype(np.float32),
            "router_b": params["router_b"].astype(np.float32),
            "error_w": params["error_w"].astype(np.float32),
            "error_b": np.array(float(params["error_b"]), dtype=np.float32),
        }

        for i, c in enumerate(comp):
            for name in ("gate_A", "gate_B", "up_A", "up_B", "down_A", "down_B"):
                ckpt_arrays[f"component_{i}_{name}"] = c[name].astype(np.float32)

        _save_npz(out_ckpt, **ckpt_arrays)

        validation_payload = {
            "schema_version": MOLR_EXPERT_VALIDATION_SCHEMA_VERSION,
            "created_at_utc": _now_utc_iso(),
            "model_spec": args.model,
            "layer": int(args.layer),
            "expert": int(args.expert),
            "status": "pass" if not failed_reasons else "fail",
            "failure_reasons": failed_reasons,
            "quality_thresholds": {
                "cosine_similarity_min": float(args.cosine_threshold),
                "error_head_pearson_r_min": float(args.error_corr_threshold),
            },
            "config": {
                "steps": int(args.steps),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "lambda_lb": float(args.lambda_lb),
                "lambda_err": float(args.lambda_err),
                "validation_samples": int(args.validation_samples),
                "seed": int(args.seed),
            },
            "initialization": init_meta,
            "validation_metrics": val,
            "training_loss_tail": loss_history_tail,
            "artifacts": {
                "checkpoint_npz": str(out_ckpt),
                "weights_npz": str(weights_path),
                "plan_json": str(plan_path),
                "covariance_npz": str(cov_path),
            },
        }
        _save_json(out_val, validation_payload)

        print(
            "[molr-train-expert] complete "
            f"layer={args.layer} expert={args.expert} "
            f"status={validation_payload['status']} "
            f"cos={val['cosine_similarity_mean']:.6f} "
            f"err_corr={val['error_head_pearson_r']:.6f} -> {out_ckpt}",
        )
        return EXIT_OK

    except MolrTrainError as exc:
        print(f"[error:molr-train-expert] {exc}", file=sys.stderr)
        return EXIT_VALIDATION_ERROR


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
