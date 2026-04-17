from __future__ import annotations

import os
from pathlib import Path

from .types import ResolvedModel


class ModelResolutionError(RuntimeError):
    """Raised when model specification resolution/download fails."""


def parse_model_spec(model_spec: str) -> tuple[str, str]:
    if ":" not in model_spec:
        raise ModelResolutionError(
            "Invalid --model format. Expected <repo_id>:<filename_or_quant>, "
            f"got '{model_spec}'.",
        )

    repo_id, filename = model_spec.split(":", 1)
    repo_id = repo_id.strip()
    filename = filename.strip()

    if not repo_id or not filename:
        raise ModelResolutionError(
            "Invalid --model format. Both repo and filename/quant must be non-empty.",
        )

    if repo_id.count("/") != 1:
        raise ModelResolutionError(
            f"Invalid repo id '{repo_id}'. Expected <user>/<model>.",
        )

    return repo_id, filename


def _resolve_cache_dir(cache_dir: str | None) -> Path | None:
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()

    for env_var in (
        "LLAMA_CACHE",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HOME",
        "XDG_CACHE_HOME",
    ):
        value = os.environ.get(env_var)
        if value:
            base = Path(value).expanduser().resolve()
            if env_var == "HF_HOME":
                return base / "hub"
            if env_var == "XDG_CACHE_HOME":
                return base / "huggingface" / "hub"
            return base

    return None


def _infer_filename(repo_id: str, quant_or_name: str, cache_dir: Path | None) -> str:
    if quant_or_name.lower().endswith(".gguf"):
        return quant_or_name

    # If user passed a quant tag (e.g. Q4_K_M), attempt to infer matching file.
    candidates: list[str] = []
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        quant_lower = quant_or_name.lower()
        for entry in files:
            if not entry.lower().endswith(".gguf"):
                continue
            basename = Path(entry).name
            if "mmproj" in basename.lower():
                continue
            if quant_lower in basename.lower():
                candidates.append(entry)
        if not candidates:
            raise ModelResolutionError(
                f"Could not resolve GGUF filename from quant tag '{quant_or_name}' in repo '{repo_id}'. "
                "Pass an explicit filename like <repo>:<file.gguf>.",
            )
        candidates.sort()
        return candidates[0]
    except ModelResolutionError:
        raise
    except Exception as exc:
        try:
            from huggingface_hub import try_to_load_from_cache
        except Exception as import_exc:
            raise ModelResolutionError(
                "huggingface_hub is required for model resolution/download. "
                "Install it with: pip install huggingface_hub",
            ) from import_exc

        cached = try_to_load_from_cache(
            repo_id=repo_id,
            filename=quant_or_name,
            cache_dir=None if cache_dir is None else str(cache_dir),
        )
        if isinstance(cached, str):
            return quant_or_name
        raise ModelResolutionError(
            f"Failed to infer filename from '{quant_or_name}': {exc}",
        ) from exc


def resolve_model_path(model_spec: str, cache_dir: str | None) -> ResolvedModel:
    repo_id, requested = parse_model_spec(model_spec)
    resolved_cache = _resolve_cache_dir(cache_dir)

    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except Exception as exc:
        raise ModelResolutionError(
            "huggingface_hub is required for model resolution/download. "
            "Install it with: pip install huggingface_hub",
        ) from exc

    try:
        filename = _infer_filename(repo_id, requested, resolved_cache)
    except ModelResolutionError:
        raise
    except Exception as exc:
        raise ModelResolutionError(str(exc)) from exc

    cached = try_to_load_from_cache(
        repo_id=repo_id,
        filename=filename,
        cache_dir=None if resolved_cache is None else str(resolved_cache),
    )

    downloaded = False
    if isinstance(cached, str):
        local_path = Path(cached).resolve()
    else:
        try:
            local_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="model",
                    cache_dir=None if resolved_cache is None else str(resolved_cache),
                )
            ).resolve()
            downloaded = True
        except Exception as exc:
            raise ModelResolutionError(
                f"Failed to download model '{repo_id}:{filename}': {exc}",
            ) from exc

    cache_dir_used = str(
        (resolved_cache if resolved_cache is not None else _infer_cache_root(local_path)).resolve(),
    )

    return ResolvedModel(
        model_spec=model_spec,
        repo_id=repo_id,
        filename=filename,
        local_path=str(local_path),
        downloaded=downloaded,
        cache_dir_used=cache_dir_used,
    )


def _infer_cache_root(local_path: Path) -> Path:
    # Typical snapshot path: <cache>/models--org--repo/snapshots/<commit>/.../file.gguf
    for parent in local_path.parents:
        if parent.name.startswith("models--"):
            return parent.parent
    return local_path.parent
