from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def ensure_gguf_import() -> Any:
    try:
        import gguf  # type: ignore

        return gguf
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        gguf_py = repo_root / "gguf-py"
        if gguf_py.is_dir() and str(gguf_py) not in sys.path:
            sys.path.insert(0, str(gguf_py))

    import gguf  # type: ignore  # noqa: E402

    return gguf
