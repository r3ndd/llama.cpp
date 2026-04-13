#!/usr/bin/env python3

"""Plan and summarize MoE lookup quality/perf evaluation matrices.

This script is intentionally offline-only:
- It does not run inference, perplexity, or benchmarks.
- It consumes existing artifacts/results and emits matrix + gate summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


PPL_PATTERNS = [
    re.compile(r"Mean\s+PPL(?:\([^\)]*\))?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\bperplexity\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\bppl\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan/summarize MoE lookup evaluation matrix from existing artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    discover = sub.add_parser(
        "discover",
        help="Build a matrix manifest from existing lookup/replaced artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    discover.add_argument(
        "--lookup-npz",
        type=Path,
        action="append",
        default=[],
        help="Lookup NPZ artifact from build-moe-lookup.py (repeatable).",
    )
    discover.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output matrix manifest JSON path.",
    )
    discover.add_argument(
        "--baseline-id",
        type=str,
        default="baseline",
        help="ID for baseline row in the generated matrix.",
    )

    summarize = sub.add_parser(
        "summarize",
        help="Generate gate summary from a filled matrix manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    summarize.add_argument("--matrix", type=Path, required=True, help="Matrix manifest JSON path.")
    summarize.add_argument("--output-md", type=Path, required=True, help="Markdown summary output path.")
    summarize.add_argument("--output-json", type=Path, default=None, help="Optional machine-readable summary JSON output.")
    summarize.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any non-baseline row is missing ppl/tok_s (or result paths that can be parsed).",
    )

    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"JSON file does not exist: {path}") from exc
    except Exception as exc:
        raise ValueError(f"failed to parse JSON '{path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return raw


def _read_lookup_metadata(lookup_npz: Path) -> Dict[str, Any]:
    try:
        npz = np.load(lookup_npz, allow_pickle=False)
    except FileNotFoundError as exc:
        raise ValueError(f"lookup NPZ does not exist: {lookup_npz}") from exc
    except Exception as exc:
        raise ValueError(f"failed to load lookup NPZ '{lookup_npz}': {exc}") from exc

    with npz:
        n_layer_total = int(np.asarray(npz["n_layer_total"])) if "n_layer_total" in npz else None
        n_expert = int(np.asarray(npz["n_expert"])) if "n_expert" in npz else None
        layers = [int(x) for x in np.asarray(npz["layers"], dtype=np.int32).tolist()] if "layers" in npz else []
        scaling_mode = None
        if "scaling_mode" in npz:
            scaling_mode = str(np.asarray(npz["scaling_mode"]))
        elif "runtime_scaling" in npz:
            scaling_mode = str(np.asarray(npz["runtime_scaling"]))

        clusters_per_layer: Dict[int, int] = {}
        for layer in layers:
            key = f"layer_{layer}_centroids"
            if key in npz:
                arr = np.asarray(npz[key])
                if arr.ndim == 2:
                    clusters_per_layer[layer] = int(arr.shape[0])

        replace_ratio_meta = None
        if "metadata_json" in npz:
            try:
                meta = json.loads(str(np.asarray(npz["metadata_json"])))
                if isinstance(meta, dict) and "replace_ratio" in meta:
                    replace_ratio_meta = float(meta["replace_ratio"])
            except Exception:
                pass

        replaced_mask = None
        if "replaced_expert_mask" in npz:
            replaced_mask = np.asarray(npz["replaced_expert_mask"], dtype=bool)

    return {
        "n_layer_total": n_layer_total,
        "n_expert": n_expert,
        "layers": layers,
        "scaling_mode": scaling_mode,
        "clusters_per_layer": clusters_per_layer,
        "replace_ratio_meta": replace_ratio_meta,
        "replaced_mask": replaced_mask,
    }


def _read_replaced_ratio(replaced_json: Path, n_expert: Optional[int]) -> Tuple[Optional[float], Dict[int, List[int]]]:
    payload = _read_json(replaced_json)
    layer_obj = payload.get("layers", payload)
    if not isinstance(layer_obj, dict):
        raise ValueError(f"replaced experts JSON must map layers to lists: {replaced_json}")

    parsed: Dict[int, List[int]] = {}
    for k, ids_raw in layer_obj.items():
        layer = int(k)
        if not isinstance(ids_raw, list):
            raise ValueError(f"layer {layer} must map to a list in {replaced_json}")
        ids = sorted(set(int(x) for x in ids_raw))
        parsed[layer] = ids

    if n_expert is None or n_expert <= 0 or not parsed:
        return None, parsed

    ratios: List[float] = []
    for ids in parsed.values():
        ratios.append(float(len(ids)) / float(n_expert))
    if not ratios:
        return None, parsed
    return float(sum(ratios) / len(ratios)), parsed


def _infer_replaced_json_for_lookup(lookup_npz: Path) -> Optional[Path]:
    candidate = lookup_npz.with_suffix(".replaced-experts.json")
    if candidate.exists():
        return candidate
    return None


def discover_matrix(lookup_npzs: List[Path], baseline_id: str) -> Dict[str, Any]:
    if not lookup_npzs:
        raise ValueError("at least one --lookup-npz is required")

    rows: List[Dict[str, Any]] = [
        {
            "id": baseline_id,
            "mode": "baseline",
            "lookup_npz": None,
            "replaced_experts_json": None,
            "replace_ratio": 0.0,
            "clusters_per_layer": None,
            "scaling_mode": None,
            "ppl": None,
            "tok_s": None,
            "ppl_result_path": None,
            "bench_result_path": None,
            "stable": None,
            "notes": "Fill ppl/tok_s or provide parseable result paths.",
        }
    ]

    remove_only_seen: set[str] = set()

    for lookup_npz in lookup_npzs:
        info = _read_lookup_metadata(lookup_npz)
        replaced_json = _infer_replaced_json_for_lookup(lookup_npz)

        replace_ratio = info["replace_ratio_meta"]
        replaced_layers = {}
        if replaced_json is not None:
            ratio_json, replaced_layers = _read_replaced_ratio(replaced_json, info["n_expert"])
            if replace_ratio is None:
                replace_ratio = ratio_json

        if replace_ratio is None and info["replaced_mask"] is not None and info["n_expert"]:
            mask = info["replaced_mask"]
            ratios = [float(np.count_nonzero(mask[layer])) / float(info["n_expert"]) for layer in info["layers"] if layer < mask.shape[0]]
            if ratios:
                replace_ratio = float(sum(ratios) / len(ratios))

        cluster_values = sorted(set(int(v) for v in info["clusters_per_layer"].values()))
        clusters_repr: Any
        if not cluster_values:
            clusters_repr = None
        elif len(cluster_values) == 1:
            clusters_repr = cluster_values[0]
        else:
            clusters_repr = cluster_values

        lookup_id = f"lookup:{lookup_npz.stem}"
        rows.append(
            {
                "id": lookup_id,
                "mode": "lookup",
                "lookup_npz": str(lookup_npz),
                "replaced_experts_json": str(replaced_json) if replaced_json else None,
                "replace_ratio": replace_ratio,
                "clusters_per_layer": clusters_repr,
                "scaling_mode": info["scaling_mode"],
                "layers": info["layers"],
                "ppl": None,
                "tok_s": None,
                "ppl_result_path": None,
                "bench_result_path": None,
                "stable": None,
                "notes": "",
            }
        )

        if replaced_json is not None:
            rj = str(replaced_json)
            if rj not in remove_only_seen:
                remove_only_seen.add(rj)
                rows.append(
                    {
                        "id": f"remove-only:{replaced_json.stem}",
                        "mode": "remove-only",
                        "lookup_npz": None,
                        "replaced_experts_json": rj,
                        "replace_ratio": replace_ratio,
                        "clusters_per_layer": None,
                        "scaling_mode": None,
                        "layers": sorted(replaced_layers.keys()),
                        "ppl": None,
                        "tok_s": None,
                        "ppl_result_path": None,
                        "bench_result_path": None,
                        "stable": None,
                        "notes": "Run with replacement list but lookup disabled/unavailable.",
                    }
                )

    return {
        "format_version": 1,
        "acceptance_gates": {
            "quality_max_ppl_delta": 2.0,
            "notes": [
                "Primary gate: at least one lookup configuration with ppl_delta <= 2.0 vs baseline.",
                "Record runtime stability observations for remove-only and lookup runs.",
            ],
        },
        "rows": rows,
    }


def _parse_ppl_from_text(text: str) -> Optional[float]:
    for pattern in PPL_PATTERNS:
        m = pattern.search(text)
        if m is not None:
            return float(m.group(1))
    return None


def parse_ppl_result(path: Path) -> float:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        raw = path.read_text(encoding="utf-8")
        if suffix == ".jsonl":
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    for key in ("mean_ppl", "ppl", "perplexity"):
                        if key in obj:
                            return float(obj[key])
        else:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                for key in ("mean_ppl", "ppl", "perplexity"):
                    if key in obj:
                        return float(obj[key])
        val = _parse_ppl_from_text(raw)
        if val is not None:
            return val
    else:
        text = path.read_text(encoding="utf-8")
        val = _parse_ppl_from_text(text)
        if val is not None:
            return val

    raise ValueError(f"could not parse perplexity value from '{path}'")


def _iter_llama_bench_rows(path: Path) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
        return
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            for row in obj:
                if isinstance(row, dict):
                    yield row
            return
        if isinstance(obj, dict):
            yield obj
            return
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)
        return
    raise ValueError(f"unsupported bench result format '{path.suffix}' for {path}")


def parse_bench_tok_s(path: Path) -> float:
    rows = list(_iter_llama_bench_rows(path))
    if not rows:
        raise ValueError(f"no rows found in bench result '{path}'")

    candidate = []
    fallback = []
    for row in rows:
        try:
            avg_ts = float(row["avg_ts"])
        except Exception:
            continue
        fallback.append(avg_ts)
        n_prompt = int(row.get("n_prompt", 0))
        n_gen = int(row.get("n_gen", 0))
        if n_prompt == 0 and n_gen > 0:
            candidate.append(avg_ts)

    values = candidate if candidate else fallback
    if not values:
        raise ValueError(f"could not parse avg_ts from bench result '{path}'")
    return float(statistics.mean(values))


def _materialize_row_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)

    if out.get("ppl") is None and out.get("ppl_result_path"):
        out["ppl"] = parse_ppl_result(Path(str(out["ppl_result_path"])))
    if out.get("tok_s") is None and out.get("bench_result_path"):
        out["tok_s"] = parse_bench_tok_s(Path(str(out["bench_result_path"])))

    return out


def _coerce_optional_bool(value: Any, field_name: str, row_id: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ValueError(f"row '{row_id}' has invalid {field_name} value: {value!r}")


def summarize_matrix(matrix: Dict[str, Any], require_complete: bool) -> Dict[str, Any]:
    rows_in = matrix.get("rows")
    if not isinstance(rows_in, list):
        raise ValueError("matrix JSON must contain a 'rows' list")

    rows = [_materialize_row_metrics(dict(r)) for r in rows_in if isinstance(r, dict)]
    baseline_rows = [r for r in rows if r.get("mode") == "baseline"]
    if len(baseline_rows) != 1:
        raise ValueError("matrix must contain exactly one baseline row")

    baseline = baseline_rows[0]
    baseline_ppl = None if baseline.get("ppl") is None else float(baseline["ppl"])
    baseline_tok_s = None if baseline.get("tok_s") is None else float(baseline["tok_s"])
    if baseline_ppl is None and baseline_tok_s is None:
        raise ValueError("baseline row must provide at least one metric (ppl and/or tok_s)")
    max_ppl_delta = float(matrix.get("acceptance_gates", {}).get("quality_max_ppl_delta", 2.0))

    computed_rows: List[Dict[str, Any]] = []
    for row in rows:
        mode = str(row.get("mode"))
        ppl = row.get("ppl")
        tok_s = row.get("tok_s")

        if require_complete and mode != "baseline" and (ppl is None or tok_s is None):
            raise ValueError(f"row '{row.get('id')}' is missing ppl/tok_s")

        ppl_delta = None if (ppl is None or baseline_ppl is None) else float(ppl) - baseline_ppl
        tok_s_delta_pct = None
        if tok_s is not None and baseline_tok_s is not None and baseline_tok_s > 0:
            tok_s_delta_pct = (float(tok_s) - baseline_tok_s) / baseline_tok_s * 100.0

        quality_pass = None if ppl_delta is None else (ppl_delta <= max_ppl_delta)
        stable_bool = _coerce_optional_bool(row.get("stable"), "stable", row.get("id"))

        row_out = dict(row)
        row_out["ppl_delta"] = ppl_delta
        row_out["tok_s_delta_pct"] = tok_s_delta_pct
        row_out["quality_gate_pass"] = quality_pass
        row_out["stability_pass"] = stable_bool
        computed_rows.append(row_out)

    lookup_rows = [r for r in computed_rows if r.get("mode") == "lookup"]
    passing_lookup = [
        r for r in lookup_rows
        if r.get("quality_gate_pass") is True and (r.get("stability_pass") is not False)
    ]
    quality_gate_candidates = [r for r in lookup_rows if r.get("quality_gate_pass") is not None]
    quality_gate_pass: Optional[bool]
    if quality_gate_candidates:
        quality_gate_pass = len([r for r in quality_gate_candidates if r.get("quality_gate_pass") is True]) > 0
    else:
        quality_gate_pass = None
    promotion_gate_pass = len(passing_lookup) > 0

    return {
        "baseline": {
            "id": baseline.get("id"),
            "ppl": baseline_ppl,
            "tok_s": baseline_tok_s,
        },
        "acceptance": {
            "quality_max_ppl_delta": max_ppl_delta,
            "quality_gate_pass": quality_gate_pass,
            "promotion_gate_pass": promotion_gate_pass,
            "best_lookup_by_ppl_delta": min(
                (
                    {"id": r.get("id"), "ppl_delta": r.get("ppl_delta"), "ppl": r.get("ppl")}
                    for r in lookup_rows
                    if r.get("ppl_delta") is not None
                ),
                key=lambda x: float(x["ppl_delta"]),
                default=None,
            ),
        },
        "rows": computed_rows,
    }


def _fmt_float(v: Any, ndigits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{float(v):.{ndigits}f}"


def _fmt_gate(v: Any) -> str:
    if v is None:
        return "N/A"
    return "PASS" if bool(v) else "FAIL"


def render_summary_markdown(summary: Dict[str, Any]) -> str:
    baseline = summary["baseline"]
    acceptance = summary["acceptance"]
    rows = summary["rows"]

    lines = []
    lines.append("# MoE Lookup Evaluation Matrix Summary")
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append(f"- Row: `{baseline['id']}`")
    lines.append(f"- PPL: {_fmt_float(baseline['ppl'])}")
    lines.append(f"- Throughput (tok/s): {_fmt_float(baseline['tok_s'])}")
    lines.append("")
    lines.append("## Acceptance gates")
    lines.append("")
    lines.append(f"- Quality gate (`ppl_delta <= {acceptance['quality_max_ppl_delta']}`): **{_fmt_gate(acceptance['quality_gate_pass'])}**")
    lines.append(f"- Promotion gate (quality pass + no stability failure): **{_fmt_gate(acceptance['promotion_gate_pass'])}**")
    best = acceptance.get("best_lookup_by_ppl_delta")
    if best is not None:
        lines.append(
            f"- Best lookup config by ppl delta: `{best['id']}` "
            f"(ppl={_fmt_float(best['ppl'])}, delta={_fmt_float(best['ppl_delta'])})"
        )
    lines.append("")
    lines.append("## Matrix")
    lines.append("")
    lines.append("| id | mode | replace_ratio | clusters | scaling | ppl | ppl_delta | tok/s | tok/s delta % | stable |")
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            "| {id} | {mode} | {rr} | {clusters} | {scaling} | {ppl} | {ppld} | {toks} | {toksd} | {stable} |".format(
                id=r.get("id", "-"),
                mode=r.get("mode", "-"),
                rr=_fmt_float(r.get("replace_ratio"), ndigits=3) if r.get("replace_ratio") is not None else "-",
                clusters=r.get("clusters_per_layer", "-") if r.get("clusters_per_layer") is not None else "-",
                scaling=r.get("scaling_mode", "-") if r.get("scaling_mode") is not None else "-",
                ppl=_fmt_float(r.get("ppl")),
                ppld=_fmt_float(r.get("ppl_delta")),
                toks=_fmt_float(r.get("tok_s"), ndigits=3),
                toksd=_fmt_float(r.get("tok_s_delta_pct"), ndigits=2),
                stable="-" if r.get("stable") is None else ("yes" if bool(r.get("stable")) else "no"),
            )
        )
    lines.append("")
    lines.append("## Runtime stability / perf observations")
    lines.append("")
    for r in rows:
        note = str(r.get("notes", "")).strip()
        if note:
            lines.append(f"- `{r.get('id')}`: {note}")
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()

    if args.cmd == "discover":
        matrix = discover_matrix(args.lookup_npz, baseline_id=args.baseline_id)
        _write_json(args.output, matrix)
        print(f"wrote matrix manifest: {args.output}")
        return 0

    if args.cmd == "summarize":
        matrix_raw = _read_json(args.matrix)
        summary = summarize_matrix(matrix_raw, require_complete=bool(args.require_complete))
        md = render_summary_markdown(summary)
        _write_text(args.output_md, md)
        if args.output_json is not None:
            _write_json(args.output_json, summary)
        print(f"wrote summary markdown: {args.output_md}")
        if args.output_json is not None:
            print(f"wrote summary json: {args.output_json}")
        return 0

    raise ValueError(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
