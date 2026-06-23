#!/usr/bin/env python3
"""Audit CurvyTron launch manifest checkpoint refs before a real launch."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from curvyzero.contracts.curvytron import curvytron_runs_volume_name


SCHEMA_ID = "curvyzero_curvytron_launch_manifest_ref_audit/v0"
ITERATION_CKPT_RE = re.compile(r"iteration_\d+\.pth\.tar\Z")
FROZEN_POLICY_KIND = "frozen_lightzero_checkpoint"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_refs_file(path: Path) -> list[tuple[int, str]]:
    refs: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        refs.append((line_number, line))
    if not refs:
        raise ValueError(f"{path} contains no checkpoint refs")
    return refs


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
    elif isinstance(value, dict):
        for item in value.values():
            if isinstance(item, dict):
                yield item


def _normalize_checkpoint_ref(ref: str) -> tuple[str, str | None]:
    text = str(ref).strip()
    if text.startswith("runs:"):
        return text.removeprefix("runs:").lstrip("/"), "runs"
    if text.startswith("control:"):
        return text.removeprefix("control:").lstrip("/"), "control"
    return text.lstrip("/"), None


def _add_ref(
    refs: dict[str, set[str]],
    *,
    ref: Any,
    source: str,
    skipped: list[dict[str, str]],
) -> None:
    if ref is None:
        return
    text = str(ref).strip()
    if not text:
        return
    normalized, volume_prefix = _normalize_checkpoint_ref(text)
    if not normalized.endswith(".pth.tar"):
        skipped.append({"source": source, "ref": text, "reason": "not a checkpoint ref"})
        return
    if volume_prefix == "control":
        refs[normalized].add(f"{source} [control-prefixed checkpoint ref]")
        return
    refs[normalized].add(source)


def _collect_mixture_refs(
    refs: dict[str, set[str]],
    mixture: Any,
    *,
    source: str,
    skipped: list[dict[str, str]],
) -> None:
    mixture_map = _as_mapping(mixture)
    for index, entry in enumerate(_iter_dicts(mixture_map.get("entries")), start=1):
        if entry.get("opponent_policy_kind") != FROZEN_POLICY_KIND:
            continue
        _add_ref(
            refs,
            ref=entry.get("opponent_checkpoint_ref"),
            source=f"{source}.entries[{index}].opponent_checkpoint_ref",
            skipped=skipped,
        )


def collect_checkpoint_refs(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    refs: dict[str, set[str]] = defaultdict(set)
    skipped: list[dict[str, str]] = []

    top_source = _as_mapping(manifest.get("top_checkpoint_source"))
    for name, row in top_source.items():
        row_map = _as_mapping(row)
        _add_ref(
            refs,
            ref=row_map.get("checkpoint_ref"),
            source=f"top_checkpoint_source.{name}.checkpoint_ref",
            skipped=skipped,
        )

    for row_index, row in enumerate(_iter_dicts(manifest.get("rows")), start=1):
        row_id = str(row.get("row_id") or row_index)
        _add_ref(
            refs,
            ref=row.get("initial_policy_checkpoint_ref"),
            source=f"rows[{row_id}].initial_policy_checkpoint_ref",
            skipped=skipped,
        )
        train_kwargs = _as_mapping(row.get("train_kwargs"))
        _add_ref(
            refs,
            ref=train_kwargs.get("initial_policy_checkpoint_ref"),
            source=f"rows[{row_id}].train_kwargs.initial_policy_checkpoint_ref",
            skipped=skipped,
        )
        _collect_mixture_refs(
            refs,
            train_kwargs.get("opponent_mixture_spec"),
            source=f"rows[{row_id}].train_kwargs.opponent_mixture_spec",
            skipped=skipped,
        )
        _collect_mixture_refs(
            refs,
            row.get("opponent_mixture_spec"),
            source=f"rows[{row_id}].opponent_mixture_spec",
            skipped=skipped,
        )
        assignment_preview = _as_mapping(row.get("opponent_assignment_preview"))
        _collect_mixture_refs(
            refs,
            {"entries": assignment_preview.get("entries")},
            source=f"rows[{row_id}].opponent_assignment_preview",
            skipped=skipped,
        )

    assignment_bank = _as_mapping(manifest.get("assignment_bank"))
    for recipe_id, artifact in _as_mapping(assignment_bank.get("assignments")).items():
        assignment = _as_mapping(artifact.get("assignment"))
        _collect_mixture_refs(
            refs,
            {"entries": assignment.get("entries")},
            source=f"assignment_bank.assignments.{recipe_id}.assignment",
            skipped=skipped,
        )

    return (
        [{"ref": ref, "sources": sorted(sources)} for ref, sources in sorted(refs.items())],
        skipped,
    )


def collect_refs_file_checkpoint_refs(
    refs_file: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    refs: dict[str, set[str]] = defaultdict(set)
    skipped: list[dict[str, str]] = []
    for line_number, ref in _load_refs_file(refs_file):
        _add_ref(
            refs,
            ref=ref,
            source=f"refs_file.line[{line_number}]",
            skipped=skipped,
        )
    return (
        [{"ref": ref, "sources": sorted(sources)} for ref, sources in sorted(refs.items())],
        skipped,
    )


def _syntax_problem(ref: str, sources: Sequence[str]) -> str | None:
    if "latest" in ref or "ckpt_best" in ref:
        return "mutable checkpoint ref"
    if not ITERATION_CKPT_RE.fullmatch(PurePosixPath(ref).name):
        return "checkpoint ref must end in iteration_N.pth.tar"
    if any("control-prefixed checkpoint ref" in source for source in sources):
        return "checkpoint refs must not use control: prefix"
    return None


def _check_local_exists(refs: Sequence[str], runs_root: Path) -> dict[str, bool]:
    return {ref: (runs_root / ref).is_file() for ref in refs}


def _modal_ls_parent(
    *,
    modal_bin: str,
    volume_name: str,
    parent: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    command = [modal_bin, "volume", "ls", volume_name, parent, "--json"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return [], {
            "parent": parent,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return [], {
            "parent": parent,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "json_error": str(exc),
        }
    if not isinstance(payload, list):
        return [], {
            "parent": parent,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "json_error": "modal volume ls JSON payload was not a list",
        }
    return payload, None


def _check_modal_exists(
    refs: Sequence[str],
    *,
    modal_bin: str,
    volume_name: str,
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    refs_by_parent: dict[str, set[str]] = defaultdict(set)
    for ref in refs:
        parent = str(PurePosixPath(ref).parent)
        refs_by_parent[parent].add(ref)

    exists: dict[str, bool] = {}
    parent_errors: list[dict[str, Any]] = []
    for parent, parent_refs in sorted(refs_by_parent.items()):
        listing, parent_error = _modal_ls_parent(
            modal_bin=modal_bin,
            volume_name=volume_name,
            parent=parent,
        )
        if parent_error is not None:
            parent_errors.append(parent_error)
        filenames = {
            str(item.get("Filename") or "").lstrip("/")
            for item in listing
            if isinstance(item, dict)
        }
        for ref in parent_refs:
            exists[ref] = ref.lstrip("/") in filenames
    return exists, parent_errors


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if (args.manifest is None) == (args.refs_file is None):
        raise ValueError("provide exactly one input: MANIFEST or --refs-file")
    runs_volume_name = str(args.runs_volume_name or curvytron_runs_volume_name()).strip()
    if not args.allow_non_v2_runs_volume and not runs_volume_name.endswith("-v2"):
        raise ValueError(f"runs volume must be all-v2 for launch audit: {runs_volume_name}")

    if args.refs_file is not None:
        input_kind = "refs_file"
        input_path = args.refs_file
        refs, skipped = collect_refs_file_checkpoint_refs(args.refs_file)
    else:
        input_kind = "manifest"
        input_path = args.manifest
        refs, skipped = collect_checkpoint_refs(_load_json(args.manifest))
    ref_texts = [entry["ref"] for entry in refs]
    local_exists = (
        _check_local_exists(ref_texts, args.runs_root) if args.runs_root is not None else {}
    )
    modal_parent_errors: list[dict[str, Any]] = []
    if args.check_modal:
        modal_exists, modal_parent_errors = _check_modal_exists(
            ref_texts,
            modal_bin=args.modal_bin,
            volume_name=runs_volume_name,
        )
    else:
        modal_exists = {}

    checked_refs = []
    bad_refs = []
    missing_refs = []
    for entry in refs:
        ref = entry["ref"]
        sources = list(entry["sources"])
        syntax_problem = _syntax_problem(ref, sources)
        status = "ok"
        if syntax_problem:
            status = "bad_ref"
            bad_refs.append({"ref": ref, "reason": syntax_problem, "sources": sources})
        local_ok = local_exists.get(ref)
        modal_ok = modal_exists.get(ref)
        existence_values = [
            value for value in (local_ok, modal_ok) if value is not None
        ]
        if not syntax_problem and existence_values and not any(existence_values):
            status = "missing"
            missing_refs.append({"ref": ref, "sources": sources})
        checked_refs.append(
            {
                "ref": ref,
                "sources": sources,
                "status": status,
                "syntax_problem": syntax_problem,
                "local_exists": local_ok,
                "modal_exists": modal_ok,
            }
        )

    existence_checked = args.runs_root is not None or args.check_modal
    ok = not bad_refs and not missing_refs and (existence_checked or args.syntax_only)
    return {
        "schema_id": SCHEMA_ID,
        "input_kind": input_kind,
        "input_path": str(input_path),
        "manifest": str(args.manifest) if args.manifest is not None else None,
        "refs_file": str(args.refs_file) if args.refs_file is not None else None,
        "runs_volume_name": runs_volume_name,
        "runs_root": str(args.runs_root) if args.runs_root is not None else None,
        "check_modal": bool(args.check_modal),
        "syntax_only": bool(args.syntax_only),
        "existence_checked": existence_checked,
        "ok": ok,
        "ref_count": len(checked_refs),
        "bad_ref_count": len(bad_refs),
        "missing_ref_count": len(missing_refs),
        "modal_parent_error_count": len(modal_parent_errors),
        "skipped_count": len(skipped),
        "bad_refs": bad_refs,
        "missing_refs": missing_refs,
        "modal_parent_errors": modal_parent_errors,
        "refs": checked_refs,
        "skipped": skipped,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument(
        "--refs-file",
        type=Path,
        default=None,
        help="audit a newline-delimited checkpoint ref file instead of a launch manifest",
    )
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--runs-volume-name", default=curvytron_runs_volume_name())
    parser.add_argument("--check-modal", action="store_true")
    parser.add_argument("--modal-bin", default="modal")
    parser.add_argument(
        "--syntax-only",
        action="store_true",
        help="allow success without local or Modal existence checks",
    )
    parser.add_argument("--allow-non-v2-runs-volume", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
