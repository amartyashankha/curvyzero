#!/usr/bin/env python3
"""Copy verified CurvyTron checkpoint refs from an old Volume into v2."""

from __future__ import annotations

import argparse
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from curvyzero.contracts.curvytron import (
    DEFAULT_CURVYTRON_RUNS_VOLUME_NAME,
    modal_volume_kwargs_for_name,
)


SCHEMA_ID = "curvyzero_curvytron_checkpoint_rematerialization/v0"
ITERATION_CKPT_RE = re.compile(r"iteration_\d+\.pth\.tar\Z")


def _load_refs_file(path: Path) -> list[str]:
    refs = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not refs:
        raise ValueError(f"{path} contains no checkpoint refs")
    if len(refs) != len(set(refs)):
        raise ValueError(f"{path} contains duplicate checkpoint refs")
    for ref in refs:
        _validate_checkpoint_ref(ref)
    return refs


def _validate_checkpoint_ref(ref: str) -> None:
    if ref.startswith(("runs:", "control:")):
        raise ValueError(f"checkpoint ref must be volume-relative, got {ref!r}")
    if "latest" in ref or "ckpt_best" in ref:
        raise ValueError(f"checkpoint ref is mutable: {ref}")
    if not ITERATION_CKPT_RE.fullmatch(PurePosixPath(ref).name):
        raise ValueError(f"checkpoint ref must end in iteration_N.pth.tar: {ref}")


def _volume_kwargs(name: str, *, source: bool) -> dict[str, Any]:
    if source and not name.endswith("-v2"):
        return {"create_if_missing": False}
    return modal_volume_kwargs_for_name(name, create_if_missing=not source)


def _remote_path(ref: str) -> str:
    return "/" + ref.lstrip("/")


def rematerialize(args: argparse.Namespace) -> dict[str, Any]:
    if not args.target_volume.endswith("-v2"):
        raise ValueError(f"target volume must be all-v2, got {args.target_volume!r}")
    if not args.allow_non_v2_source and not args.source_volume.endswith("-v2"):
        raise ValueError(
            f"source volume is non-v2: {args.source_volume!r}; pass "
            "--allow-non-v2-source for explicit migration reads"
        )

    refs = _load_refs_file(args.refs_file)
    if args.limit is not None:
        refs = refs[: args.limit]

    records: list[dict[str, Any]] = []
    if not args.dry_run:
        import modal

        source_volume = modal.Volume.from_name(
            args.source_volume,
            environment_name=args.modal_env,
            **_volume_kwargs(args.source_volume, source=True),
        )
        target_volume = modal.Volume.from_name(
            args.target_volume,
            environment_name=args.modal_env,
            **_volume_kwargs(args.target_volume, source=False),
        )
    else:
        source_volume = None
        target_volume = None

    for index, ref in enumerate(refs, start=1):
        remote_path = _remote_path(ref)
        record: dict[str, Any] = {
            "index": index,
            "ref": ref,
            "source_volume": args.source_volume,
            "target_volume": args.target_volume,
            "remote_path": remote_path,
        }
        if args.dry_run:
            records.append({"status": "dry_run", **record})
            continue
        buffer = io.BytesIO()
        bytes_read = source_volume.read_file_into_fileobj(remote_path, buffer)
        buffer.seek(0)
        with target_volume.batch_upload(force=True) as batch:
            batch.put_file(buffer, remote_path)
        records.append({"status": "copied", "bytes": bytes_read, **record})
        if args.progress_every and index % args.progress_every == 0:
            print(f"copied {index}/{len(refs)} refs", flush=True)

    copied_count = sum(1 for record in records if record["status"] == "copied")
    return {
        "schema_id": SCHEMA_ID,
        "generated_at": datetime.now(UTC).isoformat(),
        "refs_file": str(args.refs_file),
        "source_volume": args.source_volume,
        "target_volume": args.target_volume,
        "dry_run": bool(args.dry_run),
        "ref_count": len(refs),
        "copied_count": copied_count,
        "records": records,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs-file", type=Path, required=True)
    parser.add_argument("--source-volume", default="curvyzero-runs")
    parser.add_argument("--target-volume", default=DEFAULT_CURVYTRON_RUNS_VOLUME_NAME)
    parser.add_argument("--allow-non-v2-source", action="store_true")
    parser.add_argument("--modal-env", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = rematerialize(args)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not args.dry_run and report["copied_count"] != report["ref_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
