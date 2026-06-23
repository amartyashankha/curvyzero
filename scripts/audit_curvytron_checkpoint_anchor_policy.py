#!/usr/bin/env python3
"""Audit CurvyTron best-known checkpoint anchor policy for Wave A manifests."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA_ID = "curvyzero_curvytron_checkpoint_anchor_policy_audit/v0"
TOP10_REFS_FILE = (
    "docs/working/training/r18fresh_postmortem_2026-05-16/"
    "TOP10_RAW_REFS_auto-r000032-g22-555c999b.txt"
)
STATIC_TOP4NZ_REFS_FILE = (
    "artifacts/local/curvytron_no_tournament_control_20260516/source/"
    "static_top4_nonzero_refs.txt"
)
STATIC_TOP4NZ_MODAL_AUDIT = (
    "artifacts/local/curvytron_no_tournament_control_20260516/source/"
    "static_top4_nonzero_refs.ref_audit.modal.json"
)
RANK_LINE_RE = re.compile(
    r"rank=(?P<rank>\d+)\s+rating=(?P<rating>[^\s]+)\s+checkpoint_id=(?P<checkpoint_id>\S+)"
)


@dataclass(frozen=True)
class ManifestSpec:
    lane_id: str
    manifest_relpath: str


def _tonight18_relpath(family: str) -> str:
    return f"artifacts/local/curvytron_tonight18_manifests/{family}/{family}.json"


def _default_manifest_specs(non_rnd_seed_profile: str = "top4nz") -> list[ManifestSpec]:
    if non_rnd_seed_profile not in {"top4nz", "bestseed"}:
        raise ValueError(f"unknown non-RND seed profile: {non_rnd_seed_profile}")
    if non_rnd_seed_profile == "bestseed":
        static_lane_id = "static-bestseed-top4nz"
        static_family = "reward-static-bestseed-top4nz-h100-wave-a-20260623a"
    else:
        static_lane_id = "static-top4nz"
        static_family = "reward-static-top4nz-h100-wave-a-repair-20260623a"
    specs = [
        ManifestSpec(
            lane_id=static_lane_id,
            manifest_relpath=_tonight18_relpath(static_family),
        ),
    ]
    for replica in range(1, 7):
        if non_rnd_seed_profile == "bestseed":
            lane_id = f"long-horizon-bestseed-rep{replica:02d}"
            family = (
                f"reward-lhpre-bestseed-top4nz-rep{replica:02d}-h100-wave-a-20260623a"
            )
        else:
            lane_id = f"long-horizon-rep{replica:02d}"
            family = f"reward-lhpre-top4nz-rep{replica:02d}-h100-wave-a-repair-20260623a"
        specs.append(ManifestSpec(lane_id=lane_id, manifest_relpath=_tonight18_relpath(family)))
    for suffix in (
        "s25-b128-td25-cap1024",
        "s25-b128-td25-cap2048",
        "s25-b256-td25-cap2048",
    ):
        if non_rnd_seed_profile == "bestseed":
            lane_id = f"cadence-support-bestseed-{suffix}"
            family = f"reward-csupport-bestseed-top4nz-{suffix}-wave-a-20260623a"
        else:
            lane_id = f"cadence-support-{suffix}"
            family = f"reward-csupport-top4nz-{suffix}-wave-a-repair-20260623a"
        specs.append(ManifestSpec(lane_id=lane_id, manifest_relpath=_tonight18_relpath(family)))
    return specs


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _read_refs_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    refs = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("rank="):
            continue
        refs.append(line)
    return refs


def _parse_top10(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    entries: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        match = RANK_LINE_RE.fullmatch(line)
        if not match:
            index += 1
            continue
        checkpoint_ref = ""
        next_index = index + 1
        while next_index < len(lines):
            candidate = lines[next_index].strip()
            if candidate and not candidate.startswith("#"):
                checkpoint_ref = candidate
                break
            next_index += 1
        entry = match.groupdict()
        entries.append(
            {
                "rank": int(entry["rank"]),
                "rating": float(entry["rating"]),
                "checkpoint_id": entry["checkpoint_id"],
                "checkpoint_ref": checkpoint_ref,
            }
        )
        index = next_index + 1
    return entries


def _manifest_seed_report(
    *,
    repo_root: Path,
    spec: ManifestSpec,
    historical_best_seed_ref: str | None,
    top4nz_rank1_ref: str | None,
) -> dict[str, Any]:
    path = repo_root / spec.manifest_relpath
    manifest = _load_json(path)
    if manifest is None:
        return {
            "lane_id": spec.lane_id,
            "manifest": spec.manifest_relpath,
            "exists": False,
            "row_count": 0,
            "unique_initial_policy_checkpoint_refs": [],
            "uses_historical_best_seed": False,
            "uses_top4nz_rank1_seed": False,
        }
    refs = sorted(
        {
            str(row.get("initial_policy_checkpoint_ref"))
            for row in _rows(manifest)
            if row.get("initial_policy_checkpoint_ref")
        }
    )
    return {
        "lane_id": spec.lane_id,
        "manifest": spec.manifest_relpath,
        "exists": True,
        "row_count": len(_rows(manifest)),
        "unique_initial_policy_checkpoint_refs": refs,
        "unique_initial_policy_checkpoint_ref_count": len(refs),
        "uses_historical_best_seed": bool(historical_best_seed_ref and refs == [historical_best_seed_ref]),
        "uses_top4nz_rank1_seed": bool(top4nz_rank1_ref and refs == [top4nz_rank1_ref]),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    top10_entries = _parse_top10(repo_root / args.top10_refs_file)
    historical_best = next((entry for entry in top10_entries if entry.get("rank") == 1), None)
    historical_best_ref = str(historical_best.get("checkpoint_ref")) if historical_best else None
    if not historical_best_ref:
        errors.append({"message": "historical rank-1 seed ref missing", "path": args.top10_refs_file})

    top4nz_refs = _read_refs_file(repo_root / args.static_top4nz_refs_file)
    top4nz_rank1_ref = top4nz_refs[0] if top4nz_refs else None
    if not top4nz_refs:
        errors.append({"message": "static top4nz refs file missing or empty", "path": args.static_top4nz_refs_file})

    top4nz_audit = _load_json(repo_root / args.static_top4nz_modal_audit)
    top4nz_modal_ok = bool(top4nz_audit and top4nz_audit.get("ok") is True)
    if not top4nz_modal_ok:
        warnings.append(
            {
                "message": "static top4nz Modal audit is missing or not ok",
                "path": args.static_top4nz_modal_audit,
            }
        )

    manifest_reports = [
        _manifest_seed_report(
            repo_root=repo_root,
            spec=spec,
            historical_best_seed_ref=historical_best_ref,
            top4nz_rank1_ref=top4nz_rank1_ref,
        )
        for spec in _default_manifest_specs(args.non_rnd_seed_profile)
    ]
    missing_manifests = [row for row in manifest_reports if not row["exists"]]
    for row in missing_manifests:
        errors.append({"message": "manifest missing", "lane_id": row["lane_id"], "manifest": row["manifest"]})

    non_missing = [row for row in manifest_reports if row["exists"]]
    historical_best_manifest_count = sum(1 for row in non_missing if row["uses_historical_best_seed"])
    top4nz_seed_manifest_count = sum(1 for row in non_missing if row["uses_top4nz_rank1_seed"])
    mixed_seed_manifests = [
        row
        for row in non_missing
        if row.get("unique_initial_policy_checkpoint_ref_count", 0) > 1
    ]
    for row in mixed_seed_manifests:
        warnings.append(
            {
                "message": "manifest uses multiple initial policy checkpoint refs",
                "lane_id": row["lane_id"],
                "refs": row["unique_initial_policy_checkpoint_refs"],
            }
        )

    if non_missing and historical_best_manifest_count == 0:
        warnings.append(
            {
                "message": (
                    "current repaired manifests do not use the historical r18fresh rank-1 "
                    "checkpoint as their initial seed"
                ),
                "historical_best_seed_ref": historical_best_ref,
                "top4nz_seed_manifest_count": top4nz_seed_manifest_count,
            }
        )
    if args.require_best_known_seed and non_missing and historical_best_manifest_count != len(non_missing):
        errors.append(
            {
                "message": "one or more manifests violate --require-best-known-seed",
                "manifest_count": len(non_missing),
                "historical_best_seed_manifest_count": historical_best_manifest_count,
            }
        )

    if args.non_rnd_seed_profile == "bestseed":
        recommendation = (
            "Use the bestseed non-RND manifests only after fresh Modal ref, packet, "
            "capacity, and explicit launch-approval checks pass."
        )
    else:
        recommendation = (
            "Decide explicitly whether to launch the repaired top4nz-seeded manifests "
            "or switch to the prepared bestseed non-RND manifests after a fresh Modal "
            "existence audit."
        )

    report = {
        "schema_id": SCHEMA_ID,
        "ok": not errors,
        "policy": {
            "best_known_seed_default": "historical_r18fresh_rank1_iteration_180000",
            "launchable_repair_seed": "static_top4nz_rank1",
            "preferred_medium_long_seed": "historical_r18fresh_rank1_iteration_180000",
            "require_best_known_seed": bool(args.require_best_known_seed),
        },
        "non_rnd_seed_profile": args.non_rnd_seed_profile,
        "historical_top10_refs_file": args.top10_refs_file,
        "historical_best_seed": historical_best,
        "historical_top10_count": len(top10_entries),
        "static_top4nz_refs_file": args.static_top4nz_refs_file,
        "static_top4nz_ref_count": len(top4nz_refs),
        "static_top4nz_rank1_ref": top4nz_rank1_ref,
        "static_top4nz_modal_audit": {
            "path": args.static_top4nz_modal_audit,
            "ok": top4nz_modal_ok,
            "ref_count": top4nz_audit.get("ref_count") if top4nz_audit else None,
            "missing_ref_count": top4nz_audit.get("missing_ref_count") if top4nz_audit else None,
            "modal_parent_error_count": top4nz_audit.get("modal_parent_error_count") if top4nz_audit else None,
        },
        "manifest_seed_summary": {
            "manifest_count": len(non_missing),
            "historical_best_seed_manifest_count": historical_best_manifest_count,
            "top4nz_seed_manifest_count": top4nz_seed_manifest_count,
            "mixed_seed_manifest_count": len(mixed_seed_manifests),
        },
        "manifests": manifest_reports,
        "recommendation": recommendation,
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--top10-refs-file", default=TOP10_REFS_FILE)
    parser.add_argument("--static-top4nz-refs-file", default=STATIC_TOP4NZ_REFS_FILE)
    parser.add_argument("--static-top4nz-modal-audit", default=STATIC_TOP4NZ_MODAL_AUDIT)
    parser.add_argument(
        "--non-rnd-seed-profile",
        choices=("top4nz", "bestseed"),
        default="top4nz",
        help="Choose the prepared non-RND manifest family to audit.",
    )
    parser.add_argument(
        "--require-best-known-seed",
        action="store_true",
        help="Fail if repaired non-RND manifests do not all use the historical best seed.",
    )
    parser.add_argument("--output", help="Optional output path for JSON report.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
