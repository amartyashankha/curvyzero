#!/usr/bin/env python3
"""Plan staged CurvyTron Wave A launch profiles without launching jobs."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA_ID = "curvyzero_curvytron_wave_a_staged_launch_plan/v0"
SUBMIT_SCRIPT = "scripts/submit_curvytron_survivaldiag_manifest.py"
PACKET_AUDIT_SCRIPT = "scripts/audit_curvytron_wave_a_launch_packet.py"
ANCHOR_AUDIT_SCRIPT = "scripts/audit_curvytron_checkpoint_anchor_policy.py"
CAPACITY_AUDIT_SCRIPT = "scripts/audit_curvytron_wave_a_capacity.py"
EXPECTED_SELECTED_ROW_IDS = ("r005", "r011", "r017")


@dataclass(frozen=True)
class LaneSpec:
    lane_id: str
    lane_group: str
    manifest_relpath: str


@dataclass(frozen=True)
class LaneSelection:
    lane_id: str
    row_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ProfileSpec:
    profile_id: str
    tier: str
    intended_runtime: str
    max_active_h100_rows: int | None
    selections: tuple[LaneSelection, ...]
    purpose: str
    non_rnd_seed_profile: str = "top4nz"


def _tonight18_relpath(family: str) -> str:
    return f"artifacts/local/curvytron_tonight18_manifests/{family}/{family}.json"


def _default_lanes() -> dict[str, LaneSpec]:
    rnd_family = "rnd-blank-h100-wave-a-20260623a"
    lanes = {
        "rnd-blank-sweep": LaneSpec(
            lane_id="rnd-blank-sweep",
            lane_group="rnd",
            manifest_relpath=(
                "artifacts/local/curvytron_rnd_blank_sweep_manifests/"
                f"{rnd_family}/{rnd_family}.json"
            ),
        ),
        "static-top4nz": LaneSpec(
            lane_id="static-top4nz",
            lane_group="non_rnd_static",
            manifest_relpath=_tonight18_relpath(
                "reward-static-top4nz-h100-wave-a-repair-20260623a"
            ),
        ),
        "static-bestseed-top4nz": LaneSpec(
            lane_id="static-bestseed-top4nz",
            lane_group="non_rnd_static",
            manifest_relpath=_tonight18_relpath(
                "reward-static-bestseed-top4nz-h100-wave-a-20260623a"
            ),
        ),
    }
    for replica in range(1, 7):
        family = f"reward-lhpre-top4nz-rep{replica:02d}-h100-wave-a-repair-20260623a"
        lanes[f"long-horizon-rep{replica:02d}"] = LaneSpec(
            lane_id=f"long-horizon-rep{replica:02d}",
            lane_group="non_rnd_long_horizon",
            manifest_relpath=_tonight18_relpath(family),
        )
        bestseed_family = (
            f"reward-lhpre-bestseed-top4nz-rep{replica:02d}-h100-wave-a-20260623a"
        )
        lanes[f"long-horizon-bestseed-rep{replica:02d}"] = LaneSpec(
            lane_id=f"long-horizon-bestseed-rep{replica:02d}",
            lane_group="non_rnd_long_horizon",
            manifest_relpath=_tonight18_relpath(bestseed_family),
        )
    for suffix in (
        "s25-b128-td25-cap1024",
        "s25-b128-td25-cap2048",
        "s25-b256-td25-cap2048",
    ):
        family = f"reward-csupport-top4nz-{suffix}-wave-a-repair-20260623a"
        lanes[f"cadence-support-{suffix}"] = LaneSpec(
            lane_id=f"cadence-support-{suffix}",
            lane_group="non_rnd_cadence_support",
            manifest_relpath=_tonight18_relpath(family),
        )
        bestseed_family = f"reward-csupport-bestseed-top4nz-{suffix}-wave-a-20260623a"
        lanes[f"cadence-support-bestseed-{suffix}"] = LaneSpec(
            lane_id=f"cadence-support-bestseed-{suffix}",
            lane_group="non_rnd_cadence_support",
            manifest_relpath=_tonight18_relpath(bestseed_family),
        )
    return lanes


def _row_range(start: int, stop: int) -> tuple[str, ...]:
    return tuple(f"r{index:03d}" for index in range(start, stop + 1))


def _default_profiles() -> dict[str, ProfileSpec]:
    long_triad = (
        LaneSelection("static-top4nz", EXPECTED_SELECTED_ROW_IDS),
        LaneSelection("long-horizon-rep01", EXPECTED_SELECTED_ROW_IDS),
        LaneSelection("cadence-support-s25-b128-td25-cap1024", EXPECTED_SELECTED_ROW_IDS),
    )
    bestseed_long_triad = (
        LaneSelection("static-bestseed-top4nz", EXPECTED_SELECTED_ROW_IDS),
        LaneSelection("long-horizon-bestseed-rep01", EXPECTED_SELECTED_ROW_IDS),
        LaneSelection("cadence-support-bestseed-s25-b128-td25-cap1024", EXPECTED_SELECTED_ROW_IDS),
    )
    return {
        "short90": ProfileSpec(
            profile_id="short90",
            tier="short_breadth",
            intended_runtime="<=2h",
            max_active_h100_rows=100,
            purpose="Full prepared packet for short health/breadth signal.",
            selections=(
                LaneSelection("rnd-blank-sweep"),
                LaneSelection("static-top4nz"),
                *(LaneSelection(f"long-horizon-rep{replica:02d}", EXPECTED_SELECTED_ROW_IDS) for replica in range(1, 7)),
                LaneSelection("cadence-support-s25-b128-td25-cap1024", EXPECTED_SELECTED_ROW_IDS),
                LaneSelection("cadence-support-s25-b128-td25-cap2048", EXPECTED_SELECTED_ROW_IDS),
                LaneSelection("cadence-support-s25-b256-td25-cap2048", EXPECTED_SELECTED_ROW_IDS),
            ),
        ),
        "short90_bestseed": ProfileSpec(
            profile_id="short90_bestseed",
            tier="short_breadth",
            intended_runtime="<=2h",
            max_active_h100_rows=100,
            purpose=(
                "Full prepared packet for short health/breadth signal, with non-RND "
                "rows seeded from the historical r18fresh champion."
            ),
            non_rnd_seed_profile="bestseed",
            selections=(
                LaneSelection("rnd-blank-sweep"),
                LaneSelection("static-bestseed-top4nz"),
                *(
                    LaneSelection(
                        f"long-horizon-bestseed-rep{replica:02d}",
                        EXPECTED_SELECTED_ROW_IDS,
                    )
                    for replica in range(1, 7)
                ),
                LaneSelection("cadence-support-bestseed-s25-b128-td25-cap1024", EXPECTED_SELECTED_ROW_IDS),
                LaneSelection("cadence-support-bestseed-s25-b128-td25-cap2048", EXPECTED_SELECTED_ROW_IDS),
                LaneSelection("cadence-support-bestseed-s25-b256-td25-cap2048", EXPECTED_SELECTED_ROW_IDS),
            ),
        ),
        "mid36": ProfileSpec(
            profile_id="mid36",
            tier="medium_read",
            intended_runtime="2h-8h",
            max_active_h100_rows=40,
            purpose="Medium run preserving RND controls and the full static extrinsic isolate.",
            selections=(
                LaneSelection("rnd-blank-sweep", _row_range(1, 18)),
                LaneSelection("static-top4nz"),
            ),
        ),
        "mid36_bestseed": ProfileSpec(
            profile_id="mid36_bestseed",
            tier="medium_read",
            intended_runtime="2h-8h",
            max_active_h100_rows=40,
            purpose=(
                "Medium run preserving RND controls and the full static extrinsic "
                "isolate, with non-RND seeded from the historical r18fresh champion."
            ),
            non_rnd_seed_profile="bestseed",
            selections=(
                LaneSelection("rnd-blank-sweep", _row_range(1, 18)),
                LaneSelection("static-bestseed-top4nz"),
            ),
        ),
        "long18_all_weights": ProfileSpec(
            profile_id="long18_all_weights",
            tier="long_read",
            intended_runtime="8h+",
            max_active_h100_rows=20,
            purpose="Long run retaining one full RND weight ladder plus the smallest non-RND triad.",
            selections=(
                LaneSelection("rnd-blank-sweep", _row_range(1, 9)),
                *long_triad,
            ),
        ),
        "long17_no_highest_weight": ProfileSpec(
            profile_id="long17_no_highest_weight",
            tier="long_read",
            intended_runtime="8h+",
            max_active_h100_rows=20,
            purpose=(
                "Long capacity-fit run retaining stock, meter, low/mid RND weights, "
                "and the smallest non-RND triad; drops only the highest RND weight."
            ),
            selections=(
                LaneSelection("rnd-blank-sweep", _row_range(1, 8)),
                *long_triad,
            ),
        ),
        "long18_all_weights_bestseed": ProfileSpec(
            profile_id="long18_all_weights_bestseed",
            tier="long_read",
            intended_runtime="8h+",
            max_active_h100_rows=20,
            purpose=(
                "Long run retaining one full RND weight ladder plus the smallest "
                "bestseed non-RND triad."
            ),
            non_rnd_seed_profile="bestseed",
            selections=(
                LaneSelection("rnd-blank-sweep", _row_range(1, 9)),
                *bestseed_long_triad,
            ),
        ),
        "long17_no_highest_weight_bestseed": ProfileSpec(
            profile_id="long17_no_highest_weight_bestseed",
            tier="long_read",
            intended_runtime="8h+",
            max_active_h100_rows=20,
            purpose=(
                "Long capacity-fit run retaining stock, meter, low/mid RND weights, "
                "and the smallest bestseed non-RND triad; drops only the highest RND weight."
            ),
            non_rnd_seed_profile="bestseed",
            selections=(
                LaneSelection("rnd-blank-sweep", _row_range(1, 8)),
                *bestseed_long_triad,
            ),
        ),
        "long19_low_weight_replicated": ProfileSpec(
            profile_id="long19_low_weight_replicated",
            tier="long_read",
            intended_runtime="8h+",
            max_active_h100_rows=20,
            purpose="Long run focused on two replicas of stock, meter, and low positive RND weights.",
            selections=(
                LaneSelection(
                    "rnd-blank-sweep",
                    ("r001", "r002", "r003", "r004", "r005", "r010", "r011", "r012", "r013", "r014"),
                ),
                *long_triad,
            ),
        ),
        "long19_low_weight_replicated_bestseed": ProfileSpec(
            profile_id="long19_low_weight_replicated_bestseed",
            tier="long_read",
            intended_runtime="8h+",
            max_active_h100_rows=20,
            purpose=(
                "Long run focused on two replicas of stock, meter, and low positive "
                "RND weights, with a bestseed non-RND triad."
            ),
            non_rnd_seed_profile="bestseed",
            selections=(
                LaneSelection(
                    "rnd-blank-sweep",
                    ("r001", "r002", "r003", "r004", "r005", "r010", "r011", "r012", "r013", "r014"),
                ),
                *bestseed_long_triad,
            ),
        ),
    }


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


def _row_ids_label(row_ids: Sequence[str]) -> str:
    if not row_ids:
        return "all"
    ranges: list[str] = []
    start = previous = row_ids[0]

    def _suffix(row_id: str) -> int:
        try:
            return int(row_id.removeprefix("r"))
        except ValueError:
            return -1

    for row_id in row_ids[1:]:
        if _suffix(row_id) == _suffix(previous) + 1:
            previous = row_id
            continue
        ranges.append(start if start == previous else f"{start}-{previous}")
        start = previous = row_id
    ranges.append(start if start == previous else f"{start}-{previous}")
    return "-".join(ranges)


def _launch_output_relpath(manifest_relpath: str, row_ids: Sequence[str]) -> str:
    path = Path(manifest_relpath)
    suffix = ".submit.launch.json" if not row_ids else f".selected-{_row_ids_label(row_ids)}.submit.launch.json"
    return str(path.with_name(f"{path.stem}{suffix}"))


def _command(manifest_relpath: str, row_ids: Sequence[str], output_relpath: str) -> str:
    parts = [
        "uv",
        "run",
        "--extra",
        "modal",
        "python",
        SUBMIT_SCRIPT,
        manifest_relpath,
    ]
    for row_id in row_ids:
        parts.extend(["--row-id", row_id])
    parts.append("--allow-launch")
    if row_ids:
        parts.append("--allow-partial-launch")
    parts.extend(["--output", output_relpath])
    return shlex.join(parts)


def _packet_audit_output(seed_profile: str) -> str:
    suffix = "_bestseed" if seed_profile == "bestseed" else ""
    return f"artifacts/local/curvytron_wave_a_launch_packet_audit{suffix}_20260623a.json"


def _anchor_audit_output(seed_profile: str) -> str:
    suffix = "_bestseed" if seed_profile == "bestseed" else ""
    return f"artifacts/local/curvytron_checkpoint_anchor_policy_audit{suffix}_20260623a.json"


def _packet_audit_command(seed_profile: str) -> str:
    parts = ["uv", "run", "python", PACKET_AUDIT_SCRIPT]
    if seed_profile == "bestseed":
        parts.extend(["--non-rnd-seed-profile", "bestseed"])
    parts.extend(["--output", _packet_audit_output(seed_profile)])
    return shlex.join(parts)


def _anchor_audit_command(seed_profile: str) -> str:
    parts = ["uv", "run", "python", ANCHOR_AUDIT_SCRIPT]
    if seed_profile == "bestseed":
        parts.extend(["--non-rnd-seed-profile", "bestseed", "--require-best-known-seed"])
    parts.extend(["--output", _anchor_audit_output(seed_profile)])
    return shlex.join(parts)


def _capacity_audit_output(profile_id: str) -> str:
    return f"artifacts/local/curvytron_wave_a_capacity_snapshot_{profile_id}_20260623a.json"


def _capacity_audit_command(profile_id: str, requested_h100_rows: int) -> str:
    return shlex.join(
        [
            "uv",
            "run",
            "python",
            CAPACITY_AUDIT_SCRIPT,
            "--requested-h100-rows",
            str(requested_h100_rows),
            "--output",
            _capacity_audit_output(profile_id),
        ]
    )


def _prelaunch_checks(profile: ProfileSpec, total_rows: int) -> list[dict[str, Any]]:
    return [
        {
            "check": "packet_audit",
            "required": True,
            "command": _packet_audit_command(profile.non_rnd_seed_profile),
            "expected": {
                "ok": True,
                "launch_artifacts": [],
            },
        },
        {
            "check": "checkpoint_anchor_policy",
            "required": True,
            "command": _anchor_audit_command(profile.non_rnd_seed_profile),
            "expected": {
                "ok": True,
                "non_rnd_seed_profile": profile.non_rnd_seed_profile,
            },
        },
        {
            "check": "capacity_snapshot",
            "required": True,
            "command": _capacity_audit_command(profile.profile_id, total_rows),
            "expected": {
                "requested_h100_rows": total_rows,
                "approval_recommendation": "capacity_proxy_clear or operator_capacity_review_required",
                "max_active_h100_rows": profile.max_active_h100_rows,
            },
        },
    ]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rnd_key(row: dict[str, Any]) -> str | None:
    bonus = _mapping(row.get("exploration_bonus"))
    train_kwargs = _mapping(row.get("train_kwargs"))
    mode = bonus.get("mode") or train_kwargs.get("exploration_bonus_mode")
    if not mode:
        return None
    weight = bonus.get("weight", train_kwargs.get("exploration_bonus_weight"))
    return f"{mode}:{weight}"


def _reward_variant(row: dict[str, Any]) -> str | None:
    train_kwargs = _mapping(row.get("train_kwargs"))
    value = row.get("reward_variant") or train_kwargs.get("reward_variant")
    return str(value) if value is not None else None


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    lanes = _default_lanes()
    profiles = _default_profiles()
    profile = profiles.get(args.profile)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if profile is None:
        errors.append({"message": "unknown profile", "profile": args.profile, "known_profiles": sorted(profiles)})
        return {
            "schema_id": SCHEMA_ID,
            "ok": False,
            "profile": args.profile,
            "errors": errors,
            "warnings": warnings,
            "error_count": len(errors),
            "warning_count": len(warnings),
        }

    repo_root = Path(args.repo_root)
    lane_reports: list[dict[str, Any]] = []
    all_commands: list[str] = []
    total_rows = 0
    rnd_rows = 0
    non_rnd_rows = 0
    launch_artifacts: list[str] = []
    rnd_counts: dict[str, int] = {}
    reward_counts: dict[str, int] = {}

    for selection in profile.selections:
        lane = lanes.get(selection.lane_id)
        if lane is None:
            errors.append({"message": "profile references unknown lane", "lane_id": selection.lane_id})
            continue
        manifest_path = repo_root / lane.manifest_relpath
        manifest = _load_json(manifest_path)
        if manifest is None:
            errors.append(
                {
                    "message": "manifest missing",
                    "lane_id": lane.lane_id,
                    "manifest": lane.manifest_relpath,
                }
            )
            continue
        row_map = {str(row.get("row_id")): row for row in _rows(manifest) if row.get("row_id")}
        if selection.row_ids is None:
            selected_row_ids = tuple(row_map)
        else:
            selected_row_ids = selection.row_ids
        missing = [row_id for row_id in selected_row_ids if row_id not in row_map]
        if missing:
            errors.append(
                {
                    "message": "selected row ids missing from manifest",
                    "lane_id": lane.lane_id,
                    "manifest": lane.manifest_relpath,
                    "missing_row_ids": missing,
                }
            )
        selected_rows = [row_map[row_id] for row_id in selected_row_ids if row_id in row_map]
        output_relpath = _launch_output_relpath(lane.manifest_relpath, selected_row_ids if selection.row_ids else ())
        output_path = repo_root / output_relpath
        if output_path.exists():
            launch_artifacts.append(output_relpath)
        command = _command(lane.manifest_relpath, selected_row_ids if selection.row_ids else (), output_relpath)
        all_commands.append(command)
        row_count = len(selected_rows)
        total_rows += row_count
        if lane.lane_group == "rnd":
            rnd_rows += row_count
            for row in selected_rows:
                key = _rnd_key(row) or "unknown"
                rnd_counts[key] = rnd_counts.get(key, 0) + 1
        else:
            non_rnd_rows += row_count
            for row in selected_rows:
                key = _reward_variant(row) or "unknown"
                reward_counts[key] = reward_counts.get(key, 0) + 1
        lane_reports.append(
            {
                "lane_id": lane.lane_id,
                "lane_group": lane.lane_group,
                "manifest": lane.manifest_relpath,
                "selected_row_count": row_count,
                "selected_row_ids": list(selected_row_ids),
                "partial_launch": bool(selection.row_ids),
                "launch_output": output_relpath,
                "launch_artifact_exists": output_path.exists(),
                "command": command,
            }
        )

    if profile.max_active_h100_rows is not None and total_rows > profile.max_active_h100_rows:
        errors.append(
            {
                "message": "profile exceeds runtime-tier active H100 cap",
                "profile": profile.profile_id,
                "row_count": total_rows,
                "max_active_h100_rows": profile.max_active_h100_rows,
            }
        )
    if launch_artifacts:
        warnings.append(
            {
                "message": "launch output artifacts already exist; inspect before reusing commands",
                "launch_artifacts": launch_artifacts,
            }
        )
    if rnd_rows and not non_rnd_rows:
        errors.append({"message": "profile contains RND rows without non-RND controls"})

    report = {
        "schema_id": SCHEMA_ID,
        "ok": not errors,
        "profile": profile.profile_id,
        "tier": profile.tier,
        "intended_runtime": profile.intended_runtime,
        "purpose": profile.purpose,
        "non_rnd_seed_profile": profile.non_rnd_seed_profile,
        "max_active_h100_rows": profile.max_active_h100_rows,
        "total_selected_rows": total_rows,
        "rnd_selected_rows": rnd_rows,
        "non_rnd_selected_rows": non_rnd_rows,
        "rnd_counts": dict(sorted(rnd_counts.items())),
        "reward_variant_counts": dict(sorted(reward_counts.items())),
        "lanes": lane_reports,
        "prelaunch_checks": _prelaunch_checks(profile, total_rows),
        "commands": all_commands,
        "launch_artifacts": launch_artifacts,
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
    }
    return report


def _print_markdown(report: dict[str, Any]) -> None:
    print(f"# Wave A Staged Launch Plan: `{report.get('profile')}`")
    print()
    print(f"- ok: `{str(report.get('ok')).lower()}`")
    print(f"- tier: `{report.get('tier')}`")
    print(f"- intended runtime: `{report.get('intended_runtime')}`")
    print(f"- rows: `{report.get('total_selected_rows')}`")
    print(f"- RND / non-RND: `{report.get('rnd_selected_rows')}` / `{report.get('non_rnd_selected_rows')}`")
    print()
    print("## Lanes")
    print()
    print("| Lane | Rows | Partial |")
    print("| --- | ---: | --- |")
    for lane in report.get("lanes", []):
        print(
            f"| `{lane['lane_id']}` | {lane['selected_row_count']} | "
            f"`{str(lane['partial_launch']).lower()}` |"
        )
    print()
    print("## Commands")
    print()
    print("### Prelaunch Checks")
    print()
    for check in report.get("prelaunch_checks", []):
        print("```bash")
        print(check["command"])
        print("```")
    print("### Launch Commands")
    print()
    for command in report.get("commands", []):
        print("```bash")
        print(command)
        print("```")
    if report.get("errors"):
        print("## Errors")
        print()
        for error in report["errors"]:
            print(f"- {error}")
    if report.get("warnings"):
        print("## Warnings")
        print()
        for warning in report["warnings"]:
            print(f"- {warning}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--profile",
        default="mid36",
        choices=sorted(_default_profiles()),
        help="Staged launch profile to render.",
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", help="Optional output path for JSON report.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "markdown":
        _print_markdown(report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
