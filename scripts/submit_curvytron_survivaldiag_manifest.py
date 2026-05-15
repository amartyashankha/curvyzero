#!/usr/bin/env python3
"""Submit CurvyTron survivaldiag manifest rows into one deployed Modal app."""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from typing import Any, Sequence

from curvyzero.contracts.curvytron import (
    CURVYTRON_POLICY_BONUS_RENDER_MODE as TARGET_POLICY_BONUS_RENDER_MODE,
    CURVYTRON_POLICY_TRAIL_RENDER_MODE as TARGET_POLICY_TRAIL_RENDER_MODE,
    POLLER_KWARGS_ALLOWED_FOR_GROUPED_SUBMIT,
    TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT,
    curvytron_control_volume_name,
    curvytron_runs_volume_name,
    curvytron_train_app_name,
    modal_volume_kwargs_for_name,
)


ASSIGNMENT_WRITER_FUNCTION = "lightzero_curvytron_write_opponent_assignment_artifacts"
MAX_MODAL_RUN_ID_LEN = 96
REF_PREFIX_TO_VOLUME = {
    "control": curvytron_control_volume_name,
    "runs": curvytron_runs_volume_name,
}


def _call_id(call: Any) -> str | None:
    value = getattr(call, "object_id", None) or getattr(call, "id", None)
    return None if value is None else str(value)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain non-empty rows")
    return payload


def _selected_rows(manifest: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = list(manifest["rows"])
    if args.row_id:
        wanted = set(args.row_id)
        rows = [row for row in rows if str(row.get("row_id")) in wanted]
    if args.row_kind:
        wanted_kinds = set(args.row_kind)
        rows = [row for row in rows if str(row.get("row_kind")) in wanted_kinds]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("row filters selected no rows")
    seen: set[str] = set()
    for row in rows:
        run_id = str(row.get("run_id") or "")
        if not run_id:
            raise ValueError(f"row {row.get('row_id')} has empty run_id")
        if len(run_id) > MAX_MODAL_RUN_ID_LEN:
            raise ValueError(
                f"row {row.get('row_id')} run_id exceeds {MAX_MODAL_RUN_ID_LEN} chars: "
                f"{len(run_id)}"
            )
        attempt_id = str(row.get("attempt_id") or "")
        if not attempt_id:
            raise ValueError(f"row {row.get('row_id')} has empty attempt_id")
        if len(attempt_id) > MAX_MODAL_RUN_ID_LEN:
            raise ValueError(
                f"row {row.get('row_id')} attempt_id exceeds {MAX_MODAL_RUN_ID_LEN} chars: "
                f"{len(attempt_id)}"
            )
        if run_id in seen:
            raise ValueError(f"duplicate run_id in selected rows: {run_id}")
        seen.add(run_id)
    return rows


def _launch_row(
    row: dict[str, Any],
    *,
    app_name: str,
    modal_env: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    submission = row.get("deployed_app_submission")
    if not isinstance(submission, dict):
        raise ValueError(f"row {row.get('row_id')} lacks deployed_app_submission")
    train_function = str(submission.get("train_function") or "")
    poller_function = str(submission.get("poller_function") or "")
    train_kwargs = row.get("train_kwargs")
    poller_kwargs = row.get("poller_kwargs")
    if not train_function or not poller_function:
        raise ValueError(f"row {row.get('row_id')} lacks deployed function names")
    if not isinstance(train_kwargs, dict) or not isinstance(poller_kwargs, dict):
        raise ValueError(f"row {row.get('row_id')} lacks train/poller kwargs")
    missing_train_kwargs = [
        key for key in TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT if key not in train_kwargs
    ]
    if missing_train_kwargs:
        raise ValueError(
            f"row {row.get('row_id')} train_kwargs missing required keys: {missing_train_kwargs}"
        )
    _validate_initial_policy_checkpoint_ref(row, train_kwargs)
    _validate_policy_surface(row, train_kwargs)
    _validate_poller_kwargs(row, poller_kwargs)
    _validate_optional_mixture_kwargs(row, train_kwargs, poller_kwargs)

    record = {
        "row_id": row.get("row_id"),
        "label": row.get("label"),
        "run_id": row.get("run_id"),
        "attempt_id": row.get("attempt_id"),
        "app_name": app_name,
        "train_function": train_function,
        "poller_function": poller_function,
        "artifact_refs": row.get("artifact_refs"),
    }
    if dry_run:
        return {"status": "dry_run", **record}

    import modal

    poller_fn = modal.Function.from_name(
        app_name,
        poller_function,
        environment_name=modal_env,
    )
    train_fn = modal.Function.from_name(
        app_name,
        train_function,
        environment_name=modal_env,
    )
    poller_call = poller_fn.spawn(**poller_kwargs)
    train_call = train_fn.spawn(**train_kwargs)
    return {
        "status": "spawned",
        **record,
        "poller_function_call_id": _call_id(poller_call),
        "train_function_call_id": _call_id(train_call),
    }


def _validate_launch_app_name(app_name: str, rows: Sequence[dict[str, Any]]) -> None:
    expected_app_name = curvytron_train_app_name()
    if app_name != expected_app_name:
        raise ValueError(
            "manifest submission app mismatch: "
            f"selected {app_name!r}, expected current all-v2 trainer app "
            f"{expected_app_name!r}"
        )
    for row in rows:
        submission = row.get("deployed_app_submission")
        if not isinstance(submission, dict):
            raise ValueError(f"row {row.get('row_id')} lacks deployed_app_submission")
        row_app = str(submission.get("app_name") or "").strip()
        if row_app != app_name:
            raise ValueError(
                f"row {row.get('row_id')} targets app {row_app!r}, "
                f"but submission selected {app_name!r}"
            )


def _validate_policy_surface(row: dict[str, Any], train_kwargs: dict[str, Any]) -> None:
    trail_mode = str(train_kwargs.get("source_state_trail_render_mode") or "")
    bonus_mode = str(train_kwargs.get("source_state_bonus_render_mode") or "")
    if (
        trail_mode == TARGET_POLICY_TRAIL_RENDER_MODE
        and bonus_mode == TARGET_POLICY_BONUS_RENDER_MODE
    ):
        return
    raise ValueError(
        f"row {row.get('row_id')} uses policy surface "
        f"{trail_mode!r} + {bonus_mode!r}; expected "
        f"{TARGET_POLICY_TRAIL_RENDER_MODE!r} + {TARGET_POLICY_BONUS_RENDER_MODE!r}."
    )


def _validate_optional_mixture_kwargs(
    row: dict[str, Any],
    train_kwargs: dict[str, Any],
    poller_kwargs: dict[str, Any],
) -> None:
    train_spec = train_kwargs.get("opponent_mixture_spec")
    poller_spec = poller_kwargs.get("opponent_mixture_spec")
    train_assignment_ref = str(train_kwargs.get("opponent_assignment_ref") or "").strip()
    poller_assignment_ref = str(poller_kwargs.get("opponent_assignment_ref") or "").strip()
    if (
        train_spec is None
        and poller_spec is None
        and not train_assignment_ref
        and not poller_assignment_ref
    ):
        return
    if bool(train_spec is not None) == bool(train_assignment_ref):
        raise ValueError(
            f"row {row.get('row_id')} must use exactly one opponent source: "
            "opponent_mixture_spec or opponent_assignment_ref"
        )
    if train_assignment_ref:
        if poller_assignment_ref != train_assignment_ref:
            raise ValueError(f"row {row.get('row_id')} train/poller assignment refs differ")
        if poller_spec is not None:
            raise ValueError(f"row {row.get('row_id')} assignment row has poller mixture spec")
        return
    if train_spec is None:
        raise ValueError(f"row {row.get('row_id')} mixture row lacks train opponent_mixture_spec")
    if poller_spec is None:
        raise ValueError(f"row {row.get('row_id')} mixture row lacks poller opponent_mixture_spec")
    if train_spec != poller_spec:
        raise ValueError(
            f"row {row.get('row_id')} mixture train/poller opponent_mixture_spec differ"
        )

    from curvyzero.training.opponent_mixture import parse_opponent_mixture_spec

    mixture = parse_opponent_mixture_spec(train_spec)
    if mixture is None:
        raise ValueError(f"row {row.get('row_id')} mixture spec parsed as empty")
    for entry in mixture["entries"]:
        if entry["opponent_policy_kind"] != "frozen_lightzero_checkpoint":
            continue
        checkpoint_ref = str(entry.get("opponent_checkpoint_ref") or "")
        if not checkpoint_ref:
            raise ValueError(f"row {row.get('row_id')} frozen mixture entry lacks checkpoint ref")
        if "latest" in checkpoint_ref or "ckpt_best" in checkpoint_ref:
            raise ValueError(
                f"row {row.get('row_id')} frozen mixture entry uses mutable checkpoint ref"
            )
        if not re.fullmatch(r"iteration_\d+\.pth\.tar", Path(checkpoint_ref).name):
            raise ValueError(
                f"row {row.get('row_id')} frozen mixture entry must use iteration_N.pth.tar"
            )


def _validate_poller_kwargs(row: dict[str, Any], poller_kwargs: dict[str, Any]) -> None:
    unknown = sorted(set(poller_kwargs) - POLLER_KWARGS_ALLOWED_FOR_GROUPED_SUBMIT)
    if unknown:
        raise ValueError(f"row {row.get('row_id')} poller_kwargs has unsupported keys: {unknown}")


def _validate_initial_policy_checkpoint_ref(
    row: dict[str, Any],
    train_kwargs: dict[str, Any],
) -> None:
    checkpoint_ref = str(train_kwargs.get("initial_policy_checkpoint_ref") or "").strip()
    if not checkpoint_ref:
        raise ValueError(f"row {row.get('row_id')} lacks initial_policy_checkpoint_ref")
    if "latest" in checkpoint_ref or "ckpt_best" in checkpoint_ref:
        raise ValueError(
            f"row {row.get('row_id')} initial_policy_checkpoint_ref is mutable: "
            f"{checkpoint_ref}"
        )
    if not re.fullmatch(r"iteration_\d+\.pth\.tar", Path(checkpoint_ref).name):
        raise ValueError(
            f"row {row.get('row_id')} initial_policy_checkpoint_ref must use "
            f"iteration_N.pth.tar: {checkpoint_ref}"
        )
    load_mode = train_kwargs.get("initial_policy_checkpoint_load_mode")
    if load_mode != "matching_shape":
        raise ValueError(
            f"row {row.get('row_id')} initial_policy_checkpoint_load_mode must be "
            f"'matching_shape'; got {load_mode!r}"
        )


def _assignment_refs_for_rows(rows: Sequence[dict[str, Any]]) -> set[str]:
    refs = set()
    for row in rows:
        train_kwargs = row.get("train_kwargs")
        if not isinstance(train_kwargs, dict):
            continue
        ref = str(train_kwargs.get("opponent_assignment_ref") or "").strip()
        if ref:
            refs.add(ref)
    return refs


def _refresh_pointer_refs_for_rows(rows: Sequence[dict[str, Any]]) -> set[str]:
    refs = set()
    for row in rows:
        train_kwargs = row.get("train_kwargs")
        if not isinstance(train_kwargs, dict):
            continue
        ref = str(train_kwargs.get("opponent_assignment_refresh_ref") or "").strip()
        if ref:
            refs.add(ref)
    return refs


def _volume_name_for_ref(ref: str, pointer_volume: str) -> tuple[str, str]:
    text = str(ref)
    if text.startswith("control:"):
        return curvytron_control_volume_name(), text.removeprefix("control:")
    if text.startswith("runs:"):
        return curvytron_runs_volume_name(), text.removeprefix("runs:")
    if pointer_volume == "control":
        return curvytron_control_volume_name(), text
    if pointer_volume == "runs":
        return curvytron_runs_volume_name(), text
    raise ValueError(f"unknown refresh pointer volume {pointer_volume!r}")


def _publish_refresh_pointers(
    manifest: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    *,
    modal_env: str | None,
    dry_run: bool,
) -> list[dict[str, Any]]:
    needed_refs = _refresh_pointer_refs_for_rows(rows)
    if not needed_refs:
        return []
    assignment_bank = manifest.get("assignment_bank")
    if not isinstance(assignment_bank, dict):
        raise ValueError("selected refresh rows require manifest.assignment_bank")
    pointers = assignment_bank.get("refresh_pointers")
    if not isinstance(pointers, dict):
        raise ValueError("selected refresh rows require manifest.assignment_bank.refresh_pointers")

    artifacts_by_ref: dict[str, dict[str, Any]] = {}
    for recipe_id, artifact in pointers.items():
        if not isinstance(artifact, dict):
            continue
        pointer_ref = str(artifact.get("pointer_ref") or "").strip()
        if pointer_ref:
            artifacts_by_ref[pointer_ref] = {"recipe_id": recipe_id, **artifact}
    missing = sorted(needed_refs - set(artifacts_by_ref))
    if missing:
        raise ValueError(f"refresh rows reference pointers missing from assignment_bank: {missing}")

    records = []
    volumes: dict[str, Any] = {}
    if not dry_run:
        import modal

    for pointer_ref in sorted(needed_refs):
        artifact = artifacts_by_ref[pointer_ref]
        pointer_volume = str(artifact.get("pointer_volume") or "control").strip()
        volume_name, remote_ref = _volume_name_for_ref(pointer_ref, pointer_volume)
        payload = {
            "schema_id": "curvyzero_opponent_assignment_refresh_pointer/v0",
            "assignment_ref": artifact["assignment_ref"],
            "assignment_sha256": artifact["assignment_sha256"],
        }
        if artifact.get("audit") is not None:
            payload["audit"] = artifact["audit"]
        record = {
            "pointer_ref": pointer_ref,
            "recipe_id": artifact.get("recipe_id"),
            "pointer_volume": pointer_volume,
            "volume_name": volume_name,
            "remote_ref": remote_ref,
            "assignment_ref": artifact.get("assignment_ref"),
            "assignment_sha256": artifact.get("assignment_sha256"),
        }
        if dry_run:
            records.append({"status": "dry_run", **record})
            continue
        volume = volumes.get(volume_name)
        if volume is None:
            volume = modal.Volume.from_name(
                volume_name,
                environment_name=modal_env,
                **modal_volume_kwargs_for_name(volume_name),
            )
            volumes[volume_name] = volume
        content = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        upload_ref = "/" + remote_ref.lstrip("/")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(content), upload_ref)
        records.append({"status": "written", **record, "upload_ref": upload_ref})
    return records


def _publish_assignment_bank(
    manifest: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    *,
    app_name: str,
    modal_env: str | None,
    dry_run: bool,
) -> list[dict[str, Any]]:
    needed_refs = _assignment_refs_for_rows(rows)
    if not needed_refs:
        return []
    assignment_bank = manifest.get("assignment_bank")
    if not isinstance(assignment_bank, dict):
        raise ValueError("selected assignment rows require manifest.assignment_bank")
    run_id = str(assignment_bank.get("run_id") or "").strip()
    attempt_id = str(assignment_bank.get("attempt_id") or "").strip()
    target_volume = str(assignment_bank.get("target_volume") or "runs").strip()
    mirror_checkpoints_to_control = bool(
        assignment_bank.get("mirror_checkpoints_to_control", False)
    )
    assignments = assignment_bank.get("assignments")
    if not run_id or not attempt_id or not isinstance(assignments, dict):
        raise ValueError("manifest.assignment_bank must include run_id, attempt_id, assignments")

    artifacts_by_ref: dict[str, dict[str, Any]] = {}
    for recipe_id, artifact in assignments.items():
        if not isinstance(artifact, dict):
            continue
        assignment_ref = str(artifact.get("assignment_ref") or "").strip()
        if assignment_ref:
            artifacts_by_ref[assignment_ref] = {"recipe_id": recipe_id, **artifact}
    missing = sorted(needed_refs - set(artifacts_by_ref))
    if missing:
        raise ValueError(f"assignment rows reference refs missing from assignment_bank: {missing}")

    records = []
    writer_fn = None
    if not dry_run:
        import modal

        writer_fn = modal.Function.from_name(
            app_name,
            ASSIGNMENT_WRITER_FUNCTION,
            environment_name=modal_env,
        )
    for assignment_ref in sorted(needed_refs):
        artifact = artifacts_by_ref[assignment_ref]
        payload = {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "assignment": artifact["assignment"],
            "audit": artifact.get("audit"),
            "target_volume": target_volume,
            "mirror_checkpoints_to_control": mirror_checkpoints_to_control,
        }
        record = {
            "assignment_ref": assignment_ref,
            "assignment_id": artifact.get("assignment_id"),
            "recipe_id": artifact.get("recipe_id"),
            "app_name": app_name,
            "writer_function": ASSIGNMENT_WRITER_FUNCTION,
            "target_volume": target_volume,
            "mirror_checkpoints_to_control": mirror_checkpoints_to_control,
        }
        if dry_run:
            records.append({"status": "dry_run", **record})
            continue
        result = writer_fn.remote(payload)
        returned_ref = str(result.get("assignment_ref") or "")
        if returned_ref != assignment_ref:
            raise RuntimeError(
                "assignment writer returned unexpected ref for "
                f"{artifact.get('recipe_id')}: expected {assignment_ref}, got {returned_ref}"
            )
        expected_sha = str(artifact.get("assignment_sha256") or "")
        returned_sha = str(result.get("assignment_sha256") or "")
        if (
            expected_sha
            and returned_sha != expected_sha
            and not (target_volume == "control" and mirror_checkpoints_to_control)
        ):
            raise RuntimeError(
                "assignment writer returned unexpected sha for "
                f"{artifact.get('recipe_id')}: expected {expected_sha}, got {returned_sha}"
            )
        records.append({"status": "written", **record, "result": result})
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit rows from a CurvyTron survivaldiag manifest into the deployed "
            "trainer app. This avoids one ephemeral Modal app per row."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--app-name", default=None)
    parser.add_argument("--modal-env", default=None)
    parser.add_argument("--row-id", action="append", default=[])
    parser.add_argument("--row-kind", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--publish-assignments-only",
        action="store_true",
        help="write selected assignment artifacts and refresh pointers, then skip train/poller spawns",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = _load_manifest(args.manifest)
    rows = _selected_rows(manifest, args)
    app_name = args.app_name or str(manifest["guards"]["deployed_app_name"])
    _validate_launch_app_name(app_name, rows)
    dry_run = not args.allow_launch
    assignment_records = _publish_assignment_bank(
        manifest,
        rows,
        app_name=app_name,
        modal_env=args.modal_env,
        dry_run=dry_run,
    )
    refresh_pointer_records = _publish_refresh_pointers(
        manifest,
        rows,
        modal_env=args.modal_env,
        dry_run=dry_run,
    )
    records = []
    if not args.publish_assignments_only:
        records = [
            _launch_row(
                row,
                app_name=app_name,
                modal_env=args.modal_env,
                dry_run=dry_run,
            )
            for row in rows
        ]
    payload = {
        "schema_id": "curvyzero_curvytron_survivaldiag_grouped_modal_submission/v0",
        "status": "dry_run" if dry_run else "submitted",
        "dry_run": dry_run,
        "publish_assignments_only": bool(args.publish_assignments_only),
        "app_name": app_name,
        "modal_env": args.modal_env,
        "matrix_name": manifest.get("matrix_name"),
        "matrix_profile": manifest.get("matrix_profile"),
        "assignment_write_count": len(assignment_records),
        "assignment_records": assignment_records,
        "refresh_pointer_write_count": len(refresh_pointer_records),
        "refresh_pointer_records": refresh_pointer_records,
        "selected_row_count": len(rows),
        "row_count": len(records),
        "records": records,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
