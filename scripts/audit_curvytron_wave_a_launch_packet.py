#!/usr/bin/env python3
"""Audit the repaired CurvyTron Wave A launch packet without launching jobs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA_ID = "curvyzero_curvytron_wave_a_launch_packet_audit/v0"
EXPECTED_APP_NAME = "curvyzero-lightzero-curvytron-visual-survival-train-v2"
EXPECTED_TRAIN_FUNCTION = "lightzero_curvytron_visual_survival_h100_cpu40"
EXPECTED_POLLER_FUNCTION = "lightzero_curvytron_visual_survival_checkpoint_eval_poller"
EXPECTED_COMPUTE = "gpu-h100-cpu40"
EXPECTED_LAUNCH_SCRIPT = "scripts/submit_curvytron_survivaldiag_manifest.py"
EXPECTED_SELECTED_ROW_IDS = ("r005", "r011", "r017")
EXPECTED_REFS_FILE = (
    "artifacts/local/curvytron_no_tournament_control_20260516/source/"
    "static_top4_nonzero_refs.txt"
)
SUBMISSION_SCHEMA_ID = "curvyzero_curvytron_survivaldiag_grouped_modal_submission/v0"
REF_AUDIT_SCHEMA_ID = "curvyzero_curvytron_launch_manifest_ref_audit/v0"
RUNS_VOLUME_NAME = "curvyzero-runs-v2"
MAX_MODAL_RUN_ID_LEN = 96
ITERATION_CKPT_RE = re.compile(r"iteration_\d+\.pth\.tar\Z")
EXPECTED_SELECTED_RECIPES = {
    "r005": {
        "label": "sparse-slot64-blank12-wall4-rank1_46-rank1imm2-clean",
        "reward_variant": "sparse_outcome",
    },
    "r011": {
        "label": "survbonusnoout-slot64-blank12-wall4-rank1_46-rank1imm2-clean",
        "reward_variant": "survival_plus_bonus_no_outcome",
    },
    "r017": {
        "label": "survbonusout-slot64-blank12-wall4-rank1_46-rank1imm2-clean",
        "reward_variant": "survival_plus_bonus_plus_outcome",
    },
}


@dataclass(frozen=True)
class LaneSpec:
    lane_id: str
    lane_group: str
    manifest_relpath: str
    dry_run_relpath: str
    expected_manifest_rows: int
    expected_selected_rows: int
    selected_row_ids: tuple[str, ...] = ()
    modal_ref_audit_relpath: str | None = None

    @property
    def expected_full_manifest_launch(self) -> bool:
        return not self.selected_row_ids


def _tonight18_relpath(family: str, filename: str) -> str:
    return f"artifacts/local/curvytron_tonight18_manifests/{family}/{filename}"


def _default_lane_specs() -> list[LaneSpec]:
    rnd_family = "rnd-blank-h100-wave-a-20260623a"
    rnd_base = f"artifacts/local/curvytron_rnd_blank_sweep_manifests/{rnd_family}"
    specs = [
        LaneSpec(
            lane_id="rnd-blank-sweep",
            lane_group="rnd",
            manifest_relpath=f"{rnd_base}/{rnd_family}.json",
            dry_run_relpath=f"{rnd_base}/{rnd_family}.submit.dryrun.json",
            expected_manifest_rows=45,
            expected_selected_rows=45,
        ),
        LaneSpec(
            lane_id="static-top4nz",
            lane_group="non_rnd_static",
            manifest_relpath=_tonight18_relpath(
                "reward-static-top4nz-h100-wave-a-repair-20260623a",
                "reward-static-top4nz-h100-wave-a-repair-20260623a.json",
            ),
            dry_run_relpath=_tonight18_relpath(
                "reward-static-top4nz-h100-wave-a-repair-20260623a",
                "reward-static-top4nz-h100-wave-a-repair-20260623a.submit.dryrun.json",
            ),
            modal_ref_audit_relpath=_tonight18_relpath(
                "reward-static-top4nz-h100-wave-a-repair-20260623a",
                "reward-static-top4nz-h100-wave-a-repair-20260623a.ref_audit.modal.json",
            ),
            expected_manifest_rows=18,
            expected_selected_rows=18,
        ),
    ]
    for replica in range(1, 7):
        family = f"reward-lhpre-top4nz-rep{replica:02d}-h100-wave-a-repair-20260623a"
        specs.append(
            LaneSpec(
                lane_id=f"long-horizon-rep{replica:02d}",
                lane_group="non_rnd_long_horizon",
                manifest_relpath=_tonight18_relpath(family, f"{family}.json"),
                dry_run_relpath=_tonight18_relpath(
                    family,
                    f"{family}.selected-r005-r011-r017.submit.dryrun.json",
                ),
                modal_ref_audit_relpath=_tonight18_relpath(
                    family,
                    f"{family}.ref_audit.modal.json",
                ),
                expected_manifest_rows=18,
                expected_selected_rows=3,
                selected_row_ids=EXPECTED_SELECTED_ROW_IDS,
            )
        )
    for suffix in (
        "s25-b128-td25-cap1024",
        "s25-b128-td25-cap2048",
        "s25-b256-td25-cap2048",
    ):
        family = f"reward-csupport-top4nz-{suffix}-wave-a-repair-20260623a"
        specs.append(
            LaneSpec(
                lane_id=f"cadence-support-{suffix}",
                lane_group="non_rnd_cadence_support",
                manifest_relpath=_tonight18_relpath(family, f"{family}.json"),
                dry_run_relpath=_tonight18_relpath(
                    family,
                    f"{family}.selected-r005-r011-r017.submit.dryrun.json",
                ),
                modal_ref_audit_relpath=_tonight18_relpath(
                    family,
                    f"{family}.ref_audit.modal.json",
                ),
                expected_manifest_rows=18,
                expected_selected_rows=3,
                selected_row_ids=EXPECTED_SELECTED_ROW_IDS,
            )
        )
    return specs


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _error(
    errors: list[dict[str, Any]],
    *,
    lane_id: str,
    path: Path | None,
    message: str,
    detail: Any = None,
) -> None:
    entry: dict[str, Any] = {"lane_id": lane_id, "message": message}
    if path is not None:
        entry["path"] = str(path)
    if detail is not None:
        entry["detail"] = detail
    errors.append(entry)


def _warning(
    warnings: list[dict[str, Any]],
    *,
    lane_id: str,
    path: Path | None,
    message: str,
    detail: Any = None,
) -> None:
    entry: dict[str, Any] = {"lane_id": lane_id, "message": message}
    if path is not None:
        entry["path"] = str(path)
    if detail is not None:
        entry["detail"] = detail
    warnings.append(entry)


def _expect(
    errors: list[dict[str, Any]],
    *,
    condition: bool,
    lane_id: str,
    path: Path | None,
    message: str,
    detail: Any = None,
) -> None:
    if not condition:
        _error(errors, lane_id=lane_id, path=path, message=message, detail=detail)


def _rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _artifact_refs(row_or_record: dict[str, Any]) -> dict[str, Any]:
    return _mapping(row_or_record.get("artifact_refs"))


def _train_kwargs(row: dict[str, Any]) -> dict[str, Any]:
    return _mapping(row.get("train_kwargs"))


def _counter_items(counter: Counter[Any]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: str(item[0]))
    ]


def _expected_launch_relpath(spec: LaneSpec) -> str:
    return spec.dry_run_relpath.replace(".submit.dryrun.json", ".submit.launch.json")


def _validate_submission(
    submission: Any,
    *,
    errors: list[dict[str, Any]],
    lane_id: str,
    path: Path,
    row_id: str,
) -> None:
    if not isinstance(submission, dict):
        _error(
            errors,
            lane_id=lane_id,
            path=path,
            message="row lacks deployed_app_submission",
            detail={"row_id": row_id},
        )
        return
    expected = {
        "app_name": EXPECTED_APP_NAME,
        "train_function": EXPECTED_TRAIN_FUNCTION,
        "poller_function": EXPECTED_POLLER_FUNCTION,
    }
    for key, expected_value in expected.items():
        actual = submission.get(key)
        _expect(
            errors,
            condition=actual == expected_value,
            lane_id=lane_id,
            path=path,
            message=f"deployed_app_submission.{key} mismatch",
            detail={"row_id": row_id, "expected": expected_value, "actual": actual},
        )


def _validate_common_manifest(
    spec: LaneSpec,
    manifest: dict[str, Any],
    manifest_path: Path,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _rows(manifest)
    guards = _mapping(manifest.get("guards"))
    fixed_knobs = _mapping(manifest.get("fixed_knobs"))

    _expect(
        errors,
        condition=len(rows) == spec.expected_manifest_rows,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="manifest row count mismatch",
        detail={"expected": spec.expected_manifest_rows, "actual": len(rows)},
    )
    _expect(
        errors,
        condition=guards.get("expected_row_count") == spec.expected_manifest_rows,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="guards.expected_row_count mismatch",
        detail={
            "expected": spec.expected_manifest_rows,
            "actual": guards.get("expected_row_count"),
        },
    )
    _expect(
        errors,
        condition=guards.get("modal_launch_performed") is False,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="manifest guard does not state modal_launch_performed=false",
        detail=guards.get("modal_launch_performed"),
    )
    _expect(
        errors,
        condition=guards.get("operator_launch_gate_required") is True,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="manifest guard does not require operator launch gate",
        detail=guards.get("operator_launch_gate_required"),
    )
    _expect(
        errors,
        condition=guards.get("deployed_app_name") == EXPECTED_APP_NAME,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="guards.deployed_app_name mismatch",
        detail={"expected": EXPECTED_APP_NAME, "actual": guards.get("deployed_app_name")},
    )
    _expect(
        errors,
        condition=guards.get("launch_script") == EXPECTED_LAUNCH_SCRIPT,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="guards.launch_script mismatch",
        detail={"expected": EXPECTED_LAUNCH_SCRIPT, "actual": guards.get("launch_script")},
    )
    _expect(
        errors,
        condition=fixed_knobs.get("compute") == EXPECTED_COMPUTE,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="fixed_knobs.compute mismatch",
        detail={"expected": EXPECTED_COMPUTE, "actual": fixed_knobs.get("compute")},
    )

    selected_rows = rows
    if spec.selected_row_ids:
        row_by_id = {str(row.get("row_id")): row for row in rows}
        missing = [row_id for row_id in spec.selected_row_ids if row_id not in row_by_id]
        _expect(
            errors,
            condition=not missing,
            lane_id=spec.lane_id,
            path=manifest_path,
            message="selected row ids missing from manifest",
            detail=missing,
        )
        selected_rows = [row_by_id[row_id] for row_id in spec.selected_row_ids if row_id in row_by_id]

    run_ids = [str(row.get("run_id") or "") for row in rows]
    duplicate_run_ids = sorted(
        run_id for run_id, count in Counter(run_ids).items() if run_id and count > 1
    )
    _expect(
        errors,
        condition=not duplicate_run_ids,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="duplicate run_id values in manifest",
        detail=duplicate_run_ids,
    )
    for row in rows:
        row_id = str(row.get("row_id") or "")
        run_id = str(row.get("run_id") or "")
        attempt_id = str(row.get("attempt_id") or "")
        _expect(
            errors,
            condition=bool(row_id and run_id and attempt_id),
            lane_id=spec.lane_id,
            path=manifest_path,
            message="row lacks row_id, run_id, or attempt_id",
            detail={"row_id": row_id, "run_id": run_id, "attempt_id": attempt_id},
        )
        _expect(
            errors,
            condition=len(run_id) <= MAX_MODAL_RUN_ID_LEN,
            lane_id=spec.lane_id,
            path=manifest_path,
            message="row run_id exceeds Modal length limit",
            detail={"row_id": row_id, "run_id": run_id, "length": len(run_id)},
        )
        _expect(
            errors,
            condition=len(attempt_id) <= MAX_MODAL_RUN_ID_LEN,
            lane_id=spec.lane_id,
            path=manifest_path,
            message="row attempt_id exceeds Modal length limit",
            detail={"row_id": row_id, "attempt_id": attempt_id, "length": len(attempt_id)},
        )
        _validate_submission(
            row.get("deployed_app_submission"),
            errors=errors,
            lane_id=spec.lane_id,
            path=manifest_path,
            row_id=row_id,
        )

    if len(run_ids) != len(rows):
        _warning(
            warnings,
            lane_id=spec.lane_id,
            path=manifest_path,
            message="manifest rows list contained non-object entries",
        )

    return rows, selected_rows


def _validate_rnd_manifest(
    spec: LaneSpec,
    manifest: dict[str, Any],
    manifest_path: Path,
    rows: Sequence[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    guards = _mapping(manifest.get("guards"))
    fixed_knobs = _mapping(manifest.get("fixed_knobs"))
    tournament = _mapping(manifest.get("tournament"))
    _expect(
        errors,
        condition=manifest.get("matrix_profile") == "rnd_blank_sweep",
        lane_id=spec.lane_id,
        path=manifest_path,
        message="RND matrix_profile mismatch",
        detail=manifest.get("matrix_profile"),
    )
    _expect(
        errors,
        condition=guards.get("all_rows_blank_canvas_noop") is True,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="RND guard all_rows_blank_canvas_noop is not true",
        detail=guards.get("all_rows_blank_canvas_noop"),
    )
    _expect(
        errors,
        condition=guards.get("assignment_refresh_enabled") is False,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="RND assignment refresh guard is not disabled",
        detail=guards.get("assignment_refresh_enabled"),
    )
    _expect(
        errors,
        condition=guards.get("tournament_enabled") is False,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="RND tournament guard is not disabled",
        detail=guards.get("tournament_enabled"),
    )
    _expect(
        errors,
        condition=fixed_knobs.get("opponent_assignment_refresh_interval_train_iter") == 0,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="RND fixed refresh interval is not zero",
        detail=fixed_knobs.get("opponent_assignment_refresh_interval_train_iter"),
    )
    _expect(
        errors,
        condition=fixed_knobs.get("save_ckpt_after_iter") == 2500,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="RND checkpoint cadence mismatch",
        detail=fixed_knobs.get("save_ckpt_after_iter"),
    )
    _expect(
        errors,
        condition=fixed_knobs.get("opponent_runtime_mode") == "blank_canvas_noop",
        lane_id=spec.lane_id,
        path=manifest_path,
        message="RND fixed opponent runtime mode is not blank_canvas_noop",
        detail=fixed_knobs.get("opponent_runtime_mode"),
    )
    _expect(
        errors,
        condition=tournament.get("enabled") is False,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="RND tournament.enabled is not false",
        detail=tournament,
    )
    _expect(
        errors,
        condition=manifest.get("assignment_bank") is None,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="RND manifest unexpectedly has assignment_bank",
        detail=manifest.get("assignment_bank"),
    )
    _expect(
        errors,
        condition=manifest.get("training_candidate_refresh_controller") is None,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="RND manifest unexpectedly has training candidate refresh controller",
        detail=manifest.get("training_candidate_refresh_controller"),
    )

    mode_counts: Counter[str] = Counter()
    require_counts: Counter[bool] = Counter()
    weight_counts: Counter[float] = Counter()
    for row in rows:
        row_id = str(row.get("row_id") or "")
        train_kwargs = _train_kwargs(row)
        bonus = _mapping(row.get("exploration_bonus"))
        mode = str(bonus.get("mode") or train_kwargs.get("exploration_bonus_mode") or "")
        mode_counts[mode] += 1
        require_metrics = bool(train_kwargs.get("require_rnd_metrics"))
        require_counts[require_metrics] += 1
        if bonus.get("weight") is not None:
            weight_counts[float(bonus["weight"])] += 1
        _expect(
            errors,
            condition=row.get("opponent_source") == "top_level_blank_canvas_noop",
            lane_id=spec.lane_id,
            path=manifest_path,
            message="RND row does not use blank-canvas noop opponent source",
            detail={"row_id": row_id, "opponent_source": row.get("opponent_source")},
        )
        source = _mapping(row.get("initial_policy_checkpoint_source"))
        _expect(
            errors,
            condition=source.get("source") == "scratch_random_initialization",
            lane_id=spec.lane_id,
            path=manifest_path,
            message="RND row does not start from scratch random initialization",
            detail={"row_id": row_id, "source": source},
        )
        expected_require = mode != "none"
        _expect(
            errors,
            condition=require_metrics is expected_require,
            lane_id=spec.lane_id,
            path=manifest_path,
            message="RND require_rnd_metrics does not match bonus mode",
            detail={
                "row_id": row_id,
                "mode": mode,
                "expected": expected_require,
                "actual": require_metrics,
            },
        )
        refs = _artifact_refs(row)
        _expect(
            errors,
            condition=bool(refs.get("rnd_reward_model_metrics_jsonl"))
            and bool(refs.get("rnd_reward_model_metrics_latest")),
            lane_id=spec.lane_id,
            path=manifest_path,
            message="RND row lacks RND metrics artifact refs",
            detail={"row_id": row_id, "artifact_ref_keys": sorted(refs)},
        )

    _expect(
        errors,
        condition=dict(mode_counts) == {"none": 5, "rnd_meter_v0": 5, "rnd_replay_target_v0": 35},
        lane_id=spec.lane_id,
        path=manifest_path,
        message="RND exploration bonus mode counts mismatch",
        detail=_counter_items(mode_counts),
    )
    _expect(
        errors,
        condition=dict(require_counts) == {False: 5, True: 40},
        lane_id=spec.lane_id,
        path=manifest_path,
        message="RND require_rnd_metrics counts mismatch",
        detail=_counter_items(require_counts),
    )
    expected_weights = {0.0: 10, 0.003: 5, 0.01: 5, 0.03: 5, 0.1: 5, 0.3: 5, 0.6: 5, 1.0: 5}
    _expect(
        errors,
        condition=dict(weight_counts) == expected_weights,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="RND exploration bonus weight counts mismatch",
        detail=_counter_items(weight_counts),
    )
    return {
        "rnd_mode_counts": _counter_items(mode_counts),
        "rnd_require_metrics_counts": _counter_items(require_counts),
        "rnd_weight_counts": _counter_items(weight_counts),
    }


def _validate_non_rnd_manifest(
    spec: LaneSpec,
    manifest: dict[str, Any],
    manifest_path: Path,
    rows: Sequence[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    fixed_knobs = _mapping(manifest.get("fixed_knobs"))
    _expect(
        errors,
        condition=manifest.get("checkpoint_refs_file_path") == EXPECTED_REFS_FILE,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="non-RND repair checkpoint refs file mismatch",
        detail={
            "expected": EXPECTED_REFS_FILE,
            "actual": manifest.get("checkpoint_refs_file_path"),
        },
    )
    _expect(
        errors,
        condition=manifest.get("opponent_source") == "mixture",
        lane_id=spec.lane_id,
        path=manifest_path,
        message="non-RND manifest opponent_source is not mixture",
        detail=manifest.get("opponent_source"),
    )
    _expect(
        errors,
        condition=manifest.get("assignment_bank") is None,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="non-RND repair manifest unexpectedly has assignment_bank",
        detail=manifest.get("assignment_bank"),
    )
    _expect(
        errors,
        condition=fixed_knobs.get("assignment_refresh_interval_train_iter") == 0,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="non-RND fixed refresh interval is not zero",
        detail=fixed_knobs.get("assignment_refresh_interval_train_iter"),
    )
    _expect(
        errors,
        condition=fixed_knobs.get("own_checkpoint_opponent_refresh_enabled") is False,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="non-RND own checkpoint opponent refresh is not disabled",
        detail=fixed_knobs.get("own_checkpoint_opponent_refresh_enabled"),
    )
    top_source = _mapping(manifest.get("top_checkpoint_source"))
    _expect(
        errors,
        condition=len(top_source) == 4,
        lane_id=spec.lane_id,
        path=manifest_path,
        message="non-RND top checkpoint source should contain four refs",
        detail={"count": len(top_source), "keys": sorted(top_source)},
    )
    reward_counts: Counter[str] = Counter()
    for row in rows:
        row_id = str(row.get("row_id") or "")
        train_kwargs = _train_kwargs(row)
        reward_variant = row.get("reward_variant") or train_kwargs.get("reward_variant")
        reward_counts[str(reward_variant)] += 1
        _expect(
            errors,
            condition=row.get("opponent_source") == "mixture",
            lane_id=spec.lane_id,
            path=manifest_path,
            message="non-RND row opponent_source is not mixture",
            detail={"row_id": row_id, "opponent_source": row.get("opponent_source")},
        )
        forbidden_assignment_refs = {
            "row.opponent_assignment_ref": row.get("opponent_assignment_ref"),
            "row.opponent_assignment_refresh_ref": row.get("opponent_assignment_refresh_ref"),
            "train.opponent_assignment_ref": train_kwargs.get("opponent_assignment_ref"),
            "train.opponent_assignment_refresh_ref": train_kwargs.get(
                "opponent_assignment_refresh_ref"
            ),
        }
        present = {
            key: value
            for key, value in forbidden_assignment_refs.items()
            if value not in (None, "")
        }
        _expect(
            errors,
            condition=not present,
            lane_id=spec.lane_id,
            path=manifest_path,
            message="non-RND repair row unexpectedly uses assignment refs",
            detail={"row_id": row_id, "refs": present},
        )
        _expect(
            errors,
            condition=bool(train_kwargs.get("opponent_mixture_spec"))
            or bool(row.get("opponent_mixture_spec")),
            lane_id=spec.lane_id,
            path=manifest_path,
            message="non-RND repair row lacks opponent mixture spec",
            detail={"row_id": row_id},
        )
        if row_id in EXPECTED_SELECTED_RECIPES:
            expected_recipe = EXPECTED_SELECTED_RECIPES[row_id]
            label = row.get("label")
            _expect(
                errors,
                condition=label == expected_recipe["label"],
                lane_id=spec.lane_id,
                path=manifest_path,
                message="selected non-RND probe label mismatch",
                detail={
                    "row_id": row_id,
                    "expected": expected_recipe["label"],
                    "actual": label,
                },
            )
            _expect(
                errors,
                condition=reward_variant == expected_recipe["reward_variant"],
                lane_id=spec.lane_id,
                path=manifest_path,
                message="selected non-RND probe reward variant mismatch",
                detail={
                    "row_id": row_id,
                    "expected": expected_recipe["reward_variant"],
                    "actual": reward_variant,
                },
            )
    return {"reward_variant_counts": _counter_items(reward_counts)}


def _validate_modal_ref_audit(
    spec: LaneSpec,
    repo_root: Path,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    if spec.modal_ref_audit_relpath is None:
        return {}
    path = repo_root / spec.modal_ref_audit_relpath
    payload = _load_json(path)
    if payload is None:
        _error(
            errors,
            lane_id=spec.lane_id,
            path=path,
            message="missing Modal ref audit artifact",
        )
        return {"modal_ref_audit_present": False}
    expected = {
        "schema_id": REF_AUDIT_SCHEMA_ID,
        "ok": True,
        "check_modal": True,
        "existence_checked": True,
        "syntax_only": False,
        "runs_volume_name": RUNS_VOLUME_NAME,
        "ref_count": 4,
        "bad_ref_count": 0,
        "missing_ref_count": 0,
        "modal_parent_error_count": 0,
    }
    for key, expected_value in expected.items():
        _expect(
            errors,
            condition=payload.get(key) == expected_value,
            lane_id=spec.lane_id,
            path=path,
            message=f"Modal ref audit {key} mismatch",
            detail={"expected": expected_value, "actual": payload.get(key)},
        )
    bad_modal_refs = []
    for ref_entry in payload.get("refs") if isinstance(payload.get("refs"), list) else []:
        if not isinstance(ref_entry, dict):
            bad_modal_refs.append({"reason": "ref entry is not an object", "entry": ref_entry})
            continue
        ref = str(ref_entry.get("ref") or "")
        reasons = []
        if ref_entry.get("modal_exists") is not True:
            reasons.append("modal_exists is not true")
        if "latest" in ref or "ckpt_best" in ref:
            reasons.append("mutable checkpoint ref")
        if ref.startswith("control:"):
            reasons.append("control-prefixed checkpoint ref")
        if not ITERATION_CKPT_RE.fullmatch(Path(ref).name):
            reasons.append("checkpoint basename is not iteration_N.pth.tar")
        if reasons:
            bad_modal_refs.append({"ref": ref, "reasons": reasons})
    _expect(
        errors,
        condition=not bad_modal_refs,
        lane_id=spec.lane_id,
        path=path,
        message="Modal ref audit contains bad or missing refs",
        detail=bad_modal_refs,
    )
    return {
        "modal_ref_audit_present": True,
        "modal_ref_audit_ok": payload.get("ok"),
        "modal_ref_count": payload.get("ref_count"),
    }


def _validate_dry_run(
    spec: LaneSpec,
    repo_root: Path,
    manifest_selected_rows: Sequence[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    path = repo_root / spec.dry_run_relpath
    payload = _load_json(path)
    if payload is None:
        _error(errors, lane_id=spec.lane_id, path=path, message="missing dry-run artifact")
        return {"dry_run_present": False}
    records = payload.get("records") if isinstance(payload.get("records"), list) else []
    record_row_ids = [str(record.get("row_id") or "") for record in records if isinstance(record, dict)]
    manifest_selected_row_ids = [str(row.get("row_id") or "") for row in manifest_selected_rows]

    expected_common = {
        "schema_id": SUBMISSION_SCHEMA_ID,
        "dry_run": True,
        "status": "dry_run",
        "selected_row_count": spec.expected_selected_rows,
        "row_count": spec.expected_selected_rows,
        "assignment_write_count": 0,
        "refresh_pointer_write_count": 0,
        "app_name": EXPECTED_APP_NAME,
    }
    for key, expected_value in expected_common.items():
        _expect(
            errors,
            condition=payload.get(key) == expected_value,
            lane_id=spec.lane_id,
            path=path,
            message=f"dry-run {key} mismatch",
            detail={"expected": expected_value, "actual": payload.get(key)},
        )
    _expect(
        errors,
        condition=len(records) == spec.expected_selected_rows,
        lane_id=spec.lane_id,
        path=path,
        message="dry-run record count mismatch",
        detail={"expected": spec.expected_selected_rows, "actual": len(records)},
    )
    _expect(
        errors,
        condition=payload.get("training_candidate_refresh_config_record") is None,
        lane_id=spec.lane_id,
        path=path,
        message="dry-run unexpectedly writes training candidate refresh config",
        detail=payload.get("training_candidate_refresh_config_record"),
    )
    expected_row_ids = (
        list(spec.selected_row_ids)
        if spec.selected_row_ids
        else manifest_selected_row_ids
    )
    _expect(
        errors,
        condition=record_row_ids == expected_row_ids,
        lane_id=spec.lane_id,
        path=path,
        message="dry-run record row ids mismatch",
        detail={"expected": expected_row_ids, "actual": record_row_ids},
    )

    for record in records:
        if not isinstance(record, dict):
            continue
        row_id = str(record.get("row_id") or "")
        _expect(
            errors,
            condition=record.get("status") == "dry_run",
            lane_id=spec.lane_id,
            path=path,
            message="dry-run record status mismatch",
            detail={"row_id": row_id, "actual": record.get("status")},
        )
        _expect(
            errors,
            condition=record.get("train_function") == EXPECTED_TRAIN_FUNCTION,
            lane_id=spec.lane_id,
            path=path,
            message="dry-run record train function mismatch",
            detail={"row_id": row_id, "actual": record.get("train_function")},
        )
        _expect(
            errors,
            condition=record.get("poller_function") == EXPECTED_POLLER_FUNCTION,
            lane_id=spec.lane_id,
            path=path,
            message="dry-run record poller function mismatch",
            detail={"row_id": row_id, "actual": record.get("poller_function")},
        )
        _expect(
            errors,
            condition=record.get("app_name") == EXPECTED_APP_NAME,
            lane_id=spec.lane_id,
            path=path,
            message="dry-run record app mismatch",
            detail={"row_id": row_id, "actual": record.get("app_name")},
        )
        call_id_keys = sorted(key for key in record if key.endswith("_function_call_id"))
        _expect(
            errors,
            condition=not call_id_keys,
            lane_id=spec.lane_id,
            path=path,
            message="dry-run record unexpectedly contains Modal function call ids",
            detail={"row_id": row_id, "keys": call_id_keys},
        )
        if spec.lane_group == "rnd":
            refs = _artifact_refs(record)
            _expect(
                errors,
                condition=bool(refs.get("rnd_reward_model_metrics_jsonl"))
                and bool(refs.get("rnd_reward_model_metrics_latest")),
                lane_id=spec.lane_id,
                path=path,
                message="RND dry-run record lacks RND metrics artifact refs",
                detail={"row_id": row_id, "artifact_ref_keys": sorted(refs)},
            )

    return {
        "dry_run_present": True,
        "dry_run_status": payload.get("status"),
        "dry_run_record_count": len(records),
        "dry_run_selected_row_count": payload.get("selected_row_count"),
    }


def _audit_lane(spec: LaneSpec, repo_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    manifest_path = repo_root / spec.manifest_relpath
    lane_report: dict[str, Any] = {
        "lane_id": spec.lane_id,
        "lane_group": spec.lane_group,
        "manifest": spec.manifest_relpath,
        "dry_run": spec.dry_run_relpath,
        "expected_manifest_rows": spec.expected_manifest_rows,
        "expected_selected_rows": spec.expected_selected_rows,
        "expected_full_manifest_launch": spec.expected_full_manifest_launch,
    }
    manifest = _load_json(manifest_path)
    if manifest is None:
        _error(errors, lane_id=spec.lane_id, path=manifest_path, message="missing manifest")
        lane_report["manifest_present"] = False
        return lane_report, errors, warnings

    rows, selected_rows = _validate_common_manifest(
        spec,
        manifest,
        manifest_path,
        errors,
        warnings,
    )
    lane_report.update(
        {
            "manifest_present": True,
            "matrix_name": manifest.get("matrix_name"),
            "manifest_row_count": len(rows),
            "selected_manifest_row_count": len(selected_rows),
        }
    )
    if spec.lane_group == "rnd":
        lane_report.update(_validate_rnd_manifest(spec, manifest, manifest_path, rows, errors))
    else:
        lane_report.update(
            _validate_non_rnd_manifest(spec, manifest, manifest_path, rows, errors)
        )
        lane_report.update(_validate_modal_ref_audit(spec, repo_root, errors))
    lane_report.update(_validate_dry_run(spec, repo_root, selected_rows, errors))
    return lane_report, errors, warnings


def _find_launch_artifacts(repo_root: Path, specs: Sequence[LaneSpec]) -> list[str]:
    directories = {
        (repo_root / spec.manifest_relpath).parent
        for spec in specs
    }
    found: set[str] = set()
    for spec in specs:
        path = repo_root / _expected_launch_relpath(spec)
        if path.exists():
            found.add(str(path.relative_to(repo_root)))
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.submit.launch.json"):
            found.add(str(path.relative_to(repo_root)))
    return sorted(found)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    specs = _default_lane_specs()
    lanes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for spec in specs:
        lane_report, lane_errors, lane_warnings = _audit_lane(spec, repo_root)
        lanes.append(lane_report)
        errors.extend(lane_errors)
        warnings.extend(lane_warnings)

    launch_artifacts = _find_launch_artifacts(repo_root, specs)
    if launch_artifacts and not args.allow_launch_artifacts:
        errors.append(
            {
                "lane_id": "launch-artifact-check",
                "message": "launch artifacts exist; pass --allow-launch-artifacts only when auditing after a real launch",
                "detail": launch_artifacts,
            }
        )
    elif launch_artifacts:
        warnings.append(
            {
                "lane_id": "launch-artifact-check",
                "message": "launch artifacts exist and were allowed",
                "detail": launch_artifacts,
            }
        )

    expected_total_selected = sum(spec.expected_selected_rows for spec in specs)
    actual_total_selected = sum(
        int(lane.get("dry_run_selected_row_count") or 0)
        for lane in lanes
    )
    non_rnd_selected = sum(
        spec.expected_selected_rows for spec in specs if spec.lane_group != "rnd"
    )
    report = {
        "schema_id": SCHEMA_ID,
        "ok": not errors,
        "repo_root": str(repo_root),
        "allow_launch_artifacts": bool(args.allow_launch_artifacts),
        "expected_total_selected_rows": expected_total_selected,
        "actual_total_selected_rows": actual_total_selected,
        "expected_non_rnd_selected_rows": non_rnd_selected,
        "expected_rnd_selected_rows": 45,
        "expected_h100_buffer": 10,
        "lane_count": len(lanes),
        "lanes": lanes,
        "launch_artifacts": launch_artifacts,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }
    if actual_total_selected != expected_total_selected:
        report["ok"] = False
        report["error_count"] += 1
        report["errors"].append(
            {
                "lane_id": "packet-total",
                "message": "actual selected dry-run row count does not match expected packet total",
                "detail": {
                    "expected": expected_total_selected,
                    "actual": actual_total_selected,
                },
            }
        )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing the Wave A artifacts",
    )
    parser.add_argument(
        "--allow-launch-artifacts",
        action="store_true",
        help="allow existing *.submit.launch.json artifacts in the packet directories",
    )
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
