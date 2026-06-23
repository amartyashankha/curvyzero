from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "scripts" / "audit_curvytron_wave_a_launch_packet.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_curvytron_wave_a_launch_packet", AUDIT_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _submission(audit):
    return {
        "app_name": audit.EXPECTED_APP_NAME,
        "train_function": audit.EXPECTED_TRAIN_FUNCTION,
        "poller_function": audit.EXPECTED_POLLER_FUNCTION,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row_id(index: int) -> str:
    return f"r{index:03d}"


def _rnd_bonus_for_index(index: int) -> tuple[str, float, bool]:
    if index <= 5:
        return "none", 0.0, False
    if index <= 10:
        return "rnd_meter_v0", 0.0, True
    weights = [0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
    return "rnd_replay_target_v0", weights[(index - 11) // 5], True


def _artifact_refs(*, rnd: bool) -> dict:
    refs = {
        "summary": "training/lightzero-curvytron-visual-survival/run/summary.json",
        "progress_latest": "training/lightzero-curvytron-visual-survival/run/progress_latest.json",
    }
    if rnd:
        refs.update(
            {
                "rnd_reward_model_metrics_jsonl": (
                    "training/lightzero-curvytron-visual-survival/run/"
                    "rnd_reward_model_metrics.jsonl"
                ),
                "rnd_reward_model_metrics_latest": (
                    "training/lightzero-curvytron-visual-survival/run/"
                    "rnd_reward_model_metrics_latest.json"
                ),
            }
        )
    return refs


def _rnd_manifest(audit, spec) -> dict:
    rows = []
    for index in range(1, 46):
        row_id = _row_id(index)
        mode, weight, require_metrics = _rnd_bonus_for_index(index)
        rows.append(
            {
                "row_id": row_id,
                "label": f"rnd-{row_id}",
                "run_id": f"rnd-run-{row_id}",
                "attempt_id": f"try-rnd-run-{row_id}",
                "opponent_source": "top_level_blank_canvas_noop",
                "initial_policy_checkpoint_source": {
                    "source": "scratch_random_initialization",
                    "checkpoint_ref": None,
                },
                "deployed_app_submission": _submission(audit),
                "exploration_bonus": {"mode": mode, "weight": weight},
                "train_kwargs": {
                    "mode": "train",
                    "seed": 1000 + index,
                    "run_id": f"rnd-run-{row_id}",
                    "attempt_id": f"try-rnd-run-{row_id}",
                    "require_rnd_metrics": require_metrics,
                    "exploration_bonus_mode": mode,
                },
                "artifact_refs": _artifact_refs(rnd=True),
            }
        )
    return {
        "schema_id": "curvyzero_curvytron_rnd_blank_sweep_manifest/v0",
        "matrix_name": "rnd-blank-h100-wave-a-20260623a",
        "matrix_profile": "rnd_blank_sweep",
        "guards": {
            "expected_row_count": spec.expected_manifest_rows,
            "modal_launch_performed": False,
            "operator_launch_gate_required": True,
            "deployed_app_name": audit.EXPECTED_APP_NAME,
            "launch_script": audit.EXPECTED_LAUNCH_SCRIPT,
            "all_rows_blank_canvas_noop": True,
            "assignment_refresh_enabled": False,
            "tournament_enabled": False,
        },
        "fixed_knobs": {
            "compute": audit.EXPECTED_COMPUTE,
            "opponent_assignment_refresh_interval_train_iter": 0,
            "opponent_runtime_mode": "blank_canvas_noop",
            "save_ckpt_after_iter": 2500,
        },
        "tournament": {"enabled": False},
        "assignment_bank": None,
        "training_candidate_refresh_controller": None,
        "rows": rows,
    }


def _non_rnd_reward_for_row(row_id: str, index: int) -> tuple[str, str]:
    if row_id in ("r005", "r011", "r017"):
        recipe = {
            "r005": ("sparse-slot64-blank12-wall4-rank1_46-rank1imm2-clean", "sparse_outcome"),
            "r011": (
                "survbonusnoout-slot64-blank12-wall4-rank1_46-rank1imm2-clean",
                "survival_plus_bonus_no_outcome",
            ),
            "r017": (
                "survbonusout-slot64-blank12-wall4-rank1_46-rank1imm2-clean",
                "survival_plus_bonus_plus_outcome",
            ),
        }
        return recipe[row_id]
    variants = [
        "sparse_outcome",
        "survival_plus_bonus_no_outcome",
        "survival_plus_bonus_plus_outcome",
    ]
    return f"control-{row_id}", variants[(index - 1) % len(variants)]


def _non_rnd_manifest(audit, spec) -> dict:
    rows = []
    for index in range(1, 19):
        row_id = _row_id(index)
        label, reward_variant = _non_rnd_reward_for_row(row_id, index)
        rows.append(
            {
                "row_id": row_id,
                "label": label,
                "run_id": f"{spec.lane_id}-run-{row_id}",
                "attempt_id": f"try-{spec.lane_id}-run-{row_id}",
                "opponent_source": "mixture",
                "deployed_app_submission": _submission(audit),
                "reward_variant": reward_variant,
                "train_kwargs": {
                    "mode": "train",
                    "seed": 2000 + index,
                    "run_id": f"{spec.lane_id}-run-{row_id}",
                    "attempt_id": f"try-{spec.lane_id}-run-{row_id}",
                    "reward_variant": reward_variant,
                    "opponent_mixture_spec": "rank1:1.0",
                },
                "artifact_refs": _artifact_refs(rnd=False),
            }
        )
    return {
        "schema_id": "curvyzero_curvytron_tonight18_manifest/v0",
        "matrix_name": spec.manifest_relpath.rsplit("/", 1)[-1].removesuffix(".json"),
        "matrix_profile": "tonight18_reward_opponent_noise",
        "checkpoint_refs_file_path": audit.EXPECTED_REFS_FILE,
        "opponent_source": "mixture",
        "assignment_bank": None,
        "guards": {
            "expected_row_count": spec.expected_manifest_rows,
            "modal_launch_performed": False,
            "operator_launch_gate_required": True,
            "deployed_app_name": audit.EXPECTED_APP_NAME,
            "launch_script": audit.EXPECTED_LAUNCH_SCRIPT,
        },
        "fixed_knobs": {
            "compute": audit.EXPECTED_COMPUTE,
            "assignment_refresh_interval_train_iter": 0,
            "own_checkpoint_opponent_refresh_enabled": False,
        },
        "top_checkpoint_source": {
            f"rank{rank}": {
                "checkpoint_ref": (
                    "training/lightzero-curvytron-visual-survival/source-run/"
                    f"attempts/try-source/train/lightzero_exp/ckpt/iteration_{rank * 10000}.pth.tar"
                )
            }
            for rank in range(1, 5)
        },
        "rows": rows,
    }


def _dry_run_payload(audit, spec, rows: list[dict]) -> dict:
    if spec.selected_row_ids:
        selected = [row for row in rows if row["row_id"] in spec.selected_row_ids]
    else:
        selected = rows
    records = []
    for row in selected:
        records.append(
            {
                "status": "dry_run",
                "row_id": row["row_id"],
                "run_id": row["run_id"],
                "attempt_id": row["attempt_id"],
                "app_name": audit.EXPECTED_APP_NAME,
                "train_function": audit.EXPECTED_TRAIN_FUNCTION,
                "poller_function": audit.EXPECTED_POLLER_FUNCTION,
                "artifact_refs": row["artifact_refs"],
            }
        )
    return {
        "schema_id": audit.SUBMISSION_SCHEMA_ID,
        "dry_run": True,
        "status": "dry_run",
        "app_name": audit.EXPECTED_APP_NAME,
        "selected_row_count": spec.expected_selected_rows,
        "row_count": spec.expected_selected_rows,
        "assignment_write_count": 0,
        "refresh_pointer_write_count": 0,
        "training_candidate_refresh_config_record": None,
        "records": records,
    }


def _modal_ref_audit_payload(audit) -> dict:
    refs = [
        "training/lightzero-curvytron-visual-survival/source-run/"
        f"attempts/try-source/train/lightzero_exp/ckpt/iteration_{rank * 10000}.pth.tar"
        for rank in range(1, 5)
    ]
    return {
        "schema_id": audit.REF_AUDIT_SCHEMA_ID,
        "runs_volume_name": audit.RUNS_VOLUME_NAME,
        "ok": True,
        "check_modal": True,
        "syntax_only": False,
        "existence_checked": True,
        "ref_count": 4,
        "bad_ref_count": 0,
        "missing_ref_count": 0,
        "modal_parent_error_count": 0,
        "refs": [{"ref": ref, "modal_exists": True} for ref in refs],
    }


def _write_packet(repo_root: Path, audit) -> None:
    for spec in audit._default_lane_specs():
        if spec.lane_group == "rnd":
            manifest = _rnd_manifest(audit, spec)
        else:
            manifest = _non_rnd_manifest(audit, spec)
        _write_json(repo_root / spec.manifest_relpath, manifest)
        _write_json(repo_root / spec.dry_run_relpath, _dry_run_payload(audit, spec, manifest["rows"]))
        if spec.modal_ref_audit_relpath is not None:
            _write_json(repo_root / spec.modal_ref_audit_relpath, _modal_ref_audit_payload(audit))


def test_wave_a_launch_packet_audit_accepts_complete_no_launch_packet(tmp_path):
    audit = _load_script()
    _write_packet(tmp_path, audit)

    report = audit.build_report(audit.parse_args(["--repo-root", str(tmp_path)]))

    assert report["ok"] is True
    assert report["expected_total_selected_rows"] == 90
    assert report["actual_total_selected_rows"] == 90
    assert report["error_count"] == 0
    assert report["launch_artifacts"] == []


def test_wave_a_launch_packet_audit_rejects_missing_modal_ref_audit(tmp_path):
    audit = _load_script()
    _write_packet(tmp_path, audit)
    spec = audit._default_lane_specs()[1]
    (tmp_path / spec.modal_ref_audit_relpath).unlink()

    report = audit.build_report(audit.parse_args(["--repo-root", str(tmp_path)]))

    assert report["ok"] is False
    assert any(error["message"] == "missing Modal ref audit artifact" for error in report["errors"])


def test_wave_a_launch_packet_audit_rejects_launch_artifacts_by_default(tmp_path):
    audit = _load_script()
    _write_packet(tmp_path, audit)
    spec = audit._default_lane_specs()[0]
    launch_path = tmp_path / spec.dry_run_relpath.replace(".submit.dryrun.json", ".submit.launch.json")
    _write_json(launch_path, {"status": "submitted"})

    report = audit.build_report(audit.parse_args(["--repo-root", str(tmp_path)]))
    allowed = audit.build_report(
        audit.parse_args(["--repo-root", str(tmp_path), "--allow-launch-artifacts"])
    )

    assert report["ok"] is False
    assert any(error["lane_id"] == "launch-artifact-check" for error in report["errors"])
    assert allowed["ok"] is True
    assert allowed["launch_artifacts"] == [str(launch_path.relative_to(tmp_path))]


def test_wave_a_launch_packet_audit_rejects_rnd_metrics_guard_drift(tmp_path):
    audit = _load_script()
    _write_packet(tmp_path, audit)
    spec = audit._default_lane_specs()[0]
    manifest_path = tmp_path / spec.manifest_relpath
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"][5]["train_kwargs"]["require_rnd_metrics"] = False
    _write_json(manifest_path, manifest)

    report = audit.build_report(audit.parse_args(["--repo-root", str(tmp_path)]))

    assert report["ok"] is False
    assert any(
        error["message"] == "RND require_rnd_metrics does not match bonus mode"
        for error in report["errors"]
    )
