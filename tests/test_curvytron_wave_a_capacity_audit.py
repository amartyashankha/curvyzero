from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "scripts" / "audit_curvytron_wave_a_capacity.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_curvytron_wave_a_capacity", AUDIT_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _app(
    *,
    app_id: str,
    description: str,
    state: str = "deployed",
    tasks: int = 0,
) -> dict:
    return {
        "App ID": app_id,
        "Description": description,
        "State": state,
        "Tasks": str(tasks),
        "Created at": "2026-06-23 12:00:00-07:00",
        "Stopped at": None,
    }


def _base_apps(audit) -> list[dict]:
    return [
        _app(app_id="ap-train", description=audit.CURVY_TRAIN_APP_DESCRIPTION),
        _app(app_id="ap-status", description=audit.CURVY_STATUS_APP_DESCRIPTION),
    ]


def _write_app_list(path: Path, apps: list[dict]) -> Path:
    path.write_text(json.dumps(apps, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_capacity_audit_accepts_clear_proxy_capacity(tmp_path):
    audit = _load_script()
    app_list = _write_app_list(tmp_path / "apps.json", _base_apps(audit))

    report = audit.build_report(
        audit.parse_args(
            [
                "--input",
                str(app_list),
                "--requested-h100-rows",
                "90",
                "--reserved-h100-buffer",
                "10",
                "--target-h100-envelope",
                "100",
                "--snapshot-time",
                "2026-06-23T19:00:00+00:00",
            ]
        )
    )

    assert report["ok"] is True
    assert report["total_tasks"] == 0
    assert report["projected_total_tasks"] == 90
    assert report["max_additional_rows_under_task_proxy"] == 100
    assert report["full_launch_within_task_proxy"] is True
    assert report["approval_recommendation"] == "capacity_proxy_clear"
    assert report["warning_count"] == 0


def test_capacity_audit_warns_when_task_proxy_exceeds_envelope(tmp_path):
    audit = _load_script()
    apps = _base_apps(audit)
    apps.append(
        _app(
            app_id="ap-other",
            description="unrelated-h100-work",
            state=audit.DETACHED_STATE,
            tasks=58,
        )
    )
    app_list = _write_app_list(tmp_path / "apps.json", apps)

    report = audit.build_report(
        audit.parse_args(
            [
                "--input",
                str(app_list),
                "--requested-h100-rows",
                "90",
                "--target-h100-envelope",
                "100",
            ]
        )
    )

    assert report["ok"] is True
    assert report["detached_running"] == 1
    assert report["detached_tasks"] == 58
    assert report["projected_total_tasks"] == 148
    assert report["max_additional_rows_under_task_proxy"] == 42
    assert report["full_launch_within_task_proxy"] is False
    assert report["approval_recommendation"] == "operator_capacity_review_required"
    assert any("coarse proxy" in warning["note"] for warning in report["warnings"])


def test_capacity_audit_rejects_missing_curvy_train_app(tmp_path):
    audit = _load_script()
    app_list = _write_app_list(
        tmp_path / "apps.json",
        [_app(app_id="ap-status", description=audit.CURVY_STATUS_APP_DESCRIPTION)],
    )

    report = audit.build_report(audit.parse_args(["--input", str(app_list)]))

    assert report["ok"] is False
    assert report["approval_recommendation"] == "do_not_launch_until_fixed"
    assert any("train app is absent" in error["message"] for error in report["errors"])


def test_capacity_audit_warns_when_curvy_train_already_running(tmp_path):
    audit = _load_script()
    apps = _base_apps(audit)
    apps[0]["Tasks"] = "2"
    app_list = _write_app_list(tmp_path / "apps.json", apps)

    report = audit.build_report(audit.parse_args(["--input", str(app_list)]))

    assert report["ok"] is True
    assert report["curvy_train_tasks"] == 2
    assert report["approval_recommendation"] == "operator_capacity_review_required"
    assert any("train app already has active tasks" in warning["message"] for warning in report["warnings"])
