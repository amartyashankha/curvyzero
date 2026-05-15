from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "scripts" / "audit_curvytron_launch_manifest_refs.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_curvytron_launch_manifest_refs", AUDIT_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_ref(name: str = "iteration_10000.pth.tar") -> str:
    return (
        "training/lightzero-curvytron-visual-survival/run-a/attempts/try-a/"
        f"train/lightzero_exp/ckpt/{name}"
    )


def _manifest(*, extra_assignment_ref: str | None = None) -> dict:
    initial_ref = _checkpoint_ref()
    rank2_ref = extra_assignment_ref or _checkpoint_ref("iteration_20000.pth.tar")
    assignment = {
        "schema_id": "curvyzero_opponent_assignment/v1",
        "assignment_id": "assignment-a",
        "source_epoch": None,
        "source_ref": "leaderboard.json",
        "seed": 123,
        "entries": [
            {
                "name": "rank2",
                "weight": 100,
                "opponent_policy_kind": "frozen_lightzero_checkpoint",
                "opponent_runtime_mode": "normal",
                "opponent_immortal": False,
                "opponent_checkpoint_ref": rank2_ref,
            }
        ],
    }
    return {
        "schema_id": "curvyzero_curvytron_tonight18_manifest/v0",
        "top_checkpoint_source": {
            "rank1": {"checkpoint_ref": initial_ref},
            "rank2": {"checkpoint_ref": rank2_ref},
        },
        "assignment_bank": {
            "target_volume": "control",
            "assignments": {
                "recipe-a": {
                    "assignment_ref": "control:training/task/run/assignment.json",
                    "assignment": assignment,
                }
            },
        },
        "rows": [
            {
                "row_id": "r001",
                "initial_policy_checkpoint_ref": initial_ref,
                "train_kwargs": {
                    "initial_policy_checkpoint_ref": initial_ref,
                    "opponent_assignment_preview": assignment,
                    "opponent_mixture_spec": None,
                },
                "opponent_assignment_preview": assignment,
                "opponent_mixture_spec": None,
            }
        ],
    }


def test_launch_manifest_ref_audit_collects_initial_and_assignment_refs():
    audit = _load_script()
    refs, skipped = audit.collect_checkpoint_refs(_manifest())

    assert skipped == []
    assert [entry["ref"] for entry in refs] == [
        _checkpoint_ref(),
        _checkpoint_ref("iteration_20000.pth.tar"),
    ]
    sources = {entry["ref"]: entry["sources"] for entry in refs}
    assert "rows[r001].train_kwargs.initial_policy_checkpoint_ref" in sources[_checkpoint_ref()]
    assert any(
        source.startswith("assignment_bank.assignments.recipe-a.assignment")
        for source in sources[_checkpoint_ref("iteration_20000.pth.tar")]
    )


def test_launch_manifest_ref_audit_checks_local_runs_root(tmp_path):
    audit = _load_script()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    runs_root = tmp_path / "runs"
    (runs_root / _checkpoint_ref()).parent.mkdir(parents=True)
    (runs_root / _checkpoint_ref()).write_bytes(b"ckpt")

    args = audit.parse_args([str(manifest_path), "--runs-root", str(runs_root)])
    report = audit.build_report(args)

    assert report["ok"] is False
    assert report["missing_ref_count"] == 1
    assert report["missing_refs"][0]["ref"] == _checkpoint_ref("iteration_20000.pth.tar")


def test_launch_manifest_ref_audit_accepts_existing_local_refs(tmp_path):
    audit = _load_script()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    runs_root = tmp_path / "runs"
    for ref in (_checkpoint_ref(), _checkpoint_ref("iteration_20000.pth.tar")):
        (runs_root / ref).parent.mkdir(parents=True, exist_ok=True)
        (runs_root / ref).write_bytes(b"ckpt")

    args = audit.parse_args([str(manifest_path), "--runs-root", str(runs_root)])
    report = audit.build_report(args)

    assert report["ok"] is True
    assert report["missing_ref_count"] == 0
    assert report["bad_ref_count"] == 0


def test_launch_manifest_ref_audit_rejects_mutable_or_control_checkpoint_refs(tmp_path):
    audit = _load_script()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                extra_assignment_ref="control:" + _checkpoint_ref("ckpt_best.pth.tar")
            )
        ),
        encoding="utf-8",
    )

    args = audit.parse_args([str(manifest_path), "--syntax-only"])
    report = audit.build_report(args)

    assert report["ok"] is False
    assert report["bad_ref_count"] == 1
    assert report["bad_refs"][0]["reason"] == "mutable checkpoint ref"
