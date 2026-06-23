from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_curvytron_restart_source_refs.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_curvytron_restart_source_refs", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_ref(iteration: int) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/run-a/attempts/try-a/"
        f"train/lightzero_exp/ckpt/iteration_{iteration}.pth.tar"
    )


def _snapshot() -> dict:
    return {
        "leaderboard_id": "source-leaderboard",
        "snapshot_id": "source-r0",
        "context": {"rating_context_hash": "abc"},
        "rows": [
            {
                "rank": 2,
                "status": "active",
                "checkpoint_ref": _checkpoint_ref(20000),
                "checkpoint_id": "ckpt-2",
                "iteration": 20000,
                "rating": 1501.0,
            },
            {
                "rank": 1,
                "status": "active",
                "checkpoint_ref": _checkpoint_ref(10000),
                "checkpoint_id": "ckpt-1",
                "iteration": 10000,
                "rating": 1502.0,
            },
            {
                "rank": 3,
                "status": "retired",
                "checkpoint_ref": _checkpoint_ref(30000),
                "checkpoint_id": "ckpt-3",
                "iteration": 30000,
                "rating": 1500.0,
            },
        ],
    }


def test_restart_source_plan_selects_active_rows_in_rank_order(tmp_path):
    script = _load_script()
    source = tmp_path / "leaderboard.json"
    source.write_text(json.dumps(_snapshot()), encoding="utf-8")

    args = script.parse_args(
        [
            str(source),
            "--limit",
            "2",
            "--output-root",
            str(tmp_path / "out"),
            "--plan-id",
            "plan-a",
        ]
    )
    plan = script.build_plan(args)

    assert [entry["checkpoint_ref"] for entry in plan["selected"]] == [
        _checkpoint_ref(10000),
        _checkpoint_ref(20000),
    ]
    assert plan["target_volume"] == "curvyzero-runs-v2"
    assert plan["selection"]["min_required_iteration"] == 0
    assert plan["selection"]["iteration_zero_count"] == 0
    assert plan["selection"]["min_iteration"] == 10000
    assert plan["selection"]["max_iteration"] == 20000
    assert plan["rerate"]["policy_trail_render_mode"] == "browser_lines"
    assert plan["rerate"]["policy_bonus_render_mode"] == "simple_symbols"
    assert "--allow-non-v2-runs-volume" in plan["commands"]["audit_source"][0]
    assert "curvyzero-runs-v2" in plan["commands"]["audit_target_after_copy"][0]
    assert "--checkpoint-refs \"$REFS\"" in "\n".join(plan["commands"]["rating"])


def test_restart_source_plan_writes_refs_selection_and_commands(tmp_path):
    script = _load_script()
    source = tmp_path / "leaderboard.json"
    source.write_text(json.dumps(_snapshot()), encoding="utf-8")
    args = script.parse_args(
        [
            str(source),
            "--limit",
            "2",
            "--output-root",
            str(tmp_path / "out"),
            "--plan-id",
            "plan-a",
        ]
    )
    plan = script.build_plan(args)
    outputs = script._write_plan(plan, output_root=args.output_root)

    assert Path(outputs["refs_txt"]).read_text(encoding="utf-8").splitlines() == [
        _checkpoint_ref(10000),
        _checkpoint_ref(20000),
    ]
    commands = Path(outputs["commands_txt"]).read_text(encoding="utf-8")
    assert "source-refs-old-volume-audit.json" in commands
    assert "modal volume get --force curvyzero-runs" in commands
    assert "modal volume put --force curvyzero-runs-v2" in commands
    assert "source-refs-v2-target-after-copy-audit.json" in commands


def test_restart_source_plan_can_exclude_iteration_zero_rows(tmp_path):
    script = _load_script()
    payload = _snapshot()
    payload["rows"].insert(
        0,
        {
            "rank": 0,
            "status": "active",
            "checkpoint_ref": _checkpoint_ref(0),
            "checkpoint_id": "ckpt-zero",
            "iteration": 0,
            "rating": 1600.0,
        },
    )
    source = tmp_path / "leaderboard.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    args = script.parse_args(
        [
            str(source),
            "--limit",
            "2",
            "--min-iteration",
            "1",
            "--output-root",
            str(tmp_path / "out"),
            "--plan-id",
            "plan-nonzero",
        ]
    )
    plan = script.build_plan(args)

    assert [entry["checkpoint_ref"] for entry in plan["selected"]] == [
        _checkpoint_ref(10000),
        _checkpoint_ref(20000),
    ]
    assert plan["selection"]["min_required_iteration"] == 1
    assert plan["selection"]["min_iteration"] == 10000
    assert plan["selection"]["iteration_zero_count"] == 0


def test_restart_source_plan_rejects_mutable_refs(tmp_path):
    script = _load_script()
    payload = _snapshot()
    payload["rows"][1]["checkpoint_ref"] = (
        "training/lightzero-curvytron-visual-survival/run-a/attempts/try-a/"
        "train/lightzero_exp/ckpt/ckpt_best.pth.tar"
    )
    source = tmp_path / "leaderboard.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    args = script.parse_args([str(source), "--limit", "1"])
    try:
        script.build_plan(args)
    except ValueError as exc:
        assert "mutable" in str(exc)
    else:
        raise AssertionError("mutable checkpoint ref was accepted")


def test_restart_source_plan_requires_v2_target_volume(tmp_path):
    script = _load_script()
    source = tmp_path / "leaderboard.json"
    source.write_text(json.dumps(_snapshot()), encoding="utf-8")

    args = script.parse_args([str(source), "--limit", "1", "--target-volume", "curvyzero-runs"])
    try:
        script.build_plan(args)
    except ValueError as exc:
        assert "all-v2" in str(exc)
    else:
        raise AssertionError("non-v2 target volume was accepted")
