from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_curvytron_rnd_blank_sweep_manifest.py"
SUBMIT_SCRIPT = ROOT / "scripts" / "submit_curvytron_survivaldiag_manifest.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rnd_blank_sweep_manifest_covers_full_default_weight_ladder(tmp_path: Path) -> None:
    module = _load_script(
        BUILD_SCRIPT,
        "build_curvytron_rnd_blank_sweep_manifest_for_shape_test",
    )
    args = module.parse_args(
        [
            "--matrix-name",
            "rnd-blank-sweep-test",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    manifest = module.build_manifest(args)

    assert manifest["matrix_profile"] == "rnd_blank_sweep"
    assert manifest["row_count"] == 9
    assert manifest["guards"]["expected_row_count"] == 9
    assert manifest["tournament"] == {
        "enabled": False,
        "intake_enabled": False,
        "reason": "explicit user request: no tournament",
    }
    assert [point["weight"] for point in manifest["axes"]["exploration_bonus_points"]] == [
        0.0,
        0.0,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        0.6,
        1.0,
    ]
    assert [point["label"] for point in manifest["axes"]["exploration_bonus_points"]] == [
        "no-bonus",
        "measure-only",
        "bonus-0p003",
        "bonus-0p01",
        "bonus-0p03",
        "bonus-0p10",
        "bonus-0p30",
        "bonus-0p60",
        "bonus-1p00",
    ]
    assert [point["mode"] for point in manifest["axes"]["exploration_bonus_points"]] == [
        "none",
        "rnd_meter_v0",
        "rnd_replay_target_v0",
        "rnd_replay_target_v0",
        "rnd_replay_target_v0",
        "rnd_replay_target_v0",
        "rnd_replay_target_v0",
        "rnd_replay_target_v0",
        "rnd_replay_target_v0",
    ]

    for row in manifest["rows"]:
        train_kwargs = row["train_kwargs"]
        poller_kwargs = row["poller_kwargs"]
        assert row["initial_policy_checkpoint_source"] == {
            "source": "scratch_random_initialization",
            "checkpoint_ref": None,
        }
        assert train_kwargs["initial_policy_checkpoint_ref"] is None
        assert train_kwargs["opponent_runtime_mode"] == "blank_canvas_noop"
        assert train_kwargs["opponent_death_mode"] == "normal"
        assert train_kwargs["opponent_assignment_ref"] is None
        assert train_kwargs["opponent_assignment_refresh_interval_train_iter"] == 0
        assert train_kwargs["own_checkpoint_opponent_refresh_enabled"] is False
        assert train_kwargs["background_gif_enabled"] is True
        assert poller_kwargs["background_gif_enabled"] is True
        assert "assignment_bank" not in manifest
        assert "training_candidate_refresh_controller" not in manifest


def test_rnd_blank_sweep_manifest_expands_custom_weights_and_replicas(
    tmp_path: Path,
) -> None:
    module = _load_script(
        BUILD_SCRIPT,
        "build_curvytron_rnd_blank_sweep_manifest_for_custom_test",
    )
    args = module.parse_args(
        [
            "--matrix-name",
            "rnd-blank-sweep-custom-test",
            "--output-root",
            str(tmp_path / "out"),
            "--weights",
            "0",
            "0.02",
            "0.5",
            "1.0",
            "--replicas",
            "2",
        ]
    )

    manifest = module.build_manifest(args)

    assert manifest["row_count"] == 10
    assert manifest["guards"]["expected_row_count"] == 10
    assert {
        (row["replica_index"], row["train_kwargs"]["exploration_bonus_weight"])
        for row in manifest["rows"]
        if row["train_kwargs"]["exploration_bonus_mode"] == "rnd_replay_target_v0"
    } == {
        (0, 0.02),
        (0, 0.5),
        (0, 1.0),
        (1, 0.02),
        (1, 0.5),
        (1, 1.0),
    }


def test_rnd_blank_sweep_default_run_ids_are_matrix_scoped(tmp_path: Path) -> None:
    module = _load_script(
        BUILD_SCRIPT,
        "build_curvytron_rnd_blank_sweep_manifest_for_collision_test",
    )
    args_a = module.parse_args(
        [
            "--matrix-name",
            "rnd-blank-sweep-a",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    args_b = module.parse_args(
        [
            "--profile",
            "rnd_blank_meter_gate",
            "--matrix-name",
            "rnd-blank-meter-gate-b",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    manifest_a = module.build_manifest(args_a)
    manifest_b = module.build_manifest(args_b)
    run_ids_a = {row["run_id"] for row in manifest_a["rows"]}
    run_ids_b = {row["run_id"] for row in manifest_b["rows"]}
    attempt_ids_a = {row["attempt_id"] for row in manifest_a["rows"]}
    attempt_ids_b = {row["attempt_id"] for row in manifest_b["rows"]}

    assert run_ids_a.isdisjoint(run_ids_b)
    assert attempt_ids_a.isdisjoint(attempt_ids_b)
    assert all(run_id.startswith("rnd-blank-sweep-a-") for run_id in run_ids_a)
    assert all(
        run_id.startswith("rnd-blank-meter-gate-b-") for run_id in run_ids_b
    )


def test_rnd_blank_meter_gate_profile_uses_matched_stock_and_meter_only(
    tmp_path: Path,
) -> None:
    module = _load_script(
        BUILD_SCRIPT,
        "build_curvytron_rnd_blank_sweep_manifest_for_gate_test",
    )
    args = module.parse_args(
        [
            "--profile",
            "rnd_blank_meter_gate",
            "--matrix-name",
            "rnd-blank-meter-gate-test",
            "--output-root",
            str(tmp_path / "out"),
            "--replicas",
            "3",
        ]
    )

    manifest = module.build_manifest(args)

    assert manifest["matrix_profile"] == "rnd_blank_meter_gate"
    assert manifest["row_count"] == 6
    assert manifest["guards"]["expected_row_count"] == 6
    assert manifest["guards"]["positive_rnd_mode_is_experimental"] is False
    assert [
        (row["replica_index"], row["train_kwargs"]["exploration_bonus_mode"])
        for row in manifest["rows"]
    ] == [
        (0, "none"),
        (0, "rnd_meter_v0"),
        (1, "none"),
        (1, "rnd_meter_v0"),
        (2, "none"),
        (2, "rnd_meter_v0"),
    ]
    assert {
        row["train_kwargs"]["exploration_bonus_weight"] for row in manifest["rows"]
    } == {0.0}
    assert [row["label"] for row in manifest["rows"]] == [
        "no-bonus-copy00",
        "measure-only-copy00",
        "no-bonus-copy01",
        "measure-only-copy01",
        "no-bonus-copy02",
        "measure-only-copy02",
    ]
    assert [row["plain_name"] for row in manifest["rows"]] == [
        "No exploration bonus",
        "Measure bonus only",
        "No exploration bonus / copy 1",
        "Measure bonus only / copy 1",
        "No exploration bonus / copy 2",
        "Measure bonus only / copy 2",
    ]


def test_rnd_blank_sweep_submitter_dry_run_selects_all_rows(tmp_path: Path) -> None:
    build = _load_script(
        BUILD_SCRIPT,
        "build_curvytron_rnd_blank_sweep_manifest_for_submit_test",
    )
    submit = _load_script(
        SUBMIT_SCRIPT,
        "submit_curvytron_survivaldiag_manifest_for_rnd_blank_sweep_test",
    )
    args = build.parse_args(
        [
            "--matrix-name",
            "rnd-blank-sweep-submit-test",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    manifest = build.build_manifest(args)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    submit_args = submit.parse_args([str(manifest_path)])
    loaded = submit._load_manifest(manifest_path)
    rows = submit._selected_rows(loaded, submit_args)
    records = [
        submit._launch_row(
            row,
            app_name=loaded["guards"]["deployed_app_name"],
            modal_env=None,
            dry_run=True,
        )
        for row in rows
    ]

    assert len(rows) == manifest["row_count"]
    assert len(records) == manifest["row_count"]
    assert all(record["status"] == "dry_run" for record in records)
