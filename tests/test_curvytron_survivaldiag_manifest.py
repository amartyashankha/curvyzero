from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_curvytron_survivaldiag_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_curvytron_survivaldiag_manifest", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(extra_args: list[str] | None = None):
    module = _load_module()
    args = module.parse_args(["--stdout-only", *(extra_args or [])])
    return module.build_manifest(args)


def test_survivaldiag_default_shape_is_large_ready_and_dry_run_only():
    manifest = _manifest()

    assert manifest["matrix_name"] == "curvy-survive-bonus-large"
    assert manifest["matrix_profile"] == "large_ready"
    assert manifest["dry_run_only"] is True
    assert manifest["launches_modal"] is False
    assert manifest["current_launch_approved"] is False
    assert manifest["row_count"] == 300
    assert manifest["logical_pair_count"] == 150
    assert len(manifest["gated_specs"]) == 10

    block_counts = {block["block_id"]: block["row_count"] for block in manifest["blocks"]}
    assert block_counts == {
        "b20_blank_canvas_all_level_repeats": 160,
        "b21_blank_canvas_medium_high_extra": 40,
        "b22_passive_immortal_dirty_controls": 40,
        "b23_blank_canvas_compute_sentinels": 60,
    }


def test_blank_repeat_expansion_profile_is_clean_v1c_shape():
    manifest = _manifest(
        [
            "--matrix-profile",
            "blank_repeat_expansion",
            "--matrix-name",
            "survivaldiag-v1c-blankrepeat-20260513i",
            "--run-prefix",
            "survivaldiag-v1c-blankrepeat-20260513i",
            "--attempt-prefix",
            "sdv1ci-a",
        ]
    )

    assert manifest["matrix_profile"] == "blank_repeat_expansion"
    assert manifest["row_count"] == 50
    assert manifest["logical_pair_count"] == 25
    assert len(manifest["gated_specs"]) == 10

    block_counts = {block["block_id"]: block["row_count"] for block in manifest["blocks"]}
    assert block_counts == {
        "b10_blank_canvas_scale_all": 32,
        "b11_blank_canvas_medium_high_scale": 8,
        "b12_passive_immortal_dirty_extension": 4,
        "b13_compute_sentinel_extension": 6,
    }
    assert {
        row["opponent_runtime_mode"]
        for row in manifest["rows"]
        if row["block_id"] in {"b10_blank_canvas_scale_all", "b11_blank_canvas_medium_high_scale"}
    } == {"blank_canvas_noop"}
    assert all(row["mode"] == "train" for row in manifest["rows"])
    assert all("two-seat-selfplay" not in row["command_text"] for row in manifest["rows"])
    assert all(len(row["run_id"]) <= 96 for row in manifest["rows"])
    assert all(len(row["attempt_id"]) <= 96 for row in manifest["rows"])


def test_blank_core_has_matched_render_pairs_and_separate_seed_fields():
    manifest = _manifest()
    blank_rows = [
        row for row in manifest["rows"] if row["block_id"] == "b20_blank_canvas_all_level_repeats"
    ]

    assert len(blank_rows) == 160
    assert {row["stochasticity_profile_id"] for row in blank_rows} == {
        "none",
        "low",
        "medium",
        "high",
    }
    assert {row["copy_id"] for row in blank_rows} == {f"c{index:02d}" for index in range(1, 21)}

    by_pair: dict[str, list[dict]] = {}
    for row in blank_rows:
        by_pair.setdefault(row["logical_pair_id"], []).append(row)
        for field in (
            "training_seed",
            "reset_seed",
            "opponent_policy_seed",
            "opponent_behavior_seed",
            "eval_seed",
            "copy_id",
        ):
            assert field in row

    assert len(by_pair) == 80
    for pair_rows in by_pair.values():
        assert {row["render_pair_role"] for row in pair_rows} == {"fast", "browser"}
        assert len({row["training_seed"] for row in pair_rows}) == 1
        assert len({row["reset_seed"] for row in pair_rows}) == 1
        assert len({row["eval_seed"] for row in pair_rows}) == 1


def test_executable_rows_use_supported_stock_train_lane_and_high_cap():
    module = _load_module()
    manifest = _manifest()

    for row in manifest["rows"]:
        command = row["command"]
        command_text = row["command_text"]
        assert row["mode"] == "train"
        assert len(row["run_id"]) <= 96
        assert len(row["attempt_id"]) <= 96
        assert command[command.index("--mode") + 1] == "train"
        assert "two-seat-selfplay" not in command_text
        assert row["calls_stock_train_muzero"] is True
        assert row["source_max_steps"] == 65536
        assert command[command.index("--source-max-steps") + 1] == "65536"
        assert row["flags"]["max_train_iter"] == 300000
        assert command[command.index("--max-train-iter") + 1] == "300000"
        assert row["flags"]["max_env_step"] == 30000000
        assert command[command.index("--max-env-step") + 1] == "30000000"
        assert row["flags"]["save_ckpt_after_iter"] == 15000
        assert command[command.index("--save-ckpt-after-iter") + 1] == "15000"
        assert row["flags"]["background_eval_poller_max_runtime_sec"] == 64800
        assert row["reward_variant"] == "survival_plus_bonus_no_outcome"
        assert command[command.index("--reward-variant") + 1] == (
            "survival_plus_bonus_no_outcome"
        )
        assert row["flags"]["background_eval_enabled"] is True
        assert row["flags"]["background_gif_enabled"] is True
        assert str(row["training_seed"]) in row["run_id"]
        assert row["compute_label"] in row["run_id"]
        assert row["flags"]["background_eval_compute"] == "cpu"
        assert row["flags"]["background_eval_id_prefix"] == "live_checkpoint"
        assert row["flags"]["background_gif_collect_temperature"] == 1.0
        assert row["flags"]["background_gif_collect_epsilon"] == 0.25
        assert "--no-background-eval-enabled" not in command
        assert "--no-background-gif-enabled" not in command
        assert row["deployed_app_submission"]["app_name"] == (
            "curvyzero-lightzero-curvytron-visual-survival-train"
        )
        assert row["deployed_app_submission"]["spawn_order"] == ["poller", "train"]
        assert row["deployed_app_submission"]["poller_function"] == (
            "lightzero_curvytron_visual_survival_checkpoint_eval_poller"
        )
        assert row["train_kwargs"]["run_id"] == row["run_id"]
        assert row["train_kwargs"]["attempt_id"] == row["attempt_id"]
        assert row["train_kwargs"]["background_eval_launch_kind"] == "poller"
        required_train_kwargs = set(module.TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT)
        assert set(row["train_kwargs"]) >= required_train_kwargs
        assert row["train_kwargs"]["decision_ms"] == module.DECISION_MS
        assert row["train_kwargs"]["decision_ms"] < 20.0
        assert row["train_kwargs"]["env_telemetry_stride"] == 1
        assert row["train_kwargs"]["background_eval_step_detail_limit"] == 4
        assert row["poller_kwargs"]["run_id"] == row["run_id"]
        assert row["poller_kwargs"]["attempt_id"] == row["attempt_id"]
        assert row["poller_kwargs"]["exp_name_ref"].endswith(
            "/train/lightzero_exp"
        )


def test_controls_are_labeled_and_expansions_are_gated_not_commanded():
    manifest = _manifest()

    dirty_rows = [row for row in manifest["rows"] if row["row_kind"] == "dirty_control"]
    assert len(dirty_rows) == 40
    assert {row["opponent_runtime_mode"] for row in dirty_rows} == {"normal"}
    assert {row["opponent_death_mode"] for row in dirty_rows} == {"immortal"}
    assert {row["stochasticity_profile_id"] for row in dirty_rows} == {
        "none",
        "low",
        "medium",
        "high",
    }
    assert {row["row_note"] for row in dirty_rows} == {
        "passive_immortal_dirty_control_not_main_claim"
    }

    compute_rows = [
        row for row in manifest["rows"] if row["row_kind"] == "compute_sentinel"
    ]
    assert len(compute_rows) == 60
    assert {row["compute_label"] for row in compute_rows} == {
        "search16",
        "collect64",
        "batch64",
    }
    assert all("expansion_gate" not in row for row in manifest["rows"])

    ancestor_rows = [
        row for row in manifest["rows"] if row["row_kind"] == "ancestor_sentinel"
    ]
    assert ancestor_rows == []

    gated = manifest["gated_specs"]
    gated_rewards = {spec["reward_variant"] for spec in gated}
    assert gated_rewards == {"survival_only", "survival_plus_bonus_no_outcome"}
    ancestor_specs = [
        spec
        for spec in gated
        if spec["block_id"] == "b05_ancestor_checkpoint_sentinels_gated"
    ]
    assert len(ancestor_specs) == 6
    assert {spec["opponent_policy_kind"] for spec in ancestor_specs} == {
        "frozen_lightzero_checkpoint"
    }
    assert all("latest" not in spec["opponent_checkpoint_ref"] for spec in ancestor_specs)
    assert all(spec["command_omitted"] is True for spec in gated)
    assert "scripted_opponents_until_remote_e2e_canaried" in manifest["guards"][
        "gated_expansions"
    ]
    assert "ancestor_checkpoint_controls_until_exact_lane_canaried" in manifest["guards"][
        "gated_expansions"
    ]


def test_manifest_declares_rich_readout_expectations():
    manifest = _manifest()
    expected = manifest["rich_readout_expectations"]

    assert "reward_components" in expected["reward"]
    assert "bonus_pickup_count" in expected["reward"]
    assert "terminal_cause_histogram" in expected["terminal"]
    assert "action_entropy" in expected["actions"]
    assert "gif_health" in expected["artifacts"]


def test_large_ready_names_are_human_readable():
    manifest = _manifest()
    run_ids = {row["run_id"] for row in manifest["rows"]}

    assert len(run_ids) == 300
    assert all(run_id.startswith("curvy-survive-bonus-") for run_id in run_ids)
    assert not any("survbonusnoout" in run_id for run_id in run_ids)
    assert not any("blanknoop" in run_id for run_id in run_ids)
    assert not any("armed" in run_id for run_id in run_ids)
    assert not any("l4t4c40" in run_id for run_id in run_ids)
    assert {
        row["label"].split("-")[0]
        for row in manifest["rows"]
    } == {"blank", "passive"}
    assert {"base", "search16", "collect64", "batch64"} <= {
        row["compute_label"] for row in manifest["rows"]
    }


def test_manifest_records_tiny_canary_stop_warnings():
    manifest = _manifest()
    guards = manifest["guards"]

    assert guards["max_train_iter_semantics"] == (
        "not_strict_for_tiny_canaries_checked_after_collect_update_block"
    )
    assert guards["tiny_canary_stop_required"] == "--stop-after-learner-train-calls"
    assert guards["forced_stop_status_warning"] == (
        "heartbeat_or_poller_running_may_be_stale_verify_modal_app_state"
    )
