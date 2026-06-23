from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_curvytron_survivaldiag_manifest.py"
NEXT_BATCH_BUILD_SCRIPT = ROOT / "scripts" / "build_curvytron_next_batch_manifest.py"
SUBMIT_SCRIPT = ROOT / "scripts" / "submit_curvytron_survivaldiag_manifest.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_payload() -> dict:
    module = _load_script(BUILD_SCRIPT, "build_curvytron_survivaldiag_manifest_for_submit")
    args = module.parse_args(["--stdout-only"])
    return module.build_manifest(args)


def _ratings_snapshot(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "ratings": [
                    {
                        "rank": rank,
                        "checkpoint_id": f"ckpt-rank{rank}",
                        "rating": 2000.0 - rank,
                        "status": "active",
                        "run_id": f"source-run-{rank}",
                        "attempt_id": f"try-source-run-{rank}",
                        "iteration": 10000 * rank,
                        "checkpoint_ref": (
                            "training/lightzero-curvytron-visual-survival/"
                            f"source-run-{rank}/attempts/try-source-run-{rank}/train/"
                            f"lightzero_exp/ckpt/iteration_{10000 * rank}.pth.tar"
                        ),
                    }
                    for rank in range(1, 5)
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _compact_row() -> dict:
    return {
        "row_id": "compact-guard-row",
        "label": "compact guard",
        "run_id": "compact-guard-run",
        "attempt_id": "compact-guard-attempt",
        "initial_policy_checkpoint_source": {
            "source": "scratch_random_initialization",
            "checkpoint_ref": None,
        },
        "deployed_app_submission": {
            "app_name": "curvyzero-lightzero-curvytron-visual-survival-train-v2",
            "train_function": "lightzero_curvytron_visual_survival_gpu_cpu40",
            "poller_function": "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
        },
        "experiment_spec": {"seed": 11},
    }


def _next_batch_canary_manifest_payload(tmp_path: Path) -> dict:
    module = _load_script(
        NEXT_BATCH_BUILD_SCRIPT,
        "build_curvytron_next_batch_manifest_for_submit",
    )
    args = module.parse_args(
        [
            "--profile",
            "canary",
            "--matrix-name",
            "cz26-submit-canary-test",
            "--ratings-snapshot",
            str(_ratings_snapshot(tmp_path / "ratings.json")),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    return module.build_manifest(args)


def test_grouped_submitter_dry_run_selects_rows_and_preserves_two_call_shape(tmp_path):
    submit = _load_script(SUBMIT_SCRIPT, "submit_curvytron_survivaldiag_manifest_for_test")
    manifest = _manifest_payload()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = submit.parse_args([str(manifest_path), "--limit", "3"])
    loaded = submit._load_manifest(manifest_path)
    rows = submit._selected_rows(loaded, args)
    records = [
        submit._launch_row(
            row,
            app_name=loaded["guards"]["deployed_app_name"],
            modal_env=None,
            dry_run=True,
        )
        for row in rows
    ]

    assert len(records) == 3
    assert all(record["status"] == "dry_run" for record in records)
    assert all(
        record["app_name"] == "curvyzero-lightzero-curvytron-visual-survival-train-v2"
        for record in records
    )
    assert all(
        record["poller_function"]
        == "lightzero_curvytron_visual_survival_checkpoint_eval_poller"
        for record in records
    )
    assert all(
        record["train_function"] == "lightzero_curvytron_visual_survival_gpu_cpu40"
        for record in records
    )


def test_grouped_submitter_refuses_partial_launch_without_explicit_override(tmp_path):
    submit = _load_script(
        SUBMIT_SCRIPT,
        "submit_curvytron_survivaldiag_manifest_partial_launch_guard",
    )
    manifest = _manifest_payload()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = submit.parse_args([str(manifest_path), "--allow-launch", "--limit", "1"])
    loaded = submit._load_manifest(manifest_path)
    rows = submit._selected_rows(loaded, args)

    with pytest.raises(ValueError, match="refusing partial launch"):
        submit._validate_partial_launch_selection(loaded, rows, args)


def test_grouped_submitter_allows_partial_launch_with_explicit_override(tmp_path):
    submit = _load_script(
        SUBMIT_SCRIPT,
        "submit_curvytron_survivaldiag_manifest_partial_launch_override",
    )
    manifest = _manifest_payload()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = submit.parse_args(
        [
            str(manifest_path),
            "--allow-launch",
            "--limit",
            "1",
            "--allow-partial-launch",
        ]
    )
    loaded = submit._load_manifest(manifest_path)
    rows = submit._selected_rows(loaded, args)

    submit._validate_partial_launch_selection(loaded, rows, args)


def test_grouped_submitter_accepts_legacy_full_train_kwargs_in_dry_run():
    submit = _load_script(SUBMIT_SCRIPT, "submit_curvytron_survivaldiag_submit_legacy")
    manifest = _manifest_payload()
    row = manifest["rows"][0]

    assert "decision_ms" in row["train_kwargs"]
    assert "collector_env_num" in row["train_kwargs"]
    assert "batch_size" in row["train_kwargs"]
    assert "background_eval_enabled" in row["train_kwargs"]

    record = submit._launch_row(
        row,
        app_name=manifest["guards"]["deployed_app_name"],
        modal_env=None,
        dry_run=True,
    )

    assert record["status"] == "dry_run"
    assert record["train_function"] == "lightzero_curvytron_visual_survival_gpu_cpu40"
    assert (
        record["poller_function"]
        == "lightzero_curvytron_visual_survival_checkpoint_eval_poller"
    )


def test_grouped_submitter_rejects_incomplete_train_kwargs_identity_only():
    submit = _load_script(SUBMIT_SCRIPT, "submit_curvytron_survivaldiag_submit_missing")
    manifest = _manifest_payload()
    row = dict(manifest["rows"][0])
    row["train_kwargs"] = dict(row["train_kwargs"])
    row["train_kwargs"].pop("run_id")

    try:
        submit._launch_row(
            row,
            app_name=manifest["guards"]["deployed_app_name"],
            modal_env=None,
            dry_run=True,
        )
    except ValueError as exc:
        assert "train_kwargs missing required keys" in str(exc)
        assert "run_id" in str(exc)
        assert "decision_ms" not in str(exc)
    else:
        raise AssertionError("incomplete train kwargs were accepted")


def test_grouped_submitter_accepts_minimal_compact_train_kwargs_with_default_policy_surface():
    submit = _load_script(SUBMIT_SCRIPT, "submit_curvytron_survivaldiag_submit_compact")
    row = {
        "row_id": "compact-default-row",
        "label": "compact defaults",
        "run_id": "compact-default-run",
        "attempt_id": "compact-default-attempt",
        "initial_policy_checkpoint_source": {
            "source": "scratch_random_initialization",
            "checkpoint_ref": None,
        },
        "deployed_app_submission": {
            "app_name": "curvyzero-lightzero-curvytron-visual-survival-train-v2",
            "train_function": "lightzero_curvytron_visual_survival_gpu_cpu40",
            "poller_function": "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
        },
        "train_kwargs": {
            "mode": "train",
            "seed": 11,
            "run_id": "compact-default-run",
            "attempt_id": "compact-default-attempt",
        },
        "poller_kwargs": {},
    }

    assert set(row["train_kwargs"]) == set(submit.TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT)
    assert "decision_ms" not in row["train_kwargs"]
    assert "source_state_trail_render_mode" not in row["train_kwargs"]
    assert "source_state_bonus_render_mode" not in row["train_kwargs"]
    train_kwargs, poller_kwargs = submit._normalized_launch_kwargs(row)
    assert poller_kwargs["run_id"] == "compact-default-run"
    assert poller_kwargs["attempt_id"] == "compact-default-attempt"
    assert poller_kwargs["seed"] == 11
    assert submit._launch_row(
        row,
        app_name="curvyzero-lightzero-curvytron-visual-survival-train-v2",
        modal_env=None,
        dry_run=True,
    )["status"] == "dry_run"
    assert train_kwargs == row["train_kwargs"]


def test_grouped_submitter_compact_experiment_row_normalizes_before_spawn():
    submit = _load_script(
        SUBMIT_SCRIPT,
        "submit_curvytron_survivaldiag_submit_experiment_compact",
    )
    row = {
        "row_id": "compact-experiment-row",
        "label": "compact experiment",
        "run_id": "compact-experiment-run",
        "attempt_id": "compact-experiment-attempt",
        "initial_policy_checkpoint_ref": (
            "training/lightzero-curvytron-visual-survival/source-run/"
            "attempts/source-attempt/train/lightzero_exp/ckpt/iteration_10000.pth.tar"
        ),
        "deployed_app_submission": {
            "app_name": "curvyzero-lightzero-curvytron-visual-survival-train-v2",
            "train_function": "lightzero_curvytron_visual_survival_gpu_cpu40",
            "poller_function": "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
        },
        "experiment_spec": {
            "seed": 12,
            "reward_variant": "survival_plus_bonus_plus_outcome",
            "reward_outcome_alpha": 0.5,
            "opponent_policy_kind": "fixed_straight",
            "action_noise_probability": 0.10,
        },
        "poller_kwargs": {},
    }

    train_kwargs, poller_kwargs = submit._normalized_launch_kwargs(row)

    assert train_kwargs["mode"] == "train"
    assert train_kwargs["run_id"] == "compact-experiment-run"
    assert train_kwargs["initial_policy_checkpoint_load_mode"] == "matching_shape"
    assert train_kwargs["ego_action_straight_override_probability"] == 0.10
    assert train_kwargs["policy_action_repeat_max"] == 2
    assert train_kwargs["control_noise_profile_id"] == "straight_override_p10_repeat_p10"
    assert poller_kwargs["run_id"] == train_kwargs["run_id"]
    assert poller_kwargs["reward_variant"] == train_kwargs["reward_variant"]
    assert "initial_policy_checkpoint_ref" not in poller_kwargs
    assert submit._launch_row(
        row,
        app_name="curvyzero-lightzero-curvytron-visual-survival-train-v2",
        modal_env=None,
        dry_run=True,
    )["status"] == "dry_run"


def test_grouped_submitter_compact_experiment_row_rejects_unknown_scale():
    submit = _load_script(
        SUBMIT_SCRIPT,
        "submit_curvytron_survivaldiag_submit_experiment_bad_scale",
    )
    row = {
        "row_id": "compact-experiment-bad-scale",
        "label": "compact experiment bad scale",
        "run_id": "compact-experiment-bad-scale-run",
        "attempt_id": "compact-experiment-bad-scale-attempt",
        "initial_policy_checkpoint_source": {
            "source": "scratch_random_initialization",
            "checkpoint_ref": None,
        },
        "deployed_app_submission": {
            "app_name": "curvyzero-lightzero-curvytron-visual-survival-train-v2",
            "train_function": "lightzero_curvytron_visual_survival_gpu_cpu40",
            "poller_function": "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
        },
        "experiment_spec": {"seed": 12, "scale_preset": "old_fast_lane"},
        "poller_kwargs": {},
    }

    try:
        submit._normalized_launch_kwargs(row)
    except ValueError as exc:
        assert "scale_preset must be 'current_broad'" in str(exc)
    else:
        raise AssertionError("expected stale compact scale to fail")


def test_grouped_submitter_compact_row_rejects_train_only_poller_fields():
    submit = _load_script(
        SUBMIT_SCRIPT,
        "submit_curvytron_survivaldiag_submit_compact_poller_guard",
    )
    unsafe_keys = (
        "initial_policy_checkpoint_ref",
        "initial_policy_checkpoint_state_key",
        "initial_policy_checkpoint_load_mode",
        "source_state_trail_render_mode",
        "source_state_bonus_render_mode",
        "policy_observation_backend",
        "learner_seat_mode",
        "collector_env_num",
        "batch_size",
        "save_ckpt_after_iter",
        "commit_on_checkpoint",
        "opponent_assignment_refresh_ref",
    )

    for key in unsafe_keys:
        row = {
            "row_id": f"compact-poller-guard-{key}",
            "label": "compact defaults",
            "run_id": "compact-poller-guard-run",
            "attempt_id": "compact-poller-guard-attempt",
            "initial_policy_checkpoint_source": {
                "source": "scratch_random_initialization",
                "checkpoint_ref": None,
            },
            "deployed_app_submission": {
                "app_name": "curvyzero-lightzero-curvytron-visual-survival-train-v2",
                "train_function": "lightzero_curvytron_visual_survival_gpu_cpu40",
                "poller_function": (
                    "lightzero_curvytron_visual_survival_checkpoint_eval_poller"
                ),
            },
            "train_kwargs": {
                "mode": "train",
                "seed": 11,
                "run_id": "compact-poller-guard-run",
                "attempt_id": "compact-poller-guard-attempt",
            },
            "poller_kwargs": {key: "unsafe"},
        }
        try:
            submit._launch_row(
                row,
                app_name="curvyzero-lightzero-curvytron-visual-survival-train-v2",
                modal_env=None,
                dry_run=True,
            )
        except ValueError as exc:
            assert "poller_kwargs has unsupported keys" in str(exc)
            assert key in str(exc)
        else:
            raise AssertionError(f"expected poller key {key!r} to be rejected")


def test_compact_row_rejects_poller_identity_divergence():
    submit = _load_script(SUBMIT_SCRIPT, "submit_compact_poller_identity_guard")
    row = _compact_row()
    row["poller_kwargs"] = {"run_id": "other-run"}

    with pytest.raises(ValueError, match="train/poller run_id"):
        submit._launch_row(
            row,
            app_name="curvyzero-lightzero-curvytron-visual-survival-train-v2",
            modal_env=None,
            dry_run=True,
        )

    row = _compact_row()
    row["poller_overrides"] = {"attempt_id": "other-attempt"}

    with pytest.raises(ValueError, match="train/poller attempt_id"):
        submit._launch_row(
            row,
            app_name="curvyzero-lightzero-curvytron-visual-survival-train-v2",
            modal_env=None,
            dry_run=True,
        )


def test_compact_row_rejects_train_overrides_for_identity_fields():
    submit = _load_script(SUBMIT_SCRIPT, "submit_compact_train_identity_override_guard")
    for key in ("mode", "seed", "run_id", "attempt_id"):
        row = _compact_row()
        row["train_overrides"] = {key: "unsafe-override"}

        with pytest.raises(ValueError, match=f"train_overrides.*{key}"):
            submit._normalized_launch_kwargs(row)


def test_compact_row_rejects_mixed_legacy_and_experiment_spec():
    submit = _load_script(SUBMIT_SCRIPT, "submit_compact_mixed_schema_guard")
    row = _compact_row()
    row["train_kwargs"] = {
        "mode": "train",
        "seed": 99,
        "run_id": "legacy-run",
        "attempt_id": "legacy-attempt",
    }

    with pytest.raises(ValueError, match="must not include both train_kwargs and experiment_spec"):
        submit._normalized_launch_kwargs(row)


def test_compact_row_rejects_conflicting_runtime_and_top_level_refs():
    submit = _load_script(SUBMIT_SCRIPT, "submit_compact_duplicate_ref_guard")
    row = _compact_row()
    row["runtime_spec"] = {
        "initial_policy_checkpoint_ref": "training/x/iteration_10000.pth.tar"
    }
    row["initial_policy_checkpoint_ref"] = "training/y/iteration_20000.pth.tar"

    with pytest.raises(ValueError, match="initial_policy_checkpoint_ref.*conflict"):
        submit._normalized_launch_kwargs(row)

    row = _compact_row()
    row["runtime_spec"] = {"opponent_assignment_ref": "runs:a.json"}
    row["opponent_assignment_ref"] = "runs:b.json"

    with pytest.raises(ValueError, match="opponent_assignment_ref.*conflict"):
        submit._normalized_launch_kwargs(row)


def test_compact_action_noise_probability_edges_are_explicit():
    submit = _load_script(SUBMIT_SCRIPT, "submit_compact_noise_edges")

    row = _compact_row()
    row["experiment_spec"]["action_noise_probability"] = 0.0
    train_kwargs, _ = submit._normalized_launch_kwargs(row)
    assert train_kwargs["control_noise_profile_id"] == "none"
    assert train_kwargs["policy_action_repeat_max"] == 1

    row = _compact_row()
    row["experiment_spec"]["action_noise_probability"] = 1.0
    train_kwargs, _ = submit._normalized_launch_kwargs(row)
    assert train_kwargs["control_noise_profile_id"] == "straight_override_p100_repeat_p100"
    assert train_kwargs["policy_action_repeat_max"] == 2

    row = _compact_row()
    row["experiment_spec"]["action_noise_probability"] = 0.125
    with pytest.raises(ValueError, match="whole percent"):
        submit._normalized_launch_kwargs(row)


def test_compact_row_rejects_action_noise_bundle_override():
    submit = _load_script(SUBMIT_SCRIPT, "submit_compact_noise_override_guard")
    row = _compact_row()
    row["experiment_spec"]["action_noise_probability"] = 0.10
    row["train_overrides"] = {"policy_action_repeat_extra_probability": 0.20}

    with pytest.raises(ValueError, match="action_noise_probability.*train_overrides"):
        submit._normalized_launch_kwargs(row)


def test_next_batch_submitter_dry_run_includes_training_candidate_config(tmp_path):
    submit = _load_script(
        SUBMIT_SCRIPT,
        "submit_curvytron_survivaldiag_manifest_next_batch_for_test",
    )
    manifest = _next_batch_canary_manifest_payload(tmp_path)
    manifest_path = tmp_path / "next-batch-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = submit.parse_args([str(manifest_path), "--publish-assignments-only"])
    loaded = submit._load_manifest(manifest_path)
    rows = submit._selected_rows(loaded, args)
    app_name = loaded["guards"]["deployed_app_name"]

    assignment_records = submit._publish_assignment_bank(
        loaded,
        rows,
        app_name=app_name,
        modal_env=None,
        dry_run=True,
    )
    refresh_pointer_records = submit._publish_refresh_pointers(
        loaded,
        rows,
        modal_env=None,
        dry_run=True,
    )
    config_record = submit._publish_training_candidate_refresh_config(
        loaded,
        rows,
        modal_env=None,
        dry_run=True,
    )

    assert len(assignment_records) == 1
    assert len(refresh_pointer_records) == 1
    assert config_record is not None
    assert config_record["status"] == "dry_run"
    assert config_record["tournament_id"] == manifest["tournament"]["tournament_id"]
    assert config_record["rating_run_id"] == manifest["tournament"]["rating_run_id"]
    assert config_record["leaderboard_id"] == manifest["tournament"]["leaderboard_id"]
    assert config_record["refresh_pointer_count"] == 1
