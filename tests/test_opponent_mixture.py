import json

import pytest

from curvyzero.infra.modal import (
    lightzero_curvytron_visual_survival_eval as eval_mod,
)
from curvyzero.infra.modal import (
    lightzero_curvyzero_stacked_debug_visual_survival_train as train_mod,
)
from curvyzero.training.opponent_mixture import (
    OPPONENT_MIXTURE_SCHEMA_ID,
    OPPONENT_MIXTURE_SELECTION_UNIT,
    parse_opponent_mixture_spec,
    select_opponent_mixture_entry,
)


def test_opponent_mixture_parse_validates_supported_entry_shapes():
    spec = {
        "seed": 17,
        "entries": [
            {
                "name": "blank",
                "weight": 2,
                "opponent_policy_kind": "fixed_straight",
                "opponent_runtime_mode": "blank_canvas_noop",
            },
            {
                "name": "passive",
                "weight": 1,
                "opponent_policy_kind": "fixed_straight",
                "opponent_death_mode": "immortal",
            },
            {
                "name": "wall",
                "weight": 1,
                "opponent_policy_kind": "proactive_wall_avoidant",
            },
            {
                "name": "recent_001",
                "age_label": "recent",
                "weight": 1,
                "opponent_policy_kind": "frozen_lightzero_checkpoint",
                "opponent_checkpoint_ref": (
                    "training/lightzero-curvytron-visual-survival/run/checkpoints/"
                    "lightzero/iteration_123.pth.tar"
                ),
                "opponent_snapshot_ref": "recent-run-iteration-123",
            },
        ],
    }

    parsed = parse_opponent_mixture_spec(json.dumps(spec))

    assert parsed is not None
    assert parsed["schema_id"] == OPPONENT_MIXTURE_SCHEMA_ID
    assert parsed["selection_unit"] == OPPONENT_MIXTURE_SELECTION_UNIT
    assert parsed["total_weight"] == 5.0
    assert [entry["name"] for entry in parsed["entries"]] == [
        "blank",
        "passive",
        "wall",
        "recent_001",
    ]


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"name": "missing_weight", "opponent_policy_kind": "fixed_straight"}, "weight"),
        ({"name": "zero", "weight": 0, "opponent_policy_kind": "fixed_straight"}, "positive"),
        ({"name": "unknown", "weight": 1, "opponent_policy_kind": "mystery"}, "unsupported"),
        (
            {
                "name": "frozen_missing_ref",
                "weight": 1,
                "opponent_policy_kind": "frozen_lightzero_checkpoint",
            },
            "requires opponent_checkpoint_ref",
        ),
        (
            {
                "name": "typo",
                "weight": 1,
                "opponent_policy_kind": "fixed_straight",
                "opponent_rutime_mode": "blank_canvas_noop",
            },
            "unknown keys",
        ),
    ],
)
def test_opponent_mixture_rejects_ambiguous_or_invalid_entries(entry, match):
    with pytest.raises(ValueError, match=match):
        parse_opponent_mixture_spec({"entries": [entry]})


def test_opponent_mixture_selection_is_deterministic_and_episode_scoped():
    spec = parse_opponent_mixture_spec(
        {
            "seed": 3,
            "entries": [
                {
                    "name": "blank",
                    "weight": 1,
                    "opponent_policy_kind": "fixed_straight",
                    "opponent_runtime_mode": "blank_canvas_noop",
                },
                {
                    "name": "wall",
                    "weight": 1,
                    "opponent_policy_kind": "proactive_wall_avoidant",
                },
            ],
        }
    )

    first = select_opponent_mixture_entry(spec, episode_seed=123, reset_index=0)
    repeated = select_opponent_mixture_entry(spec, episode_seed=123, reset_index=0)
    later = select_opponent_mixture_entry(spec, episode_seed=123, reset_index=1)

    assert repeated == first
    assert first["selection_unit"] == OPPONENT_MIXTURE_SELECTION_UNIT
    assert later["selection_unit"] == OPPONENT_MIXTURE_SELECTION_UNIT
    assert {first["name"], later["name"]} <= {"blank", "wall"}


def test_modal_mixture_resolution_rejects_non_iteration_checkpoint_ref():
    with pytest.raises(ValueError, match="iteration_N\\.pth\\.tar"):
        train_mod._resolve_opponent_mixture_for_env(
            opponent_mixture_spec={
                "entries": [
                    {
                        "name": "recent",
                        "weight": 1,
                        "age_label": "recent",
                        "opponent_policy_kind": "frozen_lightzero_checkpoint",
                        "opponent_checkpoint_ref": (
                            "training/run/checkpoints/lightzero/custom_ref.pth.tar"
                        ),
                    }
                ]
            }
        )


def test_modal_mixture_resolution_rejects_mutable_frozen_checkpoint_ref():
    with pytest.raises(ValueError, match="immutable"):
        train_mod._resolve_opponent_mixture_for_env(
            opponent_mixture_spec={
                "entries": [
                    {
                        "name": "recent",
                        "weight": 1,
                        "age_label": "recent",
                        "opponent_policy_kind": "frozen_lightzero_checkpoint",
                        "opponent_checkpoint_ref": (
                            "training/run/checkpoints/lightzero/latest.pth.tar"
                        ),
                    }
                ]
            }
        )


def test_modal_mixture_resolution_keeps_static_frozen_refs(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "iteration_123.pth.tar"
    checkpoint_path.write_bytes(b"fake-checkpoint")

    def fake_resolve(ref, *, mount, remote_root):
        assert ref == "training/run/checkpoints/lightzero/iteration_123.pth.tar"
        assert mount == train_mod.RUNS_MOUNT
        assert remote_root == train_mod.REMOTE_ROOT
        return checkpoint_path, {"source_ref": ref, "mount": str(mount)}

    monkeypatch.setattr(train_mod.runs, "resolve_mounted_ref_or_path", fake_resolve)

    resolved = train_mod._resolve_opponent_mixture_for_env(
        opponent_mixture_spec={
            "entries": [
                {
                    "name": "somewhat_recent",
                    "weight": 1,
                    "age_label": "somewhat_recent",
                    "opponent_policy_kind": "frozen_lightzero_checkpoint",
                    "opponent_checkpoint_ref": (
                        "training/run/checkpoints/lightzero/iteration_123.pth.tar"
                    ),
                }
            ]
        }
    )

    assert resolved is not None
    entry = resolved["entries"][0]
    assert entry["age_label"] == "somewhat_recent"
    assert entry["opponent_checkpoint_ref"] == (
        "training/run/checkpoints/lightzero/iteration_123.pth.tar"
    )
    assert entry["opponent_checkpoint_path"] == str(checkpoint_path)


def test_background_eval_config_threads_opponent_mixture_to_eval_and_gif():
    mixture = parse_opponent_mixture_spec(
        {
            "entries": [
                {
                    "name": "blank",
                    "weight": 1,
                    "opponent_policy_kind": "fixed_straight",
                    "opponent_runtime_mode": "blank_canvas_noop",
                }
            ]
        }
    )
    command = {
        "seed": 123,
        "reward_variant": "survival_plus_bonus_no_outcome",
        "env_variant": "source_state_fixed_opponent",
        "source_max_steps": 65536,
        "opponent_policy_kind": "fixed_straight",
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_state_key": None,
        "opponent_death_mode": "normal",
        "opponent_runtime_mode": "normal",
        "opponent_mixture": mixture,
        "background_eval_enabled": True,
        "background_gif_enabled": True,
    }

    config = train_mod._background_eval_config_from_command(command)

    assert config["opponent_mixture"]["entries"][0]["name"] == "blank"
    assert config["selfplay_gif"]["opponent_mixture"]["entries"][0]["name"] == "blank"


def test_eval_row_records_selected_opponent_mixture_component():
    row = eval_mod._row_from_result(
        {
            "index": 0,
            "checkpoint_label": "iteration_7",
            "checkpoint_ref": "training/run/checkpoints/lightzero/iteration_7.pth.tar",
            "seed": 123,
        },
        {
            "ok": True,
            "episode": {
                "steps_survived": 42,
                "cap": 64,
                "terminal_reason": "wall",
                "opponent_mixture_enabled": True,
                "opponent_mixture_entry_name": "recent_001",
                "opponent_mixture_entry_weight": 1.5,
                "opponent_mixture_entry_index": 3,
                "opponent_mixture_age_label": "recent",
            },
            "status": {"strict_policy_model_load_ok": True},
            "artifact": {"ref": "training/run/eval/result.json"},
        },
    )

    assert row["opponent_mixture_enabled"] is True
    assert row["opponent_mixture_entry_name"] == "recent_001"
    assert row["opponent_mixture_entry_weight"] == 1.5
    assert row["opponent_mixture_entry_index"] == 3
    assert row["opponent_mixture_age_label"] == "recent"


def test_checkpoint_gif_summary_records_selected_opponent_mixture_component(
    monkeypatch,
    tmp_path,
):
    mixture = parse_opponent_mixture_spec(
        {
            "entries": [
                {
                    "name": "blank",
                    "weight": 1,
                    "opponent_policy_kind": "fixed_straight",
                    "opponent_runtime_mode": "blank_canvas_noop",
                }
            ]
        }
    )

    checkpoint_path = tmp_path / "iteration_9.pth.tar"
    checkpoint_path.write_bytes(b"fake")
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod.runs_volume, "reload", lambda: None, raising=False)
    monkeypatch.setattr(train_mod.runs_volume, "commit", lambda: None, raising=False)
    monkeypatch.setattr(train_mod, "_wait_for_visible_checkpoint", lambda _ref: checkpoint_path)
    monkeypatch.setattr(eval_mod, "_torch_load", lambda _path: {"model": {}})
    monkeypatch.setattr(eval_mod, "_find_state_dict", lambda _payload: ("model", {}))

    def fake_capture(*, variant, opponent_mixture, **_kwargs):
        assert opponent_mixture["entries"][0]["name"] == "blank"
        return {
            "variant_id": variant["variant_id"],
            "label": variant["label"],
            "policy_mode": variant["policy_mode"],
            "temperature": variant["temperature"],
            "epsilon": variant["epsilon"],
            "compatibility_role": variant["compatibility_role"],
            "ok": True,
            "gif_filename": variant["gif_filename"],
            "gif_ref": f"eval/selfplay/{variant['gif_filename']}",
            "raw_frames_ref": "eval/selfplay/raw_frames.npz",
            "telemetry_ref": "eval/selfplay/env_steps.jsonl",
            "frame_count": 2,
            "physical_steps": 1,
            "scalar_steps": 1,
            "done": True,
            "terminal_reason": "wall",
            "opponent_mixture_enabled": True,
            "opponent_mixture_entry_name": "blank",
            "opponent_mixture_entry_weight": 1.0,
            "opponent_mixture_entry_index": 0,
            "opponent_mixture_age_label": None,
            "stop_reason": "environment_done",
            "max_steps": 8,
            "step_limit_kind": "physical_step_cap",
            "configured_source_max_steps": 8,
            "effective_source_max_steps": 8,
            "frame_stride": 1,
            "fps": 8.0,
            "scale": 1,
            "frame_size": 512,
            "frame_source": "source_state_rgb_canvas_like",
            "frame_schema_id": train_mod.SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
            "frame_truth_level": train_mod.SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
            "browser_pixel_fidelity": False,
            "frame_capture_method": "fake",
            "frame_capture_method_counts": {"fake": 2},
            "frame_capture_details_sample": [],
            "action_summary": {
                "action_collapse_warning": False,
                "action_collapse_players": [],
            },
            "greedy_action_collapse_warning": False,
            "greedy_action_collapse_players": [],
            "joint_action_summary": {},
            "action_trace": {"scalar_actions": [], "joint_actions": []},
            "failure": None,
            "surface": {"env": {"env_id": "fake"}},
            "training_opponent_runtime_mode": "normal",
            "artifacts": {
                "gif": {"ref": f"eval/selfplay/{variant['gif_filename']}"},
                "raw_frames": {"ref": "eval/selfplay/raw_frames.npz"},
            },
        }

    monkeypatch.setattr(train_mod, "_capture_checkpoint_selfplay_gif_variant", fake_capture)

    summary = train_mod._run_checkpoint_selfplay_gif(
        checkpoint_ref="training/run/checkpoints/lightzero/iteration_9.pth.tar",
        checkpoint_label="iteration_9",
        eval_id="live_checkpoint_iteration_9",
        run_id="run",
        attempt_id="attempt",
        seed=123,
        max_steps=8,
        source_max_steps=8,
        num_simulations=1,
        batch_size=1,
        frame_stride=1,
        fps=8.0,
        scale=1,
        frame_size=512,
        training_env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        training_reward_variant=train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        opponent_mixture_spec=mixture,
    )

    assert summary["ok"] is True
    assert summary["opponent_mixture_enabled"] is True
    assert summary["opponent_mixture"]["entries"][0]["name"] == "blank"
    assert summary["opponent_mixture_entry_name"] == "blank"
    assert summary["gif_variants"]["eval_greedy"]["opponent_mixture_entry_name"] == "blank"
    assert summary["capture_env_variant"] == train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
