import ast
import inspect
import importlib.util
import sys
import types
from dataclasses import fields
from pathlib import Path

from curvyzero.contracts.curvytron import (
    CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM,
    CURVYTRON_DEFAULT_MAX_ENV_STEP,
    CURVYTRON_DEFAULT_MAX_TRAIN_ITER,
    CURVYTRON_DEFAULT_N_EPISODE,
    CURVYTRON_DEFAULT_NUM_SIMULATIONS,
    CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE,
    CURVYTRON_SAVE_CKPT_AFTER_ITER,
    CURVYTRON_SOURCE_MAX_STEPS,
    LEARNER_SEAT_MODE_FIXED_PLAYER_1,
    LEARNER_SEAT_MODE_RANDOM_PER_EPISODE,
)
from curvyzero.env.observation_surface_contract import (
    POLICY_OBSERVATION_CONTRACT_ID,
    POLICY_OBSERVATION_PERSPECTIVE,
    POLICY_OBSERVATION_PERSPECTIVE_OWNER,
    POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
    POLICY_OBSERVATION_SEAT_MAPPING,
)
from curvyzero.training import lightzero_config_builder as lz_config
from curvyzero.training.reward_contracts import (
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
)


def _base_main_config():
    return {
        "env": {
            "env_id": "old-env",
        },
        "policy": {
            "discount_factor": 0.997,
            "td_steps": 5,
            "model": {
                "model_type": "conv",
                "observation_shape": [4, 64, 64],
                "action_space_size": 3,
                "support_scale": 10,
                "reward_support_size": 21,
                "value_support_size": 21,
            },
            "learn": {"learner": {"hook": {"save_ckpt_after_iter": 1000}}},
        },
    }


def _install_fake_atari_template(monkeypatch):
    for module_name in ("zoo", "zoo.atari", "zoo.atari.config"):
        package = types.ModuleType(module_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, package)
    template = types.ModuleType("zoo.atari.config.atari_muzero_config")
    template.main_config = _base_main_config()
    monkeypatch.setitem(sys.modules, template.__name__, template)


def test_target_config_patches_update_lightzero_policy_and_model_fields():
    main_config = _base_main_config()

    patches = lz_config.target_config_patches(
        main_config,
        {
            "discount_factor": 1.0,
            "td_steps": 32,
            "model_support_scale": 300,
            "model_reward_support_size": 601,
            "model_value_support_size": 601,
            "model_reward_support_range": (-300.0, 301.0, 1.0),
            "model_value_support_range": (-300.0, 301.0, 1.0),
        },
    )

    assert [patch["path"] for patch in patches] == [
        "policy.discount_factor",
        "policy.td_steps",
        "policy.model.support_scale",
        "policy.model.reward_support_size",
        "policy.model.value_support_size",
        "policy.model.reward_support_range",
        "policy.model.value_support_range",
    ]
    assert all(
        patch["reason"] == "make LightZero value/reward target range match CurvyTron reward variant"
        for patch in patches
    )
    assert main_config["policy"]["discount_factor"] == 1.0
    assert main_config["policy"]["td_steps"] == 32
    assert main_config["policy"]["model"]["support_scale"] == 300
    assert main_config["policy"]["model"]["reward_support_range"] == (-300.0, 301.0, 1.0)


def test_target_config_patches_ignore_unknown_and_omitted_fields():
    main_config = _base_main_config()

    patches = lz_config.target_config_patches(
        main_config,
        {
            "td_steps": 9,
            "unknown_key": "ignored",
            "model_reward_support_range": (-7.0, 8.0, 1.0),
        },
    )

    assert [patch["path"] for patch in patches] == [
        "policy.td_steps",
        "policy.model.reward_support_range",
    ]
    assert patches[1]["new"] == [-7.0, 8.0, 1.0]
    assert main_config["policy"]["td_steps"] == 9
    assert main_config["policy"]["discount_factor"] == 0.997
    assert main_config["policy"]["model"]["value_support_size"] == 21
    assert "unknown_key" not in main_config["policy"]


def test_set_or_add_path_creates_missing_nodes_and_reports_plain_values():
    mapping = {"policy": {"model": None}}

    patch = lz_config.set_or_add_path(mapping, ("policy", "model", "shape"), (4, 64, 64))

    assert mapping["policy"]["model"]["shape"] == (4, 64, 64)
    assert patch == {
        "path": "policy.model.shape",
        "old": "<missing>",
        "new": [4, 64, 64],
    }


def test_to_plain_normalizes_json_boundary_values():
    class ArrayLike:
        def tolist(self):
            return (1, 2, {"x": ScalarLike()})

    class ScalarLike:
        def item(self):
            return 3

    assert lz_config.to_plain({1: ArrayLike(), "tuple": (4, 5)}) == {
        "1": [1, 2, {"x": 3}],
        "tuple": [4, 5],
    }


def test_checkpoint_hook_patch_helpers_report_old_and_new_values():
    main_config = _base_main_config()

    save_patch = lz_config.set_save_ckpt_after_iter(main_config, 50)
    load_patch = lz_config.set_load_ckpt_before_run(
        main_config,
        "/tmp/checkpoint.pth.tar",
        reason="resume test",
    )

    hook = main_config["policy"]["learn"]["learner"]["hook"]
    assert hook["save_ckpt_after_iter"] == 50
    assert hook["load_ckpt_before_run"] == "/tmp/checkpoint.pth.tar"
    assert save_patch == {
        "path": "policy.learn.learner.hook.save_ckpt_after_iter",
        "old": 1000,
        "new": 50,
        "reason": "first CurvyTron visual survival run checkpoints frequently for inspection",
    }
    assert load_patch == {
        "path": "policy.learn.learner.hook.load_ckpt_before_run",
        "old": None,
        "new": "/tmp/checkpoint.pth.tar",
        "reason": "resume test",
    }


def test_checkpoint_hook_patch_helpers_create_missing_nested_hook():
    main_config = {"policy": {}}

    save_patch = lz_config.set_save_ckpt_after_iter(main_config, 12)
    load_patch = lz_config.set_load_ckpt_before_run(main_config, "/tmp/ckpt.pt")

    assert main_config["policy"]["learn"]["learner"]["hook"] == {
        "save_ckpt_after_iter": 12,
        "load_ckpt_before_run": "/tmp/ckpt.pt",
    }
    assert save_patch["old"] is None
    assert load_patch["old"] is None


def test_extract_visual_survival_surface_records_public_config_contract():
    main_config = _base_main_config()
    main_config["env"].update(
        {
            "type": "curvyzero_source_state",
            "env_id": "source-state-env",
            "env_variant": "source_state_fixed_opponent",
            "action_space_size": 3,
            "collector_env_num": 256,
            "evaluator_env_num": 1,
            "n_evaluator_episode": 1,
            "source_max_steps": 1024,
            "decision_ms": 16.6666666667,
            "decision_source_frames": 1,
            "source_physics_step_ms": 16.6666666667,
            "source_max_steps_semantics": "source_physics_steps",
            "dynamic_seed": True,
            "reset_seed_strategy": "dynamic_seed_sequence_from_run_seed_and_reset_index/v0",
            "telemetry_path": "/tmp/env_steps.jsonl",
            "telemetry_stride": 1,
            "profile_env_timing_enabled": False,
            "reward_variant": "survival_plus_bonus_no_outcome",
            "reward_schema_id": "curvyzero_survival_plus_bonus_no_outcome/v0",
            "reward_policy": {"same_step_bonus_pickup_reward": True},
            "lightzero_target_config": {"discount_factor": 1.0},
            "source_state_trail_render_mode": "browser_lines",
            "source_state_bonus_render_mode": "browser_sprites",
            "policy_observation_backend": "cpu",
            "policy_trail_render_mode": "browser_lines",
            "policy_bonus_render_mode": "browser_sprites",
            "policy_observation_contract_id": "curvyzero_policy_observation_surface/v0",
            "observation_contract": {"backend": "cpu"},
            "learner_seat_mode": "random_per_episode",
            "default_trail_render_mode": "browser_lines",
            "supported_trail_render_modes": ["browser_lines"],
            "default_bonus_render_mode": "browser_sprites",
            "supported_bonus_render_modes": ["browser_sprites"],
            "default_policy_observation_backend": "cpu",
            "supported_policy_observation_backends": ["cpu", "gpu"],
            "observation_schema_id": "stacked_gray64",
            "debug_fidelity_only": False,
            "source_fidelity_claim": "source_state_backed_non_browser_pixel",
            "single_product_runtime_path": True,
            "legacy_debug_variant": False,
            "underlying_env_class": "source_state_visual_survival_wrapper",
            "runtime_env_impl_id": "natural_bonus",
            "runtime_topology": "fixed_opponent",
            "two_seat_self_play": False,
            "current_policy_two_seat_action_collection": False,
            "two_seat_self_play_status": "not_two_seat_self_play",
            "fixed_opponent_is_two_seat_self_play": False,
            "browser_pixel_fidelity": False,
            "uses_ale": False,
            "visual_surface": "gray64",
            "visual_truth_level": "source_state",
            "visual_source_state_backed": True,
            "ego_action_straight_override_probability": 0.0,
            "policy_action_repeat_min": 1,
            "policy_action_repeat_max": 1,
            "policy_action_repeat_extra_probability": 0.0,
            "policy_action_repeat_semantics": (
                "repeat_selected_policy_action_inside_one_lightzero_env_step"
            ),
            "control_noise_profile_id": "none",
            "disable_death_for_profile": False,
            "opponent_death_mode": "normal",
            "opponent_death_mode_diagnostic": False,
            "opponent_death_mode_claim": "none",
            "opponent_runtime_mode": "normal",
            "opponent_runtime_mode_claim": "normal_opponent_runtime",
            "opponent_visibility_mode": "visible_if_present_alive",
            "opponent_collision_effect": "normal",
            "opponent_trail_mode": "normal",
            "natural_bonus_spawn": True,
            "death_mode": "normal",
            "turn_commit_adapter": False,
            "opponent_policy_kind": "fixed_straight",
            "opponent_training_relation": "learner_vs_fixed_straight",
            "current_policy_self_play": False,
            "current_policy_self_play_blocker": "fixed_opponent",
            "current_policy_self_play_caveat": None,
            "trusted_current_policy_self_play": False,
            "simultaneous_game_theory_claim": False,
            "opponent_mixture": {"entries": []},
            "opponent_assignment_context": {
                "assignment_id": "assignment-a",
                "opponent_split_env_index": 0,
            },
        }
    )
    main_config["policy"].update(
        {
            "collector_env_num": 256,
            "evaluator_env_num": 1,
            "n_episode": 256,
            "num_simulations": 8,
            "batch_size": 64,
            "cuda": False,
        }
    )
    main_config["policy"]["model"].update(
        {
            "image_channel": 4,
            "frame_stack_num": 1,
            "self_supervised_learning_loss": True,
            "reward_support_range": (-300.0, 301.0, 1.0),
            "value_support_range": (-300.0, 301.0, 1.0),
        }
    )
    create_config = {
        "env": {"type": "curvyzero_source_state", "import_names": ["curvyzero.training.env"]},
        "env_manager": {"type": "subprocess"},
    }

    surface = lz_config.extract_visual_survival_surface(
        main_config,
        create_config,
        max_env_step=8192,
        max_train_iter=64,
    )

    assert surface["env_type"] == "curvyzero_source_state"
    assert surface["collector_env_num"] == 256
    assert surface["n_episode"] == 256
    assert surface["batch_size"] == 64
    assert surface["learner_seat_mode"] == "random_per_episode"
    assert surface["default_policy_observation_backend"] == "cpu"
    assert surface["supported_policy_observation_backends"] == ["cpu", "gpu"]
    assert surface["natural_bonus_spawn"] is True
    assert surface["opponent_assignment_context"] == {
        "assignment_id": "assignment-a",
        "opponent_split_env_index": 0,
    }
    assert surface["reward_policy"] == {"same_step_bonus_pickup_reward": True}
    assert surface["opponent_mixture"] == {"entries": []}
    assert surface["save_ckpt_after_iter"] == 1000


def test_eval_module_does_not_import_trainer_private_target_patch_helper():
    spec = importlib.util.find_spec(
        "curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval"
    )
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))

    private_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
        and any(alias.name == "_target_config_patches" for alias in node.names)
    ]

    assert private_imports == []


def test_eval_module_does_not_import_trainer_private_visual_survival_builder():
    spec = importlib.util.find_spec(
        "curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval"
    )
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))

    private_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
        and any(alias.name == "_build_visual_survival_configs" for alias in node.names)
    ]

    assert private_imports == []


def test_lightzero_config_builder_does_not_import_modal_modules():
    tree = ast.parse(Path(lz_config.__file__).read_text(encoding="utf-8"))

    modal_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("curvyzero.infra.modal")
    ]

    assert modal_imports == []


def _source_state_builder_kwargs(tmp_path):
    return {
        "seed": 3,
        "exp_name": tmp_path / "exp",
        "telemetry_path": tmp_path / "env_steps.jsonl",
        "cuda": False,
        "max_env_step": 1024,
        "source_max_steps": 128,
        "decision_ms": lz_config.DEFAULT_DECISION_MS,
        "collector_env_num": 256,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "n_episode": 256,
        "num_simulations": 8,
        "batch_size": 32,
        "lightzero_eval_freq": 0,
        "lightzero_multi_gpu": False,
        "max_train_iter": 7,
        "save_ckpt_after_iter": 2,
        "env_variant": lz_config.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
        "reward_outcome_alpha": 0.25,
        "ego_action_straight_override_probability": 0.0,
        "control_noise_profile_id": "none",
        "disable_death_for_profile": False,
        "env_telemetry_stride": 1,
        "env_manager_type": "base",
        "opponent_policy_kind": lz_config.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "opponent_use_cuda": False,
        "opponent_checkpoint": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_state_key": None,
        "learner_seat_mode": LEARNER_SEAT_MODE_FIXED_PLAYER_1,
        "natural_bonus_spawn": False,
        "model_support_cap": 64,
        "td_steps": 5,
    }


def _spec_from_builder_kwargs(kwargs):
    return lz_config.VisualSurvivalConfigSpec.from_builder_kwargs(kwargs)


def test_public_visual_survival_builder_builds_source_state_contract(monkeypatch, tmp_path):
    _install_fake_atari_template(monkeypatch)

    built = lz_config.build_visual_survival_configs(**_source_state_builder_kwargs(tmp_path))

    env_cfg = built["main_config"]["env"]
    surface = built["surface"]
    assert built["create_config"]["env"]["type"] == env_cfg["type"]
    assert built["main_config"]["policy"]["collector_env_num"] == 256
    assert built["main_config"]["policy"]["batch_size"] == 32
    assert built["main_config"]["policy"]["n_episode"] == 256
    assert env_cfg["collector_env_num"] == 256
    assert env_cfg["learner_seat_mode"] == LEARNER_SEAT_MODE_FIXED_PLAYER_1
    assert env_cfg["natural_bonus_spawn"] is False
    assert env_cfg["reward_policy"]["reward_outcome_alpha"] == 0.25
    assert env_cfg["lightzero_target_config"]["td_steps"] == 5
    assert env_cfg["lightzero_target_config"]["model_reward_support_size"] == 129
    assert surface["learner_seat_mode"] == LEARNER_SEAT_MODE_FIXED_PLAYER_1
    assert surface["batch_size"] == 32
    assert surface["collector_env_num"] == 256
    assert surface["natural_bonus_spawn"] is False
    assert [patch["path"] for patch in built["patches"]][-1] == "env"


def test_public_visual_survival_builder_can_add_rnd_meter_bundle(monkeypatch, tmp_path):
    _install_fake_atari_template(monkeypatch)
    kwargs = _source_state_builder_kwargs(tmp_path)
    kwargs["exploration_bonus"] = {"mode": "rnd_meter_v0", "weight": 0.0}

    built = lz_config.build_visual_survival_configs(**kwargs)

    main_config = built["main_config"]
    assert main_config["policy"]["use_rnd_model"] is True
    assert main_config["policy"]["target_model_for_intrinsic_reward_update_type"] == "assign"
    assert main_config["reward_model"]["type"] == "rnd_muzero"
    assert main_config["reward_model"]["seed"] == 3
    assert main_config["reward_model"]["intrinsic_reward_weight"] == 0.0
    assert main_config["reward_model"]["obs_shape"] == [1, 64, 64]
    assert main_config["env"]["exploration_bonus"]["mode"] == "rnd_meter_v0"
    assert built["surface"]["trainer_entrypoint"] == "lzero.entry.train_muzero_with_reward_model"
    assert built["surface"]["policy_use_rnd_model"] is True
    assert built["surface"]["reward_model_type"] == "rnd_muzero"
    assert built["surface"]["exploration_bonus"]["target_reward_effect"] == "unchanged"
    assert [patch["path"] for patch in built["patches"]][-7:] == [
        "reward_model",
        "policy.use_rnd_model",
        "policy.use_momentum_representation_network",
        "policy.target_model_for_intrinsic_reward_update_type",
        "policy.target_update_freq_for_intrinsic_reward",
        "policy.target_update_theta_for_intrinsic_reward",
        "env.exploration_bonus",
    ]


def test_public_visual_survival_builder_can_add_positive_rnd_bundle(monkeypatch, tmp_path):
    _install_fake_atari_template(monkeypatch)
    kwargs = _source_state_builder_kwargs(tmp_path)
    kwargs["exploration_bonus"] = {
        "mode": "rnd_replay_target_v0",
        "weight": 0.1,
        "rnd_batch_size": 8,
    }

    built = lz_config.build_visual_survival_configs(**kwargs)

    main_config = built["main_config"]
    assert main_config["reward_model"]["type"] == "rnd_muzero"
    assert main_config["reward_model"]["seed"] == 3
    assert main_config["reward_model"]["intrinsic_reward_weight"] == 0.1
    assert main_config["reward_model"]["batch_size"] == 8
    assert main_config["env"]["exploration_bonus"]["mode"] == "rnd_replay_target_v0"
    assert main_config["env"]["exploration_bonus"]["training_effect"] == (
        "reward_target_augmented_by_intrinsic_rnd"
    )
    target_config = main_config["env"]["lightzero_target_config"]
    assert target_config["exploration_bonus_support_adjusted"] is True
    assert target_config["exploration_bonus_intrinsic_reward_bound"] == 0.1
    assert target_config["exploration_bonus_intrinsic_reward_support_extra"] == 1
    assert target_config["exploration_bonus_intrinsic_value_support_extra"] == 13
    assert target_config["model_reward_support_requested_scale"] == 35
    assert target_config["model_value_support_requested_scale"] == 301
    assert target_config["model_reward_support_size"] == 129
    assert built["surface"]["trainer_entrypoint"] == "lzero.entry.train_muzero_with_reward_model"
    assert built["surface"]["exploration_bonus"]["target_reward_effect"] == (
        "intrinsic_weighted_addition"
    )


def test_visual_survival_surface_pins_policy_observation_perspective_contract(
    monkeypatch,
    tmp_path,
):
    _install_fake_atari_template(monkeypatch)

    built = lz_config.build_visual_survival_configs(**_source_state_builder_kwargs(tmp_path))
    surface = built["surface"]
    contract = surface["observation_contract"]

    assert surface["policy_observation_contract_id"] == POLICY_OBSERVATION_CONTRACT_ID
    assert contract["contract_id"] == POLICY_OBSERVATION_CONTRACT_ID
    assert contract["perspective_schema_id"] == POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
    assert contract["perspective"] == POLICY_OBSERVATION_PERSPECTIVE
    assert contract["perspective_owner"] == POLICY_OBSERVATION_PERSPECTIVE_OWNER
    assert contract["seat_mapping"] == POLICY_OBSERVATION_SEAT_MAPPING
    assert contract["backend"] == surface["policy_observation_backend"]
    assert contract["trail_render_mode"] == surface["source_state_trail_render_mode"]
    assert contract["bonus_render_mode"] == surface["source_state_bonus_render_mode"]
    assert contract["raw_dtype"] == "uint8"
    assert contract["model_dtype"] == "float32"
    assert contract["raw_value_range"] == [0, 255]
    assert contract["model_value_range"] == [0.0, 1.0]


def test_public_env_variant_spec_matches_trainer_facade():
    from curvyzero.infra.modal import lightzero_curvyzero_stacked_debug_visual_survival_train as train_mod

    for env_variant in lz_config.ENV_VARIANT_CHOICES:
        assert train_mod._env_variant_spec(env_variant) == lz_config.env_variant_spec(env_variant)


def test_trainer_visual_survival_facade_matches_public_builder_for_grouped_spec_cases(
    monkeypatch,
    tmp_path,
):
    _install_fake_atari_template(monkeypatch)
    from curvyzero.infra.modal import lightzero_curvyzero_stacked_debug_visual_survival_train as train_mod

    base = _source_state_builder_kwargs(tmp_path)
    cases = [
        base,
        {
            **base,
            "opponent_mixture": {
                "selection": "per_episode_weighted",
                "entries": [{"name": "blank", "weight": 1.0}],
            },
            "opponent_assignment_context": {
                "assignment_id": "assignment-typed",
                "opponent_split_env_index": 7,
            },
        },
        {
            **base,
            "opponent_policy_kind": lz_config.OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
            "opponent_use_cuda": True,
            "opponent_checkpoint": {
                "resolved_checkpoint_path": "/tmp/iteration_7.pth.tar",
                "checkpoint_ref": "training/run/checkpoints/lightzero/iteration_7.pth.tar",
            },
            "opponent_snapshot_ref": "snapshot-7",
            "opponent_checkpoint_state_key": "model",
        },
        {
            **base,
            "learner_seat_mode": LEARNER_SEAT_MODE_FIXED_PLAYER_1,
            "policy_observation_backend": lz_config.DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
        },
    ]

    for kwargs in cases:
        assert train_mod._build_visual_survival_configs(**kwargs) == (
            lz_config.build_visual_survival_configs(**kwargs)
        )


def test_visual_survival_config_spec_matches_keyword_builder(monkeypatch, tmp_path):
    _install_fake_atari_template(monkeypatch)
    kwargs = _source_state_builder_kwargs(tmp_path)
    kwargs.update(
        {
            "opponent_mixture": {
                "selection": "per_episode_weighted",
                "entries": [{"name": "blank", "weight": 1.0}],
            },
            "opponent_assignment_context": {
                "assignment_id": "assignment-typed",
                "opponent_split_env_index": 17,
            },
            "source_state_trail_render_mode": lz_config.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
            "source_state_bonus_render_mode": lz_config.DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
            "policy_observation_backend": lz_config.DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
        }
    )

    typed = lz_config.build_visual_survival_configs_from_spec(
        _spec_from_builder_kwargs(kwargs)
    )
    direct = lz_config.build_visual_survival_configs(**kwargs)

    assert isinstance(typed, lz_config.VisualSurvivalConfigResult)
    assert typed.as_dict() == direct
    assert typed.env_config["opponent_mixture"] == kwargs["opponent_mixture"]
    assert typed.env_config["opponent_assignment_context"] == kwargs[
        "opponent_assignment_context"
    ]
    assert typed.lightzero_target_config == direct["main_config"]["env"][
        "lightzero_target_config"
    ]
    assert typed.surface["batch_size"] == 32
    assert typed.surface["collector_env_num"] == 256


def test_visual_survival_config_spec_matches_keyword_builder_signature(tmp_path):
    spec = _spec_from_builder_kwargs(_source_state_builder_kwargs(tmp_path))

    assert list(inspect.signature(lz_config.build_visual_survival_configs).parameters) == list(
        spec.to_builder_kwargs()
    )


def test_visual_survival_config_spec_groups_normalized_knobs(tmp_path):
    spec = _spec_from_builder_kwargs(_source_state_builder_kwargs(tmp_path))

    assert spec.run.env_manager_type == "base"
    assert spec.training.collector_env_num == 256
    assert spec.training.batch_size == 32
    assert spec.timing.decision_ms == lz_config.DEFAULT_DECISION_MS
    assert spec.observation.learner_seat_mode == LEARNER_SEAT_MODE_FIXED_PLAYER_1
    assert spec.behavior.natural_bonus_spawn is False
    assert spec.reward.td_steps == 5
    assert spec.opponent.opponent_policy_kind == lz_config.OPPONENT_POLICY_KIND_FIXED_STRAIGHT


def test_visual_survival_config_spec_rejects_unknown_broad_kwargs(tmp_path):
    kwargs = _source_state_builder_kwargs(tmp_path)
    kwargs["stale_knob"] = True

    try:
        lz_config.VisualSurvivalConfigSpec.from_builder_kwargs(kwargs)
    except TypeError as exc:
        assert "unknown visual survival builder kwargs: stale_knob" in str(exc)
    else:
        raise AssertionError("expected unknown broad builder kwarg to fail")


def test_visual_survival_experiment_spec_is_compact_public_contract():
    public_fields = {field.name for field in fields(lz_config.VisualSurvivalExperimentSpec)}

    assert public_fields == {
        "seed",
        "exp_name",
        "telemetry_path",
        "reward_variant",
        "reward_outcome_alpha",
        "opponent_policy_kind",
        "frozen_opponent",
        "opponent_mixture",
        "opponent_assignment_context",
        "action_noise_probability",
        "scale_preset",
    }
    assert public_fields.isdisjoint(
        {
            "cuda",
            "lightzero_multi_gpu",
            "env_manager_type",
            "env_variant",
            "policy_observation_backend",
            "learner_seat_mode",
            "source_state_trail_render_mode",
            "source_state_bonus_render_mode",
            "opponent_death_mode",
            "opponent_runtime_mode",
            "model_support_cap",
            "td_steps",
            "collector_env_num",
            "batch_size",
            "decision_ms",
        }
    )


def test_visual_survival_experiment_spec_current_broad_expands_to_grouped_defaults(
    tmp_path,
):
    experiment = lz_config.VisualSurvivalExperimentSpec(
        seed=11,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        reward_outcome_alpha=0.5,
        action_noise_probability=0.10,
    )

    spec = lz_config.visual_survival_config_spec_from_experiment(experiment)

    assert spec.run.cuda is True
    assert spec.run.env_manager_type == "subprocess"
    assert spec.training.max_env_step == CURVYTRON_DEFAULT_MAX_ENV_STEP
    assert spec.training.max_train_iter == CURVYTRON_DEFAULT_MAX_TRAIN_ITER
    assert spec.training.source_max_steps == CURVYTRON_SOURCE_MAX_STEPS
    assert spec.training.collector_env_num == CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM
    assert spec.training.n_episode == CURVYTRON_DEFAULT_N_EPISODE
    assert spec.training.num_simulations == CURVYTRON_DEFAULT_NUM_SIMULATIONS
    assert spec.training.batch_size == CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE
    assert spec.training.save_ckpt_after_iter == CURVYTRON_SAVE_CKPT_AFTER_ITER
    assert spec.observation.learner_seat_mode == LEARNER_SEAT_MODE_RANDOM_PER_EPISODE
    assert spec.behavior.ego_action_straight_override_probability == 0.10
    assert spec.behavior.policy_action_repeat_max == 2
    assert spec.behavior.policy_action_repeat_extra_probability == 0.10
    assert spec.behavior.control_noise_profile_id == "straight_override_p10_repeat_p10"
    assert spec.reward.reward_outcome_alpha == 0.5
    assert spec.opponent.opponent_policy_kind == lz_config.OPPONENT_POLICY_KIND_FIXED_STRAIGHT


def test_visual_survival_experiment_spec_builds_current_broad_config(monkeypatch, tmp_path):
    _install_fake_atari_template(monkeypatch)
    experiment = lz_config.VisualSurvivalExperimentSpec(
        seed=11,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        reward_outcome_alpha=0.5,
        action_noise_probability=0.10,
    )

    built = lz_config.build_visual_survival_config_from_experiment(experiment)

    assert built.surface["collector_env_num"] == CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM
    assert built.surface["n_episode"] == CURVYTRON_DEFAULT_N_EPISODE
    assert built.surface["batch_size"] == CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE
    assert built.surface["learner_seat_mode"] == LEARNER_SEAT_MODE_RANDOM_PER_EPISODE
    assert built.surface["reward_variant"] == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME
    assert built.surface["reward_policy"]["reward_outcome_alpha"] == 0.5
    assert built.env_config["control_noise_profile_id"] == "straight_override_p10_repeat_p10"
    assert built.env_config["policy_action_repeat_max"] == 2
    assert built.env_config["policy_action_repeat_extra_probability"] == 0.10


def test_visual_survival_experiment_spec_rejects_internal_launch_knobs(tmp_path):
    forbidden_kwargs = {
        "lightzero_eval_freq": 1,
        "collector_env_num": 64,
        "batch_size": 128,
        "source_max_steps": 1024,
        "decision_ms": 33.3,
        "env_variant": "source_state_fixed_opponent",
        "env_manager_type": "base",
        "policy_observation_backend": "cpu",
        "model_support_cap": 2048,
        "td_steps": 50,
    }

    for key, value in forbidden_kwargs.items():
        try:
            lz_config.VisualSurvivalExperimentSpec(
                seed=11,
                exp_name=tmp_path / "exp",
                telemetry_path=tmp_path / "env_steps.jsonl",
                **{key: value},
            )
        except TypeError as exc:
            assert key in str(exc)
        else:
            raise AssertionError(f"expected internal experiment kwarg {key!r} to fail")


def test_visual_survival_experiment_spec_rejects_unknown_scale_preset(tmp_path):
    experiment = lz_config.VisualSurvivalExperimentSpec(
        seed=11,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        scale_preset="stale_scale",
    )

    try:
        lz_config.visual_survival_config_spec_from_experiment(experiment)
    except ValueError as exc:
        assert "unknown visual survival experiment scale preset" in str(exc)
    else:
        raise AssertionError("expected unknown experiment scale preset to fail")


def test_visual_survival_config_spec_maps_frozen_opponent_fields(tmp_path):
    kwargs = _source_state_builder_kwargs(tmp_path)
    checkpoint = {
        "resolved_checkpoint_path": "/tmp/iteration_7.pth.tar",
        "checkpoint_ref": "training/run/checkpoints/lightzero/iteration_7.pth.tar",
    }
    kwargs.update(
        {
            "opponent_policy_kind": lz_config.OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
            "opponent_use_cuda": True,
            "opponent_checkpoint": checkpoint,
            "opponent_snapshot_ref": "snapshot-7",
            "opponent_checkpoint_state_key": "model",
        }
    )

    spec = _spec_from_builder_kwargs(kwargs)
    mapped = spec.to_builder_kwargs()

    assert spec.opponent.frozen_opponent == lz_config.FrozenOpponentConfig(
        checkpoint=checkpoint,
        snapshot_ref="snapshot-7",
        checkpoint_state_key="model",
        use_cuda=True,
    )
    assert mapped["opponent_checkpoint"] == checkpoint
    assert mapped["opponent_snapshot_ref"] == "snapshot-7"
    assert mapped["opponent_checkpoint_state_key"] == "model"
    assert mapped["opponent_use_cuda"] is True


def test_visual_survival_config_spec_builds_frozen_opponent_surface(monkeypatch, tmp_path):
    _install_fake_atari_template(monkeypatch)
    kwargs = _source_state_builder_kwargs(tmp_path)
    kwargs.update(
        {
            "opponent_policy_kind": lz_config.OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
            "opponent_use_cuda": True,
            "opponent_checkpoint": {
                "resolved_checkpoint_path": "/tmp/iteration_7.pth.tar",
                "checkpoint_ref": "training/run/checkpoints/lightzero/iteration_7.pth.tar",
            },
            "opponent_snapshot_ref": "snapshot-7",
            "opponent_checkpoint_state_key": "model",
        }
    )

    built = lz_config.build_visual_survival_config(_spec_from_builder_kwargs(kwargs))

    assert built.env_config["opponent_checkpoint_path"] == "/tmp/iteration_7.pth.tar"
    assert built.env_config["opponent_checkpoint_ref"] == (
        "training/run/checkpoints/lightzero/iteration_7.pth.tar"
    )
    assert built.env_config["opponent_snapshot_ref"] == "snapshot-7"
    assert built.env_config["opponent_checkpoint_state_key"] == "model"
    assert built.env_config["opponent_use_cuda"] is True
    assert built.surface["opponent_checkpoint_ref"] == (
        "training/run/checkpoints/lightzero/iteration_7.pth.tar"
    )
    assert built.surface["opponent_snapshot_ref"] == "snapshot-7"
    assert built.surface["opponent_checkpoint_state_key"] == "model"
    assert built.surface["opponent_use_cuda"] is True
