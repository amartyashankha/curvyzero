import argparse
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / ("build_curvytron_hybrid_observation_profile_grid.py")
)
SPEC = importlib.util.spec_from_file_location(
    "curvytron_hybrid_observation_profile_grid_builder",
    SCRIPT_PATH,
)
assert SPEC is not None
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


def _args(**overrides):
    defaults = {
        "experiment_id": "unit-hybrid-profile-grid",
        "computes": ["gpu-h100", "gpu-l4-t4"],
        "batch_sizes": [512],
        "actor_count": 16,
        "steps": 100,
        "warmup_steps": 20,
        "max_ticks": 2000,
        "death_mode": "profile_no_death",
        "trail_slots": 1024,
        "body_capacity": 1024,
        "probe_simulations": [8],
        "probe_channels": 16,
        "materialize_scalar_timestep": [False, True],
        "device_latest": [False],
        "resident_chunk_probe": False,
        "resident_replay_steps": 64,
        "resident_sample_batch_size": 256,
        "resident_replay_train_steps": 1,
        "no_resident_readback_checksum": False,
        "compact_service_replay_proof": False,
        "compact_rollout_slab_probe": False,
        "compact_rollout_slab_sample_gate": False,
        "compact_rollout_slab_sample_gate_batch_size": 0,
        "compact_rollout_slab_sample_gate_interval": 1,
        "compact_rollout_slab_sample_gate_replay_pair_capacity": 4096,
        "compact_rollout_slab_learner_gate": False,
        "compact_rollout_slab_learner_gate_train_steps": 1,
        "compact_rollout_slab_learner_gate_device": "cuda",
        "compact_rollout_slab_learner_gate_include_rnd": False,
        "compact_rollout_slab_learner_gate_impl": "toy_probe",
        "compact_rollout_slab_learner_gate_support_scale": 1,
        "compact_rollout_slab_learner_gate_num_unroll_steps": 1,
        "compact_rollout_slab_action_mode": "search_feedback",
        "compact_owned_loop_entrypoint": False,
        "compact_owned_loop_policy_version_ref": "",
        "compact_owned_loop_model_version_ref": "",
        "compact_owned_loop_policy_source": "",
        "compact_owned_loop_capture_replay_store_state": False,
        "compact_root_tape_compare": False,
        "compact_root_tape_max_records": 4,
        "compact_root_tape_allow_resident_host_snapshot": False,
        "compact_root_tape_compare_fixed_shape_floor": True,
        "compact_root_tape_compare_mctx": False,
        "compact_root_tape_compare_model_compile": False,
        "compact_root_tape_compare_direct_core": False,
        "compact_root_tape_model_compile_mode": "default",
        "compact_root_tape_require_model_compile": True,
        "compact_root_tape_reference_label": "primary",
        "hybrid_device_only_stack": False,
        "hybrid_refresh_observation_stack": True,
        "hybrid_resident_observation_search": False,
        "hybrid_native_actor_buffer": False,
        "hybrid_persistent_compact_render_state_buffer": False,
        "hybrid_borrow_single_actor_render_state": False,
        "lightzero_collect_forward_probe": False,
        "lightzero_initial_inference_probe": False,
        "lightzero_array_ceiling_probe": False,
        "lightzero_mcts_arrays_boundary_probe": False,
        "mctx_compact_search_probe": False,
        "mctx_hidden_dim": 64,
        "mctx_visual_channels": 8,
        "mctx_require_gpu_backend": True,
        "lightzero_mcts_arrays_boundary_impl": "stock_facade",
        "lightzero_mcts_arrays_boundary_impls": None,
        "lightzero_mcts_arrays_boundary_input_mode": "host_uint8",
        "lightzero_array_ceiling_mode": "policy_arrays",
        "lightzero_array_ceiling_input_mode": "host_uint8",
        "lightzero_mock_service_materialize_public_output": False,
        "lightzero_consumer_temperature": 1.0,
        "lightzero_consumer_epsilon": 0.0,
        "lightzero_consumer_root_noise_weight": -1.0,
        "lightzero_consumer_use_cuda": True,
        "lightzero_consumer_collect_with_pure_policy": False,
        "compact_torch_compile_search": True,
        "compact_torch_compile_model_inference": False,
        "compact_torch_require_model_compile": False,
        "compact_torch_model_compile_mode": "reduce-overhead",
        "compact_torch_recurrent_action_shape_mode": "auto",
        "compact_torch_initial_inference_mode": "model_method",
        "compact_torch_observation_memory_format": "contiguous",
        "compact_torch_model_memory_format": "contiguous",
        "launch_mode": builder.LAUNCH_MODE_BLOCKING_STDOUT_JSON,
        "result_capture": builder.RESULT_CAPTURE_STDOUT_JSON,
        "row_timeout_sec": None,
        "result_timeout_sec": None,
        "captured_result_required": False,
        "require_terminal_compact_owned_nstep": False,
        "require_normal_death_terminal_contract": False,
        "normal_death_terminal_contract_evidence_id": "",
        "normal_death_terminal_contract_evidence_refs": [],
        "stack_storage_dtype": "uint8",
        "render_surface": "direct_gray64",
        "observation_renderer_backend": "jax_gpu_persistent_policy_framebuffer_profile",
        "quiet": False,
        "next_direct_ctree_comparison_preset": False,
        "next_fixed_root_tape_preset": False,
        "next_fixed_root_tape_large_preset": False,
        "next_fixed_root_tape_mctx_preset": False,
        "next_fixed_root_tape_compile_preset": False,
        "next_fixed_root_tape_direct_core_preset": False,
        "next_normal_death_compact_owned_preset": False,
        "next_borrowed_normal_death_compact_owned_preset": False,
        "next_terminal_nstep_compact_owned_preset": False,
        "include_l4": True,
        "stdout_only": True,
        "output_root": Path("unused"),
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_hybrid_profile_grid_uses_boundary_compute_names():
    manifest = builder.build_manifest(_args())

    assert [row["compute"] for row in manifest["rows"]] == [
        "gpu-h100",
        "gpu-h100",
        "gpu-l4-t4",
        "gpu-l4-t4",
    ]
    assert all(row["death_mode"] == "profile_no_death" for row in manifest["rows"])
    assert all("--death-mode" in row["command"] for row in manifest["rows"])
    assert all("gpu-h100-cpu40" not in row["command"] for row in manifest["rows"])
    assert all("--hybrid-observation-canary" in row["command"] for row in manifest["rows"])
    assert manifest["calls_train_muzero"] is False
    assert manifest["touches_live_runs"] is False


def test_hybrid_profile_grid_propagates_normal_death_mode():
    manifest = builder.build_manifest(_args(death_mode="normal"))

    row = manifest["rows"][0]
    assert row["death_mode"] == "normal"
    assert row["fixed_denominator"]["death_mode"] == "normal"
    flag_index = row["command"].index("--death-mode")
    assert row["command"][flag_index + 1] == "normal"


def test_hybrid_profile_grid_can_emit_resident_probe_rows_with_scalar_edge():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False, True],
            device_latest=[False, True],
            resident_chunk_probe=True,
        )
    )

    assert len(manifest["rows"]) == 2
    assert [row["materialize_scalar_timestep"] for row in manifest["rows"]] == [
        False,
        True,
    ]
    assert all(row["resident_chunk_probe"] is True for row in manifest["rows"])
    assert all(row["device_latest"] is False for row in manifest["rows"])
    assert "--no-hybrid-materialize-scalar-timestep" in manifest["rows"][0]["command"]
    assert "--hybrid-materialize-scalar-timestep" in manifest["rows"][1]["command"]
    assert all(
        "--hybrid-batched-stack-probe-device-latest" not in row["command"]
        for row in manifest["rows"]
    )


def test_hybrid_profile_grid_can_emit_lightzero_collect_forward_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_collect_forward_probe=True,
            probe_simulations=[8],
        )
    )

    assert len(manifest["rows"]) == 1
    row = manifest["rows"][0]
    assert row["lightzero_collect_forward_probe"] is True
    assert row["lightzero_initial_inference_probe"] is False
    assert "--hybrid-lightzero-collect-forward-probe" in row["command"]
    assert "--hybrid-lightzero-consumer-num-simulations" in row["command"]
    assert "--hybrid-batched-stack-probe-simulations" in row["command"]
    probe_flag_index = row["command"].index("--hybrid-batched-stack-probe-simulations")
    assert row["command"][probe_flag_index + 1] == "0"
    assert "-lzcf" in row["label"]


def test_hybrid_profile_grid_can_emit_lightzero_pure_policy_collect_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_collect_forward_probe=True,
            lightzero_consumer_collect_with_pure_policy=True,
            probe_simulations=[8],
        )
    )

    row = manifest["rows"][0]
    assert "--hybrid-lightzero-consumer-collect-with-pure-policy" in row["command"]


def test_hybrid_profile_grid_forwards_compact_torch_model_compile_mode():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="compact_torch_search_service",
            compact_torch_compile_model_inference=True,
            compact_torch_model_compile_mode="default",
            probe_simulations=[1],
        )
    )

    row = manifest["rows"][0]
    assert row["compact_torch_compile_model_inference"] is True
    assert row["compact_torch_model_compile_mode"] == "default"
    mode_flag_index = row["command"].index("--hybrid-compact-torch-model-compile-mode")
    assert row["command"][mode_flag_index + 1] == "default"


def test_hybrid_profile_grid_forwards_compact_torch_initial_inference_mode():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="compact_torch_search_service",
            compact_torch_initial_inference_mode="direct_core",
            probe_simulations=[1],
        )
    )

    row = manifest["rows"][0]
    assert row["compact_torch_initial_inference_mode"] == "direct_core"
    mode_flag_index = row["command"].index("--hybrid-compact-torch-initial-inference-mode")
    assert row["command"][mode_flag_index + 1] == "direct_core"


def test_hybrid_profile_grid_forwards_compact_torch_memory_formats():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="compact_torch_search_service",
            compact_torch_observation_memory_format="channels_last",
            compact_torch_model_memory_format="contiguous",
            probe_simulations=[1],
        )
    )

    row = manifest["rows"][0]
    assert row["compact_torch_observation_memory_format"] == "channels_last"
    assert row["compact_torch_model_memory_format"] == "contiguous"
    obs_flag_index = row["command"].index("--hybrid-compact-torch-observation-memory-format")
    assert row["command"][obs_flag_index + 1] == "channels_last"
    assert "--hybrid-compact-torch-model-memory-format" not in row["command"]


def test_hybrid_profile_grid_rejects_model_wide_channels_last():
    with pytest.raises(ValueError, match="model_memory_format=channels_last is parked"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="compact_torch_search_service",
                compact_torch_model_memory_format="channels_last",
                probe_simulations=[1],
            )
        )


def test_hybrid_profile_grid_can_emit_lightzero_initial_inference_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_initial_inference_probe=True,
            probe_simulations=[8],
        )
    )

    assert len(manifest["rows"]) == 1
    row = manifest["rows"][0]
    assert row["lightzero_initial_inference_probe"] is True
    assert row["lightzero_collect_forward_probe"] is False
    assert "--hybrid-lightzero-initial-inference-probe" in row["command"]
    assert "--hybrid-lightzero-collect-forward-probe" not in row["command"]
    probe_flag_index = row["command"].index("--hybrid-batched-stack-probe-simulations")
    assert row["command"][probe_flag_index + 1] == "0"
    assert "-lzii" in row["label"]


def test_hybrid_profile_grid_can_emit_lightzero_array_ceiling_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="recurrent_toy",
            lightzero_array_ceiling_input_mode="resident_torch_reuse",
            probe_simulations=[8],
        )
    )

    assert len(manifest["rows"]) == 1
    row = manifest["rows"][0]
    assert row["lightzero_array_ceiling_probe"] is True
    assert row["lightzero_array_ceiling_mode"] == "recurrent_toy"
    assert row["lightzero_array_ceiling_input_mode"] == "resident_torch_reuse"
    assert row["lightzero_array_ceiling_input_freshness"] == "resident_reuse_mode"
    assert row["lightzero_collect_forward_probe"] is False
    assert row["lightzero_initial_inference_probe"] is False
    assert "--hybrid-lightzero-array-ceiling-probe" in row["command"]
    assert "--hybrid-lightzero-array-ceiling-mode" in row["command"]
    assert "--hybrid-lightzero-array-ceiling-input-mode" in row["command"]
    assert "recurrent_toy" in row["command"]
    assert "resident_torch_reuse" in row["command"]
    probe_flag_index = row["command"].index("--hybrid-batched-stack-probe-simulations")
    assert row["command"][probe_flag_index + 1] == "0"
    assert "-lzarr-recurrent_toy-inresident_torch_reuse" in row["label"]


def test_hybrid_profile_grid_can_emit_dense_torch_mcts_ceiling_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="dense_torch_mcts",
            lightzero_array_ceiling_input_mode="host_uint8",
            probe_simulations=[8],
        )
    )

    row = manifest["rows"][0]
    assert row["lightzero_array_ceiling_mode"] == "dense_torch_mcts"
    assert "dense_torch_mcts" in row["command"]
    assert "-lzarr-dense_torch_mcts-inhost_uint8" in row["label"]


def test_hybrid_profile_grid_can_emit_dense_torch_compact_replay_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="dense_torch_mcts",
            lightzero_array_ceiling_input_mode="host_uint8",
            compact_service_replay_proof=True,
            probe_simulations=[16],
        )
    )

    row = manifest["rows"][0]
    assert row["compact_service_replay_proof"] is True
    assert row["lightzero_array_ceiling_mode"] == "dense_torch_mcts"
    assert "--hybrid-compact-service-replay-proof" in row["command"]
    mode_index = row["command"].index("--hybrid-lightzero-array-ceiling-mode")
    assert row["command"][mode_index + 1] == "dense_torch_mcts"
    assert "-compactreplay-lzarr-dense_torch_mcts-inhost_uint8" in row["label"]


def test_hybrid_profile_grid_can_emit_compact_torch_service_compact_replay_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="compact_torch_search_service",
            lightzero_array_ceiling_input_mode="host_uint8",
            compact_service_replay_proof=True,
            probe_simulations=[16],
        )
    )

    row = manifest["rows"][0]
    assert row["compact_service_replay_proof"] is True
    assert row["lightzero_array_ceiling_mode"] == "compact_torch_search_service"
    assert "--hybrid-compact-service-replay-proof" in row["command"]
    mode_index = row["command"].index("--hybrid-lightzero-array-ceiling-mode")
    assert row["command"][mode_index + 1] == "compact_torch_search_service"
    assert "-compactreplay-lzarr-compact_torch_search_service-inhost_uint8" in row["label"]


def test_hybrid_profile_grid_can_emit_nonsearch_floor_knobs():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="fixed_shape_search_owner",
            lightzero_array_ceiling_input_mode="host_uint8",
            compact_rollout_slab_probe=True,
            compact_rollout_slab_action_mode="scripted_random",
            hybrid_device_only_stack=True,
            hybrid_refresh_observation_stack=False,
            hybrid_native_actor_buffer=True,
        )
    )

    row = manifest["rows"][0]
    assert row["hybrid_device_only_stack"] is True
    assert row["hybrid_refresh_observation_stack"] is False
    assert row["hybrid_resident_observation_search"] is False
    assert row["hybrid_native_actor_buffer"] is True
    assert row["hybrid_persistent_compact_render_state_buffer"] is False
    assert "--hybrid-device-only-stack" in row["command"]
    assert "--no-hybrid-refresh-observation-stack" in row["command"]
    assert "--hybrid-native-actor-buffer" in row["command"]
    assert "-devicestack-norefresh-nativeactor" in row["label"]
    assert row["fixed_denominator"]["hybrid_device_only_stack"] is True
    assert row["fixed_denominator"]["hybrid_refresh_observation_stack"] is False
    assert row["fixed_denominator"]["hybrid_resident_observation_search"] is False
    assert row["fixed_denominator"]["hybrid_native_actor_buffer"] is True


def test_hybrid_profile_grid_can_emit_persistent_render_state_knob():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="fixed_shape_search_owner",
            lightzero_array_ceiling_input_mode="host_uint8",
            compact_rollout_slab_probe=True,
            hybrid_native_actor_buffer=True,
            hybrid_persistent_compact_render_state_buffer=True,
        )
    )

    row = manifest["rows"][0]
    assert row["hybrid_native_actor_buffer"] is True
    assert row["hybrid_persistent_compact_render_state_buffer"] is True
    assert "--hybrid-native-actor-buffer" in row["command"]
    assert "--hybrid-persistent-compact-render-state-buffer" in row["command"]
    assert "-nativeactor-persistentrenderstate" in row["label"]


def test_hybrid_profile_grid_can_emit_borrowed_render_state_knob():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            actor_count=1,
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="fixed_shape_search_owner",
            lightzero_array_ceiling_input_mode="host_uint8",
            compact_rollout_slab_probe=True,
            hybrid_native_actor_buffer=True,
            hybrid_borrow_single_actor_render_state=True,
        )
    )

    row = manifest["rows"][0]
    assert row["hybrid_native_actor_buffer"] is True
    assert row["hybrid_borrow_single_actor_render_state"] is True
    assert row["fixed_denominator"]["hybrid_borrow_single_actor_render_state"] is True
    assert "--hybrid-native-actor-buffer" in row["command"]
    assert "--hybrid-borrow-single-actor-render-state" in row["command"]
    assert "-nativeactor-borrowrenderstate" in row["label"]


def test_hybrid_profile_grid_borrowed_normal_death_preset_matches_fast_shape():
    args = _args(
        computes=["gpu-h100"],
        materialize_scalar_timestep=[False],
        include_l4=False,
        next_borrowed_normal_death_compact_owned_preset=True,
    )

    if args.next_borrowed_normal_death_compact_owned_preset:
        builder.apply_next_borrowed_normal_death_compact_owned_preset(args)
    manifest = builder.build_manifest(args)

    row = manifest["rows"][0]
    assert row["death_mode"] == "normal"
    assert row["batch_size"] == 1024
    assert row["actor_count"] == 1
    assert row["require_normal_death_terminal_contract"] is True
    assert row["hybrid_borrow_single_actor_render_state"] is True
    assert row["compact_torch_initial_inference_mode"] == "direct_core"
    assert "--hybrid-borrow-single-actor-render-state" in row["command"]


def test_hybrid_profile_grid_can_emit_fixed_shape_search_owner_compact_replay_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="fixed_shape_search_owner",
            lightzero_array_ceiling_input_mode="host_uint8",
            compact_service_replay_proof=True,
            probe_simulations=[16],
        )
    )

    row = manifest["rows"][0]
    assert row["compact_service_replay_proof"] is True
    assert row["lightzero_array_ceiling_mode"] == "fixed_shape_search_owner"
    assert "--hybrid-compact-service-replay-proof" in row["command"]
    mode_index = row["command"].index("--hybrid-lightzero-array-ceiling-mode")
    assert row["command"][mode_index + 1] == "fixed_shape_search_owner"
    assert "-compactreplay-lzarr-fixed_shape_search_owner-inhost_uint8" in row["label"]


def test_hybrid_profile_grid_rejects_mislabeled_fixed_shape_search_owner_input_modes():
    try:
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="fixed_shape_search_owner",
                lightzero_array_ceiling_input_mode="resident_torch_reuse",
                compact_service_replay_proof=False,
                probe_simulations=[16],
            )
        )
    except ValueError as exc:
        assert "host uint8 observations directly" in str(exc)
    else:
        raise AssertionError("expected fixed_shape_search_owner input-mode failure")


def test_hybrid_profile_grid_rejects_mislabeled_compact_torch_input_modes():
    try:
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="compact_torch_search_service",
                lightzero_array_ceiling_input_mode="resident_torch_reuse",
                compact_service_replay_proof=False,
                probe_simulations=[16],
            )
        )
    except ValueError as exc:
        assert "host uint8 observations directly" in str(exc)
    else:
        raise AssertionError("expected compact_torch_search_service input-mode failure")


def test_hybrid_profile_grid_can_emit_mock_search_service_ceiling_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="mock_search_service",
            lightzero_array_ceiling_input_mode="host_uint8",
            probe_simulations=[16],
        )
    )

    row = manifest["rows"][0]
    assert row["lightzero_array_ceiling_mode"] == "mock_search_service"
    assert "mock_search_service" in row["command"]
    assert "-lzarr-mock_search_service-inhost_uint8" in row["label"]
    assert row["fixed_denominator"]["probe_simulations"] == 16


def test_hybrid_profile_grid_can_emit_service_tax_compact_replay_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="service_tax_probe",
            lightzero_array_ceiling_input_mode="host_uint8",
            compact_service_replay_proof=True,
            probe_simulations=[16],
        )
    )

    row = manifest["rows"][0]
    assert row["compact_service_replay_proof"] is True
    assert row["lightzero_array_ceiling_mode"] == "service_tax_probe"
    assert row["lightzero_array_ceiling_input_freshness"] == "fresh_each_step_expected"
    assert "--hybrid-compact-service-replay-proof" in row["command"]
    assert "--hybrid-lightzero-array-ceiling-probe" in row["command"]
    mode_index = row["command"].index("--hybrid-lightzero-array-ceiling-mode")
    assert row["command"][mode_index + 1] == "service_tax_probe"
    assert "-compactreplay-lzarr-service_tax_probe-inhost_uint8" in row["label"]


def test_hybrid_profile_grid_can_emit_compact_rollout_slab_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            compact_rollout_slab_probe=True,
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="service_tax_probe",
            lightzero_array_ceiling_input_mode="host_uint8",
        )
    )

    row = manifest["rows"][0]
    assert row["compact_rollout_slab_probe"] is True
    assert row["compact_rollout_slab_action_mode"] == "search_feedback"
    assert row["compact_service_replay_proof"] is False
    assert "--hybrid-compact-rollout-slab-probe" in row["command"]
    assert "--hybrid-compact-rollout-slab-action-mode" in row["command"]
    assert "--hybrid-compact-service-replay-proof" not in row["command"]
    assert "-compactslab-lzarr-service_tax_probe-inhost_uint8" in row["label"]


def test_hybrid_profile_grid_can_emit_scripted_compact_rollout_slab_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            compact_rollout_slab_probe=True,
            compact_rollout_slab_action_mode="scripted_random",
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="service_tax_probe",
            lightzero_array_ceiling_input_mode="host_uint8",
        )
    )

    row = manifest["rows"][0]
    assert row["compact_rollout_slab_action_mode"] == "scripted_random"
    assert "-action-scripted_random-" in row["label"]
    flag_index = row["command"].index("--hybrid-compact-rollout-slab-action-mode")
    assert row["command"][flag_index + 1] == "scripted_random"


def test_hybrid_profile_grid_can_emit_compact_rollout_slab_sample_gate_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            compact_rollout_slab_probe=True,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=64,
            compact_rollout_slab_sample_gate_interval=8,
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="service_tax_probe",
            lightzero_array_ceiling_input_mode="host_uint8",
        )
    )

    row = manifest["rows"][0]
    assert row["compact_rollout_slab_probe"] is True
    assert row["compact_rollout_slab_sample_gate"] is True
    assert row["compact_rollout_slab_sample_gate_batch_size"] == 64
    assert row["compact_rollout_slab_sample_gate_interval"] == 8
    assert row["compact_rollout_slab_sample_gate_replay_pair_capacity"] == 4096
    assert "--hybrid-compact-rollout-slab-probe" in row["command"]
    assert "--hybrid-compact-rollout-slab-sample-gate" in row["command"]
    batch_flag_index = row["command"].index("--hybrid-compact-rollout-slab-sample-gate-batch-size")
    assert row["command"][batch_flag_index + 1] == "64"
    interval_flag_index = row["command"].index("--hybrid-compact-rollout-slab-sample-gate-interval")
    assert row["command"][interval_flag_index + 1] == "8"
    capacity_flag_index = row["command"].index(
        "--hybrid-compact-rollout-slab-sample-gate-replay-pair-capacity"
    )
    assert row["command"][capacity_flag_index + 1] == "4096"
    assert (
        "-compactslab-samplegate-b64-i8-cap4096-lzarr-service_tax_probe-inhost_uint8"
        in row["label"]
    )


def test_hybrid_profile_grid_can_emit_compact_rollout_slab_learner_gate_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            compact_rollout_slab_probe=True,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=64,
            compact_rollout_slab_sample_gate_interval=8,
            compact_rollout_slab_learner_gate=True,
            compact_rollout_slab_learner_gate_train_steps=2,
            compact_rollout_slab_learner_gate_device="cuda",
            compact_rollout_slab_learner_gate_include_rnd=True,
            compact_rollout_slab_learner_gate_impl="toy_probe",
            compact_rollout_slab_learner_gate_support_scale=3,
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="service_tax_probe",
            lightzero_array_ceiling_input_mode="host_uint8",
        )
    )

    row = manifest["rows"][0]
    assert row["compact_rollout_slab_learner_gate"] is True
    assert row["compact_rollout_slab_learner_gate_train_steps"] == 2
    assert row["compact_rollout_slab_learner_gate_device"] == "cuda"
    assert row["compact_rollout_slab_learner_gate_include_rnd"] is True
    assert row["compact_rollout_slab_learner_gate_impl"] == "toy_probe"
    assert row["compact_rollout_slab_learner_gate_support_scale"] == 3
    assert row["compact_rollout_slab_learner_gate_num_unroll_steps"] == 1
    assert "--hybrid-compact-rollout-slab-learner-gate" in row["command"]
    train_steps_index = row["command"].index(
        "--hybrid-compact-rollout-slab-learner-gate-train-steps"
    )
    assert row["command"][train_steps_index + 1] == "2"
    impl_index = row["command"].index("--hybrid-compact-rollout-slab-learner-gate-impl")
    assert row["command"][impl_index + 1] == "toy_probe"
    support_scale_index = row["command"].index(
        "--hybrid-compact-rollout-slab-learner-gate-support-scale"
    )
    assert row["command"][support_scale_index + 1] == "3"
    unroll_index = row["command"].index(
        "--hybrid-compact-rollout-slab-learner-gate-num-unroll-steps"
    )
    assert row["command"][unroll_index + 1] == "1"
    assert "--hybrid-compact-rollout-slab-learner-gate-include-rnd" in row["command"]
    assert "-learnergate-toy_probe-cuda-t2-ss3-u1-rnd-" in row["label"]


def test_hybrid_profile_grid_can_emit_explicit_compact_owned_loop_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            compact_rollout_slab_probe=True,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_learner_gate=True,
            compact_rollout_slab_learner_gate_impl="compact_muzero",
            compact_owned_loop_entrypoint=True,
            compact_owned_loop_policy_version_ref="unit-owned-policy-v1",
            compact_owned_loop_model_version_ref="unit-owned-model-v1",
            compact_owned_loop_policy_source="unit_test_builder_owned_loop",
            compact_owned_loop_capture_replay_store_state=True,
            hybrid_device_only_stack=True,
            hybrid_resident_observation_search=True,
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="compact_torch_search_service",
            lightzero_array_ceiling_input_mode="host_uint8",
        )
    )

    row = manifest["rows"][0]
    assert row["compact_owned_loop_entrypoint"] is True
    assert row["compact_owned_loop_policy_version_ref"] == "unit-owned-policy-v1"
    assert row["compact_owned_loop_model_version_ref"] == "unit-owned-model-v1"
    assert row["compact_owned_loop_policy_source"] == "unit_test_builder_owned_loop"
    assert row["compact_owned_loop_capture_replay_store_state"] is True
    assert "-ownedloop-" in row["label"]
    assert "--hybrid-compact-owned-loop-entrypoint" in row["command"]
    policy_index = row["command"].index("--hybrid-compact-owned-loop-policy-version-ref")
    assert row["command"][policy_index + 1] == "unit-owned-policy-v1"
    model_index = row["command"].index("--hybrid-compact-owned-loop-model-version-ref")
    assert row["command"][model_index + 1] == "unit-owned-model-v1"
    assert "--hybrid-compact-owned-loop-capture-replay-store-state" in row["command"]
    assert row["fixed_denominator"]["compact_owned_loop_entrypoint"] is True


def test_hybrid_profile_grid_matched_denominator_compact_owned_preset():
    args = _args(
        computes=["gpu-l4-t4"],
        batch_sizes=[1],
        steps=1,
        warmup_steps=0,
        lightzero_array_ceiling_probe=False,
    )

    builder.apply_next_matched_denominator_compact_owned_preset(args)
    manifest = builder.build_manifest(args)

    row = manifest["rows"][0]
    assert manifest["matched_denominator"]["id"] == builder.MATCHED_DENOMINATOR_ID
    assert row["matched_pair_role"] == "compact_candidate"
    assert row["speed_currency"] == builder.MATCHED_COMPACT_SPEED_CURRENCY
    assert row["row_purpose"] == builder.MATCHED_DENOMINATOR_ROW_PURPOSE
    assert row["promotion_claim"] is False
    assert row["compute"] == "gpu-h100"
    assert row["batch_size"] == 1024
    assert row["actor_count"] == 16
    assert row["steps"] == 60
    assert row["warmup_steps"] == 15
    assert row["max_ticks"] == 2000
    assert row["probe_simulations"] == 8
    assert row["compact_rollout_slab_sample_gate_batch_size"] == 512
    assert row["compact_rollout_slab_sample_gate_interval"] == 8
    assert row["compact_rollout_slab_learner_gate_impl"] == "compact_muzero"
    assert row["compact_rollout_slab_learner_gate_support_scale"] == 300
    assert row["compact_rollout_slab_learner_gate_num_unroll_steps"] == 1
    assert row["compact_owned_loop_entrypoint"] is True
    assert row["compact_owned_loop_capture_replay_store_state"] is True
    assert row["hybrid_device_only_stack"] is True
    assert row["hybrid_resident_observation_search"] is True
    assert row["hybrid_native_actor_buffer"] is True
    assert row["lightzero_array_ceiling_mode"] == "compact_torch_search_service"
    assert row["lightzero_consumer_root_noise_weight"] == 0.0
    assert row["require_terminal_compact_owned_nstep"] is False
    assert row["launch_mode"] == builder.LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT
    assert row["fixed_denominator"]["matched_denominator_id"] == row["matched_denominator_id"]
    assert "--hybrid-compact-owned-loop-entrypoint" in row["command"]
    assert "--hybrid-compact-owned-loop-capture-replay-store-state" in row["command"]
    assert "--detach" in row["command"]
    assert "--hybrid-profile-spawn-result" in row["command"]
    assert (
        row["fixed_denominator"]["compact_owned_loop_policy_version_ref"]
        == "matched-denominator-compact-owned-policy-v1"
    )


def test_hybrid_profile_grid_can_emit_terminal_nstep_compact_owned_preset():
    args = _args(include_l4=False)
    builder.apply_next_terminal_nstep_compact_owned_preset(args)

    manifest = builder.build_manifest(args)

    row = manifest["rows"][0]
    assert len(manifest["rows"]) == 1
    assert row["compute"] == "gpu-h100"
    assert row["batch_size"] == 128
    assert row["actor_count"] == 8
    assert row["steps"] == 6
    assert row["warmup_steps"] == 0
    assert row["max_ticks"] == 3
    assert row["compact_rollout_slab_probe"] is True
    assert row["compact_rollout_slab_sample_gate"] is True
    assert row["compact_rollout_slab_sample_gate_batch_size"] == 0
    assert row["compact_rollout_slab_sample_gate_interval"] == 1
    assert row["compact_rollout_slab_learner_gate"] is True
    assert row["compact_rollout_slab_learner_gate_impl"] == "compact_muzero"
    assert row["compact_rollout_slab_learner_gate_support_scale"] == 9
    assert row["compact_rollout_slab_learner_gate_num_unroll_steps"] == 2
    assert row["compact_owned_loop_entrypoint"] is True
    assert row["compact_owned_loop_policy_version_ref"]
    assert row["compact_owned_loop_policy_source"]
    assert row["compact_owned_loop_capture_replay_store_state"] is True
    assert row["hybrid_device_only_stack"] is True
    assert row["hybrid_resident_observation_search"] is True
    assert row["hybrid_native_actor_buffer"] is True
    assert row["require_terminal_compact_owned_nstep"] is True
    assert row["launch_mode"] == builder.LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT
    assert row["result_capture"] == builder.RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET
    assert "--detach" in row["command"]
    assert "--hybrid-profile-spawn-result" in row["command"]
    assert "--hybrid-compact-owned-loop-entrypoint" in row["command"]
    max_ticks_index = row["command"].index("--max-ticks")
    assert row["command"][max_ticks_index + 1] == "3"


def test_hybrid_profile_grid_can_emit_normal_death_compact_owned_preset():
    args = _args(include_l4=False)
    builder.apply_next_normal_death_compact_owned_preset(args)

    manifest = builder.build_manifest(args)

    row = manifest["rows"][0]
    assert len(manifest["rows"]) == 1
    assert row["steps"] == 64
    assert row["max_ticks"] == 2000
    assert row["max_ticks"] > row["steps"]
    assert row["death_mode"] == "normal"
    assert row["require_terminal_compact_owned_nstep"] is True
    assert row["require_normal_death_terminal_contract"] is True
    assert row["normal_death_terminal_contract_evidence_id"]
    assert row["normal_death_terminal_contract_evidence_refs"]
    assert row["compact_owned_loop_entrypoint"] is True
    assert row["compact_rollout_slab_sample_gate_batch_size"] == 256
    assert row["compact_rollout_slab_sample_gate_interval"] == 8
    assert row["compact_rollout_slab_sample_gate_replay_pair_capacity"] == 8192
    assert row["compact_rollout_slab_learner_gate_support_scale"] == 300
    assert row["compact_torch_compile_search"] is False
    assert row["compact_rollout_slab_learner_gate_num_unroll_steps"] == 2
    death_mode_index = row["command"].index("--death-mode")
    assert row["command"][death_mode_index + 1] == "normal"
    max_ticks_index = row["command"].index("--max-ticks")
    assert row["command"][max_ticks_index + 1] == "2000"
    sample_batch_index = row["command"].index(
        "--hybrid-compact-rollout-slab-sample-gate-batch-size"
    )
    assert row["command"][sample_batch_index + 1] == "256"
    assert "--no-hybrid-compact-torch-compile-search" in row["command"]
    assert "--hybrid-compact-owned-loop-entrypoint" in row["command"]


def test_hybrid_profile_grid_rejects_normal_death_contract_without_normal_mode():
    with pytest.raises(ValueError, match="death-mode normal"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                require_normal_death_terminal_contract=True,
            )
        )


def test_hybrid_profile_grid_rejects_normal_death_contract_with_sample_all_gate():
    args = _args(include_l4=False)
    builder.apply_next_normal_death_compact_owned_preset(args)
    args.compact_rollout_slab_sample_gate_batch_size = 0

    with pytest.raises(ValueError, match="bounded"):
        builder.build_manifest(args)


def test_hybrid_profile_grid_can_emit_mctx_compact_rollout_slab_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            compact_rollout_slab_probe=True,
            mctx_compact_search_probe=True,
            mctx_hidden_dim=64,
            mctx_visual_channels=8,
            probe_simulations=[16],
        )
    )

    row = manifest["rows"][0]
    assert row["compact_rollout_slab_probe"] is True
    assert row["mctx_compact_search_probe"] is True
    assert row["fixed_denominator"]["input_mode"] == "mctx_jax_host_uint8"
    assert "--hybrid-mctx-compact-search-probe" in row["command"]
    assert "--hybrid-lightzero-array-ceiling-probe" not in row["command"]
    assert "-compactslab-mctx-h64-vc8" in row["label"]


def test_hybrid_profile_grid_emits_compact_root_tape_flags():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            compact_rollout_slab_probe=True,
            compact_root_tape_compare=True,
            compact_root_tape_max_records=5,
            compact_root_tape_reference_label="primary",
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="fixed_shape_search_owner",
            probe_simulations=[8],
        )
    )

    row = manifest["rows"][0]
    assert row["compact_root_tape_compare"] is True
    assert row["compact_root_tape_max_records"] == 5
    assert row["compact_root_tape_reference_label"] == "primary"
    assert row["comparison_group"] == "compact_root_tape_fixed_denominator"
    assert row["fixed_denominator"]["compact_root_tape_compare"] is True
    assert row["fixed_denominator"]["compact_root_tape_max_records"] == 5
    assert row["fixed_denominator"]["lightzero_consumer_root_noise_weight"] == -1.0
    assert "--hybrid-compact-root-tape-compare" in row["command"]
    max_records_index = row["command"].index("--hybrid-compact-root-tape-max-records")
    assert row["command"][max_records_index + 1] == "5"


def test_hybrid_profile_grid_emits_large_fixed_root_tape_detached_preset():
    args = _args(
        computes=["gpu-l4-t4"],
        batch_sizes=[64],
        materialize_scalar_timestep=[True],
        include_l4=True,
    )

    builder.apply_next_fixed_root_tape_large_preset(args)
    manifest = builder.build_manifest(args)

    assert manifest["launch_mode"] == builder.LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT
    assert manifest["result_capture"] == builder.RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET
    assert len(manifest["rows"]) == 1
    row = manifest["rows"][0]
    assert row["compute"] == "gpu-h100"
    assert row["batch_size"] == 512
    assert row["actor_count"] == 16
    assert row["steps"] == 60
    assert row["warmup_steps"] == 15
    assert row["compact_root_tape_compare"] is True
    assert row["compact_root_tape_max_records"] == 16
    assert row["lightzero_consumer_root_noise_weight"] == 0.0
    assert row["launch_mode"] == builder.LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT
    assert row["result_capture"] == builder.RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET
    assert row["row_timeout_sec"] == 300
    assert row["result_timeout_sec"] == 7200
    assert row["captured_result_required"] is True
    assert row["require_payload_label_triad"] is True
    modal_run_index = next(
        index
        for index, part in enumerate(row["command"][:-1])
        if part == "modal" and row["command"][index + 1] == "run"
    )
    assert row["command"][modal_run_index + 1 : modal_run_index + 3] == [
        "run",
        "--detach",
    ]
    assert "--hybrid-profile-spawn-result" in row["command"]
    assert row["profile_only"] is True
    assert row["calls_train_muzero"] is False
    assert row["touches_live_runs"] is False


def test_hybrid_profile_grid_emits_fixed_root_tape_mctx_detached_preset():
    args = _args(
        computes=["gpu-l4-t4"],
        batch_sizes=[64],
        materialize_scalar_timestep=[True],
        include_l4=True,
    )

    builder.apply_next_fixed_root_tape_mctx_preset(args)
    manifest = builder.build_manifest(args)

    row = manifest["rows"][0]
    assert row["compute"] == "gpu-h100"
    assert row["batch_size"] == 512
    assert row["actor_count"] == 16
    assert row["compact_root_tape_compare"] is True
    assert row["compact_root_tape_compare_mctx"] is True
    assert row["fixed_denominator"]["compact_root_tape_compare_mctx"] is True
    assert row["fixed_denominator"]["compact_root_tape_mctx_hidden_dim"] == 64
    assert row["fixed_denominator"]["compact_root_tape_mctx_visual_channels"] == 8
    assert row["fixed_denominator"]["compact_root_tape_mctx_require_gpu_backend"] is True
    assert row["mctx_compact_search_probe"] is False
    assert "-rootmctx-" in row["label"]
    assert "--hybrid-compact-root-tape-compare-mctx" in row["command"]
    assert "--hybrid-mctx-compact-search-probe" not in row["command"]
    assert "--hybrid-mctx-num-simulations" in row["command"]
    mctx_sim_index = row["command"].index("--hybrid-mctx-num-simulations")
    assert row["command"][mctx_sim_index + 1] == "8"


def test_hybrid_profile_grid_emits_fixed_root_tape_compile_preset():
    args = _args(
        computes=["gpu-l4-t4"],
        batch_sizes=[64],
        materialize_scalar_timestep=[True],
        include_l4=True,
    )

    builder.apply_next_fixed_root_tape_compile_preset(args)
    manifest = builder.build_manifest(args)

    row = manifest["rows"][0]
    assert row["compute"] == "gpu-h100"
    assert row["probe_simulations"] == 1
    assert row["compact_root_tape_compare"] is True
    assert row["compact_root_tape_compare_fixed_shape_floor"] is False
    assert row["compact_root_tape_compare_model_compile"] is True
    assert row["compact_root_tape_model_compile_mode"] == "default"
    assert row["compact_root_tape_require_model_compile"] is True
    assert row["compact_torch_compile_model_inference"] is False
    assert row["compact_torch_compile_search"] is False
    assert row["fixed_denominator"]["compact_root_tape_compare_model_compile"] is True
    assert row["fixed_denominator"]["compact_root_tape_model_compile_mode"] == "default"
    assert "--hybrid-compact-root-tape-compare-model-compile" in row["command"]
    assert "--no-hybrid-compact-root-tape-compare-fixed-shape-floor" in row["command"]
    assert "--no-hybrid-compact-torch-compile-search" in row["command"]


def test_hybrid_profile_grid_emits_fixed_root_tape_direct_core_preset():
    args = _args(
        computes=["gpu-l4-t4"],
        batch_sizes=[64],
        materialize_scalar_timestep=[True],
        include_l4=True,
    )

    builder.apply_next_fixed_root_tape_direct_core_preset(args)
    manifest = builder.build_manifest(args)

    row = manifest["rows"][0]
    assert row["compute"] == "gpu-h100"
    assert row["probe_simulations"] == 1
    assert row["compact_root_tape_compare"] is True
    assert row["compact_root_tape_compare_fixed_shape_floor"] is False
    assert row["compact_root_tape_compare_direct_core"] is True
    assert row["compact_torch_initial_inference_mode"] == "model_method"
    assert row["compact_torch_compile_model_inference"] is False
    assert row["compact_torch_compile_search"] is False
    assert row["lightzero_consumer_root_noise_weight"] == 0.0
    assert row["fixed_denominator"]["compact_root_tape_compare_direct_core"] is True
    assert "--hybrid-compact-root-tape-compare-direct-core" in row["command"]
    assert "--hybrid-compact-torch-initial-inference-mode" not in row["command"]
    assert "--no-hybrid-compact-root-tape-compare-fixed-shape-floor" in row["command"]
    assert "--no-hybrid-compact-torch-compile-search" in row["command"]


def test_hybrid_profile_grid_rejects_detached_capture_without_modal_get():
    with pytest.raises(ValueError, match="modal_function_call_get"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                launch_mode=builder.LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT,
                result_capture=builder.RESULT_CAPTURE_STDOUT_JSON,
            )
        )


def test_hybrid_profile_grid_rejects_mctx_without_compact_slab():
    with pytest.raises(ValueError, match="requires --compact-rollout-slab-probe"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                mctx_compact_search_probe=True,
            )
        )


def test_hybrid_profile_grid_rejects_compact_root_tape_without_compact_slab():
    with pytest.raises(ValueError, match="requires --compact-rollout-slab-probe"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                compact_root_tape_compare=True,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="fixed_shape_search_owner",
            )
        )


def test_hybrid_profile_grid_rejects_compact_root_tape_resident_without_snapshot():
    with pytest.raises(ValueError, match="does not yet support resident"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                compact_rollout_slab_probe=True,
                compact_root_tape_compare=True,
                hybrid_device_only_stack=True,
                hybrid_resident_observation_search=True,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="compact_torch_search_service",
            )
        )


def test_hybrid_profile_grid_rejects_resident_search_without_device_stack():
    with pytest.raises(ValueError, match="requires --hybrid-device-only-stack"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                compact_rollout_slab_probe=True,
                hybrid_resident_observation_search=True,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="compact_torch_search_service",
                lightzero_array_ceiling_input_mode="host_uint8",
            )
        )


def test_hybrid_profile_grid_rejects_resident_search_without_compact_torch_service():
    with pytest.raises(ValueError, match="compact_torch_search_service"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                compact_rollout_slab_probe=True,
                hybrid_device_only_stack=True,
                hybrid_resident_observation_search=True,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="fixed_shape_search_owner",
                lightzero_array_ceiling_input_mode="host_uint8",
            )
        )


def test_hybrid_profile_grid_rejects_lightzero_probe_with_device_latest():
    with pytest.raises(ValueError, match="device-latest false"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                device_latest=[True],
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="service_tax_probe",
            )
        )


def test_hybrid_profile_grid_rejects_mctx_probe_with_device_latest():
    with pytest.raises(ValueError, match="device-latest false"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                device_latest=[True],
                compact_rollout_slab_probe=True,
                mctx_compact_search_probe=True,
            )
        )


def test_hybrid_profile_grid_rejects_lightzero_probe_with_non_uint8_stack():
    with pytest.raises(ValueError, match="stack-storage-dtype uint8"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                stack_storage_dtype="float32",
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="service_tax_probe",
            )
        )


def test_hybrid_profile_grid_rejects_lightzero_probe_with_wrong_surface_or_backend():
    with pytest.raises(ValueError, match="render-surface direct_gray64"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                render_surface="full_rgb",
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="service_tax_probe",
            )
        )
    with pytest.raises(ValueError, match="observation-renderer-backend"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                observation_renderer_backend="host_numpy",
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="service_tax_probe",
            )
        )


def test_hybrid_profile_grid_rejects_resident_search_without_cuda():
    with pytest.raises(ValueError, match="lightzero-consumer-use-cuda"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                compact_rollout_slab_probe=True,
                hybrid_device_only_stack=True,
                hybrid_resident_observation_search=True,
                lightzero_consumer_use_cuda=False,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="compact_torch_search_service",
                lightzero_array_ceiling_input_mode="host_uint8",
            )
        )


def test_hybrid_profile_grid_build_manifest_rejects_cli_only_preflights():
    with pytest.raises(ValueError, match="mock-service-materialize-public-output"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="service_tax_probe",
                lightzero_mock_service_materialize_public_output=True,
            )
        )
    with pytest.raises(ValueError, match="persistent-compact-render-state-buffer"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="fixed_shape_search_owner",
                compact_rollout_slab_probe=True,
                hybrid_persistent_compact_render_state_buffer=True,
            )
        )
    with pytest.raises(ValueError, match="borrow-single-actor-render-state"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="fixed_shape_search_owner",
                compact_rollout_slab_probe=True,
                hybrid_borrow_single_actor_render_state=True,
            )
        )


def test_hybrid_profile_grid_rejects_compact_root_tape_without_second_service():
    with pytest.raises(ValueError, match="secondary service"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                compact_rollout_slab_probe=True,
                compact_root_tape_compare=True,
                compact_root_tape_compare_fixed_shape_floor=False,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="fixed_shape_search_owner",
            )
        )


def test_hybrid_profile_grid_rejects_root_tape_mctx_without_root_tape():
    with pytest.raises(ValueError, match="requires --compact-root-tape-compare"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                compact_root_tape_compare_mctx=True,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="compact_torch_search_service",
            )
        )


def test_hybrid_profile_grid_rejects_root_tape_direct_core_without_root_tape():
    with pytest.raises(ValueError, match="requires --compact-root-tape-compare"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                compact_root_tape_compare_direct_core=True,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="compact_torch_search_service",
            )
        )


def test_hybrid_profile_grid_rejects_root_tape_direct_core_primary_direct_core():
    with pytest.raises(ValueError, match="model_method primary"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                compact_rollout_slab_probe=True,
                compact_root_tape_compare=True,
                compact_root_tape_compare_fixed_shape_floor=False,
                compact_root_tape_compare_direct_core=True,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="compact_torch_search_service",
                compact_torch_initial_inference_mode="direct_core",
                lightzero_consumer_root_noise_weight=0.0,
            )
        )


def test_hybrid_profile_grid_rejects_root_tape_mctx_with_mctx_primary():
    with pytest.raises(ValueError, match="non-MCTX primary"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                compact_rollout_slab_probe=True,
                compact_root_tape_compare=True,
                compact_root_tape_compare_mctx=True,
                mctx_compact_search_probe=True,
            )
        )


def test_hybrid_profile_grid_rejects_mctx_with_lightzero_probe():
    with pytest.raises(ValueError, match="cannot be combined with LightZero probes"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                compact_rollout_slab_probe=True,
                mctx_compact_search_probe=True,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="service_tax_probe",
            )
        )


def test_hybrid_profile_grid_rejects_compact_slab_without_search_service():
    with pytest.raises(ValueError, match="compact-rollout-slab-probe requires"):
        builder.build_manifest(
            _args(
                compact_rollout_slab_probe=True,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="policy_arrays",
            )
        )


def test_hybrid_profile_grid_rejects_compact_slab_and_old_replay_proof_together():
    with pytest.raises(ValueError, match="owns replay-index commits"):
        builder.build_manifest(
            _args(
                compact_rollout_slab_probe=True,
                compact_service_replay_proof=True,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="service_tax_probe",
            )
        )


def test_hybrid_profile_grid_rejects_sample_gate_without_compact_slab():
    with pytest.raises(ValueError, match="sample-gate requires"):
        builder.build_manifest(
            _args(
                compact_rollout_slab_sample_gate=True,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="service_tax_probe",
            )
        )


def test_hybrid_profile_grid_rejects_sample_gate_with_scalar_rows():
    with pytest.raises(ValueError, match="no-scalar proof"):
        builder.build_manifest(
            _args(
                materialize_scalar_timestep=[False, True],
                compact_rollout_slab_probe=True,
                compact_rollout_slab_sample_gate=True,
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="service_tax_probe",
            )
        )


def test_hybrid_profile_grid_can_emit_mock_search_service_compact_replay_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="mock_search_service",
            lightzero_array_ceiling_input_mode="host_uint8",
            compact_service_replay_proof=True,
            probe_simulations=[16],
        )
    )

    row = manifest["rows"][0]
    assert row["compact_service_replay_proof"] is True
    assert row["lightzero_array_ceiling_mode"] == "mock_search_service"
    assert "--hybrid-compact-service-replay-proof" in row["command"]
    mode_index = row["command"].index("--hybrid-lightzero-array-ceiling-mode")
    assert row["command"][mode_index + 1] == "mock_search_service"
    assert "-compactreplay-lzarr-mock_search_service-inhost_uint8" in row["label"]


def test_hybrid_profile_grid_can_emit_mock_search_service_public_output_edge():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_array_ceiling_probe=True,
            lightzero_array_ceiling_mode="mock_search_service",
            lightzero_array_ceiling_input_mode="host_uint8",
            lightzero_mock_service_materialize_public_output=True,
            probe_simulations=[16],
        )
    )

    row = manifest["rows"][0]
    assert row["lightzero_mock_service_materialize_public_output"] is True
    assert "--hybrid-lightzero-mock-service-materialize-public-output" in row["command"]


def test_hybrid_profile_grid_can_emit_lightzero_mcts_arrays_boundary_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_mcts_arrays_boundary_probe=True,
            probe_simulations=[8],
        )
    )

    assert len(manifest["rows"]) == 1
    row = manifest["rows"][0]
    assert row["lightzero_mcts_arrays_boundary_probe"] is True
    assert row["lightzero_mcts_arrays_boundary_impl"] == "stock_facade"
    assert row["lightzero_mcts_arrays_boundary_input_mode"] == "host_uint8"
    assert row["lightzero_collect_forward_probe"] is False
    assert row["lightzero_initial_inference_probe"] is False
    assert row["lightzero_array_ceiling_probe"] is False
    assert "--hybrid-lightzero-mcts-arrays-boundary-probe" in row["command"]
    assert "--hybrid-lightzero-mcts-arrays-boundary-impl" in row["command"]
    assert "--hybrid-lightzero-mcts-arrays-boundary-input-mode" in row["command"]
    assert "--hybrid-lightzero-consumer-num-simulations" in row["command"]
    probe_flag_index = row["command"].index("--hybrid-batched-stack-probe-simulations")
    assert row["command"][probe_flag_index + 1] == "0"
    assert "-lzmctsarr" in row["label"]


def test_hybrid_profile_grid_can_emit_direct_lightzero_mcts_arrays_boundary_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_mcts_arrays_boundary_probe=True,
            lightzero_mcts_arrays_boundary_impl="direct_ctree_arrays",
            lightzero_mcts_arrays_boundary_input_mode="host_uint8_pinned",
            probe_simulations=[8],
        )
    )

    row = manifest["rows"][0]
    assert row["lightzero_mcts_arrays_boundary_probe"] is True
    assert row["lightzero_mcts_arrays_boundary_impl"] == "direct_ctree_arrays"
    assert row["lightzero_mcts_arrays_boundary_input_mode"] == "host_uint8_pinned"
    assert row["lightzero_mcts_arrays_boundary_input_freshness"] == ("fresh_each_step_expected")
    assert "--hybrid-lightzero-mcts-arrays-boundary-impl" in row["command"]
    impl_index = row["command"].index("--hybrid-lightzero-mcts-arrays-boundary-impl")
    assert row["command"][impl_index + 1] == "direct_ctree_arrays"
    input_index = row["command"].index("--hybrid-lightzero-mcts-arrays-boundary-input-mode")
    assert row["command"][input_index + 1] == "host_uint8_pinned"
    assert "-lzmctsarr-direct_ctree_arrays-inhost_uint8_pinned" in row["label"]


def test_hybrid_profile_grid_can_emit_compact_service_replay_proof_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_mcts_arrays_boundary_probe=True,
            lightzero_mcts_arrays_boundary_impl="direct_ctree_gpu_latent",
            compact_service_replay_proof=True,
            probe_simulations=[8],
        )
    )

    row = manifest["rows"][0]
    assert row["compact_service_replay_proof"] is True
    assert "--hybrid-compact-service-replay-proof" in row["command"]


def test_hybrid_profile_grid_can_emit_precomputed_recurrent_ctree_boundary_rows():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_mcts_arrays_boundary_probe=True,
            lightzero_mcts_arrays_boundary_impl=("direct_ctree_gpu_latent_precomputed_recurrent"),
            probe_simulations=[16],
        )
    )

    row = manifest["rows"][0]
    assert row["lightzero_mcts_arrays_boundary_impl"] == (
        "direct_ctree_gpu_latent_precomputed_recurrent"
    )
    impl_index = row["command"].index("--hybrid-lightzero-mcts-arrays-boundary-impl")
    assert row["command"][impl_index + 1] == ("direct_ctree_gpu_latent_precomputed_recurrent")
    assert "-lzmctsarr-direct_ctree_gpu_latent_precomputed_recurrent" in row["label"]


def test_hybrid_profile_grid_can_override_lightzero_root_noise_weight():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_mcts_arrays_boundary_probe=True,
            lightzero_consumer_root_noise_weight=0.0,
            probe_simulations=[8],
        )
    )

    row = manifest["rows"][0]
    assert row["lightzero_consumer_root_noise_weight"] == 0.0
    flag_index = row["command"].index("--hybrid-lightzero-consumer-root-noise-weight")
    assert row["command"][flag_index + 1] == "0.0"


def test_next_direct_ctree_comparison_preset_emits_fixed_denominator_grid():
    args = _args(
        computes=["gpu-h100"],
        batch_sizes=[128],
        steps=1,
        warmup_steps=0,
        include_l4=True,
    )
    builder.apply_next_direct_ctree_comparison_preset(args)

    manifest = builder.build_manifest(args)

    assert len(manifest["rows"]) == 6
    assert manifest["comparison_group"] == "mcts_arrays_boundary_fixed_denominator"
    assert [
        (
            row["compute"],
            row["lightzero_mcts_arrays_boundary_impl"],
            row["batch_size"],
            row["actor_count"],
            row["probe_simulations"],
            row["steps"],
            row["warmup_steps"],
        )
        for row in manifest["rows"]
    ] == [
        ("gpu-h100", "stock_facade", 512, 16, 8, 60, 15),
        ("gpu-h100", "direct_ctree_arrays", 512, 16, 8, 60, 15),
        ("gpu-h100", "direct_ctree_gpu_latent", 512, 16, 8, 60, 15),
        ("gpu-l4-t4", "stock_facade", 512, 16, 8, 60, 15),
        ("gpu-l4-t4", "direct_ctree_arrays", 512, 16, 8, 60, 15),
        ("gpu-l4-t4", "direct_ctree_gpu_latent", 512, 16, 8, 60, 15),
    ]
    assert all(
        row["fixed_denominator"]
        == {
            "batch_size": 512,
            "actor_count": 16,
            "probe_simulations": 8,
            "steps": 60,
            "warmup_steps": 15,
            "max_ticks": 2000,
            "death_mode": "profile_no_death",
            "input_mode": "host_uint8",
            "materialize_scalar_timestep": False,
            "lightzero_consumer_root_noise_weight": -1.0,
            "hybrid_device_only_stack": False,
            "hybrid_refresh_observation_stack": True,
            "hybrid_resident_observation_search": False,
            "hybrid_native_actor_buffer": False,
            "hybrid_persistent_compact_render_state_buffer": False,
            "hybrid_borrow_single_actor_render_state": False,
        }
        for row in manifest["rows"]
    )
    for row in manifest["rows"]:
        impl_index = row["command"].index("--hybrid-lightzero-mcts-arrays-boundary-impl")
        assert row["command"][impl_index + 1] == row["lightzero_mcts_arrays_boundary_impl"]


def test_hybrid_profile_grid_labels_resident_mcts_input_as_stale_ceiling():
    manifest = builder.build_manifest(
        _args(
            computes=["gpu-h100"],
            materialize_scalar_timestep=[False],
            lightzero_mcts_arrays_boundary_probe=True,
            lightzero_mcts_arrays_boundary_impl="direct_ctree_arrays",
            lightzero_mcts_arrays_boundary_input_mode="resident_torch_reuse",
            probe_simulations=[8],
        )
    )

    row = manifest["rows"][0]
    assert row["lightzero_mcts_arrays_boundary_input_mode"] == "resident_torch_reuse"
    assert row["lightzero_mcts_arrays_boundary_input_freshness"] == "resident_reuse_mode"


def test_hybrid_profile_grid_rejects_compact_replay_for_non_search_array_ceiling():
    with pytest.raises(ValueError, match="compact-service-replay-proof requires"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                lightzero_array_ceiling_probe=True,
                lightzero_array_ceiling_mode="policy_arrays",
                compact_service_replay_proof=True,
            )
        )


def test_hybrid_profile_grid_rejects_multiple_lightzero_probe_flags_as_library():
    with pytest.raises(ValueError, match="at most one LightZero probe"):
        builder.build_manifest(
            _args(
                computes=["gpu-h100"],
                materialize_scalar_timestep=[False],
                lightzero_collect_forward_probe=True,
                lightzero_mcts_arrays_boundary_probe=True,
            )
        )
