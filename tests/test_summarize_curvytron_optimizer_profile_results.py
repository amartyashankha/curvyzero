import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / (
    "summarize_curvytron_optimizer_profile_results.py"
)
SPEC = importlib.util.spec_from_file_location("optimizer_profile_summary", SCRIPT_PATH)
assert SPEC is not None
summary = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summary)


def _attested_payload():
    return {
        "status": "complete",
        "compact": {
            "schema_id": "curvyzero_lightzero_curvytron_visual_survival_compact_output/v0",
            "mode": "profile",
            "ok": True,
            "status": "completed",
            "compute": "gpu-l4-t4-cpu40",
            "called_train_muzero": True,
            "trainer_entrypoint": "lzero.entry.train_muzero",
            "semantic_identity": {
                "schema_id": "curvyzero_optimizer_profile_semantic_identity/v0",
                "observation_raw_dtype": "uint8",
                "observation_model_dtype": "float32",
                "scalar_materialization_semantics": (
                    "batched_profile_manager_materializes_lightzero_scalar_timesteps"
                ),
                "lightzero_to_play_mode": "fixed_opponent_minus_one",
                "zero_mask_filtering_semantics": (
                    "environment_action_mask_consumed_by_lightzero"
                ),
                "consumer_semantics": (
                    "stock_lightzero_train_muzero_collect_search_replay_learner"
                ),
                "collect_search_backend": "stock",
                "collect_search_ctree_backend": "lightzero",
                "env_steps_collected_source": "collector_envstep_delta",
                "speed_currency": "stock_train_muzero_profile_env_steps_per_sec",
            },
            "command": {
                "env_variant": "source_state_fixed_opponent",
                "env_manager_type": "curvyzero_batched_profile",
                "collector_env_num": 512,
                "n_episode": 512,
                "num_simulations": 8,
                "batch_size": 64,
                "source_max_steps": 512,
                "max_train_iter": 96,
                "max_env_step": 1_000_000,
                "save_ckpt_after_iter": 999_999,
                "stop_after_learner_train_calls": 1,
                "collect_search_backend": "stock",
                "collect_search_ctree_backend": "lightzero",
                "disable_death_for_profile": True,
                "opponent_death_mode": "normal",
                "policy_observation_backend": "cpu_oracle",
                "policy_trail_render_mode": "browser_lines",
                "policy_bonus_render_mode": "simple_symbols",
                "source_state_trail_render_mode": "browser_lines",
                "source_state_bonus_render_mode": "simple_symbols",
                "reward_variant": "survival_plus_outcome",
                "policy_observation_contract_id": (
                    "curvyzero_policy_observation_surface/v1"
                ),
                "observation_contract": {
                    "contract_id": "curvyzero_policy_observation_surface/v1",
                    "backend": "cpu_oracle",
                    "surface_label": "browser_lines+simple_symbols",
                    "stack_shape": [4, 64, 64],
                    "single_frame_shape": [1, 64, 64],
                    "target_frame_size": 64,
                    "source_frame_size": 704,
                    "grayscale_method": "BT.601_luma",
                    "downsample_method": "11x11_area_average_from_704_to_64",
                    "perspective_schema_id": (
                        "curvyzero_policy_observation_controlled_player_perspective/v1"
                    ),
                },
                "exploration_bonus": {"mode": "none"},
                "skip_lightzero_eval_in_profile": True,
                "background_eval_enabled": False,
                "background_gif_enabled": False,
            },
            "counts": {
                "env_steps_collected": 1024,
                "env_steps_collected_raw": 1024,
                "env_steps_collected_source": "collector_envstep_delta",
                "env_steps_collected_uses_fallback": False,
                "mcts_search_calls": 2,
                "mcts_search_root_sum": 1024,
                "learner_train_calls": 1,
                "replay_sample_calls": 1,
            },
            "derived": {
                "steps_per_sec": 100.0,
                "steps_per_sec_currency": (
                    "stock_train_muzero_profile_env_steps_per_sec"
                ),
                "steps_per_sec_source": "collector_envstep_delta",
                "steps_per_sec_uses_fallback_denominator": False,
                "mcts_root_batch_mean": 512.0,
            },
            "timers_sec": {
                "train_muzero_wall": 10.0,
                "collector_collect": 9.0,
                "policy_forward_collect": 6.0,
                "mcts_search": 2.0,
                "batched_profile_env_manager_step": 3.0,
                "batched_profile_renderer_render": 1.0,
                "batched_profile_surface_stack_update": 1.5,
            },
            "search_backend_proof": {
                "observed_collect_search_backends": ["stock"],
                "observed_collect_search_ctree_backends": ["lightzero"],
                "flat_payload_timer_present": False,
            },
            "gpu": {
                "requested_compute": "gpu-l4-t4-cpu40",
            },
        }
    }


def test_profile_attestation_accepts_fully_labeled_speed_row():
    assert summary._profile_attestation_problems(_attested_payload()) == []
    row_path = Path("row_000_result.json")
    row = summary._profile_row_from_payload(row_path, _attested_payload())
    assert row["speed_currency"] == "stock_train_muzero_profile_env_steps_per_sec"
    assert row["promotion_status"] == "gate_a_stock_baseline"
    assert row["stock_path_replaced"] == "none"


def test_profile_summary_marks_mcts_root_fallback_as_gate_a_ineligible():
    payload = _attested_payload()
    compact = payload["compact"]
    compact["counts"]["env_steps_collected"] = 1024
    compact["counts"]["env_steps_collected_raw"] = 0
    compact["counts"]["env_steps_collected_source"] = (
        "mcts_search_root_sum_profile_fallback"
    )
    compact["counts"]["env_steps_collected_uses_fallback"] = True
    compact["semantic_identity"]["env_steps_collected_source"] = (
        "mcts_search_root_sum_profile_fallback"
    )
    compact["semantic_identity"]["speed_currency"] = (
        "stock_train_muzero_profile_mcts_roots_per_sec_fallback"
    )
    compact["derived"]["steps_per_sec_currency"] = (
        "stock_train_muzero_profile_mcts_roots_per_sec_fallback"
    )
    compact["derived"]["steps_per_sec_source"] = (
        "mcts_search_root_sum_profile_fallback"
    )
    compact["derived"]["steps_per_sec_uses_fallback_denominator"] = True

    row = summary._profile_row_from_payload(Path("row_000_result.json"), payload)

    assert row["speed_currency"] == (
        "stock_train_muzero_profile_mcts_roots_per_sec_fallback"
    )
    assert row["promotion_status"] == "gate_a_ineligible_fallback_steps"
    assert row["stock_path_replaced"] == "none"


def test_profile_attestation_requires_speed_currency_fields_and_consistency():
    payload = _attested_payload()
    compact = payload["compact"]
    compact["derived"]["steps_per_sec_source"] = (
        "mcts_search_root_sum_profile_fallback"
    )
    compact["derived"]["steps_per_sec_currency"] = "wrong_currency"
    compact["derived"]["steps_per_sec_uses_fallback_denominator"] = True
    compact["semantic_identity"]["speed_currency"] = "other_wrong_currency"

    problems = summary._profile_attestation_problems(payload)

    assert "derived.steps_per_sec_source=counts" in problems
    assert "derived.steps_per_sec_currency=source" in problems
    assert "counts.env_steps_collected_uses_fallback=derived" in problems
    assert "semantic_identity.speed_currency=derived" in problems

    compact["counts"].pop("env_steps_collected_raw")
    compact["derived"].pop("steps_per_sec_currency")
    compact["semantic_identity"].pop("speed_currency")

    problems = summary._profile_attestation_problems(payload)

    assert "counts.env_steps_collected_raw" in problems
    assert "derived.steps_per_sec_currency" in problems
    assert "semantic_identity.speed_currency" in problems


def test_markdown_summary_includes_speed_currency_for_fallback_rows():
    payload = _attested_payload()
    compact = payload["compact"]
    compact["counts"]["env_steps_collected"] = 1024
    compact["counts"]["env_steps_collected_raw"] = 0
    compact["counts"]["env_steps_collected_source"] = (
        "mcts_search_root_sum_profile_fallback"
    )
    compact["counts"]["env_steps_collected_uses_fallback"] = True
    compact["semantic_identity"]["env_steps_collected_source"] = (
        "mcts_search_root_sum_profile_fallback"
    )
    compact["semantic_identity"]["speed_currency"] = (
        "stock_train_muzero_profile_mcts_roots_per_sec_fallback"
    )
    compact["derived"]["steps_per_sec_currency"] = (
        "stock_train_muzero_profile_mcts_roots_per_sec_fallback"
    )
    compact["derived"]["steps_per_sec_source"] = (
        "mcts_search_root_sum_profile_fallback"
    )
    compact["derived"]["steps_per_sec_uses_fallback_denominator"] = True

    row = summary._profile_row_from_payload(Path("row_000_result.json"), payload)
    table = summary._markdown_table([row])

    assert "currency" in table
    assert "stock_train_muzero_profile_mcts_roots_per_sec_fallback" in table


def test_profile_attestation_flags_under_labeled_speed_row():
    payload = _attested_payload()
    payload["compact"]["called_train_muzero"] = False
    payload["compact"]["command"].pop("policy_observation_backend")
    payload["compact"]["command"]["skip_lightzero_eval_in_profile"] = False
    payload["compact"]["command"]["observation_contract"].pop("grayscale_method")
    payload["compact"]["semantic_identity"].pop("consumer_semantics")
    payload["compact"]["counts"].pop("env_steps_collected_source")

    problems = summary._profile_attestation_problems(payload)

    assert "compact.called_train_muzero=true" in problems
    assert "command.policy_observation_backend" in problems
    assert "command.skip_lightzero_eval_in_profile=true" in problems
    assert "command.observation_contract.grayscale_method" in problems
    assert "semantic_identity.consumer_semantics" in problems
    assert "counts.env_steps_collected_source" in problems


def test_profile_attestation_requires_direct_backend_self_audit_fields():
    payload = _attested_payload()
    compact = payload["compact"]
    compact["command"]["collect_search_backend"] = "direct_ctree_gpu_latent"
    compact["semantic_identity"]["collect_search_backend"] = "direct_ctree_gpu_latent"
    compact["counts"].update(
        {
            "collect_search_backend_direct_ctree_gpu_latent_calls": 4,
            "collect_search_backend_fallback_calls": 0,
            "collect_search_backend_output_rows": 256,
        }
    )

    assert summary._profile_attestation_problems(payload) == []

    compact["counts"]["collect_search_backend_fallback_calls"] = 1
    compact["counts"]["collect_search_backend_direct_ctree_gpu_latent_calls"] = 0
    compact["counts"].pop("collect_search_backend_output_rows")
    compact["semantic_identity"]["collect_search_backend"] = "stock"
    compact["semantic_identity"]["collect_search_ctree_backend"] = "flat_a3"

    problems = summary._profile_attestation_problems(payload)

    assert "counts.collect_search_backend_fallback_calls=0" in problems
    assert "counts.collect_search_backend_direct_ctree_gpu_latent_calls>0" in problems
    assert "counts.collect_search_backend_output_rows" in problems
    assert "semantic_identity.collect_search_backend=command" in problems
    assert "semantic_identity.collect_search_ctree_backend=command" in problems


def test_profile_attestation_requires_flat_a3_runtime_proof():
    payload = _attested_payload()
    compact = payload["compact"]
    compact["command"]["collect_search_backend"] = "direct_ctree_gpu_latent"
    compact["semantic_identity"]["collect_search_backend"] = "direct_ctree_gpu_latent"
    compact["command"]["collect_search_ctree_backend"] = "flat_a3"
    compact["semantic_identity"]["collect_search_ctree_backend"] = "flat_a3"
    compact["counts"].update(
        {
            "collect_search_backend_direct_ctree_gpu_latent_calls": 4,
            "collect_search_backend_fallback_calls": 0,
            "collect_search_backend_output_rows": 256,
        }
    )

    problems = summary._profile_attestation_problems(payload)

    assert "search_backend_proof.observed_collect_search_ctree_backends=flat_a3" in problems
    assert "search_backend_proof.flat_payload_timer_present=true" in problems

    compact["search_backend_proof"] = {
        "observed_collect_search_backends": ["direct_ctree_gpu_latent"],
        "observed_collect_search_ctree_backends": ["flat_a3"],
        "flat_payload_timer_present": True,
    }

    assert summary._profile_attestation_problems(payload) == []


def test_gate_a_compare_accepts_matched_stock_and_candidate_rows():
    stock = _attested_payload()
    candidate = _attested_payload()
    candidate["row_id"] = "candidate"
    compact = candidate["compact"]
    compact["command"]["collect_search_backend"] = "direct_ctree_gpu_latent"
    compact["semantic_identity"]["collect_search_backend"] = "direct_ctree_gpu_latent"
    compact["counts"].update(
        {
            "collect_search_backend_direct_ctree_gpu_latent_calls": 4,
            "collect_search_backend_fallback_calls": 0,
            "collect_search_backend_output_rows": 256,
        }
    )

    assert summary._gate_a_comparison_problems([stock, candidate]) == []


def test_gate_a_compare_rejects_unmatched_denominator_rows():
    stock = _attested_payload()
    candidate = _attested_payload()
    candidate["row_id"] = "candidate"
    compact = candidate["compact"]
    compact["command"]["collect_search_backend"] = "direct_ctree_gpu_latent"
    compact["semantic_identity"]["collect_search_backend"] = "direct_ctree_gpu_latent"
    compact["command"]["batch_size"] = 128
    compact["counts"].update(
        {
            "collect_search_backend_direct_ctree_gpu_latent_calls": 4,
            "collect_search_backend_fallback_calls": 0,
            "collect_search_backend_output_rows": 256,
        }
    )

    problems = summary._gate_a_comparison_problems([stock, candidate])

    assert any("command.batch_size mismatch" in problem for problem in problems)
