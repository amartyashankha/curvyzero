import importlib.util
from pathlib import Path
import sys

import numpy as np

from curvyzero.training import debug_actor_loop_replay as replay
from curvyzero.training import replay_chunk_v0


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_vector_actor_loop_bridge.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_vector_actor_loop_bridge", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
actor_benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = actor_benchmark
_SPEC.loader.exec_module(actor_benchmark)

_P2_SAMPLE_PATHS = (
    "scenarios/environment/source_borderless_wrap_step.json",
    "scenarios/environment/source_normal_wall_death_step.json",
)


def test_actor_bridge_rollout_blocks_feed_synthetic_steps_and_report_chunks():
    summary = actor_benchmark.benchmark_inputs(
        _P2_SAMPLE_PATHS,
        body_capacity=4,
        batch_sizes=(2,),
        repeat=2,
        warmup=0,
        rollout_steps=3,
        hidden_dim=4,
        simulations=1,
        chunk_steps=4,
    )

    assert summary["summary"]["passed"] == summary["supported_fixture_count"]
    assert summary["summary"]["failed"] == 0
    assert summary["summary"]["batch_preflight_failed"] is False
    assert summary["summary"]["status"] in {"pass", "mixed"}
    for group in summary["groups"]:
        batch = group["batches"][0]
        counts = batch["counts"]
        sample = batch["sample"]
        training = batch["training_rate_report"]
        latency = batch["latency_sec"]
        amdahl = training["amdahl_breakdown_pct_loop"]

        assert batch["rollout_steps"] == 3
        assert counts["rollout_blocks"] == 2
        assert counts["state_reset_blocks"] == 2
        assert counts["state_reset_env_rows"] == 4
        assert counts["env_step_calls"] == 6
        assert counts["env_rows"] == 12
        assert counts["chunk_stage_calls"] == counts["env_step_calls"]
        assert counts["fixture_source_env_rows"] == 4
        assert counts["synthetic_feedback_env_rows"] == 8
        assert counts["policy_row_mapping_calls"] == counts["env_step_calls"]
        assert counts["policy_rows"] == counts["active_ego_rows"]
        assert counts["active_policy_rows"] == counts["active_ego_rows"]
        assert counts["padded_policy_rows"] == 0
        assert counts["synthetic_recurrent_model_calls"] == (
            counts["active_policy_rows"] * batch["simulations"]
        )
        assert training["schema"] == actor_benchmark.TRAINING_RATE_SCHEMA_VERSION
        assert training["counts"]["staged_transition_ego_rows"] == counts["replay_rows"]
        assert training["counts"]["state_reset_env_rows"] == counts["state_reset_env_rows"]
        assert training["timing_scope"].startswith("timed actor loop only")
        assert training["loop_rates"]["staged_transition_ego_rows_per_sec_total_loop"] > 0
        assert "fixture-reset" in training["completion_proxy"]
        assert amdahl["env_step"] > 0
        assert amdahl["synthetic_policy_search_total"] > 0
        assert amdahl["non_env_step"] > 0

        assert latency["schema"] == actor_benchmark.LATENCY_SCHEMA_VERSION
        assert latency["actor_step_total_sec"]["count"] == counts["env_step_calls"]
        assert latency["env_step_sec"]["count"] == counts["env_step_calls"]
        assert latency["synthetic_policy_search_total_sec"]["count"] == counts[
            "env_step_calls"
        ]
        assert latency["replay_chunk_stage_sec"]["count"] == counts["chunk_stage_calls"]
        for latency_key in (
            "actor_step_total_sec",
            "env_step_sec",
            "synthetic_policy_search_total_sec",
            "replay_chunk_stage_sec",
        ):
            latency_summary = latency[latency_key]
            assert latency_summary["p50"] > 0
            assert latency_summary["p50"] <= latency_summary["p95"]
            assert latency_summary["p95"] <= latency_summary["p99"]
            assert latency_summary["p99"] <= latency_summary["max"]

        assert sample["bytes_per_chunk"] > 0
        assert sample["policy_row_mapping_schema"] == (
            actor_benchmark.POLICY_ROW_MAPPING_SCHEMA
        )
        assert sample["policy_rows"] == sample["live_ego_count"]
        assert sample["active_policy_rows"] == sample["live_ego_count"]
        assert sample["padded_policy_rows"] == 0
        assert sample["replay_chunk_shapes"]["obs"] == [
            4,
            2,
            group["player_count"],
            len(actor_benchmark.debug_pack.DEBUG_OBS_FEATURE_NAMES),
        ]
        assert sample["replay_chunk_dtypes"]["action"] == "int8"


def test_actor_bridge_sample_reports_contract_schema_metadata():
    payload = actor_benchmark.build_fixture_seeded_actor_bridge_sample(
        _P2_SAMPLE_PATHS,
        body_capacity=4,
        batch_size=2,
        player_count=2,
        rollout_steps=2,
        hidden_dim=4,
        simulations=1,
    )

    source = payload["source"]
    metadata = source["sample_contract_metadata"]
    chunk_metadata = metadata["chunk_level_metadata"]

    assert metadata["schema"] == (
        "curvyzero_vector_actor_loop_bridge_sample_contract_metadata/v1"
    )
    assert metadata["status"] == "sample_metadata_only"
    assert metadata["sample_shape"] == {
        "batch_size": 2,
        "player_count": source["selected_group"]["player_count"],
        "obs_dim": len(actor_benchmark.debug_pack.DEBUG_OBS_FEATURE_NAMES),
        "action_count": actor_benchmark.ACTION_COUNT,
        "event_mode": actor_benchmark.batch_rows.EVENT_MODE_DEBUG,
    }
    assert chunk_metadata["replay_schema_id"] == (
        "curvyzero_debug_actor_loop_replay_chunk/v0"
    )
    assert chunk_metadata["observation_schema_id"] == (
        actor_benchmark.debug_pack.DEBUG_OBS_SCHEMA
    )
    assert chunk_metadata["action_space_id"] == "curvyzero_source_move_action_space/v0"
    assert chunk_metadata["reward_schema_id"] == (
        actor_benchmark.debug_pack.DEBUG_REWARD_SCHEMA
    )
    assert chunk_metadata["ruleset_id"] == "curvytron-v1-reference"
    assert len(chunk_metadata["replay_schema_hash"]) == 16
    assert len(chunk_metadata["rules_hash"]) == 16
    assert len(chunk_metadata["observation_schema_hash"]) == 16
    assert len(chunk_metadata["action_space_hash"]) == 16
    assert len(chunk_metadata["reward_schema_hash"]) == 16
    assert chunk_metadata["created_at"] is None
    assert chunk_metadata["created_at_status"] == "omitted_for_deterministic_sample"
    assert "not a full source rules hash" in chunk_metadata["rules_hash_scope"]
    assert source["sample_replay_chunk"]["status"] == "not_requested"
    assert source["sample"]["policy_row_mapping_schema"] == (
        actor_benchmark.POLICY_ROW_MAPPING_SCHEMA
    )


def test_actor_bridge_sample_can_write_debug_replay_chunk(tmp_path):
    path = tmp_path / "actor-bridge-sample-chunk.npz"

    payload = actor_benchmark.build_fixture_seeded_actor_bridge_sample(
        _P2_SAMPLE_PATHS,
        body_capacity=4,
        batch_size=2,
        player_count=2,
        rollout_steps=2,
        hidden_dim=4,
        simulations=1,
        replay_chunk_path=path,
    )

    source = payload["source"]
    report = source["sample_replay_chunk"]
    selected_group = source["selected_group"]

    assert report["status"] == "written"
    assert report["path"] == str(path)
    assert report["metadata_schema_id"] == replay.REPLAY_METADATA_SCHEMA_ID
    assert report["replay_schema_id"] == replay.REPLAY_SCHEMA_ID
    assert report["write_elapsed_sec"] > 0.0
    assert report["file_bytes"] == path.stat().st_size
    assert report["write_mb_per_sec"] is not None
    assert report["write_mb_per_sec"] > 0.0

    loaded = replay.read_debug_actor_loop_replay_chunk(
        path,
        expected_metadata=report["compatibility_metadata"],
    )

    assert loaded.metadata["producer"] == actor_benchmark.BENCHMARK_ID
    assert loaded.metadata["chunk_steps"] == 2
    assert loaded.metadata["batch_size"] == 2
    assert loaded.metadata["player_count"] == selected_group["player_count"]
    assert loaded.metadata["obs_dim"] == len(actor_benchmark.debug_pack.DEBUG_OBS_FEATURE_NAMES)
    assert loaded.metadata["action_count"] == actor_benchmark.ACTION_COUNT
    assert loaded.arrays["obs"].shape == (
        2,
        2,
        selected_group["player_count"],
        len(actor_benchmark.debug_pack.DEBUG_OBS_FEATURE_NAMES),
    )
    assert loaded.arrays["action"].dtype == np.dtype("int8")
    np.testing.assert_array_equal(
        loaded.arrays["ego_mask"][-1],
        np.asarray(payload["surfaces"]["ego_mask"], dtype=bool),
    )


def test_actor_bridge_sample_can_write_replay_v0_debug_payload_chunk(tmp_path):
    path = tmp_path / "actor-bridge-sample-replay-v0-chunk.npz"

    payload = actor_benchmark.build_fixture_seeded_actor_bridge_sample(
        _P2_SAMPLE_PATHS,
        body_capacity=4,
        batch_size=2,
        player_count=2,
        rollout_steps=2,
        hidden_dim=4,
        simulations=1,
        replay_v0_chunk_path=path,
    )

    source = payload["source"]
    report = source["sample_replay_v0_chunk"]
    selected_group = source["selected_group"]

    assert report["status"] == "written"
    assert report["path"] == str(path)
    assert report["metadata_schema_id"] == replay_chunk_v0.REPLAY_METADATA_SCHEMA_ID
    assert report["replay_contract_id"] == replay_chunk_v0.REPLAY_CONTRACT_ID
    assert report["replay_schema_id"] == replay_chunk_v0.REPLAY_SCHEMA_ID
    assert report["production_training_decision"] == "blocked"
    assert "debug" in report["blocked_reason"]
    assert report["write_elapsed_sec"] > 0.0
    assert report["file_bytes"] == path.stat().st_size
    assert report["write_mb_per_sec"] is not None
    assert report["write_mb_per_sec"] > 0.0

    loaded = replay_chunk_v0.read_replay_chunk_v0(
        path,
        expected_metadata=report["compatibility_metadata"],
    )

    assert loaded.metadata["producer"] == actor_benchmark.BENCHMARK_ID
    assert loaded.metadata["chunk_steps"] == 2
    assert loaded.metadata["batch_size"] == 2
    assert loaded.metadata["player_count"] == 2
    assert loaded.metadata["obs_dim"] == len(actor_benchmark.debug_pack.DEBUG_OBS_FEATURE_NAMES)
    assert loaded.metadata["action_count"] == actor_benchmark.ACTION_COUNT
    assert loaded.metadata["observation_schema_id"] == actor_benchmark.debug_pack.DEBUG_OBS_SCHEMA
    assert loaded.metadata["reward_schema_id"] == actor_benchmark.debug_pack.DEBUG_REWARD_SCHEMA
    assert loaded.arrays["observation"].shape == (
        2,
        2,
        selected_group["player_count"],
        len(actor_benchmark.debug_pack.DEBUG_OBS_FEATURE_NAMES),
    )
    assert loaded.arrays["action"].dtype == np.dtype("int16")
    assert loaded.arrays["episode_id"].dtype.kind == "U"
    assert loaded.arrays["reset_source"].dtype.kind == "U"
    np.testing.assert_array_equal(
        loaded.arrays["done"],
        np.logical_or(loaded.arrays["terminated"], loaded.arrays["truncated"]),
    )


def test_actor_bridge_compares_debug_event_and_no_event_modes():
    summary = actor_benchmark.benchmark_inputs(
        _P2_SAMPLE_PATHS,
        body_capacity=4,
        batch_sizes=(2,),
        event_modes=(
            actor_benchmark.batch_rows.EVENT_MODE_DEBUG,
            actor_benchmark.batch_rows.EVENT_MODE_NONE,
        ),
        repeat=1,
        warmup=0,
        rollout_steps=2,
        hidden_dim=4,
        simulations=1,
        chunk_steps=4,
    )

    assert summary["summary"]["failed"] == 0
    assert summary["summary"]["batch_preflight_failed"] is False
    for group in summary["groups"]:
        assert group["preflight"]["event_match"] is True
        assert group["no_event_preflight"]["state_match"] is True
        assert len(group["event_mode_comparisons"]) == 1

        by_mode = {batch["event_mode"]: batch for batch in group["batches"]}
        debug_batch = by_mode[actor_benchmark.batch_rows.EVENT_MODE_DEBUG]
        no_event_batch = by_mode[actor_benchmark.batch_rows.EVENT_MODE_NONE]

        assert debug_batch["counts"]["env_events_emitted"] > 0
        assert no_event_batch["counts"]["env_events_emitted"] == 0
        assert debug_batch["counts"]["active_ego_rows"] == no_event_batch["counts"][
            "active_ego_rows"
        ]
        assert debug_batch["sample"]["checksum"] == no_event_batch["sample"]["checksum"]
        assert no_event_batch["timing_sec"]["env_event_overhead_sec"] == 0


def test_actor_bridge_autoresets_terminal_rows_after_replay_staging():
    summary = actor_benchmark.benchmark_inputs(
        ("scenarios/environment/source_normal_wall_death_step.json",),
        body_capacity=4,
        batch_sizes=(2,),
        repeat=1,
        warmup=0,
        rollout_steps=2,
        hidden_dim=4,
        simulations=1,
        chunk_steps=4,
    )

    assert summary["summary"]["failed"] == 0
    assert summary["summary"]["batch_preflight_failed"] is False
    batch = summary["groups"][0]["batches"][0]
    counts = batch["counts"]
    training = batch["training_rate_report"]

    assert counts["done_rows"] > 0
    assert counts["autoreset_rows"] > 0
    assert counts["final_transition_rows_before_autoreset"] == counts["autoreset_rows"]
    assert training["counts"]["final_transition_env_rows"] == counts[
        "final_transition_rows_before_autoreset"
    ]
    assert training["loop_rates"]["final_transition_env_rows_per_min_total_loop"] > 0
    assert training["loop_rates"]["completed_game_rows_per_min_total_loop"] > 0
    assert counts["chunk_stage_calls"] == counts["env_step_calls"]
    assert counts["replay_rows"] == (
        counts["env_step_calls"] * batch["batch_size"] * summary["groups"][0]["player_count"]
    )
