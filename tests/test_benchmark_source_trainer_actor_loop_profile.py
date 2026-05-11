import importlib.util
from pathlib import Path
import sys

import numpy as np

from curvyzero.env.source_trainer_adapter import config_from_source_snapshot
from curvyzero.env.source_trainer_adapter import source_snapshot_to_vector_trainer_state
from curvyzero.env.vector_trainer_observation import (
    observe_vector_1v1_egocentric_rays_batch_arrays_v0,
)
from curvyzero.env.vector_trainer_observation import observe_vector_1v1_egocentric_rays_v0


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark_source_trainer_actor_loop_profile.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "benchmark_source_trainer_actor_loop_profile",
    _SCRIPT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
profile_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = profile_script
_SPEC.loader.exec_module(profile_script)


def _circle_ray_snapshot(*, p0_alive: bool = True, p1_alive: bool = True):
    return {
        "atMs": 0.0,
        "game": {"size": 64, "borderless": False},
        "avatars": [
            {
                "id": 1,
                "name": "p0",
                "x": 10.0,
                "y": 10.0,
                "angle": 0.0,
                "alive": p0_alive,
                "bodyNum": 10,
                "bodyCount": 11,
                "radius": 1.0,
                "trailLatency": 3,
            },
            {
                "id": 2,
                "name": "p1",
                "x": 40.0,
                "y": 10.0,
                "angle": np.pi,
                "alive": p1_alive,
                "bodyNum": 7,
                "bodyCount": 8,
                "radius": 1.0,
                "trailLatency": 3,
            },
        ],
    }


def test_source_circle_ray_batch_stack_matches_scalar_rows_with_padding():
    snapshot = _circle_ray_snapshot()
    config = config_from_source_snapshot(snapshot)
    short_state = source_snapshot_to_vector_trainer_state(
        snapshot,
        config,
        world_bodies=[
            {
                "id": 1,
                "x": 14.0,
                "y": 10.0,
                "radius": 1.0,
                "avatarId": 1,
                "num": 6,
            },
        ],
        decision_ms=1000.0 / 60.0,
    )
    long_state = source_snapshot_to_vector_trainer_state(
        snapshot,
        config,
        world_bodies=[
            {
                "id": 1,
                "x": 14.0,
                "y": 10.0,
                "radius": 1.0,
                "avatarId": 1,
                "num": 6,
            },
            {
                "id": 2,
                "x": 36.0,
                "y": 10.0,
                "radius": 1.0,
                "avatarId": 2,
                "num": 3,
            },
        ],
        decision_ms=1000.0 / 60.0,
    )

    batched_state = profile_script._stack_vector_trainer_row_states([short_state, long_state])
    observation, action_mask, lightzero_action_mask, to_play = (
        observe_vector_1v1_egocentric_rays_batch_arrays_v0(
            batched_state,
            player_ids=("p0", "p1"),
            decision_ms=1000.0 / 60.0,
            max_ticks=config.max_ticks,
        )
    )

    assert batched_state["body_active"].shape == (2, 2)
    assert batched_state["body_active"][0].tolist() == [True, False]
    assert batched_state["body_owner"][0].tolist() == [0, -1]
    np.testing.assert_array_equal(
        batched_state["body_write_cursor"],
        np.asarray([1, 2], dtype=np.int32),
    )
    for row, state in enumerate((short_state, long_state)):
        scalar = observe_vector_1v1_egocentric_rays_v0(
            state,
            0,
            player_ids=("p0", "p1"),
            decision_ms=1000.0 / 60.0,
            max_ticks=config.max_ticks,
        )
        np.testing.assert_allclose(observation[row], scalar.observation)
        np.testing.assert_array_equal(action_mask[row], scalar.action_mask)
        np.testing.assert_array_equal(lightzero_action_mask[row], scalar.lightzero_action_mask)
        np.testing.assert_array_equal(to_play[row], scalar.to_play)


def test_source_circle_ray_batch_reward_helper_matches_scalar_terminal_row():
    live_snapshot = _circle_ray_snapshot()
    terminal_snapshot = _circle_ray_snapshot(p0_alive=False, p1_alive=True)
    config = config_from_source_snapshot(live_snapshot)
    states = [
        source_snapshot_to_vector_trainer_state(
            live_snapshot,
            config,
            world_bodies=[],
            decision_ms=1000.0 / 60.0,
        ),
        source_snapshot_to_vector_trainer_state(
            terminal_snapshot,
            config,
            world_bodies=[],
            decision_ms=1000.0 / 60.0,
        ),
    ]

    batched_state = profile_script._stack_vector_trainer_row_states(states)
    rewards, done, terminated, truncated = profile_script._vector_trainer_rewards_and_done(
        batched_state
    )

    for row, state in enumerate(states):
        scalar = observe_vector_1v1_egocentric_rays_v0(
            state,
            0,
            player_ids=("p0", "p1"),
            decision_ms=1000.0 / 60.0,
            max_ticks=config.max_ticks,
        )
        np.testing.assert_array_equal(rewards[row], scalar.rewards)
        assert bool(done[row]) is scalar.done
        assert bool(terminated[row]) is scalar.terminated
        assert bool(truncated[row]) is scalar.truncated


def test_source_trainer_actor_loop_profile_writes_read_validated_replay(tmp_path):
    report = profile_script.run_profile(
        batch_size=1,
        rollout_steps=1,
        step_ms=1000.0 / 60.0,
        warmup_ms=0.0,
        seed=123,
        hidden_dim=8,
        occupancy_policy="empty",
        artifact_root=tmp_path,
    )

    assert report["schema_id"] == "curvyzero_source_trainer_actor_loop_profile/v0"
    assert report["lane"] == "source_trainer_actor_loop_profile"
    assert report["contracts"]["source_derived_fields"] == ["positions", "headings", "alive"]
    assert report["contracts"]["occupancy_policy"] == "empty"
    assert report["contracts"]["occupancy_source_fields"] == []
    assert report["contracts"]["approximate_fields"] == ["occupancy"]
    assert report["loop_shape"]["observation_shape"] == [1, 1, 2, 106]
    assert report["replay_or_rollout"]["field_specs"]["reset_source"]["dtype"] != "<U1"
    assert report["denominators"]["env_transitions"] == 1
    assert report["integrity"]["replay_schema_read_validated"] is True
    assert "terminal_rows_with_final_observation" not in report["integrity"]
    assert report["integrity"]["replay_semantic_validation"] == "not performed"
    assert report["integrity"]["done_terminated_truncated_invariant_failures"] == 0
    assert "policy_action" in report["latency_sec"]
    assert report["observation_phase_profile"]["enabled"] is False
    assert report["source_body_trail"]["sample_count"] == 2
    assert report["source_body_trail"]["world_bodies_count"]["max"] == 0
    assert report["source_body_trail"]["occupancy_nonzero_count"]["max"] == 0
    assert report["source_body_trail"]["occupancy_owner_cell_count"]["p0"]["max"] == 0
    assert report["source_body_trail"]["occupancy_owner_cell_count"]["p1"]["max"] == 0
    assert any("occupancy is empty" in caveat for caveat in report["caveats"])
    assert Path(report["artifacts"]["report_json"]).exists()
    assert "profile_report_json" not in report["artifacts"]
    assert Path(report["artifacts"]["replay_v0_chunk"]).exists()


def test_source_trainer_actor_loop_profile_can_use_center_cell_source_body_occupancy(tmp_path):
    report = profile_script.run_profile(
        batch_size=1,
        rollout_steps=2,
        step_ms=1000.0 / 60.0,
        warmup_ms=0.0,
        seed=123,
        hidden_dim=8,
        occupancy_policy="source_world_bodies_center_cell_v0",
        source_setup_mode="controlled_trail",
        profile_observation_phases=True,
        artifact_root=tmp_path,
    )

    assert report["contracts"]["source_derived_fields"] == ["positions", "headings", "alive"]
    assert report["contracts"]["occupancy_policy"] == "source_world_bodies_center_cell_v0"
    assert report["contracts"]["occupancy_source_fields"] == ["world_bodies"]
    assert report["contracts"]["approximate_fields"] == ["occupancy"]
    assert report["source_body_trail"]["sample_count"] == 4
    assert report["source_body_trail"]["world_bodies_count"]["count"] == 4
    assert report["source_body_trail"]["occupancy_nonzero_count"]["count"] == 4
    assert report["source_body_trail"]["nonempty_world_body_samples"] > 0
    assert report["source_body_trail"]["nonempty_occupancy_samples"] > 0
    assert report["observation_phase_profile"]["enabled"] is True
    assert report["observation_phase_profile"]["timing_sec"]["ray_cast_sec"] > 0.0
    assert set(report["source_body_trail"]["occupancy_owner_cell_count"]) == {"p0", "p1"}
    assert any("center-cell source body raster" in caveat for caveat in report["caveats"])


def test_source_trainer_actor_loop_profile_can_use_source_body_circle_rays(tmp_path):
    report = profile_script.run_profile(
        batch_size=1,
        rollout_steps=2,
        step_ms=1000.0 / 60.0,
        warmup_ms=0.0,
        seed=123,
        hidden_dim=8,
        occupancy_policy="source_world_bodies_circle_rays_v0",
        source_setup_mode="controlled_trail",
        profile_observation_phases=True,
        artifact_root=tmp_path,
    )

    assert report["contracts"]["occupancy_policy"] == "source_world_bodies_circle_rays_v0"
    assert report["contracts"]["occupancy_source_fields"] == [
        "world_bodies",
        "avatar.bodyNum",
        "avatar.trailLatency",
        "avatar.radius",
    ]
    assert report["contracts"]["approximate_fields"] == [
        "visible_trail_history",
        "bonus_world_bodies",
    ]
    assert report["source_body_trail"]["sample_count"] == 4
    assert report["source_body_trail"]["nonempty_world_body_samples"] > 0
    assert report["source_body_trail"]["nonempty_body_circle_samples"] > 0
    assert report["source_body_trail"]["nonempty_occupancy_samples"] == 0
    assert report["observation_phase_profile"]["enabled"] is True
    assert report["observation_phase_profile"]["timing_sec"]["ray_cast_sec"] > 0.0
    assert report["integrity"]["done_terminated_truncated_invariant_failures"] == 0
    assert any("source world body circles" in caveat for caveat in report["caveats"])
