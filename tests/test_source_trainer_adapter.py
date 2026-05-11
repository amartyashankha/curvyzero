import numpy as np
import pytest

from curvyzero.env.source_env import CurvyTronSourceEnv
from curvyzero.env.source_env import SourceBodyState
from curvyzero.env.source_env import SourceWorldState
from curvyzero.env.source_trainer_adapter import SourceTrainerAdapterError
from curvyzero.env.source_trainer_adapter import config_from_source_snapshot
from curvyzero.env.source_trainer_adapter import source_snapshot_player_ids
from curvyzero.env.source_trainer_adapter import source_snapshot_to_env_state
from curvyzero.env.source_trainer_adapter import source_snapshot_to_vector_trainer_state
from curvyzero.env.trainer_observation import observe_1v1_egocentric_rays_v0
from curvyzero.env.vector_trainer_observation import observe_vector_1v1_egocentric_rays_v0


def test_source_snapshot_to_env_state_exposes_source_positions_for_trainer_shape():
    env = CurvyTronSourceEnv(random_constant=0.5)
    snapshot = env.reset(player_count=2, warmup_ms=0)

    config = config_from_source_snapshot(snapshot)
    state = source_snapshot_to_env_state(snapshot, config)
    player_ids = source_snapshot_player_ids(snapshot)

    assert config.width == int(snapshot["game"]["size"])
    assert config.height == int(snapshot["game"]["size"])
    assert config.players == 2
    assert player_ids == ("p0", "p1")
    assert state.positions.shape == (2, 2)
    assert state.headings.shape == (2,)
    assert state.alive.tolist() == [True, True]
    assert state.positions.tolist() == [
        [float(avatar["x"]), float(avatar["y"])] for avatar in snapshot["avatars"]
    ]
    np.testing.assert_allclose(
        state.headings,
        np.asarray([float(avatar["angle"]) for avatar in snapshot["avatars"]], dtype=np.float32),
    )
    assert state.occupancy.shape == (config.height, config.width)
    assert not np.any(state.occupancy)

    batch = observe_1v1_egocentric_rays_v0(state, config, player_ids=player_ids)
    assert batch.observation.shape == (2, 106)
    assert batch.action_mask.shape == (2, 3)


def test_source_snapshot_to_env_state_rejects_non_empty_occupancy_policy():
    env = CurvyTronSourceEnv(random_constant=0.5)
    snapshot = env.reset(player_count=2, warmup_ms=0)
    config = config_from_source_snapshot(snapshot)

    with pytest.raises(SourceTrainerAdapterError, match="unsupported occupancy_policy"):
        source_snapshot_to_env_state(snapshot, config, occupancy_policy="source_world")


def test_source_snapshot_to_env_state_can_mark_source_body_center_cell():
    env = CurvyTronSourceEnv(random_constant=0.5)
    snapshot = env.reset(player_count=2, warmup_ms=0)
    config = config_from_source_snapshot(snapshot)
    avatar = snapshot["avatars"][0]
    body = {
        "id": 0,
        "x": avatar["x"],
        "y": avatar["y"],
        "radius": 0.6,
        "avatarId": avatar["id"],
        "num": 0,
        "birthMs": 0.0,
        "trailLatency": 3,
    }

    state = source_snapshot_to_env_state(
        snapshot,
        config,
        occupancy_policy="source_world_bodies_center_cell_v0",
        world_bodies=[body],
    )

    x = int(round(float(avatar["x"])))
    y = int(round(float(avatar["y"])))
    assert state.occupancy[y, x] == 1


def test_source_env_exposes_body_metadata_sidecar_for_circle_observation_adapter():
    env = CurvyTronSourceEnv(random_constant=0.5)
    env.reset(player_count=2, warmup_ms=0)
    avatar = env.avatar_body_metadata_snapshot()[0]

    assert "bodyNum" in avatar
    assert "bodyCount" in avatar
    assert "radius" in avatar
    assert "trailLatency" in avatar


def test_source_snapshot_to_vector_trainer_state_uses_source_circle_latency():
    snapshot = {
        "atMs": 0.0,
        "game": {"size": 64, "borderless": False},
        "avatars": [
            {
                "id": 1,
                "name": "p0",
                "x": 10.0,
                "y": 10.0,
                "angle": 0.0,
                "alive": True,
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
                "alive": True,
                "bodyNum": 0,
                "bodyCount": 0,
                "radius": 1.0,
                "trailLatency": 3,
            },
        ],
    }
    config = config_from_source_snapshot(snapshot)
    world_bodies = [
        {
            "id": 1,
            "x": 14.0,
            "y": 10.0,
            "radius": 1.0,
            "avatarId": 1,
            "num": 9,
            "birthMs": 0.0,
            "trailLatency": 3,
        },
        {
            "id": 2,
            "x": 18.0,
            "y": 10.0,
            "radius": 1.0,
            "avatarId": 1,
            "num": 6,
            "birthMs": 0.0,
            "trailLatency": 3,
        },
    ]

    vector_state = source_snapshot_to_vector_trainer_state(
        snapshot,
        config,
        world_bodies=world_bodies,
        avatar_body_metadata=None,
        decision_ms=1000.0 / 60.0,
    )
    np.testing.assert_array_equal(vector_state["body_write_cursor"], np.asarray([2], dtype=np.int32))
    batch = observe_vector_1v1_egocentric_rays_v0(
        vector_state,
        0,
        player_ids=("p0", "p1"),
        decision_ms=1000.0 / 60.0,
        max_ticks=config.max_ticks,
    )

    arena_diagonal = np.hypot(64.0, 64.0)
    np.testing.assert_allclose(batch.rays[0, 0, 1], 7.0 / arena_diagonal, rtol=1e-6)


def test_source_world_state_iter_unique_bodies_deduplicates_multi_island_body():
    world = SourceWorldState(size=88, island_count=2)
    world.activate()
    body = SourceBodyState(x=44.0, y=44.0, radius=4.0, avatar_id=1)
    world.add_body(body)

    duplicated_slots = sum(
        1 for island in world.islands.values() for stored in island.bodies if stored is body
    )
    assert duplicated_slots > 1
    assert world.iter_unique_bodies() == (body,)


def test_source_snapshot_to_env_state_requires_alive_field():
    env = CurvyTronSourceEnv(random_constant=0.5)
    snapshot = env.reset(player_count=2, warmup_ms=0)
    del snapshot["avatars"][0]["alive"]
    config = config_from_source_snapshot(snapshot)

    with pytest.raises(SourceTrainerAdapterError, match="avatar.alive"):
        source_snapshot_to_env_state(snapshot, config)
