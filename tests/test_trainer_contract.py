from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from curvyzero.env import CurvyTronConfig, CurvyTronEnv
from curvyzero.env.state import EnvState
from curvyzero.env import trainer_contract as contract
from curvyzero.env.trainer_observation import flatten_structured_observation_v0
from curvyzero.env.trainer_observation import observe_1v1_egocentric_rays_v0
from curvyzero.env.trainer_observation import observe_egocentric_rays_v0


_ROOT = Path(__file__).resolve().parents[1]
_EMPTY_GEOMETRY_MANIFEST = (
    _ROOT / "scenarios" / "environment" / "observation" / "obs_empty_arena_geometry_v0.json"
)
_SOURCE_MOVEMENT_MANIFEST = (
    _ROOT
    / "scenarios"
    / "environment"
    / "observation"
    / "obs_source_movement_empty_multistep_v0.json"
)


def _load_empty_geometry_manifest() -> dict:
    return json.loads(_EMPTY_GEOMETRY_MANIFEST.read_text(encoding="utf-8"))


def _load_source_movement_manifest() -> dict:
    return json.loads(_SOURCE_MOVEMENT_MANIFEST.read_text(encoding="utf-8"))


def _config_from_manifest(manifest: dict) -> CurvyTronConfig:
    return CurvyTronConfig(**manifest["config"])


def _empty_occupancy(config: CurvyTronConfig) -> np.ndarray:
    return np.zeros((config.height, config.width), dtype=np.int16)


def _state_from_manifest(manifest: dict, state_id: str) -> EnvState:
    config = _config_from_manifest(manifest)
    state = manifest["states"][state_id]
    occupancy = _empty_occupancy(config)
    for cell in state["occupancy_nonzero"]:
        occupancy[int(cell["y"]), int(cell["x"])] = int(cell["owner"])
    return EnvState(
        tick=int(state["tick"]),
        positions=np.asarray(state["positions"], dtype=np.float32),
        headings=np.asarray(state["headings"], dtype=np.float32),
        alive=np.asarray(state["alive"], dtype=np.bool_),
        death_tick=np.full(config.players, -1, dtype=np.int32),
        occupancy=occupancy,
        rng=np.random.default_rng(1),
    )


def _load_source_scenario_for_manifest(manifest: dict) -> dict:
    scenario_path = _ROOT / manifest["state_source"]["source_scenario"]
    return json.loads(scenario_path.read_text(encoding="utf-8"))


def _source_expected_frames_by_tick(manifest: dict) -> dict[int, dict]:
    scenario = _load_source_scenario_for_manifest(manifest)
    return {
        int(frame["tick"]): frame
        for frame in scenario["comparison"]["expected"]["frames"]
    }


def _state_from_source_expected_frame(manifest: dict, frame: dict) -> EnvState:
    config = _config_from_manifest(manifest)
    return EnvState(
        tick=int(frame["tick"]),
        positions=np.asarray(frame["positions"], dtype=np.float32),
        headings=np.asarray(frame["headings"], dtype=np.float32),
        alive=np.asarray(frame["alive"], dtype=np.bool_),
        death_tick=np.full(config.players, -1, dtype=np.int32),
        occupancy=_empty_occupancy(config),
        rng=np.random.default_rng(1),
    )


def test_trainer_observation_schema_pins_first_non_debug_target():
    assert contract.OBSERVATION_SCHEMA_ID == "curvyzero_egocentric_rays/v0"
    assert contract.LEGACY_OBSERVATION_SCHEMA_IDS == ("curvyzero-observe-v0-rays",)
    assert contract.STRUCTURED_OBSERVATION_SHAPE == {
        "rays": (24, 4),
        "scalars": (10,),
    }
    assert contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE == (106,)
    assert contract.RAY_ANGLES_DEGREES == tuple(range(0, 360, 15))
    assert contract.RAY_CHANNEL_NAMES == (
        "wall_or_out_of_bounds",
        "own_trail",
        "opponent_trail",
        "opponent_head",
    )
    assert contract.SCALAR_NAMES == (
        "ego_alive",
        "opponent_alive",
        "tick_fraction",
        "opponent_rel_x_clipped",
        "opponent_rel_y_clipped",
        "opponent_heading_sin_relative",
        "opponent_heading_cos_relative",
        "speed_norm",
        "turn_rate_norm",
        "trail_radius_norm",
    )
    assert contract.OBSERVATION_SCHEMA["action_mask_is_separate"] is True
    assert contract.OBSERVATION_SCHEMA_HASH == "61767187ffa4a3a6"


def test_action_mask_reward_and_step_contract_are_stable():
    assert contract.ACTION_SPACE_ID == "curvyzero_turn3/v0"
    assert contract.ACTION_NAMES == ("left", "straight", "right")
    assert contract.ACTION_ID_TO_SOURCE_MOVE == (-1, 0, 1)
    assert (
        contract.ACTION_SPACE_SCHEMA["native_control_model_id"]
        == "curvytron_realtime_controls_elapsed_frames/v0"
    )
    assert (
        contract.ACTION_SPACE_SCHEMA["trainer_control_wrapper_id"]
        == "curvyzero_fixed_decision_wrapper/v0"
    )
    assert "elapsed milliseconds" in contract.ACTION_SPACE_SCHEMA["native_control_model"]
    assert (
        "trainer-style batched action step"
        in contract.ACTION_SPACE_SCHEMA["native_control_model"]
    )
    assert "fixed decision cadence" in contract.ACTION_SPACE_SCHEMA["trainer_control_wrapper"]
    assert contract.legal_action_mask(active=True) == (True, True, True)
    assert contract.legal_action_mask(active=True, allow_straight=False) == (
        True,
        False,
        True,
    )
    assert contract.legal_action_mask(active=False) == (False, False, False)
    assert contract.ACTION_SPACE_HASH == "957cf262e9a3fb1f"

    reward_schema = contract.REWARD_SCHEMA
    assert reward_schema["schema_id"] == "curvyzero_sparse_round_outcome/v0"
    assert reward_schema["alignment"] == "reward_t_plus_1_after_wrapper_decision_t"
    assert "wrapper decision boundary" in reward_schema["alignment_note"]
    assert reward_schema["nonterminal_reward"] == 0.0
    assert reward_schema["winner_reward"] == 1.0
    assert reward_schema["loser_reward"] == -1.0
    assert reward_schema["all_dead_draw_reward"] == 0.0
    assert reward_schema["pure_truncation_reward"] == 0.0
    assert reward_schema["shaping_terms"] == ()
    assert contract.REWARD_SCHEMA_HASH == "0ab8bebd84fcb2c5"

    step_returns = contract.TRAINER_ADAPTER_CONTRACT["step_returns"]
    assert (
        contract.TRAINER_ADAPTER_CONTRACT["native_control_model_id"]
        == "curvytron_realtime_controls_elapsed_frames/v0"
    )
    assert (
        contract.TRAINER_ADAPTER_CONTRACT["trainer_control_wrapper_id"]
        == "curvyzero_fixed_decision_wrapper/v0"
    )
    assert step_returns["done"] == "terminated OR truncated"
    for key in (
        "terminal_reason",
        "winner_ids",
        "loser_ids",
        "death_player_ids",
        "draw",
        "timeout",
        "truncation_reason",
        "done",
        "terminated",
        "truncated",
        "final_observation",
        "final_reward_map",
        "trace_hash",
    ):
        assert key in contract.STEP_INFO_KEYS
    assert "eval_episode_return" in contract.LIGHTZERO_TERMINAL_INFO_KEYS
    assert contract.TRAINER_ADAPTER_CONTRACT_HASH == "c25810c9cc197d27"


def test_schema_hash_changes_when_behavioral_shape_changes():
    changed = deepcopy(contract.OBSERVATION_SCHEMA)
    changed["structured_shape"]["rays"] = (32, 4)

    assert contract.stable_contract_hash(changed) != contract.OBSERVATION_SCHEMA_HASH


def test_observe_egocentric_rays_v0_returns_pinned_shapes_dtypes_and_masks():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    env.reset(seed=7)
    assert env.state is not None

    observed = observe_egocentric_rays_v0(env.state, env.config, "player_0", player_ids=env.agents)

    assert observed.rays.shape == (24, 4)
    assert observed.rays.dtype == np.float32
    assert observed.scalars.shape == (10,)
    assert observed.scalars.dtype == np.float32
    assert observed.observation.shape == (106,)
    assert observed.observation.dtype == np.float32
    assert observed.action_mask.dtype == np.bool_
    np.testing.assert_array_equal(observed.action_mask, np.array([True, True, True]))
    assert observed.lightzero_action_mask.dtype == np.int8
    np.testing.assert_array_equal(observed.lightzero_action_mask, np.array([1, 1, 1]))
    assert observed.to_play == -1
    assert np.isfinite(observed.observation).all()
    assert np.logical_and(observed.rays >= 0.0, observed.rays <= 1.0).all()
    assert observed.reward == np.float32(0.0)
    assert observed.reward_info["done"] is False
    assert observed.reward_info["reward_schema_id"] == contract.REWARD_SCHEMA_ID
    assert observed.reward_info["observation_schema_hash"] == contract.OBSERVATION_SCHEMA_HASH
    np.testing.assert_array_equal(
        observed.observation,
        flatten_structured_observation_v0(observed.rays, observed.scalars),
    )
    assert contract.OBSERVATION_SCHEMA_HASH == "61767187ffa4a3a6"


def test_observe_egocentric_rays_v0_uses_strict_left_right_mask_when_requested():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1, allow_straight_action=False))
    env.reset(seed=7)
    assert env.state is not None

    observed = observe_egocentric_rays_v0(env.state, env.config, "player_0", player_ids=env.agents)

    np.testing.assert_array_equal(
        observed.action_mask,
        np.array([True, False, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        observed.lightzero_action_mask,
        np.array([1, 0, 1], dtype=np.int8),
    )


def test_observe_egocentric_rays_v0_is_pure_and_returns_fresh_arrays():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    env.reset(seed=7)
    assert env.state is not None
    positions_before = env.state.positions.copy()
    occupancy_before = env.state.occupancy.copy()

    observed = observe_egocentric_rays_v0(env.state, env.config, "player_0", player_ids=env.agents)
    observed.rays[0, 0] = -999.0
    observed.scalars[0] = -999.0
    observed.observation[0] = -999.0
    observed.action_mask[0] = False
    observed.lightzero_action_mask[1] = 0

    again = observe_egocentric_rays_v0(env.state, env.config, "player_0", player_ids=env.agents)

    np.testing.assert_array_equal(env.state.positions, positions_before)
    np.testing.assert_array_equal(env.state.occupancy, occupancy_before)
    assert again.rays[0, 0] != np.float32(-999.0)
    assert again.scalars[0] != np.float32(-999.0)
    assert again.observation[0] != np.float32(-999.0)
    np.testing.assert_array_equal(again.action_mask, np.array([True, True, True]))
    np.testing.assert_array_equal(again.lightzero_action_mask, np.array([1, 1, 1]))


def test_observe_egocentric_rays_v0_is_ego_perspective_for_symmetric_reset():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    env.reset(seed=7)
    assert env.state is not None

    player_0 = observe_egocentric_rays_v0(env.state, env.config, "player_0", player_ids=env.agents)
    player_1 = observe_egocentric_rays_v0(env.state, env.config, "player_1", player_ids=env.agents)

    np.testing.assert_allclose(player_0.observation, player_1.observation, atol=1e-6)


def test_observe_1v1_egocentric_rays_v0_returns_both_ego_rows_in_stable_shapes():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    env.reset(seed=7)
    assert env.state is not None

    batched = observe_1v1_egocentric_rays_v0(
        env.state,
        env.config,
        player_ids=env.agents,
    )

    assert batched.player_ids == tuple(env.agents)
    assert batched.rays.shape == (2, 24, 4)
    assert batched.rays.dtype == np.float32
    assert batched.scalars.shape == (2, 10)
    assert batched.scalars.dtype == np.float32
    assert batched.observation.shape == (2, 106)
    assert batched.observation.dtype == np.float32
    assert batched.action_mask.shape == (2, 3)
    assert batched.action_mask.dtype == np.bool_
    assert batched.lightzero_action_mask.shape == (2, 3)
    assert batched.lightzero_action_mask.dtype == np.int8
    assert batched.to_play.shape == (2,)
    np.testing.assert_array_equal(batched.to_play, np.array([-1, -1], dtype=np.int64))
    np.testing.assert_array_equal(batched.rewards, np.array([0.0, 0.0], dtype=np.float32))
    assert batched.reward_map == {"player_0": 0.0, "player_1": 0.0}
    assert batched.final_reward_map is None
    assert batched.done is False
    assert batched.terminated is False
    assert batched.truncated is False

    for row, player_id in enumerate(env.agents):
        direct = observe_egocentric_rays_v0(
            env.state,
            env.config,
            player_id,
            player_ids=env.agents,
        )
        np.testing.assert_array_equal(batched.rays[row], direct.rays)
        np.testing.assert_array_equal(batched.scalars[row], direct.scalars)
        np.testing.assert_array_equal(batched.observation[row], direct.observation)
        np.testing.assert_array_equal(batched.action_mask[row], direct.action_mask)
        np.testing.assert_array_equal(
            batched.lightzero_action_mask[row],
            direct.lightzero_action_mask,
        )
        assert batched.reward_info[player_id]["ego_player_id"] == player_id
        assert batched.reward_info[player_id]["done"] is False

    payloads = batched.lightzero_payloads()
    assert set(payloads) == set(env.agents)
    for player_id in env.agents:
        row = env.agents.index(player_id)
        np.testing.assert_array_equal(
            payloads[player_id]["observation"],
            batched.observation[row],
        )
        np.testing.assert_array_equal(
            payloads[player_id]["action_mask"],
            batched.lightzero_action_mask[row],
        )
        assert payloads[player_id]["to_play"] == -1


def test_observe_1v1_egocentric_rays_v0_terminal_survivor_reward_map_and_masks():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    env.reset(seed=7)
    assert env.state is not None
    env.state.alive[:] = np.array([True, False], dtype=np.bool_)

    batched = observe_1v1_egocentric_rays_v0(
        env.state,
        env.config,
        player_ids=env.agents,
    )

    assert batched.done is True
    assert batched.terminated is True
    assert batched.truncated is False
    np.testing.assert_array_equal(batched.rewards, np.array([1.0, -1.0], dtype=np.float32))
    assert batched.reward_map == {"player_0": 1.0, "player_1": -1.0}
    assert batched.final_reward_map == {"player_0": 1.0, "player_1": -1.0}
    np.testing.assert_array_equal(
        batched.action_mask,
        np.zeros((2, 3), dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        batched.lightzero_action_mask,
        np.zeros((2, 3), dtype=np.int8),
    )

    for player_id in env.agents:
        info = batched.reward_info[player_id]
        assert info["terminal_reason"] == "survivor_win"
        assert info["winner_ids"] == ("player_0",)
        assert info["loser_ids"] == ("player_1",)
        assert info["done"] is True
        assert info["terminated"] is True
        assert info["truncated"] is False


def test_observe_1v1_egocentric_rays_v0_all_dead_draw_reward_map_and_masks():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    env.reset(seed=7)
    assert env.state is not None
    env.state.alive[:] = False

    batched = observe_1v1_egocentric_rays_v0(
        env.state,
        env.config,
        player_ids=env.agents,
    )

    assert batched.done is True
    assert batched.terminated is True
    assert batched.truncated is False
    np.testing.assert_array_equal(batched.rewards, np.array([0.0, 0.0], dtype=np.float32))
    assert batched.reward_map == {"player_0": 0.0, "player_1": 0.0}
    assert batched.final_reward_map == {"player_0": 0.0, "player_1": 0.0}
    np.testing.assert_array_equal(
        batched.action_mask,
        np.zeros((2, 3), dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        batched.lightzero_action_mask,
        np.zeros((2, 3), dtype=np.int8),
    )

    for player_id in env.agents:
        info = batched.reward_info[player_id]
        assert info["terminal_reason"] == "all_dead_draw"
        assert info["winner_ids"] == ()
        assert info["loser_ids"] == ()
        assert info["draw"] is True
        assert info["done"] is True
        assert info["terminated"] is True
        assert info["truncated"] is False


def test_observe_1v1_egocentric_rays_v0_empty_occupancy_has_no_trail_hits():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    env.reset(seed=7)
    assert env.state is not None
    env.state.occupancy[:] = 0

    batched = observe_1v1_egocentric_rays_v0(
        env.state,
        env.config,
        player_ids=env.agents,
    )

    expected = np.ones((2, 24, 2), dtype=np.float32)
    np.testing.assert_array_equal(batched.rays[:, :, 1:3], expected)


def test_empty_arena_geometry_manifest_pins_next_observation_fixture_target():
    manifest = _load_empty_geometry_manifest()

    assert manifest["schema_version"] == "observation-fixture-manifest-v0"
    assert manifest["fixture_id"] == "obs_empty_arena_geometry_v0"
    assert manifest["observation_schema_id"] == contract.OBSERVATION_SCHEMA_ID
    assert manifest["observation_schema_hash"] == contract.OBSERVATION_SCHEMA_HASH
    assert manifest["state_source"]["source_backed_observation_fidelity"] is False
    assert "browser pixel fidelity" in manifest["state_source"]["notes"][0]

    scenario_paths = manifest["state_source"]["related_source_scenarios"]
    assert scenario_paths == [
        "scenarios/environment/source_kinematics_straight_multistep.json",
        "scenarios/environment/source_borderless_wrap_step.json",
    ]
    for scenario_path in scenario_paths:
        assert (_ROOT / scenario_path).is_file()

    assert set(manifest["checks"]) == {
        "symmetric_empty",
        "no_absolute_position_leak",
        "borderless_wall_channel",
    }


def test_empty_arena_geometry_fixture_pins_perspective_and_leak_canaries():
    manifest = _load_empty_geometry_manifest()
    config = _config_from_manifest(manifest)
    player_ids = manifest["player_ids"]
    tolerance = manifest["checks"]["symmetric_empty"]["absolute_tolerance"]
    symmetric_state = _state_from_manifest(manifest, "symmetric_empty")
    translated_state = _state_from_manifest(manifest, "translated_empty")

    player_0 = observe_egocentric_rays_v0(
        symmetric_state,
        config,
        "player_0",
        player_ids=player_ids,
    )
    player_1 = observe_egocentric_rays_v0(
        symmetric_state,
        config,
        "player_1",
        player_ids=player_ids,
    )
    np.testing.assert_allclose(player_0.observation, player_1.observation, atol=tolerance)

    translated = observe_egocentric_rays_v0(
        translated_state,
        config,
        "player_0",
        player_ids=player_ids,
    )
    leak_check = manifest["checks"]["no_absolute_position_leak"]
    np.testing.assert_allclose(player_0.rays[:, 1:], translated.rays[:, 1:], atol=tolerance)
    np.testing.assert_allclose(player_0.scalars, translated.scalars, atol=tolerance)
    assert leak_check["intentionally_different_fields"] == ["rays[:, 0]"]
    assert np.max(np.abs(player_0.rays[:, 0] - translated.rays[:, 0])) > np.float32(
        tolerance
    )


def test_empty_arena_geometry_fixture_pins_borderless_wall_channel_no_hit():
    manifest = _load_empty_geometry_manifest()
    config = _config_from_manifest(manifest)
    edge_state = _state_from_manifest(manifest, "borderless_edge")
    check = manifest["checks"]["borderless_wall_channel"]
    tolerance = check["absolute_tolerance"]

    normal_wall = observe_egocentric_rays_v0(
        edge_state,
        config,
        check["ego_id"],
        player_ids=manifest["player_ids"],
        borderless=False,
    )
    borderless = observe_egocentric_rays_v0(
        edge_state,
        config,
        check["ego_id"],
        player_ids=manifest["player_ids"],
        borderless=True,
    )

    assert normal_wall.rays[0, 0] < np.float32(check["normal_wall_forward_ray_less_than"])
    np.testing.assert_allclose(
        borderless.rays[:, 0],
        np.full(24, check["borderless_wall_channel_all"], dtype=np.float32),
        atol=tolerance,
    )


def test_source_movement_observation_manifest_references_trusted_fixture_state():
    manifest = _load_source_movement_manifest()

    assert manifest["schema_version"] == "observation-fixture-manifest-v0"
    assert manifest["fixture_id"] == "obs_source_movement_empty_multistep_v0"
    assert manifest["status"] == "distilled_source_state_observation_canary"
    assert manifest["observation_schema_id"] == contract.OBSERVATION_SCHEMA_ID
    assert manifest["observation_schema_hash"] == contract.OBSERVATION_SCHEMA_HASH

    state_source = manifest["state_source"]
    assert state_source["kind"] == "source_scenario_expected_frames_distilled_state"
    assert state_source["source_backed_observation_fidelity"] is True
    assert state_source["browser_pixel_fidelity"] is False
    assert state_source["full_trail_body_fidelity"] is False
    assert "distilled into EnvState snapshots" in state_source["source_backing_scope"]

    scenario_path = _ROOT / state_source["source_scenario"]
    assert scenario_path.is_file()
    scenario = _load_source_scenario_for_manifest(manifest)
    assert scenario["id"] == state_source["source_scenario_id"]
    assert scenario["comparison"]["source_fidelity_required"] is True
    assert "source-derived" in scenario["provenance"]["labels"]
    assert scenario["source_setup"]["game"]["borderless"] is False
    assert [player["initial"]["printing"] for player in scenario["players"]] == (
        manifest["checks"]["source_fixture_reference"]["expected_initial_printing"]
    )

    frames_by_tick = _source_expected_frames_by_tick(manifest)
    expected_ticks = manifest["state_source"]["source_expected_frame_ticks"]
    assert expected_ticks == manifest["checks"]["source_fixture_reference"][
        "expected_frame_ticks"
    ]
    assert set(expected_ticks).issubset(frames_by_tick)
    assert set(manifest["checks"]) == {
        "source_fixture_reference",
        "empty_trail_channels",
        "ego_perspective_non_wall_symmetry",
        "movement_closes_forward_head_distance",
    }


def test_source_movement_observation_canary_pins_empty_trail_and_perspective_channels():
    manifest = _load_source_movement_manifest()
    config = _config_from_manifest(manifest)
    frames_by_tick = _source_expected_frames_by_tick(manifest)
    trail_check = manifest["checks"]["empty_trail_channels"]
    perspective_check = manifest["checks"]["ego_perspective_non_wall_symmetry"]
    movement_check = manifest["checks"]["movement_closes_forward_head_distance"]

    observations_by_tick: dict[int, dict[str, object]] = {}
    for tick in perspective_check["source_frame_ticks"]:
        state = _state_from_source_expected_frame(manifest, frames_by_tick[int(tick)])
        observations = {
            ego_id: observe_egocentric_rays_v0(
                state,
                config,
                ego_id,
                player_ids=manifest["player_ids"],
            )
            for ego_id in perspective_check["ego_ids"]
        }
        observations_by_tick[int(tick)] = observations

        for observed in observations.values():
            assert observed.reward == np.float32(0.0)
            np.testing.assert_array_equal(
                observed.action_mask,
                np.array([True, True, True], dtype=np.bool_),
            )
            for channel_index in trail_check["channel_indices"].values():
                np.testing.assert_allclose(
                    observed.rays[:, int(channel_index)],
                    np.full(24, trail_check["expect_no_hit"], dtype=np.float32),
                    atol=trail_check["absolute_tolerance"],
                )

        player_0 = observations["p0"]
        player_1 = observations["p1"]
        np.testing.assert_allclose(
            player_0.rays[:, 1],
            player_1.rays[:, 1],
            atol=perspective_check["absolute_tolerance"],
        )
        np.testing.assert_allclose(
            player_0.rays[:, 2],
            player_1.rays[:, 2],
            atol=perspective_check["absolute_tolerance"],
        )
        np.testing.assert_allclose(
            player_0.rays[0, 3],
            player_1.rays[0, 3],
            atol=perspective_check["absolute_tolerance"],
        )
        np.testing.assert_allclose(
            player_0.scalars,
            player_1.scalars,
            atol=perspective_check["absolute_tolerance"],
        )
        assert perspective_check["intentionally_unchecked_fields"] == [
            "rays[:, wall_or_out_of_bounds]"
        ]

    earlier_tick, later_tick = movement_check["source_frame_ticks"]
    earlier = observations_by_tick[int(earlier_tick)][movement_check["ego_id"]]
    later = observations_by_tick[int(later_tick)][movement_check["ego_id"]]
    ray_index = int(movement_check["ray_index"])
    channel_index = int(movement_check["channel_index"])
    assert later.rays[ray_index, channel_index] < earlier.rays[ray_index, channel_index]
    for tick, observed_by_ego in observations_by_tick.items():
        observed = observed_by_ego[movement_check["ego_id"]]
        assert observed.scalars[int(movement_check["tick_scalar_index"])] == np.float32(
            tick / config.max_ticks
        )


def test_observe_egocentric_rays_v0_separates_basic_trail_and_head_channels():
    config = CurvyTronConfig(
        width=20,
        height=20,
        max_ticks=100,
        action_repeat=1,
        trail_radius=0.25,
    )
    occupancy = np.zeros((config.height, config.width), dtype=np.int16)
    occupancy[8, 8] = 1
    occupancy[8, 5] = 1
    occupancy[8, 11] = 2
    occupancy[8, 14] = 2
    state = EnvState(
        tick=0,
        positions=np.array([[8.0, 8.0], [14.0, 8.0]], dtype=np.float32),
        headings=np.array([0.0, np.pi], dtype=np.float32),
        alive=np.array([True, True], dtype=np.bool_),
        death_tick=np.array([-1, -1], dtype=np.int32),
        occupancy=occupancy,
        rng=np.random.default_rng(1),
    )

    observed = observe_egocentric_rays_v0(state, config, "player_0")
    forward = observed.rays[0]
    backward = observed.rays[12]

    assert forward[2] < forward[3] < np.float32(1.0)
    assert backward[1] < np.float32(1.0)
    assert forward[1] == np.float32(1.0)


def test_observe_egocentric_rays_v0_sparse_reward_and_empty_terminal_masks():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    env.reset(seed=7)
    assert env.state is not None
    env.state.alive[1] = False

    winner = observe_egocentric_rays_v0(env.state, env.config, "player_0", player_ids=env.agents)
    loser = observe_egocentric_rays_v0(env.state, env.config, "player_1", player_ids=env.agents)

    np.testing.assert_array_equal(winner.action_mask, np.array([False, False, False]))
    assert winner.reward == np.float32(1.0)
    assert winner.reward_info["terminal_reason"] == "survivor_win"
    assert winner.reward_info["winner_ids"] == ("player_0",)
    assert winner.reward_info["loser_ids"] == ("player_1",)
    assert winner.reward_info["terminated"] is True
    assert winner.reward_info["truncated"] is False
    assert loser.reward == np.float32(-1.0)

    env.state.alive[:] = True
    env.state.tick = env.config.max_ticks
    truncated = observe_egocentric_rays_v0(
        env.state,
        env.config,
        "player_0",
        player_ids=env.agents,
    )

    np.testing.assert_array_equal(truncated.action_mask, np.array([False, False, False]))
    assert truncated.reward == np.float32(0.0)
    assert truncated.reward_info["terminal_reason"] == "timeout"
    assert truncated.reward_info["truncation_reason"] == "max_ticks"
    assert truncated.reward_info["terminated"] is False
    assert truncated.reward_info["truncated"] is True

    env.state.alive[:] = False
    env.state.tick = 0
    draw = observe_egocentric_rays_v0(
        env.state,
        env.config,
        "player_0",
        player_ids=env.agents,
    )

    assert draw.reward == np.float32(0.0)
    assert draw.reward_info["terminal_reason"] == "all_dead_draw"
    assert draw.reward_info["draw"] is True
    assert draw.reward_info["terminated"] is True
    assert draw.reward_info["truncated"] is False

    env.state.alive[:] = np.array([True, False], dtype=np.bool_)
    env.state.tick = env.config.max_ticks
    terminal_plus_truncated = observe_egocentric_rays_v0(
        env.state,
        env.config,
        "player_0",
        player_ids=env.agents,
    )

    assert terminal_plus_truncated.reward == np.float32(1.0)
    assert terminal_plus_truncated.reward_info["terminal_reason"] == "survivor_win"
    assert terminal_plus_truncated.reward_info["terminated"] is True
    assert terminal_plus_truncated.reward_info["truncated"] is True
    assert terminal_plus_truncated.reward_info["truncation_reason"] == "max_ticks"
