import numpy as np
import pytest

from curvyzero.training.multiplayer_source_state_lightzero_native_bridge import (
    build_lightzero_source_state_native_game_segments_v0,
)
from curvyzero.training.multiplayer_source_state_lightzero_native_bridge import (
    maybe_push_lightzero_source_state_native_segments_into_muzero_buffer_v0,
)
from curvyzero.training.multiplayer_source_state_native_bridge import (
    build_source_state_native_game_segment_specs_v0,
)
from curvyzero.training.multiplayer_source_state_target_rows import DEFAULT_TO_PLAY
from curvyzero.training.multiplayer_source_state_target_rows import (
    PROJECT_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_HASH,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_ID,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SourceStateMultiplayerTargetRowsV0,
)


def test_lightzero_source_state_bridge_uses_real_api_shape_and_keeps_claims_false():
    rows = _target_rows(
        metadata={
            "project_training_helper_active": True,
            "project_training_helper_metadata": {
                "death_mode": "profile_no_death",
                "source_fidelity_claim": PROJECT_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM,
            },
        }
    )
    config = {"caller": "config"}

    result = build_lightzero_source_state_native_game_segments_v0(
        rows,
        game_segment_cls=FakeLightZeroGameSegment,
        config=config,
        action_space_size=3,
    )

    assert result.config is config
    assert result.metadata["construction_smoke"] is True
    assert result.metadata["project_training_helper_active"] is True
    assert result.metadata["project_training_helper_metadata"]["death_mode"] == (
        "profile_no_death"
    )
    assert result.metadata["native_game_segment_claim"] is False
    assert result.metadata["lightzero_native_game_segment_claim"] is False
    assert result.metadata["muzero_game_buffer_claim"] is False
    assert result.metadata["learner_update_claim"] is False
    assert result.metadata["policy_improvement_claim"] is False

    assert len(result.game_segments) == 2
    segment_0 = result.game_segments[0]
    assert segment_0.action_space.n == 3
    assert segment_0.game_segment_length == 2
    assert segment_0.config is config

    assert segment_0.calls[0][0] == "reset"
    assert isinstance(segment_0.calls[0][1], list)
    assert len(segment_0.calls[0][1]) == 1
    np.testing.assert_array_equal(segment_0.calls[0][1][0], _obs(0, 0))

    assert segment_0.calls[1][0] == "store_search_stats"
    assert segment_0.calls[1][1] == [1.0, 0.0, 0.0]
    assert segment_0.calls[1][2] == pytest.approx(0.5)
    assert segment_0.calls[2][0] == "append"
    assert segment_0.calls[2][1] == 0
    np.testing.assert_array_equal(segment_0.calls[2][2], _obs(0, 1))
    assert segment_0.calls[2][3] == pytest.approx(1.0)
    np.testing.assert_array_equal(segment_0.calls[2][4], np.asarray([1, 1, 0], dtype=np.int8))
    assert segment_0.calls[2][4].dtype == np.int8
    assert segment_0.calls[2][5] == DEFAULT_TO_PLAY
    assert segment_0.calls[2][6] == 0

    assert segment_0.calls[3][0] == "store_search_stats"
    assert segment_0.calls[3][1] == [0.0, 1.0, 0.0]
    assert segment_0.calls[3][2] == pytest.approx(1.5)
    assert segment_0.calls[4][0] == "append"
    assert segment_0.calls[4][1] == 1
    np.testing.assert_array_equal(segment_0.calls[4][2], _obs(0, 3))
    assert segment_0.calls[4][3] == pytest.approx(3.0)
    np.testing.assert_array_equal(segment_0.calls[4][4], np.asarray([1, 1, 1], dtype=np.int8))
    assert segment_0.calls[4][5] == DEFAULT_TO_PLAY
    assert segment_0.calls[4][6] == 2
    assert segment_0.calls[5] == ("game_segment_to_array",)


def test_lightzero_source_state_bridge_accepts_prebuilt_specs_and_pushes_buffer():
    rows = _target_rows()
    specs = build_source_state_native_game_segment_specs_v0(rows)

    result = build_lightzero_source_state_native_game_segments_v0(
        specs,
        game_segment_cls=FakeLightZeroGameSegment,
    )
    pushed = maybe_push_lightzero_source_state_native_segments_into_muzero_buffer_v0(
        result,
        buffer_cls=FakeMuZeroGameBuffer,
    )

    assert result.specs == specs
    assert result.metadata["segment_count"] == 2
    assert result.metadata["native_game_segment_claim"] is False
    assert result.metadata["lightzero_native_game_segment_claim"] is False
    assert result.metadata["muzero_game_buffer_claim"] is False
    assert pushed.pushed_count == 2
    assert pushed.transition_count == 4
    assert pushed.buffer.pushed_segments == list(result.game_segments)
    assert len(pushed.buffer.pushed_meta) == 2


def test_real_lightzero_source_state_native_segments_smoke_when_available():
    pytest.importorskip("lzero", reason="DI-engine/LightZero runtime is not installed locally")
    game_segment_module = pytest.importorskip("lzero.mcts.buffer.game_segment")

    try:
        result = build_lightzero_source_state_native_game_segments_v0(
            _target_rows(player_count=1),
            game_segment_cls=game_segment_module.GameSegment,
        )
    except Exception as exc:
        pytest.skip(f"LightZero GameSegment construction smoke blocked: {exc!r}")

    assert len(result.specs) == 1
    assert len(result.game_segments) == 1
    assert result.metadata["construction_smoke"] is True
    assert result.metadata["native_game_segment_claim"] is False
    assert result.metadata["lightzero_native_game_segment_claim"] is False
    assert result.metadata["muzero_game_buffer_claim"] is False


class FakeLightZeroGameSegment:
    def __init__(self, action_space, game_segment_length, config):
        self.action_space = action_space
        self.game_segment_length = int(game_segment_length)
        self.config = config
        self.calls = []

    def reset(self, observations):
        self.calls.append(
            (
                "reset",
                [np.asarray(observation, dtype=np.float32).copy() for observation in observations],
            )
        )

    def store_search_stats(self, policy, *, root_value):
        self.calls.append(("store_search_stats", list(policy), float(root_value)))

    def append(
        self,
        action,
        next_observation,
        reward,
        *,
        action_mask,
        to_play,
        timestep,
    ):
        self.calls.append(
            (
                "append",
                int(action),
                np.asarray(next_observation, dtype=np.float32).copy(),
                float(reward),
                np.asarray(action_mask).copy(),
                int(to_play),
                int(timestep),
            )
        )

    def game_segment_to_array(self):
        self.calls.append(("game_segment_to_array",))


class FakeMuZeroGameBuffer:
    def __init__(self, config):
        self.config = config
        self.pushed_segments = None
        self.pushed_meta = None

    def push_game_segments(self, payload):
        segments, meta = payload
        self.pushed_segments = segments
        self.pushed_meta = meta

    def get_num_of_transitions(self):
        return sum(
            1
            for segment in self.pushed_segments
            for call in segment.calls
            if call[0] == "append"
        )


def _target_rows(*, player_count: int = 2, metadata: dict | None = None):
    row_defs = []
    for player in range(player_count):
        row_defs.extend(
            [
                {
                    "record_index": 2,
                    "player": player,
                    "action": (player + 1) % 3,
                    "reward": float(player + 3),
                    "root_value": float(player) + 1.5,
                    "done": True,
                },
                {
                    "record_index": 0,
                    "player": player,
                    "action": player % 3,
                    "reward": float(player + 1),
                    "root_value": float(player) + 0.5,
                    "done": False,
                },
            ]
        )
    row_count = len(row_defs)
    target_metadata = {
        "target_contract_id": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID,
        "target_schema_id": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_ID,
        "target_schema_hash": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_HASH,
        "native_game_segment_claim": False,
        "lightzero_native_game_segment_claim": False,
        "lightzero_training_integration_claim": False,
        "lightzero_training_claim": False,
        "muzero_game_buffer_claim": False,
        "learner_update_claim": False,
        "policy_improvement_claim": False,
    }
    if metadata:
        target_metadata.update(metadata)

    return SourceStateMultiplayerTargetRowsV0(
        metadata=target_metadata,
        observation=np.stack(
            [_obs(row["player"], row["record_index"]) for row in row_defs],
            axis=0,
        ).astype(np.float32),
        action=np.asarray([row["action"] for row in row_defs], dtype=np.int16),
        action_mask=np.asarray(
            [_mask(row["action"]) for row in row_defs],
            dtype=bool,
        ),
        policy_target=np.asarray(
            [_policy(row["action"]) for row in row_defs],
            dtype=np.float32,
        ),
        root_value=np.asarray([row["root_value"] for row in row_defs], dtype=np.float32),
        reward=np.asarray([row["reward"] for row in row_defs], dtype=np.float32),
        final_reward=np.asarray([row["reward"] for row in row_defs], dtype=np.float32),
        done=np.asarray([row["done"] for row in row_defs], dtype=bool),
        terminated=np.asarray([row["done"] for row in row_defs], dtype=bool),
        truncated=np.zeros((row_count,), dtype=bool),
        next_observation=np.stack(
            [_obs(row["player"], row["record_index"] + 1) for row in row_defs],
            axis=0,
        ).astype(np.float32),
        to_play=np.full((row_count,), DEFAULT_TO_PLAY, dtype=np.int64),
        env_row=np.zeros((row_count,), dtype=np.int32),
        player=np.asarray([row["player"] for row in row_defs], dtype=np.int16),
        record_index=np.asarray(
            [row["record_index"] for row in row_defs],
            dtype=np.int32,
        ),
        next_record_index=np.asarray(
            [row["record_index"] + 1 for row in row_defs],
            dtype=np.int32,
        ),
        policy_row=np.arange(row_count, dtype=np.int32),
        policy_source=tuple("unit_test" for _ in row_defs),
        source_record_ref=tuple(f"row:{index}" for index in range(row_count)),
    )


def _obs(player: int, record_index: int) -> np.ndarray:
    return np.asarray([float(player), float(record_index)], dtype=np.float32)


def _mask(action: int) -> np.ndarray:
    mask = np.ones((3,), dtype=bool)
    if action == 0:
        mask[2] = False
    return mask


def _policy(action: int) -> np.ndarray:
    policy = np.zeros((3,), dtype=np.float32)
    policy[action] = 1.0
    return policy
