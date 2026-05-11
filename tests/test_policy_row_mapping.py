import numpy as np
import pytest

from curvyzero.training.policy_row_mapping import (
    NOOP_ACTION_ID,
    PADDED_ROW_ID,
    build_policy_row_mapping,
    policy_rows_to_joint_action,
)


def test_policy_row_mapping_compacts_live_rows_with_legal_actions_and_ids():
    obs = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    live_mask = np.array(
        [
            [True, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    legal_action_mask = np.ones((2, 3, 3), dtype=bool)
    legal_action_mask[0, 1] = np.array([True, False, True], dtype=bool)
    legal_action_mask[0, 2] = np.array([True, True, True], dtype=bool)
    legal_action_mask[1, 0] = np.array([False, False, False], dtype=bool)

    mapping = build_policy_row_mapping(obs, live_mask, legal_action_mask)

    assert mapping.source_shape == (2, 3)
    assert mapping.action_count == 3
    assert mapping.active_count == 3
    assert mapping.capacity == 3
    np.testing.assert_array_equal(mapping.row_mask, np.array([True, True, True]))
    np.testing.assert_array_equal(mapping.env_row_id, np.array([0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(mapping.player_id, np.array([0, 1, 2], dtype=np.int16))
    np.testing.assert_array_equal(mapping.observations, obs[[0, 0, 1], [0, 1, 2]])
    np.testing.assert_array_equal(
        mapping.legal_action_mask,
        np.array(
            [
                [True, True, True],
                [True, False, True],
                [True, True, True],
            ],
            dtype=bool,
        ),
    )


def test_policy_row_mapping_can_pad_compact_rows():
    obs = np.arange(1 * 4 * 2, dtype=np.float32).reshape(1, 4, 2)
    live_mask = np.array([[True, False, True, False]], dtype=bool)
    legal_action_mask = np.zeros((1, 4, 3), dtype=bool)
    legal_action_mask[0, 0] = np.array([False, True, True], dtype=bool)
    legal_action_mask[0, 2] = np.array([True, False, False], dtype=bool)

    mapping = build_policy_row_mapping(obs, live_mask, legal_action_mask, pad_to=5)

    assert mapping.active_count == 2
    assert mapping.capacity == 5
    np.testing.assert_array_equal(
        mapping.row_mask,
        np.array([True, True, False, False, False], dtype=bool),
    )
    np.testing.assert_array_equal(mapping.observations[:2], obs[[0, 0], [0, 2]])
    np.testing.assert_array_equal(mapping.observations[2:], np.zeros((3, 2), dtype=np.float32))
    np.testing.assert_array_equal(
        mapping.legal_action_mask,
        np.array(
            [
                [False, True, True],
                [True, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(
        mapping.env_row_id,
        np.array([0, 0, PADDED_ROW_ID, PADDED_ROW_ID, PADDED_ROW_ID], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        mapping.player_id,
        np.array([0, 2, PADDED_ROW_ID, PADDED_ROW_ID, PADDED_ROW_ID], dtype=np.int16),
    )


def test_selected_policy_actions_map_back_to_joint_action_with_noop_padding():
    obs = np.zeros((2, 3, 1), dtype=np.float32)
    live_mask = np.array(
        [
            [True, False, True],
            [False, True, True],
        ],
        dtype=bool,
    )
    legal_action_mask = np.repeat(live_mask[:, :, None], 3, axis=2)
    mapping = build_policy_row_mapping(obs, live_mask, legal_action_mask, pad_to=6)

    joint_action = policy_rows_to_joint_action(
        mapping,
        np.array([2, 0, 1, 2, 2, 2], dtype=np.int8),
    )

    expected = np.full((2, 3), NOOP_ACTION_ID, dtype=np.int8)
    expected[0, 0] = 2
    expected[0, 2] = 0
    expected[1, 1] = 1
    expected[1, 2] = 2
    np.testing.assert_array_equal(joint_action, expected)


def test_selected_policy_actions_can_be_compact_for_padded_mapping():
    obs = np.zeros((1, 3, 1), dtype=np.float32)
    live_mask = np.array([[True, False, True]], dtype=bool)
    legal_action_mask = np.repeat(live_mask[:, :, None], 3, axis=2)
    mapping = build_policy_row_mapping(obs, live_mask, legal_action_mask, pad_to=4)

    joint_action = policy_rows_to_joint_action(
        mapping,
        np.array([0, 2], dtype=np.int8),
    )

    np.testing.assert_array_equal(
        joint_action,
        np.array([[0, NOOP_ACTION_ID, 2]], dtype=np.int8),
    )


def test_selected_policy_actions_reject_illegal_active_action():
    obs = np.zeros((1, 2, 1), dtype=np.float32)
    live_mask = np.array([[True, True]], dtype=bool)
    legal_action_mask = np.array(
        [
            [
                [True, True, True],
                [True, False, True],
            ]
        ],
        dtype=bool,
    )
    mapping = build_policy_row_mapping(obs, live_mask, legal_action_mask)

    with pytest.raises(ValueError, match="illegal active-row actions"):
        policy_rows_to_joint_action(mapping, np.array([2, 1], dtype=np.int8))


def test_empty_policy_mapping_round_trips_to_all_noop_joint_action():
    obs = np.zeros((1, 2, 1), dtype=np.float32)
    live_mask = np.array([[False, True]], dtype=bool)
    legal_action_mask = np.zeros((1, 2, 3), dtype=bool)
    mapping = build_policy_row_mapping(obs, live_mask, legal_action_mask, pad_to=3)

    joint_action = policy_rows_to_joint_action(mapping, np.array([], dtype=np.int8))

    assert mapping.active_count == 0
    np.testing.assert_array_equal(
        joint_action,
        np.array([[NOOP_ACTION_ID, NOOP_ACTION_ID]], dtype=np.int8),
    )
