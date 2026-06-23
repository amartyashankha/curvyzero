import numpy as np
import pytest

from curvyzero.infra.modal.mctx_synthetic_benchmark import (
    _closed_compact_plain_breakdown,
    _extract_mctx_root_values,
    _materialize_mctx_search_payload,
    _mctx_legality_summary,
    _mctx_compact_visual_search_service_profile_row,
    _validate_compact_visual_root_row_major_order,
    _validate_resident_compact_visual_latest,
)


def test_mctx_legality_summary_accepts_active_legal_rows_only() -> None:
    actions = np.array([1, 0, 2, 0], dtype=np.int32)
    weights = np.array(
        [
            [0.0, 0.75, 0.25],
            [0.9, 0.0, 0.1],
            [0.2, 0.8, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    invalid = np.array(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [False, True, True],
        ],
        dtype=bool,
    )
    active = np.array([True, True, False, False], dtype=bool)

    summary, problems = _mctx_legality_summary(actions, weights, invalid, active)

    assert problems == []
    assert summary["actions_legal"] is True
    assert summary["active_root_count"] == 2
    assert summary["inactive_root_count"] == 2
    assert summary["illegal_selected_action_count"] == 0
    assert summary["illegal_action_weight_mass_max"] == 0.0
    assert summary["legal_action_count_per_root_sample"] == [2, 2]


def test_mctx_legality_summary_flags_illegal_action_and_weight_mass() -> None:
    actions = np.array([0, 1], dtype=np.int32)
    weights = np.array(
        [
            [0.25, 0.75, 0.0],
            [0.1, 0.7, 0.2],
        ],
        dtype=np.float32,
    )
    invalid = np.array(
        [
            [True, False, False],
            [False, False, True],
        ],
        dtype=bool,
    )
    active = np.array([True, True], dtype=bool)

    summary, problems = _mctx_legality_summary(actions, weights, invalid, active)

    assert summary["actions_legal"] is False
    assert summary["illegal_selected_action_count"] == 1
    assert summary["illegal_action_weight_mass_max"] == pytest.approx(0.25)
    assert any("selected 1 illegal actions" in item for item in problems)
    assert any("assign mass to illegal actions" in item for item in problems)


def test_resident_compact_visual_latest_requires_uint8_shape() -> None:
    latest = np.zeros((2, 2, 1, 64, 64), dtype=np.uint8)

    _validate_resident_compact_visual_latest(latest, env_rows=2, players=2)

    with pytest.raises(ValueError, match="dtype must be uint8"):
        _validate_resident_compact_visual_latest(
            latest.astype(np.float32),
            env_rows=2,
            players=2,
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        _validate_resident_compact_visual_latest(latest[:1], env_rows=2, players=2)


def test_compact_visual_root_order_guard_requires_row_major_rows_and_players() -> None:
    _validate_compact_visual_root_row_major_order(
        env_rows=np.asarray([0, 0, 1, 1], dtype=np.int32),
        players=np.asarray([0, 1, 0, 1], dtype=np.int32),
        env_row_count=2,
        player_count=2,
    )

    with pytest.raises(ValueError, match="row-major"):
        _validate_compact_visual_root_row_major_order(
            env_rows=np.asarray([0, 1, 0, 1], dtype=np.int32),
            players=np.asarray([0, 0, 1, 1], dtype=np.int32),
            env_row_count=2,
            player_count=2,
        )


def test_closed_compact_plain_breakdown_separates_mechanics_from_handoff() -> None:
    breakdown = _closed_compact_plain_breakdown(
        bucket_totals={
            "env_step_sec": 10.0,
            "search_sec": 2.0,
            "root_build_sec": 1.0,
            "h2d_sec": 0.5,
            "d2h_sec": 0.25,
            "deferred_search_payload_flush_sec": 0.75,
            "replay_index_sec": 0.25,
        },
        next_step_totals={
            "actor_env_runtime_sec": 1.0,
            "actor_env_reward_sec": 0.1,
            "actor_env_post_runtime_bookkeeping_sec": 0.2,
            "actor_env_public_prepare_sec": 0.3,
            "actor_env_public_info_sec": 0.4,
            "actor_env_batch_pack_sec": 0.5,
            "actor_compact_write_sec": 0.6,
            "actor_render_state_write_sec": 2.0,
            "observation_sec": 3.0,
            "resident_stack_update_sec": 0.5,
            "stack_shift_sec": 0.7,
            "stack_latest_update_sec": 0.8,
        },
        total_sec=15.0,
    )

    assert breakdown["top_level_sec"]["env_step_sec"] == pytest.approx(10.0)
    assert breakdown["top_level_sec"]["deferred_search_payload_flush_sec"] == pytest.approx(
        0.75
    )
    assert breakdown["top_level_sec"]["unlabeled_residual_sec"] == pytest.approx(0.25)
    assert breakdown["env_step_leaf_sec"]["game_mechanics_leaf_sec"] == pytest.approx(1.3)
    assert breakdown["env_step_leaf_sec"]["public_packaging_leaf_sec"] == pytest.approx(1.8)
    assert breakdown["env_step_leaf_sec"]["observation_handoff_leaf_sec"] == pytest.approx(5.5)
    assert breakdown["env_step_leaf_fraction_of_env_step"]["game_mechanics"] == pytest.approx(
        0.13
    )


def test_mctx_compact_visual_service_profile_row_is_labeled_profile_only() -> None:
    row = _mctx_compact_visual_search_service_profile_row(
        observation_mode="curvytron_hybrid_compact_visual_sample",
        num_simulations=16,
        end_to_end_active_decisions_per_sec_median=123.5,
        closed_compact_loop_active_roots_per_sec=456.25,
        compact_search_contract={"schema_id": "curvyzero_compact_search_result/v1"},
        compact_replay_index_contract={"schema_id": "curvyzero_compact_replay_index/v1"},
    )

    assert row["backend_name"] == "mctx_hybrid_compact_visual_search_service"
    assert row["profile_only"] is True
    assert row["not_lightzero_ctree"] is True
    assert row["not_train_muzero"] is True
    assert "not_lightzero_ctree" in row["semantics"]
    assert row["metrics"]["end_to_end_active_decisions_per_sec_median"] == pytest.approx(
        123.5
    )
    assert row["metrics"]["closed_compact_loop_active_roots_per_sec"] == pytest.approx(
        456.25
    )
    assert row["contracts"]["compact_search_contract_present"] is True
    assert row["contracts"]["compact_replay_index_contract_present"] is True


def test_mctx_root_value_extractor_uses_root_node_values_for_replay_payload() -> None:
    class SearchTree:
        node_values = np.asarray(
            [
                [0.5, 99.0],
                [-0.25, 88.0],
                [1.25, 77.0],
            ],
            dtype=np.float32,
        )

    class Output:
        search_tree = SearchTree()
        action_weights = np.ones((3, 3), dtype=np.float32) / np.float32(3.0)

    values, source = _extract_mctx_root_values(Output())
    payload_bytes, root_values_present = _materialize_mctx_search_payload(Output())

    np.testing.assert_array_equal(
        values,
        np.asarray([0.5, -0.25, 1.25], dtype=np.float32),
    )
    assert source == "search_tree.node_values[:,0]"
    assert root_values_present is True
    assert payload_bytes == Output.action_weights.nbytes + values.nbytes
