import numpy as np
import pytest

from curvyzero.env import vector_autoreset, vector_reset


def _lifecycle() -> dict[str, np.ndarray]:
    return {
        "done": np.asarray([False, True, True], dtype=bool),
        "terminated": np.asarray([False, True, False], dtype=bool),
        "truncated": np.asarray([False, False, True], dtype=bool),
        "episode_id": np.asarray([10, 20, 30], dtype=np.int64),
        "episode_step": np.asarray([3, 7, 9], dtype=np.int32),
    }


def _final_observation() -> np.ndarray:
    return np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)


def _final_reward_map() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )


def _reset_seed() -> np.ndarray:
    return np.asarray([101, 202, 303], dtype=np.uint64)


def _reset_source() -> np.ndarray:
    return np.asarray(
        [
            vector_reset.RESET_SOURCE_MANUAL,
            vector_reset.RESET_SOURCE_AUTORESET,
            vector_reset.RESET_SOURCE_AUTORESET,
        ],
        dtype=np.int16,
    )


def _reset_template_and_target() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    reset_template = {
        "episode_id": np.asarray([100, 200, 300], dtype=np.int64),
        "episode_step": np.asarray([0, 0, 0], dtype=np.int32),
        "env_active": np.asarray([True, True, True], dtype=bool),
        "reset_pending": np.asarray([False, False, False], dtype=bool),
        "done": np.asarray([False, False, False], dtype=bool),
        "terminated": np.asarray([False, False, False], dtype=bool),
        "truncated": np.asarray([False, False, False], dtype=bool),
        "terminal_reason": np.asarray(
            [
                vector_reset.TERMINAL_REASON_NONE,
                vector_reset.TERMINAL_REASON_NONE,
                vector_reset.TERMINAL_REASON_NONE,
            ],
            dtype=np.int16,
        ),
        "reset_seed": np.asarray([11, 22, 33], dtype=np.uint64),
        "reset_source": np.asarray(
            [
                vector_reset.RESET_SOURCE_FIXTURE,
                vector_reset.RESET_SOURCE_FIXTURE,
                vector_reset.RESET_SOURCE_FIXTURE,
            ],
            dtype=np.int16,
        ),
        "position": np.asarray(
            [
                [[1.0, 1.5], [2.0, 2.5]],
                [[3.0, 3.5], [4.0, 4.5]],
                [[5.0, 5.5], [6.0, 6.5]],
            ],
            dtype=np.float64,
        ),
    }
    target = {name: array.copy() for name, array in reset_template.items()}
    target.update(
        {
            "episode_id": np.asarray([10, 20, 30], dtype=np.int64),
            "episode_step": np.asarray([3, 7, 9], dtype=np.int32),
            "env_active": np.asarray([True, False, False], dtype=bool),
            "reset_pending": np.asarray([False, True, True], dtype=bool),
            "done": np.asarray([False, True, True], dtype=bool),
            "terminated": np.asarray([False, True, False], dtype=bool),
            "truncated": np.asarray([False, False, True], dtype=bool),
            "terminal_reason": np.asarray(
                [
                    vector_reset.TERMINAL_REASON_NONE,
                    vector_reset.TERMINAL_REASON_SURVIVOR_WIN,
                    vector_reset.TERMINAL_REASON_TIMEOUT_TRUNCATED,
                ],
                dtype=np.int16,
            ),
            "reset_seed": np.asarray([101, 202, 303], dtype=np.uint64),
            "reset_source": np.asarray(
                [
                    vector_reset.RESET_SOURCE_MANUAL,
                    vector_reset.RESET_SOURCE_MANUAL,
                    vector_reset.RESET_SOURCE_MANUAL,
                ],
                dtype=np.int16,
            ),
        }
    )
    target["position"][0] = [[10.0, 10.5], [11.0, 11.5]]
    target["position"][1] = [[20.0, 20.5], [21.0, 21.5]]
    target["position"][2] = [[30.0, 30.5], [31.0, 31.5]]
    return reset_template, target


def test_plan_autoreset_rows_defaults_to_done_rows_and_stages_copied_terminal_data():
    lifecycle = _lifecycle()
    final_observation = _final_observation()
    final_reward_map = _final_reward_map()
    reset_seed = _reset_seed()
    reset_source = _reset_source()

    plan = vector_autoreset.plan_autoreset_rows(
        lifecycle,
        final_observation=final_observation,
        final_reward_map=final_reward_map,
        reset_seed=reset_seed,
        reset_source=reset_source,
    )

    assert plan["schema"] == vector_autoreset.AUTORESET_PLAN_SCHEMA_ID
    assert plan["surface"] == vector_autoreset.AUTORESET_SURFACE
    assert plan["mutates_state"] is False
    assert plan["mask_source"] == "done"
    assert plan["explicit_autoreset_mask"] is False
    assert plan["autoreset_count"] == 2
    np.testing.assert_array_equal(plan["row_ids"], np.asarray([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(
        plan["autoreset_mask"],
        np.asarray([False, True, True], dtype=bool),
    )
    np.testing.assert_array_equal(plan["done"], np.asarray([True, True], dtype=bool))
    np.testing.assert_array_equal(plan["terminated"], np.asarray([True, False], dtype=bool))
    np.testing.assert_array_equal(plan["truncated"], np.asarray([False, True], dtype=bool))
    np.testing.assert_array_equal(
        plan["selected_non_done_rows"],
        np.asarray([], dtype=np.int32),
    )

    snapshot = plan["final_transition_snapshot"]
    assert snapshot["schema"] == vector_autoreset.AUTORESET_TERMINAL_SNAPSHOT_SCHEMA_ID
    assert snapshot["captured_before_reset"] is True
    np.testing.assert_array_equal(snapshot["rows"], np.asarray([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(
        snapshot["final_episode_id"],
        np.asarray([20, 30], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        snapshot["final_episode_step"],
        np.asarray([7, 9], dtype=np.int32),
    )
    np.testing.assert_array_equal(snapshot["final_observation"], final_observation[1:])
    np.testing.assert_array_equal(snapshot["final_reward_map"], final_reward_map[1:])

    reset_metadata = plan["reset_metadata"]
    np.testing.assert_array_equal(reset_metadata["rows"], np.asarray([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(
        reset_metadata["reset_episode_id"],
        np.asarray([21, 31], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        reset_metadata["reset_episode_step"],
        np.asarray([0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        reset_metadata["reset_seed"],
        np.asarray([202, 303], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        reset_metadata["reset_source"],
        np.asarray(
            [
                vector_reset.RESET_SOURCE_AUTORESET,
                vector_reset.RESET_SOURCE_AUTORESET,
            ],
            dtype=np.int16,
        ),
    )
    reset_kwargs = plan["reset_arrays_kwargs"]
    np.testing.assert_array_equal(reset_kwargs["reset_mask"], plan["autoreset_mask"])
    np.testing.assert_array_equal(reset_kwargs["reset_seed"], reset_seed)
    np.testing.assert_array_equal(reset_kwargs["reset_source"], reset_source)

    final_observation[1, 0, 0] = -999.0
    final_reward_map[2, 1] = -999.0
    reset_seed[1] = 999
    np.testing.assert_array_equal(
        snapshot["final_observation"],
        np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)[1:],
    )
    np.testing.assert_array_equal(
        snapshot["final_reward_map"],
        np.asarray([[1.0, -1.0], [0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        reset_metadata["reset_seed"],
        np.asarray([202, 303], dtype=np.uint64),
    )


def test_plan_autoreset_rows_explicit_mask_can_select_non_done_rows_but_reports_them():
    plan = vector_autoreset.plan_autoreset_rows(
        _lifecycle(),
        final_observation=_final_observation(),
        final_reward_map=_final_reward_map(),
        reset_seed=_reset_seed(),
        reset_source=_reset_source(),
        autoreset_mask=np.asarray([True, False, True], dtype=bool),
    )

    assert plan["mask_source"] == "caller"
    assert plan["explicit_autoreset_mask"] is True
    np.testing.assert_array_equal(plan["row_ids"], np.asarray([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(plan["done"], np.asarray([False, True], dtype=bool))
    np.testing.assert_array_equal(plan["terminated"], np.asarray([False, False], dtype=bool))
    np.testing.assert_array_equal(plan["truncated"], np.asarray([False, True], dtype=bool))
    np.testing.assert_array_equal(
        plan["selected_non_done_rows"],
        np.asarray([0], dtype=np.int32),
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"final_observation": None}, "final_observation is required"),
        ({"final_reward_map": None}, "final_reward_map is required"),
        ({"reset_seed": None}, "reset_seed is required"),
        ({"reset_source": None}, "reset_source is required"),
    ),
)
def test_plan_autoreset_rows_rejects_missing_required_autoreset_metadata(kwargs, match):
    call_kwargs = {
        "final_observation": _final_observation(),
        "final_reward_map": _final_reward_map(),
        "reset_seed": _reset_seed(),
        "reset_source": _reset_source(),
        **kwargs,
    }

    with pytest.raises(vector_autoreset.VectorAutoresetError, match=match):
        vector_autoreset.plan_autoreset_rows(_lifecycle(), **call_kwargs)


def test_apply_autoreset_rows_snapshots_terminal_rows_before_reset_and_clears_flags():
    reset_template, target = _reset_template_and_target()
    skipped_before = {name: array[0].copy() for name, array in target.items()}
    final_observation = _final_observation()
    final_reward_map = _final_reward_map()

    result = vector_autoreset.apply_autoreset_rows(
        target,
        reset_template,
        target,
        final_observation=final_observation,
        final_reward_map=final_reward_map,
        reset_seed=_reset_seed(),
        reset_source=_reset_source(),
        snapshot_array_names=("done", "terminated", "terminal_reason", "position"),
    )

    assert result["schema"] == vector_autoreset.AUTORESET_APPLY_SCHEMA_ID
    assert result["surface"] == vector_autoreset.AUTORESET_SURFACE
    assert result["mutates_state"] is True
    assert result["autoreset_count"] == 2
    np.testing.assert_array_equal(
        result["autoreset_mask"],
        np.asarray([False, True, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        result["autoreset_rows"],
        np.asarray([1, 2], dtype=np.int32),
    )

    final_snapshot = result["final_transition_snapshot"]
    np.testing.assert_array_equal(
        final_snapshot["final_observation"],
        final_observation[1:],
    )
    np.testing.assert_array_equal(
        final_snapshot["final_reward_map"],
        final_reward_map[1:],
    )
    np.testing.assert_array_equal(
        final_snapshot["final_episode_id"],
        np.asarray([20, 30], dtype=np.int64),
    )

    reset_snapshot = result["reset_info"]["terminal_transition_snapshot"]
    np.testing.assert_array_equal(
        reset_snapshot["arrays"]["done"],
        np.asarray([True, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        reset_snapshot["arrays"]["terminated"],
        np.asarray([True, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        reset_snapshot["arrays"]["terminal_reason"],
        np.asarray(
            [
                vector_reset.TERMINAL_REASON_SURVIVOR_WIN,
                vector_reset.TERMINAL_REASON_TIMEOUT_TRUNCATED,
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_allclose(reset_snapshot["arrays"]["position"][0, 0], [20.0, 20.5])

    np.testing.assert_array_equal(
        target["episode_id"],
        np.asarray([10, 21, 31], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        target["episode_step"],
        np.asarray([3, 0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(target["done"], np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(target["terminated"], np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(target["truncated"], np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(target["reset_pending"], np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(
        target["terminal_reason"],
        np.zeros(3, dtype=np.int16),
    )
    np.testing.assert_array_equal(
        target["reset_seed"],
        np.asarray([101, 202, 303], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        target["reset_source"],
        np.asarray(
            [
                vector_reset.RESET_SOURCE_MANUAL,
                vector_reset.RESET_SOURCE_AUTORESET,
                vector_reset.RESET_SOURCE_AUTORESET,
            ],
            dtype=np.int16,
        ),
    )
    for name, before in skipped_before.items():
        np.testing.assert_array_equal(target[name][0], before)


def test_apply_autoreset_rows_reports_selected_non_done_explicit_mask():
    reset_template, target = _reset_template_and_target()

    result = vector_autoreset.apply_autoreset_rows(
        target,
        reset_template,
        target,
        final_observation=_final_observation(),
        final_reward_map=_final_reward_map(),
        reset_seed=_reset_seed(),
        reset_source=_reset_source(),
        autoreset_mask=np.asarray([True, False, True], dtype=bool),
    )

    np.testing.assert_array_equal(
        result["autoreset_rows"],
        np.asarray([0, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result["selected_non_done_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result["plan"]["selected_non_done_rows"],
        np.asarray([0], dtype=np.int32),
    )
    assert result["plan"]["mask_source"] == "caller"
    assert result["reset_info"]["reset_count"] == 2


def test_apply_autoreset_rows_allows_no_selected_rows_and_leaves_target_unchanged():
    reset_template, target = _reset_template_and_target()
    before = {name: array.copy() for name, array in target.items()}

    result = vector_autoreset.apply_autoreset_rows(
        target,
        reset_template,
        target,
        final_observation=_final_observation(),
        final_reward_map=_final_reward_map(),
        reset_seed=_reset_seed(),
        reset_source=_reset_source(),
        autoreset_mask=np.zeros(3, dtype=bool),
        snapshot_array_names=("position",),
    )

    assert result["autoreset_count"] == 0
    assert result["reset_info"]["reset_count"] == 0
    np.testing.assert_array_equal(result["autoreset_rows"], np.asarray([], dtype=np.int32))
    np.testing.assert_array_equal(
        result["reset_info"]["terminal_transition_snapshot"]["final_rows"],
        np.asarray([], dtype=np.int32),
    )
    for name, value in before.items():
        np.testing.assert_array_equal(target[name], value)


@pytest.mark.parametrize(
    ("metadata_name", "metadata_value", "match"),
    (
        (
            "final_observation",
            np.asarray([{"obs": "live"}, None, {"obs": "terminal"}], dtype=object),
            "final_observation contains missing metadata",
        ),
        (
            "final_reward_map",
            np.asarray([{"p0": 0.0}, {"p0": 1.0}, None], dtype=object),
            "final_reward_map contains missing metadata",
        ),
    ),
)
def test_apply_autoreset_rows_rejects_missing_selected_row_terminal_metadata(
    metadata_name,
    metadata_value,
    match,
):
    reset_template, target = _reset_template_and_target()
    before = {name: array.copy() for name, array in target.items()}
    kwargs = {
        "final_observation": _final_observation(),
        "final_reward_map": _final_reward_map(),
        "reset_seed": _reset_seed(),
        "reset_source": _reset_source(),
        metadata_name: metadata_value,
    }

    with pytest.raises(vector_autoreset.VectorAutoresetError, match=match):
        vector_autoreset.apply_autoreset_rows(
            target,
            reset_template,
            target,
            **kwargs,
        )

    for name, value in before.items():
        np.testing.assert_array_equal(target[name], value)


@pytest.mark.parametrize(
    ("metadata_name", "metadata_value", "match"),
    (
        (
            "final_observation",
            np.asarray([None, {"obs": "terminal"}, {"obs": "terminal"}], dtype=object),
            "final_observation contains missing metadata",
        ),
        (
            "final_reward_map",
            np.asarray([None, {"p0": 1.0, "p1": -1.0}, None], dtype=object),
            "final_reward_map contains missing metadata",
        ),
    ),
)
def test_plan_autoreset_rows_rejects_missing_selected_row_terminal_metadata(
    metadata_name,
    metadata_value,
    match,
):
    kwargs = {
        "final_observation": _final_observation(),
        "final_reward_map": _final_reward_map(),
        "reset_seed": _reset_seed(),
        "reset_source": _reset_source(),
        metadata_name: metadata_value,
    }

    with pytest.raises(vector_autoreset.VectorAutoresetError, match=match):
        vector_autoreset.plan_autoreset_rows(
            _lifecycle(),
            autoreset_mask=np.asarray([True, False, True], dtype=bool),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("mutate_lifecycle", "match"),
    (
        (
            lambda lifecycle: lifecycle.__setitem__(
                "done",
                np.asarray([False, True, False], dtype=bool),
            ),
            "done must equal terminated OR truncated",
        ),
        (
            lambda lifecycle: lifecycle.__setitem__(
                "terminated",
                np.asarray([False, False, False], dtype=bool),
            ),
            "done must equal terminated OR truncated",
        ),
        (
            lambda lifecycle: lifecycle.__setitem__(
                "truncated",
                np.asarray([False, True], dtype=bool),
            ),
            "truncated must match done shape",
        ),
    ),
)
def test_plan_autoreset_rows_keeps_done_terminated_and_truncated_distinct(
    mutate_lifecycle,
    match,
):
    lifecycle = _lifecycle()
    mutate_lifecycle(lifecycle)

    with pytest.raises(vector_autoreset.VectorAutoresetError, match=match):
        vector_autoreset.plan_autoreset_rows(
            lifecycle,
            final_observation=_final_observation(),
            final_reward_map=_final_reward_map(),
            reset_seed=_reset_seed(),
            reset_source=_reset_source(),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        (
            {"autoreset_mask": np.asarray([1, 0, 1], dtype=np.int8)},
            "autoreset_mask must be a bool array",
        ),
        (
            {
                "reset_seed": np.asarray([1, 2, 3], dtype=np.int64),
            },
            "reset_seed must be a uint64 array",
        ),
        (
            {
                "reset_source": np.asarray(
                    [
                        vector_reset.RESET_SOURCE_AUTORESET,
                        99,
                        vector_reset.RESET_SOURCE_AUTORESET,
                    ],
                    dtype=np.int16,
                )
            },
            "known reset source",
        ),
        (
            {
                "final_observation": np.zeros((2, 2, 4), dtype=np.float32),
            },
            "final_observation must have leading shape",
        ),
    ),
)
def test_plan_autoreset_rows_rejects_invalid_masks_shapes_and_reset_metadata(
    kwargs,
    match,
):
    call_kwargs = {
        "final_observation": _final_observation(),
        "final_reward_map": _final_reward_map(),
        "reset_seed": _reset_seed(),
        "reset_source": _reset_source(),
        **kwargs,
    }

    with pytest.raises(vector_autoreset.VectorAutoresetError, match=match):
        vector_autoreset.plan_autoreset_rows(_lifecycle(), **call_kwargs)
