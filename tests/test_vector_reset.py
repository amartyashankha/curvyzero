import numpy as np
import pytest

from curvyzero.env import vector_reset


def _two_row_reset_template_and_target():
    event_capacity = 3
    reset_template = {
        "episode_id": np.asarray([100, 900], dtype=np.int64),
        "episode_step": np.asarray([0, 88], dtype=np.int32),
        "env_active": np.asarray([True, False], dtype=bool),
        "reset_pending": np.asarray([False, True], dtype=bool),
        "done": np.asarray([False, True], dtype=bool),
        "terminated": np.asarray([False, True], dtype=bool),
        "truncated": np.asarray([False, False], dtype=bool),
        "terminal_reason": np.asarray(
            [
                vector_reset.TERMINAL_REASON_NONE,
                vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW,
            ],
            dtype=np.int16,
        ),
        "reset_seed": np.asarray([1001, 9001], dtype=np.uint64),
        "reset_source": np.asarray(
            [
                vector_reset.RESET_SOURCE_FIXTURE,
                vector_reset.RESET_SOURCE_FIXTURE,
            ],
            dtype=np.int16,
        ),
        "tick": np.asarray([0, 99], dtype=np.int32),
        "elapsed_ms": np.asarray([0.0, 123.5], dtype=np.float64),
        "position": np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0]],
            ],
            dtype=np.float64,
        ),
        "event_count": np.asarray([0, 2], dtype=np.int16),
        "event_mask": np.zeros((2, event_capacity), dtype=bool),
        "event_type": np.zeros((2, event_capacity), dtype=np.int16),
        "event_player": np.full((2, event_capacity), -1, dtype=np.int16),
        "event_other": np.full((2, event_capacity), -1, dtype=np.int16),
        "event_bool": np.full((2, event_capacity), -1, dtype=np.int8),
        "event_value_i": np.zeros((2, event_capacity, 2), dtype=np.int32),
        "event_value_f": np.zeros((2, event_capacity, 2), dtype=np.float64),
        "event_overflow": np.asarray([False, True], dtype=bool),
        "event_overflow_attempts": np.asarray([0, 5], dtype=np.int32),
        "timer_fired_count": np.asarray([0, 3], dtype=np.int16),
    }
    reset_template["event_mask"][1, :2] = True
    reset_template["event_type"][1, :2] = [5, 6]
    reset_template["event_player"][1, :2] = [0, 1]

    target = {name: array.copy() for name, array in reset_template.items()}
    target.update(
        {
            "episode_id": np.asarray([10, 20], dtype=np.int64),
            "episode_step": np.asarray([3, 7], dtype=np.int32),
            "env_active": np.asarray([True, False], dtype=bool),
            "reset_pending": np.asarray([False, True], dtype=bool),
            "done": np.asarray([False, True], dtype=bool),
            "terminated": np.asarray([False, True], dtype=bool),
            "truncated": np.asarray([False, False], dtype=bool),
            "terminal_reason": np.asarray(
                [
                    vector_reset.TERMINAL_REASON_NONE,
                    vector_reset.TERMINAL_REASON_SURVIVOR_WIN,
                ],
                dtype=np.int16,
            ),
            "reset_seed": np.asarray([111, 222], dtype=np.uint64),
            "reset_source": np.asarray(
                [
                    vector_reset.RESET_SOURCE_REPLAY,
                    vector_reset.RESET_SOURCE_MANUAL,
                ],
                dtype=np.int16,
            ),
            "tick": np.asarray([5, 41], dtype=np.int32),
            "elapsed_ms": np.asarray([16.0, 250.0], dtype=np.float64),
            "timer_fired_count": np.asarray([1, 4], dtype=np.int16),
        }
    )
    target["position"][0] = [[111.0, 222.0], [333.0, 444.0]]
    target["position"][1] = [[123.0, 456.0], [789.0, 101.0]]
    target["event_count"][1] = 2
    target["event_mask"][1, :2] = True
    target["event_type"][1, :2] = [3, 4]
    target["event_player"][1, :2] = [0, -1]
    target["event_other"][1, :2] = [1, 1]
    target["event_bool"][1, :2] = [1, 0]
    target["event_value_i"][1, :2] = [[7, 8], [9, 10]]
    target["event_value_f"][1, :2] = [[1.25, 2.5], [3.75, 5.0]]
    target["event_overflow"][1] = True
    target["event_overflow_attempts"][1] = 4
    return reset_template, target


def test_reset_arrays_copies_selected_template_row_then_clears_reset_surfaces():
    reset_template, target = _two_row_reset_template_and_target()

    info = vector_reset.reset_arrays(
        target,
        reset_template,
        np.asarray([False, True], dtype=bool),
        reset_seed=555,
        reset_source=vector_reset.RESET_SOURCE_AUTORESET,
        snapshot_array_names=("done", "tick", "position", "event_count", "event_type"),
    )

    assert info["schema"] == vector_reset.RESET_INFO_SCHEMA_ID
    assert info["reset_count"] == 1
    np.testing.assert_array_equal(info["reset_rows"], np.asarray([1], dtype=np.int32))
    np.testing.assert_allclose(target["position"][1], reset_template["position"][1])
    assert int(target["episode_id"][1]) == 21
    assert int(target["episode_step"][1]) == 0
    assert bool(target["env_active"][1]) is True
    assert bool(target["reset_pending"][1]) is False
    assert bool(target["done"][1]) is False
    assert bool(target["terminated"][1]) is False
    assert bool(target["truncated"][1]) is False
    assert int(target["terminal_reason"][1]) == vector_reset.TERMINAL_REASON_NONE
    assert int(target["reset_seed"][1]) == 555
    assert int(target["reset_source"][1]) == vector_reset.RESET_SOURCE_AUTORESET
    assert int(target["tick"][1]) == 0
    assert float(target["elapsed_ms"][1]) == 0.0
    assert int(target["timer_fired_count"][1]) == 0
    assert int(target["event_count"][1]) == 0
    np.testing.assert_array_equal(target["event_mask"][1], np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(target["event_type"][1], np.zeros(3, dtype=np.int16))
    np.testing.assert_array_equal(target["event_player"][1], np.full(3, -1, dtype=np.int16))
    np.testing.assert_array_equal(target["event_other"][1], np.full(3, -1, dtype=np.int16))
    np.testing.assert_array_equal(target["event_bool"][1], np.full(3, -1, dtype=np.int8))
    np.testing.assert_array_equal(target["event_value_i"][1], np.zeros((3, 2), dtype=np.int32))
    np.testing.assert_array_equal(
        target["event_value_f"][1],
        np.zeros((3, 2), dtype=np.float64),
    )
    assert bool(target["event_overflow"][1]) is False
    assert int(target["event_overflow_attempts"][1]) == 0


def test_reset_arrays_leaves_skipped_rows_unchanged():
    reset_template, target = _two_row_reset_template_and_target()
    skipped_before = {name: array[0].copy() for name, array in target.items()}

    info = vector_reset.reset_arrays(
        target,
        reset_template,
        np.asarray([False, True], dtype=bool),
        reset_seed=555,
        reset_source=vector_reset.RESET_SOURCE_AUTORESET,
        snapshot_array_names=("tick",),
    )

    for name, before in skipped_before.items():
        np.testing.assert_array_equal(target[name][0], before)
    np.testing.assert_array_equal(
        info["reset_mask"],
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        info["reset_episode_id"],
        np.asarray([10, 21], dtype=np.int64),
    )


def test_reset_arrays_snapshots_terminal_rows_before_reset_and_returns_copies():
    reset_template, target = _two_row_reset_template_and_target()

    info = vector_reset.reset_arrays(
        target,
        reset_template,
        np.asarray([False, True], dtype=bool),
        reset_seed=555,
        reset_source=vector_reset.RESET_SOURCE_AUTORESET,
        snapshot_array_names=(
            "done",
            "terminated",
            "terminal_reason",
            "tick",
            "position",
            "event_count",
            "event_type",
        ),
    )

    snapshot = info["terminal_transition_snapshot"]
    assert snapshot["schema"] == vector_reset.TERMINAL_TRANSITION_SNAPSHOT_SCHEMA_ID
    np.testing.assert_array_equal(
        snapshot["final_mask"],
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(snapshot["final_rows"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(snapshot["arrays"]["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        snapshot["arrays"]["terminated"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(snapshot["arrays"]["tick"], np.asarray([41], dtype=np.int32))
    np.testing.assert_allclose(snapshot["arrays"]["position"][0, 0], [123.0, 456.0])
    np.testing.assert_array_equal(
        snapshot["arrays"]["event_count"],
        np.asarray([2], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["event_type"][0, :2],
        np.asarray([3, 4], dtype=np.int16),
    )

    target["position"][1, 0] = [0.0, 0.0]
    snapshot["arrays"]["event_count"][0] = 99
    np.testing.assert_allclose(snapshot["arrays"]["position"][0, 0], [123.0, 456.0])
    assert int(target["event_count"][1]) == 0


def test_reset_arrays_stamps_seed_and_source_from_selected_metadata_rows():
    reset_template, target = _two_row_reset_template_and_target()

    info = vector_reset.reset_arrays(
        target,
        reset_template,
        np.asarray([False, True], dtype=bool),
        reset_seed=np.asarray([333, 555], dtype=np.uint64),
        reset_source=np.asarray(
            [
                vector_reset.RESET_SOURCE_MANUAL,
                vector_reset.RESET_SOURCE_AUTORESET,
            ],
            dtype=np.int16,
        ),
        snapshot_array_names=("reset_seed", "reset_source"),
    )

    np.testing.assert_array_equal(
        target["reset_seed"],
        np.asarray([111, 555], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        target["reset_source"],
        np.asarray(
            [
                vector_reset.RESET_SOURCE_REPLAY,
                vector_reset.RESET_SOURCE_AUTORESET,
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(info["reset_seed"], target["reset_seed"])
    np.testing.assert_array_equal(info["reset_source"], target["reset_source"])


@pytest.mark.parametrize(
    ("reset_mask", "match"),
    (
        (np.asarray([1, 0], dtype=np.int8), "reset_mask must be a bool array"),
        (np.asarray([[False, True]], dtype=bool), "reset_mask must be a bool array"),
        (np.asarray([False, True, False], dtype=bool), "reset_mask shape"),
    ),
)
def test_reset_arrays_rejects_invalid_masks(reset_mask, match):
    reset_template, target = _two_row_reset_template_and_target()

    with pytest.raises(vector_reset.VectorResetError, match=match):
        vector_reset.reset_arrays(
            target,
            reset_template,
            reset_mask,
            reset_seed=123,
            reset_source=vector_reset.RESET_SOURCE_AUTORESET,
            snapshot_array_names=("tick",),
        )


@pytest.mark.parametrize(
    ("mutate", "kwargs", "match"),
    (
        (
            lambda reset_template, target: target.pop("episode_id"),
            {},
            "missing 'episode_id'",
        ),
        (
            lambda reset_template, target: target.__setitem__(
                "episode_step",
                np.asarray([0, 1], dtype=np.int64),
            ),
            {},
            "episode_step must be an int32 array",
        ),
        (
            lambda reset_template, target: target.__setitem__(
                "terminal_reason",
                np.asarray([vector_reset.TERMINAL_REASON_NONE, 99], dtype=np.int16),
            ),
            {},
            "known terminal reason",
        ),
        (
            lambda reset_template, target: None,
            {"reset_seed": -1},
            "reset_seed scalar must be non-negative",
        ),
        (
            lambda reset_template, target: None,
            {
                "reset_source": np.asarray(
                    [vector_reset.RESET_SOURCE_MANUAL, 99],
                    dtype=np.int16,
                )
            },
            "known reset source",
        ),
    ),
)
def test_reset_arrays_rejects_invalid_lifecycle_or_reset_metadata(mutate, kwargs, match):
    reset_template, target = _two_row_reset_template_and_target()
    mutate(reset_template, target)

    reset_kwargs = {
        "reset_seed": 123,
        "reset_source": vector_reset.RESET_SOURCE_MANUAL,
        **kwargs,
    }
    with pytest.raises(vector_reset.VectorResetError, match=match):
        vector_reset.reset_arrays(
            target,
            reset_template,
            np.asarray([False, True], dtype=bool),
            snapshot_array_names=("tick",),
            **reset_kwargs,
        )
