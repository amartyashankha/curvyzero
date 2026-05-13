import json
from pathlib import Path

import numpy as np
import pytest

from curvyzero.env import vector_runtime
from curvyzero.env.config import CurvyTronReferenceDefaults
from curvyzero.env.vector_multiplayer_env import NATURAL_BONUS_EFFECT_TYPE_NAMES
from curvyzero.env.vector_multiplayer_env import NATURAL_BONUS_TYPE_NAMES
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv


SCENARIO_ROOT = Path(__file__).resolve().parents[1] / "scenarios" / "environment"


def _source_random_values(scenario_name: str) -> tuple[float, ...]:
    payload = json.loads((SCENARIO_ROOT / scenario_name).read_text(encoding="utf-8"))
    sequence = payload["source_setup"]["random"]["math_random_sequence"]
    values = [entry["value"] if isinstance(entry, dict) else entry for entry in sequence]
    return tuple(float(value) for value in values)


def _public_default_bonus_tape(type_draw: float) -> np.ndarray:
    spawn_prefix = _source_random_values("source_lifecycle_spawn_rng_order_4p.json")
    natural_bonus_values = (
        0.0,  # bonus.start_delay
        0.5,  # bonus.next_delay_after_pop
        type_draw,
        0.25,  # bonus.position.x
        0.75,  # bonus.position.y
    )
    print_manager_start_values = (0.5, 0.5, 0.5, 0.5)
    return np.asarray(
        [[*spawn_prefix, *natural_bonus_values, *print_manager_start_values]],
        dtype=np.float64,
    )


@pytest.mark.parametrize(
    ("type_draw", "expected_name", "expected_code", "expected_weighted_draw"),
    (
        (0.945, "BonusAllColor", vector_runtime.BONUS_TYPE_ALL_COLOR, 10.8675),
        (0.965, "BonusGameClear", vector_runtime.BONUS_TYPE_GAME_CLEAR, 11.0975),
    ),
)
def test_public_default_bonus_spawn_uses_corrected_weights_and_rng_accounting(
    type_draw: float,
    expected_name: str,
    expected_code: int,
    expected_weighted_draw: float,
):
    tape = _public_default_bonus_tape(type_draw)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        random_tape_capacity=tape.shape[1],
        timer_capacity=8,
        event_mode="debug-event",
        natural_bonus_spawn=True,
    )
    reset_batch = env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_random_tape_values=tape,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    reset_cursor = int(reset_batch.info["random_tape_cursor"][0])

    assert reset_cursor == (
        len(_source_random_values("source_lifecycle_spawn_rng_order_4p.json")) + 1
    )
    assert reset_batch.info["bonus_support_mode"] == "natural_spawn"
    assert reset_batch.info["natural_bonus_reset_info"]["random_calls"][-1]["label"] == (
        "bonus.start_delay"
    )

    np.testing.assert_array_equal(env.state["present"], np.ones((1, 4), dtype=bool))
    env.state["alive"][0, 2:] = False
    env.decision_ms = 0.0

    batch = env.step(
        np.asarray([[1, 1, 1, 1]], dtype=np.int16),
        timer_advance_ms=np.asarray([3000.0], dtype=np.float64),
    )

    natural_info = batch.info["natural_bonus_info"]
    assert [call["label"] for call in natural_info["random_calls"]] == [
        "bonus.next_delay_after_pop",
        f"bonus.type.{expected_name}",
        "bonus.position.x",
        "bonus.position.y",
    ]
    assert [call["tape_index"] for call in natural_info["random_calls"]] == list(
        range(reset_cursor, reset_cursor + 4)
    )
    assert natural_info["schedule_calls"][0]["delay_ms"] == pytest.approx(4500.0)

    spawn_info = natural_info["spawn_infos"][0]
    type_info = spawn_info["type_selection_info"]
    assert type_info is not None
    np.testing.assert_array_equal(spawn_info["due_rows"], np.asarray([True]))
    np.testing.assert_allclose(type_info["game_clear_probability"], [0.5])
    np.testing.assert_allclose(type_info["total_weight"], [11.5])
    np.testing.assert_allclose(type_info["weighted_draw"], [expected_weighted_draw])
    assert str(spawn_info["selected_type_name"][0]) == expected_name
    assert int(spawn_info["selected_type_code"][0]) == expected_code
    np.testing.assert_allclose(spawn_info["spawned_pos"], [[27.255, 73.745]])
    assert int(env.state["bonus_type"][0, 0]) == expected_code

    assert batch.info["step_counters"]["print_manager_delayed_start_fires"] == 4
    assert int(batch.info["random_tape_cursor"][0]) == (
        reset_cursor
        + int(natural_info["random_tape_draws"])
        + int(batch.info["step_counters"]["random_tape_draws"])
    )
    assert int(batch.info["random_tape_cursor"][0]) == tape.shape[1]


def test_public_natural_bonus_metadata_keeps_full_source_default_set():
    reference_names = CurvyTronReferenceDefaults().default_bonus_types
    runtime_names = tuple(
        vector_runtime.BONUS_TYPE_NAME_BY_CODE[int(code)]
        for code in vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES
    )

    env = VectorMultiplayerEnv(batch_size=1, player_count=2, natural_bonus_spawn=True)
    batch = env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    support = batch.info["bonus_support"]

    assert len(reference_names) == 12
    assert "BonusSelfGodzilla" not in reference_names
    assert runtime_names == reference_names
    assert tuple(NATURAL_BONUS_TYPE_NAMES) == reference_names
    assert tuple(NATURAL_BONUS_EFFECT_TYPE_NAMES) == reference_names
    assert support["source_default_natural_bonus_types"] == reference_names
    assert support["supported_natural_bonus_types"] == reference_names
    np.testing.assert_array_equal(
        support["enabled_natural_bonus_type_codes"],
        np.asarray(vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES, dtype=np.int16),
    )
    assert support["unsupported_natural_bonus_types"] == ()
    assert support["unsupported_natural_bonus_effects"] == ()
