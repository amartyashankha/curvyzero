import importlib.util
from pathlib import Path
import sys


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "seed_vector_state_from_fixtures.py"
_SPEC = importlib.util.spec_from_file_location("seed_vector_state_from_fixtures", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
fixture_seed = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fixture_seed
_SPEC.loader.exec_module(fixture_seed)


def test_seeds_visible_trail_and_draw_cursor_separately():
    result = fixture_seed.seed_fixture(
        "scenarios/environment/source_trail_normal_point_step.json"
    )

    arrays = result["arrays"]

    assert result["verification"]["js_oracle_pinned"] is True
    assert result["verification"]["python_runner_verified"] is False
    assert arrays["visible_trail_count"]["values"] == [[0]]
    assert arrays["has_visible_trail_last"]["values"] == [[False]]
    assert arrays["has_draw_cursor"]["values"] == [[True]]
    assert arrays["draw_cursor_pos"]["values"] == [[[20.0, 40.0]]]
    assert arrays["body_pos"]["shape"] == [1, 0, 2]


def test_seeds_world_body_buffer_and_own_body_counters():
    result = fixture_seed.seed_fixture(
        "scenarios/environment/source_body_own_delta4_kills_step.json"
    )

    arrays = result["arrays"]

    assert result["verification"]["python_runner_verified"] is True
    assert result["profile"]["P"] == 3
    assert result["profile"]["K"] == 1
    assert arrays["body_active"]["values"] == [[True]]
    assert arrays["body_pos"]["values"] == [[[20.0, 20.0]]]
    assert arrays["body_owner"]["values"] == [[0]]
    assert arrays["body_num"]["values"] == [[0]]
    assert arrays["body_count"]["values"][0][0] == 4
    assert arrays["live_body_num"]["values"][0][0] == 4
    assert arrays["world_body_count"]["values"] == [1]
