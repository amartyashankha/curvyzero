import importlib.util
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "compare_2p_raw_visual_observation",
    REPO_ROOT / "scripts" / "compare_2p_raw_visual_observation.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load compare_2p_raw_visual_observation script")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
run_comparison = _MODULE.run_comparison
run_suite_comparison = _MODULE.run_suite_comparison
NATURAL_BONUS_2P_FIXTURE_IDS = _MODULE.NATURAL_BONUS_2P_FIXTURE_IDS


def test_compare_2p_raw_visual_observation_matches_short_rollout():
    report = run_comparison(max_steps=4, include_original_js_reset=False)

    assert report["source_arena_size"] == 88
    assert report["raw_observation_shape"] == [1, 64, 64]
    assert report["browser_canvas_pixel_fidelity"] is False
    assert report["match"] is True


def test_compare_2p_raw_visual_observation_matches_original_js_reset_when_available():
    if shutil.which("node") is None:
        return

    report = run_comparison(max_steps=0, include_original_js_reset=True)
    js_check = report["original_js_reset_check"]

    assert js_check["available"] is True
    assert js_check["loaded_original_source"] is True
    assert js_check["source_arena_size"] == 88
    assert js_check["match"] is True


def test_compare_2p_raw_visual_observation_core_suite_matches():
    report = run_suite_comparison()

    assert report["suite_id"] == "core_2p_source_state_gray64"
    assert report["scenario_count"] == 26
    assert report["match"] is True
    assert report["max_abs_diff"] == 0
    assert report["mismatch_pixels"] == 0


def test_compare_2p_raw_visual_observation_natural_bonus_fixtures_match():
    for fixture_id in sorted(NATURAL_BONUS_2P_FIXTURE_IDS):
        report = run_comparison(
            scenario_path=(
                REPO_ROOT / "scenarios" / "environment" / f"{fixture_id}.json"
            ),
            include_original_js_reset=False,
        )

        assert report["comparison_kind"] == "natural_bonus_spawn_fixture"
        assert report["match"] is True
        assert report["max_abs_diff"] == 0
        assert report["mismatch_pixels"] == 0
        assert report["vector_natural_bonus_pop_count"] == 1
        assert report["source_random_call_count"] == report["vector_random_tape_cursor"]
