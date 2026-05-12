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
run_full_2p_visual_gate = _MODULE.run_full_2p_visual_gate
run_final_observation_autoreset_visual_gate = (
    _MODULE.run_final_observation_autoreset_visual_gate
)
run_programmatic_stress_comparison = _MODULE.run_programmatic_stress_comparison
run_typed_bonus_visual_status_gate = _MODULE.run_typed_bonus_visual_status_gate
run_visual_mismatch_canary = _MODULE.run_visual_mismatch_canary
NATURAL_BONUS_2P_FIXTURE_IDS = _MODULE.NATURAL_BONUS_2P_FIXTURE_IDS
PROGRAMMATIC_2P_STRESS_SCENARIO_IDS = _MODULE.PROGRAMMATIC_2P_STRESS_SCENARIO_IDS
TYPED_BONUS_VISUAL_STATUS_GATE_TYPES = _MODULE.TYPED_BONUS_VISUAL_STATUS_GATE_TYPES
VISUAL_MISMATCH_CANARY_IDS = _MODULE.VISUAL_MISMATCH_CANARY_IDS


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

    assert report["suite_id"] == "core_2p_source_state_canvas_gray64"
    assert report["scenario_count"] == 35
    assert report["match"] is True
    assert report["max_abs_diff"] == 0
    assert report["mismatch_pixels"] == 0


def test_compare_2p_raw_visual_observation_full_visual_gate_matches():
    report = run_full_2p_visual_gate()

    assert report["gate_id"] == "full_2p_source_state_visual_gate"
    assert report["match"] is True
    assert report["gray64"]["scenario_count"] == 35
    assert report["typed_bonus"]["case_count"] == 12
    assert report["final_observation"]["match"] is True
    assert report["visual_canaries_passed"] == len(VISUAL_MISMATCH_CANARY_IDS)
    assert report["mismatch_pixels"] == 0
    assert report["max_abs_diff"] == 0.0
    assert report["expected_canary_mismatch_pixels"] > 0
    assert report["not_a_training_ready_claim"] is True


def test_compare_2p_raw_visual_observation_programmatic_stress_scenarios_match():
    expected_terminal = {
        "source_printing_trail_point_visual_stress": False,
        "source_body_opponent_tangent_then_overlap_visual_stress": True,
        "source_body_own_latency_delta3_then_delta4_visual_stress": True,
        "source_print_manager_trail_gap_boundary_visual_stress": False,
        "source_lifecycle_survivor_score_2p_warmdown_visual_stress": False,
        "source_bonus_self_master_body_block_then_wall_death_visual_stress": True,
        "source_bonus_game_borderless_expiry_then_wall_death_visual_stress": True,
        "source_bonus_game_borderless_same_frame_expiry_wall_death_visual_stress": True,
        "source_bonus_game_clear_clears_future_collision_body_visual_stress": False,
    }

    for scenario_id in PROGRAMMATIC_2P_STRESS_SCENARIO_IDS:
        report = run_programmatic_stress_comparison(scenario_id)

        assert report["comparison_kind"] == "programmatic_source_snapshot_stress"
        assert report["match"] is True
        assert report["max_abs_diff"] == 0
        assert report["mismatch_pixels"] == 0
        assert report["terminal_seen"] is expected_terminal[scenario_id]
        assert report["expected_terminal"] is expected_terminal[scenario_id]
        assert report["visual_limits"]


def test_compare_2p_raw_visual_observation_final_observation_autoreset_gate():
    report = run_final_observation_autoreset_visual_gate()

    assert (
        report["comparison_kind"]
        == "source_state_canvas_gray64_final_observation_autoreset_gate"
    )
    assert report["match"] is True
    assert report["terminal_seen"] is True
    assert report["final_observation_present"] is True
    assert report["final_action_mask_terminal"] is True
    assert report["terminal_frame_mismatch_pixels"] == 0
    assert report["returned_stack_mismatch_pixels"] == 0
    assert report["final_stack_survived_reset"] is True
    assert report["post_reset_frame_mismatch_pixels"] > 0


def test_compare_2p_raw_visual_observation_mismatch_canaries_fail_when_visible_fact_missing():
    for canary_id in VISUAL_MISMATCH_CANARY_IDS:
        report = run_visual_mismatch_canary(canary_id)

        assert report["comparison_kind"] == "intentional_visual_mismatch_canary"
        assert report["match"] is False
        assert report["max_abs_diff"] > 0
        assert report["mismatch_pixels"] > 0
        assert report["first_mismatch"] is not None
        assert report["expected_match"] is False
        assert report["expected_hole"]


def test_compare_2p_raw_visual_observation_typed_bonus_gate_matches_source_defaults():
    report = run_typed_bonus_visual_status_gate()

    assert report["comparison_kind"] == "typed_bonus_visual_status_gate"
    assert report["source_bonus_types"] == list(TYPED_BONUS_VISUAL_STATUS_GATE_TYPES)
    assert report["case_count"] == 12
    assert report["match"] is True
    assert report["max_abs_diff"] == 0.0
    assert report["mismatch_pixels"] == 0
    assert "active map bonus type code" in report["source_backed_status_scope"]
    assert report["missing_source_backed_proof"]
    for case in report["reports"]:
        assert case["comparison_kind"] == "typed_bonus_visual_status_case"
        assert case["match"] is True
        assert case["source_mask_at_center"] == 1.0
        assert case["vector_mask_at_center"] == 1.0
        assert case["source_type_at_center"] == case["expected_type_at_center"]
        assert case["vector_type_at_center"] == case["expected_type_at_center"]
        assert case["map_type_mismatch_pixels"] == 0
        assert case["status_mismatch_pixels"] == 0
        assert case["frames_compared"] == 2


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
