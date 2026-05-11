import importlib.util
from pathlib import Path
import sys


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark_debug_visual_lightzero_adapter.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "benchmark_debug_visual_lightzero_adapter",
    _SCRIPT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
adapter_benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = adapter_benchmark
_SPEC.loader.exec_module(adapter_benchmark)


def test_debug_visual_lightzero_adapter_timing_report_contract():
    report = adapter_benchmark.run_benchmark(
        steps=3,
        seed=5,
        action_policy="fixed",
        fixed_action=1,
        source_step_ms=1000.0 / 60.0,
        source_max_steps=20,
        auto_reset_on_done=True,
    )

    assert report["schema_id"] == "curvyzero_debug_visual_lightzero_adapter_timing/v0"
    assert report["call_policy"] == "does_not_train; does_not_call_lzero_entrypoints"
    assert "No-train debug visual adapter timing only" in report["caveat"]
    assert report["adapter"]["lightzero_env_type"] == (
        "curvyzero_debug_visual_tensor_lightzero"
    )
    assert report["adapter"]["imports_lightzero_required"] is False
    assert report["workload"]["transitions"] == 3

    timed = report["timed_components"]
    assert timed["env_reset"] is True
    assert timed["env_step_total"] is True
    assert timed["render"] is True
    assert timed["stack"] is False
    assert timed["frame_stack_update_copy"] is False
    assert timed["policy_search"] is False
    assert timed["replay"] is False
    assert timed["learner"] is False

    assert report["workload"]["frame_stack_enabled"] is False
    assert report["workload"]["frame_stack_copy"] is False
    assert report["timing_sec"]["frame_stack_update_copy"] == 0.0
    assert report["throughput"]["transitions_per_sec_frame_stack_update_copy"] == 0.0
    assert report["obs_payload"]["obs_payload_shape"] == [1, 64, 64]
    assert report["obs_payload"]["obs_payload_dtype"] == "float32"
    assert report["obs_payload"]["frame_stack_shape"] is None
    assert report["obs_payload"]["frame_stack_dtype"] is None
    assert report["action_schema"]["n"] == 3
    assert report["action_schema"]["action_mask_shape"] == [3]
    assert report["reward_schema"]["shape"] == []
    assert report["reward_schema"]["dtype"] == "float32"
    assert report["throughput"]["transitions_per_sec_env_step_total"] > 0.0
    assert report["step_result"]["info_selected"]["surface"] == "debug_visual_tensor"
    assert report["step_result"]["info_selected"]["source_fidelity_level"] == "none"


def test_debug_visual_lightzero_adapter_can_time_frame_stack_without_copy():
    report = adapter_benchmark.run_benchmark(
        steps=3,
        seed=5,
        action_policy="fixed",
        fixed_action=1,
        source_step_ms=1000.0 / 60.0,
        source_max_steps=20,
        auto_reset_on_done=True,
        stack=True,
        stack_copy=False,
    )

    timed = report["timed_components"]
    assert timed["env_step_total"] is True
    assert timed["frame_stack_update_copy"] is True
    assert timed["policy_search"] is False
    assert timed["replay"] is False
    assert timed["learner"] is False

    assert report["workload"]["frame_stack_enabled"] is True
    assert report["workload"]["frame_stack_copy"] is False
    assert report["timing_sec"]["env_step_total"] > 0.0
    assert report["timing_sec"]["frame_stack_update_copy"] > 0.0
    assert report["obs_payload"]["step_shape"] == [1, 64, 64]
    assert report["obs_payload"]["obs_payload_shape"] == [4, 64, 64]
    assert report["obs_payload"]["obs_payload_dtype"] == "float32"
    assert report["obs_payload"]["frame_stack_shape"] == [4, 64, 64]
    assert report["obs_payload"]["frame_stack_dtype"] == "float32"
