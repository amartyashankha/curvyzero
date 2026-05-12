import importlib.util
import json
from pathlib import Path
import sys

import pytest

from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_BROWSER_LINES


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "benchmark_render_lane_microbench.py"
)
_SPEC = importlib.util.spec_from_file_location("benchmark_render_lane_microbench", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
render_bench = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = render_bench
_SPEC.loader.exec_module(render_bench)


def test_tiny_grid_probe_reports_canvas_schema_and_stack_timing():
    report = render_bench.run_grid_probe(
        batch_sizes=[1],
        player_counts=[2],
        trail_lengths=[0],
        bonus_counts=[0],
        trail_render_modes=[TRAIL_RENDER_MODE_BROWSER_LINES],
        kinds=[render_bench.FULL_STACK],
        iterations=1,
        warmup_iterations=0,
        allocation_mode="reuse",
        gpu_transfer="off",
        seed=7,
    )

    assert report["schema_id"] == render_bench.REPORT_SCHEMA_ID
    assert report["source_observation_schema_id"] == SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID
    assert report["cell_count"] == 1
    cell = report["cells"][0]
    assert cell["kind"] == render_bench.FULL_STACK
    assert cell["denominators"]["policy_rows"] == 2
    assert cell["denominators"]["stack_render_calls_estimated"] == 2
    assert cell["timing_sec"]["stack_update"] > 0.0
    assert cell["throughput"]["stack_update_us_per_policy_row"] > 0.0
    assert cell["integrity"]["last_stack_hash"] is not None
    assert cell["observation"]["stack_shape"] == [1, 2, 4, 64, 64]


def test_render_only_cell_times_gray64_render_without_normalize_or_stack():
    report = render_bench.run_grid_probe(
        batch_sizes=[1],
        player_counts=[2],
        trail_lengths=[0],
        bonus_counts=[0],
        trail_render_modes=[TRAIL_RENDER_MODE_BROWSER_LINES],
        kinds=[render_bench.GRAY64_RENDER_ONLY],
        iterations=1,
        warmup_iterations=0,
        allocation_mode="reuse",
        gpu_transfer="off",
        seed=11,
    )

    cell = report["cells"][0]
    assert cell["kind"] == render_bench.GRAY64_RENDER_ONLY
    assert cell["denominators"]["gray64_render_calls"] == 2
    assert cell["denominators"]["direct_normalized_frames"] == 0
    assert cell["timing_sec"]["gray64_render"] > 0.0
    assert cell["timing_sec"]["normalize"] == 0.0
    assert cell["timing_sec"]["stack_update"] == 0.0
    assert cell["integrity"]["last_raw_frame_hash"] is not None
    assert cell["integrity"]["last_stack_hash"] is None


def test_rgb_render_only_cell_times_rgb_draw_without_gray64_conversion():
    report = render_bench.run_grid_probe(
        batch_sizes=[1],
        player_counts=[2],
        trail_lengths=[0],
        bonus_counts=[0],
        trail_render_modes=[TRAIL_RENDER_MODE_BROWSER_LINES],
        kinds=[render_bench.RGB_RENDER_ONLY],
        iterations=1,
        warmup_iterations=0,
        allocation_mode="reuse",
        gpu_transfer="off",
        seed=17,
    )

    cell = report["cells"][0]
    assert cell["kind"] == render_bench.RGB_RENDER_ONLY
    assert cell["denominators"]["rgb_render_calls"] == 2
    assert cell["denominators"]["rgb_to_gray64_calls"] == 0
    assert cell["denominators"]["gray64_render_calls"] == 0
    assert cell["timing_sec"]["rgb_render"] > 0.0
    assert cell["timing_sec"]["rgb_to_gray64"] == 0.0
    assert cell["timing_sec"]["gray64_render"] == 0.0
    assert cell["integrity"]["last_rgb_frame_hash"] is not None
    assert cell["integrity"]["last_raw_frame_hash"] is None


def test_rgb_to_gray64_cell_times_conversion_from_prerendered_rgb():
    report = render_bench.run_grid_probe(
        batch_sizes=[1],
        player_counts=[2],
        trail_lengths=[0],
        bonus_counts=[0],
        trail_render_modes=[TRAIL_RENDER_MODE_BROWSER_LINES],
        kinds=[render_bench.RGB_TO_GRAY64_ONLY],
        iterations=1,
        warmup_iterations=0,
        allocation_mode="reuse",
        gpu_transfer="off",
        seed=19,
    )

    cell = report["cells"][0]
    assert cell["kind"] == render_bench.RGB_TO_GRAY64_ONLY
    assert cell["denominators"]["rgb_to_gray64_calls"] == 2
    assert cell["denominators"]["rgb_render_calls"] == 0
    assert cell["timing_sec"]["rgb_to_gray64"] > 0.0
    assert cell["timing_sec"]["rgb_render"] == 0.0
    assert cell["integrity"]["last_rgb_frame_hash"] is not None
    assert cell["integrity"]["last_raw_frame_hash"] is not None


def test_perspective_reuse_cell_times_all_player_perspectives():
    report = render_bench.run_grid_probe(
        batch_sizes=[1],
        player_counts=[2],
        trail_lengths=[0],
        bonus_counts=[0],
        trail_render_modes=[TRAIL_RENDER_MODE_BROWSER_LINES],
        kinds=[render_bench.PERSPECTIVE_REUSE_GRAY64],
        iterations=1,
        warmup_iterations=0,
        allocation_mode="reuse",
        gpu_transfer="off",
        seed=23,
    )

    cell = report["cells"][0]
    assert cell["kind"] == render_bench.PERSPECTIVE_REUSE_GRAY64
    assert cell["status"] == "ok"
    assert cell["denominators"]["perspective_reuse_gray64_calls"] == 1
    assert cell["denominators"]["perspective_reuse_gray64_frames"] == 2
    assert cell["denominators"]["gray64_render_calls"] == 0
    assert cell["timing_sec"]["perspective_reuse_gray64"] > 0.0
    assert cell["integrity"]["last_perspective_frames_hash"] is not None


def test_perspective_reuse_cell_reports_unsupported_when_helper_missing(monkeypatch):
    monkeypatch.setattr(
        render_bench,
        "_render_source_state_canvas_gray64_player_perspectives",
        None,
    )

    report = render_bench.run_grid_probe(
        batch_sizes=[1],
        player_counts=[2],
        trail_lengths=[0],
        bonus_counts=[0],
        trail_render_modes=[TRAIL_RENDER_MODE_BROWSER_LINES],
        kinds=[render_bench.PERSPECTIVE_REUSE_GRAY64],
        iterations=1,
        warmup_iterations=0,
        allocation_mode="reuse",
        gpu_transfer="off",
        seed=29,
    )

    cell = report["cells"][0]
    assert cell["status"] == "unsupported"
    assert "not importable" in cell["unsupported_reason"]
    assert cell["denominators"]["perspective_reuse_gray64_calls"] == 0
    assert cell["timing_sec"]["perspective_reuse_gray64"] == 0.0


def test_gpu_transfer_mode_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError, match="GPU transfer"):
        render_bench.run_grid_probe(
            batch_sizes=[1],
            player_counts=[2],
            trail_lengths=[0],
            bonus_counts=[0],
            trail_render_modes=[TRAIL_RENDER_MODE_BROWSER_LINES],
            kinds=[render_bench.FULL_STACK],
            iterations=1,
            warmup_iterations=0,
            allocation_mode="reuse",
            gpu_transfer="auto",
            seed=13,
        )


def test_cli_json_output_is_machine_readable(capsys):
    exit_code = render_bench.main(
        [
            "--plan",
            "grid",
            "--batch-sizes",
            "1",
            "--player-counts",
            "2",
            "--trail-lengths",
            "0",
            "--bonus-counts",
            "0",
            "--trail-render-modes",
            TRAIL_RENDER_MODE_BROWSER_LINES,
            "--cell-kinds",
            render_bench.FULL_STACK,
            "--iterations",
            "1",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_id"] == render_bench.REPORT_SCHEMA_ID
    assert payload["cell_count"] == 1
