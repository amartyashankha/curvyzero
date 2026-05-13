from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_curvytron_mixture_status.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "analyze_curvytron_mixture_status_for_test",
        SCRIPT,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_row(
    *,
    run_id: str,
    recipe_id: str,
    render_role: str,
    render_mode: str,
    copy_index: int,
    seed: int,
) -> dict:
    render_token = "rf" if render_role == "fast" else "rb"
    return {
        "run_id": run_id,
        "recipe_id": recipe_id,
        "render_role": render_role,
        "source_state_trail_render_mode": render_mode,
        "copy_index": copy_index,
        "training_seed": seed,
        "base_token": f"{render_token}-s8-c32-l32-repM-k10",
        "base_settings": {
            "token": f"{render_token}-s8-c32-l32-repM-k10",
            "num_simulations": 8,
            "collector_env_num": 32,
            "n_episode": 32,
            "batch_size": 32,
            "save_ckpt_after_iter": 10000,
        },
    }


def test_summarize_counts_only_matched_pairs_at_target():
    module = _load_script()
    manifest = {
        "matrix_name": "mix-test",
        "rows": [
            _manifest_row(
                run_id="fast-ready",
                recipe_id="r50-blank50",
                render_role="fast",
                render_mode="body_circles_fast",
                copy_index=1,
                seed=123,
            ),
            _manifest_row(
                run_id="browser-ready",
                recipe_id="r50-blank50",
                render_role="browser",
                render_mode="browser_lines",
                copy_index=1,
                seed=123,
            ),
            _manifest_row(
                run_id="fast-unmatched",
                recipe_id="r50-blank50",
                render_role="fast",
                render_mode="body_circles_fast",
                copy_index=2,
                seed=124,
            ),
            _manifest_row(
                run_id="browser-unmatched",
                recipe_id="r50-blank50",
                render_role="browser",
                render_mode="browser_lines",
                copy_index=2,
                seed=124,
            ),
        ],
    }
    status_rows = [
        {
            "short_name": "fast-ready",
            "latest_checkpoint": "iteration_10000",
            "elapsed_sec": 100.0,
            "checkpoints": [
                {"iteration": 0, "mtime": 1000.0},
                {"iteration": 10000, "mtime": 1600.0},
            ],
        },
        {
            "short_name": "browser-ready",
            "latest_checkpoint": "iteration_10000",
            "elapsed_sec": 112.5,
            "checkpoints": [
                {"iteration": 0, "mtime": 1010.0},
                {"iteration": 10000, "mtime": 1630.0},
            ],
        },
        {
            "short_name": "fast-unmatched",
            "latest_checkpoint": "iteration_10000",
            "elapsed_sec": 101.0,
            "checkpoints": [
                {"iteration": 0, "mtime": 1000.0},
                {"iteration": 10000, "mtime": 1601.0},
            ],
        },
        {
            "short_name": "browser-unmatched",
            "latest_checkpoint": "iteration_0",
            "elapsed_sec": 10.0,
            "checkpoints": [{"iteration": 0, "mtime": 1011.0}],
        },
    ]

    summary = module.summarize(manifest, status_rows, target_iteration=10000)

    assert summary["rows_at_target"] == 3
    assert summary["matched_pair_count"] == 2
    assert summary["matched_pairs_at_target"] == 1
    assert summary["matched_pairs"] == [
        {
            "recipe_id": "r50-blank50",
            "copy_index": 1,
            "training_seed": 123,
            "repeat": "repM",
            "fast_elapsed_sec": 100.0,
            "browser_elapsed_sec": 112.5,
            "browser_minus_fast_sec": 12.5,
            "fast_checkpoint_gap_sec": 600.0,
            "browser_checkpoint_gap_sec": 620.0,
            "browser_minus_fast_checkpoint_gap_sec": 20.0,
            "browser_minus_fast_checkpoint_mtime_sec": 30.0,
            "fast_run_id": "fast-ready",
            "browser_run_id": "browser-ready",
        }
    ]
    assert summary["checkpoint_gap_delta_summary"]["median_browser_minus_fast_sec"] == 20.0


def test_load_json_with_modal_noise(tmp_path):
    module = _load_script()
    path = tmp_path / "status.json"
    path.write_text("Starting app\n" + json.dumps([{"short_name": "run"}]) + "\nDone\n")

    assert module._load_json_with_modal_noise(path) == [{"short_name": "run"}]
