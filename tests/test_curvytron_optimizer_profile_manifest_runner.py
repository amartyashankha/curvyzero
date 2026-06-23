import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / (
    "run_curvytron_optimizer_profile_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("optimizer_profile_manifest_runner", SCRIPT_PATH)
assert SPEC is not None
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_profile_manifest_row_selection_accepts_old_and_new_numeric_widths():
    manifest = {
        "rows": [
            {"row_id": "01", "value": "old"},
            {"row_id": "002", "value": "new"},
        ]
    }

    assert [row["value"] for row in runner._selected_rows(manifest, runner._parse_rows(["1"]))] == [
        "old"
    ]
    assert [row["value"] for row in runner._selected_rows(manifest, runner._parse_rows(["2"]))] == [
        "new"
    ]


def test_profile_manifest_preflight_requires_compact_output_detail():
    manifest = {
        "rows": [
            {
                "row_id": "001",
                "command": [
                    "uv",
                    "run",
                    "--extra",
                    "modal",
                    "modal",
                    "run",
                    "-m",
                    runner.CURRENT_MODAL_MODULE,
                    "--mode",
                    "profile",
                    "--skip-lightzero-eval-in-profile",
                    "--no-background-eval-enabled",
                    "--no-background-gif-enabled",
                ],
            }
        ]
    }

    try:
        runner._validate_manifest(manifest, action="launch")
    except SystemExit as exc:
        assert "--output-detail compact" in str(exc)
    else:
        raise AssertionError("expected compact output detail preflight failure")


def test_profile_manifest_parser_prefers_top_level_compact_over_nested_schema():
    stdout = """
remote log
{
  "schema_id": "curvyzero_lightzero_curvytron_visual_survival_compact_output/v0",
  "ok": true,
  "semantic_identity": {
    "schema_id": "curvyzero_optimizer_profile_semantic_identity/v0",
    "called_train_muzero": true
  }
}
"""

    payload = runner._extract_last_json_object(stdout)

    assert payload["schema_id"] == (
        "curvyzero_lightzero_curvytron_visual_survival_compact_output/v0"
    )
    assert "ok" in payload


def test_profile_manifest_status_marks_compact_ok_false_as_profile_failed():
    compact = {"ok": False, "error": "collector failed before env steps"}

    status, problem = runner._profile_status_from_compact(compact)

    assert status == "profile_failed"
    assert problem == "collector failed before env steps"


def test_profile_manifest_result_line_prints_speed_currency_and_fallback_flag(capsys):
    payload = {
        "row_id": "001",
        "status": "complete",
        "problem": None,
        "compact": {
            "counts": {
                "env_steps_collected": 262144,
                "env_steps_collected_raw": 0,
                "env_steps_collected_source": "mcts_search_root_sum_profile_fallback",
            },
            "timers_sec": {"train_muzero_wall": 10.0},
            "derived": {
                "steps_per_sec": 26214.4,
                "steps_per_sec_currency": (
                    "stock_train_muzero_profile_mcts_roots_per_sec_fallback"
                ),
                "steps_per_sec_uses_fallback_denominator": True,
            },
        },
    }

    runner._print_result_line(payload)

    line = json.loads(capsys.readouterr().out)
    assert line["steps_per_sec_currency"] == (
        "stock_train_muzero_profile_mcts_roots_per_sec_fallback"
    )
    assert line["steps_per_sec_uses_fallback_denominator"] is True


def _matched_stock_row(**overrides):
    row = {
        "row_id": "001",
        "matched_denominator_id": "unit-denominator",
        "matched_pair_role": "stock_reference",
        "speed_currency": runner.MATCHED_STOCK_SPEED_CURRENCY,
        "counterpart_manifest_ref": "compact.json",
        "counterpart_row_id": "001",
        "row_purpose": runner.MATCHED_DENOMINATOR_ROW_PURPOSE,
        "promotion_claim": False,
        "fixed_denominator": {"speed_currency": runner.MATCHED_STOCK_SPEED_CURRENCY},
        "compute": "gpu-h100-cpu40",
        "env_manager_type": "subprocess",
        "collector_env_num": 512,
        "batch_size": 64,
        "num_simulations": 8,
        "collect_search_backend": "stock",
        "collect_search_ctree_backend": "lightzero",
        "exploration_bonus_mode": "none",
        "exploration_bonus_weight": 0.0,
        "source_max_steps": 512,
        "disable_death_for_profile": True,
        "command": [
            "uv",
            "run",
            "--extra",
            "modal",
            "modal",
            "run",
            "--detach",
            "-m",
            runner.CURRENT_MODAL_MODULE,
            "--mode",
            "profile",
            "--compute",
            "gpu-h100-cpu40",
            "--collector-env-num",
            "512",
            "--batch-size",
            "64",
            "--num-simulations",
            "8",
            "--collect-search-backend",
            "stock",
            "--collect-search-ctree-backend",
            "lightzero",
            "--exploration-bonus-mode",
            "none",
            "--source-max-steps",
            "512",
            "--stop-after-learner-train-calls",
            "12",
            "--skip-lightzero-eval-in-profile",
            "--no-background-eval-enabled",
            "--no-background-gif-enabled",
            "--output-detail",
            "compact",
            "--disable-death-for-profile",
            "--profile-spawn",
        ],
    }
    row.update(overrides)
    return row


def _matched_stock_compact(**overrides):
    compact = {
        "ok": True,
        "mode": "profile",
        "called_train_muzero": True,
        "trainer_entrypoint": "lzero.entry.train_muzero",
        "counts": {
            "env_steps_collected": 262144,
            "env_steps_collected_raw": 262144,
            "env_steps_collected_source": "collector_envstep_delta",
            "env_steps_collected_uses_fallback": False,
            "evaluator_eval_calls": 0,
            "learner_train_calls": 32,
            "replay_sample_calls": 32,
        },
        "derived": {
            "steps_per_sec": 950.0,
            "steps_per_sec_currency": runner.MATCHED_STOCK_SPEED_CURRENCY,
            "steps_per_sec_source": "collector_envstep_delta",
            "steps_per_sec_uses_fallback_denominator": False,
        },
    }
    compact.update(overrides)
    return compact


def test_profile_manifest_preflight_accepts_matched_stock_denominator_row():
    manifest = {"rows": [_matched_stock_row()]}

    runner._validate_manifest(manifest, action="launch-and-collect")


def test_profile_manifest_preflight_rejects_matched_stock_fallback_currency():
    row = _matched_stock_row(speed_currency="stock_train_muzero_profile_mcts_roots_per_sec_fallback")
    row["fixed_denominator"] = {"speed_currency": row["speed_currency"]}
    manifest = {"rows": [row]}

    try:
        runner._validate_manifest(manifest, action="launch-and-collect")
    except SystemExit as exc:
        assert "speed_currency" in str(exc)
    else:
        raise AssertionError("expected matched stock speed currency failure")


def test_profile_manifest_matched_stock_result_requires_raw_collector_steps():
    compact = _matched_stock_compact()

    status, problem = runner._profile_status_from_compact(
        compact,
        row=_matched_stock_row(),
    )

    assert status == "complete"
    assert problem is None

    compact["counts"]["env_steps_collected_source"] = "mcts_search_root_sum_profile_fallback"
    compact["counts"]["env_steps_collected_uses_fallback"] = True
    compact["derived"]["steps_per_sec_uses_fallback_denominator"] = True

    status, problem = runner._profile_status_from_compact(
        compact,
        row=_matched_stock_row(),
    )

    assert status == "matched_denominator_invariant_failed"
    assert "collector_envstep_delta" in problem
