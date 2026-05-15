from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "curvytron_tournament_debug_bundle.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "curvytron_tournament_debug_bundle_for_test",
        SCRIPT,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_summary_flags_manifest_rating_mismatch():
    module = _load_script()
    refs = [f"training/run-a/ckpt/iteration_{index}.pth.tar" for index in range(159)]
    manifest = {
        "schema_id": "curvyzero_curvytron_checkpoint_intake_manifest/v0",
        "tournament_id": "arena-a",
        "rating_run_id": "elo-live",
        "active": True,
        "checkpoint_refs": refs,
        "seen_checkpoint_refs": refs,
        "queued_checkpoint_refs": refs[-40:],
        "queue_partition": "q:arena-a:elo-live:test",
        "checkpoint_count": 159,
        "seen_checkpoint_count": 159,
        "queued_checkpoint_count": 40,
        "scan_spec": {"run_id_prefix": "curvy-survive", "checkpoint_selection": "latest"},
        "rating_defaults": {"continue_from_latest": False, "games_per_pair": 3},
        "queue_len": 0,
    }
    rating_config = {
        "schema_id": "curvyzero_curvytron_checkpoint_rating_config/v0",
        "tournament_id": "arena-a",
        "rating_run_id": "elo-live",
        "checkpoints": [{"checkpoint_ref": ref} for ref in refs[:9]],
        "round_count": 2,
        "continue_from_latest": False,
        "decision_source_frames": 4,
        "pair_selection": "adaptive_v0",
        "games_per_pair": 3,
        "games_per_shard": 3,
        "active_pool_limit": 100,
    }
    latest = {
        "schema_id": "curvyzero_curvytron_checkpoint_rating_snapshot/v0",
        "tournament_id": "arena-a",
        "rating_run_id": "elo-live",
        "checkpoint_count": 9,
        "pair_count": 8,
        "ratings": [
            {"checkpoint_id": f"ckpt-{index}", "status": "provisional"}
            for index in range(9)
        ],
    }

    summary = module.build_summary(
        intake_manifest=manifest,
        rating_config=rating_config,
        rating_latest=latest,
        rating_claim={
            "schema_id": "curvyzero_curvytron_checkpoint_intake_rating_claim/v0",
            "created_at": "2026-05-14T12:00:00Z",
            "queue_len_before": 0,
            "event_count": 0,
        },
    )

    assert summary["intake_manifest"]["checkpoint_count"] == 159
    assert summary["rating_config"]["checkpoint_count"] == 9
    assert summary["rating_latest"]["checkpoint_count"] == 9
    assert summary["rating_latest"]["game_count"] == 24
    assert any(
        warning == "manifest sees 159 checkpoints but rating config only has 9"
        for warning in summary["warnings"]
    )
    assert any(
        warning == "manifest sees 159 checkpoints but rating latest only has 9"
        for warning in summary["warnings"]
    )
    assert any("decision_source_frames=4" in warning for warning in summary["warnings"])
    assert any("continue_from_latest" in warning for warning in summary["warnings"])
    assert any("no active rows" in warning for warning in summary["warnings"])
    assert any("queue_len=0" in warning for warning in summary["warnings"])
    assert any("fresh rating claim exists" in warning for warning in summary["warnings"])
    assert any("event_count is smaller" in warning for warning in summary["warnings"])
    assert any("rating latest is smaller" in warning for warning in summary["warnings"])


def test_nested_rating_spec_config_reports_inner_settings():
    module = _load_script()
    nested_config = {
        "schema_id": "curvyzero_curvytron_rating_config_artifact/v0",
        "rating_spec": {
            "tournament_id": "arena-nested",
            "rating_run_id": "elo-nested",
            "checkpoints": [
                {"checkpoint_ref": f"training/run/ckpt/iteration_{index}.pth.tar"}
                for index in range(9)
            ],
            "round_count": 4,
            "continue_from_latest": True,
            "decision_source_frames": 1,
            "pair_selection": "adaptive_v0",
            "games_per_pair": 3,
            "games_per_shard": 3,
            "active_pool_limit": 100,
        },
    }

    summary = module.build_summary(rating_config=nested_config)

    config = summary["rating_config"]
    assert config["tournament_id"] == "arena-nested"
    assert config["rating_run_id"] == "elo-nested"
    assert config["checkpoint_count"] == 9
    assert config["round_count"] == 4
    assert config["continue_from_latest"] is True
    assert config["decision_source_frames"] == 1
    assert config["one_frame_decisions"] is True
    assert config["pair_selection"] == "adaptive_v0"
    assert config["games_per_pair"] == 3
    assert config["games_per_shard"] == 3
    assert config["active_pool_limit"] == 100


def test_cli_loads_wrapped_manifest_progress_and_prints_hints(tmp_path, capsys):
    module = _load_script()
    manifest_path = tmp_path / "intake_status.json"
    config_path = tmp_path / "rating_config.json"
    tick_path = tmp_path / "intake_progress.json"
    claim_path = tmp_path / "claim.json"
    queue_len_path = tmp_path / "queue_len.txt"
    manifest_path.write_text(
        "Starting app\n"
        + json.dumps(
            {
                "schema_id": "curvyzero_curvytron_checkpoint_intake_status/v0",
                "queue_len": 2,
                "manifest": {
                    "tournament_id": "arena-b",
                    "rating_run_id": "elo-b",
                    "checkpoint_refs": ["a", "b", "c"],
                    "seen_checkpoint_refs": ["a", "b", "c"],
                    "queued_checkpoint_refs": ["b", "c"],
                    "scan_spec": {"checkpoint_selection": "latest"},
                    "rating_defaults": {"continue_from_latest": True},
                },
            }
        )
        + "\nDone\n",
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "tournament_id": "arena-b",
                "rating_run_id": "elo-b",
                "checkpoints": [{"checkpoint_ref": "a"}, {"checkpoint_ref": "b"}],
                "continue_from_latest": True,
                "decision_source_frames": 1,
            }
        ),
        encoding="utf-8",
    )
    tick_path.write_text(
        json.dumps(
            {
                "schema_id": "curvyzero_curvytron_checkpoint_intake_tick_batch/v0",
                "tick_count": 1,
                "new_checkpoint_count": 2,
                "ticks": [{"new_checkpoint_count": 2}],
            }
        ),
        encoding="utf-8",
    )
    claim_path.write_text(
        json.dumps(
            {
                "schema_id": "curvyzero_curvytron_checkpoint_intake_rating_claim/v0",
                "created_at": "2026-05-14T12:34:56Z",
                "stale_after_seconds": 600,
                "queue_len_before": 0,
                "queue_len_after_repair": 0,
                "event_count": 1,
                "repaired_stale_claim": False,
            }
        ),
        encoding="utf-8",
    )
    queue_len_path.write_text("0\n", encoding="utf-8")

    exit_code = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--rating-config",
            str(config_path),
            "--claim",
            str(claim_path),
            "--claim-key",
            "rating_claim:manifest:arena-b:elo-b",
            "--queue-len-file",
            str(queue_len_path),
            "--intake-progress",
            str(tick_path),
            "--modal-hints",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "tournament_id: arena-b" in output
    assert "queued_checkpoint_count: 2" in output
    assert "Queue" in output
    assert "queue_len: 0" in output
    assert "Rating claim" in output
    assert "rating_claim_key: rating_claim:manifest:arena-b:elo-b" in output
    assert "stale_after_seconds: 600" in output
    assert "tick_count: 1" in output
    assert "manifest sees 3 checkpoints but rating config only has 2" in output
    assert "queue_len=0 but intake manifest still has 2 queued checkpoints" in output
    assert "fresh rating claim event_count is smaller than manifest seen count: 1 < 3" in output
    assert "uv run --extra modal modal volume get curvyzero-curvytron-tournaments-v2" in output
    assert "uv run --extra modal modal dict get curvyzero-curvytron-checkpoint-intake-v2 manifest:arena-b:elo-b" in output
    assert "uv run --extra modal modal queue len curvyzero-curvytron-checkpoint-events-v2 --partition q:arena-b:elo-b:" in output
    assert "tournaments/curvytron/arena-b/ratings/elo-b/latest.json" in output
