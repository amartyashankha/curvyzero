from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from curvyzero.training.opponent_leaderboard import validate_assignment_audit


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_curvytron_next_batch_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_curvytron_next_batch_manifest_for_test",
        SCRIPT,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ratings_snapshot(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "ratings": [
                    {
                        "rank": rank,
                        "checkpoint_id": f"ckpt-rank{rank}",
                        "rating": 2000.0 - rank,
                        "status": "active",
                        "run_id": f"source-run-{rank}",
                        "attempt_id": f"try-source-run-{rank}",
                        "iteration": 10000 * rank,
                        "checkpoint_ref": (
                            "training/lightzero-curvytron-visual-survival/"
                            f"source-run-{rank}/attempts/try-source-run-{rank}/train/"
                            f"lightzero_exp/ckpt/iteration_{10000 * rank}.pth.tar"
                        ),
                    }
                    for rank in range(1, 5)
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_next_batch_full_manifest_has_locked_grid_shape_and_clean_names(tmp_path: Path) -> None:
    module = _load_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    args = module.parse_args(
        [
            "--profile",
            "full",
            "--matrix-name",
            "cz26-full-test",
            "--ratings-snapshot",
            str(snapshot),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    manifest = module.build_manifest(args)

    assert manifest["row_count"] == 136
    assert manifest["grid_a_row_count"] == 96
    assert manifest["grid_b_row_count"] == 40
    assert manifest["canary_row_count"] == 0
    assert manifest["guards"]["all_rows_seed_from_rank1"] is True

    rank1_ref = manifest["top_checkpoint_source"]["rank1"]["checkpoint_ref"]
    assert rank1_ref.endswith("iteration_10000.pth.tar")
    assert {
        row["train_kwargs"]["initial_policy_checkpoint_ref"]
        for row in manifest["rows"]
    } == {rank1_ref}

    grid_a_rows = [row for row in manifest["rows"] if row["grid"] == "grid_a"]
    grid_b_rows = [row for row in manifest["rows"] if row["grid"] == "grid_b"]
    assert grid_a_rows[0]["run_id"] == "cz26a-r001-out0-n0-imm0-b20w05r1"
    assert grid_a_rows[-1]["run_id"] == "cz26a-r096-out100-n20-imm10-b20w05top2"
    assert grid_b_rows[0]["run_id"] == "cz26b-r001-out50-n0-imm0-b100"
    assert grid_b_rows[-1]["run_id"] == "cz26b-r040-out50-n10-imm10-b20w05lad4"
    assert grid_b_rows[0]["row_id"] == "r097"
    assert grid_b_rows[0]["grid_row_id"] == "r001"
    intake_seed_spec = manifest["tournament"]["intake_seed_spec"]
    assert intake_seed_spec["checkpoint_refs"] == manifest["tournament"]["seed_checkpoint_refs"]
    assert intake_seed_spec["checkpoint_selection"] == "all"
    assert intake_seed_spec["run_id_prefix"] == ""
    assert intake_seed_spec["run_ids"] == sorted(row["run_id"] for row in manifest["rows"])
    assert len(intake_seed_spec["run_ids"]) == 136

    assignments = manifest["assignment_bank"]["assignments"]
    refresh_pointers = manifest["assignment_bank"]["refresh_pointers"]
    config_refs = set(
        manifest["training_candidate_refresh_controller"]["config"]["refresh_pointers"]
    )
    row_pointer_refs = {row["opponent_assignment_refresh_ref"] for row in manifest["rows"]}
    assert len(assignments) == 24
    assert len(refresh_pointers) == 24
    assert row_pointer_refs <= config_refs
    for artifact in assignments.values():
        validate_assignment_audit(artifact["audit"], assignment=artifact["assignment"])


def test_next_batch_canary_uses_canary_batch_and_one_current_row(tmp_path: Path) -> None:
    module = _load_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    args = module.parse_args(
        [
            "--profile",
            "canary",
            "--matrix-name",
            "cz26-canary-test",
            "--ratings-snapshot",
            str(snapshot),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    manifest = module.build_manifest(args)

    assert manifest["row_count"] == 1
    row = manifest["rows"][0]
    assert row["grid"] == "canary"
    assert row["run_id"] == "cz26c-r001-out100-n0-imm0-b20w05r1"
    assert row["grid_row_id"] == "r001"
    assert manifest["tournament"]["run_id_prefixes"] == ["cz26c-"]
    intake_seed_spec = manifest["tournament"]["intake_seed_spec"]
    assert intake_seed_spec["checkpoint_refs"] == manifest["tournament"]["seed_checkpoint_refs"]
    assert intake_seed_spec["checkpoint_selection"] == "latest"
    assert intake_seed_spec["run_ids"] == [row["run_id"]]
    assert intake_seed_spec["run_id_prefix"] == ""


def test_next_batch_canary_can_make_cadence_explicit_for_e2e_probe(
    tmp_path: Path,
) -> None:
    module = _load_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    args = module.parse_args(
        [
            "--profile",
            "canary",
            "--matrix-name",
            "cz26-canary-fast-test",
            "--ratings-snapshot",
            str(snapshot),
            "--output-root",
            str(tmp_path / "out"),
            "--save-ckpt-after-iter",
            "100",
            "--assignment-refresh-interval-train-iter",
            "100",
        ]
    )

    manifest = module.build_manifest(args)
    row = manifest["rows"][0]

    assert manifest["fixed_knobs"]["save_ckpt_after_iter"] == 100
    assert manifest["fixed_knobs"]["assignment_refresh_interval_train_iter"] == 100
    assert row["train_kwargs"]["save_ckpt_after_iter"] == 100
    assert row["train_kwargs"]["opponent_assignment_refresh_interval_train_iter"] == 100


def test_next_batch_immortal_probability_is_a_slot_flag_not_a_policy_name(
    tmp_path: Path,
) -> None:
    module = _load_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    args = module.parse_args(
        [
            "--profile",
            "grid-b",
            "--matrix-name",
            "cz26-gridb-test",
            "--ratings-snapshot",
            str(snapshot),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    manifest = module.build_manifest(args)
    artifact = manifest["assignment_bank"]["assignments"]["b20w05r1-imm10"]
    entries = artifact["assignment"]["entries"]
    rank_entries = [entry for entry in entries if entry["tags"].get("source_slot") == "rank1"]

    assert {entry["name"] for entry in rank_entries} == {"rank1", "rank1_immortal"}
    assert artifact["audit"]["selection"]["recipe_code"] == "b20w05r1"
    assert artifact["audit"]["selection"]["leaderboard_immortal_probability"] == 0.10
    assert artifact["audit"]["selection"]["leaderboard_immortal_slots"] == 5.0
    assert artifact["audit"]["selection"]["hardcoded_immortal_slots"] == 16.0
    assert artifact["audit"]["selection"]["total_immortal_slots"] == 21.0
