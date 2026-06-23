from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_curvytron_tonight18_manifest.py"
SUBMIT_SCRIPT = ROOT / "scripts" / "submit_curvytron_survivaldiag_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_curvytron_tonight18_manifest_for_test",
        SCRIPT,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_submitter_module():
    spec = importlib.util.spec_from_file_location(
        "submit_curvytron_survivaldiag_manifest_for_test",
        SUBMIT_SCRIPT,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expanded_train_kwargs(module, row: dict) -> dict:
    return module._expand_train_kwargs_for_manifest(row["train_kwargs"])


def _ratings_snapshot(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "ratings": [
                    {
                        "rank": rank,
                        "checkpoint_id": f"ckpt-rank{rank}",
                        "rating": 1600.0 - rank,
                        "status": "active",
                        "run_id": f"run-rank{rank}",
                        "attempt_id": f"attempt-rank{rank}",
                        "iteration": 10000 * rank,
                        "checkpoint_ref": (
                            "training/lightzero-curvytron-visual-survival/"
                            f"run-rank{rank}/attempts/attempt-rank{rank}/train/"
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


def _checkpoint_refs_file(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                (
                    "training/lightzero-curvytron-visual-survival/"
                    f"curated-run-{rank}/attempts/curated-attempt-{rank}/train/"
                    f"lightzero_exp/ckpt/iteration_{10000 * rank}.pth.tar"
                )
                for rank in range(1, 5)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_tonight18_manifest_uses_normal_checkpoints_plus_small_immortal_wall_slot(
    tmp_path: Path,
):
    module = _load_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    args = module.parse_args(
        [
            "--ratings-snapshot",
            str(snapshot),
            "--output-root",
            str(tmp_path / "out"),
            "--opponent-source",
            "mixture",
            "--assignment-refresh-interval-train-iter",
            "0",
        ]
    )

    manifest = module.build_manifest(args)

    assert len(manifest["rows"]) == 18
    assert [recipe["recipe_id"] for recipe in manifest["axes"]["opponent_recipes"]] == [
        "slot64-blank8-wall8-rank2_16-rank1_32",
        "slot64-blank8-wall8-rank4_6-rank3_8-rank2_12-rank1_20-rank1imm2",
        "slot64-blank12-wall4-rank1_46-rank1imm2",
    ]
    assert manifest["fixed_knobs"]["learner_seat_mode"] == "random_per_episode"
    assert manifest["fixed_knobs"]["compute"] == "gpu-l4-t4-cpu40"
    assert manifest["fixed_knobs"]["collector_env_num"] == 256
    assert manifest["fixed_knobs"]["n_episode"] == 256
    assert manifest["fixed_knobs"]["num_simulations"] == 8
    assert manifest["fixed_knobs"]["batch_size"] == 64

    for row in manifest["rows"]:
        assert len(row["run_id"]) <= module.MAX_MODAL_RUN_ID_LEN
        assert len(row["attempt_id"]) <= module.MAX_MODAL_RUN_ID_LEN
        assert row["source_max_steps"] == 1_048_576
        assert row["compute"] == "gpu-l4-t4-cpu40"
        assert (
            row["deployed_app_submission"]["train_function"]
            == "lightzero_curvytron_visual_survival_gpu_cpu40"
        )
        raw_train_kwargs = row["train_kwargs"]
        train_kwargs = _expanded_train_kwargs(module, row)
        poller_kwargs = row["poller_kwargs"]
        assert row["train_kwargs_schema_id"] == module.TONIGHT18_COMPACT_TRAIN_KWARGS_SCHEMA_ID
        assert "collector_env_num" not in raw_train_kwargs
        assert "source_max_steps" not in raw_train_kwargs
        assert "learner_seat_mode" not in raw_train_kwargs
        assert "source_state_trail_render_mode" not in raw_train_kwargs
        assert "source_state_bonus_render_mode" not in raw_train_kwargs
        assert train_kwargs["collector_env_num"] == 256
        assert train_kwargs["n_episode"] == 256
        assert train_kwargs["num_simulations"] == 8
        assert train_kwargs["batch_size"] == 64
        assert train_kwargs["source_max_steps"] == 1_048_576
        assert poller_kwargs["source_max_steps"] == 1_048_576
        assert row["learner_seat_mode"] == "random_per_episode"
        assert train_kwargs["learner_seat_mode"] == "random_per_episode"
        assert "learner_seat_mode" not in poller_kwargs
        assert row["initial_policy_checkpoint_ref"].endswith("iteration_10000.pth.tar")
        assert train_kwargs["initial_policy_checkpoint_ref"] == row["initial_policy_checkpoint_ref"]
        assert train_kwargs["initial_policy_checkpoint_load_mode"] == "matching_shape"
        assert "initial_policy_checkpoint_ref" not in poller_kwargs
        mixture = row["opponent_mixture_spec"]
        assert mixture["total_weight"] == 64.0
        wall_entries = [
            entry for entry in mixture["entries"] if entry["name"] == "wall_avoidant_immortal"
        ]
        assert len(wall_entries) == 1
        assert wall_entries[0]["opponent_policy_kind"] == "proactive_wall_avoidant"
        assert wall_entries[0]["opponent_runtime_mode"] == "normal"
        assert wall_entries[0]["opponent_immortal"] is True
        assert "opponent_death_mode" not in wall_entries[0]
        assert wall_entries[0]["opponent_wall_avoidant_safe_margin"] == 20.0
        blank = next(entry for entry in mixture["entries"] if entry["name"] == "blank")
        assert blank["opponent_runtime_mode"] == "blank_canvas_noop"
        assert blank["opponent_immortal"] is True
        pressure = sum(
            entry["weight"]
            for entry in mixture["entries"]
            if entry["name"] in {"blank", "wall_avoidant_immortal"}
        )
        assert pressure / mixture["total_weight"] >= 0.20
        immortal_pressure = sum(
            entry["weight"] for entry in mixture["entries"] if entry["opponent_immortal"]
        )
        assert 0.20 <= immortal_pressure / mixture["total_weight"] <= 0.30
        frozen_entries = [
            entry
            for entry in mixture["entries"]
            if entry["opponent_policy_kind"] == "frozen_lightzero_checkpoint"
        ]
        assert frozen_entries
        assert all(
            (not entry["opponent_immortal"]) or entry["name"].endswith("_immortal")
            for entry in frozen_entries
        )
        assert all("opponent_death_mode" not in entry for entry in frozen_entries)


def test_tonight18_manifest_defaults_to_control_assignments_with_refresh(
    tmp_path: Path,
):
    module = _load_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    args = module.parse_args(
        [
            "--ratings-snapshot",
            str(snapshot),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    manifest = module.build_manifest(args)

    assert manifest["opponent_source"] == "assignment"
    assert manifest["assignment_bank"]["target_volume"] == "control"
    assert manifest["assignment_bank"]["refresh_pointer_volume"] == "control"
    assert (
        manifest["fixed_knobs"]["assignment_refresh_interval_train_iter"]
        == module.ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER
    )
    for row in manifest["rows"]:
        train_kwargs = _expanded_train_kwargs(module, row)
        assert row["opponent_mixture_spec"] is None
        assert row["opponent_assignment_ref"].startswith("control:")
        assert row["opponent_assignment_refresh_ref"].startswith("control:")
        assert train_kwargs["opponent_assignment_ref"] == row["opponent_assignment_ref"]
        assert (
            train_kwargs["opponent_assignment_refresh_ref"]
            == row["opponent_assignment_refresh_ref"]
        )
        assert (
            train_kwargs["opponent_assignment_refresh_interval_train_iter"]
            == module.ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER
        )


def test_tonight18_manifest_can_bootstrap_from_explicit_checkpoint_refs(
    tmp_path: Path,
):
    module = _load_module()
    refs_file = _checkpoint_refs_file(tmp_path / "refs.txt")
    args = module.parse_args(
        [
            "--checkpoint-refs-file",
            str(refs_file),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    manifest = module.build_manifest(args)

    assert manifest["ratings_snapshot_path"] is None
    assert manifest["checkpoint_refs_file_path"] == str(refs_file)
    assert (
        manifest["fixed_knobs"]["initial_policy_checkpoint_source"]
        == "rank1_checkpoint_from_checkpoint_refs_file"
    )
    assert manifest["top_checkpoint_source"]["rank1"]["status"] == "curated_exact_ref"
    assert manifest["top_checkpoint_source"]["rank1"]["iteration"] == 10000
    assert manifest["top_checkpoint_source"]["rank4"]["iteration"] == 40000
    for row in manifest["rows"]:
        assert row["initial_policy_checkpoint_source"]["source"] == (
            "checkpoint_refs_file_rank1_at_manifest_build_time"
        )
        assert row["initial_policy_checkpoint_ref"].endswith("iteration_10000.pth.tar")
        assert row["opponent_assignment_ref"].startswith("control:")
        entries = row["opponent_assignment_preview"]["entries"]
        assert any(entry["name"] == "rank1_immortal" for entry in entries) or any(
            entry["name"] == "wall_avoidant_immortal" for entry in entries
        )
        immortal_pressure = sum(entry["weight"] for entry in entries if entry["opponent_immortal"])
        total_slots = sum(entry["weight"] for entry in entries)
        assert total_slots == 64
        assert 0.20 <= immortal_pressure / total_slots <= 0.30


def test_tonight18_manifest_can_scratch_bootstrap_without_deleted_checkpoint_refs(
    tmp_path: Path,
):
    module = _load_module()
    args = module.parse_args(
        [
            "--scratch-bootstrap",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    manifest = module.build_manifest(args)

    assert manifest["scratch_bootstrap"] is True
    assert manifest["ratings_snapshot_path"] is None
    assert manifest["checkpoint_refs_file_path"] is None
    assert manifest["top_checkpoint_source"] == {}
    assert (
        manifest["fixed_knobs"]["initial_policy_checkpoint_source"]
        == "scratch_random_initialization"
    )
    assert manifest["opponent_source"] == "assignment"
    assert manifest["assignment_bank"]["source_ref"] == "scratch_bootstrap"
    for row in manifest["rows"]:
        assert row["initial_policy_checkpoint_ref"] is None
        assert "initial_policy_checkpoint_ref" not in row["train_kwargs"]
        assert _expanded_train_kwargs(module, row)["initial_policy_checkpoint_ref"] is None
        assert row["initial_policy_checkpoint_source"] == {
            "source": "scratch_random_initialization",
            "checkpoint_ref": None,
        }
        assert "initial_policy_checkpoint_ref" not in row["poller_kwargs"]
        entries = row["opponent_assignment_preview"]["entries"]
        rank_entries = [entry for entry in entries if str(entry["name"]).startswith("rank")]
        assert rank_entries
        assert all(
            entry["opponent_policy_kind"] == "proactive_wall_avoidant" for entry in rank_entries
        )
        assert all("opponent_checkpoint_ref" not in entry for entry in rank_entries)
        assert {entry["tags"]["rank"] for entry in rank_entries} <= {1, 2, 3, 4}
        assert all(entry["tags"]["scratch_bootstrap_placeholder"] is True for entry in rank_entries)
        immortal_pressure = sum(entry["weight"] for entry in entries if entry["opponent_immortal"])
        total_slots = sum(entry["weight"] for entry in entries)
        assert total_slots == 64
        assert 0.20 <= immortal_pressure / total_slots <= 0.30


def test_tonight18_manifest_can_use_assignment_refs_without_inline_mixtures(
    tmp_path: Path,
):
    module = _load_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    matrix_name = "curvy-night18-connected-test"
    args = module.parse_args(
        [
            "--ratings-snapshot",
            str(snapshot),
            "--output-root",
            str(tmp_path / "out"),
            "--matrix-name",
            matrix_name,
            "--run-prefix",
            "curvy-n18conn-test",
            "--attempt-prefix",
            "try-n18conn-test",
            "--opponent-source",
            "assignment",
            "--assignment-bank-run-id",
            "curvy-n18conn-test-assignments",
            "--assignment-bank-attempt-id",
            "try-n18conn-test-assignments",
        ]
    )

    manifest = module.build_manifest(args)
    outputs = module._write_outputs(
        manifest,
        output_root=tmp_path / "out",
        matrix_name=matrix_name,
    )

    assert len(manifest["rows"]) == 18
    assert manifest["opponent_source"] == "assignment"
    assert manifest["assignment_bank"]["run_id"] == "curvy-n18conn-test-assignments"
    assert set(manifest["assignment_bank"]["assignments"]) == {
        "slot64-blank8-wall8-rank2_16-rank1_32",
        "slot64-blank8-wall8-rank4_6-rank3_8-rank2_12-rank1_20-rank1imm2",
        "slot64-blank12-wall4-rank1_46-rank1imm2",
    }
    assert Path(outputs["assignments_index_json"]).exists()

    for row in manifest["rows"]:
        raw_train_kwargs = row["train_kwargs"]
        train_kwargs = _expanded_train_kwargs(module, row)
        poller_kwargs = row["poller_kwargs"]
        assert row["source_max_steps"] == 1_048_576
        assert "source_max_steps" not in raw_train_kwargs
        assert "learner_seat_mode" not in raw_train_kwargs
        assert train_kwargs["source_max_steps"] == 1_048_576
        assert poller_kwargs["source_max_steps"] == 1_048_576
        assert row["learner_seat_mode"] == "random_per_episode"
        assert train_kwargs["learner_seat_mode"] == "random_per_episode"
        assert "learner_seat_mode" not in poller_kwargs
        assert row["initial_policy_checkpoint_ref"].endswith("iteration_10000.pth.tar")
        assert train_kwargs["initial_policy_checkpoint_ref"] == row["initial_policy_checkpoint_ref"]
        assert train_kwargs["initial_policy_checkpoint_load_mode"] == "matching_shape"
        assert "initial_policy_checkpoint_ref" not in poller_kwargs
        assert row["opponent_mixture_enabled"] is False
        assert row["opponent_mixture_spec"] is None
        assert train_kwargs["opponent_mixture_spec"] is None
        assert poller_kwargs["opponent_mixture_spec"] is None
        assert train_kwargs["opponent_assignment_ref"]
        assert train_kwargs["opponent_assignment_ref"] == poller_kwargs["opponent_assignment_ref"]
        assignment = row["opponent_assignment_preview"]
        entries = assignment["entries"]
        assert {entry["name"] for entry in entries} >= {"wall_avoidant_immortal"}
        wall = next(entry for entry in entries if entry["name"] == "wall_avoidant_immortal")
        assert wall["opponent_policy_kind"] == "proactive_wall_avoidant"
        assert wall["opponent_immortal"] is True
        assert "opponent_death_mode" not in wall
        immortal_pressure = sum(entry["weight"] for entry in entries if entry["opponent_immortal"])
        total_slots = sum(entry["weight"] for entry in entries)
        assert total_slots == 64
        assert 0.20 <= immortal_pressure / total_slots <= 0.30


def test_assignment_manifest_can_use_per_recipe_control_refresh_pointers(
    tmp_path: Path,
):
    module = _load_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    matrix_name = "curvy-night18-refresh-test"
    args = module.parse_args(
        [
            "--ratings-snapshot",
            str(snapshot),
            "--output-root",
            str(tmp_path / "out"),
            "--matrix-name",
            matrix_name,
            "--run-prefix",
            "curvy-n18refresh-test",
            "--attempt-prefix",
            "try-n18refresh-test",
            "--opponent-source",
            "assignment",
            "--assignment-bank-run-id",
            "curvy-n18refresh-test-assignments",
            "--assignment-bank-attempt-id",
            "try-n18refresh-test-assignments",
            "--assignment-target-volume",
            "control",
            "--assignment-refresh-interval-train-iter",
            "25",
            "--assignment-refresh-pointer-run-id",
            "curvy-n18refresh-test-control",
            "--assignment-refresh-pointer-attempt-id",
            "try-n18refresh-test-control",
        ]
    )

    manifest = module.build_manifest(args)
    outputs = module._write_outputs(
        manifest,
        output_root=tmp_path / "out",
        matrix_name=matrix_name,
    )

    assert manifest["assignment_bank"]["target_volume"] == "control"
    assert manifest["assignment_bank"]["refresh_pointer_volume"] == "control"
    assert set(manifest["assignment_bank"]["refresh_pointers"]) == set(
        manifest["assignment_bank"]["assignments"]
    )
    index = json.loads(Path(outputs["assignments_index_json"]).read_text())
    refresh_refs = set()
    assignment_refs = set()
    for row in manifest["rows"]:
        train_kwargs = _expanded_train_kwargs(module, row)
        assignment_ref = train_kwargs["opponent_assignment_ref"]
        refresh_ref = train_kwargs["opponent_assignment_refresh_ref"]
        assignment_refs.add(assignment_ref)
        refresh_refs.add(refresh_ref)
        assert assignment_ref.startswith("control:")
        assert refresh_ref.startswith("control:")
        assert refresh_ref != assignment_ref
        assert train_kwargs["opponent_assignment_refresh_interval_train_iter"] == 25
    assert len(assignment_refs) == 3
    assert len(refresh_refs) == 3
    for recipe_id, entry in index.items():
        pointer_path = Path(entry["refresh_pointer_json"])
        pointer_payload = json.loads(pointer_path.read_text())
        assert entry["assignment_ref"].startswith("control:")
        assert entry["refresh_pointer_ref"].startswith("control:")
        assert pointer_payload["assignment_ref"] == entry["assignment_ref"]
        assert pointer_payload["schema_id"] == "curvyzero_opponent_assignment_refresh_pointer/v0"
        assert recipe_id in manifest["assignment_bank"]["refresh_pointers"]


@pytest.mark.parametrize("seat_mode", ["fixed_player_0", "fixed_player_1"])
def test_tonight18_manifest_can_emit_explicit_fixed_seat_diagnostic_mode(
    tmp_path: Path,
    seat_mode: str,
):
    module = _load_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    args = module.parse_args(
        [
            "--ratings-snapshot",
            str(snapshot),
            "--output-root",
            str(tmp_path / "out"),
            "--learner-seat-mode",
            seat_mode,
        ]
    )

    manifest = module.build_manifest(args)

    assert manifest["fixed_knobs"]["learner_seat_mode"] == seat_mode
    for row in manifest["rows"]:
        assert row["learner_seat_mode"] == seat_mode
        assert row["train_kwargs"]["learner_seat_mode"] == seat_mode
        assert "learner_seat_mode" not in row["poller_kwargs"]


def test_tonight18_manifest_can_override_lightzero_cadence_and_target_knobs(
    tmp_path: Path,
):
    module = _load_module()
    refs_file = _checkpoint_refs_file(tmp_path / "refs.txt")
    args = module.parse_args(
        [
            "--checkpoint-refs-file",
            str(refs_file),
            "--output-root",
            str(tmp_path / "out"),
            "--collector-env-num",
            "64",
            "--num-simulations",
            "25",
            "--batch-size",
            "128",
            "--model-support-cap",
            "2048",
            "--td-steps",
            "50",
            "--background-eval-num-simulations",
            "16",
            "--background-eval-batch-size",
            "128",
        ]
    )

    manifest = module.build_manifest(args)

    assert manifest["fixed_knobs"]["collector_env_num"] == 64
    assert manifest["fixed_knobs"]["n_episode"] == 64
    assert manifest["fixed_knobs"]["num_simulations"] == 25
    assert manifest["fixed_knobs"]["batch_size"] == 128
    assert manifest["fixed_knobs"]["model_support_cap"] == 2048
    assert manifest["fixed_knobs"]["td_steps"] == 50
    for row in manifest["rows"]:
        train_kwargs = row["train_kwargs"]
        poller_kwargs = row["poller_kwargs"]
        assert train_kwargs["collector_env_num"] == 64
        assert train_kwargs["n_episode"] == 64
        assert train_kwargs["num_simulations"] == 25
        assert train_kwargs["batch_size"] == 128
        assert train_kwargs["model_support_cap"] == 2048
        assert train_kwargs["td_steps"] == 50
        assert poller_kwargs["background_eval_num_simulations"] == 16
        assert poller_kwargs["background_eval_batch_size"] == 128


def test_tonight18_manifest_can_emit_own_checkpoint_refresh_control(
    tmp_path: Path,
):
    module = _load_module()
    refs_file = _checkpoint_refs_file(tmp_path / "refs.txt")
    args = module.parse_args(
        [
            "--checkpoint-refs-file",
            str(refs_file),
            "--output-root",
            str(tmp_path / "out"),
            "--opponent-source",
            "mixture",
            "--assignment-refresh-interval-train-iter",
            "2000",
            "--own-checkpoint-opponent-refresh",
        ]
    )

    manifest = module.build_manifest(args)

    assert manifest["fixed_knobs"]["own_checkpoint_opponent_refresh_enabled"] is True
    assert manifest["assignment_bank"] is None
    for row in manifest["rows"]:
        raw_train_kwargs = row["train_kwargs"]
        train_kwargs = _expanded_train_kwargs(module, row)
        assert row["opponent_source"] == "mixture"
        assert row["opponent_assignment_ref"] is None
        assert row["opponent_assignment_refresh_ref"] is None
        assert "opponent_assignment_ref" not in raw_train_kwargs
        assert train_kwargs["opponent_mixture_spec"] is not None
        assert train_kwargs["opponent_assignment_ref"] is None
        assert "opponent_assignment_refresh_ref" not in train_kwargs
        assert train_kwargs["opponent_assignment_refresh_interval_train_iter"] == 2000
        assert train_kwargs["own_checkpoint_opponent_refresh_enabled"] is True


def test_grouped_submit_dry_run_validates_assignment_bank_and_initial_checkpoint(
    tmp_path: Path,
):
    builder = _load_module()
    submitter = _load_submitter_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    matrix_name = "curvy-night18-connected-test"
    args = builder.parse_args(
        [
            "--ratings-snapshot",
            str(snapshot),
            "--output-root",
            str(tmp_path / "out"),
            "--matrix-name",
            matrix_name,
            "--run-prefix",
            "curvy-n18conn-test",
            "--attempt-prefix",
            "try-n18conn-test",
            "--opponent-source",
            "assignment",
            "--assignment-bank-run-id",
            "curvy-n18conn-test-assignments",
            "--assignment-bank-attempt-id",
            "try-n18conn-test-assignments",
        ]
    )
    manifest = builder.build_manifest(args)
    outputs = builder._write_outputs(
        manifest,
        output_root=tmp_path / "out",
        matrix_name=matrix_name,
    )
    output_path = tmp_path / "submission.json"

    submitter.main([outputs["manifest_json"], "--output", str(output_path)])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["row_count"] == 18
    assert payload["assignment_write_count"] == 3
    assert {record["status"] for record in payload["assignment_records"]} == {"dry_run"}


def test_grouped_submit_dry_run_accepts_scratch_bootstrap_initialization(
    tmp_path: Path,
):
    builder = _load_module()
    submitter = _load_submitter_module()
    matrix_name = "curvy-night18-scratch-test"
    args = builder.parse_args(
        [
            "--scratch-bootstrap",
            "--output-root",
            str(tmp_path / "out"),
            "--matrix-name",
            matrix_name,
        ]
    )
    manifest = builder.build_manifest(args)
    outputs = builder._write_outputs(
        manifest,
        output_root=tmp_path / "out",
        matrix_name=matrix_name,
    )
    output_path = tmp_path / "submission.json"

    submitter.main([outputs["manifest_json"], "--output", str(output_path)])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["row_count"] == 18
    assert payload["records"][0]["status"] == "dry_run"


def test_grouped_submit_writes_assignments_before_spawning_selected_rows(
    monkeypatch,
    tmp_path: Path,
):
    builder = _load_module()
    submitter = _load_submitter_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    args = builder.parse_args(
        [
            "--ratings-snapshot",
            str(snapshot),
            "--output-root",
            str(tmp_path / "out"),
            "--matrix-name",
            "curvy-submit-order-test",
            "--run-prefix",
            "curvy-submit-order",
            "--attempt-prefix",
            "try-submit-order",
            "--opponent-source",
            "assignment",
            "--assignment-bank-run-id",
            "curvy-submit-order-assignments",
            "--assignment-bank-attempt-id",
            "try-submit-order-assignments",
            "--assignment-target-volume",
            "control",
            "--assignment-refresh-interval-train-iter",
            "25",
            "--assignment-refresh-pointer-run-id",
            "curvy-submit-order-control",
            "--assignment-refresh-pointer-attempt-id",
            "try-submit-order-control",
        ]
    )
    manifest = builder.build_manifest(args)
    outputs = builder._write_outputs(
        manifest,
        output_root=tmp_path / "out",
        matrix_name="curvy-submit-order-test",
    )
    assignment_ref_by_id = {
        artifact["assignment_id"]: artifact["assignment_ref"]
        for artifact in manifest["assignment_bank"]["assignments"].values()
    }
    assignment_sha_by_id = {
        artifact["assignment_id"]: artifact["assignment_sha256"]
        for artifact in manifest["assignment_bank"]["assignments"].values()
    }
    events = []

    class FakeCall:
        def __init__(self, object_id: str) -> None:
            self.object_id = object_id

    class FakeFunction:
        def __init__(self, name: str) -> None:
            self.name = name

        def remote(self, payload):
            assignment_id = payload["assignment"]["assignment_id"]
            events.append(("remote", self.name, assignment_id))
            return {
                "assignment_ref": assignment_ref_by_id[assignment_id],
                "assignment_sha256": assignment_sha_by_id[assignment_id],
            }

        def spawn(self, **kwargs):
            events.append(("spawn", self.name, kwargs["run_id"]))
            return FakeCall(f"call-{self.name}")

    class FakeFunctionFactory:
        @staticmethod
        def from_name(_app_name: str, name: str, environment_name=None):
            events.append(("function_from_name", name, environment_name))
            return FakeFunction(name)

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def put_file(self, file_obj, remote_ref):
            events.append(("pointer", "volume", remote_ref, file_obj.read()))

    class FakeVolume:
        @staticmethod
        def from_name(
            name: str,
            create_if_missing: bool = False,
            environment_name=None,
            version=None,
        ):
            events.append(("volume", name, create_if_missing, environment_name, version))
            return FakeVolume()

        def batch_upload(self, force: bool = False):
            events.append(("batch_upload", force))
            return FakeBatch()

    monkeypatch.setitem(
        sys.modules,
        "modal",
        types.SimpleNamespace(Function=FakeFunctionFactory, Volume=FakeVolume),
    )
    output_path = tmp_path / "submission.json"

    submitter.main(
        [
            outputs["manifest_json"],
            "--allow-launch",
            "--limit",
            "1",
            "--allow-partial-launch",
            "--modal-env",
            "shankha-dev",
            "--output",
            str(output_path),
        ]
    )

    assert events[0] == (
        "function_from_name",
        submitter.ASSIGNMENT_WRITER_FUNCTION,
        "shankha-dev",
    )
    assert events[1][0] == "remote"
    assert events[1][1] == submitter.ASSIGNMENT_WRITER_FUNCTION
    assert events[2] == (
        "volume",
        "curvyzero-curvytron-control-v2",
        True,
        "shankha-dev",
        2,
    )
    assert events[3] == ("batch_upload", True)
    assert events[4][0:2] == ("pointer", "volume")
    assert events[4][2].startswith("/")
    assert events[5] == (
        "function_from_name",
        "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
        "shankha-dev",
    )
    assert events[6] == (
        "function_from_name",
        "lightzero_curvytron_visual_survival_gpu_cpu40",
        "shankha-dev",
    )
    assert events[7][0:2] == (
        "spawn",
        "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
    )
    assert events[8][0:2] == ("spawn", "lightzero_curvytron_visual_survival_gpu_cpu40")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["modal_env"] == "shankha-dev"
    assert payload["assignment_write_count"] == 1
    assert payload["refresh_pointer_write_count"] == 1
    assert payload["records"][0]["status"] == "spawned"


def test_grouped_submit_can_publish_assignments_without_spawning_rows(
    monkeypatch,
    tmp_path: Path,
):
    builder = _load_module()
    submitter = _load_submitter_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    args = builder.parse_args(
        [
            "--ratings-snapshot",
            str(snapshot),
            "--output-root",
            str(tmp_path / "out"),
            "--matrix-name",
            "curvy-submit-assign-only-test",
            "--run-prefix",
            "curvy-submit-assign-only",
            "--attempt-prefix",
            "try-submit-assign-only",
            "--opponent-source",
            "assignment",
            "--assignment-target-volume",
            "control",
            "--assignment-refresh-interval-train-iter",
            "25",
            "--assignment-refresh-pointer-run-id",
            "curvy-submit-assign-only-control",
            "--assignment-refresh-pointer-attempt-id",
            "try-submit-assign-only-control",
        ]
    )
    manifest = builder.build_manifest(args)
    outputs = builder._write_outputs(
        manifest,
        output_root=tmp_path / "out",
        matrix_name="curvy-submit-assign-only-test",
    )
    assignment_ref_by_id = {
        artifact["assignment_id"]: artifact["assignment_ref"]
        for artifact in manifest["assignment_bank"]["assignments"].values()
    }
    assignment_sha_by_id = {
        artifact["assignment_id"]: artifact["assignment_sha256"]
        for artifact in manifest["assignment_bank"]["assignments"].values()
    }
    events = []

    class FakeFunction:
        def __init__(self, name: str) -> None:
            self.name = name

        def remote(self, payload):
            assignment_id = payload["assignment"]["assignment_id"]
            events.append(("remote", self.name, assignment_id))
            return {
                "assignment_ref": assignment_ref_by_id[assignment_id],
                "assignment_sha256": assignment_sha_by_id[assignment_id],
            }

        def spawn(self, **kwargs):  # pragma: no cover - this test asserts it is unused.
            events.append(("spawn", self.name, kwargs.get("run_id")))

    class FakeFunctionFactory:
        @staticmethod
        def from_name(_app_name: str, name: str, environment_name=None):
            events.append(("function_from_name", name, environment_name))
            return FakeFunction(name)

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def put_file(self, file_obj, remote_ref):
            events.append(("pointer", remote_ref, file_obj.read()))

    class FakeVolume:
        @staticmethod
        def from_name(
            name: str,
            create_if_missing: bool = False,
            environment_name=None,
            version=None,
        ):
            events.append(("volume", name, create_if_missing, environment_name, version))
            return FakeVolume()

        def batch_upload(self, force: bool = False):
            events.append(("batch_upload", force))
            return FakeBatch()

    monkeypatch.setitem(
        sys.modules,
        "modal",
        types.SimpleNamespace(Function=FakeFunctionFactory, Volume=FakeVolume),
    )
    output_path = tmp_path / "submission.json"

    submitter.main(
        [
            outputs["manifest_json"],
            "--allow-launch",
            "--limit",
            "1",
            "--allow-partial-launch",
            "--modal-env",
            "shankha-dev",
            "--publish-assignments-only",
            "--output",
            str(output_path),
        ]
    )

    assert not [event for event in events if event[0] == "spawn"]
    assert [event for event in events if event[0] == "remote"]
    assert [event for event in events if event[0] == "pointer"]
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["publish_assignments_only"] is True
    assert payload["assignment_write_count"] == 1
    assert payload["refresh_pointer_write_count"] == 1
    assert payload["selected_row_count"] == 1
    assert payload["row_count"] == 0
    assert payload["records"] == []


def test_grouped_submit_rejects_app_name_that_disagrees_with_manifest(tmp_path: Path):
    builder = _load_module()
    submitter = _load_submitter_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    manifest = builder.build_manifest(
        builder.parse_args(
            [
                "--ratings-snapshot",
                str(snapshot),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
    )

    with pytest.raises(ValueError, match="targets app"):
        submitter._validate_launch_app_name(
            "curvyzero-lightzero-curvytron-visual-survival-train-v2",
            [
                {
                    **manifest["rows"][0],
                    "deployed_app_submission": {
                        **manifest["rows"][0]["deployed_app_submission"],
                        "app_name": "curvyzero-old-trainer",
                    },
                }
            ],
        )


def test_grouped_submit_rejects_train_only_initial_checkpoint_in_poller_kwargs(
    tmp_path: Path,
):
    builder = _load_module()
    submitter = _load_submitter_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    manifest = builder.build_manifest(
        builder.parse_args(
            [
                "--ratings-snapshot",
                str(snapshot),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
    )
    row = json.loads(json.dumps(manifest["rows"][0]))
    row["poller_kwargs"]["initial_policy_checkpoint_ref"] = row["initial_policy_checkpoint_ref"]

    try:
        submitter._launch_row(
            row,
            app_name="curvyzero-trainer",
            modal_env=None,
            dry_run=True,
        )
    except ValueError as exc:
        assert "unsupported keys" in str(exc)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("expected unsupported poller key to fail")


def test_grouped_submit_accepts_compact_train_kwargs_with_current_defaults():
    submitter = _load_submitter_module()
    row = {
        "row_id": "compact-default-row",
        "label": "compact defaults",
        "run_id": "compact-default-run",
        "attempt_id": "compact-default-attempt",
        "initial_policy_checkpoint_source": {
            "source": "scratch_random_initialization",
            "checkpoint_ref": None,
        },
        "deployed_app_submission": {
            "app_name": "curvyzero-lightzero-curvytron-visual-survival-train-v2",
            "train_function": "lightzero_curvytron_visual_survival_gpu_cpu40",
            "poller_function": "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
        },
        "train_kwargs": {
            "mode": "train",
            "seed": 11,
            "run_id": "compact-default-run",
            "attempt_id": "compact-default-attempt",
        },
        "poller_kwargs": {},
    }

    assert submitter._launch_row(
        row,
        app_name="curvyzero-lightzero-curvytron-visual-survival-train-v2",
        modal_env=None,
        dry_run=True,
    )["status"] == "dry_run"


def test_grouped_submit_rejects_mutable_initial_checkpoint_ref(tmp_path: Path):
    builder = _load_module()
    submitter = _load_submitter_module()
    snapshot = _ratings_snapshot(tmp_path / "ratings.json")
    manifest = builder.build_manifest(
        builder.parse_args(
            [
                "--ratings-snapshot",
                str(snapshot),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
    )
    row = json.loads(json.dumps(manifest["rows"][0]))
    row["train_kwargs"]["initial_policy_checkpoint_ref"] = (
        "training/lightzero-curvytron-visual-survival/run/attempts/a/train/"
        "lightzero_exp/ckpt/latest.pth.tar"
    )

    try:
        submitter._launch_row(
            row,
            app_name="curvyzero-trainer",
            modal_env=None,
            dry_run=True,
        )
    except ValueError as exc:
        assert "initial_policy_checkpoint_ref is mutable" in str(exc)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("expected mutable initial checkpoint ref to fail")
