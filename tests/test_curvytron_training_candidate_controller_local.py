from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from curvyzero.infra.modal import curvyzero_checkpoint_tournament as modal_arena
from curvyzero.infra.modal import (
    lightzero_curvyzero_stacked_debug_visual_survival_train as train_mod,
)
from curvyzero.training.opponent_leaderboard import (
    canonical_json_sha256,
    validate_leaderboard_snapshot,
)
from curvyzero.training.opponent_registry import (
    OPPONENT_ASSIGNMENT_SCHEMA_ID,
    canonical_assignment_json_sha256,
)
from curvyzero.tournament import curvytron_checkpoint_tournament as arena


class _FakeDict:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(str(key), default)

    def put(self, key: str, value: Any, **_kwargs: Any) -> bool:
        self.values[str(key)] = value
        return True


class _CommitPlan:
    def __init__(self, failures: list[str] | None = None) -> None:
        self.failures = list(failures or [])
        self.calls: list[str] = []

    def __call__(self, volume: object = None) -> str | None:
        name = getattr(volume, "name", "unknown")
        self.calls.append(name)
        if self.failures:
            return self.failures.pop(0)
        return None


class _FakeVolume:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeQueue:
    def __init__(self) -> None:
        self.events_by_partition: dict[str, list[dict[str, Any]]] = {}

    def put(
        self,
        value: Mapping[str, Any],
        *,
        partition: str,
        **_kwargs: Any,
    ) -> bool:
        self.events_by_partition.setdefault(str(partition), []).append(dict(value))
        return True

    def len(self, *, partition: str) -> int:
        return len(self.events_by_partition.get(str(partition), []))

    def get_many(self, max_events: int, *, partition: str, **_kwargs: Any) -> list[dict[str, Any]]:
        events = self.events_by_partition.get(str(partition), [])
        drained = events[: int(max_events)]
        del events[: int(max_events)]
        return drained


class _FakeGameMap:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def map(
        self,
        game_specs: list[dict[str, Any]],
        *,
        order_outputs: bool = False,
    ) -> list[dict[str, Any]]:
        del order_outputs
        specs = [dict(spec) for spec in game_specs]
        self.calls.append(specs)
        return [_fake_game_result(spec) for spec in specs]


class _FakeRatingLoop:
    def __init__(self) -> None:
        self.spawned: list[dict[str, Any]] = []

    def spawn(self, spec: Mapping[str, Any]) -> "_FakeRatingCall":
        self.spawned.append(dict(spec))
        return _FakeRatingCall()


class _FakeRatingCall:
    object_id = "fc-local-rating-loop"

    def get(self) -> dict[str, Any]:
        return {"ok": True, "object_id": self.object_id}


def _checkpoint_ref(run_id: str, iteration: int) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/"
        f"{run_id}/attempts/try-{run_id}/train/lightzero_exp/ckpt/"
        f"iteration_{iteration}.pth.tar"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fake_game_result(game_spec: Mapping[str, Any]) -> dict[str, Any]:
    players = list(game_spec["players"])
    seat0_ref = str(players[0].get("checkpoint_ref") or "")
    outcome = "seat_0_win" if "loop-a" in seat0_ref else "seat_1_win"
    winner = 0 if outcome == "seat_0_win" else 1
    loser = 1 - winner
    game_id = str(game_spec["game_id"])
    battle_id = str(game_spec["battle_id"])
    tournament_id = str(game_spec["tournament_id"])
    return {
        "ok": True,
        "tournament_id": tournament_id,
        "battle_id": battle_id,
        "pair_index": int(game_spec["pair_index"]),
        "game_id": game_id,
        "game_index": int(game_spec["game_index"]),
        "players": players,
        "physical_steps": 40 + int(game_spec["game_index"]),
        "score": {
            "outcome": outcome,
            "winner_seat": winner,
            "loser_seat": loser,
            "draw": False,
            "physical_steps": 40 + int(game_spec["game_index"]),
        },
        "summary_ref": arena.game_summary_ref(
            tournament_id,
            battle_id,
            game_id,
        ).as_posix(),
    }


@pytest.fixture
def controller_mounts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    tournament_mount = tmp_path / "tournament"
    runs_mount = tmp_path / "runs"
    control_mount = tmp_path / "control"
    for mount in (tournament_mount, runs_mount, control_mount):
        mount.mkdir()

    fake_dict = _FakeDict()
    commit_plan = _CommitPlan()

    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tournament_mount)
    monkeypatch.setattr(modal_arena, "RUNS_MOUNT", runs_mount)
    monkeypatch.setattr(modal_arena, "CONTROL_MOUNT", control_mount)
    monkeypatch.setattr(modal_arena, "tournament_volume", _FakeVolume("tournament"))
    monkeypatch.setattr(modal_arena, "checkpoint_volume", _FakeVolume("checkpoint"))
    monkeypatch.setattr(modal_arena, "control_volume", _FakeVolume("control"))
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "_commit_volume", commit_plan)
    monkeypatch.setattr(modal_arena, "opponent_leaderboard_state", fake_dict)

    monkeypatch.setattr(train_mod, "RUNS_MOUNT", runs_mount)
    monkeypatch.setattr(train_mod, "CONTROL_MOUNT", control_mount)
    monkeypatch.setattr(train_mod, "_safe_reload_control_volume", lambda *, reason: None)
    monkeypatch.setattr(train_mod, "_safe_reload_runs_volume", lambda *, reason: None)
    monkeypatch.setattr(
        train_mod,
        "_safe_reload_volume_for_ref",
        lambda _path_text, *, reason, default_mount_name="runs": None,
    )

    return {
        "tournament": tournament_mount,
        "runs": runs_mount,
        "control": control_mount,
        "dict": fake_dict,
        "commits": commit_plan,
    }


def _toy_rating_snapshot(
    tournament_id: str = "toy-arena",
    rating_run_id: str = "toy-elo",
    *,
    rank2_status: str = "active",
    round_index: int = 4,
) -> dict[str, Any]:
    rows = []
    for rank in range(1, 5):
        games = 500
        distinct = 25
        failure_count = 0
        status = "active"
        if rank == 2 and rank2_status == "provisional":
            games = 10
            distinct = 1
            status = ""
        if rank == 2 and rank2_status == "retired":
            status = "retired"
        rows.append(
            {
                "checkpoint_id": f"ckpt-rank{rank}",
                "checkpoint_ref": _checkpoint_ref(f"run-rank{rank}", rank * 10000),
                "run_id": f"run-rank{rank}",
                "attempt_id": f"try-run-rank{rank}",
                "label": f"rank {rank}",
                "rank": rank,
                "rating": 1800.0 - rank * 10.0,
                "games": games,
                "wins": 300 - rank,
                "losses": 150 + rank,
                "draws": 50,
                "battles": distinct,
                "rated_battles": distinct,
                "distinct_opponents": distinct,
                "failure_count": failure_count,
                "status": status,
            }
        )
    return {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "formula_version": arena.RATING_FORMULA_VERSION,
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "ratings_ref": arena.rating_latest_ref(tournament_id, rating_run_id).as_posix(),
        "context_hash": "ctx-toy",
        "roster_hash": "roster-toy",
        "round_id": f"round-{round_index:06d}",
        "round_index": round_index,
        "stable": False,
        "max_abs_delta": 12.5,
        "rating_spec": {"decision_source_frames": 1},
        "ratings": rows,
    }


def _recipe_assignment(recipe_id: str = "recipe-a") -> dict[str, Any]:
    return {
        "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
        "assignment_id": f"seed-{recipe_id}",
        "source_epoch": 0,
        "source_ref": "seed",
        "seed": 7,
        "entries": [
            {
                "name": "blank",
                "age_label": "blank_canvas",
                "weight": 12,
                "opponent_policy_kind": "fixed_straight",
                "opponent_runtime_mode": "blank_canvas_noop",
                "opponent_immortal": True,
            },
            {
                "name": "wall_avoidant_immortal",
                "age_label": "hardcoded_wall_avoidant_immortal",
                "weight": 4,
                "opponent_policy_kind": "proactive_wall_avoidant",
                "opponent_runtime_mode": "normal",
                "opponent_immortal": True,
                "opponent_wall_avoidant_safe_margin": 20.0,
            },
            {
                "name": "rank2",
                "age_label": "rank2",
                "weight": 46,
                "opponent_policy_kind": "frozen_lightzero_checkpoint",
                "opponent_checkpoint_ref": _checkpoint_ref("old-rank2", 2000),
                "opponent_immortal": False,
                "tags": {"source_slot": "rank2"},
            },
            {
                "name": "rank1_immortal",
                "age_label": "rank1_immortal",
                "weight": 2,
                "opponent_policy_kind": "frozen_lightzero_checkpoint",
                "opponent_checkpoint_ref": _checkpoint_ref("old-rank1", 1000),
                "opponent_immortal": True,
                "tags": {"source_slot": "rank1_immortal"},
            },
        ],
    }


def _scratch_placeholder_rank_assignment(recipe_id: str = "recipe-scratch") -> dict[str, Any]:
    return {
        "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
        "assignment_id": f"seed-{recipe_id}",
        "source_epoch": 0,
        "source_ref": "seed",
        "seed": 7,
        "entries": [
            {
                "name": "scratch_rank2_slot",
                "age_label": "scratch_rank2_slot",
                "weight": 32,
                "opponent_policy_kind": "fixed_straight",
                "opponent_runtime_mode": "blank_canvas_noop",
                "opponent_immortal": True,
                "tags": {"source_slot": "rank2"},
            },
            {
                "name": "scratch_rank3_numeric_slot",
                "age_label": "scratch_rank3_numeric_slot",
                "weight": 16,
                "opponent_policy_kind": "proactive_wall_avoidant",
                "opponent_runtime_mode": "normal",
                "opponent_immortal": True,
                "opponent_wall_avoidant_safe_margin": 20.0,
                "tags": {"rank": 3},
            },
            {
                "name": "wall_avoidant_immortal",
                "age_label": "hardcoded_wall_avoidant_immortal",
                "weight": 16,
                "opponent_policy_kind": "proactive_wall_avoidant",
                "opponent_runtime_mode": "normal",
                "opponent_immortal": True,
                "opponent_wall_avoidant_safe_margin": 20.0,
            },
        ],
    }


def _write_rating_latest(mount: Path, rating_snapshot: dict[str, Any]) -> None:
    _write_json(
        mount
        / arena.rating_latest_ref(
            rating_snapshot["tournament_id"],
            rating_snapshot["rating_run_id"],
        ),
        rating_snapshot,
    )
    for row in rating_snapshot.get("ratings", []):
        checkpoint_ref = row.get("checkpoint_ref")
        if not checkpoint_ref:
            continue
        checkpoint_path = modal_arena.RUNS_MOUNT / str(checkpoint_ref)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint_path.exists():
            checkpoint_path.write_bytes(f"fake checkpoint {checkpoint_ref}".encode("utf-8"))
        sidecar_path = checkpoint_path.with_name(f"{checkpoint_path.name}.metadata.json")
        if not sidecar_path.exists():
            sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_id": arena.CHECKPOINT_POLICY_METADATA_SIDECAR_SCHEMA_ID,
                        "checkpoint_ref": checkpoint_ref,
                        "policy_trail_render_mode": arena.DEFAULT_POLICY_TRAIL_RENDER_MODE,
                        "policy_bonus_render_mode": arena.DEFAULT_POLICY_BONUS_RENDER_MODE,
                        "policy_observation_backend": (
                            arena.DEFAULT_TOURNAMENT_POLICY_OBSERVATION_BACKEND
                        ),
                        "policy_observation_contract_id": (
                            arena.DEFAULT_POLICY_OBSERVATION_CONTRACT_ID
                        ),
                        "policy_observation_perspective_schema_id": (
                            arena.POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
                        ),
                        "observation_contract": arena.DEFAULT_POLICY_OBSERVATION_SURFACE,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )


def _write_training_checkpoint(
    runs_mount: Path,
    *,
    run_id: str,
    iteration: int,
) -> str:
    attempt_id = f"try-{run_id}"
    attempt_train_root = runs_mount / train_mod.runs.attempt_train_ref(
        train_mod.TASK_ID,
        run_id,
        attempt_id,
    )
    exp_name = attempt_train_root / "lightzero_exp"
    checkpoint_path = exp_name / "ckpt" / f"iteration_{int(iteration)}.pth.tar"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(f"fake checkpoint {run_id} {iteration}".encode("utf-8"))

    class FakeLearner:
        train_iter = iteration

    train_mod._write_checkpoint_progress_latest(
        run_id=run_id,
        attempt_id=attempt_id,
        attempt_train_root=attempt_train_root,
        exp_name=exp_name,
        learner=FakeLearner(),
        started_monotonic=0.0,
        checkpoint_metadata={
            "policy_trail_render_mode": arena.DEFAULT_POLICY_TRAIL_RENDER_MODE,
            "policy_bonus_render_mode": arena.DEFAULT_POLICY_BONUS_RENDER_MODE,
            "policy_observation_backend": arena.DEFAULT_TOURNAMENT_POLICY_OBSERVATION_BACKEND,
            "policy_observation_contract_id": arena.DEFAULT_POLICY_OBSERVATION_CONTRACT_ID,
            "policy_observation_perspective_schema_id": (
                arena.POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
            ),
            "observation_contract": arena.DEFAULT_POLICY_OBSERVATION_SURFACE,
            "decision_source_frames": 1,
            "learner_seat_mode": "random",
        },
    )
    return train_mod.runs.file_ref(checkpoint_path, mount=runs_mount)


def _write_control_assignment_and_pointer(
    control_mount: Path,
    *,
    recipe_id: str = "recipe-a",
    pointer_ref: str = "control:control/recipes/recipe-a/pointer.json",
    assignment: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    assignment_payload = assignment or _recipe_assignment(recipe_id)
    assignment_ref = f"control:control/recipes/{recipe_id}/assignment.json"
    assignment_sha = canonical_assignment_json_sha256(assignment_payload)
    pointer = {
        "schema_id": "curvyzero_opponent_assignment_refresh_pointer/v0",
        "assignment_ref": assignment_ref,
        "assignment_sha256": assignment_sha,
        "recipe_id": recipe_id,
        "generation": 0,
    }
    _write_json(control_mount / assignment_ref.removeprefix("control:"), assignment_payload)
    _write_json(control_mount / pointer_ref.removeprefix("control:"), pointer)
    return pointer_ref, assignment_ref, assignment_payload


def _run_controller(controller_mounts: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    rating_snapshot = _toy_rating_snapshot()
    _write_rating_latest(controller_mounts["tournament"], rating_snapshot)
    pointer_ref, _assignment_ref, _assignment = _write_control_assignment_and_pointer(
        controller_mounts["control"]
    )
    spec = {
        "tournament_id": rating_snapshot["tournament_id"],
        "rating_run_id": rating_snapshot["rating_run_id"],
        "leaderboard_id": "toy-training",
        "snapshot_id": "toy-snapshot",
        "expected_round_id": rating_snapshot["round_id"],
        "expected_round_index": rating_snapshot["round_index"],
        "expected_rating_context_hash": rating_snapshot["context_hash"],
        "expected_roster_hash": rating_snapshot["roster_hash"],
        "expected_rating_snapshot_sha256": canonical_json_sha256(rating_snapshot),
        "refresh_pointers": [pointer_ref],
        "assignment_bank_run_id": "toy-bank",
        "assignment_bank_attempt_id": "try-toy-bank",
    }
    spec.update(overrides)
    return modal_arena.curvytron_training_candidate_refresh.local(spec)


def _run_controller_with_assignment(
    controller_mounts: dict[str, Any],
    *,
    rating_snapshot: dict[str, Any],
    assignment: dict[str, Any],
    **overrides: Any,
) -> dict[str, Any]:
    _write_rating_latest(controller_mounts["tournament"], rating_snapshot)
    pointer_ref, _assignment_ref, _assignment = _write_control_assignment_and_pointer(
        controller_mounts["control"],
        assignment=assignment,
    )
    spec = {
        "tournament_id": rating_snapshot["tournament_id"],
        "rating_run_id": rating_snapshot["rating_run_id"],
        "leaderboard_id": "toy-training",
        "snapshot_id": "toy-snapshot",
        "expected_round_id": rating_snapshot["round_id"],
        "expected_round_index": rating_snapshot["round_index"],
        "expected_rating_context_hash": rating_snapshot["context_hash"],
        "expected_roster_hash": rating_snapshot["roster_hash"],
        "expected_rating_snapshot_sha256": canonical_json_sha256(rating_snapshot),
        "refresh_pointers": [pointer_ref],
        "assignment_bank_run_id": "toy-bank",
        "assignment_bank_attempt_id": "try-toy-bank",
    }
    spec.update(overrides)
    return modal_arena.curvytron_training_candidate_refresh.local(spec)


def test_training_candidate_refresh_writes_leaderboard_and_rewrites_control_pointer(
    controller_mounts: dict[str, Any],
) -> None:
    result = _run_controller(controller_mounts)

    assert result["rewritten_pointer_count"] == 1
    assert result["pointer_published"] is True
    assert result["rating_stable"] is False
    assert result["active_count"] == 4
    assert controller_mounts["dict"].values["current:toy-training"]["snapshot_id"] == "toy-snapshot"

    snapshot_ref = modal_arena._leaderboard_snapshot_ref("toy-training", "toy-snapshot")
    latest_ref = modal_arena._leaderboard_latest_ref("toy-training")
    snapshot = validate_leaderboard_snapshot(
        json.loads((controller_mounts["tournament"] / snapshot_ref).read_text(encoding="utf-8"))
    )
    latest = validate_leaderboard_snapshot(
        json.loads((controller_mounts["tournament"] / latest_ref).read_text(encoding="utf-8"))
    )
    assert latest["snapshot_sha256"] == snapshot["snapshot_sha256"]
    assert snapshot["context"]["snapshot_kind"] == "training_candidate"

    pointer_ref = result["rewritten_pointers"][0]["pointer_ref"]
    pointer = json.loads(
        (controller_mounts["control"] / pointer_ref.removeprefix("control:")).read_text(
            encoding="utf-8"
        )
    )
    assignment_ref = pointer["assignment_ref"]
    assignment = json.loads(
        (controller_mounts["control"] / assignment_ref.removeprefix("control:")).read_text(
            encoding="utf-8"
        )
    )
    assert pointer["assignment_sha256"] == canonical_assignment_json_sha256(assignment)
    entries = {entry["name"]: entry for entry in assignment["entries"]}
    assert entries["blank"]["opponent_immortal"] is True
    assert entries["wall_avoidant_immortal"]["opponent_immortal"] is True
    assert entries["rank2"]["weight"] == 46.0
    assert entries["rank2"]["opponent_checkpoint_ref"].startswith("control:")
    assert entries["rank2"]["tags"]["checkpoint_id"] == "ckpt-rank2"
    assert entries["rank1_immortal"]["opponent_immortal"] is True
    assert entries["rank1_immortal"]["opponent_checkpoint_ref"].startswith("control:")
    assert entries["rank1_immortal"]["tags"]["checkpoint_id"] == "ckpt-rank1"
    checkpoint_copies = result["rewritten_pointers"][0]["checkpoint_copies"]
    assert {copy["source_ref"] for copy in checkpoint_copies} == {
        _checkpoint_ref("run-rank1", 10000),
        _checkpoint_ref("run-rank2", 20000),
    }
    for checkpoint_copy in checkpoint_copies:
        assert checkpoint_copy["target_ref"].startswith("control:")
        target = controller_mounts["control"] / checkpoint_copy["target_ref"].removeprefix(
            "control:"
        )
        assert target.is_file()
        metadata_path = target.with_name(f"{target.name}.metadata.json")
        assert metadata_path.is_file()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["checkpoint_ref"] == checkpoint_copy["target_ref"]
        assert metadata["source_checkpoint_ref"] == checkpoint_copy["source_ref"]
        assert (
            metadata["policy_observation_contract_id"]
            == arena.DEFAULT_POLICY_OBSERVATION_CONTRACT_ID
        )
        assert (
            metadata["policy_observation_perspective_schema_id"]
            == arena.POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
        )


def test_training_candidate_refresh_replaces_tagged_hardcoded_rank_slot(
    controller_mounts: dict[str, Any],
) -> None:
    result = _run_controller_with_assignment(
        controller_mounts,
        rating_snapshot=_toy_rating_snapshot(),
        assignment=_scratch_placeholder_rank_assignment(),
    )

    assignment_ref = result["rewritten_pointers"][0]["assignment_ref"]
    assignment = json.loads(
        (controller_mounts["control"] / assignment_ref.removeprefix("control:")).read_text(
            encoding="utf-8"
        )
    )
    entries = {entry["name"]: entry for entry in assignment["entries"]}
    slot = entries["scratch_rank2_slot"]
    assert slot["name"] == "scratch_rank2_slot"
    assert slot["weight"] == 32.0
    assert slot["opponent_immortal"] is True
    assert slot["opponent_policy_kind"] == "frozen_lightzero_checkpoint"
    assert slot["opponent_runtime_mode"] == "normal"
    assert slot["opponent_checkpoint_ref"].startswith("control:")
    assert "opponent_wall_avoidant_safe_margin" not in slot
    assert slot["tags"]["rank"] == 2
    assert slot["tags"]["source_slot"] == "rank2"
    numeric_slot = entries["scratch_rank3_numeric_slot"]
    assert numeric_slot["weight"] == 16.0
    assert numeric_slot["opponent_policy_kind"] == "frozen_lightzero_checkpoint"
    assert numeric_slot["opponent_runtime_mode"] == "normal"
    assert numeric_slot["opponent_checkpoint_ref"].startswith("control:")
    assert "opponent_wall_avoidant_safe_margin" not in numeric_slot
    assert numeric_slot["tags"]["rank"] == 3
    assert numeric_slot["tags"]["source_slot"] == "rank3"
    assert entries["wall_avoidant_immortal"]["opponent_policy_kind"] == "proactive_wall_avoidant"


def test_training_candidate_refresh_rejects_corrupted_input_pointer_sha(
    controller_mounts: dict[str, Any],
) -> None:
    rating_snapshot = _toy_rating_snapshot()
    _write_rating_latest(controller_mounts["tournament"], rating_snapshot)
    pointer_ref, _assignment_ref, _assignment = _write_control_assignment_and_pointer(
        controller_mounts["control"]
    )
    pointer_path = controller_mounts["control"] / pointer_ref.removeprefix("control:")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["assignment_sha256"] = "0" * 64
    _write_json(pointer_path, pointer)

    with pytest.raises(ValueError, match="assignment_sha256 mismatch"):
        modal_arena.curvytron_training_candidate_refresh.local(
            {
                "tournament_id": rating_snapshot["tournament_id"],
                "rating_run_id": rating_snapshot["rating_run_id"],
                "leaderboard_id": "toy-training",
                "snapshot_id": "toy-snapshot",
                "refresh_pointers": [pointer_ref],
            }
        )


def test_training_candidate_refresh_rejects_checkpoint_without_policy_metadata(
    controller_mounts: dict[str, Any],
) -> None:
    rating_snapshot = _toy_rating_snapshot()
    _write_rating_latest(controller_mounts["tournament"], rating_snapshot)
    bad_checkpoint_ref = _checkpoint_ref("run-rank1", 10000)
    (
        controller_mounts["runs"] / arena.checkpoint_policy_metadata_sidecar_ref(bad_checkpoint_ref)
    ).unlink()
    pointer_ref, _assignment_ref, _assignment = _write_control_assignment_and_pointer(
        controller_mounts["control"]
    )

    with pytest.raises(ValueError, match="missing required policy observation metadata"):
        modal_arena.curvytron_training_candidate_refresh.local(
            {
                "tournament_id": rating_snapshot["tournament_id"],
                "rating_run_id": rating_snapshot["rating_run_id"],
                "leaderboard_id": "toy-training",
                "snapshot_id": "toy-snapshot",
                "refresh_pointers": [pointer_ref],
            }
        )


def test_training_candidate_refresh_aborts_on_commit_failure_before_pointer_rewrite(
    controller_mounts: dict[str, Any],
) -> None:
    rating_snapshot = _toy_rating_snapshot()
    _write_rating_latest(controller_mounts["tournament"], rating_snapshot)
    pointer_ref, _assignment_ref, _assignment = _write_control_assignment_and_pointer(
        controller_mounts["control"]
    )
    before = (controller_mounts["control"] / pointer_ref.removeprefix("control:")).read_text(
        encoding="utf-8"
    )
    controller_mounts["commits"].failures = ["tournament commit failed"]

    with pytest.raises(RuntimeError, match="failed to commit training candidate leaderboard"):
        modal_arena.curvytron_training_candidate_refresh.local(
            {
                "tournament_id": rating_snapshot["tournament_id"],
                "rating_run_id": rating_snapshot["rating_run_id"],
                "leaderboard_id": "toy-training",
                "snapshot_id": "toy-snapshot",
                "refresh_pointers": [pointer_ref],
            }
        )

    after = (controller_mounts["control"] / pointer_ref.removeprefix("control:")).read_text(
        encoding="utf-8"
    )
    assert after == before
    assert controller_mounts["dict"].values == {}


def test_training_candidate_refresh_aborts_on_assignment_commit_failure_before_pointer_rewrite(
    controller_mounts: dict[str, Any],
) -> None:
    rating_snapshot = _toy_rating_snapshot()
    _write_rating_latest(controller_mounts["tournament"], rating_snapshot)
    pointer_ref, _assignment_ref, _assignment = _write_control_assignment_and_pointer(
        controller_mounts["control"]
    )
    before = (controller_mounts["control"] / pointer_ref.removeprefix("control:")).read_text(
        encoding="utf-8"
    )
    controller_mounts["commits"].failures = [None, "control assignment commit failed"]

    with pytest.raises(RuntimeError, match="failed to commit training candidate assignments"):
        modal_arena.curvytron_training_candidate_refresh.local(
            {
                "tournament_id": rating_snapshot["tournament_id"],
                "rating_run_id": rating_snapshot["rating_run_id"],
                "leaderboard_id": "toy-training",
                "snapshot_id": "toy-snapshot",
                "refresh_pointers": [pointer_ref],
            }
        )

    after = (controller_mounts["control"] / pointer_ref.removeprefix("control:")).read_text(
        encoding="utf-8"
    )
    assert after == before
    assert controller_mounts["dict"].values == {}


def test_training_candidate_refresh_aborts_on_reload_failure_before_writes(
    controller_mounts: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rating_snapshot = _toy_rating_snapshot()
    _write_rating_latest(controller_mounts["tournament"], rating_snapshot)
    pointer_ref, _assignment_ref, _assignment = _write_control_assignment_and_pointer(
        controller_mounts["control"]
    )
    before = (controller_mounts["control"] / pointer_ref.removeprefix("control:")).read_text(
        encoding="utf-8"
    )
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: "reload failed")

    with pytest.raises(RuntimeError, match="reload failed"):
        modal_arena.curvytron_training_candidate_refresh.local(
            {
                "tournament_id": rating_snapshot["tournament_id"],
                "rating_run_id": rating_snapshot["rating_run_id"],
                "leaderboard_id": "toy-training",
                "snapshot_id": "toy-snapshot",
                "refresh_pointers": [pointer_ref],
            }
        )

    after = (controller_mounts["control"] / pointer_ref.removeprefix("control:")).read_text(
        encoding="utf-8"
    )
    assert after == before
    assert controller_mounts["dict"].values == {}


def test_training_candidate_refresh_rejects_stale_pointer_overwrite(
    controller_mounts: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rating_snapshot = _toy_rating_snapshot()
    _write_rating_latest(controller_mounts["tournament"], rating_snapshot)
    pointer_ref, _assignment_ref, _assignment = _write_control_assignment_and_pointer(
        controller_mounts["control"]
    )
    pointer_path = controller_mounts["control"] / pointer_ref.removeprefix("control:")
    changed = {"done": False}
    original_write = modal_arena._write_json_by_volume_ref

    def racing_write(ref: str, payload: dict[str, Any], *, commit: bool = True):
        if not changed["done"] and str(ref).endswith("/assignment.json"):
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer["assignment_sha256"] = "1" * 64
            _write_json(pointer_path, pointer)
            changed["done"] = True
        return original_write(ref, payload, commit=commit)

    monkeypatch.setattr(modal_arena, "_write_json_by_volume_ref", racing_write)

    with pytest.raises(ValueError, match="changed during controller refresh"):
        modal_arena.curvytron_training_candidate_refresh.local(
            {
                "tournament_id": rating_snapshot["tournament_id"],
                "rating_run_id": rating_snapshot["rating_run_id"],
                "leaderboard_id": "toy-training",
                "snapshot_id": "toy-snapshot",
                "refresh_pointers": [pointer_ref],
            }
        )


def test_training_candidate_refresh_missing_active_rank_does_not_change_pointer(
    controller_mounts: dict[str, Any],
) -> None:
    rating_snapshot = _toy_rating_snapshot(rank2_status="provisional")
    _write_rating_latest(controller_mounts["tournament"], rating_snapshot)
    pointer_ref, _assignment_ref, _assignment = _write_control_assignment_and_pointer(
        controller_mounts["control"]
    )
    before = (controller_mounts["control"] / pointer_ref.removeprefix("control:")).read_text(
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="leaderboard missing active ranks"):
        modal_arena.curvytron_training_candidate_refresh.local(
            {
                "tournament_id": rating_snapshot["tournament_id"],
                "rating_run_id": rating_snapshot["rating_run_id"],
                "leaderboard_id": "toy-training",
                "snapshot_id": "toy-snapshot",
                "refresh_pointers": [pointer_ref],
            }
        )

    after = (controller_mounts["control"] / pointer_ref.removeprefix("control:")).read_text(
        encoding="utf-8"
    )
    assert after == before


def test_training_candidate_refresh_strict_missing_tagged_rank_slot_does_not_change_pointer(
    controller_mounts: dict[str, Any],
) -> None:
    rating_snapshot = _toy_rating_snapshot(rank2_status="provisional")
    _write_rating_latest(controller_mounts["tournament"], rating_snapshot)
    pointer_ref, _assignment_ref, _assignment = _write_control_assignment_and_pointer(
        controller_mounts["control"],
        assignment=_scratch_placeholder_rank_assignment(),
    )
    before = (controller_mounts["control"] / pointer_ref.removeprefix("control:")).read_text(
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="leaderboard missing active ranks"):
        modal_arena.curvytron_training_candidate_refresh.local(
            {
                "tournament_id": rating_snapshot["tournament_id"],
                "rating_run_id": rating_snapshot["rating_run_id"],
                "leaderboard_id": "toy-training",
                "snapshot_id": "toy-snapshot",
                "refresh_pointers": [pointer_ref],
            }
        )

    after = (controller_mounts["control"] / pointer_ref.removeprefix("control:")).read_text(
        encoding="utf-8"
    )
    assert after == before


def test_training_candidate_refresh_allow_partial_leaves_missing_placeholder_slot(
    controller_mounts: dict[str, Any],
) -> None:
    result = _run_controller_with_assignment(
        controller_mounts,
        rating_snapshot=_toy_rating_snapshot(rank2_status="provisional"),
        assignment=_scratch_placeholder_rank_assignment(),
        allow_partial_assignment=True,
    )

    assignment_ref = result["rewritten_pointers"][0]["assignment_ref"]
    assignment = json.loads(
        (controller_mounts["control"] / assignment_ref.removeprefix("control:")).read_text(
            encoding="utf-8"
        )
    )
    entries = {entry["name"]: entry for entry in assignment["entries"]}
    slot = entries["scratch_rank2_slot"]
    assert slot["opponent_policy_kind"] == "fixed_straight"
    assert slot["opponent_runtime_mode"] == "blank_canvas_noop"
    assert "opponent_checkpoint_ref" not in slot
    audit_ref = result["rewritten_pointers"][0]["audit_write"]["ref"]
    audit = json.loads((controller_mounts["control"] / audit_ref).read_text(encoding="utf-8"))
    assert audit["missing_ranks"] == [2]


def test_training_candidate_refresh_frozen_entry_without_rank_still_fails(
    controller_mounts: dict[str, Any],
) -> None:
    assignment = _recipe_assignment()
    assignment["entries"] = [
        {
            "name": "legacy_checkpoint_slot",
            "age_label": "legacy_checkpoint_slot",
            "weight": 25,
            "opponent_policy_kind": "frozen_lightzero_checkpoint",
            "opponent_runtime_mode": "normal",
            "opponent_checkpoint_ref": _checkpoint_ref("old-legacy", 1000),
            "opponent_immortal": False,
        }
    ]

    with pytest.raises(ValueError, match="cannot infer leaderboard rank for frozen entry"):
        _run_controller_with_assignment(
            controller_mounts,
            rating_snapshot=_toy_rating_snapshot(),
            assignment=assignment,
        )


def test_training_candidate_refresh_written_pointer_is_trainer_visible(
    controller_mounts: dict[str, Any],
) -> None:
    result = _run_controller(controller_mounts)
    pointer_ref = result["rewritten_pointers"][0]["pointer_ref"]
    assignment_ref = result["rewritten_pointers"][0]["assignment_ref"]
    resolved = train_mod._resolve_opponent_assignment_for_env(
        opponent_assignment_ref=pointer_ref,
        reload_volume_before_read=True,
        reload_checkpoint_volume_before_read=True,
    )

    assert resolved is not None
    assert resolved["assignment_ref"] == assignment_ref.removeprefix("control:")
    assert resolved["assignment_pointer"]["pointed_assignment_ref"] == assignment_ref
    assert resolved["assignment_sha256"] == result["rewritten_pointers"][0]["assignment_sha256"]
    entries = {entry["name"]: entry for entry in resolved["opponent_mixture"]["entries"]}
    assert entries["rank2"]["opponent_checkpoint_resolution"]["mount"] == "control"
    assert entries["rank2"]["opponent_checkpoint_file"]["bytes"] > 0
    lineage_rows = [
        json.loads(line)
        for line in (
            controller_mounts["tournament"]
            / arena.rating_root_ref("toy-arena", "toy-elo")
            / "feedback_loop"
            / "lineage_events.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assignment_sha = result["rewritten_pointers"][0]["assignment_sha256"]
    assert [row["stage"] for row in lineage_rows[-2:]] == [
        "training_candidate_assignment_written",
        "assignment_pointer_rewritten",
    ]
    assert lineage_rows[-2]["assignment_sha256"] == assignment_sha
    assert lineage_rows[-1]["assignment_sha256"] == assignment_sha
    assert resolved["assignment_sha256"] == assignment_sha


def test_training_candidate_assignment_sha_reaches_trainer_load_and_apply_lineage(
    controller_mounts: dict[str, Any],
) -> None:
    result = _run_controller(controller_mounts)
    pointer_ref = result["rewritten_pointers"][0]["pointer_ref"]
    assignment_ref = result["rewritten_pointers"][0]["assignment_ref"]
    assignment_sha = result["rewritten_pointers"][0]["assignment_sha256"]
    resolved = train_mod._resolve_opponent_assignment_for_env(
        opponent_assignment_ref=pointer_ref,
        reload_volume_before_read=True,
        reload_checkpoint_volume_before_read=True,
    )
    assert resolved is not None

    run_id = "synthetic-trainer"
    attempt_id = "try-synthetic-trainer"
    attempt_train_root = controller_mounts["runs"] / train_mod.runs.attempt_train_ref(
        train_mod.TASK_ID,
        run_id,
        attempt_id,
    )
    trainer_lineage_path = train_mod.lineage_events_path(attempt_train_root)
    train_mod._append_trainer_assignment_loaded_lineage(
        trainer_lineage_path,
        status="synthetic",
        run_id=run_id,
        attempt_id=attempt_id,
        opponent_assignment_ref=pointer_ref,
        opponent_assignment=resolved,
    )

    class FakePolicy:
        def __init__(self) -> None:
            self.reset_calls: list[list[int]] = []

        def reset(self, env_ids: list[int]) -> None:
            self.reset_calls.append(env_ids)

    class FakeEnvManager:
        env_num = 64

        def __init__(self) -> None:
            self.ready_obs: dict[int, dict[str, Any]] = {}
            self.last_reset_info: list[dict[str, Any]] = []

        def reset(self, reset_param: Mapping[int, Mapping[str, Any]]) -> None:
            self.ready_obs = {env_id: {"ready": True} for env_id in reset_param}
            self.last_reset_info = []
            for env_id in sorted(reset_param):
                context = dict(reset_param[env_id]["opponent_assignment_context"])
                self.last_reset_info.append(
                    {
                        "opponent_assignment_id": context["assignment_id"],
                        "opponent_assignment_ref": context["assignment_ref"],
                        "opponent_assignment_sha256": context["assignment_sha256"],
                        "opponent_assignment_refresh_index": context["refresh_index"],
                        **{
                            key: value
                            for key, value in context.items()
                            if key.startswith("opponent_split_")
                        },
                    }
                )

    class FakeCollector:
        def __init__(self) -> None:
            self._env = FakeEnvManager()
            self._policy = FakePolicy()
            self.reset_stats: list[int] = []

        def _reset_stat(self, env_id: int) -> None:
            self.reset_stats.append(env_id)

    collector = FakeCollector()
    ready_report = train_mod._apply_opponent_assignment_refresh_to_collector_env(
        collector=collector,
        opponent_assignment=resolved,
        refresh_index=1,
    )
    assert ready_report["ok"] is True
    assert collector._policy.reset_calls == [list(range(64))]
    assert collector.reset_stats == list(range(64))

    train_mod._append_trainer_assignment_applied_lineage(
        trainer_lineage_path,
        run_id=run_id,
        attempt_id=attempt_id,
        event={
            "decision": "applied",
            "train_iter": 2000,
            "bucket": 1,
            "refresh_index": 1,
            "assignment_id": ready_report["assignment_id"],
            "assignment_ref": ready_report["assignment_ref"],
            "assignment_sha256": ready_report["assignment_sha256"],
            "env_ready_report": ready_report,
        },
    )

    tournament_lineage_rows = [
        json.loads(line)
        for line in (
            controller_mounts["tournament"]
            / arena.rating_root_ref("toy-arena", "toy-elo")
            / "feedback_loop"
            / "lineage_events.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    trainer_lineage_rows = [
        json.loads(line) for line in trainer_lineage_path.read_text(encoding="utf-8").splitlines()
    ]
    tournament_assignment_rows = [
        row
        for row in tournament_lineage_rows
        if row["stage"]
        in {
            "training_candidate_assignment_written",
            "assignment_pointer_rewritten",
        }
    ]
    assert [row["stage"] for row in tournament_assignment_rows[-2:]] == [
        "training_candidate_assignment_written",
        "assignment_pointer_rewritten",
    ]
    assert all(row["assignment_sha256"] == assignment_sha for row in tournament_assignment_rows)
    assert [row["stage"] for row in trainer_lineage_rows] == [
        "trainer_assignment_loaded",
        "trainer_assignment_applied",
    ]
    assert trainer_lineage_rows[0]["assignment_ref"] == assignment_ref.removeprefix("control:")
    assert trainer_lineage_rows[0]["assignment_sha256"] == assignment_sha
    assert trainer_lineage_rows[0]["assignment_pointer"]["pointed_assignment_ref"] == (
        assignment_ref
    )
    assert trainer_lineage_rows[1]["assignment_sha256"] == assignment_sha
    assert trainer_lineage_rows[1]["env_ready_report"]["ok"] is True


def test_checkpoint_intake_rating_leaderboard_assignment_trainer_lineage_chain(
    controller_mounts: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tournament_id = "loop-arena"
    rating_run_id = "loop-elo"
    leaderboard_id = "loop-leaderboard"
    training_leaderboard_id = "loop-training"
    checkpoint_refs = [
        _write_training_checkpoint(
            controller_mounts["runs"],
            run_id="loop-a",
            iteration=10,
        ),
        _write_training_checkpoint(
            controller_mounts["runs"],
            run_id="loop-b",
            iteration=10,
        ),
    ]
    intake_state = _FakeDict()
    intake_queue = _FakeQueue()
    rating_loop = _FakeRatingLoop()
    game_map = _FakeGameMap()
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", intake_state)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", intake_queue)
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", rating_loop)
    monkeypatch.setattr(modal_arena, "curvytron_tournament_game", game_map)

    seed_result = modal_arena.curvytron_checkpoint_intake_seed.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_count": 1,
            "pair_selection": arena.RATING_PAIR_SELECTION_ALL_PAIRS,
            "games_per_pair": 3,
            "games_per_shard": 1,
            "decision_source_frames": 1,
            "save_gif": False,
            "max_steps": 64,
        }
    )
    submit_result = modal_arena.curvytron_checkpoint_intake_submit.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoint_refs": checkpoint_refs,
        }
    )
    drain_result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "max_events": 10,
            "spawn_rating": True,
        }
    )
    assert seed_result["checkpoint_count"] == 0
    assert submit_result["accepted_checkpoint_refs"] == checkpoint_refs
    assert drain_result["event_count"] == 2
    assert drain_result["rating_call_id"] == "fc-local-rating-loop"
    assert len(rating_loop.spawned) == 1

    round_result = modal_arena.curvytron_rating_round.local(
        {**rating_loop.spawned[0], "round_index": 0}
    )
    assert round_result["rated_pair_count"] == 1
    assert game_map.calls

    rating_latest = json.loads(
        (
            controller_mounts["tournament"] / arena.rating_latest_ref(tournament_id, rating_run_id)
        ).read_text(encoding="utf-8")
    )
    rating_snapshot_sha = canonical_json_sha256(rating_latest)
    publish_result = modal_arena.curvytron_opponent_leaderboard_publish.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "leaderboard_id": leaderboard_id,
            "snapshot_id": "loop-snapshot",
            "expected_round_id": rating_latest["round_id"],
            "expected_round_index": rating_latest["round_index"],
            "expected_rating_context_hash": rating_latest["context_hash"],
            "expected_roster_hash": rating_latest["roster_hash"],
            "expected_rating_snapshot_sha256": rating_snapshot_sha,
            "active_min_distinct_opponents": 0,
            "active_min_valid_games": 0,
            "max_active_rank": 100,
        }
    )
    assert publish_result["pointer_published"] is True
    assert publish_result["active_count"] == 2

    pointer_ref, _assignment_ref, _assignment = _write_control_assignment_and_pointer(
        controller_mounts["control"]
    )
    refresh_result = modal_arena.curvytron_training_candidate_refresh.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "leaderboard_id": training_leaderboard_id,
            "snapshot_id": "loop-training-snapshot",
            "generation": 1,
            "expected_round_id": rating_latest["round_id"],
            "expected_round_index": rating_latest["round_index"],
            "expected_rating_context_hash": rating_latest["context_hash"],
            "expected_roster_hash": rating_latest["roster_hash"],
            "expected_rating_snapshot_sha256": rating_snapshot_sha,
            "refresh_pointers": [pointer_ref],
            "assignment_bank_run_id": "loop-bank",
            "assignment_bank_attempt_id": "try-loop-bank",
            "active_min_distinct_opponents": 0,
            "active_min_valid_games": 0,
            "max_active_rank": 100,
        }
    )
    assignment_ref = refresh_result["rewritten_pointers"][0]["assignment_ref"]
    assignment_sha = refresh_result["rewritten_pointers"][0]["assignment_sha256"]
    resolved = train_mod._resolve_opponent_assignment_for_env(
        opponent_assignment_ref=pointer_ref,
        reload_volume_before_read=True,
        reload_checkpoint_volume_before_read=True,
    )
    assert resolved is not None
    assert resolved["assignment_sha256"] == assignment_sha

    run_id = "loop-consumer"
    attempt_id = "try-loop-consumer"
    attempt_train_root = controller_mounts["runs"] / train_mod.runs.attempt_train_ref(
        train_mod.TASK_ID,
        run_id,
        attempt_id,
    )
    trainer_lineage_path = train_mod.lineage_events_path(attempt_train_root)
    train_mod._append_trainer_assignment_loaded_lineage(
        trainer_lineage_path,
        status="synthetic",
        run_id=run_id,
        attempt_id=attempt_id,
        opponent_assignment_ref=pointer_ref,
        opponent_assignment=resolved,
    )
    train_mod._append_trainer_assignment_applied_lineage(
        trainer_lineage_path,
        run_id=run_id,
        attempt_id=attempt_id,
        event={
            "decision": "applied",
            "train_iter": 0,
            "bucket": 0,
            "refresh_index": 1,
            "assignment_ref": assignment_ref.removeprefix("control:"),
            "assignment_sha256": assignment_sha,
            "env_ready_report": {"ok": True, "assignment_sha256": assignment_sha},
        },
    )

    producer_lineage_rows = []
    for checkpoint_ref in checkpoint_refs:
        checkpoint_path = controller_mounts["runs"] / checkpoint_ref
        producer_lineage_path = train_mod.lineage_events_path(checkpoint_path.parents[2])
        producer_lineage_rows.extend(
            json.loads(line)
            for line in producer_lineage_path.read_text(encoding="utf-8").splitlines()
        )
    tournament_lineage_rows = [
        json.loads(line)
        for line in (
            controller_mounts["tournament"]
            / arena.rating_root_ref(tournament_id, rating_run_id)
            / "feedback_loop"
            / "lineage_events.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    trainer_lineage_rows = [
        json.loads(line) for line in trainer_lineage_path.read_text(encoding="utf-8").splitlines()
    ]

    assert {row["checkpoint_ref"] for row in producer_lineage_rows} == set(checkpoint_refs)
    expected_tournament_stages = [
        "checkpoint_intake_seen",
        "checkpoint_intake_enqueued",
        "rating_spawn_claimed",
        "rating_round_started",
        "rating_latest_written",
        "rating_round_reduced",
        "leaderboard_published",
        "training_candidate_assignment_written",
        "assignment_pointer_rewritten",
    ]
    tournament_stages = [row["stage"] for row in tournament_lineage_rows]
    positions = {}
    for stage in expected_tournament_stages:
        positions[stage] = tournament_stages.index(stage)
    ordered_positions = [positions[stage] for stage in expected_tournament_stages]
    assert ordered_positions == sorted(ordered_positions)
    assert [row["stage"] for row in trainer_lineage_rows] == [
        "trainer_assignment_loaded",
        "trainer_assignment_applied",
    ]
    assert all(
        row["assignment_sha256"] == assignment_sha
        for row in tournament_lineage_rows
        if row["stage"] in {"training_candidate_assignment_written", "assignment_pointer_rewritten"}
    )
    assert all(row["assignment_sha256"] == assignment_sha for row in trainer_lineage_rows)
