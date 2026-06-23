from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from curvyzero.infra.modal import curvyzero_checkpoint_tournament as modal_arena
from curvyzero.env.observation_surface_contract import (
    POLICY_OBSERVATION_CONTRACT_ID,
    POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
    POLICY_OBSERVATION_SEAT_MAPPING,
)
from curvyzero.training.opponent_leaderboard import canonical_json_sha256
from curvyzero.tournament import curvytron_checkpoint_tournament as arena


def _checkpoint_ref(run_id: str, iteration: int) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/"
        f"{run_id}/checkpoints/lightzero/iteration_{iteration}.pth.tar"
    )


def _train_exp_checkpoint_ref(run_id: str, attempt_id: str, iteration: int) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/"
        f"{run_id}/attempts/{attempt_id}/train/lightzero_exp/ckpt/"
        f"iteration_{iteration}.pth.tar"
    )


def test_intake_submit_validation_accepts_current_policy_render_defaults():
    modal_arena._reject_submit_cli_scheduler_overrides(
        checkpoint_iteration=-1,
        checkpoint_selection=arena.CHECKPOINT_SELECTION_LATEST,
        games_per_pair=arena.DEFAULT_GAMES_PER_PAIR,
        games_per_shard=arena.DEFAULT_GAMES_PER_SHARD,
        reuse_policies_per_shard=arena.DEFAULT_REUSE_POLICIES_PER_SHARD,
        round_count=arena.DEFAULT_RATING_ROUND_COUNT,
        continue_from_latest=False,
        pairs_per_round=0,
        placement_min_games=0,
        placement_min_opponents=20,
        pair_selection=arena.DEFAULT_RATING_PAIR_SELECTION,
        initial_rating=arena.DEFAULT_RATING_INITIAL_RATING,
        active_pool_limit=arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT,
        stop_when_stable=False,
        seed=0,
        max_steps=arena.DEFAULT_MAX_STEPS,
        decision_ms=arena.DEFAULT_DECISION_MS,
        decision_source_frames=arena.DEFAULT_DECISION_SOURCE_FRAMES,
        source_physics_step_ms=arena.DEFAULT_SOURCE_PHYSICS_STEP_MS,
        policy_mode=arena.POLICY_MODE_EVAL,
        collect_temperature=arena.DEFAULT_COLLECT_TEMPERATURE,
        collect_epsilon=arena.DEFAULT_COLLECT_EPSILON,
        policy_trail_render_mode=arena.DEFAULT_POLICY_TRAIL_RENDER_MODE,
        policy_bonus_render_mode=arena.DEFAULT_POLICY_BONUS_RENDER_MODE,
        num_simulations=arena.DEFAULT_NUM_SIMULATIONS,
        save_gif=arena.DEFAULT_SAVE_GIF,
        gif_sample_games_per_pair=arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR,
        gif_sample_strategy=arena.DEFAULT_GIF_SAMPLE_STRATEGY,
        intake_enqueue_existing=False,
        intake_spawn_rating=False,
        intake_spawn_if_existing=False,
        intake_allow_rating_overrides=False,
        intake_max_events=100,
        intake_claim_stale_after_seconds=modal_arena.DEFAULT_CHECKPOINT_INTAKE_CLAIM_STALE_SECONDS,
        intake_active=True,
        max_runs=0,
    )


def _fake_game(pair: dict, index: int, outcome: str, *, ok: bool = True) -> dict:
    if outcome == "seat_0_win":
        winner, loser, draw = 0, 1, False
    elif outcome == "seat_1_win":
        winner, loser, draw = 1, 0, False
    elif outcome == "draw":
        winner, loser, draw = None, None, True
    else:
        winner, loser, draw = None, None, False
    return {
        "ok": ok,
        "game_id": f"game-{index:06d}",
        "game_index": index,
        "players": pair["players"],
        "physical_steps": 10 + index,
        "score": {
            "outcome": outcome,
            "winner_seat": winner,
            "loser_seat": loser,
            "draw": draw,
            "physical_steps": 10 + index,
        },
        "summary_ref": (
            f"tournaments/curvytron/arena-a/battles/{pair['battle_id']}/"
            f"games/game-{index:06d}/summary.json"
        ),
    }


def _distinct_opponents_by_checkpoint(pairs: list[dict]) -> dict[str, set[str]]:
    opponents: dict[str, set[str]] = {}
    for pair in pairs:
        left = str(pair["players"][0]["checkpoint_id"])
        right = str(pair["players"][1]["checkpoint_id"])
        opponents.setdefault(left, set()).add(right)
        opponents.setdefault(right, set()).add(left)
    return opponents


def _write_review_fixture(tmp_path):
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    marker = tmp_path / arena.tournament_marker_ref(tournament_id)
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    latest_path = tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id)
    latest_path.parent.mkdir(parents=True)
    modal_arena.runs.write_json(
        latest_path,
        {
            "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": "round-000000",
            "ratings": [
                {
                    "rank": 1,
                    "checkpoint_id": "ckpt-a",
                    "label": "A",
                    "rating": 1600.0,
                    "games": 6,
                    "wins": 3,
                    "losses": 1,
                    "draws": 2,
                    "win_rate": 0.5,
                    "distinct_opponents": 2,
                    "failure_count": 0,
                },
                {
                    "rank": 2,
                    "checkpoint_id": "ckpt-b",
                    "label": "B",
                    "rating": 1500.0,
                    "games": 6,
                    "wins": 2,
                    "losses": 2,
                    "draws": 2,
                    "win_rate": 1 / 3,
                    "distinct_opponents": 2,
                    "failure_count": 0,
                },
                {
                    "rank": 3,
                    "checkpoint_id": "ckpt-c",
                    "label": "C",
                    "rating": 1400.0,
                    "games": 6,
                    "wins": 1,
                    "losses": 3,
                    "draws": 2,
                    "win_rate": 1 / 6,
                    "distinct_opponents": 2,
                    "failure_count": 0,
                },
            ],
        },
    )
    modal_arena.runs.write_json(
        tmp_path / arena.tournament_battle_index_ref(tournament_id),
        {
            "schema_id": "curvyzero_curvytron_checkpoint_tournament_battle_index/v0",
            "tournament_id": tournament_id,
            "total": 3,
            "rows": [
                {
                    "tournament_id": tournament_id,
                    "battle_id": "battle-bc",
                    "players": [
                        {"checkpoint_id": "ckpt-b", "label": "B", "seat": 0},
                        {"checkpoint_id": "ckpt-c", "label": "C", "seat": 1},
                    ],
                    "tally": {
                        "completed_count": 3,
                        "failure_count": 0,
                        "draw_count": 1,
                        "wins_by_checkpoint": {"ckpt-b": 1, "ckpt-c": 1},
                        "wins_by_seat": {"seat_0": 1, "seat_1": 1},
                        "average_physical_steps": 10.0,
                    },
                    "summary_ref": ("tournaments/curvytron/arena-a/battles/battle-bc/battle.json"),
                    "first_gif_ref": None,
                    "updated_ts": 3.0,
                    "ok": True,
                },
                {
                    "tournament_id": tournament_id,
                    "battle_id": "battle-ac",
                    "players": [
                        {"checkpoint_id": "ckpt-c", "label": "C", "seat": 0},
                        {"checkpoint_id": "ckpt-a", "label": "A", "seat": 1},
                    ],
                    "tally": {
                        "completed_count": 3,
                        "failure_count": 0,
                        "draw_count": 1,
                        "wins_by_checkpoint": {"ckpt-c": 1, "ckpt-a": 1},
                        "wins_by_seat": {"seat_0": 1, "seat_1": 1},
                        "average_physical_steps": 11.0,
                    },
                    "summary_ref": ("tournaments/curvytron/arena-a/battles/battle-ac/battle.json"),
                    "first_gif_ref": None,
                    "updated_ts": 2.0,
                    "ok": True,
                },
                {
                    "tournament_id": tournament_id,
                    "battle_id": "battle-ab",
                    "players": [
                        {"checkpoint_id": "ckpt-a", "label": "A", "seat": 0},
                        {"checkpoint_id": "ckpt-b", "label": "B", "seat": 1},
                    ],
                    "tally": {
                        "completed_count": 6,
                        "failure_count": 0,
                        "draw_count": 2,
                        "wins_by_checkpoint": {"ckpt-a": 3, "ckpt-b": 1},
                        "wins_by_seat": {"seat_0": 3, "seat_1": 1},
                        "average_physical_steps": 12.0,
                    },
                    "summary_ref": ("tournaments/curvytron/arena-a/battles/battle-ab/battle.json"),
                    "first_gif_ref": (
                        "tournaments/curvytron/arena-a/battles/battle-ab/games/game-000000/game.gif"
                    ),
                    "updated_ts": 1.0,
                    "ok": True,
                },
            ],
        },
    )
    battle_ab_ref = arena.battle_summary_ref(tournament_id, "battle-ab")
    modal_arena.runs.write_json(
        tmp_path / battle_ab_ref,
        {
            "schema_id": arena.BATTLE_SCHEMA_ID,
            "ok": True,
            "tournament_id": tournament_id,
            "battle_id": "battle-ab",
            "pair_index": 0,
            "players": [
                {"checkpoint_id": "ckpt-a", "label": "A", "seat": 0},
                {"checkpoint_id": "ckpt-b", "label": "B", "seat": 1},
            ],
            "tally": {
                "completed_count": 6,
                "failure_count": 0,
                "draw_count": 2,
                "wins_by_checkpoint": {"ckpt-a": 3, "ckpt-b": 1},
                "wins_by_seat": {"seat_0": 3, "seat_1": 1},
                "average_physical_steps": 12.0,
            },
            "first_gif_ref": (
                "tournaments/curvytron/arena-a/battles/battle-ab/games/game-000000/game.gif"
            ),
            "game_summary_ref_count": 6,
            "result_detail_mode": "shard_tally",
            "summary_ref": battle_ab_ref.as_posix(),
        },
    )
    outcomes = [
        "seat_0_win",
        "seat_0_win",
        "draw",
        "seat_1_win",
        "seat_0_win",
        "draw",
    ]
    for index, outcome in enumerate(outcomes):
        game_id = f"game-{index:06d}"
        gif_ref = (
            f"tournaments/curvytron/arena-a/battles/battle-ab/games/{game_id}/game.gif"
            if index < 5
            else None
        )
        modal_arena.runs.write_json(
            tmp_path / arena.game_summary_ref(tournament_id, "battle-ab", game_id),
            {
                "schema_id": arena.GAME_SCHEMA_ID,
                "ok": True,
                "tournament_id": tournament_id,
                "battle_id": "battle-ab",
                "pair_index": 0,
                "game_id": game_id,
                "game_index": index,
                "seed": 1000 + index,
                "physical_steps": 10 + index,
                "score": {
                    "outcome": outcome,
                    "winner_seat": 0
                    if outcome == "seat_0_win"
                    else 1
                    if outcome == "seat_1_win"
                    else None,
                    "loser_seat": 1
                    if outcome == "seat_0_win"
                    else 0
                    if outcome == "seat_1_win"
                    else None,
                    "draw": outcome == "draw",
                    "physical_steps": 10 + index,
                },
                "gif_ref": gif_ref,
                "summary_ref": arena.game_summary_ref(
                    tournament_id,
                    "battle-ab",
                    game_id,
                ).as_posix(),
            },
        )
    return tournament_id, rating_run_id


def test_tournament_lineage_event_is_scoped_to_rating_run(tmp_path, monkeypatch):
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)

    result = modal_arena._append_tournament_lineage_event(
        stage="leaderboard_published",
        tournament_id="arena-a",
        rating_run_id="elo-a",
        leaderboard_id="leaderboard-a",
        snapshot_id="snapshot-a",
    )

    assert result["ok"] is True
    path = (
        tmp_path
        / arena.rating_root_ref("arena-a", "elo-a")
        / "feedback_loop"
        / "lineage_events.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "event_id": result["event_id"],
            "leaderboard_id": "leaderboard-a",
            "observed_at": rows[0]["observed_at"],
            "rating_run_id": "elo-a",
            "schema_id": "curvyzero_feedback_loop_lineage_event/v1",
            "snapshot_id": "snapshot-a",
            "stage": "leaderboard_published",
            "status": "ok",
            "tournament_id": "arena-a",
        }
    ]


def _write_live_rating_fixture(tmp_path):
    tournament_id = "arena-live"
    rating_run_id = "elo-live"
    marker = tmp_path / arena.tournament_marker_ref(tournament_id)
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    refs = [
        _checkpoint_ref("run-a", 0),
        _checkpoint_ref("run-b", 10),
        _checkpoint_ref("run-c", 20),
    ]
    spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": refs,
            "games_per_pair": 3,
            "games_per_shard": 2,
            "save_gif": True,
            "gif_sample_games_per_pair": 1,
        }
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_config_ref(tournament_id, rating_run_id),
        {
            "schema_id": arena.RATING_CONFIG_SCHEMA_ID,
            "rating_spec": spec,
        },
    )
    round_id = arena.rating_round_id(0)
    pair_specs = arena.build_rating_round_pair_specs(spec, round_index=0)
    input_payload = {
        "schema_id": arena.RATING_ROUND_SCHEMA_ID,
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "round_id": round_id,
        "round_index": 0,
        "rating_spec": spec,
        "pair_count": len(pair_specs),
        "game_count": sum(int(pair["games_per_pair"]) for pair in pair_specs),
        "pair_specs": pair_specs,
    }
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        input_payload,
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_progress_ref(tournament_id, rating_run_id),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 0,
            "status": "running",
            "phase": "games_running",
            "pair_count": len(pair_specs),
            "game_count": input_payload["game_count"],
            "completed_pair_count": 0,
            "completed_game_count": 0,
        },
    )
    pair = pair_specs[0]
    games = [
        {
            **_fake_game(pair, index, "seat_0_win"),
            "tournament_id": tournament_id,
            "battle_id": pair["battle_id"],
            "pair_index": pair["pair_index"],
        }
        for index in range(3)
    ]
    shard_id = "shard-000000-games-000000-000002"
    gif_ref = arena.game_gif_ref(
        tournament_id,
        pair["battle_id"],
        "game-000000",
    ).as_posix()
    modal_arena.runs.write_json(
        tmp_path / arena.game_shard_summary_ref(tournament_id, pair["battle_id"], shard_id),
        {
            "schema_id": arena.GAME_SHARD_SCHEMA_ID,
            "tournament_id": tournament_id,
            "battle_id": pair["battle_id"],
            "pair_index": pair["pair_index"],
            "shard_id": shard_id,
            "shard_index": 0,
            "game_count": 3,
            "games": [arena._compact_game_result(game) for game in games],
            "tally": arena.tally_game_results(games),
            "game_summary_ref_count": 3,
            "first_gif_ref": gif_ref,
            "sample_gif_refs": [gif_ref],
        },
    )
    return tournament_id, rating_run_id, spec, pair


def test_build_pair_specs_defaults_to_unordered_no_self_pairs() -> None:
    refs = [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10), _checkpoint_ref("run-c", 20)]

    pairs = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=refs,
        games_per_pair=5,
        seed=100,
    )

    assert len(pairs) == 3
    assert [pair["pair_index"] for pair in pairs] == [0, 1, 2]
    assert pairs[0]["games_per_pair"] == 5
    assert pairs[0]["seed"] == 100
    assert pairs[1]["seed"] == 10_100
    assert pairs[0]["players"][0]["seat"] == 0
    assert pairs[0]["players"][1]["seat"] == 1
    assert pairs[0]["players"][0]["checkpoint_ref"] == refs[0]


def test_games_per_pair_must_be_odd() -> None:
    refs = [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)]

    with pytest.raises(ValueError, match="games_per_pair must be odd"):
        arena.build_pair_specs(
            tournament_id="arena-a",
            checkpoints=refs,
            games_per_pair=2,
        )

    with pytest.raises(ValueError, match="games_per_pair must be odd"):
        arena.normalize_rating_spec(
            {
                "tournament_id": "arena-a",
                "rating_run_id": "elo-test",
                "checkpoints": refs,
                "games_per_pair": 10,
            }
        )

    with pytest.raises(ValueError, match="games_per_pair must be odd"):
        arena.estimate_tournament_plan(checkpoint_count=2, games_per_pair=50)


def test_default_games_per_pair_is_21() -> None:
    refs = [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)]

    spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
        }
    )

    assert arena.DEFAULT_GAMES_PER_PAIR == 21
    assert arena.DEFAULT_GAMES_PER_SHARD == 1
    assert spec["games_per_pair"] == 21
    assert spec["games_per_shard"] == 1
    assert arena.DEFAULT_MAX_STEPS == 1_048_576
    assert spec["max_steps"] == 1_048_576


def test_rating_spec_default_active_pool_keeps_all_submitted_checkpoints() -> None:
    refs = [_checkpoint_ref(f"run-{index:03d}", index) for index in range(212)]

    spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
        }
    )

    assert arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT == 100
    assert spec["active_pool_limit"] == 212


def test_build_game_specs_for_pair_uses_stable_ids_and_seeds() -> None:
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=[_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
        games_per_pair=3,
        seed=7,
        max_steps=64,
    )[0]

    games = arena.build_game_specs_for_pair(pair)

    assert [game["game_id"] for game in games] == [
        "game-000000",
        "game-000001",
        "game-000002",
    ]
    assert [game["seed"] for game in games] == [7, 8, 9]
    assert all(game["max_steps"] == 64 for game in games)


def test_build_game_specs_randomizes_balanced_seat_order_by_default() -> None:
    checkpoints = [
        {"checkpoint_ref": _checkpoint_ref("run-a", 0), "checkpoint_id": "ckpt-a"},
        {"checkpoint_ref": _checkpoint_ref("run-b", 10), "checkpoint_id": "ckpt-b"},
    ]
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=checkpoints,
        games_per_pair=5,
        seed=17,
    )[0]

    games = arena.build_game_specs_for_pair(pair)
    repeat_games = arena.build_game_specs_for_pair(dict(pair))
    swapped = [bool(game["seat_order"]["swapped"]) for game in games]

    assert pair["seat_order_mode"] == arena.SEAT_ORDER_BALANCED_RANDOM
    assert [game["seat_order"] for game in repeat_games] == [game["seat_order"] for game in games]
    assert set(swapped) == {False, True}
    assert abs(swapped.count(False) - swapped.count(True)) == 1
    for game in games:
        assert game["battle_players"] == pair["players"]
        assert game["seat_order_mode"] == arena.SEAT_ORDER_BALANCED_RANDOM
        assert [player["checkpoint_id"] for player in game["players"]] == game["seat_order"][
            "seat_to_checkpoint_id"
        ]

    fixed_pair = arena.normalize_pair_spec({**pair, "seat_order_mode": arena.SEAT_ORDER_FIXED})
    fixed_games = arena.build_game_specs_for_pair(fixed_pair)

    assert all(not game["seat_order"]["swapped"] for game in fixed_games)
    assert all(game["players"] == pair["players"] for game in fixed_games)


def test_failure_game_summary_payload_is_scoreable_without_volume_write() -> None:
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=[_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
        games_per_pair=3,
        seed=17,
    )[0]
    game = arena.build_game_specs_for_pair(pair)[0]

    summary = arena.failure_game_summary_payload(game, OSError("primary volume failure"))

    assert summary["ok"] is False
    assert summary["battle_id"] == pair["battle_id"]
    assert summary["pair_index"] == pair["pair_index"]
    assert summary["game_id"] == "game-000000"
    assert summary["seed"] == 17
    assert summary["players"] == game["players"]
    assert summary["error_type"] == "OSError"
    tally = arena.tally_game_results([summary])
    assert tally["game_count"] == 1
    assert tally["failure_count"] == 1
    assert tally["completed_count"] == 0


def test_failure_game_summary_or_inline_survives_failure_summary_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=[_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
        games_per_pair=3,
        seed=17,
    )[0]
    game = arena.build_game_specs_for_pair(pair)[0]

    def fail_to_write_summary(*args: object, **kwargs: object) -> dict[str, object]:
        raise OSError("summary volume write failed")

    monkeypatch.setattr(
        modal_arena.arena,
        "failure_game_summary",
        fail_to_write_summary,
    )

    result = modal_arena._failure_game_summary_or_inline(
        game,
        RuntimeError("primary game failure"),
        artifact_mount=tmp_path,
    )

    assert result["ok"] is False
    assert result["error_type"] == "RuntimeError"
    assert result["failure_summary_write_error_type"] == "OSError"
    assert "summary volume write failed" in result["failure_summary_write_error"]
    assert result["summary_ref"] is None
    assert str(result["intended_summary_ref"]).endswith("/summary.json")
    compact = arena._compact_game_result(result)
    assert compact["failure_summary_write_error_type"] == "OSError"
    assert compact["intended_summary_ref"] == result["intended_summary_ref"]


def test_source_timing_settings_pass_through_pair_game_and_rating_specs() -> None:
    refs = [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)]
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=refs,
        games_per_pair=3,
        decision_ms=120.0,
        decision_source_frames=6,
        source_physics_step_ms=20.0,
    )[0]
    game = arena.build_game_specs_for_pair(pair)[0]
    shard = arena.build_game_shard_specs_for_pair(pair, games_per_shard=3)[0]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "decision_ms": 120.0,
            "decision_source_frames": 6,
            "source_physics_step_ms": 20.0,
        }
    )
    rating_pair = arena.build_rating_round_pair_specs(rating_spec)[0]

    assert pair["decision_ms"] == 120.0
    assert pair["decision_source_frames"] == 6
    assert pair["source_physics_step_ms"] == 20.0
    assert game["decision_ms"] == 120.0
    assert game["decision_source_frames"] == 6
    assert game["source_physics_step_ms"] == 20.0
    assert shard["game_specs"][0]["decision_source_frames"] == 6
    assert rating_pair["decision_source_frames"] == 6
    assert rating_pair["source_physics_step_ms"] == 20.0


def test_compact_summary_reader_ignores_large_action_trace(tmp_path: Path) -> None:
    refs = [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)]
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=refs,
        games_per_pair=1,
        seed=17,
    )[0]
    path = (
        tmp_path
        / "tournaments/curvytron/arena-a/battles"
        / pair["battle_id"]
        / "games/game-000000/summary.json"
    )
    path.parent.mkdir(parents=True)
    payload = {
        "ok": True,
        "game_id": "game-000000",
        "game_index": 0,
        "seed": 123,
        "score": {
            "outcome": "seat_1_win",
            "winner_seat": 1,
            "loser_seat": 0,
            "draw": False,
            "physical_steps": 321,
        },
        "seat_order": {
            "mode": "balanced_random",
            "seat_to_logical_index": [1, 0],
        },
        "seat_order_mode": "balanced_random",
        "action_trace": [{"step": i, "payload": "x" * 1000} for i in range(64)],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    compact = modal_arena._read_compact_game_summary(path, pair=pair, mount=tmp_path)

    assert compact["ok"] is True
    assert compact["score"]["outcome"] == "seat_1_win"
    assert compact["physical_steps"] == 321
    assert compact["players"][0]["checkpoint_id"] == pair["players"][1]["checkpoint_id"]
    assert compact["players"][1]["checkpoint_id"] == pair["players"][0]["checkpoint_id"]
    assert compact["battle_players"] == pair["players"]
    assert "action_trace" not in compact


def test_build_game_shard_specs_for_pair_chunks_existing_game_specs() -> None:
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=[_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
        games_per_pair=5,
        games_per_shard=2,
        reuse_policies_per_shard=False,
        seed=11,
    )[0]

    direct_games = arena.build_game_specs_for_pair(pair)
    shards = arena.build_game_shard_specs_for_pair(pair)
    flattened = [game for shard in shards for game in shard["game_specs"]]

    assert [shard["game_count"] for shard in shards] == [2, 2, 1]
    assert [shard["game_index_start"] for shard in shards] == [0, 2, 4]
    assert [shard["game_index_end"] for shard in shards] == [1, 3, 4]
    assert all(shard["reuse_policies"] is False for shard in shards)
    assert flattened == direct_games
    with pytest.raises(ValueError):
        arena.build_game_shard_specs_for_pair(pair, games_per_shard=0)


def test_modal_game_work_specs_can_use_game_or_shard_work() -> None:
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=[_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
        games_per_pair=3,
    )[0]

    default_kind, default_specs = modal_arena._build_game_work_specs([pair])
    game_kind, game_specs = modal_arena._build_game_work_specs(
        [pair],
        games_per_shard=1,
    )
    shard_kind, shard_specs = modal_arena._build_game_work_specs(
        [pair],
        games_per_shard=2,
    )

    assert default_kind == "game"
    assert len(default_specs) == 3
    assert game_kind == "game"
    assert len(game_specs) == 3
    assert shard_kind == "shard"
    assert len(shard_specs) == 2
    assert shard_specs[0]["reuse_policies"] is True


def test_flatten_game_results_from_shards_is_order_stable() -> None:
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=[_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
        games_per_pair=3,
    )[0]
    games = [
        {
            **_fake_game(pair, index, outcome),
            "tournament_id": pair["tournament_id"],
            "battle_id": pair["battle_id"],
            "pair_index": pair["pair_index"],
        }
        for index, outcome in enumerate(["seat_0_win", "draw", "seat_1_win"])
    ]

    flattened = modal_arena._flatten_game_results_from_shards(
        [
            {"games": [games[2], games[1]]},
            {"games": [games[0], games[1]]},
        ]
    )
    summary = arena.summarize_pair_results(pair, flattened)

    assert [game["game_index"] for game in flattened] == [0, 1, 2]
    assert summary["tally"]["completed_count"] == 3


def test_shard_tally_pair_summary_matches_game_list_rating() -> None:
    refs = [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "games_per_pair": 5,
            "min_valid_fraction": 0.5,
        }
    )
    pair = arena.build_rating_round_pair_specs(rating_spec)[0]
    games = [
        _fake_game(pair, 0, "seat_0_win"),
        _fake_game(pair, 1, "draw"),
        _fake_game(pair, 2, "seat_1_win"),
        _fake_game(pair, 3, "unfinished", ok=False),
        _fake_game(pair, 4, "unfinished", ok=False),
    ]
    game_summary = arena.summarize_pair_results(pair, games)
    tally_summary = arena.summarize_pair_from_tally(
        pair,
        tally=arena.merge_game_tallies(
            [
                arena.tally_game_results(games[:2]),
                arena.tally_game_results(games[2:]),
            ]
        ),
    )

    game_snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=[game_summary],
        rating_spec=rating_spec,
        created_at="2026-05-13T00:00:00Z",
    )
    tally_snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=[tally_summary],
        rating_spec=rating_spec,
        created_at="2026-05-13T00:00:00Z",
    )

    assert "games" not in tally_summary
    assert tally_summary["tally"] == game_summary["tally"]
    assert tally_snapshot["pair_rating_results"] == game_snapshot["pair_rating_results"]
    assert tally_snapshot["ratings"] == game_snapshot["ratings"]


def test_rating_counts_wins_from_each_games_actual_seat_order() -> None:
    checkpoints = [
        {"checkpoint_ref": _checkpoint_ref("run-a", 0), "checkpoint_id": "ckpt-a"},
        {"checkpoint_ref": _checkpoint_ref("run-b", 10), "checkpoint_id": "ckpt-b"},
    ]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": checkpoints,
            "games_per_pair": 3,
        }
    )
    pair = arena.build_rating_round_pair_specs(rating_spec)[0]
    swapped_players = [
        {**pair["players"][1], "seat": 0},
        {**pair["players"][0], "seat": 1},
    ]
    games = [
        {
            **_fake_game(pair, 0, "seat_0_win"),
            "players": swapped_players,
            "seat_order": {
                "swapped": True,
                "seat_to_checkpoint_id": ["ckpt-b", "ckpt-a"],
            },
        },
        _fake_game(pair, 1, "seat_0_win"),
        _fake_game(pair, 2, "draw"),
    ]

    summary = arena.summarize_pair_results(pair, games)
    rating = arena.rating_result_from_pair_summary(summary, rating_spec)
    standings = arena.standings_from_pair_results([summary])

    assert rating["wins_a"] == 1
    assert rating["wins_b"] == 1
    assert rating["draws"] == 1
    assert rating["score_a"] == pytest.approx(0.5)
    rows = {row["checkpoint_id"]: row for row in standings["standings"]}
    assert rows["ckpt-a"]["wins"] == 1
    assert rows["ckpt-b"]["wins"] == 1


def test_tally_only_rating_rejects_seat_win_fallback() -> None:
    checkpoints = [
        {"checkpoint_ref": _checkpoint_ref("run-a", 0), "checkpoint_id": "ckpt-a"},
        {"checkpoint_ref": _checkpoint_ref("run-b", 10), "checkpoint_id": "ckpt-b"},
    ]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": checkpoints,
            "games_per_pair": 3,
        }
    )
    pair = arena.build_rating_round_pair_specs(rating_spec)[0]
    summary = arena.summarize_pair_from_tally(
        pair,
        tally={
            "game_count": 3,
            "completed_count": 3,
            "failure_count": 0,
            "draw_count": 1,
            "wins_by_seat": {"seat_0": 1, "seat_1": 1},
        },
    )

    with pytest.raises(ValueError, match="wins_by_checkpoint"):
        arena.rating_result_from_pair_summary(summary, rating_spec)


def test_preloaded_policy_entries_reorder_to_actual_game_seats() -> None:
    players = [
        {"checkpoint_ref": _checkpoint_ref("run-b", 10), "checkpoint_id": "ckpt-b"},
        {"checkpoint_ref": _checkpoint_ref("run-a", 0), "checkpoint_id": "ckpt-a"},
    ]
    entries = [
        {"policy": "policy-a", "checkpoint_ref": _checkpoint_ref("run-a", 0)},
        {"policy": "policy-b", "checkpoint_ref": _checkpoint_ref("run-b", 10)},
    ]

    ordered = arena._preloaded_policy_entries_for_players(entries, players)

    assert [entry["policy"] for entry in ordered] == ["policy-b", "policy-a"]


def test_modal_shard_tally_summarizer_writes_lean_pair_summary(tmp_path) -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
            "games_per_shard": 2,
        }
    )
    pair = arena.build_rating_round_pair_specs(rating_spec)[0]
    games = [
        {
            **_fake_game(pair, index, outcome),
            "tournament_id": pair["tournament_id"],
            "battle_id": pair["battle_id"],
            "pair_index": pair["pair_index"],
        }
        for index, outcome in enumerate(["seat_0_win", "draw", "seat_1_win"])
    ]
    shard_results = [
        {
            "battle_id": pair["battle_id"],
            "pair_index": pair["pair_index"],
            "shard_id": "shard-000000",
            "shard_index": 0,
            "game_count": 2,
            "tally": arena.tally_game_results(games[:2]),
            "game_summary_ref_count": 2,
            "summary_ref": arena.game_shard_summary_ref(
                pair["tournament_id"],
                pair["battle_id"],
                "shard-000000",
            ).as_posix(),
            "sample_gif_refs": [
                "tournaments/curvytron/arena-a/battles/battle-a/shards/sample-a.gif"
            ],
        },
        {
            "battle_id": pair["battle_id"],
            "pair_index": pair["pair_index"],
            "shard_id": "shard-000001",
            "shard_index": 1,
            "game_count": 1,
            "tally": arena.tally_game_results(games[2:]),
            "game_summary_ref_count": 1,
            "summary_ref": arena.game_shard_summary_ref(
                pair["tournament_id"],
                pair["battle_id"],
                "shard-000001",
            ).as_posix(),
        },
    ]

    pair_results, game_count = modal_arena._summarize_pair_results_from_shard_tallies(
        mount=tmp_path,
        pair_specs=[pair],
        shard_results=list(reversed(shard_results)),
        started_at="2026-05-13T00:00:00Z",
        work_summary={"work_kind": "shard", "parent_result_mode": "shard_tallies"},
        rating_run_id=rating_spec["rating_run_id"],
        round_id=arena.rating_round_id(0),
        round_index=0,
    )
    progress = modal_arena._rating_progress_from_pair_results(
        input_payload={
            "tournament_id": rating_spec["tournament_id"],
            "rating_run_id": rating_spec["rating_run_id"],
            "round_id": arena.rating_round_id(0),
            "round_index": 0,
            "game_count": 3,
            "pair_specs": [pair],
        },
        pair_results=pair_results,
        work_summary={"work_kind": "shard", "parent_result_mode": "shard_tallies"},
    )
    written_summary = modal_arena._read_json(
        tmp_path / arena.battle_summary_ref(pair["tournament_id"], pair["battle_id"])
    )

    assert game_count == 3
    assert len(pair_results) == 1
    assert "games" not in pair_results[0]
    assert pair_results[0]["game_summary_ref_count"] == 3
    assert pair_results[0]["shard_summary_ref_count"] == 2
    assert len(pair_results[0]["shard_summary_refs"]) == 2
    assert len(pair_results[0]["sample_gif_refs"]) == 1
    assert pair_results[0]["tally"]["completed_count"] == 3
    assert pair_results[0]["result_detail_mode"] == "shard_tally"
    assert written_summary["rating_run_id"] == "elo-test"
    assert progress["completed_game_count"] == 3
    assert progress["completed_pair_count"] == 1
    assert progress["count_basis"] == "shard_tallies"


def test_gif_sampling_limits_saved_games_per_pair() -> None:
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=[_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
        games_per_pair=5,
        save_gif=True,
        gif_sample_games_per_pair=3,
    )[0]

    games = arena.build_game_specs_for_pair(pair)

    assert [game["save_gif"] for game in games] == [True, False, True, False, True]
    assert [game["gif_sample_strategy"] for game in games] == [
        "evenly_spaced",
        "evenly_spaced",
        "evenly_spaced",
        "evenly_spaced",
        "evenly_spaced",
    ]

    first_n = {**pair, "gif_sample_strategy": "first_n"}
    assert [game["save_gif"] for game in arena.build_game_specs_for_pair(first_n)] == [
        True,
        True,
        True,
        False,
        False,
    ]

    no_gifs = {**pair, "gif_sample_games_per_pair": 0}
    assert [game["save_gif"] for game in arena.build_game_specs_for_pair(no_gifs)] == [
        False,
        False,
        False,
        False,
        False,
    ]

    all_gifs = {**pair, "gif_sample_games_per_pair": -1}
    assert [game["save_gif"] for game in arena.build_game_specs_for_pair(all_gifs)] == [
        True,
        True,
        True,
        True,
        True,
    ]


def test_compact_game_result_keeps_grouping_keys() -> None:
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=[_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
        games_per_pair=1,
    )[0]
    game = arena.build_game_specs_for_pair(pair)[0]
    compact = arena._compact_game_result(
        {
            **game,
            "ok": True,
            "players": pair["players"],
            "score": {"outcome": "draw", "draw": True},
            "checkpoint_reload_error": "reload failed",
            "commit_error": "commit failed",
            "summary_timing_rewrite_error": "rewrite failed",
        }
    )

    assert compact["tournament_id"] == "arena-a"
    assert compact["battle_id"] == pair["battle_id"]
    assert compact["pair_index"] == pair["pair_index"]
    assert compact["checkpoint_reload_error"] == "reload failed"
    assert compact["commit_error"] == "commit failed"
    assert compact["summary_timing_rewrite_error"] == "rewrite failed"


def test_tournament_plan_estimate_counts_pairs_games_and_gif_samples() -> None:
    estimate = arena.estimate_tournament_plan(
        checkpoint_count=50,
        games_per_pair=51,
        save_gif=True,
        gif_sample_games_per_pair=1,
    )

    assert estimate["pair_count"] == 1225
    assert estimate["game_count"] == 62_475
    assert estimate["game_call_count"] == 62_475
    assert estimate["approx_game_worker_commit_count"] == 62_475
    assert estimate["gif_count"] == 1225

    sharded = arena.estimate_tournament_plan(
        checkpoint_count=50,
        games_per_pair=51,
        games_per_shard=10,
    )

    assert sharded["game_count"] == 62_475
    assert sharded["game_call_count"] == 7_350
    assert sharded["approx_game_worker_commit_count"] == 7_350

    probe = arena.estimate_tournament_plan(
        checkpoint_count=50,
        games_per_pair=3,
        pairs_per_round=5,
        save_gif=False,
    )

    assert probe["pair_candidate_count"] == 1225
    assert probe["pair_count"] == 5
    assert probe["game_count"] == 15


def test_default_gif_frame_size_is_full_raw_canvas_size(tmp_path) -> None:
    Image = pytest.importorskip("PIL.Image")
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=[_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
    )[0]
    game = arena.build_game_specs_for_pair(pair)[0]
    path = tmp_path / "game.gif"
    frames = np.zeros(
        (2, arena.DEFAULT_FRAME_SIZE, arena.DEFAULT_FRAME_SIZE, 3),
        dtype=np.uint8,
    )

    info = arena._save_gif(frames, path, fps=arena.DEFAULT_GIF_FPS)

    assert arena.DEFAULT_GIF_FPS == 800.0
    assert info["duration_ms_per_frame"] == 1
    assert game["frame_size"] == 704
    assert info["pixel_size"] == [704, 704]
    with Image.open(path) as image:
        assert image.size == (704, 704)


def test_tournament_render_contract_pins_policy_surface_and_full_gif() -> None:
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=[
            {
                "checkpoint_ref": _checkpoint_ref("run-a", 0),
                "policy_trail_render_mode": arena.DEFAULT_POLICY_TRAIL_RENDER_MODE,
                "policy_bonus_render_mode": "simple_symbols",
            },
            {
                "checkpoint_ref": _checkpoint_ref("run-b", 10),
                "policy_trail_render_mode": "browser_lines",
                "policy_bonus_render_mode": arena.DEFAULT_POLICY_BONUS_RENDER_MODE,
            },
        ],
        games_per_pair=1,
        frame_size=64,
        save_frames_npz=True,
    )[0]
    game = arena.build_game_specs_for_pair(pair)[0]

    assert pair["policy_trail_render_mode"] == arena.DEFAULT_POLICY_TRAIL_RENDER_MODE
    assert pair["policy_bonus_render_mode"] == arena.DEFAULT_POLICY_BONUS_RENDER_MODE
    assert pair["policy_observation_backend"] == arena.DEFAULT_TOURNAMENT_POLICY_OBSERVATION_BACKEND
    assert pair["gif_trail_render_mode"] == "browser_lines"
    assert pair["frame_size"] == 704
    assert game["policy_trail_render_mode"] == arena.DEFAULT_POLICY_TRAIL_RENDER_MODE
    assert game["policy_bonus_render_mode"] == arena.DEFAULT_POLICY_BONUS_RENDER_MODE
    assert game["policy_observation_backend"] == arena.DEFAULT_TOURNAMENT_POLICY_OBSERVATION_BACKEND
    assert game["gif_trail_render_mode"] == "browser_lines"
    assert game["frame_size"] == 704
    assert game["players"][0]["policy_trail_render_mode"] == "browser_lines"
    assert game["players"][1]["policy_trail_render_mode"] == "browser_lines"
    assert game["players"][0]["policy_bonus_render_mode"] == "simple_symbols"
    assert game["players"][1]["policy_bonus_render_mode"] == "simple_symbols"
    assert game["players"][0]["policy_observation_backend"] == "cpu_oracle"
    assert game["players"][1]["policy_observation_backend"] == "cpu_oracle"


def test_tournament_rejects_legacy_policy_surface() -> None:
    with pytest.raises(ValueError, match="policy surface must be"):
        arena.build_pair_specs(
            tournament_id="arena-a",
            checkpoints=[
                {
                    "checkpoint_ref": _checkpoint_ref("run-a", 0),
                    "policy_trail_render_mode": "body_circles_fast",
                    "policy_bonus_render_mode": "simple_symbols",
                },
                _checkpoint_ref("run-b", 10),
            ],
            games_per_pair=1,
        )


def test_tournament_rejects_lab_policy_observation_backend() -> None:
    with pytest.raises(ValueError, match="policy observation backend"):
        arena.build_pair_specs(
            tournament_id="arena-a",
            checkpoints=[
                {
                    "checkpoint_ref": _checkpoint_ref("run-a", 0),
                    "policy_observation_backend": "jax_gpu",
                },
                _checkpoint_ref("run-b", 10),
            ],
            games_per_pair=1,
        )


def test_checkpoint_spec_reads_policy_render_mode_from_observation_contract() -> None:
    checkpoint = arena.normalize_checkpoint_spec(
        {
            "checkpoint_ref": _checkpoint_ref("run-a", 0),
            "observation_contract": {
                "trail_render_mode": arena.DEFAULT_POLICY_TRAIL_RENDER_MODE,
                "bonus_render_mode": "simple_symbols",
                "backend": "cpu_oracle",
            },
        },
    )

    assert checkpoint["policy_trail_render_mode"] == arena.DEFAULT_POLICY_TRAIL_RENDER_MODE
    assert checkpoint["policy_bonus_render_mode"] == "simple_symbols"
    assert checkpoint["policy_observation_backend"] == "cpu_oracle"
    assert checkpoint["observation_contract"]["backend"] == "cpu_oracle"


def test_checkpoint_label_from_ref_includes_run_and_iteration() -> None:
    checkpoint = arena.normalize_checkpoint_spec(
        (
            "training/lightzero-curvytron-visual-survival/"
            "curvy-survive-bonus-blank-browser-heavy-collect64-r298-s1141899/"
            "attempts/try-blank-browser-heavy-collect64-r298-s1141899/"
            "train/lightzero_exp/ckpt/iteration_300773.pth.tar"
        )
    )

    assert checkpoint["label"] == "blank-browser-heavy-collect64-r298 i300773"


def test_checkpoint_labels_disambiguate_duplicate_visible_names() -> None:
    checkpoints = arena.normalize_checkpoint_specs(
        [
            (
                "training/lightzero-curvytron-visual-survival/"
                "curvy-survive-bonus-blank-fast-heavy-base-r159-s1113201/"
                "attempts/a/train/lightzero_exp/ckpt/iteration_300000.pth.tar"
            ),
            (
                "training/lightzero-curvytron-visual-survival/"
                "curvy-survive-bonus-blank-fast-heavy-base-r159-s1119999/"
                "attempts/b/train/lightzero_exp/ckpt/iteration_300000.pth.tar"
            ),
        ]
    )

    labels = [checkpoint["label"] for checkpoint in checkpoints]
    assert labels == [
        "blank-fast-heavy-base-r159 i300000 (s1113201)",
        "blank-fast-heavy-base-r159 i300000 (s1119999)",
    ]


def test_checkpoint_policy_render_mode_falls_back_to_run_metadata(tmp_path) -> None:
    ref = _checkpoint_ref("run-a", 0)
    run_json = tmp_path / "training/lightzero-curvytron-visual-survival/run-a/run.json"
    run_json.parent.mkdir(parents=True)
    run_json.write_text(
        json.dumps(
            {
                "config": {
                    "source_state_trail_render_mode": arena.DEFAULT_POLICY_TRAIL_RENDER_MODE,
                    "source_state_bonus_render_mode": "simple_symbols",
                    "decision_ms": 200.0,
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        arena._checkpoint_policy_trail_render_mode_from_ref(ref, mount=tmp_path)
        == arena.DEFAULT_POLICY_TRAIL_RENDER_MODE
    )
    assert (
        arena._checkpoint_policy_bonus_render_mode_from_ref(ref, mount=tmp_path) == "simple_symbols"
    )
    assert arena._checkpoint_runtime_settings_from_ref(ref, mount=tmp_path)["decision_ms"] == 200.0


def test_checkpoint_policy_metadata_sidecar_overrides_run_metadata(tmp_path) -> None:
    ref = _checkpoint_ref("run-a", 0)
    run_json = tmp_path / "training/lightzero-curvytron-visual-survival/run-a/run.json"
    run_json.parent.mkdir(parents=True)
    run_json.write_text(
        json.dumps(
            {
                "config": {
                    "source_state_trail_render_mode": "body_circles_fast",
                    "source_state_bonus_render_mode": "none",
                    "decision_ms": 200.0,
                }
            }
        ),
        encoding="utf-8",
    )
    sidecar_ref = arena.checkpoint_policy_metadata_sidecar_ref(ref)
    sidecar_path = tmp_path / sidecar_ref
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(
            {
                "schema_id": arena.CHECKPOINT_POLICY_METADATA_SIDECAR_SCHEMA_ID,
                "policy_trail_render_mode": "browser_lines",
                "policy_bonus_render_mode": "simple_symbols",
                "policy_observation_backend": "cpu_oracle",
                "policy_observation_contract_id": POLICY_OBSERVATION_CONTRACT_ID,
                "policy_observation_perspective_schema_id": (
                    POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
                ),
                "observation_contract": {
                    "contract_id": POLICY_OBSERVATION_CONTRACT_ID,
                    "perspective_schema_id": POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
                    "trail_render_mode": "browser_lines",
                    "bonus_render_mode": "simple_symbols",
                    "backend": "cpu_oracle",
                },
                "decision_ms": 1000.0 / 60.0,
                "model_env_variant": "source_state_fixed_opponent",
                "model_reward_variant": "survival_plus_bonus_no_outcome",
            }
        ),
        encoding="utf-8",
    )

    assert (
        arena.checkpoint_policy_metadata_from_ref(ref, mount=tmp_path)["schema_id"]
        == arena.CHECKPOINT_POLICY_METADATA_SIDECAR_SCHEMA_ID
    )
    assert (
        arena._checkpoint_policy_trail_render_mode_from_ref(
            ref,
            mount=tmp_path,
        )
        == "browser_lines"
    )
    assert (
        arena._checkpoint_policy_bonus_render_mode_from_ref(
            ref,
            mount=tmp_path,
        )
        == "simple_symbols"
    )
    assert (
        arena._checkpoint_policy_observation_backend_from_ref(
            ref,
            mount=tmp_path,
        )
        == "cpu_oracle"
    )
    assert (
        arena._checkpoint_policy_observation_contract_id_from_ref(
            ref,
            mount=tmp_path,
        )
        == POLICY_OBSERVATION_CONTRACT_ID
    )
    assert (
        arena._checkpoint_policy_observation_perspective_schema_id_from_ref(
            ref,
            mount=tmp_path,
        )
        == POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
    )
    assert arena._checkpoint_runtime_settings_from_ref(ref, mount=tmp_path)[
        "decision_ms"
    ] == pytest.approx(1000.0 / 60.0)
    assert arena._checkpoint_model_contract_from_ref(ref, mount=tmp_path) == {
        "model_env_variant": "source_state_fixed_opponent",
        "model_reward_variant": "survival_plus_bonus_no_outcome",
    }


def test_loaded_policy_metadata_wins_for_raw_checkpoint_specs() -> None:
    player = arena.normalize_checkpoint_spec(_checkpoint_ref("run-a", 0))
    load = {
        "policy_trail_render_mode": "body_circles_fast",
        "policy_bonus_render_mode": "simple_symbols",
        "policy_observation_backend": "cpu_oracle",
    }

    assert (
        arena._policy_metadata_value_for_player(
            player,
            load,
            "policy_trail_render_mode",
            "browser_lines",
        )
        == "body_circles_fast"
    )
    assert (
        arena._policy_metadata_value_for_player(
            player,
            load,
            "policy_bonus_render_mode",
            None,
        )
        == "simple_symbols"
    )
    assert (
        arena._policy_metadata_value_for_player(
            player,
            load,
            "policy_observation_backend",
            "cpu_oracle",
        )
        == "cpu_oracle"
    )


def test_policy_loader_recovers_model_contract_from_checkpoint_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    from curvyzero.infra.modal import lightzero_curvytron_visual_survival_eval as eval_mod

    ref = (
        "training/lightzero-curvytron-visual-survival/run-a/attempts/attempt-a/"
        "train/lightzero_exp/ckpt/iteration_0.pth.tar"
    )
    checkpoint_path = tmp_path / ref
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    command_path = (
        tmp_path
        / "training/lightzero-curvytron-visual-survival/run-a/attempts/attempt-a/command.json"
    )
    command_path.write_text(
        json.dumps(
            {
                "policy_trail_render_mode": "browser_lines",
                "policy_bonus_render_mode": "simple_symbols",
                "policy_observation_backend": "cpu_oracle",
                "policy_observation_contract_id": POLICY_OBSERVATION_CONTRACT_ID,
                "policy_observation_perspective_schema_id": (
                    POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
                ),
                "env_variant": "source_state_fixed_opponent",
                "training_reward_variant": "survival_plus_bonus_no_outcome",
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeEnv:
        def close(self) -> None:
            pass

    monkeypatch.setattr(eval_mod, "_torch_load", lambda _path: {"weights": {}})
    monkeypatch.setattr(eval_mod, "_find_state_dict", lambda _payload: ("weights", {}))

    def fake_make_policy_and_env(**kwargs):
        captured.update(kwargs)
        return object(), FakeEnv(), {"schema": "fake"}

    monkeypatch.setattr(eval_mod, "_make_policy_and_env", fake_make_policy_and_env)

    loaded = arena._load_policy_from_checkpoint(
        checkpoint_ref=ref,
        checkpoint_state_key=None,
        seed=1,
        source_max_steps=16,
        num_simulations=2,
        batch_size=1,
        telemetry_path=tmp_path / "telemetry.jsonl",
        mount=tmp_path,
        remote_root=None,
        model_env_variant=None,
        model_reward_variant=None,
    )

    assert captured["model_env_variant"] == "source_state_fixed_opponent"
    assert captured["model_reward_variant"] == "survival_plus_bonus_no_outcome"
    assert loaded["model_env_variant"] == "source_state_fixed_opponent"
    assert loaded["model_reward_variant"] == "survival_plus_bonus_no_outcome"
    assert loaded["model_contract_source"]["metadata"] == {
        "model_env_variant": "source_state_fixed_opponent",
        "model_reward_variant": "survival_plus_bonus_no_outcome",
    }
    assert loaded["policy_observation_contract_id"] == POLICY_OBSERVATION_CONTRACT_ID
    assert (
        loaded["policy_observation_perspective_schema_id"]
        == POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
    )


def test_policy_loader_rejects_checkpoint_without_observation_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    from curvyzero.infra.modal import lightzero_curvytron_visual_survival_eval as eval_mod

    ref = (
        "training/lightzero-curvytron-visual-survival/run-a/attempts/attempt-a/"
        "train/lightzero_exp/ckpt/iteration_0.pth.tar"
    )
    checkpoint_path = tmp_path / ref
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(eval_mod, "_torch_load", lambda _path: {"weights": {}})

    with pytest.raises(ValueError, match="required policy observation metadata"):
        arena._load_policy_from_checkpoint(
            checkpoint_ref=ref,
            checkpoint_state_key=None,
            seed=1,
            source_max_steps=16,
            num_simulations=2,
            batch_size=1,
            telemetry_path=tmp_path / "telemetry.jsonl",
            mount=tmp_path,
            remote_root=None,
            model_env_variant=None,
            model_reward_variant=None,
        )


def test_loaded_policy_observation_metadata_rejects_missing_contract_fields() -> None:
    with pytest.raises(ValueError, match="policy_observation_contract_id"):
        arena._require_loaded_policy_observation_metadata(
            checkpoint_ref=_checkpoint_ref("run-a", 0),
            policy_trail_render_mode="browser_lines",
            policy_bonus_render_mode="simple_symbols",
            policy_observation_backend="cpu_oracle",
            policy_observation_contract_id=None,
            policy_observation_perspective_schema_id=(POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID),
        )

    with pytest.raises(ValueError, match="policy_observation_perspective_schema_id"):
        arena._require_loaded_policy_observation_metadata(
            checkpoint_ref=_checkpoint_ref("run-a", 0),
            policy_trail_render_mode="browser_lines",
            policy_bonus_render_mode="simple_symbols",
            policy_observation_backend="cpu_oracle",
            policy_observation_contract_id=POLICY_OBSERVATION_CONTRACT_ID,
            policy_observation_perspective_schema_id=None,
        )


def test_loaded_policy_observation_metadata_rejects_wrong_contract() -> None:
    with pytest.raises(ValueError, match="policy observation contract"):
        arena._require_loaded_policy_observation_metadata(
            checkpoint_ref=_checkpoint_ref("run-a", 0),
            policy_trail_render_mode="browser_lines",
            policy_bonus_render_mode="simple_symbols",
            policy_observation_backend="cpu_oracle",
            policy_observation_contract_id="legacy/raw-player-zero/v0",
            policy_observation_perspective_schema_id=(POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID),
        )


def test_loaded_policy_observation_metadata_rejects_wrong_perspective_schema() -> None:
    with pytest.raises(ValueError, match="policy observation perspective schema"):
        arena._require_loaded_policy_observation_metadata(
            checkpoint_ref=_checkpoint_ref("run-a", 0),
            policy_trail_render_mode="browser_lines",
            policy_bonus_render_mode="simple_symbols",
            policy_observation_backend="cpu_oracle",
            policy_observation_contract_id=POLICY_OBSERVATION_CONTRACT_ID,
            policy_observation_perspective_schema_id="legacy/raw-player-colors/v0",
        )


def test_loaded_policy_observation_contracts_reject_shadowed_nested_mismatch() -> None:
    with pytest.raises(ValueError, match="observation_contract.contract_id"):
        arena._require_loaded_policy_observation_contracts_consistent(
            checkpoint_ref=_checkpoint_ref("run-a", 0),
            payloads=[
                {
                    "policy_observation_contract_id": POLICY_OBSERVATION_CONTRACT_ID,
                    "observation_contract": {
                        "contract_id": "legacy/raw-player-zero/v0",
                    },
                }
            ],
            policy_trail_render_mode="browser_lines",
            policy_bonus_render_mode="simple_symbols",
            policy_observation_backend="cpu_oracle",
            policy_observation_contract_id=POLICY_OBSERVATION_CONTRACT_ID,
            policy_observation_perspective_schema_id=(POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID),
        )


def test_loaded_policy_observation_contracts_reject_shadowed_perspective_mismatch() -> None:
    with pytest.raises(ValueError, match="observation_contract.perspective_schema_id"):
        arena._require_loaded_policy_observation_contracts_consistent(
            checkpoint_ref=_checkpoint_ref("run-a", 0),
            payloads=[
                {
                    "policy_observation_perspective_schema_id": (
                        POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
                    ),
                    "observation_contract": {
                        "perspective_schema_id": "legacy/raw-player-colors/v0",
                    },
                }
            ],
            policy_trail_render_mode="browser_lines",
            policy_bonus_render_mode="simple_symbols",
            policy_observation_backend="cpu_oracle",
            policy_observation_contract_id=POLICY_OBSERVATION_CONTRACT_ID,
            policy_observation_perspective_schema_id=(POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID),
        )


def test_source_frame_runtime_settings_use_source_substeps() -> None:
    pair = arena.normalize_pair_spec(
        {
            "tournament_id": "arena-a",
            "players": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
        }
    )

    settings = arena._source_frame_runtime_settings(
        {},
        pair,
        [],
        max_steps=64,
    )

    assert settings["decision_source_frames"] == 1
    assert settings["decision_ms"] == pytest.approx(16.666666666666668)
    assert settings["source_max_ticks"] == 64


def test_source_frame_runtime_settings_reject_checkpoint_runtime_mismatch() -> None:
    pair = arena.normalize_pair_spec(
        {
            "tournament_id": "arena-a",
            "players": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "decision_source_frames": 1,
            "source_physics_step_ms": 20.0,
            "decision_ms": 20.0,
        }
    )

    with pytest.raises(ValueError, match="source_physics_step_ms does not match"):
        arena._source_frame_runtime_settings(
            {},
            pair,
            [
                {
                    "runtime_settings": {
                        "decision_source_frames": 1,
                        "source_physics_step_ms": 1000.0 / 60.0,
                        "decision_ms": 1000.0 / 60.0,
                    }
                },
                {
                    "runtime_settings": {
                        "decision_source_frames": 1,
                        "source_physics_step_ms": 1000.0 / 60.0,
                        "decision_ms": 1000.0 / 60.0,
                    }
                },
            ],
            max_steps=64,
        )


def test_checkpoint_game_uses_per_seat_policy_modes_and_full_rich_gif(
    tmp_path,
    monkeypatch,
) -> None:
    from curvyzero.env import vector_multiplayer_env as env_mod
    from curvyzero.env import vector_visual_observation as visual_mod
    from curvyzero.training import curvytron_current_policy_selfplay_smoke as stack_mod

    stack_surfaces: list[tuple[str, str]] = []
    render_calls: list[dict[str, object]] = []
    policy_calls: list[tuple[int, float]] = []
    env_kwargs: list[dict[str, object]] = []

    class FakeBatch:
        def __init__(self, *, done: bool = False) -> None:
            self.action_mask = np.ones((1, 2, env_mod.ACTION_COUNT), dtype=np.float32)
            self.done = np.asarray([done], dtype=bool)
            self.truncated = np.asarray([done], dtype=bool)
            self.info = {
                "death_count": [0],
                "draw": [done],
                "terminal_reason_name": ["timeout" if done else "running"],
            }

    class FakeEnv:
        batch_size = 1
        player_count = 2

        def __init__(self, *args, **kwargs) -> None:
            env_kwargs.append(dict(kwargs))
            self.state = object()
            self.step_count = 0

        def reset(self, *, seed: int):
            self.step_count = 0
            return FakeBatch(done=False)

        def step(self, actions, *, timer_advance_ms: float):
            self.step_count += 1
            return FakeBatch(done=True)

    class FakeStack:
        def __init__(
            self,
            *,
            batch_size: int,
            player_count: int,
            trail_render_mode: str,
            bonus_render_mode: str,
        ):
            self.trail_render_mode = trail_render_mode
            self.bonus_render_mode = bonus_render_mode
            stack_surfaces.append((trail_render_mode, bonus_render_mode))

        def update(self, env, *, copy: bool = True):
            value = (
                7.0
                if (self.trail_render_mode, self.bonus_render_mode)
                == (arena.DEFAULT_POLICY_TRAIL_RENDER_MODE, arena.DEFAULT_POLICY_BONUS_RENDER_MODE)
                else 11.0
            )
            observation = np.zeros((1, 2, 4, 64, 64), dtype=np.float32)
            observation[0, 0] = value
            observation[0, 1] = value + 10.0
            return observation

    def fake_render(state, *, row: int, frame_size: int, trail_render_mode: str):
        render_calls.append({"frame_size": frame_size, "trail_render_mode": trail_render_mode})
        return np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

    def fake_policy_action(
        *,
        policy,
        observation,
        policy_mode: str,
        collect_temperature: float,
        collect_epsilon: float,
    ):
        policy_calls.append(
            (
                int(observation["to_play"]),
                float(np.asarray(observation["observation"])[0, 0, 0]),
            )
        )
        return {"ok": True, "action": 0, "compact_output": {}}

    monkeypatch.setattr(env_mod, "VectorMultiplayerEnv", FakeEnv)
    monkeypatch.setattr(stack_mod, "SourceStateGray64Stack4", FakeStack)
    monkeypatch.setattr(visual_mod, "render_source_state_rgb_canvas_like", fake_render)
    monkeypatch.setattr(arena, "_policy_action", fake_policy_action)

    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=[_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
        games_per_pair=1,
        max_steps=2,
        save_frames_npz=True,
        frame_size=64,
    )[0]
    game = arena.build_game_specs_for_pair(pair)[0]

    summary = arena.run_checkpoint_game(
        game,
        artifact_mount=tmp_path,
        preloaded_policy_entries=[
            {
                "policy": object(),
                "policy_trail_render_mode": "browser_lines",
                "policy_bonus_render_mode": "simple_symbols",
            },
            {
                "policy": object(),
                "policy_trail_render_mode": "browser_lines",
                "policy_bonus_render_mode": "simple_symbols",
            },
        ],
    )

    assert stack_surfaces == [("browser_lines", "simple_symbols")]
    assert env_kwargs[0]["decision_source_frames"] == 1
    assert env_kwargs[0]["decision_ms"] == pytest.approx(16.666666666666668)
    assert env_kwargs[0]["max_ticks"] == 2
    assert render_calls
    assert {call["frame_size"] for call in render_calls} == {704}
    assert {call["trail_render_mode"] for call in render_calls} == {"browser_lines"}
    assert policy_calls == [(-1, 7.0), (-1, 17.0)]
    assert summary["frame_size"] == 704
    assert summary["gif_trail_render_mode"] == "browser_lines"
    assert summary["decision_source_frames"] == 1
    assert summary["decision_ms"] == pytest.approx(1000.0 / 60.0)
    assert summary["source_physics_step_ms"] == pytest.approx(1000.0 / 60.0)
    assert summary["source_max_ticks"] == 2
    assert summary["render_contract"]["gif_frame_size"] == 704
    assert summary["render_contract"]["gif_trail_render_mode"] == "browser_lines"
    assert summary["policy_observation_perspective"]["perspective"] == ("controlled_player_view")
    assert summary["policy_observation_perspective"]["seat_mapping"] == (
        POLICY_OBSERVATION_SEAT_MAPPING
    )
    assert summary["render_contract"]["policy_observation_perspective"] == (
        "controlled_player_view"
    )
    assert summary["policy_trail_render_modes"] == {
        "seat_0": "browser_lines",
        "seat_1": "browser_lines",
    }
    assert summary["policy_bonus_render_modes"] == {
        "seat_0": "simple_symbols",
        "seat_1": "simple_symbols",
    }


def test_pair_spec_rejects_unknown_policy_mode() -> None:
    with pytest.raises(ValueError):
        arena.build_pair_specs(
            tournament_id="arena-a",
            checkpoints=[_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            policy_mode="mystery",
        )


def test_score_from_info_scores_first_death() -> None:
    score = arena.score_from_info(
        {
            "death_count": [1],
            "death_player": [[1, -1]],
            "terminal_reason_name": ["round_survivor_win"],
        },
        done=True,
        truncated=False,
        physical_steps=42,
        max_steps=512,
    )

    assert score["outcome"] == "seat_0_win"
    assert score["winner_seat"] == 0
    assert score["loser_seat"] == 1
    assert score["score_reason"] == "single_player_death"


def test_score_from_info_draws_simultaneous_death_and_timeout() -> None:
    simultaneous = arena.score_from_info(
        {
            "death_count": [2],
            "death_player": [[0, 1]],
            "terminal_reason_name": ["simultaneous_death"],
        },
        done=True,
        truncated=False,
        physical_steps=12,
        max_steps=64,
    )
    timeout = arena.score_from_info(
        {"death_count": [0], "terminal_reason_name": ["timeout"]},
        done=True,
        truncated=True,
        physical_steps=64,
        max_steps=64,
    )

    assert simultaneous["draw"] is True
    assert simultaneous["score_reason"] == "simultaneous_death_same_public_step"
    assert timeout["draw"] is True
    assert timeout["score_reason"] == "draw_or_timeout"


def test_pair_summary_and_standings_are_recomputable() -> None:
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=[_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
        games_per_pair=3,
    )[0]
    games = [
        _fake_game(pair, 0, "seat_0_win"),
        _fake_game(pair, 1, "seat_1_win"),
        _fake_game(pair, 2, "draw"),
    ]

    summary = arena.summarize_pair_results(pair, games)
    standings = arena.standings_from_pair_results([summary])

    assert summary["tally"]["completed_count"] == 3
    assert summary["tally"]["wins_by_seat"] == {"seat_0": 1, "seat_1": 1}
    assert summary["tally"]["draw_count"] == 1
    assert standings["checkpoint_count"] == 2
    assert {row["games"] for row in standings["standings"]} == {3}


def test_rating_snapshot_uses_batch_elo_and_is_order_stable() -> None:
    refs = [
        _checkpoint_ref("run-a", 0),
        _checkpoint_ref("run-b", 10),
        _checkpoint_ref("run-c", 20),
    ]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "games_per_pair": 3,
        }
    )
    pairs = arena.build_rating_round_pair_specs(rating_spec)
    summaries = [
        arena.summarize_pair_results(
            pairs[0],
            [
                _fake_game(pairs[0], 0, "seat_0_win"),
                _fake_game(pairs[0], 1, "seat_0_win"),
                _fake_game(pairs[0], 2, "draw"),
            ],
        ),
        arena.summarize_pair_results(
            pairs[1],
            [
                _fake_game(pairs[1], 0, "seat_1_win"),
                _fake_game(pairs[1], 1, "seat_1_win"),
                _fake_game(pairs[1], 2, "draw"),
            ],
        ),
        arena.summarize_pair_results(
            pairs[2],
            [
                _fake_game(pairs[2], 0, "seat_0_win"),
                _fake_game(pairs[2], 1, "draw"),
                _fake_game(pairs[2], 2, "draw"),
            ],
        ),
    ]

    snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=list(reversed(summaries)),
        rating_spec=rating_spec,
        round_index=0,
        created_at="2026-05-13T00:00:00Z",
    )
    stable_again = arena.rating_snapshot_from_pair_results(
        pair_results=summaries,
        rating_spec=rating_spec,
        round_index=0,
        created_at="2026-05-13T00:00:00Z",
    )

    assert snapshot["schema_id"] == arena.RATING_SNAPSHOT_SCHEMA_ID
    assert snapshot["rated_pair_count"] == 3
    assert snapshot["ratings"] == stable_again["ratings"]
    assert snapshot["pair_rating_results"] == stable_again["pair_rating_results"]
    assert snapshot["ratings"][0]["rating"] > arena.DEFAULT_RATING_INITIAL_RATING


def test_rating_snapshot_status_requires_placement_evidence_target() -> None:
    refs = [_checkpoint_ref(f"run-{index:02d}", index * 10) for index in range(21)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "games_per_pair": 21,
            "placement_min_games": 420,
            "placement_min_opponents": 20,
        }
    )
    checkpoints = rating_spec["checkpoints"]
    previous_snapshot = {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "context_hash": arena.rating_context_hash(rating_spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(checkpoints),
        "ratings": [
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "rating": 1500.0 + index,
                "games": 420,
                "opponent_ids": [
                    checkpoints[(index + offset) % len(checkpoints)]["checkpoint_id"]
                    for offset in range(1, 21)
                ],
            }
            for index, checkpoint in enumerate(checkpoints)
        ],
    }
    previous_snapshot["ratings"][0]["opponent_ids"] = previous_snapshot["ratings"][0][
        "opponent_ids"
    ][:5]

    snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=[],
        rating_spec=rating_spec,
        previous_snapshot=previous_snapshot,
        round_index=1,
        created_at="2026-05-13T00:00:00Z",
    )
    rows_by_id = {row["checkpoint_id"]: row for row in snapshot["ratings"]}

    assert rows_by_id[checkpoints[0]["checkpoint_id"]]["status"] == "provisional"
    assert rows_by_id[checkpoints[1]["checkpoint_id"]]["status"] == "active"


def test_rating_pair_specs_carry_shard_settings() -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 5,
            "games_per_shard": 2,
            "reuse_policies_per_shard": False,
        }
    )

    pair = arena.build_rating_round_pair_specs(rating_spec)[0]
    shards = arena.build_game_shard_specs_for_pair(pair)

    assert pair["games_per_shard"] == 2
    assert pair["reuse_policies_per_shard"] is False
    assert [shard["game_count"] for shard in shards] == [2, 2, 1]
    assert all(shard["reuse_policies"] is False for shard in shards)


def test_adaptive_v0_requires_pair_budget() -> None:
    with pytest.raises(ValueError, match="requires pairs_per_round"):
        arena.normalize_rating_spec(
            {
                "tournament_id": "arena-a",
                "rating_run_id": "elo-test",
                "pair_selection": "adaptive_v0",
                "checkpoints": [
                    _checkpoint_ref("run-a", 0),
                    _checkpoint_ref("run-b", 10),
                ],
            }
        )


def test_adaptive_v0_pair_specs_are_budgeted_unique_and_tagged() -> None:
    refs = [_checkpoint_ref(f"run-{index:02d}", index * 10) for index in range(12)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 7,
            "games_per_pair": 3,
            "placement_min_games": 0,
            "placement_min_opponents": 0,
            "seed": 123,
        }
    )
    previous_snapshot = {
        "ratings": [
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "rating": 1500.0 + index * 10.0,
                "games": index * 20,
                "distinct_opponents": min(index, 4),
                "rated_battles": index,
                "last_round_delta": float(index % 3),
            }
            for index, checkpoint in enumerate(rating_spec["checkpoints"])
        ]
    }

    pairs = arena.build_rating_round_pair_specs(
        rating_spec,
        previous_snapshot=previous_snapshot,
        round_index=2,
    )

    assert len(pairs) == 7
    assert len({pair["pair_key"] for pair in pairs}) == len(pairs)
    assert {pair["scheduled_round_index"] for pair in pairs} == {2}
    assert {pair["schedule_reason"] for pair in pairs}
    assert all(pair.get("schedule", {}).get("reason") for pair in pairs)


def test_adaptive_v0_handles_large_rating_roster_without_sigmoid_overflow() -> None:
    refs = [_checkpoint_ref(f"run-{index:04d}", index * 10) for index in range(1600)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 25,
            "games_per_pair": 3,
            "placement_min_games": 0,
            "placement_min_opponents": 0,
            "seed": 123,
        }
    )
    previous_snapshot = {
        "ratings": [
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "rating": 1500.0 - index,
                "games": 100,
                "distinct_opponents": 25,
                "rated_battles": 25,
                "last_round_delta": 0.0,
            }
            for index, checkpoint in enumerate(rating_spec["checkpoints"])
        ]
    }

    pairs = arena.build_rating_round_pair_specs(
        rating_spec,
        previous_snapshot=previous_snapshot,
        round_index=3,
    )

    assert len(pairs) == 25
    assert len({pair["pair_key"] for pair in pairs}) == 25


def test_adaptive_v0_covers_new_checkpoints_when_budget_allows() -> None:
    refs = [_checkpoint_ref(f"run-{index:02d}", index * 10) for index in range(20)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 10,
            "games_per_pair": 3,
            "placement_min_opponents": 1,
            "seed": 123,
        }
    )

    pairs = arena.build_rating_round_pair_specs(rating_spec, round_index=0)
    played = {player["checkpoint_id"] for pair in pairs for player in pair["players"]}

    assert 10 <= len(pairs) <= 20
    assert played == {checkpoint["checkpoint_id"] for checkpoint in rating_spec["checkpoints"]}
    assert {pair["schedule_reason"] for pair in pairs} == {arena.SCHEDULE_REASON_PLACEMENT}


def test_adaptive_v0_covers_mixed_new_checkpoints_with_bounded_placement() -> None:
    refs = [_checkpoint_ref(f"run-{index:02d}", index * 10) for index in range(10)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 6,
            "games_per_pair": 3,
            "placement_min_opponents": 1,
            "seed": 321,
        }
    )
    established = rating_spec["checkpoints"][:4]
    new = rating_spec["checkpoints"][4:]
    previous_snapshot = {
        "ratings": [
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "rating": 1500.0 + index * 20.0,
                "games": 21,
                "distinct_opponents": 3,
                "opponent_ids": [established[(index + 1) % len(established)]["checkpoint_id"]],
            }
            for index, checkpoint in enumerate(established)
        ]
    }

    pairs = arena.build_rating_round_pair_specs(
        rating_spec,
        previous_snapshot=previous_snapshot,
        round_index=1,
    )
    new_ids = {checkpoint["checkpoint_id"] for checkpoint in new}
    played = {player["checkpoint_id"] for pair in pairs for player in pair["players"]}

    assert len(pairs) == 6
    assert new_ids <= played
    placement_pairs = [
        pair for pair in pairs if pair["schedule_reason"] == arena.SCHEDULE_REASON_PLACEMENT
    ]
    assert len(placement_pairs) == 6
    assert new_ids <= {
        player["checkpoint_id"] for pair in placement_pairs for player in pair["players"]
    }


def test_adaptive_v0_respects_pair_budget_even_with_new_checkpoints() -> None:
    refs = [_checkpoint_ref(f"run-{index:02d}", index * 10) for index in range(5)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 1,
            "games_per_pair": 3,
            "seed": 456,
        }
    )

    pairs = arena.build_rating_round_pair_specs(rating_spec, round_index=0)
    played = {player["checkpoint_id"] for pair in pairs for player in pair["players"]}

    distinct = _distinct_opponents_by_checkpoint(pairs)

    assert rating_spec["placement_min_games"] == 12
    assert rating_spec["placement_min_opponents"] == 20
    assert len(pairs) == 1
    assert len(played) == 2
    assert all(len(opponents) >= 1 for opponents in distinct.values())
    assert {pair["schedule_reason"] for pair in pairs} == {arena.SCHEDULE_REASON_PLACEMENT}


def test_adaptive_v0_revisits_undercovered_roster_with_bounded_placement_wave() -> None:
    refs = [_checkpoint_ref(f"run-{index:02d}", index * 10) for index in range(24)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 228,
            "games_per_pair": 21,
            "seed": 789,
        }
    )
    checkpoints = rating_spec["checkpoints"]
    previous_snapshot = {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "context_hash": arena.rating_context_hash(rating_spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(checkpoints),
        "ratings": [
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "rating": 1500.0 + index,
                "games": 21,
                "distinct_opponents": 1,
                "opponent_ids": [
                    checkpoints[index ^ 1]["checkpoint_id"],
                ],
            }
            for index, checkpoint in enumerate(checkpoints)
        ],
    }

    pairs = arena.build_rating_round_pair_specs(
        rating_spec,
        previous_snapshot=previous_snapshot,
        round_index=1,
    )
    scheduled = _distinct_opponents_by_checkpoint(pairs)

    assert rating_spec["placement_min_games"] == 420
    assert rating_spec["placement_min_opponents"] == 20
    assert len(pairs) == 228
    assert {pair["schedule_reason"] for pair in pairs} == {arena.SCHEDULE_REASON_PLACEMENT}
    opponent_counts = []
    for index, checkpoint in enumerate(checkpoints):
        checkpoint_id = checkpoint["checkpoint_id"]
        total_opponents = set(scheduled[checkpoint_id])
        total_opponents.add(checkpoints[index ^ 1]["checkpoint_id"])
        opponent_counts.append(len(total_opponents))
    assert min(opponent_counts) >= 19


def test_adaptive_v0_caps_placement_opponents_to_small_roster() -> None:
    refs = [_checkpoint_ref(f"run-{index:02d}", index * 10) for index in range(7)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 21,
            "games_per_pair": 21,
            "placement_min_opponents": 20,
            "seed": 790,
        }
    )

    pairs = arena.build_rating_round_pair_specs(rating_spec, round_index=1)
    distinct = _distinct_opponents_by_checkpoint(pairs)

    assert rating_spec["placement_min_games"] == 126
    assert len(pairs) == 21
    assert all(len(opponents) == 6 for opponents in distinct.values())
    assert {pair["schedule_reason"] for pair in pairs} == {arena.SCHEDULE_REASON_PLACEMENT}


def test_adaptive_v0_uses_opponent_ids_over_stale_distinct_scalar() -> None:
    refs = [_checkpoint_ref(f"run-{index:02d}", index * 10) for index in range(8)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 28,
            "games_per_pair": 21,
            "placement_min_opponents": 7,
            "seed": 791,
        }
    )
    checkpoints = rating_spec["checkpoints"]
    previous_snapshot = {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "context_hash": arena.rating_context_hash(rating_spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(checkpoints),
        "ratings": [
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "rating": 1500.0,
                "games": 21,
                "distinct_opponents": 20,
                "opponent_ids": [checkpoints[index ^ 1]["checkpoint_id"]],
            }
            for index, checkpoint in enumerate(checkpoints)
        ],
    }

    pairs = arena.build_rating_round_pair_specs(
        rating_spec,
        previous_snapshot=previous_snapshot,
        round_index=2,
    )
    scheduled = _distinct_opponents_by_checkpoint(pairs)

    assert pairs
    for index, checkpoint in enumerate(checkpoints):
        checkpoint_id = checkpoint["checkpoint_id"]
        total_opponents = set(scheduled[checkpoint_id])
        total_opponents.add(checkpoints[index ^ 1]["checkpoint_id"])
        assert len(total_opponents) == 7


def test_adaptive_v0_ignores_distinct_scalar_without_opponent_ids() -> None:
    refs = [_checkpoint_ref(f"run-{index:02d}", index * 10) for index in range(8)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 28,
            "games_per_pair": 21,
            "placement_min_opponents": 7,
            "seed": 795,
        }
    )
    checkpoints = rating_spec["checkpoints"]
    previous_snapshot = {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "context_hash": arena.rating_context_hash(rating_spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(checkpoints),
        "ratings": [
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "rating": 1500.0,
                "games": 420,
                "distinct_opponents": 7,
            }
            for checkpoint in checkpoints
        ],
    }

    pairs = arena.build_rating_round_pair_specs(
        rating_spec,
        previous_snapshot=previous_snapshot,
        round_index=2,
    )
    scheduled = _distinct_opponents_by_checkpoint(pairs)

    assert pairs
    assert {pair["schedule_reason"] for pair in pairs} == {arena.SCHEDULE_REASON_PLACEMENT}
    for checkpoint in checkpoints:
        assert len(scheduled[checkpoint["checkpoint_id"]]) == 7


def test_adaptive_v0_places_undercovered_checkpoint_against_strong_unseen_opponent_first() -> None:
    refs = [_checkpoint_ref(f"run-{index:02d}", index * 10) for index in range(6)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 1,
            "games_per_pair": 21,
            "placement_min_opponents": 2,
            "seed": 792,
        }
    )
    checkpoints = rating_spec["checkpoints"]
    previous_snapshot = {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "context_hash": arena.rating_context_hash(rating_spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(checkpoints),
        "ratings": [
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "rating": 1000.0 + index,
                "games": 21,
                "distinct_opponents": 1,
                "opponent_ids": [f"prior-opponent-{index}"],
            }
            for index, checkpoint in enumerate(checkpoints)
        ],
    }

    pairs = arena.build_rating_round_pair_specs(
        rating_spec,
        previous_snapshot=previous_snapshot,
        round_index=3,
    )

    assert pairs[0]["players"][1]["checkpoint_id"] == checkpoints[-1]["checkpoint_id"]


def _ranked_established_snapshot(
    rating_spec: dict,
    *,
    opponent_count: int = 20,
    undercovered_index: int | None = None,
) -> dict:
    checkpoints = rating_spec["checkpoints"]
    rows = []
    for index, checkpoint in enumerate(checkpoints):
        opponents = [
            checkpoints[(index + offset) % len(checkpoints)]["checkpoint_id"]
            for offset in range(1, opponent_count + 1)
        ]
        games = 420
        if index == undercovered_index:
            opponents = opponents[:-1]
            games = 399
        rows.append(
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "rank": len(checkpoints) - index,
                "rating": 1000.0 + index,
                "games": games,
                "distinct_opponents": len(opponents),
                "opponent_ids": opponents,
                "rated_battles": max(1, len(opponents)),
                "last_round_delta": float(index % 5),
            }
        )
    return {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "context_hash": arena.rating_context_hash(rating_spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(checkpoints),
        "ratings": rows,
    }


def test_adaptive_v0_placement_coverage_runs_before_top_band_bias() -> None:
    refs = [_checkpoint_ref(f"run-{index:02d}", index * 10) for index in range(40)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 30,
            "games_per_pair": 21,
            "placement_min_games": 420,
            "placement_min_opponents": 20,
            "seed": 793,
        }
    )
    previous_snapshot = _ranked_established_snapshot(
        rating_spec,
        undercovered_index=3,
    )

    pairs = arena.build_rating_round_pair_specs(
        rating_spec,
        previous_snapshot=previous_snapshot,
        round_index=4,
    )

    first_pair_ids = {player["checkpoint_id"] for player in pairs[0]["players"]}
    undercovered_id = rating_spec["checkpoints"][3]["checkpoint_id"]
    top_band_ids = {checkpoint["checkpoint_id"] for checkpoint in rating_spec["checkpoints"][30:40]}

    assert pairs[0]["schedule_reason"] == arena.SCHEDULE_REASON_PLACEMENT
    assert undercovered_id in first_pair_ids
    assert not first_pair_ids <= top_band_ids
    first_non_placement = next(
        (
            index
            for index, pair in enumerate(pairs)
            if pair["schedule_reason"] != arena.SCHEDULE_REASON_PLACEMENT
        ),
        len(pairs),
    )
    assert all(
        pair["schedule_reason"] == arena.SCHEDULE_REASON_PLACEMENT
        for pair in pairs[:first_non_placement]
    )


def test_adaptive_v0_top_band_bias_boosts_leaders_without_starving_other_phases() -> None:
    refs = [_checkpoint_ref(f"run-{index:02d}", index * 10) for index in range(40)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 80,
            "games_per_pair": 21,
            "placement_min_games": 420,
            "placement_min_opponents": 20,
            "seed": 794,
        }
    )
    previous_snapshot = _ranked_established_snapshot(rating_spec)

    pairs = arena.build_rating_round_pair_specs(
        rating_spec,
        previous_snapshot=previous_snapshot,
        round_index=5,
    )
    appearances: dict[str, int] = {}
    for pair in pairs:
        for player in pair["players"]:
            checkpoint_id = player["checkpoint_id"]
            appearances[checkpoint_id] = appearances.get(checkpoint_id, 0) + 1
    reason_counts: dict[str, int] = {}
    for pair in pairs:
        reason = pair["schedule_reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    top10_ids = [checkpoint["checkpoint_id"] for checkpoint in rating_spec["checkpoints"][30:40]]
    top20_ids = [checkpoint["checkpoint_id"] for checkpoint in rating_spec["checkpoints"][20:40]]
    lower_half_ids = [
        checkpoint["checkpoint_id"] for checkpoint in rating_spec["checkpoints"][0:20]
    ]
    top10_average = sum(appearances.get(item, 0) for item in top10_ids) / len(top10_ids)
    top20_average = sum(appearances.get(item, 0) for item in top20_ids) / len(top20_ids)
    lower_half_average = sum(appearances.get(item, 0) for item in lower_half_ids) / len(
        lower_half_ids
    )

    assert top10_average > lower_half_average
    assert top20_average > lower_half_average
    assert all(appearances.get(item, 0) > 0 for item in lower_half_ids)
    assert reason_counts[arena.SCHEDULE_REASON_UNCERTAIN] > 0
    assert reason_counts[arena.SCHEDULE_REASON_RANDOM_BRIDGE] > 0


def test_schedule_metadata_survives_pair_summary() -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                _checkpoint_ref("run-a", 0),
                _checkpoint_ref("run-b", 10),
                _checkpoint_ref("run-c", 20),
            ],
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 1,
            "games_per_pair": 3,
        }
    )
    pair = arena.build_rating_round_pair_specs(rating_spec)[0]

    summary = arena.summarize_pair_results(
        pair,
        [
            _fake_game(pair, 0, "seat_0_win"),
            _fake_game(pair, 1, "draw"),
            _fake_game(pair, 2, "seat_1_win"),
        ],
    )

    assert summary["pair_key"] == pair["pair_key"]
    assert summary["schedule_reason"] == pair["schedule_reason"]
    assert summary["schedule"]["reason"] == pair["schedule_reason"]


def test_pair_history_accumulates_by_canonical_pair_key_across_seat_order() -> None:
    checkpoints = [
        {"checkpoint_ref": _checkpoint_ref("run-a", 0), "checkpoint_id": "ckpt-a"},
        {"checkpoint_ref": _checkpoint_ref("run-b", 10), "checkpoint_id": "ckpt-b"},
    ]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": checkpoints,
            "games_per_pair": 3,
        }
    )
    pair_key = arena.rating_pair_key("ckpt-a", "ckpt-b")
    pair_ab = arena.normalize_pair_spec(
        {
            "tournament_id": "arena-a",
            "battle_id": "battle-ab",
            "players": [
                {"seat": 0, **rating_spec["checkpoints"][0]},
                {"seat": 1, **rating_spec["checkpoints"][1]},
            ],
            "games_per_pair": 3,
            "pair_key": pair_key,
        }
    )
    pair_ba = arena.normalize_pair_spec(
        {
            "tournament_id": "arena-a",
            "battle_id": "battle-ba",
            "players": [
                {"seat": 0, **rating_spec["checkpoints"][1]},
                {"seat": 1, **rating_spec["checkpoints"][0]},
            ],
            "games_per_pair": 3,
            "pair_key": pair_key,
        }
    )
    summary_ab = arena.summarize_pair_results(
        pair_ab,
        [
            _fake_game(pair_ab, 0, "seat_0_win"),
            _fake_game(pair_ab, 1, "seat_0_win"),
            _fake_game(pair_ab, 2, "draw"),
        ],
    )
    summary_ba = arena.summarize_pair_results(
        pair_ba,
        [
            _fake_game(pair_ba, 0, "seat_0_win"),
            _fake_game(pair_ba, 1, "seat_1_win"),
            _fake_game(pair_ba, 2, "draw"),
        ],
    )

    first = arena.pair_history_from_pair_results(
        [summary_ab],
        rating_spec=rating_spec,
        round_index=0,
    )
    second = arena.pair_history_from_pair_results(
        [summary_ba],
        previous_pair_history=first,
        rating_spec=rating_spec,
        round_index=1,
    )

    assert len(second["rows"]) == 1
    row = second["rows"][0]
    assert row["pair_key"] == pair_key
    assert row["battle_count"] == 2
    assert row["game_count"] == 6
    assert row["draw_count"] == 2
    assert row["last_battle_id"] == "battle-ba"
    assert row["wins_by_checkpoint"]["ckpt-a"] == 3
    assert row["wins_by_checkpoint"]["ckpt-b"] == 1


def test_pair_history_rejects_pool_hash_mismatch() -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                _checkpoint_ref("run-a", 0),
                _checkpoint_ref("run-b", 10),
            ],
            "games_per_pair": 3,
        }
    )

    with pytest.raises(ValueError, match="pool_hash"):
        arena.pair_history_from_pair_results(
            [],
            previous_pair_history={"pool_hash": "not-the-current-pool"},
            rating_spec=rating_spec,
        )


def test_rating_context_hash_changes_for_evaluator_not_roster() -> None:
    base = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                _checkpoint_ref("run-a", 0),
                _checkpoint_ref("run-b", 10),
            ],
            "games_per_pair": 3,
            "max_steps": 64,
        }
    )
    expanded_roster = arena.normalize_rating_spec(
        {
            **base,
            "checkpoints": [
                *base["checkpoints"],
                {"checkpoint_ref": _checkpoint_ref("run-c", 20), "checkpoint_id": "ckpt-c"},
            ],
        }
    )
    changed_context = arena.normalize_rating_spec({**base, "max_steps": 128})
    changed_policy_batch = arena.normalize_rating_spec(
        {**base, "policy_batch_size": base["policy_batch_size"] + 16}
    )
    with pytest.raises(ValueError, match="policy surface must be"):
        arena.normalize_rating_spec({**base, "policy_bonus_render_mode": "browser_sprites"})
    with pytest.raises(ValueError, match="policy observation backend"):
        arena.normalize_rating_spec({**base, "policy_observation_backend": "jax_gpu"})
    non_context_change = arena.normalize_rating_spec(
        {
            **base,
            "games_per_pair": 5,
            "games_per_shard": 5,
            "save_gif": True,
            "gif_sample_games_per_pair": 3,
        }
    )

    assert arena.rating_context_hash(base) == arena.rating_context_hash(expanded_roster)
    assert arena.rating_context_hash(base) == arena.rating_context_hash(non_context_change)
    assert arena.rating_pool_hash(base["checkpoints"]) != arena.rating_pool_hash(
        expanded_roster["checkpoints"]
    )
    assert arena.rating_pool_hash(base["checkpoints"]) != arena.rating_pool_hash(
        [
            {
                **base["checkpoints"][0],
                "policy_observation_backend": "jax_gpu",
            },
            base["checkpoints"][1],
        ]
    )
    assert arena.rating_context_hash(base) != arena.rating_context_hash(changed_context)
    assert arena.rating_context_hash(base) != arena.rating_context_hash(changed_policy_batch)


def test_pair_history_with_context_hash_allows_roster_expansion() -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                {"checkpoint_ref": _checkpoint_ref("run-a", 0), "checkpoint_id": "ckpt-a"},
                {"checkpoint_ref": _checkpoint_ref("run-b", 10), "checkpoint_id": "ckpt-b"},
            ],
            "games_per_pair": 3,
        }
    )
    pair = arena.build_rating_round_pair_specs(rating_spec)[0]
    summary = arena.summarize_pair_results(
        pair,
        [
            _fake_game(pair, 0, "seat_0_win"),
            _fake_game(pair, 1, "draw"),
            _fake_game(pair, 2, "seat_1_win"),
        ],
    )
    first = arena.pair_history_from_pair_results(
        [summary],
        rating_spec=rating_spec,
        round_index=0,
    )
    expanded_spec = arena.normalize_rating_spec(
        {
            **rating_spec,
            "checkpoints": [
                *rating_spec["checkpoints"],
                {"checkpoint_ref": _checkpoint_ref("run-c", 20), "checkpoint_id": "ckpt-c"},
            ],
        }
    )
    second = arena.pair_history_from_pair_results(
        [],
        previous_pair_history=first,
        rating_spec=expanded_spec,
        round_index=1,
    )

    assert second["pool_hash"] != first["pool_hash"]
    assert second["roster_hash"] == second["pool_hash"]
    assert second["context_hash"] == first["context_hash"]
    assert second["rows"] == first["rows"]


def test_pair_history_rejects_context_hash_mismatch() -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                {"checkpoint_ref": _checkpoint_ref("run-a", 0), "checkpoint_id": "ckpt-a"},
                {"checkpoint_ref": _checkpoint_ref("run-b", 10), "checkpoint_id": "ckpt-b"},
            ],
            "games_per_pair": 3,
            "max_steps": 64,
        }
    )
    history = arena.pair_history_from_pair_results([], rating_spec=rating_spec)
    changed_context = arena.normalize_rating_spec({**rating_spec, "max_steps": 128})

    with pytest.raises(ValueError, match="context_hash"):
        arena.pair_history_from_pair_results(
            [],
            previous_pair_history=history,
            rating_spec=changed_context,
        )


def test_pair_history_rejects_checkpoint_id_replacement() -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                {"checkpoint_ref": _checkpoint_ref("run-a", 0), "checkpoint_id": "ckpt-a"},
                {"checkpoint_ref": _checkpoint_ref("run-b", 10), "checkpoint_id": "ckpt-b"},
            ],
            "games_per_pair": 3,
        }
    )
    history = arena.pair_history_from_pair_results([], rating_spec=rating_spec)
    replaced = arena.normalize_rating_spec(
        {
            **rating_spec,
            "checkpoints": [
                {
                    "checkpoint_ref": _checkpoint_ref("run-a-replaced", 99),
                    "checkpoint_id": "ckpt-a",
                },
                rating_spec["checkpoints"][1],
            ],
        }
    )

    assert arena.rating_context_hash(replaced) == arena.rating_context_hash(rating_spec)
    with pytest.raises(ValueError, match="checkpoint_roster"):
        arena.pair_history_from_pair_results(
            [],
            previous_pair_history=history,
            rating_spec=replaced,
        )


def test_adaptive_scheduler_state_rejects_context_hash_mismatch() -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                _checkpoint_ref("run-a", 0),
                _checkpoint_ref("run-b", 10),
                _checkpoint_ref("run-c", 20),
            ],
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 1,
            "games_per_pair": 3,
            "max_steps": 64,
        }
    )
    changed_context = arena.normalize_rating_spec({**rating_spec, "max_steps": 128})
    scheduler_state = {
        "pool_hash": arena.rating_pool_hash(rating_spec["checkpoints"]),
        "context_hash": arena.rating_context_hash(changed_context),
    }

    with pytest.raises(ValueError, match="scheduler_state context_hash"):
        arena.build_rating_round_pair_specs(
            rating_spec,
            scheduler_state=scheduler_state,
        )


def test_adaptive_scheduler_state_rejects_checkpoint_id_replacement() -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                {"checkpoint_ref": _checkpoint_ref("run-a", 0), "checkpoint_id": "ckpt-a"},
                {"checkpoint_ref": _checkpoint_ref("run-b", 10), "checkpoint_id": "ckpt-b"},
                {"checkpoint_ref": _checkpoint_ref("run-c", 20), "checkpoint_id": "ckpt-c"},
            ],
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 1,
            "games_per_pair": 3,
        }
    )
    replaced = arena.normalize_rating_spec(
        {
            **rating_spec,
            "checkpoints": [
                {
                    "checkpoint_ref": _checkpoint_ref("run-a-replaced", 99),
                    "checkpoint_id": "ckpt-a",
                },
                *rating_spec["checkpoints"][1:],
            ],
        }
    )
    scheduler_state = {
        "pool_hash": arena.rating_pool_hash(rating_spec["checkpoints"]),
        "context_hash": arena.rating_context_hash(rating_spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(rating_spec["checkpoints"]),
    }

    with pytest.raises(ValueError, match="scheduler_state checkpoint_roster"):
        arena.build_rating_round_pair_specs(
            replaced,
            scheduler_state=scheduler_state,
        )


def test_rating_snapshot_rejects_previous_context_hash_mismatch() -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                _checkpoint_ref("run-a", 0),
                _checkpoint_ref("run-b", 10),
            ],
            "games_per_pair": 3,
            "max_steps": 64,
        }
    )
    changed_context = arena.normalize_rating_spec({**rating_spec, "max_steps": 128})
    previous_snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=[],
        rating_spec=changed_context,
        round_index=0,
    )

    with pytest.raises(ValueError, match="previous snapshot context_hash"):
        arena.rating_snapshot_from_pair_results(
            pair_results=[],
            rating_spec=rating_spec,
            previous_snapshot=previous_snapshot,
            round_index=1,
        )


def test_rating_snapshot_rejects_checkpoint_id_replacement() -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                {"checkpoint_ref": _checkpoint_ref("run-a", 0), "checkpoint_id": "ckpt-a"},
                {"checkpoint_ref": _checkpoint_ref("run-b", 10), "checkpoint_id": "ckpt-b"},
            ],
            "games_per_pair": 3,
        }
    )
    previous_snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=[],
        rating_spec=rating_spec,
        round_index=0,
    )
    replaced = arena.normalize_rating_spec(
        {
            **rating_spec,
            "checkpoints": [
                {
                    "checkpoint_ref": _checkpoint_ref("run-a-replaced", 99),
                    "checkpoint_id": "ckpt-a",
                },
                rating_spec["checkpoints"][1],
            ],
        }
    )

    with pytest.raises(ValueError, match="previous snapshot checkpoint_roster"):
        arena.rating_snapshot_from_pair_results(
            pair_results=[],
            rating_spec=replaced,
            previous_snapshot=previous_snapshot,
            round_index=1,
        )


def test_rating_loop_start_state_continues_from_latest_with_new_checkpoint(
    tmp_path,
) -> None:
    base_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                _checkpoint_ref("run-a", 0),
                _checkpoint_ref("run-b", 10),
            ],
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 1,
            "games_per_pair": 3,
            "max_steps": 64,
        }
    )
    pair = arena.build_rating_round_pair_specs(base_spec, round_index=2)[0]
    previous_snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=[
            arena.summarize_pair_results(
                pair,
                [
                    _fake_game(pair, 0, "seat_0_win"),
                    _fake_game(pair, 1, "seat_0_win"),
                    _fake_game(pair, 2, "draw"),
                ],
            )
        ],
        rating_spec=base_spec,
        round_index=2,
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_latest_ref("arena-a", "elo-test"),
        modal_arena._slim_rating_snapshot(previous_snapshot),
    )
    expanded_spec = arena.normalize_rating_spec(
        {
            **base_spec,
            "continue_from_latest": True,
            "checkpoints": [
                *base_spec["checkpoints"],
                _checkpoint_ref("run-c", 20),
            ],
        }
    )

    state = modal_arena._rating_loop_start_state(tmp_path, expanded_spec)
    pairs = arena.build_rating_round_pair_specs(
        expanded_spec,
        previous_snapshot=state["previous_snapshot"],
        pair_history=state["previous_pair_history"],
        scheduler_state=state["scheduler_state"],
        round_index=state["start_round_index"],
    )
    new_id = expanded_spec["checkpoints"][2]["checkpoint_id"]
    played = {player["checkpoint_id"] for pair_spec in pairs for player in pair_spec["players"]}

    assert state["continued_from_latest"] is True
    assert state["start_round_index"] == 3
    assert state["previous_snapshot"]["round_id"] == "round-000002"
    assert new_id in played
    assert pairs[0]["scheduled_round_index"] == 3
    assert pairs[0]["schedule_reason"] == arena.SCHEDULE_REASON_PLACEMENT


def test_rating_loop_start_state_skips_repaired_orphan_round(
    tmp_path,
) -> None:
    base_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                _checkpoint_ref("run-a", 0),
                _checkpoint_ref("run-b", 10),
            ],
            "games_per_pair": 3,
            "max_steps": 64,
        }
    )
    previous_snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=[],
        rating_spec=base_spec,
        round_index=2,
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_latest_ref("arena-a", "elo-test"),
        modal_arena._slim_rating_snapshot(previous_snapshot),
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref("arena-a", "elo-test", arena.rating_round_id(3)),
        {"round_index": 3, "game_count": 999},
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_progress_ref("arena-a", "elo-test", arena.rating_round_id(3)),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "round_id": arena.rating_round_id(3),
            "round_index": 3,
            "status": "skipped",
            "phase": "stale_orphan_round_skipped",
        },
    )
    expanded_spec = arena.normalize_rating_spec(
        {
            **base_spec,
            "continue_from_latest": True,
            "checkpoints": [
                *base_spec["checkpoints"],
                _checkpoint_ref("run-c", 20),
            ],
        }
    )

    assert modal_arena._rating_writer_has_finished(
        tmp_path,
        tournament_id="arena-a",
        rating_run_id="elo-test",
    )
    assert (
        modal_arena._oldest_unrated_rating_round(
            tmp_path,
            tournament_id="arena-a",
            rating_run_id="elo-test",
        )
        is None
    )
    state = modal_arena._rating_loop_start_state(tmp_path, expanded_spec)

    assert state["continued_from_latest"] is True
    assert state["start_round_index"] == 4
    assert state["previous_snapshot"]["round_id"] == "round-000002"


def test_rating_writer_finished_trusts_completed_latest_over_stale_root_progress(
    tmp_path,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    round_id = arena.rating_round_id(9)
    modal_arena.runs.write_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id),
        {
            "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 9,
            "ratings_ref": "ratings.json",
            "ended_at": "2026-05-17T04:08:00Z",
            "global_outputs_published": True,
        },
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_progress_ref(tournament_id, rating_run_id),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 9,
            "status": "running",
            "phase": "games_running",
            "completed_game_count": 31,
            "game_count": 126,
        },
    )

    assert modal_arena._rating_writer_has_finished(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )


def test_rating_progress_root_pointer_is_monotonic(tmp_path) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    newer_progress = {
        "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "round_id": arena.rating_round_id(11),
        "round_index": 11,
        "status": "running",
        "phase": "games_running",
    }
    older_progress = {
        "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "round_id": arena.rating_round_id(10),
        "round_index": 10,
        "status": "complete",
        "phase": "reduced",
    }
    modal_arena.runs.write_json(
        tmp_path / arena.rating_progress_ref(tournament_id, rating_run_id),
        newer_progress,
    )

    result = modal_arena._write_rating_progress(tmp_path, older_progress)

    root_progress = modal_arena._read_json(
        tmp_path / arena.rating_progress_ref(tournament_id, rating_run_id)
    )
    round_progress = modal_arena._read_json(
        tmp_path
        / arena.rating_round_progress_ref(
            tournament_id,
            rating_run_id,
            arena.rating_round_id(10),
        )
    )
    assert result["root_progress_write_skipped"] is True
    assert root_progress["round_id"] == arena.rating_round_id(11)
    assert round_progress["round_id"] == arena.rating_round_id(10)


def test_rating_round_outputs_do_not_publish_older_round_over_newer_latest(
    tmp_path,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
        }
    )
    newer_snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=[],
        rating_spec=rating_spec,
        round_index=11,
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id),
        modal_arena._slim_rating_snapshot(newer_snapshot),
    )
    pairs = arena.build_rating_round_pair_specs(rating_spec)
    pair = pairs[0]
    games = [_fake_game(pair, index, "seat_0_win") for index in range(3)]
    pair_result = arena.summarize_pair_results(pair, games)
    pair_result["summary_ref"] = arena.battle_summary_ref(
        tournament_id,
        pair["battle_id"],
    ).as_posix()

    result = modal_arena._write_rating_round_outputs(
        tmp_path,
        spec=rating_spec,
        round_id=arena.rating_round_id(10),
        round_index=10,
        pair_results=[pair_result],
        pair_specs=pairs,
        game_count=3,
        started_at="2026-05-16T00:00:00Z",
        previous_snapshot=None,
    )

    latest = modal_arena._read_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id)
    )
    round_ratings = modal_arena._read_json(
        tmp_path
        / arena.rating_round_ratings_ref(
            tournament_id,
            rating_run_id,
            arena.rating_round_id(10),
        )
    )
    assert latest["round_index"] == 11
    assert round_ratings["round_index"] == 10
    assert result["global_outputs_published"] is False
    assert result["snapshot"]["latest_write_skipped"] is True
    lineage_rows = [
        json.loads(line)
        for line in (
            tmp_path
            / arena.rating_root_ref(tournament_id, rating_run_id)
            / "feedback_loop"
            / "lineage_events.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert lineage_rows[-1]["stage"] == "rating_latest_written"
    assert lineage_rows[-1]["status"] == "skipped"
    assert lineage_rows[-1]["reason"] == "newer_round_already_latest"
    assert lineage_rows[-1]["round_index"] == 10
    assert lineage_rows[-1]["latest_round_index_before_write"] == 11


def test_rating_latest_publish_blocks_lower_checkpoint_count(tmp_path) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    modal_arena.runs.write_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id),
        {
            "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": arena.rating_round_id(10),
            "round_index": 10,
            "checkpoint_count": 681,
            "ratings": [{"checkpoint_id": f"ckpt-{index}"} for index in range(681)],
        },
    )

    decision = modal_arena._publish_rating_latest_snapshot_if_current(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        snapshot={
            "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": arena.rating_round_id(11),
            "round_index": 11,
            "checkpoint_count": 588,
            "ratings": [{"checkpoint_id": f"ckpt-{index}"} for index in range(588)],
        },
    )

    latest = modal_arena._read_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id)
    )
    assert decision["publish"] is False
    assert decision["reason"] == "higher_checkpoint_count_already_latest"
    assert decision["latest_checkpoint_count"] == 681
    assert decision["snapshot_checkpoint_count"] == 588
    assert latest["round_index"] == 10
    assert latest["checkpoint_count"] == 681


def test_rating_latest_publish_blocks_snapshot_marked_non_global(tmp_path) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    modal_arena.runs.write_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id),
        {
            "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": arena.rating_round_id(11),
            "round_index": 11,
            "checkpoint_count": 681,
        },
    )

    decision = modal_arena._publish_rating_latest_snapshot_if_current(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        snapshot={
            "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": arena.rating_round_id(10),
            "round_index": 10,
            "checkpoint_count": 681,
            "global_outputs_published": False,
            "latest_write_skipped_reason": "newer_round_already_latest",
        },
    )

    latest = modal_arena._read_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id)
    )
    assert decision["publish"] is False
    assert decision["reason"] == "newer_round_already_latest"
    assert latest["round_index"] == 11


def test_rating_loop_start_state_rejects_changed_context(tmp_path) -> None:
    base_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                _checkpoint_ref("run-a", 0),
                _checkpoint_ref("run-b", 10),
            ],
            "games_per_pair": 3,
            "max_steps": 64,
        }
    )
    previous_snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=[],
        rating_spec=base_spec,
        round_index=0,
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_latest_ref("arena-a", "elo-test"),
        modal_arena._slim_rating_snapshot(previous_snapshot),
    )
    changed_spec = arena.normalize_rating_spec(
        {
            **base_spec,
            "continue_from_latest": True,
            "max_steps": 128,
        }
    )

    with pytest.raises(ValueError, match="latest snapshot context_hash"):
        modal_arena._rating_loop_start_state(tmp_path, changed_spec)


def test_rating_spec_with_latest_roster_restores_checkpoint_ids(tmp_path) -> None:
    base_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                {
                    "checkpoint_id": "ckpt-left",
                    "label": "left",
                    "checkpoint_ref": _checkpoint_ref("run-a", 10),
                },
                {
                    "checkpoint_id": "ckpt-right",
                    "label": "right",
                    "checkpoint_ref": _checkpoint_ref("run-b", 20),
                },
            ],
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 1,
            "games_per_pair": 21,
        }
    )
    latest = arena.rating_snapshot_from_pair_results(
        pair_results=[],
        rating_spec=base_spec,
        round_index=0,
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_latest_ref("arena-a", "elo-test"),
        latest,
    )

    restored = modal_arena._rating_spec_with_latest_roster(
        tmp_path,
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "continue_from_latest": True,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 40,
            "games_per_pair": 21,
            "placement_min_opponents": 20,
        },
    )

    assert [row["checkpoint_id"] for row in restored["checkpoints"]] == [
        "ckpt-left",
        "ckpt-right",
    ]
    assert arena.rating_pool_hash(restored["checkpoints"]) == arena.rating_pool_hash(
        base_spec["checkpoints"]
    )


def test_named_artifact_refs_include_pair_spec_and_provisional_latest() -> None:
    refs = [
        arena.battle_pair_spec_ref("arena-a", "battle-a"),
        arena.rating_provisional_latest_ref("arena-a", "elo-test"),
        arena.rating_run_results_ref("arena-a", "elo-test"),
        arena.tournament_intake_manifest_ref("arena-a", "elo-test"),
        arena.tournament_intake_latest_tick_ref("arena-a", "elo-test"),
    ]

    for ref in refs:
        assert arena.validate_tournament_artifact_ref(ref) == ref


def test_intake_manifest_builds_seen_refs_and_queue_partition() -> None:
    discovery = {
        "checkpoint_refs": [
            _checkpoint_ref("run-a", 10),
            _checkpoint_ref("run-b", 20),
        ],
        "rows": [],
        "found_count": 2,
    }

    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={
            "run_id_prefix": "run-",
            "max_runs": 20,
            "checkpoint_selection": arena.CHECKPOINT_SELECTION_LATEST,
        },
        rating_defaults={"games_per_pair": 21, "save_gif": True},
        discovery=discovery,
        existing={"seen_checkpoint_refs": [_checkpoint_ref("run-old", 0)]},
    )

    assert manifest["manifest_key"] == "manifest:arena-a:elo-test"
    assert manifest["queue_partition"].startswith("q:arena-a:elo-test:")
    assert len(manifest["queue_partition"]) <= 64
    assert manifest["checkpoint_count"] == 2
    assert manifest["seen_checkpoint_count"] == 3
    assert manifest["queued_checkpoint_count"] == 0
    assert _checkpoint_ref("run-old", 0) in manifest["seen_checkpoint_refs"]


def test_intake_manifest_write_merge_does_not_shrink_newer_checkpoint_pool() -> None:
    stale_refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    fresh_refs = [*stale_refs, _checkpoint_ref("run-c", 30)]
    current = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": fresh_refs},
        rating_defaults={
            "pairs_per_round": 300,
            "placement_min_games": None,
        },
        discovery={"checkpoint_refs": fresh_refs},
    )
    stale_candidate = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": stale_refs},
        rating_defaults={
            "pairs_per_round": 300,
            "placement_min_games": 420,
        },
        discovery={"checkpoint_refs": stale_refs},
        existing=current,
    )

    merged = modal_arena._intake_manifest_with_merged_pool(stale_candidate, current)

    assert merged["checkpoint_refs"] == fresh_refs
    assert merged["checkpoint_count"] == 3
    assert merged["seen_checkpoint_count"] == 3
    assert merged["rating_defaults"]["placement_min_games"] is None
    assert merged["scan_spec"]["checkpoint_refs"] == fresh_refs
    assert [row["checkpoint_ref"] for row in merged["discovery"]["rows"]] == fresh_refs


def test_intake_manifest_write_merge_accepts_newer_wider_checkpoint_pool() -> None:
    stale_refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    fresh_refs = [*stale_refs, _checkpoint_ref("run-c", 30)]
    current = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": stale_refs},
        rating_defaults={
            "pairs_per_round": 300,
            "placement_min_games": 420,
        },
        discovery={"checkpoint_refs": stale_refs},
    )
    fresh_candidate = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": fresh_refs},
        rating_defaults={
            "pairs_per_round": 300,
            "placement_min_games": None,
        },
        discovery={"checkpoint_refs": fresh_refs},
        existing=current,
    )

    merged = modal_arena._intake_manifest_with_merged_pool(fresh_candidate, current)

    assert merged["checkpoint_refs"] == fresh_refs
    assert merged["checkpoint_count"] == 3
    assert merged["rating_defaults"]["placement_min_games"] is None


def test_intake_manifest_tracks_queued_refs_separately_from_seen_refs() -> None:
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": refs},
        rating_defaults={},
        discovery={"checkpoint_refs": refs},
    )

    assert manifest["seen_checkpoint_count"] == 2
    assert manifest["queued_checkpoint_count"] == 0

    queued = modal_arena._mark_intake_manifest_queued(manifest, [refs[0]])

    assert queued["seen_checkpoint_count"] == 2
    assert queued["queued_checkpoint_count"] == 1
    assert queued["queued_checkpoint_refs"] == [refs[0]]


def test_intake_discovery_accepts_explicit_checkpoint_refs() -> None:
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]

    discovery = modal_arena._discover_checkpoint_refs_from_scan_spec(
        {"checkpoint_refs": refs},
        mount=Path("/unused"),
    )

    assert discovery["selection"] == "explicit_refs"
    assert discovery["found_count"] == 2
    assert discovery["checkpoint_refs"] == refs
    assert [row["iteration"] for row in discovery["rows"]] == [10, 20]


def test_explicit_submit_scan_spec_is_not_a_live_run_watch() -> None:
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]

    scan_spec = modal_arena._explicit_checkpoint_refs_scan_spec(refs)

    assert scan_spec["checkpoint_refs"] == sorted(refs)
    assert scan_spec["run_ids"] == ""
    assert scan_spec["run_id_prefix"] == ""
    assert modal_arena._intake_scan_spec_is_live_watch(scan_spec) is False


def test_exact_refs_added_to_live_watch_preserve_run_scan() -> None:
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    live_scan = {
        "run_ids": "run-a,run-b",
        "run_id_prefix": "",
        "max_runs": 0,
        "checkpoint_iteration": None,
        "checkpoint_selection": arena.CHECKPOINT_SELECTION_ALL,
    }

    scan_spec = modal_arena._scan_spec_with_checkpoint_refs(refs, live_scan)

    assert scan_spec["checkpoint_refs"] == sorted(refs)
    assert scan_spec["run_ids"] == "run-a,run-b"
    assert scan_spec["checkpoint_selection"] == arena.CHECKPOINT_SELECTION_ALL
    assert modal_arena._intake_scan_spec_is_live_watch(scan_spec) is True


def test_explicit_refs_plus_prefix_discovery_keeps_watching_new_checkpoints(
    tmp_path,
) -> None:
    seed_ref = _checkpoint_ref("seed-run", 10)
    live_ref = _train_exp_checkpoint_ref("cz26c-r001", "try-cz26c-r001", 100)
    live_path = tmp_path / live_ref
    live_path.parent.mkdir(parents=True)
    live_path.write_bytes(b"checkpoint")

    discovery = modal_arena._discover_checkpoint_refs_from_scan_spec(
        {
            "checkpoint_refs": [seed_ref],
            "run_ids": "",
            "run_id_prefix": "cz26c-",
            "max_runs": 0,
            "checkpoint_iteration": None,
            "checkpoint_selection": arena.CHECKPOINT_SELECTION_ALL,
        },
        mount=tmp_path,
    )

    assert discovery["selection"] == "explicit_refs_plus_run_watch"
    assert discovery["checkpoint_refs"] == [seed_ref, live_ref]
    assert modal_arena._intake_scan_spec_is_live_watch(discovery) is True


def test_manifest_pool_merge_preserves_existing_live_watch_scan_spec() -> None:
    old_ref = _checkpoint_ref("run-a", 10)
    new_ref = _checkpoint_ref("run-b", 20)
    current = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={
            "run_ids": "run-a",
            "run_id_prefix": "",
            "max_runs": 0,
            "checkpoint_iteration": None,
            "checkpoint_selection": arena.CHECKPOINT_SELECTION_ALL,
        },
        rating_defaults={"continue_from_latest": True},
        discovery={"checkpoint_refs": [old_ref]},
    )
    candidate = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec=modal_arena._explicit_checkpoint_refs_scan_spec([new_ref]),
        rating_defaults={},
        discovery={"checkpoint_refs": [new_ref]},
    )

    merged = modal_arena._intake_manifest_with_merged_pool(candidate, current)

    assert merged["checkpoint_refs"] == sorted([old_ref, new_ref])
    assert merged["scan_spec"]["checkpoint_refs"] == sorted([old_ref, new_ref])
    assert merged["scan_spec"]["run_ids"] == "run-a"
    assert merged["scan_spec"]["checkpoint_selection"] == arena.CHECKPOINT_SELECTION_ALL
    assert modal_arena._intake_scan_spec_is_live_watch(merged["scan_spec"]) is True


def test_manifest_pool_merge_can_upgrade_static_refs_to_live_watch_scan_spec() -> None:
    seed_ref = _checkpoint_ref("seed-run", 10)
    current = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec=modal_arena._explicit_checkpoint_refs_scan_spec([seed_ref]),
        rating_defaults={"continue_from_latest": True},
        discovery={"checkpoint_refs": [seed_ref]},
    )
    candidate = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={
            "checkpoint_refs": [seed_ref],
            "run_ids": "",
            "run_id_prefix": "cz26c-",
            "max_runs": 0,
            "checkpoint_iteration": None,
            "checkpoint_selection": arena.CHECKPOINT_SELECTION_LATEST,
        },
        rating_defaults={"continue_from_latest": True},
        discovery={"checkpoint_refs": [seed_ref]},
    )

    merged = modal_arena._intake_manifest_with_merged_pool(candidate, current)

    assert merged["checkpoint_refs"] == [seed_ref]
    assert merged["scan_spec"]["checkpoint_refs"] == [seed_ref]
    assert merged["scan_spec"]["run_id_prefix"] == "cz26c-"
    assert merged["scan_spec"]["checkpoint_selection"] == arena.CHECKPOINT_SELECTION_LATEST
    assert modal_arena._intake_scan_spec_is_live_watch(merged["scan_spec"]) is True


def test_rating_run_existing_output_guard_uses_named_refs(tmp_path: Path) -> None:
    assert not modal_arena._rating_run_has_existing_output(
        tmp_path,
        tournament_id="arena-a",
        rating_run_id="elo-test",
    )

    progress_path = tmp_path / arena.rating_progress_ref("arena-a", "elo-test")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("{}", encoding="utf-8")

    assert modal_arena._rating_run_has_existing_output(
        tmp_path,
        tournament_id="arena-a",
        rating_run_id="elo-test",
    )


def test_intake_rating_spec_preserves_five_gif_samples_per_battle() -> None:
    manifest = {
        "tournament_id": "arena-a",
        "rating_run_id": "elo-test",
        "checkpoint_refs": [
            _checkpoint_ref("run-a", 10),
            _checkpoint_ref("run-b", 20),
            _checkpoint_ref("run-c", 30),
        ],
        "rating_defaults": {
            "round_count": 1,
            "pair_selection": arena.RATING_PAIR_SELECTION_ADAPTIVE_V0,
            "pairs_per_round": 2,
            "games_per_pair": 21,
            "games_per_shard": 21,
            "placement_min_games": 0,
            "placement_min_opponents": 0,
            "save_gif": True,
            "gif_sample_games_per_pair": 5,
            "gif_sample_strategy": "evenly_spaced",
            "max_steps": 64,
            "num_simulations": 1,
        },
    }

    spec = modal_arena._intake_rating_spec_from_manifest(manifest)
    pairs = arena.build_rating_round_pair_specs(spec, round_index=0)

    assert spec["save_gif"] is True
    assert spec["gif_sample_games_per_pair"] == 5
    assert spec["games_per_pair"] == 21
    assert spec["games_per_shard"] == 21
    assert len(pairs) == 2
    assert all(len(arena.gif_sample_indices_for_pair(pair)) == 5 for pair in pairs)


def test_intake_drain_does_not_consume_events_when_rating_exists(
    tmp_path,
    monkeypatch,
) -> None:
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": refs},
        rating_defaults={},
        discovery={"checkpoint_refs": refs},
    )
    events = [{"checkpoint_ref": ref} for ref in refs]

    class FakeState:
        def get(self, key, default=None):
            if key == manifest["manifest_key"]:
                return manifest
            return default

        def put(self, *_args, **_kwargs):
            raise AssertionError("drain should not claim an existing rating run")

    class FakeQueue:
        def __init__(self):
            self.get_many_calls = 0

        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(events)

        def get_many(self, *_args, **_kwargs):
            self.get_many_calls += 1
            return events

    fake_queue = FakeQueue()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", fake_queue)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(
        modal_arena,
        "_rating_run_has_existing_output",
        lambda *_args, **_kwargs: True,
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "max_events": 10,
            "spawn_rating": True,
        }
    )

    assert result["event_count"] == 0
    assert result["queue_len_before"] == 2
    assert result["spawn_skipped_reason"] == "rating_run_already_exists"
    assert fake_queue.get_many_calls == 0


def test_intake_drain_claims_before_consuming_events(tmp_path, monkeypatch) -> None:
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": refs},
        rating_defaults={},
        discovery={"checkpoint_refs": refs},
    )
    events = [{"checkpoint_ref": ref} for ref in refs]
    call_order: list[str] = []

    class FakeState:
        def __init__(self):
            self.values = {manifest["manifest_key"]: manifest}

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value, skip_if_exists=False):
            call_order.append("claim")
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    class FakeQueue:
        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(events)

        def get_many(self, *_args, **_kwargs):
            call_order.append("consume")
            return events

    class FakeCall:
        object_id = "fc-test"

    class FakeRatingLoop:
        def spawn(self, _spec):
            call_order.append("spawn")
            return FakeCall()

    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", FakeQueue())
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", FakeRatingLoop())
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "_commit_volume", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        modal_arena,
        "_rating_run_has_existing_output",
        lambda *_args, **_kwargs: False,
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "max_events": 10,
            "spawn_rating": True,
            "pairs_per_round": 1,
            "games_per_pair": 3,
        }
    )

    assert result["event_count"] == 2
    assert result["rating_call_id"] == "fc-test"
    assert call_order[0] == "claim"
    assert "consume" in call_order
    assert call_order[-1] == "spawn"
    lineage_rows = [
        json.loads(line)
        for line in (
            tmp_path
            / arena.rating_root_ref("arena-a", "elo-test")
            / "feedback_loop"
            / "lineage_events.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert lineage_rows[-1]["stage"] == "rating_spawn_claimed"
    assert lineage_rows[-1]["status"] == "ok"
    assert lineage_rows[-1]["claim_kind"] == "rating_loop"
    assert lineage_rows[-1]["rating_call_id"] == "fc-test"
    assert lineage_rows[-1]["event_count"] == 2


def test_opponent_leaderboard_publish_writes_snapshot_latest_and_pointer(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-a"
    latest_path = tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id)
    latest_path.parent.mkdir(parents=True)
    rating_snapshot = {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "formula_version": arena.RATING_FORMULA_VERSION,
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "ratings_ref": arena.rating_latest_ref(tournament_id, rating_run_id).as_posix(),
        "context_hash": "ctx-a",
        "roster_hash": "roster-a",
        "round_id": "round-000003",
        "round_index": 3,
        "stable": True,
        "max_abs_delta": 0.0,
        "rating_spec": {
            "decision_source_frames": 1,
        },
        "ratings": [
            {
                "checkpoint_id": "ckpt-a",
                "checkpoint_ref": _checkpoint_ref("run-a", 100000),
                "label": "run-a i100000",
                "rank": 1,
                "rating": 1700.0,
                "games": 500,
                "wins": 300,
                "losses": 150,
                "draws": 50,
                "battles": 20,
                "rated_battles": 20,
                "distinct_opponents": 20,
                "failure_count": 0,
            }
        ],
    }
    latest_path.write_text(json.dumps(rating_snapshot), encoding="utf-8")

    class FakeDict:
        def __init__(self) -> None:
            self.values = {}

        def put(self, key, value, **_kwargs):
            self.values[key] = value
            return True

    fake_dict = FakeDict()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "_commit_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "opponent_leaderboard_state", fake_dict)

    result = modal_arena.curvytron_opponent_leaderboard_publish.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "leaderboard_id": "main",
            "snapshot_id": "snapshot-001",
            "expected_round_id": "round-000003",
            "expected_round_index": 3,
            "expected_rating_context_hash": "ctx-a",
            "expected_roster_hash": "roster-a",
            "expected_rating_snapshot_sha256": canonical_json_sha256(rating_snapshot),
        }
    )

    snapshot_ref = modal_arena._leaderboard_snapshot_ref("main", "snapshot-001")
    latest_ref = modal_arena._leaderboard_latest_ref("main")
    assert (tmp_path / snapshot_ref).is_file()
    assert (tmp_path / latest_ref).is_file()
    assert result["row_count"] == 1
    assert result["active_count"] == 1
    assert result["rating_snapshot_sha256"] == canonical_json_sha256(rating_snapshot)
    assert result["rating_stable"] is True
    assert result["rating_max_abs_delta"] == 0.0
    assert result["pointer_key"] == "current:main"
    assert fake_dict.values["current:main"]["snapshot_id"] == "snapshot-001"


def test_opponent_leaderboard_publish_allows_complete_unstable_training_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-a"
    latest_path = tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id)
    latest_path.parent.mkdir(parents=True)
    latest_path.write_text(
        json.dumps(
            {
                "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
                "formula_version": arena.RATING_FORMULA_VERSION,
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "context_hash": "ctx-a",
                "roster_hash": "roster-a",
                "round_id": "round-000003",
                "round_index": 3,
                "stable": False,
                "max_abs_delta": 23.3,
                "rating_spec": {"decision_source_frames": 1},
                "ratings": [
                    {
                        "checkpoint_id": "ckpt-a",
                        "checkpoint_ref": _checkpoint_ref("run-a", 100000),
                        "label": "run-a i100000",
                        "rank": 1,
                        "rating": 1700.0,
                        "games": 500,
                        "wins": 300,
                        "losses": 150,
                        "draws": 50,
                        "battles": 20,
                        "rated_battles": 20,
                        "distinct_opponents": 20,
                        "failure_count": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "_commit_volume", lambda _volume: None)

    class FakeDict:
        def __init__(self) -> None:
            self.values = {}

        def put(self, key, value, **_kwargs):
            self.values[key] = value
            return True

    fake_dict = FakeDict()
    monkeypatch.setattr(modal_arena, "opponent_leaderboard_state", fake_dict)
    result = modal_arena.curvytron_opponent_leaderboard_publish.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "leaderboard_id": "main",
            "snapshot_id": "snapshot-unstable",
        }
    )

    assert result["diagnostic_only"] is False
    assert result["pointer_published"] is True
    assert result["rating_stable"] is False
    assert result["rating_max_abs_delta"] == 23.3
    assert fake_dict.values["current:main"]["snapshot_id"] == "snapshot-unstable"


def test_opponent_leaderboard_publish_honors_zero_canary_active_thresholds(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "tiny-canary"
    rating_run_id = "elo-tiny-canary"
    latest_path = tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id)
    latest_path.parent.mkdir(parents=True)
    rating_snapshot = {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "formula_version": arena.RATING_FORMULA_VERSION,
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "context_hash": "ctx-tiny",
        "roster_hash": "roster-tiny",
        "round_id": "round-000000",
        "round_index": 0,
        "stable": True,
        "max_abs_delta": 0.1,
        "rating_spec": {"decision_source_frames": 1},
        "ratings": [
            {
                "checkpoint_id": "ckpt-a",
                "checkpoint_ref": _checkpoint_ref("run-a", 0),
                "label": "run-a i0",
                "rank": 1,
                "rating": 1500.0,
                "games": 21,
                "wins": 12,
                "losses": 9,
                "draws": 0,
                "failure_count": 0,
            }
        ],
    }
    latest_path.write_text(json.dumps(rating_snapshot), encoding="utf-8")

    class FakeDict:
        def __init__(self) -> None:
            self.values = {}

        def put(self, key, value, **_kwargs):
            self.values[key] = value
            return True

    fake_dict = FakeDict()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "_commit_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "opponent_leaderboard_state", fake_dict)

    result = modal_arena.curvytron_opponent_leaderboard_publish.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "leaderboard_id": "tiny",
            "snapshot_id": "snapshot-tiny",
            "active_min_valid_games": 0,
            "active_min_distinct_opponents": 0,
        }
    )

    assert result["active_count"] == 1
    assert result["diagnostic_only"] is False
    assert result["pointer_published"] is True


def test_rating_round_outputs_write_pair_history_and_scheduler_state(tmp_path) -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": [
                _checkpoint_ref("run-a", 0),
                _checkpoint_ref("run-b", 10),
                _checkpoint_ref("run-c", 20),
            ],
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 1,
            "games_per_pair": 3,
        }
    )
    round_id = arena.rating_round_id(0)
    pair = arena.build_rating_round_pair_specs(rating_spec, round_index=0)[0]
    summary = arena.summarize_pair_results(
        pair,
        [
            _fake_game(pair, 0, "seat_0_win"),
            _fake_game(pair, 1, "draw"),
            _fake_game(pair, 2, "seat_1_win"),
        ],
    )
    summary["summary_ref"] = arena.battle_summary_ref(
        "arena-a",
        pair["battle_id"],
    ).as_posix()

    outputs = modal_arena._write_rating_round_outputs(
        tmp_path,
        spec=rating_spec,
        round_id=round_id,
        round_index=0,
        pair_results=[summary],
        pair_specs=[pair],
        game_count=3,
        started_at="2026-05-13T00:00:00Z",
        previous_snapshot=None,
    )

    snapshot = outputs["snapshot"]
    history_path = tmp_path / arena.rating_pair_history_ref("arena-a", "elo-test")
    scheduler_path = tmp_path / arena.rating_scheduler_state_ref("arena-a", "elo-test")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))

    assert (
        snapshot["pair_history_ref"]
        == arena.rating_pair_history_ref(
            "arena-a",
            "elo-test",
        ).as_posix()
    )
    assert (
        snapshot["scheduler_state_ref"]
        == arena.rating_scheduler_state_ref(
            "arena-a",
            "elo-test",
        ).as_posix()
    )
    assert outputs["pair_history"]["rows"][0]["pair_key"] == pair["pair_key"]
    assert history["rows"][0]["pair_key"] == pair["pair_key"]
    assert history["context_hash"] == arena.rating_context_hash(rating_spec)
    assert scheduler["context_hash"] == arena.rating_context_hash(rating_spec)
    assert scheduler["pair_selection"] == "adaptive_v0"
    assert scheduler["schedule_reason_counts"][pair["schedule_reason"]] == 1
    assert scheduler["pair_history_row_count"] == 1


def test_rating_snapshot_skips_pairs_with_too_many_failures() -> None:
    refs = [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)]
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "checkpoints": refs,
            "games_per_pair": 5,
            "min_valid_fraction": 0.75,
        }
    )
    pair = arena.build_rating_round_pair_specs(rating_spec)[0]
    summary = arena.summarize_pair_results(
        pair,
        [
            _fake_game(pair, 0, "seat_0_win"),
            _fake_game(pair, 1, "unfinished", ok=False),
            _fake_game(pair, 2, "unfinished", ok=False),
            _fake_game(pair, 3, "unfinished", ok=False),
            _fake_game(pair, 4, "unfinished", ok=False),
        ],
    )

    snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=[summary],
        rating_spec=rating_spec,
    )

    assert snapshot["rated_pair_count"] == 0
    assert snapshot["invalid_pair_count"] == 1
    assert {row["rating"] for row in snapshot["ratings"]} == {arena.DEFAULT_RATING_INITIAL_RATING}
    assert snapshot["pair_rating_results"][0]["rating_skip_reason"] == "not_enough_valid_games"


def test_tournament_artifact_ref_validation_is_strict() -> None:
    good = "tournaments/curvytron/arena-a/battles/pair-000000/games/game-000000/game.gif"
    assert arena.validate_tournament_artifact_ref(good).as_posix() == good
    rating_ref = "tournaments/curvytron/arena-a/ratings/elo/latest.json"
    assert arena.validate_tournament_artifact_ref(rating_ref).as_posix() == rating_ref

    with pytest.raises(ValueError):
        arena.validate_tournament_artifact_ref("../escape/game.gif")
    with pytest.raises(ValueError):
        arena.validate_tournament_artifact_ref(
            "training/lightzero-curvytron-visual-survival/run/raw.gif"
        )
    with pytest.raises(ValueError):
        arena.validate_tournament_artifact_ref(
            "tournaments/curvytron/arena-a/battles/pair/game.exe"
        )


def test_modal_browser_lists_tournaments_and_battles(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id = "arena-a"
    battle_id = "pair-000000"
    marker = tmp_path / arena.tournament_marker_ref(tournament_id)
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    (marker.parent / "tournament.json").write_text(
        json.dumps({"status": "running", "updated_at": "2026-05-13T00:00:00Z"}),
        encoding="utf-8",
    )
    battle_path = tmp_path / arena.battle_summary_ref(tournament_id, battle_id)
    battle_path.parent.mkdir(parents=True)
    battle_path.write_text(
        json.dumps(
            {
                "ok": True,
                "tournament_id": tournament_id,
                "battle_id": battle_id,
                "players": [
                    {"checkpoint_id": "a", "label": "A"},
                    {"checkpoint_id": "b", "label": "B"},
                ],
                "tally": {"completed_count": 2, "failure_count": 0},
            }
        ),
        encoding="utf-8",
    )

    tournaments = modal_arena._list_tournaments(tmp_path)
    battles = modal_arena._list_battles(
        tmp_path,
        tournament_id=tournament_id,
        limit=10,
        offset=0,
    )

    assert tournaments[0]["tournament_id"] == tournament_id
    assert battles["total"] == 1
    assert battles["rows"][0]["battle_id"] == battle_id
    assert battles["rows"][0]["summary_ref"].endswith("/battle.json")


def test_modal_browser_uses_battle_index_when_present(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id = "arena-a"
    marker = tmp_path / arena.tournament_marker_ref(tournament_id)
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    index_path = tmp_path / arena.tournament_battle_index_ref(tournament_id)
    index_path.write_text(
        json.dumps(
            {
                "schema_id": "curvyzero_curvytron_checkpoint_tournament_battle_index/v0",
                "tournament_id": tournament_id,
                "total": 1,
                "rows": [
                    {
                        "tournament_id": tournament_id,
                        "battle_id": "pair-indexed",
                        "players": [],
                        "tally": {"completed_count": 5},
                        "ok": True,
                        "summary_ref": (
                            "tournaments/curvytron/arena-a/battles/pair-indexed/battle.json"
                        ),
                        "updated_ts": 10.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    battles = modal_arena._list_battles(
        tmp_path,
        tournament_id=tournament_id,
        limit=10,
        offset=0,
    )

    assert battles["source"] == "battle_index"
    assert battles["total"] == 1
    assert battles["rows"][0]["battle_id"] == "pair-indexed"


def test_modal_browser_filters_battle_index_by_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, _rating_run_id = _write_review_fixture(tmp_path)

    battles = modal_arena._list_battles(
        tmp_path,
        tournament_id=tournament_id,
        checkpoint_id="ckpt-a",
        limit=10,
        offset=0,
    )

    assert battles["source"] == "battle_index"
    assert battles["checkpoint_id"] == "ckpt-a"
    assert battles["total"] == 2
    assert {row["battle_id"] for row in battles["rows"]} == {"battle-ab", "battle-ac"}


def test_write_battle_index_writes_checkpoint_indexes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id = "arena-a"
    pair_results = [
        {
            "tournament_id": tournament_id,
            "battle_id": "battle-ab",
            "pair_index": 0,
            "pair_key": arena.rating_pair_key("ckpt-a", "ckpt-b"),
            "schedule_reason": arena.SCHEDULE_REASON_PLACEMENT,
            "schedule_priority": 1.0,
            "scheduled_round_index": 0,
            "schedule": {
                "reason": arena.SCHEDULE_REASON_PLACEMENT,
                "priority": 1.0,
            },
            "players": [
                {"checkpoint_id": "ckpt-a", "label": "A", "seat": 0},
                {"checkpoint_id": "ckpt-b", "label": "B", "seat": 1},
            ],
            "tally": {"completed_count": 3, "failure_count": 0},
            "summary_ref": arena.battle_summary_ref(
                tournament_id,
                "battle-ab",
            ).as_posix(),
            "ok": True,
        },
        {
            "tournament_id": tournament_id,
            "battle_id": "battle-ac",
            "pair_index": 1,
            "players": [
                {"checkpoint_id": "ckpt-a", "label": "A", "seat": 0},
                {"checkpoint_id": "ckpt-c", "label": "C", "seat": 1},
            ],
            "tally": {"completed_count": 3, "failure_count": 0},
            "summary_ref": arena.battle_summary_ref(
                tournament_id,
                "battle-ac",
            ).as_posix(),
            "ok": True,
        },
    ]

    payload = modal_arena._write_battle_index(
        tournament_id,
        pair_results,
        mount=tmp_path,
    )
    checkpoint_path = tmp_path / arena.tournament_checkpoint_battle_index_ref(
        tournament_id,
        "ckpt-a",
    )
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_rows = {row["battle_id"]: row for row in checkpoint_payload["rows"]}
    battles = modal_arena._list_battle_index(
        tmp_path,
        tournament_id=tournament_id,
        checkpoint_id="ckpt-a",
        limit=10,
        offset=0,
    )

    assert payload["checkpoint_index_count"] == 3
    assert payload["ref"] == arena.tournament_battle_index_ref(tournament_id).as_posix()
    assert (
        checkpoint_payload["ref"]
        == arena.tournament_checkpoint_battle_index_ref(
            tournament_id,
            "ckpt-a",
        ).as_posix()
    )
    assert checkpoint_payload["total"] == 2
    assert checkpoint_rows["battle-ab"]["schedule_reason"] == arena.SCHEDULE_REASON_PLACEMENT
    assert checkpoint_rows["battle-ab"]["pair_key"] == arena.rating_pair_key("ckpt-a", "ckpt-b")
    assert battles["source"] == "checkpoint_battle_index"
    assert battles["total"] == 2
    assert {row["battle_id"] for row in battles["rows"]} == {"battle-ab", "battle-ac"}


def test_checkpoint_battle_index_filters_stale_wrong_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id = "arena-a"
    checkpoint_id = "ckpt-a"
    checkpoint_path = tmp_path / arena.tournament_checkpoint_battle_index_ref(
        tournament_id,
        checkpoint_id,
    )
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema_id": "curvyzero_curvytron_checkpoint_tournament_battle_index/v0",
                "tournament_id": tournament_id,
                "checkpoint_id": checkpoint_id,
                "total": 2,
                "rows": [
                    {
                        "tournament_id": tournament_id,
                        "battle_id": "battle-ab",
                        "checkpoint_ids": ["ckpt-a", "ckpt-b"],
                        "players": [
                            {"checkpoint_id": "ckpt-a", "label": "A", "seat": 0},
                            {"checkpoint_id": "ckpt-b", "label": "B", "seat": 1},
                        ],
                        "updated_ts": 2.0,
                    },
                    {
                        "tournament_id": tournament_id,
                        "battle_id": "battle-cd",
                        "checkpoint_ids": ["ckpt-c", "ckpt-d"],
                        "players": [
                            {"checkpoint_id": "ckpt-c", "label": "C", "seat": 0},
                            {"checkpoint_id": "ckpt-d", "label": "D", "seat": 1},
                        ],
                        "updated_ts": 1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    battles = modal_arena._list_battle_index(
        tmp_path,
        tournament_id=tournament_id,
        checkpoint_id=checkpoint_id,
        limit=10,
        offset=0,
    )

    assert battles["source"] == "checkpoint_battle_index"
    assert battles["total"] == 1
    assert battles["rows"][0]["battle_id"] == "battle-ab"


def test_modal_browser_lists_rating_runs_and_standings(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    marker = tmp_path / arena.tournament_marker_ref(tournament_id)
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    latest_path = tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id)
    latest_path.parent.mkdir(parents=True)
    latest_path.write_text(
        json.dumps(
            {
                "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "round_id": "round-000000",
                "round_index": 0,
                "checkpoint_count": 1,
                "rated_pair_count": 1,
                "max_abs_delta": 12.0,
                "stable": False,
                "ratings": [
                    {
                        "rank": 1,
                        "checkpoint_id": "ckpt-a",
                        "label": "A",
                        "rating": 1512.0,
                        "games": 3,
                        "wins": 2,
                        "losses": 0,
                        "draws": 1,
                        "failure_count": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (latest_path.parent / "config.json").write_text(
        json.dumps({"formula_version": arena.RATING_FORMULA_VERSION}),
        encoding="utf-8",
    )

    rating_runs = modal_arena._list_rating_runs(tmp_path, tournament_id=tournament_id)
    snapshot = modal_arena._read_rating_snapshot(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

    assert rating_runs[0]["rating_run_id"] == rating_run_id
    assert rating_runs[0]["latest_ref"].endswith("/latest.json")
    assert snapshot["ratings"][0]["rating"] == 1512.0


def test_review_checkpoint_payload_uses_rankings_and_battle_index(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, rating_run_id = _write_review_fixture(tmp_path)

    payload = modal_arena._review_checkpoint_payload(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        checkpoint_id="ckpt-a",
    )

    assert payload["selected_tournament_id"] == tournament_id
    assert payload["rating_run_id"] == rating_run_id
    assert payload["ranking"]["checkpoint_id"] == "ckpt-a"
    assert payload["total"] == 2
    assert [row["opponent"]["checkpoint_id"] for row in payload["rows"]] == [
        "ckpt-b",
        "ckpt-c",
    ]
    assert payload["rows"][0]["checkpoint_wins"] == 3
    assert payload["rows"][0]["opponent_wins"] == 1
    assert payload["rows"][0]["first_gif_ref"].endswith("/game.gif")


def test_review_checkpoint_payload_uses_checkpoint_index_without_live_shard_scan(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, rating_run_id = _write_review_fixture(tmp_path)
    checkpoint_ref = arena.tournament_checkpoint_battle_index_ref(
        tournament_id,
        "ckpt-a",
    )
    modal_arena.runs.write_json(
        tmp_path / checkpoint_ref,
        {
            "schema_id": "curvyzero_curvytron_checkpoint_tournament_battle_index/v0",
            "tournament_id": tournament_id,
            "checkpoint_id": "ckpt-a",
            "ref": checkpoint_ref.as_posix(),
            "total": 1,
            "rows": [
                {
                    "tournament_id": tournament_id,
                    "rating_run_id": rating_run_id,
                    "battle_id": "battle-ab",
                    "players": [
                        {"checkpoint_id": "ckpt-a", "label": "A", "seat": 0},
                        {"checkpoint_id": "ckpt-b", "label": "B", "seat": 1},
                    ],
                    "checkpoint_ids": ["ckpt-a", "ckpt-b"],
                    "tally": {
                        "completed_count": 6,
                        "failure_count": 0,
                        "draw_count": 2,
                        "wins_by_checkpoint": {"ckpt-a": 3, "ckpt-b": 1},
                        "wins_by_seat": {"seat_0": 3, "seat_1": 1},
                        "average_physical_steps": 12.0,
                    },
                    "first_gif_ref": (
                        "tournaments/curvytron/arena-a/battles/battle-ab/games/game-000000/game.gif"
                    ),
                    "updated_ts": 10.0,
                    "ok": True,
                }
            ],
        },
    )

    def fail_live_scan(*_args, **_kwargs):
        raise AssertionError("checkpoint index path should not scan live shards")

    monkeypatch.setattr(modal_arena, "_read_battle_shard_summaries", fail_live_scan)

    payload = modal_arena._review_checkpoint_payload(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        checkpoint_id="ckpt-a",
    )

    assert payload["source"] == "checkpoint_battle_index"
    assert payload["total"] == 1
    assert payload["rows"][0]["battle_id"] == "battle-ab"
    assert payload["rows"][0]["completed_count"] == 6


def test_review_checkpoint_payload_does_not_request_unbounded_battle_index(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, rating_run_id = _write_review_fixture(tmp_path)
    original = modal_arena._list_battle_index
    seen_limits: list[int] = []

    def recording_index(*args, **kwargs):
        seen_limits.append(int(kwargs["limit"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(modal_arena, "_list_battle_index", recording_index)

    payload = modal_arena._review_checkpoint_payload(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        checkpoint_id="ckpt-a",
        limit=1,
    )

    assert payload["limit"] == 1
    assert payload["total"] == 2
    assert seen_limits
    assert max(seen_limits) <= modal_arena.MAX_LIMIT


def test_review_battle_payload_reads_shard_game_index_before_scanning(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, _rating_run_id = _write_review_fixture(tmp_path)
    shard_ref = arena.game_shard_summary_ref(
        tournament_id,
        "battle-ab",
        "shard-000000",
    )
    shard_games = []
    for index, outcome in enumerate(["seat_0_win", "draw"]):
        game_id = f"game-{index:06d}"
        shard_games.append(
            {
                "ok": True,
                "game_id": game_id,
                "game_index": index,
                "seed": 2000 + index,
                "physical_steps": 20 + index,
                "score": {
                    "outcome": outcome,
                    "winner_seat": 0 if outcome == "seat_0_win" else None,
                    "loser_seat": 1 if outcome == "seat_0_win" else None,
                    "draw": outcome == "draw",
                    "physical_steps": 20 + index,
                },
                "gif_ref": (
                    f"tournaments/curvytron/arena-a/battles/battle-ab/games/{game_id}/game.gif"
                ),
                "summary_ref": arena.game_summary_ref(
                    tournament_id,
                    "battle-ab",
                    game_id,
                ).as_posix(),
            }
        )
    modal_arena.runs.write_json(
        tmp_path / shard_ref,
        {
            "schema_id": arena.GAME_SHARD_SCHEMA_ID,
            "battle_id": "battle-ab",
            "shard_id": "shard-000000",
            "games": shard_games,
        },
    )
    battle_ref = arena.battle_summary_ref(tournament_id, "battle-ab")
    battle_summary = modal_arena._read_json(tmp_path / battle_ref)
    battle_summary["tally"]["game_count"] = 2
    battle_summary["game_summary_ref_count"] = 2
    battle_summary["shard_summary_refs"] = [shard_ref.as_posix()]
    modal_arena.runs.write_json(tmp_path / battle_ref, battle_summary)

    payload = modal_arena._review_battle_payload(
        tmp_path,
        tournament_id=tournament_id,
        battle_id="battle-ab",
    )

    assert payload["game_count"] == 2
    assert payload["game_sources"] == ["shard_summary_refs"]
    assert [game["seed"] for game in payload["games"]] == [2000, 2001]
    assert len(payload["sample_gifs"]) == 2


def test_review_battle_payload_reads_battle_summary_before_index(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, _rating_run_id = _write_review_fixture(tmp_path)

    def fail_index(*_args, **_kwargs):
        raise AssertionError("battle index should not be read for direct battle detail")

    monkeypatch.setattr(modal_arena, "_list_battle_index", fail_index)

    payload = modal_arena._review_battle_payload(
        tmp_path,
        tournament_id=tournament_id,
        battle_id="battle-ab",
    )

    assert payload["source"] == "battle_summary"
    assert payload["summary"]["battle_id"] == "battle-ab"
    assert payload["game_count"] == 6


def test_review_battle_payload_can_open_live_battle_before_battle_summary(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id = "arena-a"
    marker = tmp_path / arena.tournament_marker_ref(tournament_id)
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    pair = {
        "battle_id": "battle-live",
        "players": [
            {"seat": 0, "checkpoint_id": "ckpt-a", "label": "A"},
            {"seat": 1, "checkpoint_id": "ckpt-b", "label": "B"},
        ],
    }
    game = {
        **_fake_game(pair, 0, "seat_0_win"),
        "gif_ref": ("tournaments/curvytron/arena-a/battles/battle-live/games/game-000000/game.gif"),
    }
    modal_arena.runs.write_json(
        tmp_path / arena.game_summary_ref(tournament_id, "battle-live", "game-000000"),
        game,
    )

    def fail_index(*_args, **_kwargs):
        raise AssertionError("live battle detail should not scan the full battle index")

    monkeypatch.setattr(modal_arena, "_list_battle_index", fail_index)

    payload = modal_arena._review_battle_payload(
        tmp_path,
        tournament_id=tournament_id,
        battle_id="battle-live",
    )

    assert payload["source"] == "battle_dir"
    assert payload["game_count"] == 1
    assert payload["games"][0]["outcome"] == "seat_0_win"
    assert payload["sample_gif_count"] == 1


def test_best_rating_snapshot_builds_provisional_rows_from_live_shards(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, spec, _pair = _write_live_rating_fixture(tmp_path)

    snapshot = modal_arena._build_provisional_rating_snapshot_for_run(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

    ids = [row["checkpoint_id"] for row in spec["checkpoints"]]
    rows_by_id = {row["checkpoint_id"]: row for row in snapshot["ratings"]}
    assert snapshot["provisional"] is True
    assert snapshot["source"] == "live_shard_summaries"
    assert snapshot["completed_pair_count"] == 1
    assert snapshot["completed_game_count"] == 3
    assert rows_by_id[ids[0]]["rating"] > rows_by_id[ids[1]]["rating"]
    assert snapshot["live_pair_results"][0]["sample_gif_refs"]


def test_review_rankings_payload_builds_live_provisional_without_written_latest(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, _spec, _pair = _write_live_rating_fixture(tmp_path)

    payload = modal_arena._review_rankings_payload(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        allow_live_provisional=True,
    )

    assert payload["provisional"] is True
    assert payload["source"] == "live_shard_summaries"
    assert payload["total"] == 3
    assert payload["rows"][0]["games"] == 3


def test_review_rankings_route_can_return_live_provisional_on_fresh_request(
    tmp_path,
    monkeypatch,
) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    TestClient = fastapi_testclient.TestClient

    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_web_reload_volume", lambda *_args, **_kwargs: None)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, _spec, _pair = _write_live_rating_fixture(tmp_path)
    client = TestClient(modal_arena._build_fastapi_app(object()))

    response = client.get(
        "/api/review/rankings",
        params={
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "fresh": "true",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provisional"] is True
    assert payload["source"] == "live_shard_summaries"
    assert payload["total"] == 3
    assert payload["rows"][0]["games"] == 3


def test_rating_standings_route_can_return_live_provisional_on_fresh_request(
    tmp_path,
    monkeypatch,
) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    TestClient = fastapi_testclient.TestClient

    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_web_reload_volume", lambda *_args, **_kwargs: None)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, _spec, _pair = _write_live_rating_fixture(tmp_path)
    client = TestClient(modal_arena._build_fastapi_app(object()))

    response = client.get(
        "/api/rating-standings",
        params={
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "fresh": "true",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provisional"] is True
    assert payload["source"] == "live_shard_summaries"
    assert payload["total"] == 3
    assert payload["rows"][0]["games"] == 3


def test_best_rating_snapshot_reads_written_provisional_file(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, _spec, _pair = _write_live_rating_fixture(tmp_path)
    provisional = modal_arena._build_provisional_rating_snapshot_for_run(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    modal_arena.runs.write_json(
        tmp_path / modal_arena._rating_provisional_latest_ref(tournament_id, rating_run_id),
        modal_arena._slim_provisional_rating_snapshot(provisional),
    )

    snapshot = modal_arena._read_best_rating_snapshot_for_run(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

    assert snapshot["provisional"] is True
    assert snapshot["ratings"]
    assert "live_pair_results" not in snapshot


def test_cached_live_rating_progress_reads_small_progress_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, _spec, _pair = _write_live_rating_fixture(tmp_path)

    progress = modal_arena._read_cached_live_rating_progress(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

    assert progress["status"] == "running"
    assert progress["phase"] == "games_running"
    assert progress["pair_count"] == 3
    assert progress["game_count"] == 9
    assert progress["completed_pair_count"] == 0
    assert progress["completed_game_count"] == 0
    assert "count_basis" not in progress


def test_cached_live_rating_progress_force_live_reads_shard_progress(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, _spec, _pair = _write_live_rating_fixture(tmp_path)

    progress = modal_arena._read_cached_live_rating_progress(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        force_live=True,
    )

    assert progress["status"] == "running"
    assert progress["phase"] == "games_running"
    assert progress["count_basis"] == "shard_summary_files"
    assert progress["completed_pair_count"] == 1
    assert progress["completed_game_count"] == 3


def test_live_rating_progress_reads_live_shard_progress(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, _spec, _pair = _write_live_rating_fixture(tmp_path)

    progress = modal_arena._read_live_rating_progress(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

    assert progress["status"] == "running"
    assert progress["phase"] == "games_running"
    assert progress["pair_count"] == 3
    assert progress["game_count"] == 9
    assert progress["count_basis"] == "shard_summary_files"
    assert progress["completed_pair_count"] == 1
    assert progress["partial_pair_count"] == 0
    assert progress["completed_game_count"] == 3


def test_live_rating_progress_reports_volume_scan_errors_without_crashing(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, _spec, _pair = _write_live_rating_fixture(tmp_path)
    original_iterdir = Path.iterdir

    def flaky_iterdir(path: Path):
        if path.name == "battles":
            raise OSError("temporary volume scan failure")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", flaky_iterdir)

    progress = modal_arena._read_live_rating_progress(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

    assert progress["status"] == "running"
    assert progress["pair_count"] == 3
    assert progress["scan_errors"]
    assert progress["scan_errors"][0]["operation"] == "iterdir"
    assert progress["scan_errors"][0]["error_type"] == "OSError"


def test_diagnostic_pair_progress_counts_game_summaries_without_shards(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, _spec, pair = _write_live_rating_fixture(tmp_path)
    shard_summary_path = tmp_path / arena.game_shard_summary_ref(
        tournament_id,
        pair["battle_id"],
        "shard-000000-games-000000-000002",
    )
    shard_summary_path.unlink()
    for index in range(2):
        game = {
            **_fake_game(pair, index, "seat_0_win"),
            "tournament_id": tournament_id,
            "battle_id": pair["battle_id"],
            "pair_index": pair["pair_index"],
        }
        modal_arena.runs.write_json(
            tmp_path
            / arena.game_summary_ref(
                tournament_id,
                pair["battle_id"],
                game["game_id"],
            ),
            game,
        )

    progress, _games_by_battle = modal_arena._rating_round_progress_payload(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=arena.rating_round_id(0),
        load_summaries=False,
        pair_only=True,
        count_game_summaries=True,
    )

    assert progress["count_basis"] == "shard_and_game_summary_files"
    assert progress["completed_game_count"] == 2
    assert progress["completed_pair_count"] == 0
    assert progress["partial_pair_count"] == 1


def test_progress_merges_written_provisional_counts(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, _spec, _pair = _write_live_rating_fixture(tmp_path)
    provisional = modal_arena._build_provisional_rating_snapshot_for_run(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    modal_arena.runs.write_json(
        tmp_path / modal_arena._rating_provisional_latest_ref(tournament_id, rating_run_id),
        modal_arena._slim_provisional_rating_snapshot(provisional),
    )

    progress = modal_arena._read_cached_live_rating_progress(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

    assert progress["phase"] == "games_running_with_provisional_ratings"
    assert progress["provisional_ratings_written"] is True
    assert progress["completed_pair_count"] == 1
    assert progress["completed_game_count"] == 3


def test_review_checkpoint_payload_enriches_stale_battle_index_from_live_shards(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, spec, pair = _write_live_rating_fixture(tmp_path)
    checkpoint_id = spec["checkpoints"][0]["checkpoint_id"]
    modal_arena.runs.write_json(
        tmp_path / arena.tournament_battle_index_ref(tournament_id),
        {
            "schema_id": "curvyzero_curvytron_checkpoint_tournament_battle_index/v0",
            "tournament_id": tournament_id,
            "total": 1,
            "rows": [
                {
                    "tournament_id": tournament_id,
                    "battle_id": pair["battle_id"],
                    "pair_index": pair["pair_index"],
                    "players": pair["players"],
                    "checkpoint_ids": [player["checkpoint_id"] for player in pair["players"]],
                    "summary_ref": arena.battle_summary_ref(
                        tournament_id,
                        pair["battle_id"],
                    ).as_posix(),
                    "updated_ts": 1.0,
                }
            ],
        },
    )

    payload = modal_arena._review_checkpoint_payload(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        checkpoint_id=checkpoint_id,
        limit=10,
    )

    assert payload["source"] == "battle_index"
    assert payload["total"] == 1
    assert payload["rows"][0]["completed_count"] == 3
    assert payload["rows"][0]["sample_gif_refs"]
    assert payload["rows"][0]["first_gif_ref"].endswith("/game.gif")


def test_progress_merge_keeps_complete_status(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, rating_run_id, _spec, _pair = _write_live_rating_fixture(tmp_path)
    provisional = modal_arena._build_provisional_rating_snapshot_for_run(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    modal_arena.runs.write_json(
        tmp_path / modal_arena._rating_provisional_latest_ref(tournament_id, rating_run_id),
        modal_arena._slim_provisional_rating_snapshot(provisional),
    )

    progress = modal_arena._merge_progress_with_provisional_snapshot(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        progress={"status": "complete", "phase": "ratings_written"},
    )

    assert progress == {"status": "complete", "phase": "ratings_written"}


def test_progress_refresh_spawn_is_throttled(monkeypatch) -> None:
    modal_arena._WEB_CACHE.clear()
    calls: list[dict[str, object]] = []

    class FakeCall:
        object_id = "fc-refresh"

    class FakeProgressFunction:
        @staticmethod
        def spawn(payload):
            calls.append(dict(payload))
            return FakeCall()

    monkeypatch.setattr(
        modal_arena,
        "curvytron_rating_progress",
        FakeProgressFunction,
    )

    first = modal_arena._maybe_spawn_rating_progress_refresh(
        tournament_id="arena-live",
        rating_run_id="elo-live",
        progress={"status": "running", "round_index": 2},
        min_interval_seconds=60.0,
    )
    second = modal_arena._maybe_spawn_rating_progress_refresh(
        tournament_id="arena-live",
        rating_run_id="elo-live",
        progress={"status": "running", "round_index": 2},
        min_interval_seconds=60.0,
    )
    complete = modal_arena._maybe_spawn_rating_progress_refresh(
        tournament_id="arena-live",
        rating_run_id="elo-live",
        progress={"status": "complete", "round_index": 2},
        min_interval_seconds=60.0,
    )

    assert first == "fc-refresh"
    assert second == "fc-refresh"
    assert complete == ""
    assert calls == [
        {
            "tournament_id": "arena-live",
            "rating_run_id": "elo-live",
            "round_id": "round-000002",
            "round_index": 2,
            "load_summaries": False,
        }
    ]


def test_provisional_artifacts_write_battle_index_for_checkpoint_drilldown(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, spec, pair = _write_live_rating_fixture(tmp_path)
    provisional = modal_arena._build_provisional_rating_snapshot_for_run(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

    artifacts = modal_arena._write_provisional_rating_artifacts(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        snapshot=provisional,
    )
    payload = modal_arena._review_checkpoint_payload(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        checkpoint_id=spec["checkpoints"][0]["checkpoint_id"],
        limit=10,
    )

    assert artifacts["battle_index"]["total"] == 1
    assert payload["source"] == "checkpoint_battle_index"
    assert payload["total"] == 1
    assert payload["rows"][0]["battle_id"] == pair["battle_id"]


def test_review_checkpoint_payload_falls_back_to_checkpoint_live_shards(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, spec, pair = _write_live_rating_fixture(tmp_path)
    provisional = modal_arena._build_provisional_rating_snapshot_for_run(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    modal_arena.runs.write_json(
        tmp_path / modal_arena._rating_provisional_latest_ref(tournament_id, rating_run_id),
        modal_arena._slim_provisional_rating_snapshot(provisional),
    )

    payload = modal_arena._review_checkpoint_payload(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        checkpoint_id=spec["checkpoints"][0]["checkpoint_id"],
        limit=10,
    )

    assert payload["source"] == "checkpoint_round_input"
    assert payload["total"] == 2
    row_by_id = {row["battle_id"]: row for row in payload["rows"]}
    assert row_by_id[pair["battle_id"]]["sample_gif_refs"]


def test_review_battle_payload_reads_live_shard_summary_without_battle_json(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, _rating_run_id, _spec, pair = _write_live_rating_fixture(tmp_path)

    payload = modal_arena._review_battle_payload(
        tmp_path,
        tournament_id=tournament_id,
        battle_id=pair["battle_id"],
    )

    assert payload["source"] == "battle_dir"
    assert payload["game_count"] == 3
    assert payload["game_sources"] == ["shard_summary_refs"]
    assert payload["sample_gif_count"] == 1


def test_best_rating_snapshot_prefers_final_latest_over_provisional(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, _spec, _pair = _write_live_rating_fixture(tmp_path)
    modal_arena.runs.write_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id),
        {
            "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": arena.rating_round_id(0),
            "ratings": [{"checkpoint_id": "final", "rank": 1, "rating": 1600.0}],
        },
    )

    snapshot = modal_arena._read_best_rating_snapshot_for_run(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

    assert snapshot.get("provisional") is not True
    assert snapshot["ratings"][0]["checkpoint_id"] == "final"


def test_render_page_links_rankings_to_checkpoint_battle_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, rating_run_id = _write_review_fixture(tmp_path)
    tournaments = modal_arena._list_tournaments(tmp_path)
    rating_runs = modal_arena._list_rating_latest_runs(
        tmp_path,
        tournament_id=tournament_id,
    )
    snapshot = modal_arena._read_rating_snapshot_for_run(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    battles = modal_arena._list_battles(
        tmp_path,
        tournament_id=tournament_id,
        checkpoint_id="ckpt-a",
        limit=10,
        offset=0,
    )

    html = modal_arena._render_page(
        tournaments=tournaments,
        selected_tournament_id=tournament_id,
        selected_rating_run_id=rating_run_id,
        selected_checkpoint_id="ckpt-a",
        rating_runs=rating_runs,
        rating_snapshot=snapshot,
        rating_progress={},
        battles=battles,
    )

    assert "Leaderboard" in html
    assert "rankings-scroll" in html
    assert "checkpoint_id=ckpt-a" in html
    assert "Battles" in html
    assert "battles-scroll" in html
    assert "battle_id=battle-ab" in html
    assert "battle_id=battle-ab#battle-detail" in html
    battles_html = html.split("<h2>Battles</h2>", 1)[1]
    assert battles_html.index("battle_id=battle-ab#battle-detail") < battles_html.index(
        "battle_id=battle-ac#battle-detail"
    )
    assert "/battle?tournament_id=arena-a" not in html
    assert "/gif?ref=tournaments/curvytron/arena-a/battles/battle-ab/" in html
    assert "/meta?ref=tournaments/curvytron/arena-a/battles/battle-ab/battle.json" in html


def test_render_page_battle_table_has_client_sort_controls(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, rating_run_id = _write_review_fixture(tmp_path)
    tournaments = modal_arena._list_tournaments(tmp_path)
    rating_runs = modal_arena._list_rating_latest_runs(
        tmp_path,
        tournament_id=tournament_id,
    )
    snapshot = modal_arena._read_rating_snapshot_for_run(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    battles = modal_arena._list_battles(
        tmp_path,
        tournament_id=tournament_id,
        checkpoint_id="ckpt-a",
        limit=10,
        offset=0,
    )

    html = modal_arena._render_page(
        tournaments=tournaments,
        selected_tournament_id=tournament_id,
        selected_rating_run_id=rating_run_id,
        selected_checkpoint_id="ckpt-a",
        rating_runs=rating_runs,
        rating_snapshot=snapshot,
        rating_progress={},
        battles=battles,
    )

    assert "data-battle-table" in html
    assert 'data-sort-key="rank"' in html
    assert 'data-sort-direction="asc"' in html
    assert 'data-battle-sort="rank"' in html
    assert 'data-battle-sort="avgSteps"' in html
    assert 'data-battle-sort="failures"' in html
    assert 'data-sort-rank="2"' in html
    assert 'data-sort-avg-steps="12"' in html
    assert 'data-sort-failures="0"' in html
    assert "const applyBattleSort" in html
    assert "nextDirection" in html
    assert 'aria-sort="ascending"' in html


def test_render_page_expands_selected_battle_games_and_gifs_in_place(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, rating_run_id = _write_review_fixture(tmp_path)
    tournaments = modal_arena._list_tournaments(tmp_path)
    rating_runs = modal_arena._list_rating_latest_runs(
        tmp_path,
        tournament_id=tournament_id,
    )
    snapshot = modal_arena._read_rating_snapshot_for_run(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    battles = modal_arena._list_battles(
        tmp_path,
        tournament_id=tournament_id,
        checkpoint_id="ckpt-a",
        limit=10,
        offset=0,
    )
    battle_detail = modal_arena._review_battle_payload(
        tmp_path,
        tournament_id=tournament_id,
        battle_id="battle-ab",
        gif_sample_limit=10,
    )

    html = modal_arena._render_page(
        tournaments=tournaments,
        selected_tournament_id=tournament_id,
        selected_rating_run_id=rating_run_id,
        selected_checkpoint_id="ckpt-a",
        rating_runs=rating_runs,
        rating_snapshot=snapshot,
        rating_progress={},
        battles=battles,
        selected_battle_id="battle-ab",
        battle_detail=battle_detail,
    )

    assert 'id="battle-detail"' in html
    assert "GIF Samples" in html
    assert "Games" in html
    assert "game-000003" in html
    assert "seat_1_win" in html
    assert (
        "/gif?ref=tournaments/curvytron/arena-a/battles/battle-ab/games/game-000004/game.gif"
        in html
    )
    assert 'loading="lazy"' in html
    assert 'decoding="async"' in html
    assert (
        "/meta?ref=tournaments/curvytron/arena-a/battles/battle-ab/games/game-000005/summary.json"
        in html
    )
    assert "/battle?" not in html


def test_render_page_dropdowns_update_url_and_clear_stale_selection() -> None:
    html = modal_arena._render_page(
        tournaments=[
            {"tournament_id": "arena-a", "status": "running", "updated_ts": 2.0},
            {"tournament_id": "arena-b", "status": "running", "updated_ts": 1.0},
        ],
        selected_tournament_id="arena-a",
        selected_rating_run_id="elo-a",
        selected_checkpoint_id="ckpt-a",
        rating_runs=[
            {
                "tournament_id": "arena-a",
                "rating_run_id": "elo-a",
                "status": "running",
            },
            {
                "tournament_id": "arena-a",
                "rating_run_id": "elo-b",
                "status": "running",
            },
        ],
        rating_snapshot={},
        rating_progress={},
        battles={"rows": [], "total": 0},
        selected_battle_id="battle-a",
    )

    assert 'id="tournament-picker"' in html
    assert 'data-picker="tournament"' in html
    assert 'data-picker="rating"' in html
    assert 'tournamentSelect.addEventListener("change"' in html
    assert 'ratingSelect.addEventListener("change"' in html
    assert "window.location.assign(url.toString())" in html
    assert 'url.searchParams.delete("checkpoint_id")' in html
    assert 'url.searchParams.delete("battle_id")' in html
    assert 'url.searchParams.delete("fresh")' in html
    assert 'url.searchParams.delete("rating_run_id")' in html


def test_render_page_labels_current_tournament_and_rating_options() -> None:
    html = modal_arena._render_page(
        tournaments=[
            {
                "tournament_id": modal_arena.CURRENT_TOURNAMENT_ID,
                "status": "running",
                "is_current": True,
                "updated_ts": 2.0,
            },
            {"tournament_id": "arena-old", "status": "complete", "updated_ts": 1.0},
        ],
        selected_tournament_id=modal_arena.CURRENT_TOURNAMENT_ID,
        selected_rating_run_id=modal_arena.CURRENT_RATING_RUN_ID,
        selected_checkpoint_id="",
        rating_runs=[
            {
                "tournament_id": modal_arena.CURRENT_TOURNAMENT_ID,
                "rating_run_id": modal_arena.CURRENT_RATING_RUN_ID,
                "status": "complete",
                "is_current": True,
            },
            {
                "tournament_id": modal_arena.CURRENT_TOURNAMENT_ID,
                "rating_run_id": "elo-old",
                "status": "complete",
            },
        ],
        rating_snapshot={},
        rating_progress={},
        battles={"rows": [], "total": 0},
    )

    assert f"{modal_arena.CURRENT_TOURNAMENT_ID} (current)" in html
    assert f"{modal_arena.CURRENT_RATING_RUN_ID} (current) (complete)" in html


def test_tournament_visibility_can_hide_all_except_keep(tmp_path) -> None:
    _write_review_fixture(tmp_path)
    modal_arena._write_tournament_marker_at(tmp_path, "arena-b")

    dry_run = modal_arena._update_tournament_visibility(
        tmp_path,
        action="hide_except",
        keep_tournament_ids="arena-a",
        dry_run=True,
    )
    assert dry_run["changed_count"] == 1
    assert (tmp_path / arena.tournament_marker_ref("arena-b")).exists()

    result = modal_arena._update_tournament_visibility(
        tmp_path,
        action="hide_except",
        keep_tournament_ids="arena-a",
        dry_run=False,
    )
    rows_by_id = {row["tournament_id"]: row for row in result["rows"]}
    assert result["changed_count"] == 1
    assert rows_by_id["arena-a"]["visible"] is True
    assert rows_by_id["arena-b"]["visible"] is False
    assert not (tmp_path / arena.tournament_marker_ref("arena-b")).exists()


def test_pending_rating_progress_payload_is_writeable(tmp_path) -> None:
    progress = modal_arena._pending_rating_progress(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        round_id=arena.rating_round_id(0),
        round_index=0,
        reason="waiting_for_round_input",
    )

    modal_arena._write_rating_progress(tmp_path, progress)
    written = modal_arena._read_json(tmp_path / arena.rating_progress_ref("arena-a", "elo-test"))

    assert written["status"] == "pending"
    assert written["phase"] == "waiting_for_round_input"
    assert written["input_ref"].endswith("/input.json")


def test_empty_waiting_rating_progress_does_not_block_round_start(tmp_path) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    round_id = arena.rating_round_id(1)
    progress = modal_arena._pending_rating_progress(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        round_index=1,
        reason="waiting_for_round_input",
    )
    modal_arena._write_rating_progress(tmp_path, progress)

    artifacts = modal_arena._rating_round_existing_blocking_artifacts(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
    )

    assert artifacts == []


def test_nonempty_rating_progress_blocks_round_start(tmp_path) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    round_id = arena.rating_round_id(1)
    progress = modal_arena._pending_rating_progress(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        round_index=1,
        reason="waiting_for_round_input",
    )
    progress["pair_count"] = 3
    modal_arena._write_rating_progress(tmp_path, progress)

    artifacts = modal_arena._rating_round_existing_blocking_artifacts(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
    )

    assert artifacts == [arena.rating_round_progress_ref(tournament_id, rating_run_id, round_id)]


def test_rating_loop_round_call_uses_spawn(monkeypatch) -> None:
    class FakeCall:
        def get(self):
            return {"round_id": "round-000004", "ok": True}

    class FakeRatingRound:
        def __init__(self) -> None:
            self.spawned: list[dict[str, object]] = []

        def spawn(self, spec):
            self.spawned.append(dict(spec))
            return FakeCall()

        def remote(self, _spec):
            raise AssertionError("rating loop must spawn round workers")

    fake = FakeRatingRound()
    monkeypatch.setattr(modal_arena, "curvytron_rating_round", fake)

    result = modal_arena._spawn_rating_round_and_get({"round_index": 4})

    assert result == {"round_id": "round-000004", "ok": True}
    assert fake.spawned == [{"round_index": 4}]


def test_checkpoint_intake_drain_tick_spawns_drain_without_local_mutation(monkeypatch) -> None:
    manifest_key = "manifest:arena-a:elo-test"
    manifest = {
        "tournament_id": "arena-a",
        "rating_run_id": "elo-test",
        "active": True,
        "checkpoint_refs": ["a", "b", "c"],
        "rating_defaults": {"continue_from_latest": True},
    }

    class FakeState:
        def get(self, key, default=None):
            if key == modal_arena.CHECKPOINT_INTAKE_ACTIVE_KEYS:
                return [manifest_key]
            if key == manifest_key:
                return manifest
            return default

    class FakeCall:
        object_id = "fc-drain"

    class FakeDrain:
        def __init__(self) -> None:
            self.spawned: list[dict[str, object]] = []

        def spawn(self, spec):
            self.spawned.append(dict(spec))
            return FakeCall()

        def local(self, _spec):
            raise AssertionError("scheduled drain tick must not run drain.local")

    fake_drain = FakeDrain()
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "curvytron_checkpoint_intake_drain", fake_drain)
    monkeypatch.setattr(
        modal_arena,
        "_feedback_loop_status_payload",
        lambda **_kwargs: {
            "status": "ready_for_next_rating_batch",
            "intake": {"new_checkpoints_not_in_latest_rating": 3, "queue_len": 3},
            "flags": [],
            "current_game_batch": None,
        },
    )
    monkeypatch.setattr(
        modal_arena,
        "_feedback_loop_control_decision",
        lambda _status, *, action: {
            "action": action,
            "spawn_drain": True,
            "reason": "spawn_next_bounded_rating_batch",
            "new_checkpoints_not_in_latest_rating": 3,
            "queue_len": 3,
        },
    )

    result = modal_arena.curvytron_checkpoint_intake_drain_tick.local()

    assert result["spawned_drain_count"] == 1
    assert result["drain_call_ids"] == ["fc-drain"]
    assert fake_drain.spawned == [
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "max_events": 100,
            "spawn_rating": True,
            "continue_from_latest": True,
            "spawn_if_existing": True,
            "rating_round_stale_after_seconds": modal_arena.DEFAULT_RATING_ROUND_STALE_SECONDS,
        }
    ]


def test_checkpoint_intake_drain_tick_respects_feedback_loop_gate(monkeypatch) -> None:
    manifest_key = "manifest:arena-a:elo-test"
    manifest = {
        "tournament_id": "arena-a",
        "rating_run_id": "elo-test",
        "active": True,
        "checkpoint_refs": ["a", "b", "c"],
        "rating_defaults": {"continue_from_latest": True},
    }

    class FakeState:
        def get(self, key, default=None):
            if key == modal_arena.CHECKPOINT_INTAKE_ACTIVE_KEYS:
                return [manifest_key]
            if key == manifest_key:
                return manifest
            return default

    class FakeDrain:
        def spawn(self, _spec):
            raise AssertionError("blocked feedback-loop status must not spawn a drain")

    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "curvytron_checkpoint_intake_drain", FakeDrain())
    monkeypatch.setattr(
        modal_arena,
        "_feedback_loop_status_payload",
        lambda **_kwargs: {
            "status": "rating_game_batch_active",
            "intake": {"new_checkpoints_not_in_latest_rating": 3, "queue_len": 3},
            "flags": ["rating_game_batch_active"],
            "current_game_batch": {"round_id": "round-000123"},
        },
    )
    monkeypatch.setattr(
        modal_arena,
        "_feedback_loop_control_decision",
        lambda _status, *, action: {
            "action": action,
            "spawn_drain": False,
            "reason": "blocked_active_game_batch",
            "new_checkpoints_not_in_latest_rating": 3,
            "queue_len": 3,
        },
    )

    result = modal_arena.curvytron_checkpoint_intake_drain_tick.local()

    assert result["spawned_drain_count"] == 0
    assert result["drain_call_ids"] == []
    assert result["results"][0]["spawned"] is False
    assert result["results"][0]["decision"]["reason"] == "blocked_active_game_batch"


def test_review_battle_payload_reads_game_summaries_and_samples_gifs(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, _rating_run_id = _write_review_fixture(tmp_path)

    payload = modal_arena._review_battle_payload(
        tmp_path,
        tournament_id=tournament_id,
        battle_id="battle-ab",
    )

    assert payload["summary"]["battle_id"] == "battle-ab"
    assert payload["summary"]["result_detail_mode"] == "shard_tally"
    assert payload["game_count"] == 6
    assert payload["game_sources"] == ["game_summary_scan"]
    assert [game["seed"] for game in payload["games"][:2]] == [1000, 1001]
    assert payload["games"][3]["outcome"] == "seat_1_win"
    assert payload["games"][3]["physical_steps"] == 13
    assert payload["sample_gif_count"] == 5
    assert payload["sample_gifs"][0]["gif_ref"].endswith("game-000000/game.gif")
    assert payload["sample_gifs"][-1]["gif_ref"].endswith("game-000004/game.gif")


def test_review_battle_payload_pages_game_summaries(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, _rating_run_id = _write_review_fixture(tmp_path)

    payload = modal_arena._review_battle_payload(
        tmp_path,
        tournament_id=tournament_id,
        battle_id="battle-ab",
        game_limit=2,
        game_offset=2,
    )

    assert payload["game_count"] == 6
    assert payload["game_rows_returned"] == 2
    assert payload["game_limit"] == 2
    assert payload["game_offset"] == 2
    assert payload["has_newer_games"] is True
    assert payload["has_older_games"] is True
    assert [game["seed"] for game in payload["games"]] == [1002, 1003]


def test_review_battle_payload_prefers_summary_gif_samples(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, _rating_run_id = _write_review_fixture(tmp_path)
    battle_ref = arena.battle_summary_ref(tournament_id, "battle-ab")
    summary = modal_arena._read_json(tmp_path / battle_ref)
    summary["sample_gif_refs"] = [
        "tournaments/curvytron/arena-a/battles/battle-ab/games/sample-a/game.gif",
        "tournaments/curvytron/arena-a/battles/battle-ab/games/sample-b/game.gif",
    ]
    modal_arena.runs.write_json(tmp_path / battle_ref, summary)

    payload = modal_arena._review_battle_payload(
        tmp_path,
        tournament_id=tournament_id,
        battle_id="battle-ab",
        game_limit=1,
    )

    assert payload["sample_gif_count"] == 2
    assert [sample["gif_ref"] for sample in payload["sample_gifs"]] == summary["sample_gif_refs"]
    assert payload["games"][0]["gif_ref"].endswith("game-000000/game.gif")


def test_review_battle_payload_samples_game_summary_refs_beyond_game_page(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, _rating_run_id = _write_review_fixture(tmp_path)
    battle_ref = arena.battle_summary_ref(tournament_id, "battle-ab")
    summary = modal_arena._read_json(tmp_path / battle_ref)
    summary["game_summary_refs"] = [
        arena.game_summary_ref(
            tournament_id,
            "battle-ab",
            f"game-{index:06d}",
        ).as_posix()
        for index in range(5)
    ]
    modal_arena.runs.write_json(tmp_path / battle_ref, summary)

    payload = modal_arena._review_battle_payload(
        tmp_path,
        tournament_id=tournament_id,
        battle_id="battle-ab",
        gif_sample_limit=10,
        game_limit=1,
    )

    assert payload["game_rows_returned"] == 1
    assert payload["sample_gif_count"] == 5
    assert payload["sample_gifs"][0]["gif_ref"].endswith("game-000000/game.gif")
    assert payload["sample_gifs"][-1]["gif_ref"].endswith("game-000004/game.gif")


def test_render_battle_page_lists_games_and_gif_samples(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id, rating_run_id = _write_review_fixture(tmp_path)
    payload = modal_arena._review_battle_payload(
        tmp_path,
        tournament_id=tournament_id,
        battle_id="battle-ab",
    )

    html = modal_arena._render_battle_page(
        payload=payload,
        rating_run_id=rating_run_id,
        checkpoint_id="ckpt-a",
    )

    assert "GIF Samples" in html
    assert "game-000003" in html
    assert "seat_1_win" in html
    assert (
        "/gif?ref=tournaments/curvytron/arena-a/battles/battle-ab/games/game-000004/game.gif"
        in html
    )
    assert (
        "/meta?ref=tournaments/curvytron/arena-a/battles/battle-ab/games/game-000005/summary.json"
        in html
    )


def test_slim_rating_snapshot_removes_pair_details() -> None:
    snapshot = {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "ratings": [{"checkpoint_id": "a", "rating": 1500.0}],
        "pair_rating_results": [{"battle_id": "pair-a"}],
        "pair_rating_results_ref": "tournaments/curvytron/a/ratings/elo/rounds/r/results.json",
    }

    slim = modal_arena._slim_rating_snapshot(snapshot)

    assert "pair_rating_results" not in slim
    assert slim["pair_rating_results_ref"].endswith("/results.json")


def test_checkpoint_count_guard_rejects_incomplete_prefix_discovery() -> None:
    with pytest.raises(ValueError, match="checkpoint discovery incomplete"):
        modal_arena._assert_checkpoint_count(
            refs=["a", "b"],
            discovery={"found_count": 2, "missing_count": 0},
            max_runs=3,
        )

    modal_arena._assert_checkpoint_count(
        refs=["a", "b"],
        discovery={"found_count": 2, "missing_count": 0},
        max_runs=3,
        allow_missing_checkpoints=True,
    )


def test_checkpoint_discovery_finds_latest_real_lightzero_weight(tmp_path) -> None:
    run_id = "survivaldiag-v1b-20260513h-001-test"
    attempt_id = "attempt-a"
    latest_attempt = tmp_path / modal_arena.runs.latest_attempt_ref(
        modal_arena.TRAINING_TASK_ID,
        run_id,
    )
    latest_attempt.parent.mkdir(parents=True)
    latest_attempt.write_text(json.dumps({"attempt_id": attempt_id}), encoding="utf-8")
    ckpt_dir = (
        tmp_path
        / modal_arena.runs.attempt_train_ref(
            modal_arena.TRAINING_TASK_ID,
            run_id,
            attempt_id,
        )
        / "lightzero_exp"
        / "ckpt"
    )
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "iteration_5000.pth.tar").write_bytes(b"old")
    latest_weight = ckpt_dir / "iteration_10000.pth.tar"
    latest_weight.write_bytes(b"new")
    sidecar = latest_weight.with_name("iteration_10000.pth.tar.metadata.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_id": arena.CHECKPOINT_POLICY_METADATA_SIDECAR_SCHEMA_ID,
                "policy_trail_render_mode": "browser_lines",
                "policy_bonus_render_mode": "simple_symbols",
                "policy_observation_backend": "cpu_oracle",
                "model_env_variant": "source_state_fixed_opponent",
                "model_reward_variant": "survival_plus_bonus_no_outcome",
            }
        ),
        encoding="utf-8",
    )
    resume_dir = (
        tmp_path
        / modal_arena.runs.checkpoints_root_ref(modal_arena.TRAINING_TASK_ID, run_id)
        / "lightzero_resume_state"
    )
    resume_dir.mkdir(parents=True)
    (resume_dir / "iteration_20000.resume_state.pkl").write_bytes(b"not-a-weight-file")

    discovery = modal_arena._discover_latest_checkpoint_refs(
        tmp_path,
        run_id_prefix="survivaldiag-v1b-20260513h",
    )

    assert discovery["found_count"] == 1
    assert discovery["missing_count"] == 0
    assert discovery["rows"][0]["iteration"] == 10000
    assert discovery["rows"][0]["checkpoint_metadata_ref"].endswith(
        "iteration_10000.pth.tar.metadata.json"
    )
    assert discovery["rows"][0]["policy_trail_render_mode"] == "browser_lines"
    assert discovery["rows"][0]["policy_bonus_render_mode"] == "simple_symbols"
    assert discovery["rows"][0]["policy_observation_backend"] == "cpu_oracle"
    assert discovery["rows"][0]["model_env_variant"] == "source_state_fixed_opponent"
    assert discovery["rows"][0]["model_reward_variant"] == "survival_plus_bonus_no_outcome"
    assert discovery["checkpoint_refs"] == [
        (
            "training/lightzero-curvytron-visual-survival/"
            f"{run_id}/attempts/{attempt_id}/train/lightzero_exp/ckpt/"
            "iteration_10000.pth.tar"
        )
    ]


def test_intake_rating_checkpoints_preserve_discovered_policy_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(modal_arena, "RUNS_MOUNT", tmp_path)
    ref = (
        "training/lightzero-curvytron-visual-survival/run-a/attempts/attempt-a/"
        "train/lightzero_exp/ckpt/iteration_10000.pth.tar"
    )
    checkpoint_path = tmp_path / ref
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    sidecar_path = checkpoint_path.with_name("iteration_10000.pth.tar.metadata.json")
    sidecar_path.write_text(
        json.dumps(
            {
                "schema_id": arena.CHECKPOINT_POLICY_METADATA_SIDECAR_SCHEMA_ID,
                "policy_observation_contract_id": "surface/custom",
                "observation_contract": {
                    "trail_render_mode": "browser_lines",
                    "bonus_render_mode": "simple_symbols",
                    "backend": "cpu_oracle",
                },
                "policy_trail_render_mode": "browser_lines",
                "policy_bonus_render_mode": "simple_symbols",
                "policy_observation_backend": "cpu_oracle",
                "model_env_variant": "source_state_fixed_opponent",
                "model_reward_variant": "survival_plus_bonus_no_outcome",
                "decision_source_frames": 1,
                "source_physics_step_ms": 1000.0 / 60.0,
                "learner_seat_mode": "random_per_episode",
            }
        ),
        encoding="utf-8",
    )
    discovery_row = modal_arena._checkpoint_discovery_row_from_ref(
        ref,
        mount=tmp_path,
        found=True,
    )
    manifest = {
        "checkpoint_refs": [ref],
        "discovery": {"rows": [discovery_row]},
    }

    checkpoints = modal_arena._intake_manifest_rating_checkpoints(
        manifest,
        continue_from_latest=False,
    )

    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    assert checkpoint["checkpoint_ref"] == ref
    assert checkpoint["run_id"] == "run-a"
    assert checkpoint["attempt_id"] == "attempt-a"
    assert checkpoint["iteration"] == 10000
    assert checkpoint["checkpoint_size_bytes"] == len(b"checkpoint")
    assert checkpoint["policy_observation_contract_id"] == "surface/custom"
    assert checkpoint["observation_contract"] == {
        "trail_render_mode": "browser_lines",
        "bonus_render_mode": "simple_symbols",
        "backend": "cpu_oracle",
    }
    assert checkpoint["policy_trail_render_mode"] == "browser_lines"
    assert checkpoint["policy_bonus_render_mode"] == "simple_symbols"
    assert checkpoint["policy_observation_backend"] == "cpu_oracle"
    assert checkpoint["model_env_variant"] == "source_state_fixed_opponent"
    assert checkpoint["model_reward_variant"] == "survival_plus_bonus_no_outcome"
    assert checkpoint["decision_source_frames"] == 1
    assert checkpoint["source_physics_step_ms"] == 1000.0 / 60.0
    assert checkpoint["learner_seat_mode"] == "random_per_episode"
    assert checkpoint["checkpoint_metadata_ref"] == f"{ref}.metadata.json"


def test_checkpoint_discovery_scans_timestamped_lightzero_exp_dirs(tmp_path) -> None:
    run_id = "curvy-mix2clean-r50-example"
    attempt_id = "attempt-a"
    latest_attempt = tmp_path / modal_arena.runs.latest_attempt_ref(
        modal_arena.TRAINING_TASK_ID,
        run_id,
    )
    latest_attempt.parent.mkdir(parents=True)
    latest_attempt.write_text(json.dumps({"attempt_id": attempt_id}), encoding="utf-8")
    train_root = tmp_path / modal_arena.runs.attempt_train_ref(
        modal_arena.TRAINING_TASK_ID,
        run_id,
        attempt_id,
    )
    fixed = train_root / "lightzero_exp" / "ckpt" / "iteration_0.pth.tar"
    timestamped = train_root / "lightzero_exp_260513_123802" / "ckpt" / "iteration_180000.pth.tar"
    fixed.parent.mkdir(parents=True)
    timestamped.parent.mkdir(parents=True)
    fixed.write_bytes(b"stale")
    timestamped.write_bytes(b"fresh")

    discovery = modal_arena._discover_checkpoint_refs(
        tmp_path,
        run_ids=[run_id],
    )

    assert discovery["checkpoint_scan_glob"] == "train/lightzero_exp*/ckpt/iteration_*.pth.tar"
    assert discovery["found_count"] == 1
    assert discovery["rows"][0]["iteration"] == 180000
    assert discovery["rows"][0]["exp_dir_name"] == "lightzero_exp_260513_123802"
    assert (
        "lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar"
        in discovery["checkpoint_refs"][0]
    )


def test_checkpoint_discovery_iteration_filter_scans_timestamped_dirs(tmp_path) -> None:
    run_id = "curvy-mix2clean-r50-example"
    attempt_id = "attempt-a"
    latest_attempt = tmp_path / modal_arena.runs.latest_attempt_ref(
        modal_arena.TRAINING_TASK_ID,
        run_id,
    )
    latest_attempt.parent.mkdir(parents=True)
    latest_attempt.write_text(json.dumps({"attempt_id": attempt_id}), encoding="utf-8")
    train_root = tmp_path / modal_arena.runs.attempt_train_ref(
        modal_arena.TRAINING_TASK_ID,
        run_id,
        attempt_id,
    )
    for iteration in (0, 40000, 180000):
        ckpt = (
            train_root / "lightzero_exp_260513_123802" / "ckpt" / f"iteration_{iteration}.pth.tar"
        )
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"weights")

    discovery = modal_arena._discover_checkpoint_refs(
        tmp_path,
        run_ids=[run_id],
        checkpoint_selection="iteration",
        checkpoint_iteration=40000,
    )

    assert discovery["checkpoint_selection"] == "iteration"
    assert discovery["found_count"] == 1
    assert discovery["rows"][0]["iteration"] == 40000
    assert (
        "lightzero_exp_260513_123802/ckpt/iteration_40000.pth.tar"
        in discovery["checkpoint_refs"][0]
    )


def test_checkpoint_discovery_all_returns_all_nonempty_weight_files(tmp_path) -> None:
    run_id = "curvy-mix2clean-r50-example"
    attempt_id = "attempt-a"
    latest_attempt = tmp_path / modal_arena.runs.latest_attempt_ref(
        modal_arena.TRAINING_TASK_ID,
        run_id,
    )
    latest_attempt.parent.mkdir(parents=True)
    latest_attempt.write_text(json.dumps({"attempt_id": attempt_id}), encoding="utf-8")
    train_root = tmp_path / modal_arena.runs.attempt_train_ref(
        modal_arena.TRAINING_TASK_ID,
        run_id,
        attempt_id,
    )
    fixed = train_root / "lightzero_exp" / "ckpt" / "iteration_0.pth.tar"
    timestamped = train_root / "lightzero_exp_260513_123802" / "ckpt" / "iteration_180000.pth.tar"
    empty = train_root / "lightzero_exp_260513_123802" / "ckpt" / "iteration_190000.pth.tar"
    resume = (
        tmp_path
        / modal_arena.runs.checkpoints_root_ref(modal_arena.TRAINING_TASK_ID, run_id)
        / "lightzero_resume_state"
        / "iteration_200000.resume_state.pkl"
    )
    fixed.parent.mkdir(parents=True)
    timestamped.parent.mkdir(parents=True)
    resume.parent.mkdir(parents=True)
    fixed.write_bytes(b"fixed")
    timestamped.write_bytes(b"timestamped")
    empty.write_bytes(b"")
    resume.write_bytes(b"resume")

    discovery = modal_arena._discover_checkpoint_refs(
        tmp_path,
        run_ids=[run_id],
        checkpoint_selection="all",
    )

    assert discovery["checkpoint_selection"] == "all"
    assert discovery["found_count"] == 2
    assert discovery["found_run_count"] == 1
    assert [row["iteration"] for row in discovery["rows"]] == [0, 180000]
    assert {row["exp_dir_name"] for row in discovery["rows"]} == {
        "lightzero_exp",
        "lightzero_exp_260513_123802",
    }
    assert all(not ref.endswith("resume_state.pkl") for ref in discovery["checkpoint_refs"])


def test_checkpoint_count_guard_does_not_treat_max_runs_as_checkpoint_count_for_all_selection() -> (
    None
):
    modal_arena._assert_checkpoint_count(
        refs=["a", "b", "c", "d"],
        discovery={
            "checkpoint_selection": "all",
            "found_count": 4,
            "missing_count": 0,
            "selected_run_count": 2,
        },
        max_runs=2,
    )


def test_checkpoint_discovery_prefix_limit_selects_most_recent_checkpoints(tmp_path) -> None:
    def write_checkpoint(run_id: str, iteration: int, mtime_ns: int) -> None:
        attempt_id = f"attempt-{run_id}"
        latest_attempt = tmp_path / modal_arena.runs.latest_attempt_ref(
            modal_arena.TRAINING_TASK_ID,
            run_id,
        )
        latest_attempt.parent.mkdir(parents=True)
        latest_attempt.write_text(json.dumps({"attempt_id": attempt_id}), encoding="utf-8")
        ckpt = (
            tmp_path
            / modal_arena.runs.attempt_train_ref(
                modal_arena.TRAINING_TASK_ID,
                run_id,
                attempt_id,
            )
            / "lightzero_exp"
            / "ckpt"
            / f"iteration_{iteration}.pth.tar"
        )
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"weights")
        os.utime(ckpt, ns=(mtime_ns, mtime_ns))

    write_checkpoint("prefix-a", iteration=100, mtime_ns=100)
    write_checkpoint("prefix-b", iteration=50, mtime_ns=300)
    write_checkpoint("prefix-c", iteration=200, mtime_ns=200)

    discovery = modal_arena._discover_latest_checkpoint_refs(
        tmp_path,
        run_id_prefix="prefix",
        max_runs=2,
    )

    assert discovery["selection"] == "latest_checkpoint_mtime"
    assert discovery["requested_run_count"] == 3
    assert discovery["selected_run_count"] == 2
    assert [row["run_id"] for row in discovery["rows"]] == ["prefix-b", "prefix-c"]
    assert discovery["found_count"] == 2


def test_rating_progress_scans_committed_game_summaries(tmp_path) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
        }
    )
    round_id = arena.rating_round_id(0)
    pairs = arena.build_rating_round_pair_specs(rating_spec)
    input_path = tmp_path / arena.rating_round_input_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    modal_arena.runs.write_json(
        input_path,
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 0,
            "started_at": "2026-05-13T00:00:00Z",
            "rating_spec": rating_spec,
            "pair_count": len(pairs),
            "game_count": 3,
            "pair_specs": pairs,
        },
    )
    pair = pairs[0]
    for index, outcome in enumerate(["seat_0_win", "draw"]):
        game = {
            **_fake_game(pair, index, outcome),
            "tournament_id": tournament_id,
            "battle_id": pair["battle_id"],
            "pair_index": pair["pair_index"],
        }
        modal_arena.runs.write_json(
            tmp_path / arena.game_summary_ref(tournament_id, pair["battle_id"], game["game_id"]),
            game,
        )

    progress, games_by_battle = modal_arena._rating_round_progress_payload(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
    )
    path_only_progress, _ = modal_arena._rating_round_progress_payload(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        load_summaries=False,
    )
    pair_only_progress, _ = modal_arena._rating_round_progress_payload(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        load_summaries=False,
        pair_only=True,
    )

    assert progress["schema_id"] == arena.RATING_PROGRESS_SCHEMA_ID
    assert progress["completed_game_count"] == 2
    assert progress["game_count"] == 3
    assert progress["started_pair_count"] == 1
    assert progress["partial_pair_count"] == 1
    assert progress["completed_pair_count"] == 0
    assert progress["status"] == "running"
    assert list(games_by_battle) == [pair["battle_id"]]
    assert path_only_progress["completed_game_count"] == 2
    assert path_only_progress["unknown_result_count"] == 2
    assert path_only_progress["result_counts_known"] is False
    assert pair_only_progress["completed_game_count"] == 2
    assert pair_only_progress["partial_pair_count"] == 1
    assert pair_only_progress["estimated_seen_game_count"] == 2
    assert pair_only_progress["count_basis"] == "shard_and_game_summary_files"
    assert pair_only_progress["result_counts_known"] is False

    modal_arena.runs.write_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id),
        {
            "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "rated_pair_count": 1,
            "max_abs_delta": 0.0,
            "stable": True,
        },
    )
    complete_pair_only_progress, _ = modal_arena._rating_round_progress_payload(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        load_summaries=False,
        pair_only=True,
    )

    assert complete_pair_only_progress["status"] == "complete"
    assert complete_pair_only_progress["phase"] == "ratings_written"
    assert complete_pair_only_progress["completed_game_count"] == 3
    assert complete_pair_only_progress["completed_pair_count"] == 1
    assert complete_pair_only_progress["completion_fraction"] == 1.0
    assert complete_pair_only_progress["count_basis"] == "latest_snapshot"
    assert complete_pair_only_progress["recent_started_pairs"] == []


def test_live_rating_progress_counts_started_battle_dirs(tmp_path) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
        }
    )
    round_id = arena.rating_round_id(0)
    pairs = arena.build_rating_round_pair_specs(rating_spec)
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 0,
            "started_at": "2026-05-13T00:00:00Z",
            "rating_spec": rating_spec,
            "pair_count": len(pairs),
            "game_count": 3,
            "pair_specs": pairs,
        },
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_progress_ref(tournament_id, rating_run_id),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 0,
            "status": "running",
            "phase": "game_map_started",
            "pair_count": len(pairs),
            "game_count": 3,
            "started_pair_count": 0,
            "estimated_seen_game_count": None,
        },
    )
    battle_dir = tmp_path / arena.battle_root_ref(tournament_id, pairs[0]["battle_id"])
    battle_dir.mkdir(parents=True)

    progress = modal_arena._read_live_rating_progress(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

    assert progress["count_basis"] == "shard_and_game_summary_files"
    assert progress["phase"] == "games_running"
    assert progress["started_pair_count"] == 1
    assert progress["completed_pair_count"] == 0
    assert progress["estimated_seen_game_count"] == 0
    assert progress["estimated_completion_fraction"] == 0.0


def test_live_rating_progress_counts_completed_shard_summaries(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    modal_arena._WEB_CACHE.clear()
    tournament_id, rating_run_id, _spec, _pair = _write_live_rating_fixture(tmp_path)

    progress = modal_arena._read_live_rating_progress(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

    assert progress["count_basis"] == "shard_summary_files"
    assert progress["started_pair_count"] == 1
    assert progress["completed_pair_count"] == 1
    assert progress["completed_game_count"] == 3
    assert progress["recent_started_pairs"][0]["seen_game_count"] == 3
    assert progress["recent_started_pairs"][0]["complete"] is True


def test_rating_reduce_rebuilds_latest_from_game_summaries(tmp_path) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
        }
    )
    round_id = arena.rating_round_id(0)
    pairs = arena.build_rating_round_pair_specs(rating_spec)
    input_path = tmp_path / arena.rating_round_input_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    modal_arena.runs.write_json(
        input_path,
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 0,
            "started_at": "2026-05-13T00:00:00Z",
            "rating_spec": rating_spec,
            "pair_count": len(pairs),
            "game_count": 3,
            "pair_specs": pairs,
        },
    )
    pair = pairs[0]
    for index, outcome in enumerate(["seat_0_win", "seat_0_win", "seat_0_win"]):
        game = {
            **_fake_game(pair, index, outcome),
            "tournament_id": tournament_id,
            "battle_id": pair["battle_id"],
            "pair_index": pair["pair_index"],
        }
        modal_arena.runs.write_json(
            tmp_path / arena.game_summary_ref(tournament_id, pair["battle_id"], game["game_id"]),
            game,
        )

    result = modal_arena._reduce_rating_round_from_summaries(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
    )
    latest = modal_arena._read_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id)
    )
    results = modal_arena._read_json(
        tmp_path / arena.rating_round_results_ref(tournament_id, rating_run_id, round_id)
    )
    progress = modal_arena._read_json(
        tmp_path / arena.rating_progress_ref(tournament_id, rating_run_id)
    )

    assert result["snapshot"]["rated_pair_count"] == 1
    assert latest["ratings"][0]["rating"] > arena.DEFAULT_RATING_INITIAL_RATING
    assert "pair_rating_results" not in latest
    assert results["pair_rating_results"][0]["valid_games"] == 3
    assert progress["status"] == "complete"
    lineage_rows = [
        json.loads(line)
        for line in (
            tmp_path
            / arena.rating_root_ref(tournament_id, rating_run_id)
            / "feedback_loop"
            / "lineage_events.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert lineage_rows[-1]["stage"] == "rating_latest_written"
    assert lineage_rows[-1]["status"] == "ok"
    assert (
        lineage_rows[-1]["latest_ref"]
        == arena.rating_latest_ref(
            tournament_id,
            rating_run_id,
        ).as_posix()
    )
    assert lineage_rows[-1]["round_index"] == 0


def test_rating_round_resumes_existing_input_by_reducing_game_summaries(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "_commit_volume", lambda *_args, **_kwargs: None)
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
        }
    )
    round_id = arena.rating_round_id(0)
    pairs = arena.build_rating_round_pair_specs(rating_spec)
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "pool_hash": arena.rating_pool_hash(rating_spec["checkpoints"]),
            "roster_hash": arena.rating_pool_hash(rating_spec["checkpoints"]),
            "context_hash": arena.rating_context_hash(rating_spec),
            "checkpoint_roster": arena.rating_roster_by_checkpoint(rating_spec["checkpoints"]),
            "round_id": round_id,
            "round_index": 0,
            "started_at": "2026-05-13T00:00:00Z",
            "rating_spec": rating_spec,
            "pair_count": len(pairs),
            "game_count": 3,
            "pair_specs": pairs,
        },
    )
    pair = pairs[0]
    for index, outcome in enumerate(["seat_0_win", "seat_0_win", "seat_1_win"]):
        game = {
            **_fake_game(pair, index, outcome),
            "tournament_id": tournament_id,
            "battle_id": pair["battle_id"],
            "pair_index": pair["pair_index"],
        }
        modal_arena.runs.write_json(
            tmp_path / arena.game_summary_ref(tournament_id, pair["battle_id"], game["game_id"]),
            game,
        )

    result = modal_arena.curvytron_rating_round.local({**rating_spec, "round_index": 0})

    latest = modal_arena._read_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id)
    )
    assert result["resumed_existing_round"] is True
    assert result["work_summary"]["source"] == "existing_input_reduce"
    assert result["snapshot"]["rated_pair_count"] == 1
    assert result["pair_count"] == 1
    assert latest["round_id"] == round_id


def test_rating_round_existing_incomplete_input_returns_running_status(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
            "max_steps": 64,
        }
    )
    round_id = arena.rating_round_id(0)
    pairs = arena.build_rating_round_pair_specs(rating_spec)
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "pool_hash": arena.rating_pool_hash(rating_spec["checkpoints"]),
            "roster_hash": arena.rating_pool_hash(rating_spec["checkpoints"]),
            "context_hash": arena.rating_context_hash(rating_spec),
            "checkpoint_roster": arena.rating_roster_by_checkpoint(rating_spec["checkpoints"]),
            "round_id": round_id,
            "round_index": 0,
            "started_at": "2026-05-13T00:00:00Z",
            "rating_spec": rating_spec,
            "pair_count": len(pairs),
            "game_count": 3,
            "pair_specs": pairs,
        },
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_progress_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 0,
            "status": "running",
            "phase": "game_map_started",
            "pair_count": len(pairs),
            "game_count": 3,
            "completed_pair_count": 0,
            "completed_game_count": 0,
        },
    )

    result = modal_arena.curvytron_rating_round.local({**rating_spec, "round_index": 0})

    assert result["status"] == "running_existing_round"
    assert result["phase"] == "existing_input_incomplete"
    assert result["pair_count"] == 1
    assert result["game_count"] == 3
    assert result["rated_pair_count"] == 0


def test_rating_round_existing_different_input_returns_running_status(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    existing_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
            "max_steps": 64,
        }
    )
    requested_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [
                _checkpoint_ref("run-a", 0),
                _checkpoint_ref("run-b", 10),
                _checkpoint_ref("run-c", 20),
            ],
            "games_per_pair": 3,
            "max_steps": 64,
        }
    )
    round_id = arena.rating_round_id(0)
    pairs = arena.build_rating_round_pair_specs(existing_spec)
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "pool_hash": arena.rating_pool_hash(existing_spec["checkpoints"]),
            "roster_hash": arena.rating_pool_hash(existing_spec["checkpoints"]),
            "context_hash": arena.rating_context_hash(existing_spec),
            "checkpoint_roster": arena.rating_roster_by_checkpoint(existing_spec["checkpoints"]),
            "round_id": round_id,
            "round_index": 0,
            "started_at": "2026-05-13T00:00:00Z",
            "rating_spec": existing_spec,
            "pair_count": len(pairs),
            "game_count": 3,
            "pair_specs": pairs,
        },
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_progress_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 0,
            "status": "running",
            "phase": "game_map_started",
            "pair_count": len(pairs),
            "game_count": 3,
            "completed_pair_count": 0,
            "completed_game_count": 0,
        },
    )

    result = modal_arena.curvytron_rating_round.local({**requested_spec, "round_index": 0})

    assert result["status"] == "running_existing_round"
    assert result["phase"] == "existing_input_different_spec"
    assert result["pair_count"] == 1
    assert result["game_count"] == 3
    assert result["rated_pair_count"] == 0
    assert result["work_summary"]["source"] == "existing_input_different_spec"


def test_rating_round_skip_decision_keeps_newer_zero_output_different_spec_until_stale(
    tmp_path,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    existing_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
            "max_steps": 64,
        }
    )
    desired_spec = arena.normalize_rating_spec(
        {
            **existing_spec,
            "checkpoints": [
                _checkpoint_ref("run-a", 0),
                _checkpoint_ref("run-b", 10),
                _checkpoint_ref("run-c", 20),
            ],
        }
    )
    round_id = arena.rating_round_id(4)
    pairs = arena.build_rating_round_pair_specs(existing_spec)
    input_path = tmp_path / arena.rating_round_input_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    progress_path = tmp_path / arena.rating_round_progress_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    modal_arena.runs.write_json(
        input_path,
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "pool_hash": arena.rating_pool_hash(existing_spec["checkpoints"]),
            "roster_hash": arena.rating_pool_hash(existing_spec["checkpoints"]),
            "context_hash": arena.rating_context_hash(existing_spec),
            "checkpoint_roster": arena.rating_roster_by_checkpoint(existing_spec["checkpoints"]),
            "round_id": round_id,
            "round_index": 4,
            "rating_spec": existing_spec,
            "pair_count": len(pairs),
            "game_count": 3,
            "pair_specs": pairs,
        },
    )
    modal_arena.runs.write_json(
        progress_path,
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 4,
            "status": "running",
            "phase": "game_map_started",
            "pair_count": len(pairs),
            "game_count": 3,
            "completed_pair_count": 0,
            "completed_game_count": 0,
        },
    )

    decision = modal_arena._rating_round_skip_decision(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        desired_checkpoint_count=3,
        desired_rating_spec=desired_spec,
        stale_after_seconds=24 * 60 * 60,
        scan_output_progress=False,
    )

    assert decision["skip"] is False
    assert decision["reason"] == "not_skippable"
    assert decision["different_spec"] is True
    assert "different spec" in decision["different_spec_error"]

    old_mtime = os.path.getmtime(input_path) - 120.0
    os.utime(input_path, (old_mtime, old_mtime))
    os.utime(progress_path, (old_mtime, old_mtime))

    decision = modal_arena._rating_round_skip_decision(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        desired_checkpoint_count=3,
        desired_rating_spec=desired_spec,
        stale_after_seconds=60,
        scan_output_progress=False,
    )

    assert decision["skip"] is True
    assert decision["reason"] == "different_spec_zero_output"
    assert decision["different_spec"] is True
    assert "different spec" in decision["different_spec_error"]


def test_rating_round_skip_decision_counts_game_summaries_before_skip(
    tmp_path,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    existing_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
            "games_per_shard": 3,
            "max_steps": 64,
        }
    )
    round_id = arena.rating_round_id(5)
    pairs = arena.build_rating_round_pair_specs(existing_spec)
    pair = pairs[0]
    input_path = tmp_path / arena.rating_round_input_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    progress_path = tmp_path / arena.rating_round_progress_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    modal_arena.runs.write_json(
        input_path,
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "pool_hash": arena.rating_pool_hash(existing_spec["checkpoints"]),
            "roster_hash": arena.rating_pool_hash(existing_spec["checkpoints"]),
            "context_hash": arena.rating_context_hash(existing_spec),
            "checkpoint_roster": arena.rating_roster_by_checkpoint(
                existing_spec["checkpoints"]
            ),
            "round_id": round_id,
            "round_index": 5,
            "rating_spec": existing_spec,
            "pair_count": len(pairs),
            "game_count": 3,
            "pair_specs": pairs,
        },
    )
    modal_arena.runs.write_json(
        progress_path,
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 5,
            "status": "running",
            "phase": "game_map_started",
            "pair_count": len(pairs),
            "game_count": 3,
            "completed_pair_count": 0,
            "completed_game_count": 0,
        },
    )
    old_ts = max(0, int(input_path.stat().st_mtime) - 700)
    os.utime(input_path, (old_ts, old_ts))
    os.utime(progress_path, (old_ts, old_ts))
    modal_arena.runs.write_json(
        tmp_path
        / arena.game_summary_ref(
            tournament_id,
            pair["battle_id"],
            "game-000000",
        ),
        {
            "schema_id": arena.GAME_SCHEMA_ID,
            "tournament_id": tournament_id,
            "battle_id": pair["battle_id"],
            "game_id": "game-000000",
            "pair_index": pair["pair_index"],
            "ok": True,
        },
    )

    decision = modal_arena._rating_round_skip_decision(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        desired_checkpoint_count=3,
        desired_rating_spec={**existing_spec, "checkpoints": [
            *existing_spec["checkpoints"],
            _checkpoint_ref("run-c", 20),
        ]},
        stale_after_seconds=600,
        scan_output_progress=True,
    )

    assert decision["skip"] is False
    assert decision["reason"] == "not_skippable"
    assert decision["completed_game_count"] == 1
    assert decision["started_pair_count"] == 1
    assert decision["latest_result_ts"] is not None
    assert decision["is_stale"] is False


def test_rating_round_skip_decision_reloads_tournament_volume_before_output_scan(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    reload_calls = []
    monkeypatch.setattr(
        modal_arena,
        "_reload_volume",
        lambda volume: reload_calls.append(volume),
    )
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
            "max_steps": 64,
        }
    )
    round_id = arena.rating_round_id(6)
    pairs = arena.build_rating_round_pair_specs(rating_spec)
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 6,
            "rating_spec": rating_spec,
            "pair_count": len(pairs),
            "game_count": 3,
            "pair_specs": pairs,
        },
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_progress_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 6,
            "status": "running",
            "phase": "game_map_started",
            "pair_count": len(pairs),
            "game_count": 3,
            "completed_game_count": 0,
        },
    )

    modal_arena._rating_round_skip_decision(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        desired_checkpoint_count=2,
        desired_rating_spec=rating_spec,
        stale_after_seconds=24 * 60 * 60,
        scan_output_progress=True,
    )

    assert reload_calls == [modal_arena.tournament_volume]


def test_rating_round_existing_skipped_input_returns_skipped_status(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
            "max_steps": 64,
        }
    )
    round_id = arena.rating_round_id(0)
    pairs = arena.build_rating_round_pair_specs(rating_spec)
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "pool_hash": arena.rating_pool_hash(rating_spec["checkpoints"]),
            "roster_hash": arena.rating_pool_hash(rating_spec["checkpoints"]),
            "context_hash": arena.rating_context_hash(rating_spec),
            "checkpoint_roster": arena.rating_roster_by_checkpoint(rating_spec["checkpoints"]),
            "round_id": round_id,
            "round_index": 0,
            "started_at": "2026-05-13T00:00:00Z",
            "rating_spec": rating_spec,
            "pair_count": len(pairs),
            "game_count": 3,
            "pair_specs": pairs,
        },
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_progress_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 0,
            "status": "skipped",
            "phase": "stale_orphan_round_skipped",
            "pair_count": len(pairs),
            "game_count": 3,
            "completed_pair_count": 0,
            "completed_game_count": 0,
        },
    )

    result = modal_arena.curvytron_rating_round.local({**rating_spec, "round_index": 0})

    assert result["status"] == "skipped"
    assert result["phase"] == "stale_orphan_round_skipped"
    assert result["work_summary"]["source"] == "existing_input_skipped"
    assert result["rated_pair_count"] == 0


def test_feedback_loop_status_is_compact_and_flags_active_zero_started_batch() -> None:
    result = modal_arena._feedback_loop_status_from_state(
        tournament_id="cz26-live",
        rating_run_id="elo-cz26-live",
        manifest={
            "active": True,
            "checkpoint_count": 545,
            "queued_checkpoint_count": 545,
            "queue_partition": "q:cz26",
            "rating_defaults": {
                "pair_selection": "adaptive_v0",
                "pairs_per_round": 300,
                "active_pool_limit": 100,
                "games_per_pair": 21,
                "save_gif": False,
                "continue_from_latest": True,
            },
            "checkpoint_refs": ["should-not-leak"],
        },
        manifest_load={
            "manifest_key": "manifest:cz26-live:elo-cz26-live",
            "manifest_source": "dict",
            "manifest_state_repaired": False,
        },
        active_manifest_keys=["manifest:cz26-live:elo-cz26-live"],
        queue_len=14,
        latest_snapshot={
            "round_id": "round-000007",
            "round_index": 7,
            "checkpoint_count": 414,
            "ratings": [{"status": "active", "iteration": 20000}],
            "stable": False,
            "max_abs_delta": 31.4,
        },
        batch_window=[
            {
                "round_id": "round-000008",
                "round_index": 8,
                "exists": True,
                "ratings_written": False,
                "status": "running",
                "phase": "game_map_started",
                "checkpoint_count": 414,
                "pair_count": 300,
                "game_count": 6300,
                "started_pair_count": 0,
                "completed_pair_count": 0,
                "completed_game_count": 0,
            }
        ],
        trainer_refresh_state={
            "status": "refreshed",
            "generation": 3,
            "snapshot_id": "auto-r000007-g3-test",
            "row_count": 414,
            "active_count": 100,
            "rewritten_pointer_count": 24,
        },
    )

    assert result["status"] == "rating_game_batch_active"
    assert "latest_rating_behind_intake" in result["flags"]
    assert "active_game_batch_zero_started" in result["flags"]
    assert result["intake"]["checkpoint_count"] == 545
    assert result["intake"]["new_checkpoints_not_in_latest_rating"] == 131
    assert result["latest_rating"]["checkpoint_count"] == 414
    assert result["trainer_refresh"]["rewritten_pointer_count"] == 24
    assert "checkpoint_refs" not in result["intake"]


def test_unrated_round_index_scan_finds_far_ahead_blocker(tmp_path) -> None:
    tournament_id = "cz26-live"
    rating_run_id = "elo-cz26-live"
    modal_arena.runs.write_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id),
        {
            "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": "round-000010",
            "round_index": 10,
            "checkpoint_count": 588,
            "ratings": [],
        },
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, "round-000024"),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": "round-000024",
            "round_index": 24,
            "checkpoint_count": 1494,
            "pair_count": 526,
            "game_count": 11046,
            "rating_spec": {
                "pair_selection": "adaptive_v0",
                "pairs_per_round": 300,
                "active_pool_limit": 100,
                "games_per_pair": 21,
            },
        },
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_progress_ref(
            tournament_id,
            rating_run_id,
            "round-000024",
        ),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": "round-000024",
            "round_index": 24,
            "status": "running_existing_round",
            "phase": "existing_input_different_spec",
            "pair_count": 526,
            "game_count": 11046,
            "completed_pair_count": 0,
            "completed_game_count": 0,
        },
    )

    indices = modal_arena._unrated_rating_round_indices(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        after_round_index=10,
    )

    assert indices == [24]


def test_feedback_loop_status_reports_active_game_output_probe() -> None:
    result = modal_arena._feedback_loop_status_from_state(
        tournament_id="cz26-live",
        rating_run_id="elo-cz26-live",
        manifest={
            "active": True,
            "checkpoint_count": 828,
            "queued_checkpoint_count": 828,
            "queue_partition": "q:cz26",
            "rating_defaults": {
                "pair_selection": "adaptive_v0",
                "pairs_per_round": 300,
                "active_pool_limit": 100,
                "games_per_pair": 21,
                "save_gif": False,
                "continue_from_latest": True,
            },
        },
        manifest_load={
            "manifest_key": "manifest:cz26-live:elo-cz26-live",
            "manifest_source": "dict",
            "manifest_state_repaired": False,
        },
        active_manifest_keys=["manifest:cz26-live:elo-cz26-live"],
        queue_len=0,
        latest_snapshot={
            "round_id": "round-000009",
            "round_index": 9,
            "checkpoint_count": 414,
            "ratings": [],
        },
        batch_window=[
            {
                "round_id": "round-000014",
                "round_index": 14,
                "exists": True,
                "ratings_written": False,
                "status": "running",
                "phase": "game_map_started",
                "checkpoint_count": 681,
                "pair_count": 319,
                "game_count": 6699,
                "started_pair_count": 0,
                "completed_pair_count": 0,
                "completed_game_count": 0,
            }
        ],
        trainer_refresh_state={},
        current_batch_recovery_probe={
            "scan_output_progress": True,
            "skip": False,
            "reason": "not_skippable",
            "completed_game_count": 23,
            "started_pair_count": 2,
            "latest_result_ts": 1_779_000_000.0,
            "latest_result_age_seconds": 12.5,
            "progress_scan_error": None,
            "progress_scan_error_blocks_skip": False,
            "stale_after_seconds": 600,
            "is_stale": False,
        },
    )

    assert "active_game_batch_zero_started" in result["flags"]
    assert "active_game_batch_has_game_output" in result["flags"]
    assert result["current_game_batch"]["recovery_probe"]["completed_game_count"] == 23
    assert result["current_game_batch"]["recovery_probe"]["latest_result_age_seconds"] == 12.5
    assert result["operator_next_action"] == (
        "game outputs are landing for the active batch; wait for completion or reduce"
    )


def test_feedback_loop_status_warns_when_active_batch_uses_old_pool() -> None:
    result = modal_arena._feedback_loop_status_from_state(
        tournament_id="cz26-live",
        rating_run_id="elo-cz26-live",
        manifest={
            "active": True,
            "checkpoint_count": 1888,
            "queued_checkpoint_count": 1888,
            "queue_partition": "q:cz26",
            "rating_defaults": {
                "pair_selection": "adaptive_v0",
                "pairs_per_round": 300,
                "active_pool_limit": 100,
                "games_per_pair": 21,
                "save_gif": False,
                "continue_from_latest": True,
            },
        },
        manifest_load={
            "manifest_key": "manifest:cz26-live:elo-cz26-live",
            "manifest_source": "dict",
            "manifest_state_repaired": False,
        },
        active_manifest_keys=["manifest:cz26-live:elo-cz26-live"],
        queue_len=0,
        latest_snapshot={
            "round_id": "round-000015",
            "round_index": 15,
            "checkpoint_count": 919,
            "ratings": [],
        },
        batch_window=[
            {
                "round_id": "round-000029",
                "round_index": 29,
                "exists": True,
                "ratings_written": False,
                "status": "running",
                "phase": "game_map_started",
                "checkpoint_count": 919,
                "pair_count": 438,
                "game_count": 9198,
                "started_pair_count": 1,
                "completed_pair_count": 0,
                "completed_game_count": 0,
            }
        ],
        trainer_refresh_state={},
        current_batch_recovery_probe={
            "scan_output_progress": True,
            "skip": False,
            "reason": "activity_probe",
            "completed_game_count": 2,
            "started_pair_count": 1,
            "latest_result_ts": 1_779_010_390.7052872,
            "latest_result_age_seconds": 52.0,
            "progress_scan_error": None,
            "progress_scan_error_blocks_skip": False,
            "stale_after_seconds": 600,
            "is_stale": False,
        },
    )

    assert result["pool_status"] == {
        "intake_checkpoint_count": 1888,
        "latest_rating_checkpoint_count": 919,
        "active_game_batch_checkpoint_count": 919,
        "new_checkpoints_not_in_latest_rating": 969,
        "active_batch_newer_than_latest": False,
        "active_batch_not_covering_new_checkpoints": True,
        "active_batch_missing_from_intake_count": 969,
    }
    assert "latest_rating_behind_intake" in result["flags"]
    assert "active_game_batch_has_game_output" in result["flags"]
    assert "active_game_batch_not_covering_new_checkpoints" in result["flags"]
    assert result["current_game_batch"]["pool_status"] == result["pool_status"]
    assert result["operator_next_action"] == (
        "active game batch is running but appears to cover only the already-rated "
        "pool; do not call catch-up validated, let it finish/recover, then start "
        "a full-pool drain with spec-count proof"
    )


def test_feedback_loop_status_flags_multiple_active_game_batches() -> None:
    result = modal_arena._feedback_loop_status_from_state(
        tournament_id="cz26-live",
        rating_run_id="elo-cz26-live",
        manifest={
            "active": True,
            "checkpoint_count": 1088,
            "queued_checkpoint_count": 1088,
            "queue_partition": "q:cz26",
            "rating_defaults": {
                "pair_selection": "adaptive_v0",
                "pairs_per_round": 300,
                "active_pool_limit": 100,
                "games_per_pair": 21,
                "save_gif": False,
                "continue_from_latest": True,
            },
        },
        manifest_load={
            "manifest_key": "manifest:cz26-live:elo-cz26-live",
            "manifest_source": "dict",
            "manifest_state_repaired": False,
        },
        active_manifest_keys=["manifest:cz26-live:elo-cz26-live"],
        queue_len=211,
        latest_snapshot={
            "round_id": "round-000011",
            "round_index": 11,
            "checkpoint_count": 681,
            "ratings": [],
        },
        batch_window=[
            {
                "round_id": "round-000016",
                "round_index": 16,
                "exists": True,
                "ratings_written": False,
                "status": "running",
                "phase": "input_written",
                "checkpoint_count": 960,
                "pair_count": 459,
                "game_count": 9639,
                "started_pair_count": 0,
                "completed_pair_count": 0,
                "completed_game_count": 0,
            },
            {
                "round_id": "round-000017",
                "round_index": 17,
                "exists": True,
                "ratings_written": False,
                "status": "running",
                "phase": "game_map_started",
                "checkpoint_count": 960,
                "pair_count": 459,
                "game_count": 9639,
                "started_pair_count": 0,
                "completed_pair_count": 0,
                "completed_game_count": 0,
            },
        ],
        trainer_refresh_state={},
        current_batch_recovery_probe={
            "scan_output_progress": True,
            "skip": False,
            "reason": "activity_probe",
            "completed_game_count": 21,
            "started_pair_count": 1,
            "latest_result_ts": 1_779_002_390.7052872,
            "latest_result_age_seconds": 45.0,
            "progress_scan_error": None,
            "progress_scan_error_blocks_skip": False,
            "stale_after_seconds": 600,
            "is_stale": False,
        },
    )

    assert result["active_game_batch_count"] == 2
    assert "multiple_active_game_batches" in result["flags"]
    assert result["current_game_batch"]["round_id"] == "round-000017"
    assert result["operator_next_action"] == (
        "game outputs are landing, but multiple active game-batch artifacts exist; "
        "do not spawn more, wait for completion/reduction, then repair stale leftovers"
    )


def test_feedback_loop_status_flags_stale_active_game_output() -> None:
    result = modal_arena._feedback_loop_status_from_state(
        tournament_id="cz26-live",
        rating_run_id="elo-cz26-live",
        manifest={
            "active": True,
            "checkpoint_count": 1091,
            "queued_checkpoint_count": 1091,
            "queue_partition": "q:cz26",
            "rating_defaults": {
                "pair_selection": "adaptive_v0",
                "pairs_per_round": 300,
                "active_pool_limit": 100,
                "games_per_pair": 21,
                "save_gif": False,
                "continue_from_latest": True,
            },
        },
        manifest_load={
            "manifest_key": "manifest:cz26-live:elo-cz26-live",
            "manifest_source": "dict",
            "manifest_state_repaired": False,
        },
        active_manifest_keys=["manifest:cz26-live:elo-cz26-live"],
        queue_len=230,
        latest_snapshot={
            "round_id": "round-000011",
            "round_index": 11,
            "checkpoint_count": 681,
            "ratings": [],
        },
        batch_window=[
            {
                "round_id": "round-000017",
                "round_index": 17,
                "exists": True,
                "ratings_written": False,
                "status": "running",
                "phase": "game_map_started",
                "checkpoint_count": 960,
                "pair_count": 459,
                "game_count": 9639,
                "started_pair_count": 0,
                "completed_pair_count": 0,
                "completed_game_count": 0,
            },
        ],
        trainer_refresh_state={},
        current_batch_recovery_probe={
            "scan_output_progress": True,
            "skip": False,
            "reason": "activity_probe",
            "completed_game_count": 21,
            "started_pair_count": 1,
            "latest_result_ts": 1_779_002_390.7052872,
            "latest_result_age_seconds": 620.0,
            "progress_scan_error": None,
            "progress_scan_error_blocks_skip": False,
            "stale_after_seconds": 600,
            "is_stale": True,
        },
    )

    assert "active_game_batch_has_game_output" in result["flags"]
    assert "active_game_batch_output_stale" in result["flags"]
    assert result["current_game_batch"]["recovery_probe"]["is_stale"] is True
    assert result["operator_next_action"] == (
        "game output exists but is stale; let drain recovery scan the full output set "
        "before deciding whether to skip"
    )


def test_feedback_loop_status_flags_old_active_batch_for_partial_reduce() -> None:
    result = modal_arena._feedback_loop_status_from_state(
        tournament_id="cz26-live",
        rating_run_id="elo-cz26-live",
        manifest={
            "active": True,
            "checkpoint_count": 1091,
            "queued_checkpoint_count": 1091,
            "queue_partition": "q:cz26",
            "rating_defaults": {
                "pair_selection": "adaptive_v0",
                "pairs_per_round": 300,
                "active_pool_limit": 100,
                "games_per_pair": 21,
                "save_gif": False,
                "continue_from_latest": True,
            },
        },
        manifest_load={
            "manifest_key": "manifest:cz26-live:elo-cz26-live",
            "manifest_source": "dict",
            "manifest_state_repaired": False,
        },
        active_manifest_keys=["manifest:cz26-live:elo-cz26-live"],
        queue_len=230,
        latest_snapshot={
            "round_id": "round-000011",
            "round_index": 11,
            "checkpoint_count": 681,
            "ratings": [],
        },
        batch_window=[
            {
                "round_id": "round-000017",
                "round_index": 17,
                "exists": True,
                "ratings_written": False,
                "status": "running",
                "phase": "game_map_started",
                "checkpoint_count": 960,
                "pair_count": 300,
                "game_count": 6300,
                "started_pair_count": 0,
                "completed_pair_count": 0,
                "completed_game_count": 0,
                "updated_at": "2000-01-01T00:00:00Z",
            },
        ],
        trainer_refresh_state={},
        current_batch_recovery_probe={
            "scan_output_progress": True,
            "skip": False,
            "reason": "activity_probe",
            "completed_game_count": 21,
            "started_pair_count": 1,
            "latest_result_ts": 1_779_002_390.7052872,
            "latest_result_age_seconds": 45.0,
            "progress_scan_error": None,
            "progress_scan_error_blocks_skip": False,
            "stale_after_seconds": 600,
            "is_stale": False,
        },
    )

    assert "active_game_batch_partial_reduce_due" in result["flags"]
    assert result["current_game_batch"]["partial_reduce_due"] is True

    decision = modal_arena._feedback_loop_control_decision(
        result,
        action="drain-if-ready",
    )

    assert decision["spawn_drain"] is True
    assert decision["reason"] == "spawn_active_game_batch_recovery_scan"


def test_feedback_loop_control_decision_spawns_only_when_ready() -> None:
    ready = {
        "status": "ready_for_next_rating_batch",
        "flags": ["latest_rating_behind_intake"],
        "current_game_batch": None,
        "intake": {
            "new_checkpoints_not_in_latest_rating": 412,
            "queue_len": 0,
        },
    }

    decision = modal_arena._feedback_loop_control_decision(
        ready,
        action="drain-if-ready",
    )

    assert decision["spawn_drain"] is True
    assert decision["reason"] == "spawn_next_bounded_rating_batch"


def test_feedback_loop_control_decision_blocks_active_game_batch() -> None:
    active = {
        "status": "rating_game_batch_active",
        "flags": ["rating_game_batch_active", "latest_rating_behind_intake"],
        "current_game_batch": {"round_id": "round-000018"},
        "intake": {
            "new_checkpoints_not_in_latest_rating": 412,
            "queue_len": 0,
        },
    }

    decision = modal_arena._feedback_loop_control_decision(active, action="drain")

    assert decision["spawn_drain"] is False
    assert decision["reason"] == "blocked_active_game_batch"


def test_feedback_loop_control_decision_allows_stale_active_recovery_scan() -> None:
    active = {
        "status": "rating_game_batch_active",
        "flags": [
            "rating_game_batch_active",
            "latest_rating_behind_intake",
            "active_game_batch_output_stale",
        ],
        "current_game_batch": {"round_id": "round-000018"},
        "intake": {
            "new_checkpoints_not_in_latest_rating": 412,
            "queue_len": 0,
        },
    }

    decision = modal_arena._feedback_loop_control_decision(active, action="drain")

    assert decision["spawn_drain"] is True
    assert decision["reason"] == "spawn_active_game_batch_recovery_scan"


def test_feedback_loop_control_decision_allows_old_pool_recovery_scan() -> None:
    active = {
        "status": "rating_game_batch_active",
        "flags": [
            "rating_game_batch_active",
            "latest_rating_behind_intake",
            "active_game_batch_not_covering_new_checkpoints",
        ],
        "current_game_batch": {"round_id": "round-000035"},
        "intake": {
            "new_checkpoints_not_in_latest_rating": 688,
            "queue_len": 0,
        },
    }

    decision = modal_arena._feedback_loop_control_decision(active, action="drain")

    assert decision["spawn_drain"] is True
    assert decision["reason"] == "spawn_active_game_batch_recovery_scan"


def test_local_entrypoint_blocks_unsafe_live_control_modes() -> None:
    with pytest.raises(ValueError, match="temporary scheduled Modal app"):
        modal_arena._assert_safe_local_entrypoint_mode("loop-status")


def test_local_entrypoint_allows_non_control_modes() -> None:
    modal_arena._assert_safe_local_entrypoint_mode("current")


def test_rating_round_activity_probe_checks_expected_battle_dirs(tmp_path) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    round_id = arena.rating_round_id(0)
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
            "games_per_shard": 1,
        }
    )
    pair_specs = arena.build_rating_round_pair_specs(rating_spec)
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 0,
            "checkpoint_roster": arena.rating_roster_by_checkpoint(rating_spec["checkpoints"]),
            "pair_count": len(pair_specs),
            "game_count": 3,
            "pair_specs": pair_specs,
            "rating_spec": rating_spec,
        },
    )
    pair = pair_specs[0]
    modal_arena.runs.write_json(
        tmp_path / arena.game_summary_ref(tournament_id, pair["battle_id"], "game-000000"),
        {
            "schema_id": arena.GAME_SCHEMA_ID,
            "tournament_id": tournament_id,
            "battle_id": pair["battle_id"],
            "game_id": "game-000000",
            "pair_index": pair["pair_index"],
            "ok": True,
        },
    )

    probe = modal_arena._rating_round_activity_probe(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        max_pairs=8,
    )

    assert probe["has_output"] is True
    assert probe["sampled_pair_count"] == 1
    assert probe["seen_pair_count"] == 1
    assert probe["completed_game_count"] == 1
    assert probe["latest_result_ts"] is not None
    assert probe["latest_result_age_seconds"] is not None
    assert probe["stale_after_seconds"] == modal_arena.DEFAULT_RATING_ROUND_STALE_SECONDS


def test_rating_game_batch_status_summary_reports_spec_and_roster_counts(tmp_path) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    round_index = 2
    round_id = arena.rating_round_id(round_index)
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [
                _checkpoint_ref("run-a", 0),
                _checkpoint_ref("run-b", 10),
                _checkpoint_ref("run-c", 20),
            ],
            "games_per_pair": 3,
        }
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": round_index,
            "checkpoint_roster": arena.rating_roster_by_checkpoint(
                rating_spec["checkpoints"]
            ),
            "pair_count": 1,
            "game_count": 3,
            "rating_spec": rating_spec,
        },
    )

    summary = modal_arena._rating_game_batch_status_summary(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_index=round_index,
    )

    assert summary["checkpoint_count"] == 3
    assert summary["rating_spec_checkpoint_count"] == 3
    assert summary["checkpoint_roster_count"] == 3


def test_rating_game_batch_status_summary_reports_skip_decision(tmp_path) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    round_index = 15
    round_id = arena.rating_round_id(round_index)
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": round_index,
            "checkpoint_count": 919,
            "pair_count": 438,
            "game_count": 9198,
            "rating_spec": {
                "pair_selection": "adaptive_v0",
                "pairs_per_round": 300,
                "active_pool_limit": 100,
                "games_per_pair": 21,
                "save_gif": False,
            },
        },
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_progress_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": round_index,
            "status": "skipped",
            "phase": "stale_orphan_round_skipped",
            "skip_reason": "stale_incomplete_round",
            "completed_game_count": 21,
            "started_pair_count": 1,
            "skip_decision": {
                "reason": "stale_incomplete_round",
                "input_checkpoint_count": 919,
                "desired_checkpoint_count": 956,
                "pair_count": 438,
                "game_count": 9198,
                "completed_game_count": 21,
                "started_pair_count": 1,
                "stale_after_seconds": 600,
                "stale_age_seconds": 612.5,
                "latest_result_ts": 1_779_001_044.421591,
                "newest_real_activity_ts": 1_779_001_044.421591,
                "is_stale": True,
                "scan_output_progress": True,
                "progress_scan_error": None,
                "progress_scan_error_blocks_skip": False,
            },
        },
    )

    summary = modal_arena._rating_game_batch_status_summary(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_index=round_index,
    )

    assert summary["status"] == "skipped"
    assert summary["skip_reason"] == "stale_incomplete_round"
    assert summary["skip_decision"]["completed_game_count"] == 21
    assert summary["skip_decision"]["desired_checkpoint_count"] == 956
    assert summary["skip_decision"]["scan_output_progress"] is True
    assert summary["skip_decision"]["progress_scan_error_blocks_skip"] is False


def test_rating_reduce_rebuilds_latest_from_shard_summaries(tmp_path) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
            "games_per_shard": 2,
        }
    )
    round_id = arena.rating_round_id(0)
    pairs = arena.build_rating_round_pair_specs(rating_spec)
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 0,
            "started_at": "2026-05-13T00:00:00Z",
            "rating_spec": rating_spec,
            "pair_count": len(pairs),
            "game_count": 3,
            "pair_specs": pairs,
        },
    )
    pair = pairs[0]
    games = [
        {
            **_fake_game(pair, index, outcome),
            "tournament_id": tournament_id,
            "battle_id": pair["battle_id"],
            "pair_index": pair["pair_index"],
        }
        for index, outcome in enumerate(["seat_0_win", "seat_0_win", "seat_1_win"])
    ]
    for shard_index, shard_games in enumerate([games[:2], games[2:]]):
        shard_id = f"shard-{shard_index:06d}"
        modal_arena.runs.write_json(
            tmp_path / arena.game_shard_summary_ref(tournament_id, pair["battle_id"], shard_id),
            {
                "schema_id": arena.GAME_SHARD_SCHEMA_ID,
                "tournament_id": tournament_id,
                "battle_id": pair["battle_id"],
                "pair_index": pair["pair_index"],
                "shard_id": shard_id,
                "shard_index": shard_index,
                "game_count": len(shard_games),
                "tally": arena.tally_game_results(shard_games),
            },
        )

    result = modal_arena._reduce_rating_round_from_summaries(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
    )
    latest = modal_arena._read_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id)
    )
    results = modal_arena._read_json(
        tmp_path / arena.rating_round_results_ref(tournament_id, rating_run_id, round_id)
    )
    progress = modal_arena._read_json(
        tmp_path / arena.rating_progress_ref(tournament_id, rating_run_id)
    )
    battle_summary = modal_arena._read_json(
        tmp_path / arena.battle_summary_ref(tournament_id, pair["battle_id"])
    )

    assert result["reduced_from"] == "shard_tallies"
    assert result["game_count"] == 3
    assert result["snapshot"]["rated_pair_count"] == 1
    assert latest["ratings"][0]["rating"] > arena.DEFAULT_RATING_INITIAL_RATING
    assert results["pair_rating_results"][0]["valid_games"] == 3
    assert progress["status"] == "complete"
    assert progress["completed_game_count"] == 3
    assert progress["work_summary"]["parent_result_mode"] == "volume_shard_tallies"
    assert battle_summary["result_detail_mode"] == "shard_tally"
    assert battle_summary["shard_summary_ref_count"] == 2


def test_rating_reduce_rebuilds_legacy_shard_tally_wins_from_games(
    tmp_path,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 0), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
            "games_per_shard": 2,
        }
    )
    round_id = arena.rating_round_id(0)
    pairs = arena.build_rating_round_pair_specs(rating_spec)
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 0,
            "started_at": "2026-05-13T00:00:00Z",
            "rating_spec": rating_spec,
            "pair_count": len(pairs),
            "game_count": 3,
            "pair_specs": pairs,
        },
    )
    pair = pairs[0]
    games = [
        {
            **_fake_game(pair, index, outcome),
            "tournament_id": tournament_id,
            "battle_id": pair["battle_id"],
            "pair_index": pair["pair_index"],
        }
        for index, outcome in enumerate(["seat_0_win", "seat_0_win", "seat_1_win"])
    ]
    for shard_index, shard_games in enumerate([games[:2], games[2:]]):
        shard_id = f"shard-{shard_index:06d}"
        tally = arena.tally_game_results(shard_games)
        tally.pop("wins_by_checkpoint", None)
        modal_arena.runs.write_json(
            tmp_path / arena.game_shard_summary_ref(tournament_id, pair["battle_id"], shard_id),
            {
                "schema_id": arena.GAME_SHARD_SCHEMA_ID,
                "tournament_id": tournament_id,
                "battle_id": pair["battle_id"],
                "pair_index": pair["pair_index"],
                "shard_id": shard_id,
                "shard_index": shard_index,
                "game_count": len(shard_games),
                "games": shard_games,
                "tally": tally,
            },
        )

    result = modal_arena._reduce_rating_round_from_summaries(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
    )
    battle_summary = modal_arena._read_json(
        tmp_path / arena.battle_summary_ref(tournament_id, pair["battle_id"])
    )

    assert result["reduced_from"] == "shard_tallies"
    assert result["snapshot"]["rated_pair_count"] == 1
    assert battle_summary["tally"]["wins_by_checkpoint"]
    assert len(battle_summary["games"]) == 3


def test_modal_browser_lists_running_rating_from_progress(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    marker = tmp_path / arena.tournament_marker_ref(tournament_id)
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    progress_path = tmp_path / arena.rating_progress_ref(tournament_id, rating_run_id)
    modal_arena.runs.write_json(
        progress_path,
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": "round-000000",
            "round_index": 0,
            "status": "running",
            "phase": "games_running",
            "pair_count": 10,
            "game_count": 500,
            "completed_pair_count": 2,
            "completed_game_count": 125,
            "completion_fraction": 0.25,
            "updated_at": "2026-05-13T00:00:00Z",
        },
    )

    rows = modal_arena._list_rating_runs(tmp_path, tournament_id=tournament_id)
    progress = modal_arena._read_rating_progress(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

    assert rows[0]["rating_run_id"] == rating_run_id
    assert rows[0]["status"] == "running"
    assert rows[0]["completed_game_count"] == 125
    assert rows[0]["latest_ref"] is None
    assert rows[0]["progress_ref"].endswith("/progress.json")
    assert progress["completion_fraction"] == 0.25


def test_render_page_shows_running_rating_progress_without_latest() -> None:
    html = modal_arena._render_page(
        tournaments=[
            {
                "tournament_id": "arena-running",
                "status": "rating",
                "updated_ts": 1.0,
            }
        ],
        selected_tournament_id="arena-running",
        selected_rating_run_id="elo-running",
        selected_checkpoint_id="",
        rating_runs=[
            {
                "tournament_id": "arena-running",
                "rating_run_id": "elo-running",
                "status": "running",
            }
        ],
        rating_snapshot={},
        rating_progress={
            "tournament_id": "arena-running",
            "rating_run_id": "elo-running",
            "status": "running",
            "phase": "games_running",
            "started_pair_count": 46,
            "pair_count": 3000,
            "estimated_seen_game_count": 920,
            "game_count": 60000,
            "estimated_completion_fraction": 920 / 60000,
            "updated_at": "2026-05-13T00:00:00Z",
            "recent_started_pairs": [
                {
                    "battle_id": "battle-live",
                    "pair_index": 12,
                    "expected_game_count": 10,
                    "complete": False,
                }
            ],
        },
        battles={"rows": [], "total": 0},
    )

    assert "Progress" in html
    assert "elo-running (running)" in html
    assert "Leaderboard rows will appear after this rating round writes" in html
    assert "leaderboard is pending, not missing" in html
    assert "Rating run exists, but no rating rows were found" not in html
    assert "46/3000" in html
    assert "920/60000" in html
    assert "Recent Battles" in html
    assert "battle-live" in html
    assert "pair 12" in html
    assert 'id="progress-panel"' in html
    assert 'data-tournament-id="arena-running"' in html
    assert 'data-has-ratings="false"' in html
    assert "/api/rating-progress?" in html
    assert 'params.set("fresh", "true")' not in html
    assert "window.setInterval" not in html
    assert "window.setTimeout" in html


def test_render_page_shows_provisional_live_rankings() -> None:
    html = modal_arena._render_page(
        tournaments=[
            {
                "tournament_id": "arena-running",
                "status": "rating",
                "updated_ts": 1.0,
            }
        ],
        selected_tournament_id="arena-running",
        selected_rating_run_id="elo-running",
        selected_checkpoint_id="",
        rating_runs=[
            {
                "tournament_id": "arena-running",
                "rating_run_id": "elo-running",
                "status": "running",
            }
        ],
        rating_snapshot={
            "tournament_id": "arena-running",
            "rating_run_id": "elo-running",
            "round_id": "round-000000",
            "provisional": True,
            "ratings": [
                {
                    "rank": 1,
                    "checkpoint_id": "ckpt-a",
                    "label": "A",
                    "rating": 1510.0,
                    "games": 2,
                    "wins": 2,
                    "losses": 0,
                    "draws": 0,
                    "win_rate": 1.0,
                    "distinct_opponents": 1,
                    "failure_count": 0,
                }
            ],
        },
        rating_progress={
            "tournament_id": "arena-running",
            "rating_run_id": "elo-running",
            "status": "running",
            "phase": "games_running",
            "started_pair_count": 1,
            "pair_count": 3,
            "estimated_seen_game_count": 2,
            "game_count": 6,
            "estimated_completion_fraction": 1 / 3,
            "updated_at": "2026-05-13T00:00:00Z",
        },
        battles={"rows": [], "total": 0},
    )

    assert "Live Leaderboard" in html
    assert "updating from finished games" in html
    assert "Rankings will appear as soon as finished games are visible" not in html
    assert "A" in html


def test_dynamic_web_headers_disable_browser_cache() -> None:
    assert modal_arena.DYNAMIC_HEADERS["Cache-Control"] == "no-store, max-age=0"
    assert modal_arena.DYNAMIC_HEADERS["Pragma"] == "no-cache"
    assert modal_arena.DYNAMIC_HEADERS["Expires"] == "0"


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/", {}),
        ("/battle", {"battle_id": "battle-a"}),
    ],
)
def test_page_loads_throttle_volume_reload_unless_fresh(
    tmp_path,
    monkeypatch,
    path,
    params,
) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    TestClient = fastapi_testclient.TestClient
    calls: list[dict[str, object]] = []

    def fake_reload(_volume, *, force=False, min_interval_sec=None):
        calls.append({"force": force, "min_interval_sec": min_interval_sec})
        return None

    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_web_reload_volume", fake_reload)
    client = TestClient(modal_arena._build_fastapi_app(object()))

    response = client.get(path, params=params)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["expires"] == "0"
    assert calls
    assert calls[0]["force"] is False
    assert calls[0]["min_interval_sec"] == modal_arena.WEB_PAGE_RELOAD_MIN_INTERVAL_SECONDS

    calls.clear()
    fresh_response = client.get(path, params={**params, "fresh": "true"})

    assert fresh_response.status_code == 200
    assert calls
    assert calls[0]["force"] is True
    assert calls[0]["min_interval_sec"] == modal_arena.WEB_PAGE_RELOAD_MIN_INTERVAL_SECONDS


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/api/tournaments", {}),
        ("/api/ratings", {}),
        ("/api/rating-standings", {}),
        ("/api/review/rankings", {}),
        ("/api/review/checkpoint", {"checkpoint_id": "ckpt-a"}),
        ("/api/review/battle", {"battle_id": "battle-a"}),
        ("/api/battles", {}),
    ],
)
def test_fresh_api_requests_force_volume_reload(
    tmp_path,
    monkeypatch,
    path,
    params,
) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    TestClient = fastapi_testclient.TestClient

    calls: list[dict[str, object]] = []

    def fake_reload(_volume, *, force=False, min_interval_sec=None):
        calls.append({"force": force, "min_interval_sec": min_interval_sec})
        return None

    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_web_reload_volume", fake_reload)
    client = TestClient(modal_arena._build_fastapi_app(object()))

    response = client.get(path, params={**params, "fresh": "true"})

    assert response.status_code == 200
    assert calls
    assert calls[0]["force"] is True


def test_web_reload_volume_throttles_after_reload_error(monkeypatch) -> None:
    class FailingVolume:
        def __init__(self) -> None:
            self.calls = 0

        def reload(self) -> None:
            self.calls += 1
            raise RuntimeError("busy")

    volume = FailingVolume()
    monkeypatch.setattr(modal_arena, "_LAST_WEB_VOLUME_RELOAD_TS", 0.0)

    error = modal_arena._web_reload_volume(volume, min_interval_sec=60.0)
    skipped = modal_arena._web_reload_volume(volume, min_interval_sec=60.0)

    assert "RuntimeError: busy" == error
    assert skipped is None
    assert volume.calls == 1
