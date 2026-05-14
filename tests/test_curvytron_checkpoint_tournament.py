from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from curvyzero.infra.modal import curvyzero_checkpoint_tournament as modal_arena
from curvyzero.tournament import curvytron_checkpoint_tournament as arena


def _checkpoint_ref(run_id: str, iteration: int) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/"
        f"{run_id}/checkpoints/lightzero/iteration_{iteration}.pth.tar"
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
                    "summary_ref": (
                        "tournaments/curvytron/arena-a/battles/battle-bc/battle.json"
                    ),
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
                    "summary_ref": (
                        "tournaments/curvytron/arena-a/battles/battle-ac/battle.json"
                    ),
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
                    "summary_ref": (
                        "tournaments/curvytron/arena-a/battles/battle-ab/battle.json"
                    ),
                    "first_gif_ref": (
                        "tournaments/curvytron/arena-a/battles/battle-ab/"
                        "games/game-000000/game.gif"
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
                "tournaments/curvytron/arena-a/battles/battle-ab/"
                "games/game-000000/game.gif"
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
                    "winner_seat": 0 if outcome == "seat_0_win" else 1 if outcome == "seat_1_win" else None,
                    "loser_seat": 1 if outcome == "seat_0_win" else 0 if outcome == "seat_1_win" else None,
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
    assert spec["games_per_pair"] == 21
    assert arena.DEFAULT_MAX_STEPS == 8_000
    assert spec["max_steps"] == 8_000


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

    game_kind, game_specs = modal_arena._build_game_work_specs(
        [pair],
        games_per_shard=1,
    )
    shard_kind, shard_specs = modal_arena._build_game_work_specs(
        [pair],
        games_per_shard=2,
    )

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
        }
    )

    assert compact["tournament_id"] == "arena-a"
    assert compact["battle_id"] == pair["battle_id"]
    assert compact["pair_index"] == pair["pair_index"]


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

    assert game["frame_size"] == 704
    assert info["pixel_size"] == [704, 704]
    with Image.open(path) as image:
        assert image.size == (704, 704)


def test_tournament_render_contract_splits_policy_and_gif_modes() -> None:
    pair = arena.build_pair_specs(
        tournament_id="arena-a",
        checkpoints=[
            {
                "checkpoint_ref": _checkpoint_ref("run-a", 0),
                "policy_trail_render_mode": "body_circles_fast",
            },
            {
                "checkpoint_ref": _checkpoint_ref("run-b", 10),
                "policy_trail_render_mode": "browser_lines",
            },
        ],
        games_per_pair=1,
        trail_render_mode="body_circles_fast",
        gif_trail_render_mode="body_circles_fast",
        frame_size=64,
        save_frames_npz=True,
    )[0]
    game = arena.build_game_specs_for_pair(pair)[0]

    assert pair["policy_trail_render_mode"] == "body_circles_fast"
    assert pair["gif_trail_render_mode"] == "browser_lines"
    assert pair["frame_size"] == 704
    assert game["policy_trail_render_mode"] == "body_circles_fast"
    assert game["gif_trail_render_mode"] == "browser_lines"
    assert game["frame_size"] == 704
    assert game["players"][0]["policy_trail_render_mode"] == "body_circles_fast"
    assert game["players"][1]["policy_trail_render_mode"] == "browser_lines"


def test_checkpoint_spec_reads_policy_render_mode_from_observation_contract() -> None:
    checkpoint = arena.normalize_checkpoint_spec(
        {
            "checkpoint_ref": _checkpoint_ref("run-a", 0),
            "observation_contract": {"trail_render_mode": "body_circles_fast"},
        },
    )

    assert checkpoint["policy_trail_render_mode"] == "body_circles_fast"


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
                    "source_state_trail_render_mode": "body_circles_fast",
                    "decision_ms": 200.0,
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        arena._checkpoint_policy_trail_render_mode_from_ref(ref, mount=tmp_path)
        == "body_circles_fast"
    )
    assert arena._checkpoint_runtime_settings_from_ref(ref, mount=tmp_path)[
        "decision_ms"
    ] == 200.0


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

    assert settings["decision_source_frames"] == 12
    assert settings["decision_ms"] == pytest.approx(200.0)
    assert settings["source_max_ticks"] == 64 * 12


def test_checkpoint_game_uses_per_seat_policy_modes_and_full_rich_gif(
    tmp_path,
    monkeypatch,
) -> None:
    from curvyzero.env import vector_multiplayer_env as env_mod
    from curvyzero.env import vector_visual_observation as visual_mod
    from curvyzero.training import curvytron_current_policy_selfplay_smoke as stack_mod

    stack_modes: list[str] = []
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
        def __init__(self, *, batch_size: int, player_count: int, trail_render_mode: str):
            self.trail_render_mode = trail_render_mode
            stack_modes.append(trail_render_mode)

        def update(self, env, *, copy: bool = True):
            value = 7.0 if self.trail_render_mode == "body_circles_fast" else 11.0
            return np.full((1, 2, 4, 64, 64), value, dtype=np.float32)

    def fake_render(state, *, row: int, frame_size: int, trail_render_mode: str):
        render_calls.append(
            {"frame_size": frame_size, "trail_render_mode": trail_render_mode}
        )
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
            {"policy": object(), "policy_trail_render_mode": "body_circles_fast"},
            {"policy": object(), "policy_trail_render_mode": "browser_lines"},
        ],
    )

    assert sorted(stack_modes) == ["body_circles_fast", "browser_lines"]
    assert env_kwargs[0]["decision_source_frames"] == 12
    assert env_kwargs[0]["decision_ms"] == pytest.approx(200.0)
    assert env_kwargs[0]["max_ticks"] == 2 * 12
    assert render_calls
    assert {call["frame_size"] for call in render_calls} == {704}
    assert {call["trail_render_mode"] for call in render_calls} == {"browser_lines"}
    assert policy_calls == [(0, 7.0), (1, 11.0)]
    assert summary["frame_size"] == 704
    assert summary["gif_trail_render_mode"] == "browser_lines"
    assert summary["decision_source_frames"] == 12
    assert summary["decision_ms"] == pytest.approx(200.0)
    assert summary["source_physics_step_ms"] == pytest.approx(1000.0 / 60.0)
    assert summary["source_max_ticks"] == 2 * 12
    assert summary["render_contract"]["gif_frame_size"] == 704
    assert summary["render_contract"]["gif_trail_render_mode"] == "browser_lines"
    assert summary["policy_trail_render_modes"] == {
        "seat_0": "body_circles_fast",
        "seat_1": "browser_lines",
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
    played = {
        player["checkpoint_id"]
        for pair in pairs
        for player in pair["players"]
    }

    assert 10 <= len(pairs) <= 20
    assert played == {
        checkpoint["checkpoint_id"] for checkpoint in rating_spec["checkpoints"]
    }
    assert {pair["schedule_reason"] for pair in pairs} == {
        arena.SCHEDULE_REASON_PLACEMENT
    }


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
    played = {
        player["checkpoint_id"]
        for pair in pairs
        for player in pair["players"]
    }

    assert len(pairs) == 6
    assert new_ids <= played
    placement_pairs = [
        pair
        for pair in pairs
        if pair["schedule_reason"] == arena.SCHEDULE_REASON_PLACEMENT
    ]
    assert len(placement_pairs) == 3
    assert new_ids <= {
        player["checkpoint_id"]
        for pair in placement_pairs
        for player in pair["players"]
    }


def test_adaptive_v0_expands_budget_to_cover_new_checkpoints() -> None:
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
    played = {
        player["checkpoint_id"]
        for pair in pairs
        for player in pair["players"]
    }

    distinct = _distinct_opponents_by_checkpoint(pairs)

    assert rating_spec["placement_min_games"] == 12
    assert rating_spec["placement_min_opponents"] == 20
    assert len(pairs) == 3
    assert played == {
        checkpoint["checkpoint_id"] for checkpoint in rating_spec["checkpoints"]
    }
    assert all(len(opponents) >= 1 for opponents in distinct.values())
    assert {pair["schedule_reason"] for pair in pairs} == {
        arena.SCHEDULE_REASON_PLACEMENT
    }


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
        "checkpoint_roster": arena.rating_roster_by_checkpoint(
            checkpoints
        ),
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
    assert {pair["schedule_reason"] for pair in pairs} == {
        arena.SCHEDULE_REASON_PLACEMENT
    }
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
    assert {pair["schedule_reason"] for pair in pairs} == {
        arena.SCHEDULE_REASON_PLACEMENT
    }


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
    assert {pair["schedule_reason"] for pair in pairs} == {
        arena.SCHEDULE_REASON_PLACEMENT
    }
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

    first_pair_ids = {
        player["checkpoint_id"]
        for player in pairs[0]["players"]
    }
    undercovered_id = rating_spec["checkpoints"][3]["checkpoint_id"]
    top_band_ids = {
        checkpoint["checkpoint_id"]
        for checkpoint in rating_spec["checkpoints"][30:40]
    }

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
    lower_half_average = (
        sum(appearances.get(item, 0) for item in lower_half_ids) / len(lower_half_ids)
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
    assert arena.rating_context_hash(base) == arena.rating_context_hash(
        non_context_change
    )
    assert arena.rating_pool_hash(base["checkpoints"]) != arena.rating_pool_hash(
        expanded_roster["checkpoints"]
    )
    assert arena.rating_context_hash(base) != arena.rating_context_hash(changed_context)


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
                {"checkpoint_ref": _checkpoint_ref("run-a-replaced", 99), "checkpoint_id": "ckpt-a"},
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
                {"checkpoint_ref": _checkpoint_ref("run-a-replaced", 99), "checkpoint_id": "ckpt-a"},
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
                {"checkpoint_ref": _checkpoint_ref("run-a-replaced", 99), "checkpoint_id": "ckpt-a"},
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
    played = {
        player["checkpoint_id"]
        for pair_spec in pairs
        for player in pair_spec["players"]
    }

    assert state["continued_from_latest"] is True
    assert state["start_round_index"] == 3
    assert state["previous_snapshot"]["round_id"] == "round-000002"
    assert new_id in played
    assert pairs[0]["scheduled_round_index"] == 3
    assert pairs[0]["schedule_reason"] == arena.SCHEDULE_REASON_PLACEMENT


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


def test_intake_tick_enqueues_only_refs_not_already_seen(monkeypatch) -> None:
    old_ref = _checkpoint_ref("run-a", 10)
    new_ref = _checkpoint_ref("run-a", 20)
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"run_ids": "run-a", "checkpoint_selection": "all"},
        rating_defaults={},
        discovery={"checkpoint_refs": [old_ref]},
    )

    class FakeState:
        def __init__(self):
            self.values = {
                modal_arena.CHECKPOINT_INTAKE_ACTIVE_KEYS: [manifest["manifest_key"]],
                manifest["manifest_key"]: manifest,
            }

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value):
            self.values[key] = value
            return True

    class FakeQueue:
        def __init__(self):
            self.events = []

        def put(self, value, **_kwargs):
            self.events.append(value)
            return True

    fake_state = FakeState()
    fake_queue = FakeQueue()
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", fake_state)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", fake_queue)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "_commit_volume", lambda _volume: None)
    monkeypatch.setattr(
        modal_arena,
        "_write_intake_manifest_artifact",
        lambda _manifest: {"ref": "manifest.json"},
    )
    monkeypatch.setattr(
        modal_arena,
        "_write_intake_tick_artifact",
        lambda _tick: {"ref": "tick.json"},
    )
    monkeypatch.setattr(
        modal_arena,
        "_discover_checkpoint_refs_from_scan_spec",
        lambda _scan_spec, *, mount: {
            "checkpoint_refs": [old_ref, new_ref],
            "found_count": 2,
            "missing_count": 0,
            "checkpoint_selection": "all",
        },
    )

    result = modal_arena.curvytron_checkpoint_intake_tick.local({})

    assert result["new_checkpoint_count"] == 1
    assert [event["checkpoint_ref"] for event in fake_queue.events] == [new_ref]
    updated_manifest = fake_state.values[manifest["manifest_key"]]
    assert updated_manifest["seen_checkpoint_count"] == 2
    assert updated_manifest["queued_checkpoint_refs"] == [new_ref]


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


def test_intake_scan_run_ids_are_live_but_explicit_refs_are_frozen(tmp_path) -> None:
    run_id = "watch-run-a"
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
    (ckpt_dir / "iteration_10.pth.tar").write_bytes(b"old")
    explicit_ref = (
        modal_arena.runs.attempt_train_ref(
            modal_arena.TRAINING_TASK_ID,
            run_id,
            attempt_id,
        )
        / "lightzero_exp"
        / "ckpt"
        / "iteration_10.pth.tar"
    ).as_posix()

    run_before = modal_arena._discover_checkpoint_refs_from_scan_spec(
        {"run_ids": [run_id]},
        mount=tmp_path,
    )
    (ckpt_dir / "iteration_20.pth.tar").write_bytes(b"new")
    explicit_after = modal_arena._discover_checkpoint_refs_from_scan_spec(
        {"checkpoint_refs": [explicit_ref]},
        mount=tmp_path,
    )
    run_after = modal_arena._discover_checkpoint_refs_from_scan_spec(
        {"run_ids": [run_id]},
        mount=tmp_path,
    )

    assert explicit_after["selection"] == "explicit_refs"
    assert explicit_after["checkpoint_refs"] == [explicit_ref]
    assert run_before["checkpoint_refs"][0].endswith("iteration_10.pth.tar")
    assert run_after["checkpoint_refs"][0].endswith("iteration_20.pth.tar")


def test_intake_rating_spec_continuation_uses_seen_checkpoint_pool() -> None:
    old_ref = _checkpoint_ref("run-a", 10)
    current_refs = [_checkpoint_ref("run-a", 20), _checkpoint_ref("run-b", 20)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"run_ids": "run-a,run-b", "checkpoint_selection": "latest"},
        rating_defaults={"continue_from_latest": True, "pairs_per_round": 1},
        discovery={"checkpoint_refs": current_refs},
        existing={"seen_checkpoint_refs": [old_ref]},
    )

    spec = modal_arena._intake_rating_spec_from_manifest(manifest)

    assert spec["continue_from_latest"] is True
    spec_refs = {
        str(row.get("checkpoint_ref") if isinstance(row, dict) else row)
        for row in spec["checkpoints"]
    }
    assert spec_refs == {old_ref, *current_refs}


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


def test_intake_drain_continues_existing_rating_when_requested(
    tmp_path,
    monkeypatch,
) -> None:
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": refs},
        rating_defaults={
            "continue_from_latest": True,
            "pairs_per_round": 1,
            "games_per_pair": 3,
        },
        discovery={"checkpoint_refs": refs},
    )
    manifest = modal_arena._mark_intake_manifest_queued(manifest, refs)
    events = [{"checkpoint_ref": ref} for ref in refs]
    captured_spec: dict[str, object] = {}

    class FakeState:
        def __init__(self):
            self.values = {
                manifest["manifest_key"]: manifest,
                f"rating_claim:{manifest['manifest_key']}": {"old": True},
            }

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    class FakeQueue:
        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(events)

        def get_many(self, *_args, **_kwargs):
            return events

    class FakeCall:
        object_id = "fc-continue"

    class FakeRatingLoop:
        def spawn(self, spec):
            captured_spec.update(spec)
            return FakeCall()

    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", FakeQueue())
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", FakeRatingLoop())
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

    assert result["event_count"] == 2
    assert result["rating_call_id"] == "fc-continue"
    assert result["rating_claim_key"] != f"rating_claim:{manifest['manifest_key']}"
    assert result["spawn_skipped_reason"] == ""
    assert captured_spec["continue_from_latest"] is True


def test_intake_drain_tick_uses_continuation_defaults(monkeypatch) -> None:
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": refs},
        rating_defaults={"continue_from_latest": True},
        discovery={"checkpoint_refs": refs},
    )
    captured_specs: list[dict[str, object]] = []

    class FakeState:
        def get(self, key, default=None):
            if key == modal_arena.CHECKPOINT_INTAKE_ACTIVE_KEYS:
                return [manifest["manifest_key"]]
            if key == manifest["manifest_key"]:
                return manifest
            return default

    class FakeDrain:
        def local(self, spec):
            captured_specs.append(dict(spec))
            return {
                "event_count": 2,
                "rating_call_id": "fc-test",
            }

    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "curvytron_checkpoint_intake_drain", FakeDrain())

    result = modal_arena.curvytron_checkpoint_intake_drain_tick.local()

    assert result["drained_manifest_count"] == 1
    assert captured_specs[0]["continue_from_latest"] is True
    assert captured_specs[0]["spawn_if_existing"] is True


def _write_leaderboard_publish_rating_snapshot(
    tmp_path: Path,
    *,
    tournament_id: str = "arena-a",
    rating_run_id: str = "elo-a",
    provisional: bool = False,
) -> None:
    latest_ref = (
        modal_arena._rating_provisional_latest_ref(tournament_id, rating_run_id)
        if provisional
        else arena.rating_latest_ref(tournament_id, rating_run_id)
    )
    latest_path = tmp_path / latest_ref
    latest_path.parent.mkdir(parents=True)
    payload = {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "formula_version": arena.RATING_FORMULA_VERSION,
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "ratings_ref": latest_ref.as_posix(),
        "context_hash": "ctx-a",
        "roster_hash": "roster-a",
        "round_index": 3,
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
    if provisional:
        payload["provisional"] = True
    latest_path.write_text(json.dumps(payload), encoding="utf-8")


def test_opponent_leaderboard_publish_writes_snapshot_latest_and_pointer(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-a"
    _write_leaderboard_publish_rating_snapshot(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

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
        }
    )

    snapshot_ref = modal_arena._leaderboard_snapshot_ref("main", "snapshot-001")
    latest_ref = modal_arena._leaderboard_latest_ref("main")
    assert (tmp_path / snapshot_ref).is_file()
    assert (tmp_path / latest_ref).is_file()
    assert result["row_count"] == 1
    assert result["active_count"] == 1
    assert result["pointer_key"] == "current:main"
    assert result["pointer_published"] is True
    assert fake_dict.values["current:main"]["snapshot_id"] == "snapshot-001"


def test_opponent_leaderboard_publish_rejects_provisional_without_opt_in(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-a"
    _write_leaderboard_publish_rating_snapshot(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        provisional=True,
    )
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)

    def fail_commit(_volume):
        raise AssertionError("provisional publish should fail before commit")

    monkeypatch.setattr(modal_arena, "_commit_volume", fail_commit)

    with pytest.raises(ValueError, match="refusing to publish provisional"):
        modal_arena.curvytron_opponent_leaderboard_publish.local(
            {
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "leaderboard_id": "main",
                "snapshot_id": "snapshot-001",
            }
        )


def test_opponent_leaderboard_publish_commits_before_pointer_update(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-a"
    _write_leaderboard_publish_rating_snapshot(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    call_order = []

    class FakeDict:
        def put(self, key, value, **_kwargs):
            call_order.append(("put", key, value["snapshot_id"]))
            return True

    def fake_commit(_volume):
        call_order.append(("commit",))
        return None

    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "_commit_volume", fake_commit)
    monkeypatch.setattr(modal_arena, "opponent_leaderboard_state", FakeDict())

    result = modal_arena.curvytron_opponent_leaderboard_publish.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "leaderboard_id": "main",
            "snapshot_id": "snapshot-001",
        }
    )

    assert result["pointer_published"] is True
    assert call_order == [("commit",), ("put", "current:main", "snapshot-001")]


def test_opponent_leaderboard_publish_skips_pointer_update_on_commit_error(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-a"
    _write_leaderboard_publish_rating_snapshot(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )

    class FakeDict:
        def __init__(self) -> None:
            self.values = {}

        def put(self, key, value, **_kwargs):
            self.values[key] = value
            return True

    fake_dict = FakeDict()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "_commit_volume", lambda _volume: "commit failed")
    monkeypatch.setattr(modal_arena, "opponent_leaderboard_state", fake_dict)

    result = modal_arena.curvytron_opponent_leaderboard_publish.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "leaderboard_id": "main",
            "snapshot_id": "snapshot-001",
        }
    )

    assert result["commit_error"] == "commit failed"
    assert result["pointer_published"] is False
    assert fake_dict.values == {}


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

    assert snapshot["pair_history_ref"] == arena.rating_pair_history_ref(
        "arena-a",
        "elo-test",
    ).as_posix()
    assert snapshot["scheduler_state_ref"] == arena.rating_scheduler_state_ref(
        "arena-a",
        "elo-test",
    ).as_posix()
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
    assert {row["rating"] for row in snapshot["ratings"]} == {
        arena.DEFAULT_RATING_INITIAL_RATING
    }
    assert snapshot["pair_rating_results"][0]["rating_skip_reason"] == "not_enough_valid_games"


def test_tournament_artifact_ref_validation_is_strict() -> None:
    good = (
        "tournaments/curvytron/arena-a/battles/pair-000000/games/"
        "game-000000/game.gif"
    )
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
                            "tournaments/curvytron/arena-a/battles/"
                            "pair-indexed/battle.json"
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
    checkpoint_rows = {
        row["battle_id"]: row
        for row in checkpoint_payload["rows"]
    }
    battles = modal_arena._list_battle_index(
        tmp_path,
        tournament_id=tournament_id,
        checkpoint_id="ckpt-a",
        limit=10,
        offset=0,
    )

    assert payload["checkpoint_index_count"] == 3
    assert payload["ref"] == arena.tournament_battle_index_ref(tournament_id).as_posix()
    assert checkpoint_payload["ref"] == arena.tournament_checkpoint_battle_index_ref(
        tournament_id,
        "ckpt-a",
    ).as_posix()
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
                        "tournaments/curvytron/arena-a/battles/battle-ab/"
                        "games/game-000000/game.gif"
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
                    f"tournaments/curvytron/arena-a/battles/battle-ab/"
                    f"games/{game_id}/game.gif"
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
        "gif_ref": (
            "tournaments/curvytron/arena-a/battles/battle-live/"
            "games/game-000000/game.gif"
        ),
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

    assert progress["count_basis"] == "shard_summary_files"
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

    assert "Rankings" in html
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
    assert "/gif?ref=tournaments/curvytron/arena-a/battles/battle-ab/games/game-000004/game.gif" in html
    assert 'loading="lazy"' in html
    assert 'decoding="async"' in html
    assert "/meta?ref=tournaments/curvytron/arena-a/battles/battle-ab/games/game-000005/summary.json" in html
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
    assert 'window.location.assign(url.toString())' in html
    assert 'url.searchParams.delete("checkpoint_id")' in html
    assert 'url.searchParams.delete("battle_id")' in html
    assert 'url.searchParams.delete("fresh")' in html
    assert 'url.searchParams.delete("rating_run_id")' in html


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
    written = modal_arena._read_json(
        tmp_path / arena.rating_progress_ref("arena-a", "elo-test")
    )

    assert written["status"] == "pending"
    assert written["phase"] == "waiting_for_round_input"
    assert written["input_ref"].endswith("/input.json")


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
    assert "/gif?ref=tournaments/curvytron/arena-a/battles/battle-ab/games/game-000004/game.gif" in html
    assert "/meta?ref=tournaments/curvytron/arena-a/battles/battle-ab/games/game-000005/summary.json" in html


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
    (ckpt_dir / "iteration_10000.pth.tar").write_bytes(b"new")
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
    assert discovery["checkpoint_refs"] == [
        (
            "training/lightzero-curvytron-visual-survival/"
            f"{run_id}/attempts/{attempt_id}/train/lightzero_exp/ckpt/"
            "iteration_10000.pth.tar"
        )
    ]


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
    timestamped = (
        train_root
        / "lightzero_exp_260513_123802"
        / "ckpt"
        / "iteration_180000.pth.tar"
    )
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
    assert "lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar" in discovery["checkpoint_refs"][0]


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
            train_root
            / "lightzero_exp_260513_123802"
            / "ckpt"
            / f"iteration_{iteration}.pth.tar"
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
    assert "lightzero_exp_260513_123802/ckpt/iteration_40000.pth.tar" in discovery["checkpoint_refs"][0]


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
    timestamped = (
        train_root
        / "lightzero_exp_260513_123802"
        / "ckpt"
        / "iteration_180000.pth.tar"
    )
    empty = (
        train_root
        / "lightzero_exp_260513_123802"
        / "ckpt"
        / "iteration_190000.pth.tar"
    )
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


def test_checkpoint_count_guard_does_not_treat_max_runs_as_checkpoint_count_for_all_selection() -> None:
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
            tmp_path
            / arena.game_summary_ref(tournament_id, pair["battle_id"], game["game_id"]),
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
    assert pair_only_progress["completed_game_count"] == 0
    assert pair_only_progress["partial_pair_count"] == 0
    assert pair_only_progress["estimated_seen_game_count"] == 3
    assert pair_only_progress["count_basis"] == "shard_summary_files"
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
    assert complete_pair_only_progress["recent_started_pairs"][0]["complete"] is True


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

    assert progress["count_basis"] == "shard_summary_files"
    assert progress["phase"] == "games_running"
    assert progress["started_pair_count"] == 1
    assert progress["completed_pair_count"] == 0
    assert progress["estimated_seen_game_count"] == 3
    assert progress["estimated_completion_fraction"] == 1.0


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
            tmp_path
            / arena.game_summary_ref(tournament_id, pair["battle_id"], game["game_id"]),
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
    assert "Rankings will appear as soon as finished games are visible" in html
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

    assert "Live Rankings" in html
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
