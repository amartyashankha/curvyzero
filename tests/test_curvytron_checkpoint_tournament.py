from __future__ import annotations

import json

import pytest

from curvyzero.infra.modal import curvyzero_checkpoint_tournament as modal_arena
from curvyzero.tournament import curvytron_checkpoint_tournament as arena


def _checkpoint_ref(run_id: str, iteration: int) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/"
        f"{run_id}/checkpoints/lightzero/iteration_{iteration}.pth.tar"
    )


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
    games = []
    for index, winner in enumerate([0, 1, None]):
        draw = winner is None
        games.append(
            {
                "ok": True,
                "game_id": f"game-{index:06d}",
                "game_index": index,
                "players": pair["players"],
                "physical_steps": 10 + index,
                "score": {
                    "outcome": "draw" if draw else f"seat_{winner}_win",
                    "winner_seat": winner,
                    "loser_seat": None if draw else 1 - int(winner),
                    "draw": draw,
                    "physical_steps": 10 + index,
                },
                "summary_ref": f"tournaments/curvytron/arena-a/battles/{pair['battle_id']}/games/game-{index:06d}/summary.json",
            }
        )

    summary = arena.summarize_pair_results(pair, games)
    standings = arena.standings_from_pair_results([summary])

    assert summary["tally"]["completed_count"] == 3
    assert summary["tally"]["wins_by_seat"] == {"seat_0": 1, "seat_1": 1}
    assert summary["tally"]["draw_count"] == 1
    assert standings["checkpoint_count"] == 2
    assert {row["games"] for row in standings["standings"]} == {3}


def test_tournament_artifact_ref_validation_is_strict() -> None:
    good = (
        "tournaments/curvytron/arena-a/battles/pair-000000/games/"
        "game-000000/game.gif"
    )
    assert arena.validate_tournament_artifact_ref(good).as_posix() == good

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
    monkeypatch.setattr(modal_arena, "RUNS_MOUNT", tmp_path)
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
