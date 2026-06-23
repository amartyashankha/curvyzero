from __future__ import annotations

from curvyzero.tournament import curvytron_checkpoint_tournament as arena
from curvyzero.training.opponent_leaderboard import (
    build_leaderboard_snapshot_from_rating_snapshot,
    validate_leaderboard_snapshot,
)


def _checkpoint_ref(index: int) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/"
        f"online-sim-run-{index:04d}/train/lightzero_exp_260516_000000/"
        f"ckpt/iteration_{index * 10}.pth.tar"
    )


def _rating_spec(
    count: int,
    *,
    pairs_per_round: int,
    placement_min_opponents: int = 20,
    games_per_pair: int = 21,
) -> dict:
    return arena.normalize_rating_spec(
        {
            "tournament_id": "online-sim-arena",
            "rating_run_id": "online-sim-elo",
            "checkpoints": [_checkpoint_ref(index) for index in range(count)],
            "pair_selection": arena.RATING_PAIR_SELECTION_ADAPTIVE_V0,
            "pairs_per_round": pairs_per_round,
            "games_per_pair": games_per_pair,
            "placement_min_games": placement_min_opponents * games_per_pair,
            "placement_min_opponents": placement_min_opponents,
            "active_pool_limit": arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT,
            "seed": 20260516,
        }
    )


def _mature_snapshot(
    spec: dict,
    *,
    established_count: int,
    new_count: int = 0,
) -> dict:
    checkpoints = spec["checkpoints"]
    rows = []
    for index, checkpoint in enumerate(checkpoints):
        checkpoint_id = str(checkpoint["checkpoint_id"])
        if index < established_count:
            opponent_count = min(20, max(0, established_count - 1))
            opponent_ids = [
                checkpoints[(index + offset) % established_count]["checkpoint_id"]
                for offset in range(1, opponent_count + 1)
            ]
            rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "label": checkpoint.get("label"),
                    "checkpoint_ref": checkpoint.get("checkpoint_ref"),
                    "rating": 2000.0 - float(index),
                    "games": 420,
                    "wins": 210,
                    "losses": 210,
                    "draws": 0,
                    "failure_count": 0,
                    "battles": opponent_count,
                    "rated_battles": opponent_count,
                    "opponent_ids": opponent_ids,
                    "distinct_opponents": len(opponent_ids),
                    "status": "active",
                    "rank": index + 1,
                    "last_round_delta": 0.0,
                }
            )
        elif index < established_count + new_count:
            rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "label": checkpoint.get("label"),
                    "checkpoint_ref": checkpoint.get("checkpoint_ref"),
                    "rating": 1500.0,
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "failure_count": 0,
                    "battles": 0,
                    "rated_battles": 0,
                    "opponent_ids": [],
                    "distinct_opponents": 0,
                    "status": "provisional",
                    "rank": 0,
                    "last_round_delta": 0.0,
                }
            )
        else:
            rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "label": checkpoint.get("label"),
                    "checkpoint_ref": checkpoint.get("checkpoint_ref"),
                    "rating": 800.0 - float(index),
                    "games": 420,
                    "wins": 100,
                    "losses": 320,
                    "draws": 0,
                    "failure_count": 0,
                    "battles": 20,
                    "rated_battles": 20,
                    "opponent_ids": [
                        checkpoints[offset]["checkpoint_id"]
                        for offset in range(min(20, established_count))
                    ],
                    "distinct_opponents": min(20, established_count),
                    "status": "active",
                    "rank": index + 1,
                    "last_round_delta": 0.0,
                }
            )
    return {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "context_hash": arena.rating_context_hash(spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(checkpoints),
        "ratings": rows,
    }


def _pair_summary(
    pair: dict,
    *,
    wins_by_checkpoint: dict[str, int],
    draw_count: int = 0,
) -> dict:
    games_per_pair = int(pair["games_per_pair"])
    assert sum(wins_by_checkpoint.values()) + draw_count == games_per_pair
    return {
        "battle_id": pair["battle_id"],
        "pair_index": int(pair["pair_index"]),
        "pair_key": pair["pair_key"],
        "players": pair["players"],
        "settings": {"games_per_pair": games_per_pair},
        "tally": {
            "game_count": games_per_pair,
            "completed_count": games_per_pair,
            "wins_by_checkpoint": wins_by_checkpoint,
            "draw_count": draw_count,
            "failure_count": 0,
        },
    }


def test_toy_weak_new_batch_is_placed_then_retired_from_top100() -> None:
    established_count = 100
    new_count = 10
    spec = _rating_spec(
        established_count + new_count,
        pairs_per_round=20 * new_count,
    )
    previous_snapshot = _mature_snapshot(
        spec,
        established_count=established_count,
        new_count=new_count,
    )
    pairs = arena.build_rating_round_pair_specs(
        spec,
        previous_snapshot=previous_snapshot,
        round_index=0,
    )
    checkpoint_ids = [checkpoint["checkpoint_id"] for checkpoint in spec["checkpoints"]]
    new_ids = set(checkpoint_ids[established_count:])
    assert len(pairs) == 20 * new_count
    assert all(pair["schedule_reason"] == arena.SCHEDULE_REASON_PLACEMENT for pair in pairs)

    summaries = []
    for pair in pairs:
        left = str(pair["players"][0]["checkpoint_id"])
        right = str(pair["players"][1]["checkpoint_id"])
        if left in new_ids and right not in new_ids:
            wins = {left: 0, right: int(pair["games_per_pair"])}
        elif right in new_ids and left not in new_ids:
            wins = {left: int(pair["games_per_pair"]), right: 0}
        else:
            wins = {left: 0, right: 0}
        summaries.append(_pair_summary(pair, wins_by_checkpoint=wins))

    snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=summaries,
        rating_spec=spec,
        previous_snapshot=previous_snapshot,
        round_index=0,
    )
    rows_by_id = {row["checkpoint_id"]: row for row in snapshot["ratings"]}
    assert {rows_by_id[checkpoint_id]["status"] for checkpoint_id in new_ids} == {
        "retired"
    }
    assert sum(row["status"] == "active" for row in snapshot["ratings"]) == 100

    pair_history = arena.pair_history_from_pair_results(
        summaries,
        rating_spec=spec,
        round_index=0,
    )
    next_pairs = arena.build_rating_round_pair_specs(
        spec,
        previous_snapshot=snapshot,
        pair_history=pair_history,
        round_index=1,
    )
    next_ids = {
        player["checkpoint_id"]
        for pair in next_pairs
        for player in pair["players"]
    }
    assert not (next_ids & new_ids)


def test_toy_clone_draw_swarm_has_deterministic_top100_public_pool() -> None:
    spec = _rating_spec(120, pairs_per_round=7140, placement_min_opponents=20)
    pairs = arena.build_rating_round_pair_specs(spec, round_index=0)
    summaries = [
        _pair_summary(
            pair,
            wins_by_checkpoint={
                pair["players"][0]["checkpoint_id"]: 0,
                pair["players"][1]["checkpoint_id"]: 0,
            },
            draw_count=int(pair["games_per_pair"]),
        )
        for pair in pairs
    ]

    snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=summaries,
        rating_spec=spec,
        round_index=0,
    )
    assert snapshot["max_abs_delta"] == 0.0
    assert snapshot["stable"] is True
    assert sum(row["status"] == "active" for row in snapshot["ratings"]) == 100
    assert sum(row["status"] == "retired" for row in snapshot["ratings"]) == 20

    leaderboard = validate_leaderboard_snapshot(
        build_leaderboard_snapshot_from_rating_snapshot(
            snapshot,
            leaderboard_id="online-sim",
            snapshot_id="snapshot-001",
            active_min_distinct_opponents=20,
            active_min_valid_games=420,
            max_active_rank=100,
        )
    )
    assert sum(row["status"] == "active" for row in leaderboard["rows"]) == 100
    assert sum(row["status"] == "retired" for row in leaderboard["rows"]) == 20


def test_adaptive_v0_scale_1000_existing_50_new_is_budgeted_and_top100_only() -> None:
    established_count = 1000
    new_count = 50
    spec = _rating_spec(
        established_count + new_count,
        pairs_per_round=1000,
    )
    previous_snapshot = _mature_snapshot(
        spec,
        established_count=established_count,
        new_count=new_count,
    )

    pairs = arena.build_rating_round_pair_specs(
        spec,
        previous_snapshot=previous_snapshot,
        round_index=0,
    )
    checkpoint_ids = [checkpoint["checkpoint_id"] for checkpoint in spec["checkpoints"]]
    new_ids = set(checkpoint_ids[established_count:])
    old_tail_ids = set(checkpoint_ids[100:established_count])
    played_ids = {
        player["checkpoint_id"]
        for pair in pairs
        for player in pair["players"]
    }

    assert len(pairs) == 1000
    assert len({pair["pair_key"] for pair in pairs}) == len(pairs)
    assert new_ids <= played_ids
    assert not (old_tail_ids & played_ids)
