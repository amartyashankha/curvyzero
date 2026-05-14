from __future__ import annotations

from collections.abc import Mapping, Sequence

from curvyzero.tournament import curvytron_checkpoint_tournament as arena


def _checkpoint(index: int) -> dict[str, str]:
    return {
        "checkpoint_ref": (
            "training/lightzero-curvytron-visual-survival/"
            f"scheduler-fairness-{index:02d}/checkpoints/lightzero/"
            f"iteration_{index * 10}.pth.tar"
        ),
        "checkpoint_id": f"ckpt-{index:02d}",
    }


def _pair_ids(pair: Mapping[str, object]) -> tuple[str, str]:
    players = pair["players"]
    assert isinstance(players, Sequence)
    return (
        str(players[0]["checkpoint_id"]),
        str(players[1]["checkpoint_id"]),
    )


def _history_row(
    left_id: str,
    right_id: str,
    *,
    battle_count: int = 1,
    game_count: int = 3,
    round_index: int = 0,
) -> dict[str, object]:
    pair_key = arena.rating_pair_key(left_id, right_id)
    return {
        "pair_key": pair_key,
        "checkpoint_ids": sorted([left_id, right_id]),
        "battle_count": battle_count,
        "rated_battle_count": battle_count,
        "game_count": battle_count * game_count,
        "valid_game_count": battle_count * game_count,
        "draw_count": 0,
        "failure_count": 0,
        "wins_by_checkpoint": {},
        "last_round_index": round_index,
        "last_battle_id": f"synthetic-{pair_key}",
        "last_summary_ref": None,
    }


def _pair_history(
    rating_spec: Mapping[str, object],
    rows: Sequence[Mapping[str, object]] = (),
    *,
    updated_round_index: int = -1,
) -> dict[str, object]:
    checkpoints = rating_spec["checkpoints"]
    assert isinstance(checkpoints, Sequence)
    pool_hash = arena.rating_pool_hash(checkpoints)
    return {
        "schema_id": arena.PAIR_HISTORY_SCHEMA_ID,
        "tournament_id": rating_spec["tournament_id"],
        "rating_run_id": rating_spec["rating_run_id"],
        "pool_hash": pool_hash,
        "roster_hash": pool_hash,
        "context_hash": arena.rating_context_hash(rating_spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(checkpoints),
        "updated_round_index": updated_round_index,
        "rows": sorted((dict(row) for row in rows), key=lambda row: str(row["pair_key"])),
    }


def _record_pair(
    pair_history: Mapping[str, object],
    left_id: str,
    right_id: str,
    *,
    game_count: int,
    round_index: int,
) -> dict[str, object]:
    rows = {
        str(row["pair_key"]): dict(row)
        for row in pair_history.get("rows", [])
        if isinstance(row, Mapping)
    }
    pair_key = arena.rating_pair_key(left_id, right_id)
    row = rows.get(
        pair_key,
        _history_row(
            left_id,
            right_id,
            battle_count=0,
            game_count=game_count,
            round_index=round_index,
        ),
    )
    row["battle_count"] = int(row.get("battle_count") or 0) + 1
    row["rated_battle_count"] = int(row.get("rated_battle_count") or 0) + 1
    row["game_count"] = int(row.get("game_count") or 0) + game_count
    row["valid_game_count"] = int(row.get("valid_game_count") or 0) + game_count
    row["last_round_index"] = round_index
    row["last_battle_id"] = f"synthetic-round-{round_index}-{pair_key}"
    rows[pair_key] = row
    return {
        **dict(pair_history),
        "updated_round_index": round_index,
        "rows": sorted(rows.values(), key=lambda item: str(item["pair_key"])),
    }


def _snapshot(
    rating_spec: Mapping[str, object],
    *,
    games_by_id: Mapping[str, int],
    opponents_by_id: Mapping[str, set[str]],
    rating_by_id: Mapping[str, float],
) -> dict[str, object]:
    checkpoints = rating_spec["checkpoints"]
    assert isinstance(checkpoints, Sequence)
    rows = []
    for checkpoint in checkpoints:
        checkpoint_id = str(checkpoint["checkpoint_id"])
        opponents = sorted(opponents_by_id.get(checkpoint_id, set()))
        rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "rating": float(rating_by_id[checkpoint_id]),
                "games": int(games_by_id.get(checkpoint_id, 0)),
                "distinct_opponents": len(opponents),
                "opponent_ids": opponents,
                "rated_battles": max(1, len(opponents)),
                "last_round_delta": 0.0,
                "failure_count": 0,
            }
        )
    return {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "context_hash": arena.rating_context_hash(rating_spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(checkpoints),
        "ratings": rows,
    }


def _leading_placement_count(pairs: Sequence[Mapping[str, object]]) -> int:
    for index, pair in enumerate(pairs):
        if pair["schedule_reason"] != arena.SCHEDULE_REASON_PLACEMENT:
            return index
    return len(pairs)


def test_adaptive_v0_reaches_placement_floors_across_rounds_before_repeats() -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-fairness",
            "checkpoints": [_checkpoint(index) for index in range(14)],
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 4,
            "games_per_pair": 3,
            "placement_min_games": 15,
            "placement_min_opponents": 4,
            "seed": 90210,
        }
    )
    checkpoint_ids = [str(checkpoint["checkpoint_id"]) for checkpoint in rating_spec["checkpoints"]]
    undercovered_ids = set(checkpoint_ids[:3])
    established_ids = checkpoint_ids[3:]
    opponents_by_id = {checkpoint_id: set() for checkpoint_id in checkpoint_ids}
    games_by_id = {checkpoint_id: 60 for checkpoint_id in checkpoint_ids}
    rating_by_id = {
        checkpoint_id: 1000.0 + index * 25.0
        for index, checkpoint_id in enumerate(checkpoint_ids)
    }

    for checkpoint_id in undercovered_ids:
        opponents_by_id[checkpoint_id] = undercovered_ids - {checkpoint_id}
        games_by_id[checkpoint_id] = 6
    for index, checkpoint_id in enumerate(established_ids):
        opponents_by_id[checkpoint_id] = {
            established_ids[(index + offset) % len(established_ids)]
            for offset in range(1, 5)
        }

    repeated_top_rows = [
        _history_row(established_ids[-1], established_ids[-2], battle_count=8),
        _history_row(established_ids[-1], established_ids[-3], battle_count=8),
        _history_row(established_ids[-2], established_ids[-3], battle_count=8),
    ]
    pair_history = _pair_history(rating_spec, repeated_top_rows)
    placement_pairs = []
    rounds_with_non_placement = []

    for round_index in range(3):
        previous_snapshot = _snapshot(
            rating_spec,
            games_by_id=games_by_id,
            opponents_by_id=opponents_by_id,
            rating_by_id=rating_by_id,
        )
        pairs = arena.build_rating_round_pair_specs(
            rating_spec,
            previous_snapshot=previous_snapshot,
            pair_history=pair_history,
            round_index=round_index,
        )
        placement_count = _leading_placement_count(pairs)

        assert placement_count > 0
        assert all(
            pair["schedule_reason"] != arena.SCHEDULE_REASON_PLACEMENT
            for pair in pairs[placement_count:]
        )
        if placement_count < len(pairs):
            rounds_with_non_placement.append(round_index)

        for pair in pairs[:placement_count]:
            left_id, right_id = _pair_ids(pair)
            undercovered_in_pair = {left_id, right_id} & undercovered_ids
            assert undercovered_in_pair
            assert pair["schedule"]["prior_battle_count"] == 0
            for checkpoint_id in undercovered_in_pair:
                opponent_id = right_id if checkpoint_id == left_id else left_id
                assert opponent_id not in opponents_by_id[checkpoint_id]
            placement_pairs.append(pair)

        for pair in pairs:
            left_id, right_id = _pair_ids(pair)
            opponents_by_id[left_id].add(right_id)
            opponents_by_id[right_id].add(left_id)
            games_by_id[left_id] += int(pair["games_per_pair"])
            games_by_id[right_id] += int(pair["games_per_pair"])
            pair_history = _record_pair(
                pair_history,
                left_id,
                right_id,
                game_count=int(pair["games_per_pair"]),
                round_index=round_index,
            )

    assert len(placement_pairs) == 9
    assert rounds_with_non_placement == [2]
    for checkpoint_id in undercovered_ids:
        assert games_by_id[checkpoint_id] >= rating_spec["placement_min_games"]
        assert len(opponents_by_id[checkpoint_id]) >= rating_spec["placement_min_opponents"]


def test_adaptive_v0_prefers_unplayed_near_rating_pair_over_played_pair() -> None:
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-fairness",
            "checkpoints": [_checkpoint(index) for index in range(4)],
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 1,
            "games_per_pair": 3,
            "placement_min_games": 0,
            "placement_min_opponents": 0,
            "seed": 90211,
        }
    )
    checkpoint_ids = [str(checkpoint["checkpoint_id"]) for checkpoint in rating_spec["checkpoints"]]
    low_id, fresh_id, played_id, top_id = checkpoint_ids
    previous_snapshot = _snapshot(
        rating_spec,
        games_by_id={checkpoint_id: 30 for checkpoint_id in checkpoint_ids},
        opponents_by_id={checkpoint_id: set() for checkpoint_id in checkpoint_ids},
        rating_by_id={
            low_id: 1200.0,
            fresh_id: 1598.0,
            played_id: 1599.0,
            top_id: 1600.0,
        },
    )
    pair_history = _pair_history(
        rating_spec,
        [_history_row(top_id, played_id, battle_count=5)],
    )

    pairs = arena.build_rating_round_pair_specs(
        rating_spec,
        previous_snapshot=previous_snapshot,
        pair_history=pair_history,
        round_index=7,
    )

    assert len(pairs) == 1
    assert pairs[0]["schedule_reason"] == arena.SCHEDULE_REASON_NEAR_RATING
    assert pairs[0]["pair_key"] == arena.rating_pair_key(top_id, fresh_id)
    assert pairs[0]["pair_key"] != arena.rating_pair_key(top_id, played_id)
    assert pairs[0]["schedule"]["prior_battle_count"] == 0
