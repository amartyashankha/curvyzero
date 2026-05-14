from __future__ import annotations

import math

from curvyzero.tournament import curvytron_checkpoint_tournament as arena


def _checkpoint_ref(index: int) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/"
        f"guardrail-run-{index:03d}/checkpoints/lightzero/iteration_{index}.pth.tar"
    )


def _rating_spec(count: int, *, pairs_per_round: int) -> dict:
    return arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "scheduler-guardrails",
            "checkpoints": [_checkpoint_ref(index) for index in range(count)],
            "pair_selection": "adaptive_v0",
            "pairs_per_round": pairs_per_round,
            "seed": 101,
        }
    )


def _played_checkpoint_ids(pairs: list[dict]) -> set[str]:
    return {
        player["checkpoint_id"]
        for pair in pairs
        for player in pair["players"]
    }


def _established_snapshot(
    rating_spec: dict,
    *,
    undercovered_index: int | None = None,
) -> dict:
    checkpoints = rating_spec["checkpoints"]
    rows = []
    for index, checkpoint in enumerate(checkpoints):
        opponent_count = 20
        games = 420
        if index == undercovered_index:
            opponent_count = 19
            games = 399
        opponents = [
            checkpoints[(index + offset) % len(checkpoints)]["checkpoint_id"]
            for offset in range(1, opponent_count + 1)
        ]
        rows.append(
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "rating": 1000.0 + index,
                "games": games,
                "distinct_opponents": len(opponents),
                "opponent_ids": opponents,
                "rated_battles": len(opponents),
            }
        )
    return {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "context_hash": arena.rating_context_hash(rating_spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(checkpoints),
        "ratings": rows,
    }


def test_default_placement_targets_do_not_expand_to_full_need() -> None:
    rating_spec = _rating_spec(40, pairs_per_round=30)

    pairs = arena.build_rating_round_pair_specs(rating_spec, round_index=0)

    assert len(pairs) == 30
    assert {pair["schedule_reason"] for pair in pairs} == {
        arena.SCHEDULE_REASON_PLACEMENT
    }


def test_absurdly_small_budget_only_expands_to_first_touch_floor() -> None:
    rating_spec = _rating_spec(5, pairs_per_round=1)

    pairs = arena.build_rating_round_pair_specs(rating_spec, round_index=0)

    assert len(pairs) == math.ceil(5 / 2)
    assert _played_checkpoint_ids(pairs) == {
        checkpoint["checkpoint_id"] for checkpoint in rating_spec["checkpoints"]
    }


def test_large_new_roster_keeps_requested_round_budget() -> None:
    rating_spec = _rating_spec(424, pairs_per_round=212)

    pairs = arena.build_rating_round_pair_specs(rating_spec, round_index=0)

    assert len(pairs) == 212
    assert len(pairs) != 4240


def test_single_undercovered_established_checkpoint_does_not_consume_round() -> None:
    rating_spec = _rating_spec(40, pairs_per_round=30)
    previous_snapshot = _established_snapshot(rating_spec, undercovered_index=3)

    pairs = arena.build_rating_round_pair_specs(
        rating_spec,
        previous_snapshot=previous_snapshot,
        round_index=1,
    )

    placement_count = len(
        [
            pair
            for pair in pairs
            if pair["schedule_reason"] == arena.SCHEDULE_REASON_PLACEMENT
        ]
    )
    assert len(pairs) == 30
    assert placement_count == 1
