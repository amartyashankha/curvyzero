from __future__ import annotations

from typing import Any

import pytest

from curvyzero.tournament import curvytron_checkpoint_tournament as arena


def _checkpoint_ref(run_id: str, iteration: int) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/"
        f"{run_id}/checkpoints/lightzero/iteration_{iteration}.pth.tar"
    )


def _require_helper(name: str) -> Any:
    helper = getattr(arena, name, None)
    if helper is None:
        pytest.skip(
            "CurvyTron reintroduction validation helper "
            f"arena.{name} is not implemented yet"
        )
    return helper


def _base_rating_spec() -> dict[str, Any]:
    return arena.normalize_rating_spec(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-live",
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 2,
            "games_per_pair": 3,
            "seed": 17,
            "checkpoints": [
                {
                    "checkpoint_id": "ckpt-champion",
                    "label": "Champion",
                    "checkpoint_ref": _checkpoint_ref("run-champion", 100),
                    "checkpoint_state_key": "model",
                    "model_env_variant": "source_state_visual_survival",
                    "model_reward_variant": "survival_v3",
                    "policy_trail_render_mode": "browser_lines",
                },
                {
                    "checkpoint_id": "ckpt-anchor",
                    "label": "Anchor",
                    "checkpoint_ref": _checkpoint_ref("run-anchor", 80),
                    "checkpoint_state_key": "model",
                    "model_env_variant": "source_state_visual_survival",
                    "model_reward_variant": "survival_v3",
                    "policy_trail_render_mode": "browser_lines",
                },
                {
                    "checkpoint_id": "ckpt-middle",
                    "label": "Middle",
                    "checkpoint_ref": _checkpoint_ref("run-middle", 60),
                    "checkpoint_state_key": "model",
                    "model_env_variant": "source_state_visual_survival",
                    "model_reward_variant": "survival_v3",
                    "policy_trail_render_mode": "browser_lines",
                },
            ],
        }
    )


def _snapshot(rating_spec: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "rank": 1,
            "checkpoint_id": "ckpt-champion",
            "label": "Champion",
            "checkpoint_ref": _checkpoint_ref("run-champion", 100),
            "checkpoint_state_key": "model",
            "model_env_variant": "source_state_visual_survival",
            "model_reward_variant": "survival_v3",
            "policy_trail_render_mode": "browser_lines",
            "rating": 1725.0,
            "games": 42,
            "wins": 23,
            "losses": 12,
            "draws": 7,
            "distinct_opponents": 2,
            "opponent_ids": ["ckpt-anchor", "ckpt-middle"],
            "rated_battles": 2,
            "status": "active",
        },
        {
            "rank": 2,
            "checkpoint_id": "ckpt-anchor",
            "label": "Anchor",
            "checkpoint_ref": _checkpoint_ref("run-anchor", 80),
            "checkpoint_state_key": "model",
            "model_env_variant": "source_state_visual_survival",
            "model_reward_variant": "survival_v3",
            "policy_trail_render_mode": "browser_lines",
            "rating": 1510.0,
            "games": 42,
            "wins": 17,
            "losses": 15,
            "draws": 10,
            "distinct_opponents": 2,
            "opponent_ids": ["ckpt-champion", "ckpt-middle"],
            "rated_battles": 2,
            "status": "active",
        },
        {
            "rank": 3,
            "checkpoint_id": "ckpt-middle",
            "label": "Middle",
            "checkpoint_ref": _checkpoint_ref("run-middle", 60),
            "checkpoint_state_key": "model",
            "model_env_variant": "source_state_visual_survival",
            "model_reward_variant": "survival_v3",
            "policy_trail_render_mode": "browser_lines",
            "rating": 1480.0,
            "games": 42,
            "wins": 14,
            "losses": 18,
            "draws": 10,
            "distinct_opponents": 2,
            "opponent_ids": ["ckpt-champion", "ckpt-anchor"],
            "rated_battles": 2,
            "status": "active",
        },
    ]
    return {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "formula_version": arena.RATING_FORMULA_VERSION,
        "tournament_id": rating_spec["tournament_id"],
        "rating_run_id": rating_spec["rating_run_id"],
        "round_id": "round-000009",
        "round_index": 9,
        "pool_hash": arena.rating_pool_hash(rating_spec["checkpoints"]),
        "roster_hash": arena.rating_pool_hash(rating_spec["checkpoints"]),
        "context_hash": arena.rating_context_hash(rating_spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(
            rating_spec["checkpoints"]
        ),
        "rating_spec": {
            key: value for key, value in rating_spec.items() if key != "checkpoints"
        },
        "checkpoint_count": len(rows),
        "ratings": rows,
    }


def _pair_history(rating_spec: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for left, right, battle_count in [
        ("ckpt-champion", "ckpt-anchor", 4),
        ("ckpt-champion", "ckpt-middle", 3),
        ("ckpt-anchor", "ckpt-middle", 5),
    ]:
        rows.append(
            {
                "pair_key": arena.rating_pair_key(left, right),
                "checkpoint_ids": sorted([left, right]),
                "battle_count": battle_count,
                "rated_battle_count": battle_count,
                "game_count": battle_count * 3,
                "valid_game_count": battle_count * 3,
                "draw_count": battle_count,
                "failure_count": 0,
                "wins_by_checkpoint": {left: battle_count, right: battle_count - 1},
                "last_round_index": 9,
                "last_battle_id": f"battle-{left}-vs-{right}",
                "last_summary_ref": (
                    "tournaments/curvytron/arena-a/battles/"
                    f"battle-{left}-vs-{right}/battle.json"
                ),
            }
        )
    return {
        "schema_id": arena.PAIR_HISTORY_SCHEMA_ID,
        "tournament_id": rating_spec["tournament_id"],
        "rating_run_id": rating_spec["rating_run_id"],
        "pool_hash": arena.rating_pool_hash(rating_spec["checkpoints"]),
        "roster_hash": arena.rating_pool_hash(rating_spec["checkpoints"]),
        "context_hash": arena.rating_context_hash(rating_spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(
            rating_spec["checkpoints"]
        ),
        "updated_round_index": 9,
        "rows": rows,
    }


def test_extract_reintroduction_checkpoint_specs_preserves_identity_and_render_metadata() -> None:
    extract_specs = _require_helper("extract_reintroduction_checkpoint_specs")
    rating_spec = _base_rating_spec()

    specs = extract_specs(_snapshot(rating_spec))

    specs_by_id = {spec["checkpoint_id"]: spec for spec in specs}
    champion = specs_by_id["ckpt-champion"]
    assert champion["checkpoint_id"] == "ckpt-champion"
    assert champion["checkpoint_ref"] == _checkpoint_ref("run-champion", 100)
    assert champion["checkpoint_state_key"] == "model"
    assert champion["model_env_variant"] == "source_state_visual_survival"
    assert champion["model_reward_variant"] == "survival_v3"
    assert champion["policy_trail_render_mode"] == "browser_lines"


def test_reintroduction_validation_spec_defaults_to_rank_one_target() -> None:
    normalize_spec = _require_helper("normalize_reintroduction_validation_spec")
    rating_spec = _base_rating_spec()

    spec = normalize_spec(
        {
            "rating_spec": rating_spec,
            "rating_snapshot": _snapshot(rating_spec),
            "pair_history": _pair_history(rating_spec),
        }
    )

    assert spec["target_rank"] == 1
    assert spec["target_checkpoint_id"] == "ckpt-champion"


def test_reintroduction_purged_fork_removes_target_without_target_pair_history() -> None:
    build_fork = _require_helper("build_reintroduction_validation_fork")
    rating_spec = _base_rating_spec()

    fork = build_fork(
        rating_spec=rating_spec,
        previous_snapshot=_snapshot(rating_spec),
        pair_history=_pair_history(rating_spec),
        target_checkpoint_id="ckpt-champion",
    )

    fork_spec = arena.normalize_rating_spec(fork["rating_spec"])
    checkpoint_ids = {checkpoint["checkpoint_id"] for checkpoint in fork_spec["checkpoints"]}
    snapshot_ids = {
        row["checkpoint_id"] for row in fork["previous_snapshot"].get("ratings", [])
    }
    history_pairs = [
        set(row["checkpoint_ids"]) for row in fork["pair_history"].get("rows", [])
    ]

    assert "ckpt-champion" not in checkpoint_ids
    assert "ckpt-champion" not in snapshot_ids
    assert history_pairs
    assert all("ckpt-champion" not in pair for pair in history_pairs)
    assert {frozenset(pair) for pair in history_pairs} == {
        frozenset({"ckpt-anchor", "ckpt-middle"})
    }
    assert fork["target_checkpoint"]["checkpoint_id"] == "ckpt-champion"


def test_reintroduced_target_has_no_prior_pair_history_and_can_be_scheduled() -> None:
    build_fork = _require_helper("build_reintroduction_validation_fork")
    reintroduce = _require_helper("reintroduce_checkpoint_in_validation_fork")
    rating_spec = _base_rating_spec()
    fork = build_fork(
        rating_spec=rating_spec,
        previous_snapshot=_snapshot(rating_spec),
        pair_history=_pair_history(rating_spec),
        target_checkpoint_id="ckpt-champion",
    )

    state = reintroduce(fork)
    target_id = state["target_checkpoint"]["checkpoint_id"]
    pair_history = state.get("pair_history")
    pairs = arena.build_rating_round_pair_specs(
        state["rating_spec"],
        previous_snapshot=state.get("previous_snapshot"),
        scheduler_state=state.get("scheduler_state"),
        pair_history=pair_history,
        round_index=int(state.get("round_index", 10)),
    )

    target_pairs = [
        pair
        for pair in pairs
        if target_id in {player["checkpoint_id"] for player in pair["players"]}
    ]
    target_history_rows = [
        row
        for row in (pair_history or {}).get("rows", [])
        if target_id in set(row.get("checkpoint_ids") or [])
    ]

    assert target_history_rows == []
    assert target_pairs
    assert target_pairs[0]["schedule_reason"] == arena.SCHEDULE_REASON_PLACEMENT
    assert target_pairs[0]["schedule"]["prior_battle_count"] == 0
