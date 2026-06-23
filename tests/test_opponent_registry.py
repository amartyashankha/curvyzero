import json

import pytest

from curvyzero.training.opponent_mixture import OPPONENT_MIXTURE_SCHEMA_ID
from curvyzero.training.opponent_registry import (
    OPPONENT_ASSIGNMENT_SCHEMA_ID,
    canonical_assignment_json_sha256,
    parse_opponent_assignment_snapshot,
)


def test_opponent_assignment_snapshot_parses_to_existing_mixture_contract():
    assignment = parse_opponent_assignment_snapshot(
        {
            "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
            "assignment_id": "overnight_pool_001",
            "source_epoch": 42,
            "source_ref": "training/opponents/overnight_pool_001.json",
            "seed": 7,
            "entries": [
                {
                    "name": "recent_winner",
                    "weight": 32,
                    "age_label": "recent",
                    "tags": ["tournament_winner"],
                    "opponent_policy_kind": "frozen_lightzero_checkpoint",
                    "opponent_checkpoint_ref": (
                        "training/run/checkpoints/lightzero/iteration_120000.pth.tar"
                    ),
                },
                {
                    "name": "blank_canvas",
                    "weight": 32,
                    "opponent_policy_kind": "fixed_straight",
                    "opponent_runtime_mode": "blank_canvas_noop",
                    "opponent_immortal": True,
                },
            ],
        }
    )

    assert assignment is not None
    assert assignment["schema_id"] == OPPONENT_ASSIGNMENT_SCHEMA_ID
    assert assignment["assignment_id"] == "overnight_pool_001"
    assert assignment["source_epoch"] == 42
    mixture = assignment["opponent_mixture"]
    assert mixture["schema_id"] == OPPONENT_MIXTURE_SCHEMA_ID
    assert mixture["seed"] == 7
    assert mixture["total_weight"] == 64.0
    assert [entry["name"] for entry in mixture["entries"]] == [
        "recent_winner",
        "blank_canvas",
    ]


def test_opponent_assignment_snapshot_accepts_json_string():
    assignment = parse_opponent_assignment_snapshot(
        json.dumps(
            {
                "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
                "assignment_id": "json_assignment",
                "entries": [
                    {
                        "name": "blank",
                        "weight": 1,
                        "opponent_policy_kind": "fixed_straight",
                        "opponent_immortal": True,
                    }
                ],
            }
        )
    )

    assert assignment is not None
    assert assignment["assignment_id"] == "json_assignment"


@pytest.mark.parametrize(
    "checkpoint_ref",
    [
        "training/run/checkpoints/lightzero/latest.pth.tar",
        "training/run/checkpoints/lightzero/ckpt_best.pth.tar",
        "training/run/checkpoints/lightzero/custom_ref.pth.tar",
    ],
)
def test_opponent_assignment_snapshot_rejects_mutable_or_non_iteration_refs(
    checkpoint_ref,
):
    with pytest.raises(ValueError, match="immutable exact iteration_N"):
        parse_opponent_assignment_snapshot(
            {
                "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
                "assignment_id": "bad_assignment",
                "entries": [
                    {
                        "name": "bad_frozen_ref",
                        "weight": 1,
                        "opponent_policy_kind": "frozen_lightzero_checkpoint",
                        "opponent_checkpoint_ref": checkpoint_ref,
                    }
                ],
            }
        )


def test_opponent_assignment_snapshot_requires_traceable_assignment_id():
    with pytest.raises(ValueError, match="assignment_id"):
        parse_opponent_assignment_snapshot(
            {
                "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
                "entries": [
                    {
                        "name": "blank",
                        "weight": 1,
                        "opponent_policy_kind": "fixed_straight",
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "schema_id",
    [
        None,
        "",
        "curvyzero_opponent_assignment/v1",
        "curvyzero_opponent_mixture/v0",
    ],
)
def test_opponent_assignment_snapshot_requires_exact_schema_id(schema_id):
    payload = {
        "assignment_id": "schema-test",
        "entries": [
            {
                "name": "blank",
                "weight": 1,
                "opponent_policy_kind": "fixed_straight",
            }
        ],
    }
    if schema_id is not None:
        payload["schema_id"] = schema_id

    with pytest.raises(ValueError, match="schema_id"):
        parse_opponent_assignment_snapshot(payload)


def test_canonical_assignment_json_sha256_is_stable_and_sensitive():
    assignment_a = {
        "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
        "assignment_id": "hash-test",
        "seed": 7,
        "entries": [
            {
                "name": "blank",
                "weight": 1,
                "opponent_policy_kind": "fixed_straight",
                "opponent_immortal": True,
            }
        ],
    }
    assignment_b = {
        "entries": [
            {
                "opponent_policy_kind": "fixed_straight",
                "opponent_immortal": True,
                "weight": 1,
                "name": "blank",
            }
        ],
        "seed": 7,
        "assignment_id": "hash-test",
        "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
    }
    changed = {
        **assignment_a,
        "entries": [
            {
                "name": "blank",
                "weight": 2,
                "opponent_policy_kind": "fixed_straight",
                "opponent_immortal": True,
            }
        ],
    }

    assert canonical_assignment_json_sha256(assignment_a) == (
        canonical_assignment_json_sha256(json.dumps(assignment_b, indent=2))
    )
    assert canonical_assignment_json_sha256(assignment_a) != (
        canonical_assignment_json_sha256(changed)
    )
