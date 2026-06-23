from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from curvyzero.training.compact_policy_refresh_handoff import (
    COMPACT_POLICY_REFRESH_SEARCH_WORKER_STATE_SCHEMA_ID,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    build_compact_policy_refresh_handoff_v1,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_model_state_digest_v1,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_policy_refresh_metadata_from_state_v1,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    validate_compact_policy_refresh_handoff_v1,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    validate_compact_policy_refresh_search_worker_state_v1,
)


def test_compact_policy_refresh_handoff_accepts_distinct_refreshed_worker() -> None:
    parts = _valid_policy_refresh_parts()

    lineage = _build_policy_refresh_handoff(parts)

    validate_compact_policy_refresh_handoff_v1(lineage)
    assert lineage["checkpoint_id"] == "unit-policy-refresh-ckpt"
    assert lineage["learner_model_state_digest"] == parts.learner_digest
    assert lineage["search_worker_model_state_digest"] == parts.learner_digest
    assert lineage["search_worker_distinct_from_learner"] is True
    assert lineage["root_rows_stamped"] is True
    assert lineage["action_rows_stamped"] is True
    assert lineage["replay_rows_stamped"] is True
    assert lineage["sample_rows_stamped"] is True
    assert lineage["calls_train_muzero"] is False
    assert lineage["promotion_claim"] is False


def test_compact_model_state_digest_checks_finiteness_without_requiring_float_inputs() -> None:
    digest = compact_model_state_digest_v1(
        {
            "int_counter": np.asarray([1, 2, 3], dtype=np.int64),
            "bool_mask": np.asarray([True, False, True], dtype=np.bool_),
            "float_weight": np.asarray([0.25, 1.5], dtype=np.float32),
        }
    )

    assert len(digest) == 64
    with pytest.raises(ValueError, match="finite numeric values"):
        compact_model_state_digest_v1(
            {"bad_weight": np.asarray([0.0, np.nan], dtype=np.float32)}
        )
    with pytest.raises(ValueError, match="numeric values"):
        compact_model_state_digest_v1({"bad_label": np.asarray(["not-a-weight"])})


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda parts: parts.search_worker_state.update(
                {"model_state_digest": "stale-digest"}
            ),
            "digest mismatch",
        ),
        (
            lambda parts: parts.search_worker_state.update(
                {"learner_update_count": 2}
            ),
            "update count mismatch",
        ),
        (
            lambda parts: parts.search_worker_state.update(
                {"policy_version_ref": "policy:stale"}
            ),
            "policy_version_ref mismatch",
        ),
        (
            lambda parts: parts.search_worker_state.update({"refresh_applied": False}),
            "refresh_applied",
        ),
        (
            lambda parts: parts.search_worker_state.update({"cache_cleared": False}),
            "cache_cleared",
        ),
        (
            lambda parts: parts.search_worker_state.update(
                {"search_worker_model_object_id": id(parts.learner_model)}
            ),
            "distinct search worker",
        ),
        (
            lambda parts: parts.action_metadata.update(
                {"compact_policy_refresh_search_worker_refreshed": False}
            ),
            "action metadata missing refreshed marker",
        ),
        (
            lambda parts: parts.sample_metadata.update(
                {"model_version_ref": "model:stale"}
            ),
            "sample metadata model_version_ref mismatch",
        ),
    ],
)
def test_compact_policy_refresh_handoff_rejects_stale_worker_or_rows(
    mutate,
    match: str,
) -> None:
    parts = _valid_policy_refresh_parts()
    mutate(parts)

    with pytest.raises(ValueError, match=match):
        _build_policy_refresh_handoff(parts)


def _valid_policy_refresh_parts() -> SimpleNamespace:
    learner_model = {
        "weight": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "bias": np.asarray([0.25], dtype=np.float32),
    }
    learner_digest = compact_model_state_digest_v1(learner_model)
    resume_state = SimpleNamespace(
        trainer_id="unit-policy-refresh-trainer",
        policy_version_ref="policy:update-3",
        model_version_ref="model:update-3",
        policy_source="unit_test_policy_refresh",
        learner_update_count=3,
    )
    search_worker_state = {
        "schema_id": COMPACT_POLICY_REFRESH_SEARCH_WORKER_STATE_SCHEMA_ID,
        "search_impl": "unit_test_compact_torch_search",
        "policy_version_ref": resume_state.policy_version_ref,
        "model_version_ref": resume_state.model_version_ref,
        "policy_source": resume_state.policy_source,
        "learner_update_count": resume_state.learner_update_count,
        "model_state_digest": learner_digest,
        "search_worker_model_object_id": id(learner_model) + 1,
        "search_worker_object_id": id(learner_model) + 2,
        "refresh_count": 2,
        "refresh_applied": True,
        "cache_cleared": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }
    validate_compact_policy_refresh_search_worker_state_v1(search_worker_state)
    row_metadata = compact_policy_refresh_metadata_from_state_v1(search_worker_state)
    return SimpleNamespace(
        learner_model=learner_model,
        learner_digest=learner_digest,
        resume_state=resume_state,
        search_worker_state=search_worker_state,
        root_metadata=dict(row_metadata),
        action_metadata=dict(row_metadata),
        replay_metadata=dict(row_metadata),
        sample_metadata=dict(row_metadata),
    )


def _build_policy_refresh_handoff(parts: SimpleNamespace) -> dict[str, object]:
    return build_compact_policy_refresh_handoff_v1(
        checkpoint_id="unit-policy-refresh-ckpt",
        resume_state=parts.resume_state,
        learner_model=parts.learner_model,
        search_worker_state=parts.search_worker_state,
        root_metadata=parts.root_metadata,
        action_metadata=parts.action_metadata,
        replay_metadata=parts.replay_metadata,
        sample_metadata=parts.sample_metadata,
        evidence_refs=(
            "tests/test_compact_policy_refresh_handoff.py::"
            "test_compact_policy_refresh_handoff_accepts_distinct_refreshed_worker",
        ),
    )
