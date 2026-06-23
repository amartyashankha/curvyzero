from __future__ import annotations

import pytest

from curvyzero.contracts.curvytron_naming import (
    CURVYTRON_CANARY_BATCH,
    CURVYTRON_GRID_A_BATCH,
    action_noise_tag,
    curvytron_attempt_id,
    curvytron_row_id,
    curvytron_run_id,
    leaderboard_immortal_tag,
    reward_alpha_tag,
)


def test_current_curvytron_run_name_is_short_and_explanatory() -> None:
    run_id = curvytron_run_id(
        batch=CURVYTRON_GRID_A_BATCH,
        row_number=17,
        reward_tag=reward_alpha_tag(0.33),
        noise_tag=action_noise_tag(0.10),
        immortal_tag=leaderboard_immortal_tag(0.0),
        recipe_code="b20w05r1",
    )

    assert run_id == "cz26a-r017-out33-n10-imm0-b20w05r1"
    assert curvytron_attempt_id(run_id) == "try-cz26a-r017-out33-n10-imm0-b20w05r1"


def test_canary_batch_name_and_row_id() -> None:
    assert CURVYTRON_CANARY_BATCH == "cz26c"
    assert curvytron_row_id(1) == "r001"


def test_grid_b_out50_reward_alpha_name_is_supported() -> None:
    run_id = curvytron_run_id(
        batch="cz26b",
        row_number=3,
        reward_tag=reward_alpha_tag(0.5),
        noise_tag=action_noise_tag(0.0),
        immortal_tag=leaderboard_immortal_tag(0.10),
        recipe_code="b100",
    )

    assert run_id == "cz26b-r003-out50-n0-imm10-b100"


@pytest.mark.parametrize("alpha", [0.2, 2.0])
def test_reward_alpha_tag_rejects_unplanned_visible_names(alpha: float) -> None:
    with pytest.raises(ValueError, match="unsupported reward outcome alpha"):
        reward_alpha_tag(alpha)
