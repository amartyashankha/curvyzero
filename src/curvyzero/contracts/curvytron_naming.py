"""Short operator-facing names for current CurvyTron experiments."""

from __future__ import annotations

import re


CURRENT_CURVYTRON_EXPERIMENT_PREFIX = "cz26"
CURVYTRON_GRID_A_BATCH = f"{CURRENT_CURVYTRON_EXPERIMENT_PREFIX}a"
CURVYTRON_GRID_B_BATCH = f"{CURRENT_CURVYTRON_EXPERIMENT_PREFIX}b"
CURVYTRON_CANARY_BATCH = f"{CURRENT_CURVYTRON_EXPERIMENT_PREFIX}c"

REWARD_ALPHA_TAGS: dict[float, str] = {
    0.0: "out0",
    0.33: "out33",
    0.5: "out50",
    0.67: "out67",
    1.0: "out100",
}

ACTION_NOISE_TAGS: dict[float, str] = {
    0.0: "n0",
    0.10: "n10",
    0.20: "n20",
}

LEADERBOARD_IMMORTAL_TAGS: dict[float, str] = {
    0.0: "imm0",
    0.10: "imm10",
}

RUN_ID_PATTERN = re.compile(
    r"^cz26[abc]-r[0-9]{3}-out(?:0|33|50|67|100)-n(?:0|10|20)-imm(?:0|10)-[a-z0-9]+(?:-[a-z0-9]+)*$"
)


def reward_alpha_tag(alpha: float) -> str:
    key = round(float(alpha), 2)
    try:
        return REWARD_ALPHA_TAGS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported reward outcome alpha for short name: {alpha!r}") from exc


def action_noise_tag(probability: float) -> str:
    key = round(float(probability), 2)
    try:
        return ACTION_NOISE_TAGS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported action noise probability for short name: {probability!r}") from exc


def leaderboard_immortal_tag(probability: float) -> str:
    key = round(float(probability), 2)
    try:
        return LEADERBOARD_IMMORTAL_TAGS[key]
    except KeyError as exc:
        raise ValueError(
            f"unsupported leaderboard immortal probability for short name: {probability!r}"
        ) from exc


def curvytron_row_id(row_number: int) -> str:
    row = int(row_number)
    if row < 1 or row > 999:
        raise ValueError(f"row_number must be in 1..999, got {row_number!r}")
    return f"r{row:03d}"


def curvytron_run_id(
    *,
    batch: str,
    row_number: int,
    reward_tag: str,
    noise_tag: str,
    immortal_tag: str,
    recipe_code: str,
) -> str:
    row_id = curvytron_row_id(row_number)
    run_id = f"{batch}-{row_id}-{reward_tag}-{noise_tag}-{immortal_tag}-{recipe_code}"
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"invalid current CurvyTron run id: {run_id!r}")
    return run_id


def curvytron_attempt_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"invalid current CurvyTron run id: {run_id!r}")
    return f"try-{run_id}"
