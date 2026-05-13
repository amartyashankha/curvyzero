"""Probe a scripted wall-avoidant CurvyTron opponent in the vector env.

This is intentionally local tooling, not trainer launch plumbing. It exercises
the real VectorMultiplayerEnv transition while making selected players death
immune so the wall policy can be judged on geometry instead of early round
termination.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv


LEFT_ACTION = 0
STRAIGHT_ACTION = 1
RIGHT_ACTION = 2
ACTION_TO_SOURCE_MOVE = {
    LEFT_ACTION: -1.0,
    STRAIGHT_ACTION: 0.0,
    RIGHT_ACTION: 1.0,
}
ACTION_NAMES = {
    LEFT_ACTION: "left",
    STRAIGHT_ACTION: "straight",
    RIGHT_ACTION: "right",
    -1: "dead_noop",
}
WALL_NAMES = ("left", "right", "top", "bottom")
INTERIOR_NORMALS = np.asarray(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class PolicySpec:
    kind: str
    safe_margin: float
    trigger_margin: float | None = None
    normal_bias: float = 0.0
    lookahead_steps: int = 6
    behavior_seed: int = 0
    description: str = ""


def _clearances(env: VectorMultiplayerEnv, *, player: int) -> np.ndarray:
    state = env.state
    pos = state["pos"][:, player]
    radius = state["radius"][:, player]
    map_size = state["map_size"]
    return np.stack(
        [
            pos[:, 0] - radius,
            map_size - (pos[:, 0] + radius),
            pos[:, 1] - radius,
            map_size - (pos[:, 1] + radius),
        ],
        axis=1,
    )


def _heading_vectors(heading: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.cos(heading), np.sin(heading)


def _post_action_heading(
    env: VectorMultiplayerEnv,
    *,
    player: int,
    action: int,
) -> np.ndarray:
    state = env.state
    heading = state["heading"][:, player]
    angular_velocity = state["angular_velocity_per_ms"][:, player]
    decision_ms = np.full(env.batch_size, env.decision_ms, dtype=np.float64)
    return heading + ACTION_TO_SOURCE_MOVE[action] * angular_velocity * decision_ms


def _choose_turn_toward_vector(
    env: VectorMultiplayerEnv,
    *,
    player: int,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    target_norm = np.hypot(target_x, target_y)
    heading_x, heading_y = _heading_vectors(env.state["heading"][:, player])
    target_x = np.where(target_norm > 1e-9, target_x / target_norm, heading_x)
    target_y = np.where(target_norm > 1e-9, target_y / target_norm, heading_y)

    left_heading = _post_action_heading(env, player=player, action=LEFT_ACTION)
    right_heading = _post_action_heading(env, player=player, action=RIGHT_ACTION)
    left_score = np.cos(left_heading) * target_x + np.sin(left_heading) * target_y
    right_score = np.cos(right_heading) * target_x + np.sin(right_heading) * target_y
    return np.where(right_score > left_score, RIGHT_ACTION, LEFT_ACTION).astype(np.int16)


def _turn_or_straight_toward_vector(
    env: VectorMultiplayerEnv,
    *,
    player: int,
    target_x: np.ndarray,
    target_y: np.ndarray,
    straight_dot: float,
) -> np.ndarray:
    """Choose straight when already aligned, else the better legal turn."""

    target_norm = np.hypot(target_x, target_y)
    heading_x, heading_y = _heading_vectors(env.state["heading"][:, player])
    unit_x = np.where(target_norm > 1e-9, target_x / target_norm, heading_x)
    unit_y = np.where(target_norm > 1e-9, target_y / target_norm, heading_y)
    actions = _choose_turn_toward_vector(
        env,
        player=player,
        target_x=unit_x,
        target_y=unit_y,
    )
    aligned = (heading_x * unit_x + heading_y * unit_y) >= straight_dot
    actions[aligned] = STRAIGHT_ACTION
    actions[~env.state["alive"][:, player]] = -1
    return actions


def _nearest_wall_geometry(
    env: VectorMultiplayerEnv,
    *,
    player: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clearances = _clearances(env, player=player)
    nearest_wall = clearances.argmin(axis=1)
    nearest_clearance = clearances[np.arange(env.batch_size), nearest_wall]
    normals = INTERIOR_NORMALS[nearest_wall]
    return clearances, nearest_clearance, nearest_wall, normals


def wall_avoidant_actions(
    env: VectorMultiplayerEnv,
    *,
    player: int,
    safe_margin: float,
    trigger_margin: float | None = None,
) -> np.ndarray:
    """Return scripted actions for one player across env rows.

    The policy builds a nearest-wall danger field in world coordinates. Away
    vectors point from dangerous walls toward the arena interior. When all wall
    clearances exceed ``safe_margin`` it goes straight; otherwise it chooses the
    left/right turn whose one-step heading aligns better with that away vector.
    """

    state = env.state
    heading = state["heading"][:, player]
    clearances, nearest_clearance, _, _ = _nearest_wall_geometry(env, player=player)
    resolved_trigger_margin = safe_margin if trigger_margin is None else trigger_margin

    left_clearance = clearances[:, 0]
    right_clearance = clearances[:, 1]
    top_clearance = clearances[:, 2]
    bottom_clearance = clearances[:, 3]
    danger_left = np.maximum(0.0, safe_margin - left_clearance)
    danger_right = np.maximum(0.0, safe_margin - right_clearance)
    danger_top = np.maximum(0.0, safe_margin - top_clearance)
    danger_bottom = np.maximum(0.0, safe_margin - bottom_clearance)
    away_x = danger_left - danger_right
    away_y = danger_top - danger_bottom
    away_norm = np.hypot(away_x, away_y)
    no_field = away_norm <= 1e-9
    away_x = np.where(no_field, np.cos(heading), away_x / np.maximum(away_norm, 1e-9))
    away_y = np.where(no_field, np.sin(heading), away_y / np.maximum(away_norm, 1e-9))

    actions = _choose_turn_toward_vector(
        env,
        player=player,
        target_x=away_x,
        target_y=away_y,
    )
    actions[nearest_clearance > resolved_trigger_margin] = STRAIGHT_ACTION
    actions[~state["alive"][:, player]] = -1
    return actions


def lazy_weave_actions(
    env: VectorMultiplayerEnv,
    *,
    player: int,
    safe_margin: float,
    decision_index: int,
    behavior_seed: int,
) -> np.ndarray:
    """Mostly-straight S-curve trail maker with force-field wall override."""

    baseline = wall_avoidant_actions(env, player=player, safe_margin=safe_margin)
    _, nearest_clearance, _, _ = _nearest_wall_geometry(env, player=player)
    rows = np.arange(env.batch_size, dtype=np.int64)
    period = 18
    turn_window = 3
    phase = (decision_index + rows * 5 + behavior_seed) % period
    cycle = (decision_index // period + rows + behavior_seed) % 2
    rhythm_action = np.full(env.batch_size, STRAIGHT_ACTION, dtype=np.int16)
    rhythm_action[phase < turn_window] = np.where(
        cycle[phase < turn_window] == 0,
        LEFT_ACTION,
        RIGHT_ACTION,
    )
    actions = np.where(
        nearest_clearance <= safe_margin,
        baseline,
        rhythm_action,
    ).astype(np.int16)
    actions[~env.state["alive"][:, player]] = -1
    return actions


def wall_follower_actions(
    env: VectorMultiplayerEnv,
    *,
    player: int,
    safe_margin: float,
    behavior_seed: int,
) -> np.ndarray:
    """Drift into a loose wall lane and follow it with legal turns."""

    baseline = wall_avoidant_actions(env, player=player, safe_margin=safe_margin)
    _, nearest_clearance, _, normals = _nearest_wall_geometry(env, player=player)
    rows = np.arange(env.batch_size, dtype=np.int64)
    orbit_sign = np.where((rows + behavior_seed) % 2 == 0, 1.0, -1.0)
    tangent_x = orbit_sign * normals[:, 1]
    tangent_y = -orbit_sign * normals[:, 0]

    lane_margin = safe_margin * 2.75
    lane_band = max(1.0, safe_margin * 1.25)
    correction = np.clip((lane_margin - nearest_clearance) / lane_band, -0.7, 1.2)
    target_x = tangent_x + correction * normals[:, 0]
    target_y = tangent_y + correction * normals[:, 1]
    lane_actions = _turn_or_straight_toward_vector(
        env,
        player=player,
        target_x=target_x,
        target_y=target_y,
        straight_dot=0.92,
    )
    actions = np.where(
        nearest_clearance <= safe_margin,
        baseline,
        lane_actions,
    ).astype(np.int16)
    actions[~env.state["alive"][:, player]] = -1
    return actions


def waypoint_patrol_actions(
    env: VectorMultiplayerEnv,
    *,
    player: int,
    safe_margin: float,
    decision_index: int,
    behavior_seed: int,
) -> np.ndarray:
    """Patrol four interior waypoints while wall danger takes priority."""

    baseline = wall_avoidant_actions(env, player=player, safe_margin=safe_margin)
    _, nearest_clearance, _, _ = _nearest_wall_geometry(env, player=player)
    rows = np.arange(env.batch_size, dtype=np.int64)
    phase = ((decision_index // 36) + rows + behavior_seed) % 4
    reverse = ((rows + behavior_seed) % 2) == 1
    phase = np.where(reverse, 3 - phase, phase)
    waypoints = np.asarray(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.25, 0.75],
        ],
        dtype=np.float64,
    )
    target = waypoints[phase]
    state = env.state
    pos = state["pos"][:, player]
    map_size = state["map_size"]
    target_x = target[:, 0] * map_size - pos[:, 0]
    target_y = target[:, 1] * map_size - pos[:, 1]
    patrol_actions = _turn_or_straight_toward_vector(
        env,
        player=player,
        target_x=target_x,
        target_y=target_y,
        straight_dot=0.95,
    )
    actions = np.where(
        nearest_clearance <= safe_margin,
        baseline,
        patrol_actions,
    ).astype(np.int16)
    actions[~state["alive"][:, player]] = -1
    return actions


def jitter_force_field_actions(
    env: VectorMultiplayerEnv,
    *,
    player: int,
    safe_margin: float,
    decision_index: int,
    behavior_seed: int,
) -> np.ndarray:
    """Sparse deterministic jitter when safe, force-field escape near walls."""

    baseline = wall_avoidant_actions(env, player=player, safe_margin=safe_margin)
    _, nearest_clearance, _, _ = _nearest_wall_geometry(env, player=player)
    rows = np.arange(env.batch_size, dtype=np.int64)
    hashed = (
        (rows + 1) * np.int64(1_103_515_245)
        + np.int64(decision_index + 17) * np.int64(12_345)
        + np.int64(behavior_seed) * np.int64(2_654_435_761)
    ) & np.int64(0x7FFFFFFF)
    bucket = hashed % 10_000
    jitter_actions = np.full(env.batch_size, STRAIGHT_ACTION, dtype=np.int16)
    jitter_actions[bucket < 450] = LEFT_ACTION
    jitter_actions[bucket >= 9_550] = RIGHT_ACTION
    actions = np.where(
        nearest_clearance <= safe_margin,
        baseline,
        jitter_actions,
    ).astype(np.int16)
    actions[~env.state["alive"][:, player]] = -1
    return actions


def margin_reflection_actions(
    env: VectorMultiplayerEnv,
    *,
    player: int,
    safe_margin: float,
    trigger_margin: float | None = None,
    normal_bias: float = 0.0,
) -> np.ndarray:
    """Choose turns toward a wall-reflected heading before contact.

    This stays inside the left/straight/right action contract. It does not
    teleport or clamp the avatar; it only turns toward the direction an ideal
    wall reflection would have produced.
    """

    state = env.state
    heading = state["heading"][:, player]
    _, nearest_clearance, _, normals = _nearest_wall_geometry(env, player=player)
    resolved_trigger_margin = safe_margin if trigger_margin is None else trigger_margin

    heading_x, heading_y = _heading_vectors(heading)
    inward_dot = heading_x * normals[:, 0] + heading_y * normals[:, 1]
    reflected_x = np.where(
        inward_dot < 0.0,
        heading_x - 2.0 * inward_dot * normals[:, 0],
        heading_x,
    )
    reflected_y = np.where(
        inward_dot < 0.0,
        heading_y - 2.0 * inward_dot * normals[:, 1],
        heading_y,
    )
    target_x = reflected_x + normal_bias * normals[:, 0]
    target_y = reflected_y + normal_bias * normals[:, 1]

    actions = _choose_turn_toward_vector(
        env,
        player=player,
        target_x=target_x,
        target_y=target_y,
    )
    moving_comfortably_inward = inward_dot > 0.35
    actions[(nearest_clearance > resolved_trigger_margin) | moving_comfortably_inward] = (
        STRAIGHT_ACTION
    )
    actions[~state["alive"][:, player]] = -1
    return actions


def predictive_reflection_actions(
    env: VectorMultiplayerEnv,
    *,
    player: int,
    safe_margin: float,
    lookahead_steps: int,
    normal_bias: float,
) -> np.ndarray:
    """Reflection-like policy that triggers from projected straight-line danger."""

    state = env.state
    pos = state["pos"][:, player].astype(np.float64, copy=True)
    heading = state["heading"][:, player].astype(np.float64, copy=True)
    radius = state["radius"][:, player]
    map_size = state["map_size"]
    speed = state["speed"][:, player]
    distance = speed * env.decision_ms / 1000.0

    predicted_clearance = _clearances(env, player=player)
    predicted_min = predicted_clearance.min(axis=1)
    predicted_wall = predicted_clearance.argmin(axis=1)
    heading_x, heading_y = _heading_vectors(heading)
    for _ in range(max(1, int(lookahead_steps))):
        pos[:, 0] += heading_x * distance
        pos[:, 1] += heading_y * distance
        clearances = np.stack(
            [
                pos[:, 0] - radius,
                map_size - (pos[:, 0] + radius),
                pos[:, 1] - radius,
                map_size - (pos[:, 1] + radius),
            ],
            axis=1,
        )
        step_min = clearances.min(axis=1)
        worse = step_min < predicted_min
        predicted_min = np.where(worse, step_min, predicted_min)
        predicted_wall = np.where(worse, clearances.argmin(axis=1), predicted_wall)

    normals = INTERIOR_NORMALS[predicted_wall]
    current_x, current_y = _heading_vectors(state["heading"][:, player])
    inward_dot = current_x * normals[:, 0] + current_y * normals[:, 1]
    reflected_x = np.where(
        inward_dot < 0.0,
        current_x - 2.0 * inward_dot * normals[:, 0],
        current_x,
    )
    reflected_y = np.where(
        inward_dot < 0.0,
        current_y - 2.0 * inward_dot * normals[:, 1],
        current_y,
    )
    actions = _choose_turn_toward_vector(
        env,
        player=player,
        target_x=reflected_x + normal_bias * normals[:, 0],
        target_y=reflected_y + normal_bias * normals[:, 1],
    )
    actions[predicted_min > safe_margin] = STRAIGHT_ACTION
    actions[~state["alive"][:, player]] = -1
    return actions


def rollout_clearance_actions(
    env: VectorMultiplayerEnv,
    *,
    player: int,
    safe_margin: float,
    lookahead_steps: int,
) -> np.ndarray:
    """Choose the legal action whose repeated arc keeps the largest clearance."""

    state = env.state
    pos0 = state["pos"][:, player].astype(np.float64, copy=True)
    heading0 = state["heading"][:, player].astype(np.float64, copy=True)
    radius = state["radius"][:, player]
    map_size = state["map_size"]
    speed = state["speed"][:, player]
    angular_velocity = state["angular_velocity_per_ms"][:, player]
    decision_ms = np.full(env.batch_size, env.decision_ms, dtype=np.float64)
    distance = speed * env.decision_ms / 1000.0

    scores: dict[int, np.ndarray] = {}
    for action in (LEFT_ACTION, STRAIGHT_ACTION, RIGHT_ACTION):
        pos = pos0.copy()
        heading = heading0.copy()
        min_clearance = _clearances(env, player=player).min(axis=1)
        move = ACTION_TO_SOURCE_MOVE[action]
        for _ in range(max(1, int(lookahead_steps))):
            heading += move * angular_velocity * decision_ms
            pos[:, 0] += np.cos(heading) * distance
            pos[:, 1] += np.sin(heading) * distance
            clearances = np.stack(
                [
                    pos[:, 0] - radius,
                    map_size - (pos[:, 0] + radius),
                    pos[:, 1] - radius,
                    map_size - (pos[:, 1] + radius),
                ],
                axis=1,
            )
            min_clearance = np.minimum(min_clearance, clearances.min(axis=1))
        scores[action] = min_clearance

    actions = np.full(env.batch_size, STRAIGHT_ACTION, dtype=np.int16)
    best_score = scores[STRAIGHT_ACTION].copy()
    for action in (LEFT_ACTION, RIGHT_ACTION):
        better = scores[action] > best_score + 1e-9
        actions[better] = action
        best_score = np.where(better, scores[action], best_score)
    straight_is_safe = scores[STRAIGHT_ACTION] > safe_margin
    actions[straight_is_safe] = STRAIGHT_ACTION
    actions[~state["alive"][:, player]] = -1
    return actions


def select_policy_actions(
    env: VectorMultiplayerEnv,
    *,
    player: int,
    spec: PolicySpec,
    decision_index: int,
) -> np.ndarray:
    if spec.kind == "proactive_force_field":
        return wall_avoidant_actions(
            env,
            player=player,
            safe_margin=spec.safe_margin,
            trigger_margin=spec.trigger_margin,
        )
    if spec.kind == "lazy_weave":
        return lazy_weave_actions(
            env,
            player=player,
            safe_margin=spec.safe_margin,
            decision_index=decision_index,
            behavior_seed=spec.behavior_seed,
        )
    if spec.kind == "wall_follower":
        return wall_follower_actions(
            env,
            player=player,
            safe_margin=spec.safe_margin,
            behavior_seed=spec.behavior_seed,
        )
    if spec.kind == "waypoint_patrol":
        return waypoint_patrol_actions(
            env,
            player=player,
            safe_margin=spec.safe_margin,
            decision_index=decision_index,
            behavior_seed=spec.behavior_seed,
        )
    if spec.kind == "jitter_force_field":
        return jitter_force_field_actions(
            env,
            player=player,
            safe_margin=spec.safe_margin,
            decision_index=decision_index,
            behavior_seed=spec.behavior_seed,
        )
    if spec.kind == "reactive_reflection_proxy":
        return wall_avoidant_actions(
            env,
            player=player,
            safe_margin=spec.safe_margin,
            trigger_margin=0.0 if spec.trigger_margin is None else spec.trigger_margin,
        )
    if spec.kind == "margin_reflection":
        return margin_reflection_actions(
            env,
            player=player,
            safe_margin=spec.safe_margin,
            trigger_margin=spec.trigger_margin,
            normal_bias=spec.normal_bias,
        )
    if spec.kind == "predictive_reflection":
        return predictive_reflection_actions(
            env,
            player=player,
            safe_margin=spec.safe_margin,
            lookahead_steps=spec.lookahead_steps,
            normal_bias=spec.normal_bias,
        )
    if spec.kind == "rollout_clearance":
        return rollout_clearance_actions(
            env,
            player=player,
            safe_margin=spec.safe_margin,
            lookahead_steps=spec.lookahead_steps,
        )
    raise ValueError(f"unknown policy kind {spec.kind!r}")


def _death_immunity_ids(mode: str, opponent_player: int) -> tuple[int, ...]:
    if mode == "all":
        return (0, 1)
    if mode == "opponent":
        return (opponent_player,)
    if mode == "none":
        return ()
    raise ValueError("death_immunity must be 'all', 'opponent', or 'none'")


def run_probe(
    *,
    seed: int,
    batch_size: int,
    steps: int,
    safe_margin: float,
    opponent_player: int,
    ego_action: int,
    both_players_wall_avoidant: bool,
    policy_kind: str,
    trigger_margin: float | None = None,
    normal_bias: float = 0.0,
    lookahead_steps: int = 6,
    behavior_seed: int = 0,
    death_immunity: str = "all",
) -> dict[str, Any]:
    if policy_kind not in {
        "proactive_force_field",
        "lazy_weave",
        "wall_follower",
        "waypoint_patrol",
        "jitter_force_field",
        "reactive_reflection_proxy",
        "margin_reflection",
        "predictive_reflection",
        "rollout_clearance",
    }:
        raise ValueError(
            "policy_kind must be one of proactive_force_field, lazy_weave, "
            "wall_follower, waypoint_patrol, jitter_force_field, "
            "reactive_reflection_proxy, margin_reflection, predictive_reflection, "
            "rollout_clearance"
        )
    resolved_trigger_margin = (
        0.0
        if policy_kind == "reactive_reflection_proxy" and trigger_margin is None
        else trigger_margin
    )
    spec = PolicySpec(
        kind=policy_kind,
        safe_margin=safe_margin,
        trigger_margin=resolved_trigger_margin,
        normal_bias=normal_bias,
        lookahead_steps=lookahead_steps,
        behavior_seed=behavior_seed,
    )
    immunity_ids = _death_immunity_ids(death_immunity, opponent_player)
    env = VectorMultiplayerEnv(
        batch_size=batch_size,
        player_count=2,
        seed=seed,
        max_ticks=steps + 1,
        body_capacity=max(4096, steps * 3),
        natural_bonus_spawn=False,
        death_immunity_player_ids=immunity_ids,
    )
    env.reset(seed=seed)
    action_counts: Counter[int] = Counter()
    total_step_counters: Counter[str] = Counter()
    out_of_bounds_counts = np.zeros(batch_size, dtype=np.int32)
    min_clearance_seen = np.full(batch_size, np.inf, dtype=np.float64)
    max_outside_by_row = np.zeros(batch_size, dtype=np.float64)
    first_out_of_bounds_step = np.full(batch_size, -1, dtype=np.int32)
    first_turn_step = np.full(batch_size, -1, dtype=np.int32)
    death_step = np.full((batch_size, 2), -1, dtype=np.int32)
    first_oob_pre_clearance = np.full(batch_size, np.nan, dtype=np.float64)
    first_oob_pre_wall = np.full(batch_size, -1, dtype=np.int16)
    first_oob_pre_inward_dot = np.full(batch_size, np.nan, dtype=np.float64)
    first_oob_action = np.full(batch_size, -1, dtype=np.int16)
    initial_pos = env.state["pos"][:, opponent_player].copy()
    initial_heading = env.state["heading"][:, opponent_player].copy()
    executed_steps = 0

    for step_index in range(steps):
        actions = np.full((batch_size, 2), int(ego_action), dtype=np.int16)
        _, pre_nearest, pre_wall, pre_normals = _nearest_wall_geometry(
            env,
            player=opponent_player,
        )
        pre_heading_x, pre_heading_y = _heading_vectors(env.state["heading"][:, opponent_player])
        pre_inward_dot = pre_heading_x * pre_normals[:, 0] + pre_heading_y * pre_normals[:, 1]
        actions[:, opponent_player] = select_policy_actions(
            env,
            player=opponent_player,
            spec=spec,
            decision_index=step_index,
        )
        if both_players_wall_avoidant:
            actions[:, 1 - opponent_player] = select_policy_actions(
                env,
                player=1 - opponent_player,
                spec=spec,
                decision_index=step_index,
            )
        for action in actions[:, opponent_player]:
            action_counts[int(action)] += 1
        newly_turning = (actions[:, opponent_player] != STRAIGHT_ACTION) & (first_turn_step < 0)
        first_turn_step[newly_turning] = step_index + 1
        pre_alive = env.state["alive"][:, :2].copy()
        batch = env.step(actions)
        executed_steps = step_index + 1
        total_step_counters.update(
            {
                str(name): int(value)
                for name, value in batch.info.get("step_counters", {}).items()
                if int(value)
            }
        )
        post_alive = env.state["alive"][:, :2]
        newly_dead = pre_alive & ~post_alive & (death_step < 0)
        death_step[newly_dead] = step_index + 1

        clearances = _clearances(env, player=opponent_player)
        nearest = clearances.min(axis=1)
        min_clearance_seen = np.minimum(min_clearance_seen, nearest)
        outside = np.maximum(0.0, -nearest)
        max_outside_by_row = np.maximum(max_outside_by_row, outside)
        outside_mask = outside > 0.0
        newly_outside = outside_mask & (first_out_of_bounds_step < 0)
        first_out_of_bounds_step[newly_outside] = step_index + 1
        first_oob_pre_clearance[newly_outside] = pre_nearest[newly_outside]
        first_oob_pre_wall[newly_outside] = pre_wall[newly_outside]
        first_oob_pre_inward_dot[newly_outside] = pre_inward_dot[newly_outside]
        first_oob_action[newly_outside] = actions[newly_outside, opponent_player]
        out_of_bounds_counts += outside_mask
        if bool(batch.done.any()):
            break

    inside_all_steps = out_of_bounds_counts == 0
    failure_rows = np.flatnonzero(~inside_all_steps)
    sample_failures = []
    for row in failure_rows[:10]:
        row_int = int(row)
        sample_failures.append(
            {
                "row": row_int,
                "first_out_of_bounds_step": int(first_out_of_bounds_step[row_int]),
                "first_turn_step": int(first_turn_step[row_int]),
                "out_of_bounds_steps": int(out_of_bounds_counts[row_int]),
                "initial_pos": initial_pos[row_int].astype(float).tolist(),
                "initial_heading": float(initial_heading[row_int]),
                "min_clearance": float(min_clearance_seen[row_int]),
                "max_outside": float(max_outside_by_row[row_int]),
                "pre_oob_clearance": float(first_oob_pre_clearance[row_int]),
                "pre_oob_wall": (
                    WALL_NAMES[int(first_oob_pre_wall[row_int])]
                    if first_oob_pre_wall[row_int] >= 0
                    else None
                ),
                "pre_oob_inward_dot": float(first_oob_pre_inward_dot[row_int]),
                "first_oob_action": ACTION_NAMES.get(
                    int(first_oob_action[row_int]),
                    str(int(first_oob_action[row_int])),
                ),
            }
        )
    first_turn_rows = first_turn_step >= 0
    action_total = max(1, sum(action_counts.values()))
    return {
        "seed": seed,
        "batch_size": batch_size,
        "steps": steps,
        "executed_steps": executed_steps,
        "safe_margin": safe_margin,
        "trigger_margin": resolved_trigger_margin,
        "normal_bias": normal_bias,
        "lookahead_steps": lookahead_steps,
        "behavior_seed": behavior_seed,
        "policy_kind": policy_kind,
        "opponent_player": opponent_player,
        "ego_action": ego_action,
        "both_players_wall_avoidant": both_players_wall_avoidant,
        "death_immunity_player_ids": list(immunity_ids),
        "action_counts": {str(k): int(v) for k, v in sorted(action_counts.items())},
        "action_mix": {
            ACTION_NAMES.get(k, str(k)): round(int(v) / action_total, 4)
            for k, v in sorted(action_counts.items())
        },
        "rows_inside_all_steps": int(inside_all_steps.sum()),
        "rows_with_oob": int((~inside_all_steps).sum()),
        "opponent_death_rows": int((death_step[:, opponent_player] >= 0).sum()),
        "ego_death_rows": int((death_step[:, 1 - opponent_player] >= 0).sum()),
        "first_turn_step": {
            "rows": int(first_turn_rows.sum()),
            "min": int(first_turn_step[first_turn_rows].min()) if first_turn_rows.any() else None,
            "median": (
                float(np.median(first_turn_step[first_turn_rows]))
                if first_turn_rows.any()
                else None
            ),
        },
        "min_clearance": {
            "min": float(min_clearance_seen.min()),
            "median": float(np.median(min_clearance_seen)),
            "p10": float(np.percentile(min_clearance_seen, 10)),
        },
        "max_outside": {
            "max": float(max_outside_by_row.max()),
            "median": float(np.median(max_outside_by_row)),
        },
        "out_of_bounds_steps": {
            "total": int(out_of_bounds_counts.sum()),
            "max_row": int(out_of_bounds_counts.max()),
        },
        "step_counters": dict(sorted(total_step_counters.items())),
        "body_write_cursor_max": int(env.state["body_write_cursor"].max()),
        "body_write_cursor_median": float(np.median(env.state["body_write_cursor"])),
        "sample_failure_cases": sample_failures,
    }


def default_suite(safe_margin: float, lookahead_steps: int) -> list[PolicySpec]:
    return [
        PolicySpec(
            kind="reactive_reflection_proxy",
            safe_margin=safe_margin,
            trigger_margin=0.0,
            description="contact-only proxy; waits for wall hit",
        ),
        PolicySpec(
            kind="margin_reflection",
            safe_margin=safe_margin,
            normal_bias=0.0,
            description="lead-margin pure reflected heading",
        ),
        PolicySpec(
            kind="margin_reflection",
            safe_margin=safe_margin,
            normal_bias=0.75,
            description="lead-margin reflected heading plus inward bias",
        ),
        PolicySpec(
            kind="predictive_reflection",
            safe_margin=safe_margin,
            normal_bias=0.75,
            lookahead_steps=lookahead_steps,
            description="straight-line lookahead selects reflected wall normal",
        ),
        PolicySpec(
            kind="rollout_clearance",
            safe_margin=safe_margin,
            lookahead_steps=lookahead_steps,
            description="short legal-action arc rollout",
        ),
        PolicySpec(
            kind="proactive_force_field",
            safe_margin=safe_margin,
            description="nearest-wall force field",
        ),
    ]


def hand_designed_suite(safe_margin: float) -> list[PolicySpec]:
    return [
        PolicySpec(
            kind="proactive_force_field",
            safe_margin=safe_margin,
            description="baseline nearest-wall force field",
        ),
        PolicySpec(
            kind="lazy_weave",
            safe_margin=safe_margin,
            description="mostly-straight periodic S-curves plus wall override",
        ),
        PolicySpec(
            kind="wall_follower",
            safe_margin=safe_margin,
            description="loose perimeter lane follower plus wall override",
        ),
        PolicySpec(
            kind="waypoint_patrol",
            safe_margin=safe_margin,
            description="four interior waypoints plus wall override",
        ),
        PolicySpec(
            kind="jitter_force_field",
            safe_margin=safe_margin,
            description="sparse deterministic jitter plus wall override",
        ),
    ]


def _settings_label(result: dict[str, Any], description: str = "") -> str:
    parts = [f"M={result['safe_margin']:g}"]
    if result.get("trigger_margin") is not None:
        parts.append(f"trigger={result['trigger_margin']:g}")
    if result.get("normal_bias"):
        parts.append(f"normal_bias={result['normal_bias']:g}")
    if result.get("policy_kind") == "predictive_reflection":
        parts.append(f"lookahead={result['lookahead_steps']}")
    if result.get("policy_kind") == "rollout_clearance":
        parts.append(f"lookahead={result['lookahead_steps']}")
    if result.get("behavior_seed"):
        parts.append(f"behavior_seed={result['behavior_seed']}")
    if description:
        parts.append(description)
    return ", ".join(parts)


def markdown_table(results: list[dict[str, Any]], descriptions: list[str] | None = None) -> str:
    descriptions = descriptions or ["" for _ in results]
    lines = [
        "| Policy variant | Settings | Steps | Starts | OOB/death results | Action mix |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for result, description in zip(results, descriptions, strict=True):
        mix = result["action_mix"]
        action_mix = (
            f"L {mix.get('left', 0.0):.3f}, "
            f"S {mix.get('straight', 0.0):.3f}, "
            f"R {mix.get('right', 0.0):.3f}"
        )
        deaths = (
            f"oob rows {result['rows_with_oob']}; "
            f"opp deaths {result['opponent_death_rows']}; ego deaths {result['ego_death_rows']}; "
            f"min clearance {result['min_clearance']['min']:.3f}"
        )
        lines.append(
            "| "
            f"{result['policy_kind']} | "
            f"{_settings_label(result, description)} | "
            f"{result['executed_steps']}/{result['steps']} | "
            f"{result['batch_size']} | "
            f"{deaths} | "
            f"{action_mix} |"
        )
    return "\n".join(lines)


def run_suite(
    *,
    seed: int,
    batch_size: int,
    steps: int,
    safe_margin: float,
    opponent_player: int,
    ego_action: int,
    both_players_wall_avoidant: bool,
    lookahead_steps: int,
    suite_kind: str,
    behavior_seed: int,
    death_immunity: str,
) -> dict[str, Any]:
    if suite_kind == "wall_stress":
        specs = default_suite(safe_margin, lookahead_steps)
    elif suite_kind == "hand_designed":
        specs = hand_designed_suite(safe_margin)
    else:
        raise ValueError("suite_kind must be 'wall_stress' or 'hand_designed'")
    results = [
        run_probe(
            seed=seed,
            batch_size=batch_size,
            steps=steps,
            safe_margin=spec.safe_margin,
            opponent_player=opponent_player,
            ego_action=ego_action,
            both_players_wall_avoidant=both_players_wall_avoidant,
            policy_kind=spec.kind,
            trigger_margin=spec.trigger_margin,
            normal_bias=spec.normal_bias,
            lookahead_steps=spec.lookahead_steps,
            behavior_seed=behavior_seed + spec.behavior_seed,
            death_immunity=death_immunity,
        )
        for spec in specs
    ]
    descriptions = [spec.description for spec in specs]
    return {
        "seed": seed,
        "batch_size": batch_size,
        "steps": steps,
        "death_immunity": death_immunity,
        "suite_kind": suite_kind,
        "results": results,
        "markdown_table": markdown_table(results, descriptions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--safe-margin", type=float, default=20.0)
    parser.add_argument("--opponent-player", type=int, choices=(0, 1), default=1)
    parser.add_argument("--ego-action", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--both-players-wall-avoidant", action="store_true")
    parser.add_argument(
        "--policy-kind",
        choices=(
            "proactive_force_field",
            "lazy_weave",
            "wall_follower",
            "waypoint_patrol",
            "jitter_force_field",
            "reactive_reflection_proxy",
            "margin_reflection",
            "predictive_reflection",
            "rollout_clearance",
        ),
        default="proactive_force_field",
    )
    parser.add_argument("--trigger-margin", type=float, default=None)
    parser.add_argument("--normal-bias", type=float, default=0.0)
    parser.add_argument("--lookahead-steps", type=int, default=6)
    parser.add_argument("--behavior-seed", type=int, default=0)
    parser.add_argument("--death-immunity", choices=("all", "opponent", "none"), default="all")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument(
        "--suite-kind",
        choices=("wall_stress", "hand_designed"),
        default="wall_stress",
    )
    args = parser.parse_args()
    if args.suite:
        result = run_suite(
            seed=args.seed,
            batch_size=args.batch_size,
            steps=args.steps,
            safe_margin=args.safe_margin,
            opponent_player=args.opponent_player,
            ego_action=args.ego_action,
            both_players_wall_avoidant=args.both_players_wall_avoidant,
            lookahead_steps=args.lookahead_steps,
            suite_kind=args.suite_kind,
            behavior_seed=args.behavior_seed,
            death_immunity=args.death_immunity,
        )
    else:
        result = run_probe(
            seed=args.seed,
            batch_size=args.batch_size,
            steps=args.steps,
            safe_margin=args.safe_margin,
            opponent_player=args.opponent_player,
            ego_action=args.ego_action,
            both_players_wall_avoidant=args.both_players_wall_avoidant,
            policy_kind=args.policy_kind,
            trigger_margin=args.trigger_margin,
            normal_bias=args.normal_bias,
            lookahead_steps=args.lookahead_steps,
            behavior_seed=args.behavior_seed,
            death_immunity=args.death_immunity,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
