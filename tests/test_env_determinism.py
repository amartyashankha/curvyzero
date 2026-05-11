import numpy as np

from curvyzero.env import CurvyTronConfig, CurvyTronEnv


def rollout(seed: int, action_sequence: list[dict[str, int]]) -> list[np.ndarray]:
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    observations = env.reset(seed=seed)
    trajectory = [observations["player_0"]]
    for actions in action_sequence:
        result = env.step(actions)
        trajectory.append(result.observations["player_0"])
        if result.terminated["player_0"] or result.truncated["player_0"]:
            break
    return trajectory


def test_same_seed_and_actions_are_deterministic():
    actions = [
        {"player_0": 1, "player_1": 1},
        {"player_0": 2, "player_1": 0},
        {"player_0": 2, "player_1": 0},
        {"player_0": 1, "player_1": 1},
    ]

    first = rollout(123, actions)
    second = rollout(123, actions)

    assert len(first) == len(second)
    for first_obs, second_obs in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_obs, second_obs)
