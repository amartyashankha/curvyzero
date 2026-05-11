from curvyzero.env import CurvyTronConfig, CurvyTronEnv


def test_wall_collision_terminates_round():
    env = CurvyTronEnv(
        CurvyTronConfig(width=16, height=16, spawn_margin=2, action_repeat=1, max_ticks=32)
    )
    env.reset(seed=1)

    result = None
    for _ in range(32):
        result = env.step({"player_0": 1, "player_1": 1})
        if result.terminated["player_0"]:
            break

    assert result is not None
    assert result.terminated["player_0"]
    assert result.terminated["player_1"]
