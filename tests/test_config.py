from curvyzero.env import CurvyTronConfig


def test_rules_hash_is_stable_for_same_config():
    first = CurvyTronConfig()
    second = CurvyTronConfig()

    assert first.rules_hash == second.rules_hash


def test_rules_hash_changes_for_behavior_change():
    first = CurvyTronConfig()
    second = CurvyTronConfig(speed=1.25)

    assert first.rules_hash != second.rules_hash
