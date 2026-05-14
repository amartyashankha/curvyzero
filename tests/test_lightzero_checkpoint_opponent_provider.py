import numpy as np

from curvyzero.training import lightzero_checkpoint_opponent_provider as provider


def test_infer_model_support_config_from_checkpoint_head_shapes():
    state_dict = {
        "dynamics_network.fc_reward_head.3.weight": np.zeros((3, 32), dtype=np.float32),
        "prediction_network.fc_value.3.weight": np.zeros((3, 32), dtype=np.float32),
    }

    assert provider._infer_model_support_config_from_state_dict(state_dict) == {
        "reward_support_size": 3,
        "value_support_size": 3,
        "support_scale": 1,
    }


def test_apply_inferred_model_support_config_patches_lightzero_model_config():
    main_config = {
        "policy": {
            "model": {
                "support_scale": 300,
                "reward_support_size": 601,
                "value_support_size": 601,
            }
        }
    }

    provider._apply_inferred_model_support_config(
        main_config,
        {
            "support_scale": 1,
            "reward_support_size": 3,
            "value_support_size": 3,
        },
    )

    assert main_config["policy"]["model"]["support_scale"] == 1
    assert main_config["policy"]["model"]["reward_support_size"] == 3
    assert main_config["policy"]["model"]["value_support_size"] == 3
