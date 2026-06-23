import sys
import json

import numpy as np

from curvyzero.env.observation_surface_contract import (
    DEFAULT_POLICY_OBSERVATION_BACKEND,
    POLICY_BONUS_RENDER_MODE,
    POLICY_OBSERVATION_CONTRACT_ID,
    POLICY_OBSERVATION_PERSPECTIVE,
    POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
    POLICY_OBSERVATION_SEAT_MAPPING,
    POLICY_STACK_SHAPE,
    POLICY_TRAIL_RENDER_MODE,
)
from curvyzero.training import lightzero_checkpoint_opponent_provider as provider


def _valid_policy_observation_sidecar() -> dict:
    return {
        "schema_id": "curvyzero_checkpoint_policy_metadata/v0",
        "policy_trail_render_mode": POLICY_TRAIL_RENDER_MODE,
        "policy_bonus_render_mode": POLICY_BONUS_RENDER_MODE,
        "policy_observation_backend": DEFAULT_POLICY_OBSERVATION_BACKEND,
        "policy_observation_contract_id": POLICY_OBSERVATION_CONTRACT_ID,
        "policy_observation_perspective_schema_id": (
            POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
        ),
        "observation_contract": {
            "contract_id": POLICY_OBSERVATION_CONTRACT_ID,
            "perspective_schema_id": POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
            "perspective": POLICY_OBSERVATION_PERSPECTIVE,
            "seat_mapping": POLICY_OBSERVATION_SEAT_MAPPING,
            "trail_render_mode": POLICY_TRAIL_RENDER_MODE,
            "bonus_render_mode": POLICY_BONUS_RENDER_MODE,
            "backend": DEFAULT_POLICY_OBSERVATION_BACKEND,
            "stack_shape": list(POLICY_STACK_SHAPE),
        },
    }


def test_checkpoint_provider_requires_policy_observation_metadata(tmp_path):
    checkpoint = tmp_path / "iteration_0.pth.tar"
    checkpoint.write_bytes(b"checkpoint")

    with np.testing.assert_raises_regex(ValueError, "required policy observation metadata"):
        provider.require_checkpoint_policy_observation_metadata(
            checkpoint_path=checkpoint,
            checkpoint_payload={"model": {}},
        )


def test_checkpoint_provider_accepts_current_policy_observation_sidecar(tmp_path):
    checkpoint = tmp_path / "iteration_0.pth.tar"
    checkpoint.write_bytes(b"checkpoint")
    provider.checkpoint_policy_metadata_sidecar_path(checkpoint).write_text(
        json.dumps(_valid_policy_observation_sidecar()),
        encoding="utf-8",
    )

    metadata = provider.require_checkpoint_policy_observation_metadata(
        checkpoint_path=checkpoint,
        checkpoint_payload={"model": {}},
    )

    assert metadata == {
        "policy_trail_render_mode": POLICY_TRAIL_RENDER_MODE,
        "policy_bonus_render_mode": POLICY_BONUS_RENDER_MODE,
        "policy_observation_backend": DEFAULT_POLICY_OBSERVATION_BACKEND,
        "policy_observation_contract_id": POLICY_OBSERVATION_CONTRACT_ID,
        "policy_observation_perspective_schema_id": (
            POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
        ),
    }


def test_checkpoint_provider_rejects_shadowed_nested_contract_mismatch(tmp_path):
    checkpoint = tmp_path / "iteration_0.pth.tar"
    checkpoint.write_bytes(b"checkpoint")
    sidecar = _valid_policy_observation_sidecar()
    sidecar["observation_contract"]["contract_id"] = "legacy/raw-player-zero/v0"
    provider.checkpoint_policy_metadata_sidecar_path(checkpoint).write_text(
        json.dumps(sidecar),
        encoding="utf-8",
    )

    with np.testing.assert_raises_regex(ValueError, "observation_contract.contract_id"):
        provider.require_checkpoint_policy_observation_metadata(
            checkpoint_path=checkpoint,
            checkpoint_payload={"model": {}},
        )


def test_checkpoint_provider_rejects_lab_policy_observation_backend(tmp_path):
    checkpoint = tmp_path / "iteration_0.pth.tar"
    checkpoint.write_bytes(b"checkpoint")
    sidecar = _valid_policy_observation_sidecar()
    sidecar["policy_observation_backend"] = "jax_gpu"
    provider.checkpoint_policy_metadata_sidecar_path(checkpoint).write_text(
        json.dumps(sidecar),
        encoding="utf-8",
    )

    with np.testing.assert_raises_regex(ValueError, "policy_observation_backend"):
        provider.require_checkpoint_policy_observation_metadata(
            checkpoint_path=checkpoint,
            checkpoint_payload={"model": {}},
        )


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


def test_checkpoint_opponent_policy_forward_uses_non_board_game_to_play(monkeypatch):
    calls = []

    class FakeTorch:
        float32 = np.float32

        @staticmethod
        def as_tensor(value, *, dtype, device):
            del device
            return np.asarray(value, dtype=dtype)

        @staticmethod
        def no_grad():
            class Context:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, traceback):
                    return False

            return Context()

    class FakeEvalMode:
        def forward(self, obs_tensor, *, action_mask, to_play, ready_env_id):
            calls.append(
                {
                    "obs_shape": tuple(obs_tensor.shape),
                    "action_mask": np.asarray(action_mask).copy(),
                    "to_play": list(to_play),
                    "ready_env_id": np.asarray(ready_env_id).copy(),
                }
            )
            return {"action": 1}

    class FakePolicy:
        eval_mode = FakeEvalMode()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    provider._policy_eval_forward(
        FakePolicy(),
        observation=np.zeros((4, 64, 64), dtype=np.float32),
        legal_action_mask=np.array([True, True, False]),
        player_id=1,
        device="cpu",
    )

    assert len(calls) == 1
    assert calls[0]["obs_shape"] == (1, 4, 64, 64)
    np.testing.assert_array_equal(
        calls[0]["action_mask"],
        np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
    )
    assert calls[0]["to_play"] == [-1]
    np.testing.assert_array_equal(calls[0]["ready_env_id"], np.asarray([0]))
