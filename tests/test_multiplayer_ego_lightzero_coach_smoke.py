import numpy as np

from curvyzero.env.multiplayer_ego_wrapper import MULTIPLAYER_EGO_WRAPPER_ID
from curvyzero.env.vector_multiplayer_env import DEBUG_METADATA_OBSERVATION_SCHEMA_ID
from curvyzero.training.multiplayer_ego_lightzero_coach_smoke import (
    MULTIPLAYER_EGO_LIGHTZERO_COACH_SMOKE_ID,
    MultiplayerEgoLightZeroCoachSmokeRequest,
    build_multiplayer_ego_lightzero_coach_smoke_report,
)
from curvyzero.training.multiplayer_opponent_policy import FIXED_ACTION_OPPONENT_POLICY_ID


def test_multiplayer_ego_coach_smoke_is_metadata_only_wrapper_path():
    report = build_multiplayer_ego_lightzero_coach_smoke_report(
        MultiplayerEgoLightZeroCoachSmokeRequest(seed=11, batch_size=2, player_count=3)
    )

    assert report["ok"] is True
    assert report["smoke_id"] == MULTIPLAYER_EGO_LIGHTZERO_COACH_SMOKE_ID
    assert report["call_policy"] == (
        "does_not_train; does_not_import_lightzero; does_not_call_trainer"
    )
    assert report["metadata_only"] is True
    assert report["trainer_observation_claim"] is False
    assert report["trainer_replay_claim"] is False
    assert report["learned_observation_claim"] is False
    assert report["joint_action_mcts_claim"] is False
    assert report["uses_ale"] is False
    assert report["registered_lightzero_env"] is False
    assert report["semantics"]["wrapper_id"] == MULTIPLAYER_EGO_WRAPPER_ID
    assert report["semantics"]["forked_curvytron_trainer_path"] is False

    assert report["coach_config_surface"]["trainable"] is False
    assert report["coach_config_surface"]["model_observation_shape"] is None
    assert report["coach_config_surface"]["model_observation_schema_id"] is None
    assert report["coach_config_surface"]["public_observation_schema_id"] == (
        DEBUG_METADATA_OBSERVATION_SCHEMA_ID
    )


def test_multiplayer_ego_coach_smoke_proves_reset_action_map_and_sidecar_shapes():
    report = build_multiplayer_ego_lightzero_coach_smoke_report(
        MultiplayerEgoLightZeroCoachSmokeRequest(
            seed=5,
            batch_size=2,
            player_count=4,
            ego_player_id=(0, 2),
            selected_ego_action_id=2,
        )
    )

    assert report["validation_problems"] == []
    assert report["reset_surface"]["observation"]["shape"] == [2, 4, 6]
    assert report["reset_surface"]["action_mask"]["shape"] == [2, 4, 3]
    assert report["policy_rows_surface"]["source_shape"] == [2, 4]
    assert report["policy_rows_surface"]["active_count"] == 2
    assert report["policy_rows_surface"]["capacity"] == 2
    assert report["policy_rows_surface"]["observation_schema_id"] == (
        DEBUG_METADATA_OBSERVATION_SCHEMA_ID
    )
    np.testing.assert_array_equal(report["policy_rows_surface"]["ego_player_id"], [0, 2])

    joint_action = np.asarray(report["action_map_surface"]["joint_action"])
    assert joint_action.shape == (2, 4)
    np.testing.assert_array_equal(joint_action[:, [0, 2]].diagonal(), np.asarray([2, 2]))
    assert report["action_map_surface"]["sidecar_wrapper_id"] == MULTIPLAYER_EGO_WRAPPER_ID
    assert report["action_map_surface"]["sidecar_flags"]["metadata_only"] is True
    assert report["action_map_surface"]["sidecar_flags"]["learned_observation_claim"] is False
    assert report["action_map_surface"]["opponent_policy_id"] == FIXED_ACTION_OPPONENT_POLICY_ID

    assert report["step_surface"]["wrapper_joint_action_summary"]["shape"] == [2, 4]
    assert report["step_surface"]["multiplayer_ego_wrapper_id"] == MULTIPLAYER_EGO_WRAPPER_ID
    assert report["step_surface"]["metadata_only"] is True
    assert report["step_surface"]["learned_observation_claim"] is False
