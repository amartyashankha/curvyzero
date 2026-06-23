from __future__ import annotations

import inspect
import importlib.util
import sys
from pathlib import Path

import pytest

from curvyzero.contracts import curvytron as contract
from curvyzero.infra.modal import curvyzero_checkpoint_tournament as tournament_app
from curvyzero.infra.modal import lightzero_curvyzero_stacked_debug_visual_survival_train as train_app
from curvyzero.tournament.curvytron import contracts as tournament_contracts


ROOT = Path(__file__).resolve().parents[1]


def _load_script(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_current_modal_names_are_shared():
    assert contract.curvytron_runs_volume_name() == "curvyzero-runs-v2"
    assert (
        contract.curvytron_tournament_volume_name()
        == "curvyzero-curvytron-tournaments-v2"
    )
    assert contract.curvytron_control_volume_name() == "curvyzero-curvytron-control-v2"
    assert (
        contract.curvytron_checkpoint_intake_dict_name()
        == "curvyzero-curvytron-checkpoint-intake-v2"
    )
    assert (
        contract.curvytron_checkpoint_intake_queue_name()
        == "curvyzero-curvytron-checkpoint-events-v2"
    )
    assert (
        contract.curvytron_opponent_leaderboard_dict_name()
        == "curvyzero-curvytron-opponent-leaderboard-live-v2"
    )
    assert (
        contract.curvytron_current_tournament_id()
        == "cz26-live-20260517a"
    )
    assert (
        contract.curvytron_current_rating_run_id()
        == "elo-cz26-live-20260517a"
    )
    assert contract.curvytron_current_gif_run_prefixes() == (
        "rnd-blank-current-",
        "rnd-blank-sweep-fastckpt-20260519a-",
    )
    assert (
        contract.curvytron_training_candidate_refresh_config_ref()
        == "control:training/lightzero-curvytron-visual-survival/"
        "cz26-control/attempts/try-cz26-control/"
        "opponents/training_candidate_refresh_config.json"
    )
    pointer_root = (
        "control:training/lightzero-curvytron-visual-survival/cz26-control/"
        "attempts/try-cz26-control/opponents/refresh_pointers"
    )
    assert contract.curvytron_training_candidate_refresh_pointers() == (
        f"{pointer_root}/b100/imm0/refresh_pointer.json",
        f"{pointer_root}/b100/imm10/refresh_pointer.json",
        f"{pointer_root}/b10w05r1/imm0/refresh_pointer.json",
        f"{pointer_root}/b10w05r1/imm10/refresh_pointer.json",
        f"{pointer_root}/b20w05lad4/imm0/refresh_pointer.json",
        f"{pointer_root}/b20w05lad4/imm10/refresh_pointer.json",
        f"{pointer_root}/b20w05r1/imm0/refresh_pointer.json",
        f"{pointer_root}/b20w05r1/imm10/refresh_pointer.json",
        f"{pointer_root}/b20w05top2/imm0/refresh_pointer.json",
        f"{pointer_root}/b20w05top2/imm10/refresh_pointer.json",
        f"{pointer_root}/b20w10r1/imm0/refresh_pointer.json",
        f"{pointer_root}/b20w10r1/imm10/refresh_pointer.json",
        f"{pointer_root}/b20w20lad4s/imm0/refresh_pointer.json",
        f"{pointer_root}/b20w20lad4s/imm10/refresh_pointer.json",
        f"{pointer_root}/b25w25r1/imm0/refresh_pointer.json",
        f"{pointer_root}/b25w25r1/imm10/refresh_pointer.json",
        f"{pointer_root}/b30w05r1/imm0/refresh_pointer.json",
        f"{pointer_root}/b30w05r1/imm10/refresh_pointer.json",
        f"{pointer_root}/b50r1/imm0/refresh_pointer.json",
        f"{pointer_root}/b50r1/imm10/refresh_pointer.json",
        f"{pointer_root}/r1/imm0/refresh_pointer.json",
        f"{pointer_root}/r1/imm10/refresh_pointer.json",
        f"{pointer_root}/w100/imm0/refresh_pointer.json",
        f"{pointer_root}/w100/imm10/refresh_pointer.json",
    )

    assert contract.modal_volume_kwargs_for_name("curvyzero-runs-v2")["version"] == 2
    assert (
        contract.modal_volume_kwargs_for_name(
            "curvyzero-curvytron-tournaments-v2"
        )["version"]
        == 2
    )
    assert (
        contract.modal_volume_kwargs_for_name("curvyzero-curvytron-control-v2")[
            "version"
        ]
        == 2
    )
    with pytest.raises(ValueError, match="all-v2"):
        contract.modal_volume_kwargs_for_name("curvyzero-runs")
    with pytest.raises(ValueError, match="all-v2"):
        contract.modal_volume_kwargs_for_name("curvyzero-curvytron-control")


def test_current_modal_name_overrides_must_remain_v2(monkeypatch):
    monkeypatch.setenv("CURVYZERO_RUNS_VOLUME_NAME", "curvyzero-runs")

    with pytest.raises(ValueError, match="all-v2"):
        contract.curvytron_runs_volume_name()


def test_train_and_tournament_apps_use_shared_runtime_defaults():
    assert train_app.TASK_ID == contract.CURVYTRON_TRAINING_TASK_ID
    assert train_app.VOLUME_NAME == contract.curvytron_runs_volume_name()
    assert train_app.CONTROL_VOLUME_NAME == contract.curvytron_control_volume_name()
    assert train_app.DEFAULT_SOURCE_MAX_STEPS == contract.CURVYTRON_SOURCE_MAX_STEPS
    assert train_app.DEFAULT_MAX_ENV_STEP == contract.CURVYTRON_DEFAULT_MAX_ENV_STEP
    assert train_app.DEFAULT_MAX_TRAIN_ITER == contract.CURVYTRON_DEFAULT_MAX_TRAIN_ITER
    assert train_app.DEFAULT_COMPUTE == contract.CURVYTRON_DEFAULT_TRAIN_COMPUTE
    assert (
        train_app.DEFAULT_COLLECTOR_ENV_NUM
        == contract.CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM
    )
    assert train_app.DEFAULT_N_EPISODE == contract.CURVYTRON_DEFAULT_N_EPISODE
    assert train_app.DEFAULT_NUM_SIMULATIONS == contract.CURVYTRON_DEFAULT_NUM_SIMULATIONS
    assert train_app.DEFAULT_BATCH_SIZE == contract.CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE
    assert train_app.DEFAULT_SAVE_CKPT_AFTER_ITER == contract.CURVYTRON_SAVE_CKPT_AFTER_ITER
    assert train_app.DEFAULT_COMMIT_ON_CHECKPOINT is contract.CURVYTRON_COMMIT_ON_CHECKPOINT
    assert train_app.DEFAULT_BACKGROUND_GIF_FPS == contract.CURVYTRON_BACKGROUND_GIF_FPS

    assert tournament_app.TRAINING_TASK_ID == contract.CURVYTRON_TRAINING_TASK_ID
    assert tournament_app.CHECKPOINT_VOLUME_NAME == contract.curvytron_runs_volume_name()
    assert tournament_app.TOURNAMENT_VOLUME_NAME == contract.curvytron_tournament_volume_name()
    assert tournament_app.CURRENT_TOURNAMENT_ID == contract.curvytron_current_tournament_id()
    assert tournament_app.CURRENT_RATING_RUN_ID == contract.curvytron_current_rating_run_id()
    assert tournament_app.TRAIN_APP_NAME == contract.curvytron_train_app_name()
    assert tournament_app.GIF_BROWSER_APP_NAME == contract.curvytron_gif_browser_app_name()
    assert (
        tournament_app.DEFAULT_CHECKPOINT_INTAKE_SCAN_SECONDS
        == contract.DEFAULT_CURVYTRON_CHECKPOINT_INTAKE_SCAN_SECONDS
    )
    assert tournament_contracts.DEFAULT_MAX_STEPS == contract.CURVYTRON_SOURCE_MAX_STEPS
    assert tournament_contracts.DEFAULT_DECISION_MS == contract.CURVYTRON_DECISION_MS
    assert tournament_contracts.DEFAULT_GIF_FPS == contract.CURVYTRON_TOURNAMENT_GIF_FPS
    assert (
        tournament_contracts.DEFAULT_GIF_MIN_FRAME_DURATION_MS
        == contract.CURVYTRON_TOURNAMENT_GIF_MIN_FRAME_DURATION_MS
    )


def test_tournament_cli_current_modes_default_to_current_lane():
    assert (
        tournament_app._cli_tournament_id_for_mode(
            mode="intake-status",
            tournament_id="",
        )
        == contract.curvytron_current_tournament_id()
    )
    assert (
        tournament_app._cli_rating_run_id_for_mode(
            mode="training-candidate-auto-refresh",
            tournament_id="",
            rating_run_id=tournament_contracts.DEFAULT_RATING_RUN_ID,
        )
        == contract.curvytron_current_rating_run_id()
    )
    assert (
        tournament_app._cli_rating_run_id_for_mode(
            mode="intake-status",
            tournament_id=contract.curvytron_current_tournament_id(),
            rating_run_id="custom-elo",
        )
        == "custom-elo"
    )
    assert (
        tournament_app._cli_tournament_id_for_mode(
            mode="rating",
            tournament_id="explicit-arena",
        )
        == "explicit-arena"
    )


def test_auto_refresh_cli_payload_does_not_smuggle_current_lane_defaults():
    payload = tournament_app._training_candidate_auto_refresh_cli_payload(
        tournament_id="arena-canary",
        rating_run_id="elo-canary",
        leaderboard_id="arena-canary-elo-canary-training",
    )

    assert payload == {
        "tournament_id": "arena-canary",
        "rating_run_id": "elo-canary",
        "leaderboard_id": "arena-canary-elo-canary-training",
    }
    assert "refresh_pointers" not in payload
    assert "assignment_bank_run_id" not in payload
    assert "assignment_seed" not in payload
    assert "min_active_count" not in payload
    assert "active_min_valid_games" not in payload
    assert "active_min_distinct_opponents" not in payload


def test_auto_refresh_cli_payload_keeps_explicit_overrides_only():
    payload = tournament_app._training_candidate_auto_refresh_cli_payload(
        tournament_id="arena-canary",
        rating_run_id="elo-canary",
        leaderboard_id="arena-canary-elo-canary-training",
        config_ref="control:canary/config.json",
        refresh_pointer_refs=["control:path/a.json"],
        assignment_bank_run_id="canary-bank",
        assignment_bank_attempt_id="try-canary-bank",
        assignment_id_prefix="canary-auto",
        assignment_seed=123,
        generation=7,
        min_active_count=2,
        allow_partial_assignment=True,
        active_min_valid_games=21,
        active_min_distinct_opponents=1,
        max_active_rank=50,
    )

    assert payload["config_ref"] == "control:canary/config.json"
    assert payload["refresh_pointers"] == ["control:path/a.json"]
    assert payload["assignment_bank_run_id"] == "canary-bank"
    assert payload["assignment_bank_attempt_id"] == "try-canary-bank"
    assert payload["assignment_id_prefix"] == "canary-auto"
    assert payload["assignment_seed"] == 123
    assert payload["generation"] == 7
    assert payload["min_active_count"] == 2
    assert payload["allow_partial_assignment"] is True
    assert payload["active_min_valid_games"] == 21
    assert payload["active_min_distinct_opponents"] == 1
    assert payload["max_active_rank"] == 50


def test_auto_refresh_cli_entrypoint_exposes_config_ref_override():
    signature = inspect.signature(tournament_app.main.info.raw_f)

    assert "training_candidate_refresh_config_ref" in signature.parameters


def test_current_lane_static_status_uses_shared_contract():
    status = tournament_app._current_lane_static_status()

    assert status["current"]["tournament_id"] == contract.curvytron_current_tournament_id()
    assert status["current"]["rating_run_id"] == contract.curvytron_current_rating_run_id()
    assert status["current"]["training_candidate_refresh_pointers"] == list(
        contract.curvytron_training_candidate_refresh_pointers()
    )
    assert (
        status["current"]["training_candidate_refresh_config_ref"]
        == contract.curvytron_training_candidate_refresh_config_ref()
    )
    assert status["app"]["trainer"] == contract.curvytron_train_app_name()
    assert status["app"]["gif_browser"] == contract.curvytron_gif_browser_app_name()
    assert (
        status["automation"]["checkpoint_intake_subscriber_tick_seconds"]
        == contract.DEFAULT_CURVYTRON_CHECKPOINT_INTAKE_SCAN_SECONDS
    )
    assert status["tournament_defaults"]["save_gif"] is True
    assert status["tournament_defaults"]["gif_sample_games_per_pair"] == 5


def test_manifest_and_submitter_scripts_use_shared_current_contract():
    tonight18 = _load_script(
        "scripts/build_curvytron_tonight18_manifest.py",
        "build_curvytron_tonight18_manifest_contract_test",
    )
    survivaldiag = _load_script(
        "scripts/build_curvytron_survivaldiag_manifest.py",
        "build_curvytron_survivaldiag_manifest_contract_test",
    )
    mixture = _load_script(
        "scripts/build_curvytron_opponent_mixture_manifest.py",
        "build_curvytron_opponent_mixture_manifest_contract_test",
    )
    submitter = _load_script(
        "scripts/submit_curvytron_survivaldiag_manifest.py",
        "submit_curvytron_survivaldiag_manifest_contract_test",
    )

    for module in (tonight18, survivaldiag, mixture):
        assert module.TASK_ID == contract.CURVYTRON_TRAINING_TASK_ID
        assert module.SOURCE_MAX_STEPS == contract.CURVYTRON_SOURCE_MAX_STEPS
        assert module.DECISION_MS == contract.CURVYTRON_DECISION_MS

    assert (
        tonight18.TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT
        == contract.TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT
    )
    assert (
        survivaldiag.TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT
        == contract.TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT
    )
    assert (
        submitter.TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT
        == contract.TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT
    )
    assert (
        submitter.POLLER_KWARGS_ALLOWED_FOR_GROUPED_SUBMIT
        == contract.POLLER_KWARGS_ALLOWED_FOR_GROUPED_SUBMIT
    )
    assert contract.TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT == (
        "mode",
        "seed",
        "run_id",
        "attempt_id",
    )
    assert set(contract.TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT).isdisjoint(
        {
            "decision_ms",
            "collector_env_num",
            "batch_size",
            "source_state_trail_render_mode",
            "source_state_bonus_render_mode",
            "background_eval_enabled",
        }
    )
    assert tonight18.DEFAULT_COLLECTOR_ENV_NUM == contract.CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM
    assert tonight18.DEFAULT_N_EPISODE == contract.CURVYTRON_DEFAULT_N_EPISODE
    assert tonight18.DEFAULT_BATCH_SIZE == contract.CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE
    assert tonight18.DEFAULT_NUM_SIMULATIONS == contract.CURVYTRON_DEFAULT_NUM_SIMULATIONS
    assert tonight18.DEFAULT_MAX_ENV_STEP == contract.CURVYTRON_DEFAULT_MAX_ENV_STEP
    assert tonight18.DEFAULT_MAX_TRAIN_ITER == contract.CURVYTRON_DEFAULT_MAX_TRAIN_ITER
    assert tonight18.CURVYTRON_DEFAULT_TRAIN_COMPUTE == contract.CURVYTRON_DEFAULT_TRAIN_COMPUTE
    assert contract.CURVYTRON_DEFAULT_TRAIN_COMPUTE == "gpu-l4-t4-cpu40"
    assert contract.CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE == 64


def test_visual_survival_train_default_kwargs_fill_compact_grouped_submit_payload():
    kwargs = {"mode": "train", "seed": 11, "run_id": "r", "attempt_id": "a"}

    train_app._apply_visual_survival_train_default_kwargs(kwargs)

    assert kwargs["decision_ms"] == train_app.DEFAULT_DECISION_MS
    assert kwargs["collector_env_num"] == train_app.DEFAULT_COLLECTOR_ENV_NUM
    assert kwargs["batch_size"] == train_app.DEFAULT_BATCH_SIZE
    assert (
        kwargs["source_state_trail_render_mode"]
        == train_app.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE
    )
    assert (
        kwargs["source_state_bonus_render_mode"]
        == train_app.DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE
    )
    assert kwargs["background_eval_enabled"] == train_app.DEFAULT_BACKGROUND_EVAL_ENABLED
