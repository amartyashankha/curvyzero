from __future__ import annotations

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
        == "curvy-e2e-allv2-canary-live-20260515a"
    )
    assert (
        contract.curvytron_current_rating_run_id()
        == "elo-e2e-allv2-canary-live-20260515a"
    )

    assert "version" not in contract.modal_volume_kwargs_for_name("curvyzero-runs")
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
    assert "version" not in contract.modal_volume_kwargs_for_name(
        "curvyzero-curvytron-control"
    )


def test_current_modal_name_overrides_must_remain_v2(monkeypatch):
    monkeypatch.setenv("CURVYZERO_RUNS_VOLUME_NAME", "curvyzero-runs")

    with pytest.raises(ValueError, match="all-v2"):
        contract.curvytron_runs_volume_name()


def test_train_and_tournament_apps_use_shared_runtime_defaults():
    assert train_app.TASK_ID == contract.CURVYTRON_TRAINING_TASK_ID
    assert train_app.VOLUME_NAME == contract.curvytron_runs_volume_name()
    assert train_app.CONTROL_VOLUME_NAME == contract.curvytron_control_volume_name()
    assert train_app.DEFAULT_SOURCE_MAX_STEPS == contract.CURVYTRON_SOURCE_MAX_STEPS
    assert train_app.DEFAULT_SAVE_CKPT_AFTER_ITER == contract.CURVYTRON_SAVE_CKPT_AFTER_ITER
    assert train_app.DEFAULT_COMMIT_ON_CHECKPOINT is contract.CURVYTRON_COMMIT_ON_CHECKPOINT
    assert train_app.DEFAULT_BACKGROUND_GIF_FPS == contract.CURVYTRON_BACKGROUND_GIF_FPS

    assert tournament_app.TRAINING_TASK_ID == contract.CURVYTRON_TRAINING_TASK_ID
    assert tournament_app.CHECKPOINT_VOLUME_NAME == contract.curvytron_runs_volume_name()
    assert tournament_app.TOURNAMENT_VOLUME_NAME == contract.curvytron_tournament_volume_name()
    assert tournament_app.CURRENT_TOURNAMENT_ID == contract.curvytron_current_tournament_id()
    assert tournament_app.CURRENT_RATING_RUN_ID == contract.curvytron_current_rating_run_id()
    assert tournament_contracts.DEFAULT_MAX_STEPS == contract.CURVYTRON_SOURCE_MAX_STEPS
    assert tournament_contracts.DEFAULT_DECISION_MS == contract.CURVYTRON_DECISION_MS


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
