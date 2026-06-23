from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from curvyzero.training import compact_promotion_readiness_learning_quality as quality


def test_capture_from_artifacts_stock_happy_path(tmp_path):
    module = _load_capture_module()
    inputs = _capture_inputs(tmp_path, role="stock_reference")
    output_root = tmp_path / "out"

    assert module.main(_cli_args(inputs, role="stock_reference", output_root=output_root)) == 0

    capture_path = output_root / "unit-run" / "stock_reference_capture.json"
    capture = _read_json(capture_path)
    assert capture["schema_id"] == quality.COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_SCHEMA_ID
    assert capture["role"] == "stock_reference"
    assert capture["route"] == "stock_train_muzero_reference"
    assert capture["called_train_muzero"] is True
    _assert_required_artifact_refs(capture)
    arm = quality.compact_matched_learning_quality_arm_from_capture_v1(
        capture,
        expected_role="stock_reference",
    )
    assert arm["calls_train_muzero"] is True


def test_capture_from_artifacts_compact_happy_path(tmp_path):
    module = _load_capture_module()
    inputs = _capture_inputs(tmp_path, role="compact_candidate")
    output_root = tmp_path / "out"

    assert module.main(_cli_args(inputs, role="compact_candidate", output_root=output_root)) == 0

    capture_path = output_root / "unit-run" / "compact_candidate_capture.json"
    capture = _read_json(capture_path)
    assert capture["schema_id"] == quality.COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_SCHEMA_ID
    assert capture["role"] == "compact_candidate"
    assert capture["route"] == "compact_owned_trainer"
    assert capture["calls_train_muzero"] is False
    assert capture["real_compact_owned_training_work"] is True
    assert capture["model_identity"]["model_identity_scope"] == "candidate_loaded_checkpoint"
    _assert_required_artifact_refs(capture)
    arm = quality.compact_matched_learning_quality_arm_from_capture_v1(
        capture,
        expected_role="compact_candidate",
    )
    assert arm["calls_train_muzero"] is False


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("producer_ran_training", False, "producer_ran_training=true"),
        ("producer_ran_training", None, "producer_ran_training=true"),
        ("producer_ran_pre_eval", False, "producer_ran_pre_eval=true"),
        ("producer_ran_post_eval", False, "producer_ran_post_eval=true"),
        ("feeds_builder", False, "feeds_builder must be true"),
        ("support_only", True, "support_only must be false"),
        ("support_only", None, "support_only must be false"),
    ],
)
def test_capture_from_artifacts_rejects_no_training_provenance(
    tmp_path,
    key,
    value,
    match,
):
    module = _load_capture_module()
    inputs = _capture_inputs(tmp_path, role="stock_reference")
    provenance = _read_json(inputs["capture_provenance"])
    if value is None:
        del provenance[key]
    else:
        provenance[key] = value
    _write_json(inputs["capture_provenance"], provenance)

    with pytest.raises(quality.CompactMatchedLearningQualityError, match=match):
        module.main(_cli_args(inputs, role="stock_reference", output_root=tmp_path / "out"))


@pytest.mark.parametrize(
    ("role", "mutator", "match"),
    [
        (
            "stock_reference",
            lambda p: p["stock_train_muzero"].update({"called_train_muzero": False}),
            "must call train_muzero",
        ),
        (
            "stock_reference",
            lambda p: p.update({"role": "compact_candidate"}),
            "role mismatch",
        ),
        (
            "compact_candidate",
            lambda p: p["compact_owned_training"].update({"calls_train_muzero": True}),
            "calls_train_muzero must be false",
        ),
        (
            "compact_candidate",
            lambda p: p["compact_owned_training"].update(
                {"real_compact_owned_training_work": False}
            ),
            "real compact-owned work missing",
        ),
        (
            "compact_candidate",
            lambda p: p.update({"role": "stock_reference"}),
            "role mismatch",
        ),
    ],
)
def test_capture_from_artifacts_rejects_role_specific_lies(
    tmp_path,
    role,
    mutator,
    match,
):
    module = _load_capture_module()
    inputs = _capture_inputs(tmp_path, role=role)
    provenance = _read_json(inputs["capture_provenance"])
    mutator(provenance)
    _write_json(inputs["capture_provenance"], provenance)

    with pytest.raises(quality.CompactMatchedLearningQualityError, match=match):
        module.main(_cli_args(inputs, role=role, output_root=tmp_path / "out"))


@pytest.mark.parametrize(
    ("role", "output_name", "match"),
    [
        (
            "stock_reference",
            "compact_candidate_capture.json",
            "stock_reference capture output must be named stock_reference_capture.json",
        ),
        (
            "compact_candidate",
            "stock_reference_capture.json",
            "compact_candidate capture output must be named compact_candidate_capture.json",
        ),
        (
            "stock_reference",
            "stock_reference_capture_preview.json",
            "must not use preview capture filenames",
        ),
    ],
)
def test_capture_from_artifacts_rejects_output_filename_mismatch(
    tmp_path,
    role,
    output_name,
    match,
):
    module = _load_capture_module()
    inputs = _capture_inputs(tmp_path, role=role)

    with pytest.raises(ValueError, match=match):
        module.main(
            _cli_args(
                inputs,
                role=role,
                output=tmp_path / "out" / output_name,
            )
        )


def test_capture_from_artifacts_writes_only_capture_file(tmp_path):
    module = _load_capture_module()
    inputs = _capture_inputs(tmp_path, role="stock_reference")
    output_root = tmp_path / "out"

    assert module.main(_cli_args(inputs, role="stock_reference", output_root=output_root)) == 0

    run_dir = output_root / "unit-run"
    assert sorted(path.name for path in run_dir.iterdir()) == ["stock_reference_capture.json"]
    assert not (run_dir / "matched_learning_quality_canary_report.json").exists()
    assert not (run_dir / "stock_reference_arm.json").exists()
    assert not (run_dir / "compact_candidate_arm.json").exists()


def test_capture_from_artifacts_rejects_existing_output_without_overwrite(tmp_path):
    module = _load_capture_module()
    inputs = _capture_inputs(tmp_path, role="stock_reference")
    output_root = tmp_path / "out"

    assert module.main(_cli_args(inputs, role="stock_reference", output_root=output_root)) == 0

    with pytest.raises(FileExistsError, match="capture output already exists"):
        module.main(_cli_args(inputs, role="stock_reference", output_root=output_root))


def test_capture_from_artifacts_can_explicitly_overwrite_existing_output(tmp_path):
    module = _load_capture_module()
    inputs = _capture_inputs(tmp_path, role="stock_reference")
    output_root = tmp_path / "out"
    args = _cli_args(inputs, role="stock_reference", output_root=output_root)

    assert module.main(args) == 0
    assert module.main([*args, "--overwrite"]) == 0


@pytest.mark.parametrize(
    ("role", "mutator", "match"),
    [
        (
            "stock_reference",
            lambda d: d.update({"speed_currency": "stock_train_muzero_profile_env_steps_per_sec"}),
            "profile-only speed currency",
        ),
        (
            "stock_reference",
            lambda d: d.update({"uses_fallback_denominator": True}),
            "uses_fallback_denominator must be false",
        ),
        (
            "stock_reference",
            lambda d: d.update({"learner_train_calls": 0}),
            "learner_train_calls must be > 0",
        ),
        (
            "compact_candidate",
            lambda d: d.update({"sample_batch_count_delta": 0}),
            "sample_batch_count_delta must be > 0",
        ),
        (
            "compact_candidate",
            lambda d: d.update({"learner_loss_mean": "not-finite"}),
            "learner_loss_mean must be finite",
        ),
    ],
)
def test_capture_from_artifacts_rejects_fake_denominators(
    tmp_path,
    role,
    mutator,
    match,
):
    module = _load_capture_module()
    inputs = _capture_inputs(tmp_path, role=role)
    denominators = _read_json(inputs["denominators"])
    mutator(denominators)
    _write_json(inputs["denominators"], denominators)

    with pytest.raises(quality.CompactMatchedLearningQualityError, match=match):
        module.main(_cli_args(inputs, role=role, output_root=tmp_path / "out"))


def test_capture_from_artifacts_rejects_missing_required_artifact_file(tmp_path):
    module = _load_capture_module()
    inputs = _capture_inputs(tmp_path, role="compact_candidate")
    inputs["final_checkpoint"].unlink()

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="capture artifact path missing",
    ):
        module.main(_cli_args(inputs, role="compact_candidate", output_root=tmp_path / "out"))


def test_capture_from_artifacts_rejects_training_artifact_schema_mismatch(tmp_path):
    module = _load_capture_module()
    inputs = _capture_inputs(tmp_path, role="stock_reference")
    artifact = _read_json(inputs["training_artifact"])
    artifact["schema_id"] = "fake-training-artifact/v1"
    _write_json(inputs["training_artifact"], artifact)

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="training_artifact schema mismatch",
    ):
        module.main(_cli_args(inputs, role="stock_reference", output_root=tmp_path / "out"))


def test_capture_from_artifacts_rejects_eval_summary_seed_set_mismatch(tmp_path):
    module = _load_capture_module()
    inputs = _capture_inputs(tmp_path, role="compact_candidate")
    summary = _read_json(inputs["pre_eval_summary"])
    summary["eval_seed_set"] = [101, 999]
    _write_json(inputs["pre_eval_summary"], summary)

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="pre_train eval summary eval_seed_set mismatch",
    ):
        module.main(_cli_args(inputs, role="compact_candidate", output_root=tmp_path / "out"))


def _load_capture_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_matched_learning_quality_capture_from_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_matched_learning_quality_capture_from_artifacts_direct_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_args(
    inputs: Mapping[str, Path],
    *,
    role: str,
    output_root: Path | None = None,
    output: Path | None = None,
) -> list[str]:
    args = [
        "--role",
        role,
        "--run-id",
        "unit-run",
        "--capture-id",
        f"unit-{role}-capture",
        "--candidate-checkpoint-id",
        "unit-compact-ckpt",
        "--denominator-id",
        "unit-matched-quality-denominator",
        "--quality-horizon",
        "pre_post_100_env_steps",
        "--hardware-class",
        "local-test",
        "--source-fingerprint-json",
        str(inputs["source_fingerprint"]),
        "--model-identity-json",
        str(inputs["model_identity"]),
        "--eval-settings-json",
        str(inputs["eval_settings"]),
        "--denominators-json",
        str(inputs["denominators"]),
        "--capture-provenance-json",
        str(inputs["capture_provenance"]),
        "--training-artifact",
        str(inputs["training_artifact"]),
        "--pre-eval-summary",
        str(inputs["pre_eval_summary"]),
        "--post-eval-summary",
        str(inputs["post_eval_summary"]),
        "--initial-checkpoint",
        str(inputs["initial_checkpoint"]),
        "--final-checkpoint",
        str(inputs["final_checkpoint"]),
    ]
    if output is not None:
        args.extend(["--output", str(output)])
    if output_root is not None:
        args.extend(["--output-root", str(output_root)])
    return args


def _capture_inputs(tmp_path: Path, *, role: str) -> dict[str, Path]:
    root = tmp_path / f"{role}_inputs"
    paths = {
        "source_fingerprint": root / "source_fingerprint.json",
        "model_identity": root / "model_identity.json",
        "eval_settings": root / "eval_settings.json",
        "denominators": root / "denominators.json",
        "capture_provenance": root / "capture_provenance.json",
        "training_artifact": root / "training_artifact.json",
        "pre_eval_summary": root / "pre_eval_summary.json",
        "post_eval_summary": root / "post_eval_summary.json",
        "initial_checkpoint": root / "initial_checkpoint.pt",
        "final_checkpoint": root / "final_checkpoint.pt",
    }
    _write_json(paths["source_fingerprint"], _source_fingerprint())
    _write_json(paths["model_identity"], _model_identity(role))
    _write_json(paths["eval_settings"], _eval_settings())
    _write_json(paths["denominators"], _denominators(role))
    _write_json(paths["capture_provenance"], _capture_provenance(role))
    _write_json(paths["initial_checkpoint"], {"kind": "initial_checkpoint", "role": role})
    _write_json(paths["final_checkpoint"], {"kind": "final_checkpoint", "role": role})
    _write_json(paths["training_artifact"], _training_artifact(role, paths))
    _write_json(paths["pre_eval_summary"], _eval_summary(role, mean=10.0, step=0))
    _write_json(paths["post_eval_summary"], _eval_summary(role, mean=13.0, step=100))
    return paths


def _source_fingerprint() -> dict[str, Any]:
    return {
        "git_commit": "unit-commit",
        "git_status_dirty": True,
        "producer_script": "unit-quality-producer.py",
        "producer_route": "unit-quality-producer",
        "image_digest": "unit-image",
        "matched_surface": _matched_surface(),
    }


def _matched_surface() -> dict[str, Any]:
    return {
        "env_variant": "source_state_fixed_opponent",
        "reward_variant": "survival_plus_bonus_no_outcome",
        "policy_observation_backend": "cpu_oracle",
        "opponent_policy_kind": "fixed_anchor",
        "eval_seed_set": [101, 102],
        "eval_seed_rng_seed": 20260833,
    }


def _model_identity(role: str) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "initial_model_state_digest": f"{role}-initial",
        "final_model_state_digest": f"{role}-final",
        "model_state_digest_changed": True,
    }
    if role == "compact_candidate":
        identity["model_identity_scope"] = "candidate_loaded_checkpoint"
    return identity


def _eval_settings() -> dict[str, Any]:
    return {
        "observation_schema_id": "curvytron_source_state_v1",
        "policy_observation_backend": "cpu_oracle",
        "eval_seed_set": [101, 102],
        "eval_seed_rng_seed": 20260833,
        "eval_episode_count": 2,
        "source_max_steps": 2000,
        "eval_max_steps": 2000,
        "num_simulations": 8,
        "batch_size": 2,
        "reward_variant": "survival_plus_bonus_no_outcome",
        "reward_target_effect": "extrinsic_reward_only",
        "death_mode": "normal",
        "terminal_target_mode": "stock_terminal_no_bootstrap_return_discount_1.0",
        "root_noise": 0.0,
        "dirichlet_alpha": 0.3,
        "policy_noise": 0.0,
        "rnd_enabled": False,
        "exploration_bonus_mode": "none",
        "opponent_policy_ref": "fixed-anchor",
        "opponent_policy_kind": "fixed_anchor",
        "opponent_runtime_mode": "source_state",
        "opponent_death_mode": "normal",
        "natural_bonus_spawn": False,
        "training_seed_policy": "fixed",
        "initialization_source": "same_repo_state",
        "num_unroll_steps": 2,
        "td_steps": 2,
        "discount": 1.0,
        "support_scale": 300,
    }


def _training_artifact(role: str, paths: Mapping[str, Path]) -> dict[str, Any]:
    identity = _model_identity(role)
    base: dict[str, Any] = {
        "schema_id": (
            quality.COMPACT_MATCHED_LEARNING_QUALITY_STOCK_TRAINING_ARTIFACT_SCHEMA_ID
            if role == "stock_reference"
            else quality.COMPACT_MATCHED_LEARNING_QUALITY_COMPACT_TRAINING_ARTIFACT_SCHEMA_ID
        ),
        "ok": True,
        "role": role,
        "route": (
            "stock_train_muzero_reference"
            if role == "stock_reference"
            else "compact_owned_trainer"
        ),
        "run_id": "unit-run",
        "profile_only": False,
        "support_only": False,
        "touches_live_runs": False,
        "model_identity": identity,
        "denominators": _denominators(role),
    }
    if role == "stock_reference":
        base.update(
            {
                "called_train_muzero": True,
                "train_muzero_entrypoint": "lzero.entry.train_muzero",
                "checkpoint_selection": {
                    "initial_ref": "iteration_0.pth.tar",
                    "final_ref": "iteration_100.pth.tar",
                },
                "downloaded_checkpoints": {
                    "initial_path": str(paths["initial_checkpoint"].resolve()),
                    "initial_sha256": _file_sha256(paths["initial_checkpoint"]),
                    "final_path": str(paths["final_checkpoint"].resolve()),
                    "final_sha256": _file_sha256(paths["final_checkpoint"]),
                },
            }
        )
    else:
        base.update(
            {
                "calls_train_muzero": False,
                "real_compact_owned_training_work": True,
                "compact_training_entrypoint": (
                    "curvyzero.training.compact_owned_trainer."
                    "CompactOwnedTrainerV1.record_step"
                ),
                "sample_source": "compact_env_search_replay_rows",
                "synthetic_resident_sample": False,
                "real_env_search_replay_rows": True,
                "initial_checkpoint_path": str(paths["initial_checkpoint"].resolve()),
                "final_checkpoint_path": str(paths["final_checkpoint"].resolve()),
                "pre_eval_checkpoint_ref": "iteration_0.pth.tar",
                "post_eval_checkpoint_ref": "iteration_100.pth.tar",
            }
        )
    return base


def _denominators(role: str) -> dict[str, Any]:
    if role == "stock_reference":
        return {
            "denominator_currency": "stock_train_muzero_learning_quality",
            "env_step_currency": "stock_train_muzero_raw_env_steps",
            "speed_currency": "stock_train_muzero_raw_env_steps_per_sec",
            "collector_envstep_delta": 100,
            "learner_train_calls": 4,
            "replay_sample_calls": 4,
            "checkpoint_iteration_delta": 1,
            "training_wall_sec": 10.0,
            "uses_fallback_denominator": False,
            "wall_sec_used_for_speed_claim": False,
        }
    return {
        "denominator_currency": "compact_owned_learning_quality",
        "env_step_currency": "compact_owned_trainer_env_steps",
        "speed_currency": "compact_trainer_env_steps_per_sec",
        "learner_update_count_delta": 4,
        "sample_batch_count_delta": 4,
        "compact_rollout_rows": 100,
        "compact_sample_rows": 64,
        "training_wall_sec": 8.0,
        "learner_loss_mean": 0.5,
        "uses_fallback_denominator": False,
        "wall_sec_used_for_speed_claim": False,
    }


def _capture_provenance(role: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_id": quality.COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_PROVENANCE_SCHEMA_ID,
        "role": role,
        "producer_id": f"unit-{role}-producer",
        "producer_ran_training": True,
        "producer_ran_pre_eval": True,
        "producer_ran_post_eval": True,
        "feeds_builder": True,
        "support_only": False,
        "training_artifact_ref": "training_artifact",
        "pre_eval_artifact_ref": "pre_eval_summary",
        "post_eval_artifact_ref": "post_eval_summary",
    }
    if role == "stock_reference":
        base["stock_train_muzero"] = {
            "called_train_muzero": True,
            "train_muzero_entrypoint": "lzero.entry.train_muzero",
        }
    else:
        base["compact_owned_training"] = {
            "calls_train_muzero": False,
            "real_compact_owned_training_work": True,
            "compact_training_entrypoint": (
                "curvyzero.training.compact_owned_trainer."
                "CompactOwnedTrainerV1.record_step"
            ),
        }
    return base


def _eval_summary(role: str, *, mean: float, step: int) -> dict[str, Any]:
    return {
        "checkpoint": f"iteration_{step}.pth.tar",
        "eval_seed_set": [101, 102],
        "eval_max_steps": 2000,
        "seeds": 2,
        "mean_steps": mean,
        "median_steps": mean,
        "min_steps": mean - 1,
        "max_steps": mean + 1,
        "ok_count": 2,
        "capped_count": 0,
        "failure_count": 0,
        "outcome_histogram": {"loss": 2},
    }


def _assert_required_artifact_refs(capture: Mapping[str, Any]) -> None:
    refs = {ref["kind"]: ref for ref in capture["artifact_refs"]}
    assert set(refs) == set(quality.REQUIRED_CAPTURE_ARTIFACT_KINDS)
    for ref in refs.values():
        path = Path(ref["path"])
        assert path.is_absolute()
        assert path.is_file()
        assert ref["sha256"] == _file_sha256(path)
        assert ref["required"] is True


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
