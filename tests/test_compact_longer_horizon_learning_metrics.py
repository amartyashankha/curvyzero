from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from curvyzero.training import compact_longer_horizon_learning_metrics as metrics


def test_longer_horizon_report_validates_three_checkpoint_trace(tmp_path):
    payload = _build_report(tmp_path)

    assert payload["schema_id"] == metrics.COMPACT_LONGER_HORIZON_LEARNING_METRICS_SCHEMA_ID
    assert payload["ok"] is True
    assert payload["readiness"]["longer_horizon_compact_learning_metrics"] is True
    assert payload["readiness"]["promotion_readiness_complete"] is False
    assert payload["non_claims"]["promotion_claim"] is False
    assert payload["non_claims"]["calls_train_muzero"] is False
    assert len(payload["checkpoint_series"]) == 3
    assert payload["cumulative_denominators"]["learner_update_count_delta"] == 2
    metrics.validate_compact_longer_horizon_learning_metrics_v1(payload)


def test_longer_horizon_report_rejects_two_checkpoints(tmp_path):
    with pytest.raises(
        metrics.CompactLongerHorizonLearningMetricsError,
        match="at least three checkpoint",
    ):
        _build_report(tmp_path, checkpoint_count=2)


def test_longer_horizon_report_rejects_unchanged_adjacent_digest(tmp_path):
    def mutate(points: list[dict[str, Any]]) -> None:
        points[2]["model_state_digest"] = points[1]["model_state_digest"]

    with pytest.raises(
        metrics.CompactLongerHorizonLearningMetricsError,
        match="adjacent model digest did not change",
    ):
        _build_report(tmp_path, point_mutator=mutate)


def test_longer_horizon_report_rejects_nonmonotonic_denominator(tmp_path):
    def mutate(points: list[dict[str, Any]]) -> None:
        points[2]["cumulative_denominators"]["learner_update_count_delta"] = 0
        points[2]["interval_denominators"]["learner_update_count_delta"] = -1

    with pytest.raises(
        metrics.CompactLongerHorizonLearningMetricsError,
        match="non-monotonic",
    ):
        _build_report(tmp_path, point_mutator=mutate)


def test_longer_horizon_report_rejects_eval_seed_drift(tmp_path):
    def mutate(points: list[dict[str, Any]]) -> None:
        eval_path = Path(points[2]["eval_summary_path"])
        payload = _read_json(eval_path)
        payload["eval_seed_set"] = [101, 999]
        _write_json(eval_path, payload)

    with pytest.raises(
        metrics.CompactLongerHorizonLearningMetricsError,
        match="eval_seed_set mismatch",
    ):
        _build_report(tmp_path, point_mutator=mutate)


def test_longer_horizon_report_rejects_claim_flip(tmp_path):
    payload = _build_report(tmp_path)
    payload["non_claims"]["promotion_claim"] = True

    with pytest.raises(
        metrics.CompactLongerHorizonLearningMetricsError,
        match="promotion_claim must be false",
    ):
        metrics.validate_compact_longer_horizon_learning_metrics_v1(payload)


def test_longer_horizon_producer_rejects_nonempty_output_before_training(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    run_id = "unit-longer-horizon-stale"
    output_root = tmp_path / "out"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "stale.txt").write_text("stale\n", encoding="utf-8")
    train_called = False

    def fail_if_called(**_kwargs):
        nonlocal train_called
        train_called = True
        raise AssertionError("training should not run with stale output")

    monkeypatch.setattr(module, "_train_compact_longer_horizon_trace", fail_if_called)

    with pytest.raises(FileExistsError, match="already exists and is not empty"):
        module.main(
            [
                "--run-id",
                run_id,
                "--output-root",
                str(output_root),
            ]
        )

    assert train_called is False


def test_longer_horizon_producer_fast_path_writes_report(tmp_path, monkeypatch):
    module = _load_producer_module()
    lifecycle_path, compatibility_path, compact_path = _write_input_reports(tmp_path)
    context = module.candidate.CompactCandidateContext(
        model=object(),
        optimizer=object(),
        checkpoint=object(),
        surface={"policy": {"observation_shape": [4, 64, 64], "action_space_size": 3}},
        checkpoint_path=compact_path,
        checkpoint_sha256=_file_sha256(compact_path),
        checkpoint_id="unit-compact-ckpt",
    )

    def fake_load_context(**_kwargs):
        return context

    def fake_train(**_kwargs):
        return _producer_snapshots(module)

    def fake_save_compact(_checkpoint, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"compact {path.name}\n", encoding="utf-8")
        return path

    def fake_save_stock(**kwargs):
        path = Path(kwargs["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"stock {path.name}\n", encoding="utf-8")
        return {"checkpoint_path": path, "sidecar_path": path.with_suffix(".json")}

    def fake_eval(**kwargs):
        args = kwargs["args"]
        checkpoint_path = Path(kwargs["checkpoint_path"]).resolve()
        output_path = Path(kwargs["output_path"])
        index = int(checkpoint_path.name.removeprefix("iteration_").removesuffix(".pth.tar"))
        seeds = module.candidate._eval_seed_values(
            args=args,
            fallback_seed=int(kwargs["seed"]),
        )
        payload = _eval_summary(
            checkpoint_path,
            mean=10.0 + index,
            step=index,
            seeds=seeds,
        )
        _write_json(output_path, payload)
        return payload

    monkeypatch.setattr(module.candidate, "_load_compact_candidate_context", fake_load_context)
    monkeypatch.setattr(module, "_train_compact_longer_horizon_trace", fake_train)
    monkeypatch.setattr(module, "_save_compact_checkpoint", fake_save_compact)
    monkeypatch.setattr(module.candidate, "_save_stock_checkpoint_export", fake_save_stock)
    monkeypatch.setattr(module.candidate, "_run_eval_summary", fake_eval)

    run_id = "unit-longer-horizon-fast"
    assert module.main(
        [
            "--run-id",
            run_id,
            "--output-root",
            str(tmp_path / "out"),
            "--unified-lifecycle-report",
            str(lifecycle_path),
            "--compatibility-report",
            str(compatibility_path),
        ]
    ) == 0

    run_dir = tmp_path / "out" / run_id
    report_path = run_dir / "longer_horizon_learning_metrics_report.json"
    manifest_path = run_dir / "longer_horizon_learning_metrics_producer_manifest.json"
    assert report_path.is_file()
    assert manifest_path.is_file()
    assert not (run_dir / "matched_learning_quality_canary_report.json").exists()
    report = _read_json(report_path)
    metrics.validate_compact_longer_horizon_learning_metrics_v1(report)


def _build_report(
    tmp_path: Path,
    *,
    checkpoint_count: int = 3,
    point_mutator=None,
) -> dict[str, Any]:
    lifecycle_path, compatibility_path, _compact_path = _write_input_reports(tmp_path)
    eval_settings = _eval_settings()
    points = [_checkpoint_point(tmp_path, index, eval_settings) for index in range(checkpoint_count)]
    if point_mutator is not None:
        point_mutator(points)
    return metrics.build_compact_longer_horizon_learning_metrics_v1(
        run_id="unit-longer-horizon",
        compatibility_report_path=compatibility_path,
        unified_lifecycle_report_path=lifecycle_path,
        checkpoint_series=points,
        source_fingerprint={
            "git_commit": "unit",
            "producer_script": "unit",
            "producer_route": "unit",
        },
        eval_settings=eval_settings,
        training_settings=_training_settings(),
        created_at="2026-05-30T00:00:00+00:00",
    )


def _checkpoint_point(
    tmp_path: Path,
    index: int,
    eval_settings: Mapping[str, Any],
) -> dict[str, Any]:
    compact_path = tmp_path / f"compact_iteration_{index}.pt"
    stock_path = tmp_path / f"iteration_{index}.pth.tar"
    eval_path = tmp_path / f"eval_{index}.json"
    compact_path.write_text(f"compact checkpoint {index}\n", encoding="utf-8")
    stock_path.write_text(f"stock export {index}\n", encoding="utf-8")
    _write_json(eval_path, _eval_summary(stock_path, mean=10.0 + index, step=index))
    cumulative = _denominators(index)
    previous = _denominators(max(0, index - 1))
    interval = {
        key: cumulative[key] - previous.get(key, 0)
        for key in cumulative
    }
    if index == 0:
        interval = {key: 0 for key in cumulative}
    return {
        "candidate_checkpoint_id": "unit-compact-ckpt",
        "point_id": "pre_train" if index == 0 else "post_train" if index == 2 else f"checkpoint_{index}",
        "checkpoint_index": index,
        "checkpoint_step": index,
        "checkpoint_id": f"unit-ckpt-{index}",
        "compact_checkpoint_path": str(compact_path),
        "stock_export_path": str(stock_path),
        "eval_summary_path": str(eval_path),
        "model_state_digest": chr(ord("a") + index) * 64,
        "digest_changed_from_previous": index > 0,
        "trainer_counters": {
            "train_step": index,
            "learner_update_count": index,
            "sample_batch_count": index,
            "policy_refresh_count": index,
        },
        "interval_denominators": interval,
        "cumulative_denominators": cumulative,
        "learner_metrics": {} if index == 0 else _learner_metrics(index),
        "loop_metrics": {},
        "training_metrics_lineage": None if index == 0 else _lineage(index),
        "sample_source": "compact_env_search_replay_rows",
        "compact_training_entrypoint": metrics.COMPACT_RECORD_STEP_ENTRYPOINT,
        "eval_seed_set": list(eval_settings["eval_seed_set"]),
        "eval_max_steps": int(eval_settings["eval_max_steps"]),
    }


def _write_input_reports(tmp_path: Path) -> tuple[Path, Path, Path]:
    compact_path = tmp_path / "loaded_compact_checkpoint.pt"
    compact_path.write_text("loaded compact checkpoint\n", encoding="utf-8")
    lifecycle_path = tmp_path / "unified_lifecycle_report.json"
    compatibility_path = tmp_path / "compatibility_report.json"
    _write_json(
        lifecycle_path,
        {
            "schema_id": "curvyzero_compact_unified_lifecycle_smoke/v1",
            "ok": True,
            "checkpoint_id": "unit-compact-ckpt",
            "compact_checkpoint_path": str(compact_path),
            "lifecycle_gates_complete": True,
            "promotion_claim": False,
            "promotion_eligible": False,
        },
    )
    _write_json(
        compatibility_path,
        {
            "schema_id": "curvyzero_compact_coach_compatibility_refresh/v1",
            "ok": True,
            "candidate_checkpoint_id": "unit-compact-ckpt",
            "promotion_eligible": True,
            "coach_speed_row_gate": True,
            "promotion_claim": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
        },
    )
    return lifecycle_path, compatibility_path, compact_path


def _eval_settings() -> dict[str, Any]:
    return {
        "eval_seed_set": [101, 102],
        "eval_episode_count": 2,
        "eval_max_steps": 128,
        "num_simulations": 1,
        "batch_size": 2,
        "reward_variant": "survival_plus_bonus_no_outcome",
        "death_mode": "normal",
        "root_noise": 0.0,
        "policy_noise": 0.0,
        "rnd_enabled": False,
        "opponent_policy_kind": "fixed_straight",
    }


def _training_settings() -> dict[str, Any]:
    return {
        "route": "compact_owned_trainer",
        "sample_source": "compact_env_search_replay_rows",
        "compact_training_entrypoint": metrics.COMPACT_RECORD_STEP_ENTRYPOINT,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "training_speedup_claim": False,
        "wall_sec_used_for_speed_claim": False,
    }


def _eval_summary(
    checkpoint_path: Path,
    *,
    mean: float,
    step: int,
    seeds: list[int] | tuple[int, ...] = (101, 102),
) -> dict[str, Any]:
    checkpoint = str(checkpoint_path.resolve())
    seed_values = [int(seed) for seed in seeds]
    return {
        "checkpoint": checkpoint,
        "checkpoint_ref": checkpoint,
        "checkpoint_step": int(step),
        "eval_seed_set": seed_values,
        "eval_max_steps": 128,
        "survival_aggregate_table": [
            {
                "checkpoint": f"iteration_{step}",
                "checkpoint_ref": checkpoint,
                "checkpoint_step": int(step),
                "seeds": len(seed_values),
                "eval_episode_count": len(seed_values),
                "mean_steps": float(mean),
                "median_steps": float(mean),
                "min_steps": float(mean) - 1.0,
                "max_steps": float(mean) + 1.0,
                "ok_count": len(seed_values),
                "capped_count": 0,
                "failure_count": 0,
                "outcome_histogram": {"loss": len(seed_values)},
            }
        ],
        "survival_table": [
            {
                "checkpoint": f"iteration_{step}",
                "checkpoint_ref": checkpoint,
                "seed": seed,
                "steps": float(mean),
                "cap": 128,
                "terminal": "loss",
                "ok": True,
            }
            for seed in seed_values
        ],
    }


def _denominators(index: int) -> dict[str, Any]:
    if index == 0:
        return {
            "train_step_delta": 0,
            "learner_update_count_delta": 0,
            "sample_batch_count_delta": 0,
            "record_step_calls": 0,
            "appended_replay_entry_count": 0,
            "sampled_count": 0,
            "trained_count": 0,
            "compact_rollout_rows": 0,
            "compact_sample_rows": 0,
            "replay_store_entry_count": 0,
            "replay_store_index_row_count": 0,
        }
    return {
        "train_step_delta": index,
        "learner_update_count_delta": index,
        "sample_batch_count_delta": index,
        "record_step_calls": index,
        "appended_replay_entry_count": index,
        "sampled_count": index,
        "trained_count": index,
        "compact_rollout_rows": index * 2,
        "compact_sample_rows": index * 2,
        "replay_store_entry_count": index,
        "replay_store_index_row_count": index * 2,
    }


def _learner_metrics(index: int) -> dict[str, Any]:
    return {
        "learner_loss": 0.2 + index,
        "learner_policy_loss": 0.1 + index,
        "learner_value_loss": 0.1 + index,
        "learner_reward_loss": 0.1 + index,
        "learner_grad_norm_before_clip": 1.0 + index,
        "learner_sample_rows": 2,
        "learner_train_steps": 1,
    }


def _lineage(index: int) -> dict[str, Any]:
    return {
        "compact_training_metrics_lineage_schema_id": (
            "curvyzero_compact_training_metrics_lineage/v1"
        ),
        "training_metrics_lineage_status": "compact_training_metrics_lineage_v1",
        "checkpoint_id": f"unit-ckpt-{index}",
        "trainer_id": "unit-trainer",
        "policy_version_ref": f"policy-{index}",
        "model_version_ref": f"model-{index}",
        "policy_source": "unit",
        "train_step": index,
        "learner_update_count": index,
        "sample_batch_count": index,
        "learner_metric_keys": [
            "compact_muzero_learner_loss",
            "compact_muzero_learner_policy_loss",
            "compact_muzero_learner_value_loss",
            "compact_muzero_learner_reward_loss",
            "compact_muzero_learner_grad_norm_before_clip",
        ],
        "learner_loss": 0.2 + index,
        "learner_policy_loss": 0.1 + index,
        "learner_value_loss": 0.1 + index,
        "learner_reward_loss": 0.1 + index,
        "learner_grad_norm_before_clip": 1.0 + index,
        "learner_sample_rows": 2,
        "learner_train_steps": 1,
        "learner_input_bytes": 0,
        "resident_sample_used": True,
        "device_replay_index_rows_sample": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "promotion_claim": False,
        "training_speed_claim": False,
        "replay_store_state_schema_id": "unit_replay_state/v1",
        "replay_store_entry_count": index,
        "replay_store_index_row_count": index * 2,
        "compact_owned_loop_schema_id": "curvyzero_compact_owned_loop/v1",
        "sample_gate_calls": index,
        "sample_gate_sample_rows": index * 2,
        "learner_gate_updates": index,
        "learner_gate_sample_rows": index * 2,
        "learner_gate_input_bytes": 0,
        "search_provenance": {
            "search_impl": "compact_torch_search",
            "num_simulations": 1,
            "active_root_count": 2,
            "root_batch_schema_id": "unit_root_batch/v1",
            "search_result_schema_id": "unit_search_result/v1",
            "replay_payload_schema_id": "unit_replay_payload/v1",
            "search_replay_payload_digest": f"digest-{index}",
        },
        "evidence_refs": [f"unit-ref-{index}"],
    }


def _producer_snapshots(module) -> list[Any]:
    zero = _denominators(0)
    snapshots = []
    for index in range(3):
        cumulative = _denominators(index)
        previous = _denominators(max(0, index - 1))
        interval = (
            {key: 0 for key in cumulative}
            if index == 0
            else {key: cumulative[key] - previous.get(key, 0) for key in cumulative}
        )
        snapshots.append(
            module.CompactLongerHorizonSnapshot(
                checkpoint_index=index,
                checkpoint_step=index,
                checkpoint_id=f"unit-ckpt-{index}",
                checkpoint=object(),
                model_state_digest=chr(ord("a") + index) * 64,
                digest_changed_from_previous=index > 0,
                trainer_counters={},
                interval_denominators=interval if index > 0 else zero,
                cumulative_denominators=cumulative,
                learner_metrics={} if index == 0 else _learner_metrics(index),
                loop_metrics={},
                training_metrics_lineage=None if index == 0 else _lineage(index),
            )
        )
    return snapshots


def _load_producer_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_longer_horizon_learning_metrics_producer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_longer_horizon_learning_metrics_producer_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
