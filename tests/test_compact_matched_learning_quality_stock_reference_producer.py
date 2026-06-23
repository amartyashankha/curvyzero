from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from curvyzero.training import compact_promotion_readiness_learning_quality as quality


def test_stock_reference_producer_writes_valid_capture(tmp_path, monkeypatch):
    module = _load_producer_module()
    _patch_stock_producer(module, monkeypatch)
    output_root = tmp_path / "out"

    assert module.main(_cli_args(output_root=output_root)) == 0

    run_dir = output_root / "unit-stock-reference-producer"
    capture_path = run_dir / "stock_reference_capture.json"
    manifest_path = run_dir / "stock_reference_producer_manifest.json"
    capture = _read_json(capture_path)
    manifest = _read_json(manifest_path)

    assert capture["role"] == "stock_reference"
    assert capture["route"] == "stock_train_muzero_reference"
    assert capture["called_train_muzero"] is True
    assert capture["denominators"]["denominator_currency"] == (
        "stock_train_muzero_learning_quality"
    )
    assert capture["denominators"]["collector_envstep_delta"] == 128
    assert capture["denominators"]["learner_train_calls"] == 4
    assert capture["denominators"]["replay_sample_calls"] == 4
    assert capture["denominators"]["checkpoint_iteration_delta"] == 1
    assert capture["eval_settings"]["stock_internal_evaluator_surface"] == {
        "evaluator_env_num": 2,
        "n_evaluator_episode": 2,
        "env_num_matches_episode_count": True,
        "eval_seed_count": 2,
        "evalenv_full_episode_surface": True,
        "empty_ready_set_workaround_formalized": True,
    }
    assert manifest["producer_guardrails"]["eval_bound_to_exact_checkpoint_refs"] is True
    assert (
        manifest["producer_guardrails"][
            "stock_internal_evaluator_env_num_matches_episode_count"
        ]
        is True
    )
    assert not (run_dir / "matched_learning_quality_canary_report.json").exists()
    quality.compact_matched_learning_quality_arm_from_capture_v1(
        capture,
        expected_role="stock_reference",
    )


def test_stock_reference_producer_rejects_nonempty_output_dir_before_training(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    train_called = False

    def fail_if_called(_args):
        nonlocal train_called
        train_called = True
        raise AssertionError("stock train should not run when output dir is stale")

    monkeypatch.setattr(module, "_run_stock_train", fail_if_called)
    output_root = tmp_path / "out"
    run_dir = output_root / "unit-stock-reference-producer"
    run_dir.mkdir(parents=True)
    (run_dir / "stock_reference_capture.json").write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists and is not empty"):
        module.main(_cli_args(output_root=output_root))

    assert train_called is False


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda r: r.update({"mode": "profile"}), "mode must be train"),
        (lambda r: r.update({"called_train_muzero": False}), "must call train_muzero"),
        (
            lambda r: r["command"].update({"stop_after_learner_train_calls": 1}),
            "stop_after_learner_train_calls must be 0",
        ),
        (
            lambda r: r["command"].update({"env_manager_type": "curvyzero_batched_profile"}),
            "env_manager_type=base",
        ),
        (
            lambda r: r["command"].update({"max_env_step": 999}),
            "max_env_step mismatch",
        ),
        (
            lambda r: r["command"].update({"batch_size": 999}),
            "batch_size mismatch",
        ),
        (
            lambda r: r["command"].update({"opponent_runtime_mode": "diagnostic"}),
            "opponent_runtime_mode mismatch",
        ),
        (
            lambda r: r.update({"auto_resume": {"found": True}}),
            "auto_resume found",
        ),
        (
            lambda r: r["target_audit"]["counts"].update({"replay_sample_calls": 0}),
            "replay_sample_calls must be > 0",
        ),
    ],
)
def test_stock_reference_producer_rejects_fake_train_evidence(
    tmp_path,
    monkeypatch,
    mutator,
    match,
):
    module = _load_producer_module()
    _patch_stock_producer(
        module,
        monkeypatch,
        train_mutator=mutator,
    )

    with pytest.raises(ValueError, match=match):
        module.main(_cli_args(output_root=tmp_path / "out"))


def test_stock_reference_producer_rejects_missing_iteration_zero(tmp_path, monkeypatch):
    module = _load_producer_module()
    _patch_stock_producer(
        module,
        monkeypatch,
        train_mutator=lambda r: r["checkpoint_mirror"].update(
            {
                "copied_checkpoints": [
                    {"ref": "training/curvytron/run/checkpoints/lightzero/iteration_1.pth.tar"}
                ]
            }
        ),
    )

    with pytest.raises(ValueError, match="iteration_0"):
        module.main(_cli_args(output_root=tmp_path / "out"))


def test_stock_reference_producer_rejects_missing_learner_metrics(tmp_path, monkeypatch):
    module = _load_producer_module()
    _patch_stock_producer(
        module,
        monkeypatch,
        status_mutator=lambda s: s.update({"learner_metrics_latest_exists": False}),
    )

    with pytest.raises(ValueError, match="learner_metrics_latest must exist"):
        module.main(_cli_args(output_root=tmp_path / "out"))


def test_stock_reference_producer_rejects_zero_learner_train_calls(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    _patch_stock_producer(
        module,
        monkeypatch,
        status_mutator=lambda s: s.update({"learner_train_call_index": 0}),
    )

    with pytest.raises(ValueError, match="learner_train_call_index must be > 0"):
        module.main(_cli_args(output_root=tmp_path / "out"))


def test_stock_reference_producer_rejects_eval_checkpoint_mismatch(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    _patch_stock_producer(
        module,
        monkeypatch,
        eval_builder=lambda ref, point_id, iteration: _eval_summary(
            "training/curvytron/other/checkpoints/lightzero/iteration_999.pth.tar",
            mean=10.0 if point_id == "pre_train" else 13.0,
            iteration=iteration,
        ),
    )

    with pytest.raises(ValueError, match="checkpoint_ref does not match"):
        module.main(_cli_args(output_root=tmp_path / "out"))


def test_stock_reference_producer_rejects_eval_seed_mismatch(tmp_path, monkeypatch):
    module = _load_producer_module()
    _patch_stock_producer(
        module,
        monkeypatch,
        eval_builder=lambda ref, point_id, iteration: _eval_summary(
            ref,
            mean=10.0 if point_id == "pre_train" else 13.0,
            iteration=iteration,
            seeds=[20260833] if point_id == "pre_train" else [20260834],
        ),
    )

    with pytest.raises(ValueError, match="seed sets must match"):
        module.main(_cli_args(output_root=tmp_path / "out"))


def test_stock_reference_producer_rejects_saturated_eval_horizon(tmp_path, monkeypatch):
    module = _load_producer_module()
    _patch_stock_producer(
        module,
        monkeypatch,
        eval_builder=lambda ref, point_id, iteration: _eval_summary(
            ref,
            mean=128.0,
            iteration=iteration,
            capped_count=2,
            outcome="cap",
        ),
    )

    with pytest.raises(ValueError, match="stock_reference saturated eval horizon"):
        module.main(_cli_args(output_root=tmp_path / "out"))


def test_stock_reference_producer_rejects_stale_eval_horizon_before_saturation(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    _patch_stock_producer(
        module,
        monkeypatch,
        eval_builder=lambda ref, point_id, iteration: _eval_summary(
            ref,
            mean=64.0,
            iteration=iteration,
            eval_max_steps=64,
            capped_count=2,
            outcome="cap",
        ),
    )

    with pytest.raises(ValueError, match="eval_max_steps mismatch"):
        module.main(_cli_args(output_root=tmp_path / "out"))


def test_stock_reference_producer_rejects_unchanged_checkpoint_digest(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    _patch_stock_producer(
        module,
        monkeypatch,
        download_writer=lambda ref, path: path.write_text(
            "same checkpoint bytes\n",
            encoding="utf-8",
        ),
    )

    with pytest.raises(ValueError, match="model state digest did not change"):
        module.main(_cli_args(output_root=tmp_path / "out"))


def test_stock_reference_producer_accepts_compact_return_wall_timer(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    _patch_stock_producer(
        module,
        monkeypatch,
        train_mutator=lambda r: (
            r.update({"train_result": "compact modal return"}),
            r.update({"timers_sec": {"train_muzero_wall": 7.5}}),
        ),
    )

    assert module.main(_cli_args(output_root=tmp_path / "out")) == 0
    capture = _read_json(
        tmp_path
        / "out"
        / "unit-stock-reference-producer"
        / "stock_reference_capture.json"
    )
    assert capture["denominators"]["training_wall_sec"] == 7.5


def test_stock_reference_producer_loads_train_summary_for_slim_modal_return(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    loaded_refs: list[str] = []

    def fake_download_json_ref(ref, **kwargs):
        del kwargs
        loaded_refs.append(str(ref))
        summary = _train_result()
        summary["train_result"]["elapsed_sec"] = 8.5
        return summary

    _patch_stock_producer(
        module,
        monkeypatch,
        train_mutator=lambda r: r.update({"train_result": None}),
    )
    monkeypatch.setattr(module, "_download_volume_json_ref", fake_download_json_ref)

    assert module.main(_cli_args(output_root=tmp_path / "out")) == 0
    capture = _read_json(
        tmp_path
        / "out"
        / "unit-stock-reference-producer"
        / "stock_reference_capture.json"
    )
    assert loaded_refs == ["training/curvytron/unit/attempts/a/train/summary.json"]
    assert capture["denominators"]["training_wall_sec"] == 8.5


def test_stock_reference_producer_rejects_unsupported_no_natural_bonus_spawn(tmp_path):
    module = _load_producer_module()

    with pytest.raises(ValueError, match="natural_bonus_spawn=true only"):
        module.main(
            [
                *_cli_args(output_root=tmp_path / "out"),
                "--no-natural-bonus-spawn",
            ]
        )


@pytest.mark.parametrize(
    ("extra_arg", "value", "match"),
    [
        ("--decision-source-frames", "2", "decision_source_frames=1 only"),
        ("--source-physics-step-ms", "20.0", "source_physics_step_ms=50/3 only"),
        ("--n-evaluator-episode", "1", "n_evaluator_episode must be >="),
        (
            "--n-evaluator-episode",
            "3",
            "n_evaluator_episode == evaluator_env_num",
        ),
    ],
)
def test_stock_reference_producer_rejects_unsupported_fixed_train_surface_args(
    tmp_path,
    extra_arg,
    value,
    match,
):
    module = _load_producer_module()

    with pytest.raises(ValueError, match=match):
        module.main(
            [
                *_cli_args(output_root=tmp_path / "out"),
                extra_arg,
                value,
            ]
        )


def test_stock_reference_producer_rejects_hidden_larger_evalenv_default(tmp_path):
    module = _load_producer_module()

    with pytest.raises(ValueError, match="evaluator_env_num == eval_seed_count"):
        module.main(
            [
                *_cli_args(output_root=tmp_path / "out"),
                "--eval-seed-count",
                "32",
                "--evaluator-env-num",
                "2",
                "--n-evaluator-episode",
                "2",
            ]
        )


def test_stock_reference_producer_rejects_train_natural_bonus_mismatch(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    _patch_stock_producer(
        module,
        monkeypatch,
        train_mutator=lambda r: r["command"].update({"natural_bonus_spawn": False}),
    )

    with pytest.raises(ValueError, match="natural_bonus_spawn mismatch"):
        module.main(_cli_args(output_root=tmp_path / "out"))


def test_stock_reference_producer_uses_deployed_modal_functions(monkeypatch):
    module = _load_producer_module()
    resolved: list[tuple[str, str, str | None]] = []
    remote_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class FakeFunction:
        def __init__(self, function_name: str) -> None:
            self.function_name = function_name

        def remote(self, *args, **kwargs):
            remote_calls.append((self.function_name, args, kwargs))
            if self.function_name == "curvytron_run_status":
                return [{"status": "ok"}]
            if (
                self.function_name
                == "lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect"
            ):
                return {"ok": True, "checkpoint_ref": kwargs["checkpoint_ref"]}
            return {"ok": True, "mode": kwargs["mode"]}

    def fake_deployed_modal_function(*, app_name, function_name, modal_env):
        resolved.append((app_name, function_name, modal_env))
        return FakeFunction(function_name)

    monkeypatch.setattr(module, "_deployed_modal_function", fake_deployed_modal_function)
    args = module._parse_args(
        [
            *_cli_args(output_root=Path("out")),
            "--train-app-name",
            "train-app",
            "--status-app-name",
            "status-app",
            "--modal-env",
            "dev",
        ]
    )

    assert module._run_stock_train(args)["mode"] == "train"
    assert module._load_stock_status(args, train_result={})["status"] == "ok"
    assert module._run_stock_eval(
        args=args,
        checkpoint_ref="training/run/checkpoints/lightzero/iteration_0.pth.tar",
        checkpoint_iteration=0,
        point_id="pre_train",
    )["checkpoint_ref"].endswith("iteration_0.pth.tar")

    assert resolved == [
        ("train-app", "lightzero_curvytron_visual_survival_gpu_cpu40", "dev"),
        ("status-app", "curvytron_run_status", "dev"),
        (
            "train-app",
            "lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect",
            "dev",
        ),
    ]
    assert remote_calls[1][1] == (["unit-stock-reference-producer"], ["unit-stock-reference-attempt"])
    assert remote_calls[2][2]["eval_seed_rng_seed"] == 20260833


def test_stock_reference_eval_seed_rng_seed_can_be_overridden(monkeypatch):
    module = _load_producer_module()
    remote_calls: list[dict[str, Any]] = []

    class FakeFunction:
        def remote(self, **kwargs):
            remote_calls.append(kwargs)
            return {"ok": True, "checkpoint_ref": kwargs["checkpoint_ref"]}

    monkeypatch.setattr(
        module,
        "_deployed_modal_function",
        lambda **_kwargs: FakeFunction(),
    )
    args = module._parse_args(
        [
            *_cli_args(output_root=Path("out")),
            "--eval-seed-rng-seed",
            "12345",
        ]
    )

    module._run_stock_eval(
        args=args,
        checkpoint_ref="training/run/checkpoints/lightzero/iteration_0.pth.tar",
        checkpoint_iteration=0,
        point_id="pre_train",
    )

    assert remote_calls[0]["eval_seed_rng_seed"] == 12345


def test_stock_reference_producer_wires_documented_1024x8_eval_shape(monkeypatch):
    module = _load_producer_module()
    remote_calls: list[dict[str, Any]] = []

    class FakeFunction:
        def remote(self, **kwargs):
            remote_calls.append(kwargs)
            return {"ok": True, "checkpoint_ref": kwargs["checkpoint_ref"]}

    monkeypatch.setattr(
        module,
        "_deployed_modal_function",
        lambda **_kwargs: FakeFunction(),
    )
    args = module._parse_args(
        [
            *_cli_args(output_root=Path("out")),
            "--eval-steps",
            "1024",
            "--eval-seed-count",
            "8",
            "--eval-seed-rng-seed",
            "20260833",
            "--eval-compute",
            "cpu",
            "--step-detail-limit",
            "0",
        ]
    )

    module._run_stock_eval(
        args=args,
        checkpoint_ref="training/run/checkpoints/lightzero/iteration_124.pth.tar",
        checkpoint_iteration=124,
        point_id="post_train",
    )

    assert len(remote_calls) == 1
    assert remote_calls[0]["compute"] == "cpu"
    assert remote_calls[0]["seed"] == 20260833
    assert remote_calls[0]["eval_seed_count"] == 8
    assert remote_calls[0]["eval_seed_rng_seed"] == 20260833
    assert remote_calls[0]["max_eval_steps"] == 1024
    assert remote_calls[0]["step_detail_limit"] == 0
    assert remote_calls[0]["checkpoint_label"] == "iteration_124"


def test_stock_reference_producer_wires_documented_2048x32_evalenv_surface(monkeypatch):
    module = _load_producer_module()
    args = module._parse_args(
        [
            *_cli_args(output_root=Path("out")),
            "--eval-steps",
            "2048",
            "--eval-seed-count",
            "32",
            "--evaluator-env-num",
            "32",
            "--n-evaluator-episode",
            "32",
        ]
    )

    assert module._stock_train_kwargs(args)["evaluator_env_num"] == 32
    assert module._stock_train_kwargs(args)["n_evaluator_episode"] == 32
    assert module._stock_internal_evaluator_surface(args) == {
        "evaluator_env_num": 32,
        "n_evaluator_episode": 32,
        "env_num_matches_episode_count": True,
        "eval_seed_count": 32,
        "evalenv_full_episode_surface": True,
        "empty_ready_set_workaround_formalized": True,
    }


def test_stock_reference_download_uses_v2_volume_kwargs(tmp_path, monkeypatch):
    module = _load_producer_module()
    calls: list[dict[str, Any]] = []

    class FakeVolume:
        @staticmethod
        def from_name(name, **kwargs):
            calls.append({"name": name, **kwargs})
            return FakeVolume()

        def read_file_into_fileobj(self, remote_path, buffer):
            assert remote_path == "/training/run/checkpoints/lightzero/iteration_0.pth.tar"
            buffer.write(b"checkpoint")

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Volume=FakeVolume))

    output_path = tmp_path / "iteration_0.pth.tar"
    module._download_volume_ref(
        "training/run/checkpoints/lightzero/iteration_0.pth.tar",
        output_path,
        volume_name="curvyzero-runs-v2",
        modal_env="dev",
    )

    assert output_path.read_bytes() == b"checkpoint"
    assert calls == [
        {
            "name": "curvyzero-runs-v2",
            "environment_name": "dev",
            "create_if_missing": False,
            "version": 2,
        }
    ]


def _load_producer_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_matched_learning_quality_stock_reference_producer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_matched_learning_quality_stock_reference_producer_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patch_stock_producer(
    module,
    monkeypatch,
    *,
    train_mutator: Callable[[dict[str, Any]], None] | None = None,
    status_mutator: Callable[[dict[str, Any]], None] | None = None,
    eval_builder: Callable[[str, str, int], Mapping[str, Any]] | None = None,
    download_writer: Callable[[str, Path], None] | None = None,
) -> None:
    def fake_run_stock_train(args):
        del args
        result = _train_result()
        if train_mutator is not None:
            train_mutator(result)
        return result

    def fake_load_stock_status(args, *, train_result):
        del args, train_result
        status = {
            "learner_metrics_latest_exists": True,
            "learner_train_call_index": 4,
            "learner_collector_envstep": 128,
        }
        if status_mutator is not None:
            status_mutator(status)
        return status

    def fake_download(ref, output_path, **kwargs):
        del kwargs
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if download_writer is not None:
            download_writer(ref, path)
        else:
            path.write_text(f"checkpoint bytes for {ref}\n", encoding="utf-8")

    def fake_run_stock_eval(*, args, checkpoint_ref, checkpoint_iteration, point_id):
        del args
        if eval_builder is not None:
            return eval_builder(str(checkpoint_ref), str(point_id), int(checkpoint_iteration))
        return _eval_summary(
            str(checkpoint_ref),
            mean=10.0 if point_id == "pre_train" else 13.0,
            iteration=int(checkpoint_iteration),
        )

    monkeypatch.setattr(module, "_run_stock_train", fake_run_stock_train)
    monkeypatch.setattr(module, "_load_stock_status", fake_load_stock_status)
    monkeypatch.setattr(module, "_download_volume_ref", fake_download)
    monkeypatch.setattr(module, "_run_stock_eval", fake_run_stock_eval)


def _cli_args(*, output_root: Path) -> list[str]:
    return [
        "--run-id",
        "unit-stock-reference-producer",
        "--attempt-id",
        "unit-stock-reference-attempt",
        "--candidate-checkpoint-id",
        "unit-compact-ckpt",
        "--output-root",
        str(output_root),
    ]


def _train_result() -> dict[str, Any]:
    return {
        "ok": True,
        "status": "completed",
        "mode": "train",
        "compute": "gpu-l4-t4-cpu40",
        "run_id": "unit-stock-reference-producer",
        "attempt_id": "unit-stock-reference-attempt",
        "called_train_muzero": True,
        "trainer_entrypoint": "lzero.entry.train_muzero",
        "summary_ref": "training/curvytron/unit/attempts/a/train/summary.json",
        "command": {
            "seed": 20260530,
            "max_env_step": 512,
            "max_train_iter": 1,
            "source_max_steps": 1_048_576,
            "collector_env_num": 2,
            "evaluator_env_num": 2,
            "n_episode": 2,
            "n_evaluator_episode": 2,
            "num_simulations": 1,
            "batch_size": 2,
            "save_ckpt_after_iter": 1,
            "decision_ms": 50.0 / 3.0,
            "env_variant": "source_state_fixed_opponent",
            "reward_variant": "survival_plus_bonus_no_outcome",
            "policy_observation_backend": "cpu_oracle",
            "collect_search_backend": "stock",
            "collect_search_ctree_backend": "lightzero",
            "env_manager_type": "base",
            "stop_after_learner_train_calls": 0,
            "background_eval_enabled": False,
            "decision_source_frames": 1,
            "source_physics_step_ms": 50.0 / 3.0,
            "opponent_policy_kind": "fixed_straight",
            "opponent_runtime_mode": "normal",
            "opponent_death_mode": "normal",
            "natural_bonus_spawn": True,
        },
        "train_result": {
            "ok": True,
            "elapsed_sec": 10.0,
        },
        "target_audit": {
            "counts": {
                "replay_sample_calls": 4,
            }
        },
        "checkpoint_mirror": {
            "copied_checkpoints": [
                {"ref": "training/curvytron/unit/checkpoints/lightzero/iteration_0.pth.tar"},
                {"ref": "training/curvytron/unit/checkpoints/lightzero/iteration_1.pth.tar"},
                {"ref": "training/curvytron/unit/checkpoints/lightzero/ckpt_best.pth.tar"},
            ]
        },
        "auto_resume": {"found": False},
        "runtime_compute": {"gpu": "unit"},
        "artifact_refs": {},
    }


def _eval_summary(
    checkpoint_ref: str,
    *,
    mean: float,
    iteration: int,
    seeds: list[int] | None = None,
    eval_max_steps: int = 128,
    capped_count: int = 0,
    outcome: str = "loss",
) -> dict[str, Any]:
    seed_values = seeds or [20260833, 20260834]
    return {
        "ok": True,
        "checkpoint_ref": checkpoint_ref,
        "eval_max_steps": int(eval_max_steps),
        "selection": {
            "checkpoint_ref": checkpoint_ref,
            "checkpoint_refs": checkpoint_ref,
            "eval_seed_values": seed_values,
        },
        "config": {
            "checkpoint_ref": checkpoint_ref,
        },
        "survival_aggregate_table": [
            {
                "checkpoint": f"iteration_{iteration}",
                "checkpoint_ref": checkpoint_ref,
                "checkpoint_step": int(iteration),
                "seeds": len(seed_values),
                "mean_steps": mean,
                "median_steps": mean,
                "min_steps": mean - 1.0,
                "max_steps": mean + 1.0,
                "ok_count": len(seed_values),
                "capped_count": capped_count,
                "failure_count": 0,
                "outcome_histogram": {outcome: len(seed_values)},
            }
        ],
        "survival_table": [
            {
                "checkpoint": f"iteration_{iteration}",
                "seed": seed,
                "steps": mean,
                "cap": int(eval_max_steps),
                "terminal": outcome,
                "ok": True,
            }
            for seed in seed_values
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
