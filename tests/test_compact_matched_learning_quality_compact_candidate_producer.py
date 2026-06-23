from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def test_compact_candidate_producer_resident_sample_scaffold_fails_quality_gate(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    lifecycle_path = _write_lifecycle(tmp_path)
    _patch_fast_producer(
        module,
        tmp_path,
        monkeypatch,
        sample_source="deterministic_resident_lightzero_sample",
    )
    output_root = tmp_path / "out"

    with pytest.raises(ValueError, match="compact_env_search_replay_rows sample source"):
        module.main(
            [
                "--run-id",
                "unit-compact-candidate-producer",
                "--output-root",
                str(output_root),
                "--unified-lifecycle-report",
                str(lifecycle_path),
            ]
        )

    run_dir = output_root / "unit-compact-candidate-producer"
    assert not (run_dir / "compact_candidate_capture.json").exists()
    assert not (run_dir / "matched_learning_quality_canary_report.json").exists()


def test_compact_candidate_producer_rejects_nonempty_output_dir_before_training(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    lifecycle_path = _write_lifecycle(tmp_path)
    train_called = False

    def fail_if_called(**_kwargs):
        nonlocal train_called
        train_called = True
        raise AssertionError("compact training should not run when output dir is stale")

    monkeypatch.setattr(module, "_train_compact_candidate_once", fail_if_called)
    output_root = tmp_path / "out"
    run_dir = output_root / "unit-compact-candidate-producer"
    run_dir.mkdir(parents=True)
    (run_dir / "compact_candidate_capture.json").write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists and is not empty"):
        module.main(
            [
                "--run-id",
                "unit-compact-candidate-producer",
                "--output-root",
                str(output_root),
                "--unified-lifecycle-report",
                str(lifecycle_path),
            ]
        )

    assert train_called is False


def test_compact_candidate_producer_rejects_unchanged_model_digest(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    lifecycle_path = _write_lifecycle(tmp_path)
    _patch_fast_producer(
        module,
        tmp_path,
        monkeypatch,
        training_mutator=lambda module, result: module.CompactCandidateTrainingResult(
            final_checkpoint=result.final_checkpoint,
            initial_model_state_digest=result.initial_model_state_digest,
            final_model_state_digest=result.initial_model_state_digest,
            trainer_counters_before=result.trainer_counters_before,
            trainer_counters_after=result.trainer_counters_after,
            learner_telemetry=result.learner_telemetry,
            learner_update_count_delta=result.learner_update_count_delta,
            sample_batch_count_delta=result.sample_batch_count_delta,
            compact_rollout_rows=result.compact_rollout_rows,
            compact_sample_rows=result.compact_sample_rows,
            training_wall_sec=result.training_wall_sec,
            learner_loss_mean=result.learner_loss_mean,
            sample_source=result.sample_source,
        ),
    )

    with pytest.raises(ValueError, match="model state digest did not change"):
        module.main(
            [
                "--run-id",
                "unit-unchanged-digest",
                "--output-root",
                str(tmp_path / "out"),
                "--unified-lifecycle-report",
                str(lifecycle_path),
            ]
        )


def test_compact_candidate_producer_rejects_eval_checkpoint_mismatch(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    lifecycle_path = _write_lifecycle(tmp_path)
    _patch_fast_producer(
        module,
        tmp_path,
        monkeypatch,
        eval_summary_builder=lambda checkpoint_path, mean, step: _eval_summary(
            Path("iteration_999.pth.tar"),
            mean=mean,
            step=step,
        ),
    )

    with pytest.raises(ValueError, match="checkpoint reference does not match"):
        module.main(
            [
                "--run-id",
                "unit-eval-mismatch",
                "--output-root",
                str(tmp_path / "out"),
                "--unified-lifecycle-report",
                str(lifecycle_path),
            ]
        )


def test_compact_candidate_producer_rejects_same_basename_different_checkpoint_dir(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    lifecycle_path = _write_lifecycle(tmp_path)
    _patch_fast_producer(
        module,
        tmp_path,
        monkeypatch,
        eval_summary_builder=lambda checkpoint_path, mean, step: _eval_summary(
            Path("other-run") / checkpoint_path.name,
            mean=mean,
            step=step,
        ),
    )

    with pytest.raises(ValueError, match="checkpoint reference does not match"):
        module.main(
            [
                "--run-id",
                "unit-eval-same-basename-wrong-dir",
                "--output-root",
                str(tmp_path / "out"),
                "--unified-lifecycle-report",
                str(lifecycle_path),
            ]
        )


def test_compact_candidate_producer_rejects_zero_sample_rows(
    tmp_path,
    monkeypatch,
):
    module = _load_producer_module()
    lifecycle_path = _write_lifecycle(tmp_path)
    _patch_fast_producer(
        module,
        tmp_path,
        monkeypatch,
        training_mutator=lambda module, result: module.CompactCandidateTrainingResult(
            final_checkpoint=result.final_checkpoint,
            initial_model_state_digest=result.initial_model_state_digest,
            final_model_state_digest=result.final_model_state_digest,
            trainer_counters_before=result.trainer_counters_before,
            trainer_counters_after=result.trainer_counters_after,
            learner_telemetry=result.learner_telemetry,
            learner_update_count_delta=result.learner_update_count_delta,
            sample_batch_count_delta=result.sample_batch_count_delta,
            compact_rollout_rows=result.compact_rollout_rows,
            compact_sample_rows=0,
            training_wall_sec=result.training_wall_sec,
            learner_loss_mean=result.learner_loss_mean,
            sample_source=result.sample_source,
        ),
    )

    with pytest.raises(ValueError, match="compact_sample_rows must be positive"):
        module.main(
            [
                "--run-id",
                "unit-zero-sample-rows",
                "--output-root",
                str(tmp_path / "out"),
                "--unified-lifecycle-report",
                str(lifecycle_path),
            ]
        )


def test_compact_candidate_env_search_replay_training_calls_trainer_record_step(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    module = _load_producer_module()
    checkpoint_path = tmp_path / "loaded_compact_checkpoint.pt"
    checkpoint_path.write_text("unit compact checkpoint\n", encoding="utf-8")
    model = _TinyMuZero(torch)
    context = module.CompactCandidateContext(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        checkpoint=SimpleNamespace(
            resume_state=SimpleNamespace(
                trainer_id="unit-trainer",
                train_step=0,
                learner_update_count=1,
                sample_batch_count=0,
                policy_version_ref="unit-policy-v1",
                model_version_ref="unit-model-v1",
                policy_source="unit_test_env_search_replay",
            )
        ),
        surface={
            "policy": {
                "observation_shape": [4, 64, 64],
                "action_space_size": 3,
                "support_scale": 1,
            }
        },
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_file_sha256(checkpoint_path),
        checkpoint_id="unit-compact-ckpt",
    )
    args = module._parse_args(
        [
            "--run-id",
            "unit-env-search-replay",
            "--compact-env-steps",
            "4",
            "--compact-warmup-steps",
            "1",
            "--compact-sample-batch-size",
            "1",
            "--compact-replay-pair-capacity",
            "8",
            "--batch-size",
            "1",
            "--train-steps",
            "1",
            "--num-simulations",
            "1",
            "--learner-device",
            "cpu",
        ]
    )

    result = module._train_compact_candidate_env_search_replay_once(
        args=args,
        context=context,
        run_id="unit-env-search-replay",
    )
    module._validate_training_result(result)

    assert result.sample_source == module.COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE
    assert result.learner_update_count_delta > 0
    assert result.sample_batch_count_delta > 0
    assert result.compact_rollout_rows > 0
    assert result.compact_sample_rows > 0
    assert result.learner_telemetry["compact_muzero_learner_calls_train_muzero"] is False
    assert "record_step" in module._compact_training_entrypoint(result)


class _TinyMuZero:
    def __init__(self, torch):
        super().__init__()
        self._module = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 8),
            torch.nn.Tanh(),
        )
        self._action_embedding = torch.nn.Embedding(3, 8)
        self._policy_head = torch.nn.Linear(8, 3)
        self._value_head = torch.nn.Linear(8, 3)
        self._reward_head = torch.nn.Linear(8, 3)

    def parameters(self):
        yield from self._module.parameters()
        yield from self._action_embedding.parameters()
        yield from self._policy_head.parameters()
        yield from self._value_head.parameters()
        yield from self._reward_head.parameters()

    def to(self, device):
        self._module.to(device)
        self._action_embedding.to(device)
        self._policy_head.to(device)
        self._value_head.to(device)
        self._reward_head.to(device)
        return self

    def train(self):
        self._module.train()
        self._action_embedding.train()
        self._policy_head.train()
        self._value_head.train()
        self._reward_head.train()
        return self

    def state_dict(self):
        return {
            "module": self._module.state_dict(),
            "action_embedding": self._action_embedding.state_dict(),
            "policy_head": self._policy_head.state_dict(),
            "value_head": self._value_head.state_dict(),
            "reward_head": self._reward_head.state_dict(),
        }

    def load_state_dict(self, state):
        self._module.load_state_dict(state["module"])
        self._action_embedding.load_state_dict(state["action_embedding"])
        self._policy_head.load_state_dict(state["policy_head"])
        self._value_head.load_state_dict(state["value_head"])
        self._reward_head.load_state_dict(state["reward_head"])

    def initial_inference(self, obs):
        from lzero.model.common import MZNetworkOutput

        latent = self._module(obs)
        return MZNetworkOutput(
            self._value_head(latent),
            self._value_head(latent).new_zeros(self._value_head(latent).shape),
            self._policy_head(latent),
            latent,
        )

    def recurrent_inference(self, latent_state, action):
        import torch

        from lzero.model.common import MZNetworkOutput

        next_latent = torch.tanh(
            latent_state + self._action_embedding(action.reshape(-1).long())
        )
        return MZNetworkOutput(
            self._value_head(next_latent),
            self._reward_head(next_latent),
            self._policy_head(next_latent),
            next_latent,
        )


def _load_producer_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_matched_learning_quality_compact_candidate_producer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_matched_learning_quality_compact_candidate_producer_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patch_fast_producer(
    module,
    tmp_path: Path,
    monkeypatch,
    *,
    training_mutator: Callable[[Any, Any], Any] | None = None,
    eval_summary_builder: Callable[[Path, float, int], Mapping[str, Any]] | None = None,
    sample_source: str | None = None,
) -> None:
    loaded_checkpoint_path = tmp_path / "loaded_compact_checkpoint.pt"
    loaded_checkpoint_path.write_text("unit compact checkpoint\n", encoding="utf-8")
    context = module.CompactCandidateContext(
        model=object(),
        optimizer=object(),
        checkpoint=object(),
        surface={
            "policy": {
                "observation_shape": [4, 64, 64],
                "action_space_size": 3,
                "support_scale": 300,
            }
        },
        checkpoint_path=loaded_checkpoint_path,
        checkpoint_sha256=_file_sha256(loaded_checkpoint_path),
        checkpoint_id="unit-compact-ckpt",
    )

    def fake_load_context(**kwargs):
        del kwargs
        return context

    def fake_train(**kwargs):
        del kwargs
        result_sample_source = sample_source or module.COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE
        result = module.CompactCandidateTrainingResult(
            final_checkpoint=object(),
            initial_model_state_digest="a" * 64,
            final_model_state_digest="b" * 64,
            trainer_counters_before={
                "train_step": 1,
                "learner_update_count": 1,
                "sample_batch_count": 1,
                "policy_refresh_count": 0,
            },
            trainer_counters_after={
                "train_step": 2,
                "learner_update_count": 2,
                "sample_batch_count": 2,
                "policy_refresh_count": 1,
            },
            learner_telemetry={
                "compact_muzero_learner_calls_train_muzero": False,
                "compact_muzero_learner_loss": 0.125,
                "compact_muzero_learner_sample_rows": 2,
            },
            learner_update_count_delta=1,
            sample_batch_count_delta=1,
            compact_rollout_rows=2,
            compact_sample_rows=2,
            training_wall_sec=0.25,
            learner_loss_mean=0.125,
            sample_source=result_sample_source,
        )
        if training_mutator is not None:
            return training_mutator(module, result)
        return result

    def fake_save_export(**kwargs):
        path = Path(kwargs["path"])
        _write_json(path, {"kind": "stock_export", "name": path.name})
        return {"checkpoint_path": path, "sidecar_path": path.with_suffix(".json")}

    def fake_run_eval(**kwargs):
        args = kwargs["args"]
        checkpoint_path = Path(kwargs["checkpoint_path"])
        output_path = Path(kwargs["output_path"])
        step = 0 if checkpoint_path.name == "iteration_0.pth.tar" else 1
        mean = 10.0 if step == 0 else 13.0
        builder = eval_summary_builder or _eval_summary
        payload = dict(builder(checkpoint_path, mean, step))
        seeds = module._eval_seed_values(args=args, fallback_seed=int(kwargs["seed"]))
        payload["eval_seed_set"] = seeds
        payload["eval_max_steps"] = int(args.eval_steps)
        aggregate = payload.get("survival_aggregate_table")
        if isinstance(aggregate, list) and aggregate:
            aggregate[0]["seeds"] = len(seeds)
            aggregate[0]["eval_episode_count"] = len(seeds)
            aggregate[0]["ok_count"] = len(seeds)
        payload["survival_table"] = [
            {
                "checkpoint": f"iteration_{step}",
                "seed": seed,
                "steps": mean,
                "cap": int(args.eval_steps),
                "terminal": "loss",
                "ok": True,
            }
            for seed in seeds
        ]
        _write_json(output_path, payload)
        return payload

    monkeypatch.setattr(module, "_load_compact_candidate_context", fake_load_context)
    monkeypatch.setattr(module, "_train_compact_candidate_once", fake_train)
    monkeypatch.setattr(module, "_save_stock_checkpoint_export", fake_save_export)
    monkeypatch.setattr(module, "_run_eval_summary", fake_run_eval)


def _write_lifecycle(tmp_path: Path) -> Path:
    checkpoint_path = tmp_path / "source_compact_checkpoint.pt"
    checkpoint_path.write_text("source checkpoint\n", encoding="utf-8")
    lifecycle_path = tmp_path / "unified_lifecycle_report.json"
    _write_json(
        lifecycle_path,
        {
            "schema_id": "curvyzero_compact_unified_lifecycle_smoke/v1",
            "ok": True,
            "checkpoint_id": "unit-compact-ckpt",
            "compact_checkpoint_path": str(checkpoint_path),
            "lifecycle_gates_complete": True,
            "promotion_claim": False,
            "promotion_eligible": False,
        },
    )
    return lifecycle_path


def _eval_summary(checkpoint_path: Path, mean: float, step: int) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(step),
        "eval_seed_set": [20260833, 20260834],
        "eval_max_steps": 128,
        "seeds": 2,
        "mean_steps": float(mean),
        "median_steps": float(mean),
        "min_steps": float(mean) - 1.0,
        "max_steps": float(mean) + 1.0,
        "ok_count": 2,
        "capped_count": 0,
        "failure_count": 0,
        "outcome_histogram": {"loss": 2},
        "survival_aggregate_table": [
            {
                "checkpoint": f"iteration_{step}",
                "checkpoint_ref": str(checkpoint_path),
                "checkpoint_step": int(step),
                "seeds": 2,
                "eval_episode_count": 2,
                "mean_steps": float(mean),
                "median_steps": float(mean),
                "min_steps": float(mean) - 1.0,
                "max_steps": float(mean) + 1.0,
                "ok_count": 2,
                "capped_count": 0,
                "failure_count": 0,
                "outcome_histogram": {"loss": 2},
            }
        ],
        "survival_table": [
            {
                "checkpoint": f"iteration_{step}",
                "seed": seed,
                "steps": float(mean),
                "cap": 128,
                "terminal": "loss",
                "ok": True,
            }
            for seed in (20260833, 20260834)
        ],
    }


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
