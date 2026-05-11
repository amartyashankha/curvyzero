"""Run a tiny optional-Torch PPO learner smoke over the repo-native rollout shape.

This is an on-policy plumbing smoke, not policy-quality evidence. It preserves
the existing dry-run actor artifact shape and profile fields, then adds the
smallest PPO pieces that need Torch: collect with the same tiny policy/value
model that is updated, masked policy/value forward, GAE/returns, one clipped PPO
update, metrics, and a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ModuleNotFoundError:  # pragma: no cover - depends on local environment.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Categorical = None  # type: ignore[assignment]

from repo_native_ppo_actor_loop_dry_run import ACTION_COUNT
from repo_native_ppo_actor_loop_dry_run import CurvyTronConfig
from repo_native_ppo_actor_loop_dry_run import CurvyTronEnv
from repo_native_ppo_actor_loop_dry_run import LIGHTZERO_FLAT_OBSERVATION_SHAPE
from repo_native_ppo_actor_loop_dry_run import PLAYER_COUNT
from repo_native_ppo_actor_loop_dry_run import _build_report
from repo_native_ppo_actor_loop_dry_run import _canonical_timing
from repo_native_ppo_actor_loop_dry_run import build_policy_row_mapping
from repo_native_ppo_actor_loop_dry_run import observe_1v1_egocentric_rays_v0
from repo_native_ppo_actor_loop_dry_run import policy_rows_to_joint_action


SCHEMA_ID = "curvyzero_repo_native_ppo_learner_smoke/v0"


if torch is not None:

    class TinyActorCritic(nn.Module):
        """Small shared ego-row actor/value model for smoke-scale PPO plumbing."""

        def __init__(self, obs_dim: int, hidden_dim: int, action_count: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.policy = nn.Linear(hidden_dim, action_count)
            self.value = nn.Linear(hidden_dim, 1)

        def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.body(observation)
            return self.policy(hidden), self.value(hidden).squeeze(-1)

else:

    class TinyActorCritic:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("Torch is required for TinyActorCritic")


def run_learner_smoke(
    *,
    batch_size: int,
    rollout_steps: int,
    seed: int,
    artifact_root: Path,
    hidden_dim: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    update_epochs: int,
) -> dict[str, Any]:
    if torch is None:
        return _write_skip_report(
            artifact_root=artifact_root,
            batch_size=batch_size,
            rollout_steps=rollout_steps,
            seed=seed,
            reason="torch is not importable; pyproject does not declare it as a dependency",
        )

    profile_started = time.perf_counter()
    artifact_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    obs_dim = int(LIGHTZERO_FLAT_OBSERVATION_SHAPE[0])
    model = TinyActorCritic(obs_dim=obs_dim, hidden_dim=hidden_dim, action_count=ACTION_COUNT).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    actor_report = _run_policy_sampled_actor_loop(
        batch_size=batch_size,
        rollout_steps=rollout_steps,
        seed=seed,
        artifact_root=artifact_root,
        model=model,
        device=device,
        write_rollout_npz=True,
    )
    rollout_path = Path(actor_report["artifacts"]["rollout_npz"])
    rollout = np.load(rollout_path)

    observation_np = rollout["observation"].astype(np.float32, copy=False)
    legal_action_mask_np = rollout["legal_action_mask"].astype(np.bool_, copy=False)
    live_mask_np = rollout["live_mask"].astype(np.bool_, copy=False)
    action_np = rollout["action"].astype(np.int64, copy=False)
    old_logprob_np = rollout["action_logprob"].astype(np.float32, copy=False)
    value_np = rollout["value"].astype(np.float32, copy=False)
    reward_np = rollout["reward"].astype(np.float32, copy=False)
    done_np = rollout["done"].astype(np.bool_, copy=False)

    _validate_rollout_shape(
        observation_np=observation_np,
        legal_action_mask_np=legal_action_mask_np,
        live_mask_np=live_mask_np,
        action_np=action_np,
        old_logprob_np=old_logprob_np,
        value_np=value_np,
        reward_np=reward_np,
        done_np=done_np,
        expected_batch_size=batch_size,
        expected_rollout_steps=rollout_steps,
    )

    observation = torch.as_tensor(observation_np, dtype=torch.float32, device=device)
    legal_action_mask = torch.as_tensor(legal_action_mask_np, dtype=torch.bool, device=device)
    live_mask = torch.as_tensor(live_mask_np, dtype=torch.bool, device=device)
    action = torch.as_tensor(action_np, dtype=torch.long, device=device)
    old_logprob = torch.as_tensor(old_logprob_np, dtype=torch.float32, device=device)
    behavior_value = torch.as_tensor(value_np, dtype=torch.float32, device=device)
    reward = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
    done = torch.as_tensor(done_np, dtype=torch.bool, device=device)
    active_mask = live_mask & legal_action_mask.any(dim=-1)

    timers = {
        "target_construction_sec": 0.0,
        "learner_update_sec": 0.0,
        "checkpoint_publish_sec": 0.0,
    }

    started = time.perf_counter()
    with torch.no_grad():
        values = torch.where(active_mask, behavior_value, torch.zeros_like(behavior_value))
        advantages, returns = _compute_gae(
            reward=reward,
            done=done,
            values=values,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
    timers["target_construction_sec"] += time.perf_counter() - started

    started = time.perf_counter()
    model.train()
    flat_active = active_mask.reshape(-1)
    flat_observation = observation.reshape(-1, obs_dim)[flat_active]
    flat_legal_action_mask = legal_action_mask.reshape(-1, ACTION_COUNT)[flat_active]
    flat_action = action.reshape(-1)[flat_active]
    flat_old_logprob = old_logprob.reshape(-1)[flat_active]
    flat_advantage = advantages.reshape(-1)[flat_active]
    flat_return = returns.reshape(-1)[flat_active]

    if flat_observation.numel() == 0:
        raise RuntimeError("learner smoke collected zero active policy rows")

    advantage_std = flat_advantage.std(unbiased=False)
    flat_advantage = (flat_advantage - flat_advantage.mean()) / advantage_std.clamp_min(1e-8)

    last_metrics: dict[str, float] = {}
    for epoch in range(update_epochs):
        logits, new_value = model(flat_observation)
        masked_logits = logits.masked_fill(~flat_legal_action_mask, -1.0e9)
        distribution = Categorical(logits=masked_logits)
        new_logprob = distribution.log_prob(flat_action)
        entropy = distribution.entropy()

        log_ratio = new_logprob - flat_old_logprob
        ratio = log_ratio.exp()
        unclipped_pg = -flat_advantage * ratio
        clipped_pg = -flat_advantage * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        policy_loss = torch.max(unclipped_pg, clipped_pg).mean()
        value_loss = 0.5 * (new_value - flat_return).pow(2).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5).item())
        optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
            value_var = torch.var(flat_return, unbiased=False)
            explained_variance = (
                1.0
                - torch.var(flat_return - new_value.detach(), unbiased=False) / value_var
                if float(value_var.item()) > 0.0
                else torch.tensor(0.0, device=device)
            )
            last_metrics = {
                "epoch": float(epoch),
                "loss": float(loss.detach().item()),
                "policy_loss": float(policy_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "entropy": float(entropy_loss.detach().item()),
                "approx_kl": float(approx_kl.detach().item()),
                "clip_fraction": float(clip_fraction.detach().item()),
                "grad_norm": grad_norm,
                "explained_variance": float(explained_variance.detach().item()),
            }
    timers["learner_update_sec"] += time.perf_counter() - started

    started = time.perf_counter()
    checkpoint_path = artifact_root / "checkpoint_step_000001.pt"
    torch.save(
        {
            "schema_id": SCHEMA_ID,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "obs_dim": obs_dim,
                "hidden_dim": hidden_dim,
                "action_count": ACTION_COUNT,
            },
        },
        checkpoint_path,
    )
    timers["checkpoint_publish_sec"] += time.perf_counter() - started

    masked_action_violation_count = int(
        (~legal_action_mask.reshape(-1, ACTION_COUNT)[flat_active].gather(
            1,
            flat_action.reshape(-1, 1),
        )).sum()
    )
    metrics = {
        "schema_id": SCHEMA_ID,
        "status": "no_quality_learner_smoke",
        "collection_policy_kind": "tiny_actor_critic_sampled",
        "active_policy_rows": int(flat_active.sum().item()),
        "masked_action_violation_count": masked_action_violation_count,
        "behavior_value_abs_mean": float(behavior_value[active_mask].abs().mean().item()),
        "update_epochs": int(update_epochs),
        **last_metrics,
    }
    metrics_path = artifact_root / "ppo_metrics.jsonl"
    metrics_path.write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "schema_id": SCHEMA_ID,
        "status": "no_quality_learner_smoke",
        "caveats": [
            "not a policy-quality result",
            "not a LightZero replacement",
            "not source-fidelity evidence",
            "single tiny on-policy collection/update only",
        ],
        "run": {
            "batch_size": batch_size,
            "player_count": PLAYER_COUNT,
            "rollout_steps": rollout_steps,
            "seed": seed,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "learner_config": {
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_coef": clip_coef,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
            "update_epochs": update_epochs,
            "collection_policy_kind": "tiny_actor_critic_sampled",
        },
        "rollout_field_shapes": {
            "observation": list(observation_np.shape),
            "legal_action_mask": list(legal_action_mask_np.shape),
            "live_mask": list(live_mask_np.shape),
            "action": list(action_np.shape),
            "action_logprob": list(old_logprob_np.shape),
            "value": list(value_np.shape),
            "reward": list(reward_np.shape),
            "done": list(done_np.shape),
            "advantage": list(advantages.shape),
            "return": list(returns.shape),
        },
        "collection_report": {
            "schema_id": actor_report["schema_id"],
            "rollout_schema_id": actor_report["replay_or_rollout"]["schema_id"],
            "status": actor_report["status"],
            "throughput": actor_report["throughput"],
            "latency_sec": actor_report["latency_sec"],
            "timing_sec": actor_report["timing_sec"],
            "policy_search": actor_report["policy_search"],
            "artifacts": actor_report["artifacts"],
            "field_specs": actor_report["replay_or_rollout"]["field_specs"],
        },
        "timing_sec": {
            **actor_report["timing_sec"],
            **timers,
        },
        "metrics": metrics,
        "artifacts": {
            "artifact_root": str(artifact_root),
            "rollout_npz": str(rollout_path),
            "collection_report_json": actor_report["artifacts"]["report_json"],
            "ppo_metrics_jsonl": str(metrics_path),
            "checkpoint": str(checkpoint_path),
            "report_json": str(artifact_root / "learner_report.json"),
        },
    }
    report_path = artifact_root / "learner_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _run_policy_sampled_actor_loop(
    *,
    batch_size: int,
    rollout_steps: int,
    seed: int,
    artifact_root: Path,
    model: TinyActorCritic,
    device: torch.device,
    write_rollout_npz: bool = True,
) -> dict[str, Any]:
    """Collect a tiny rollout with the same model instance the learner updates."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")

    artifact_root.mkdir(parents=True, exist_ok=True)
    config = CurvyTronConfig()
    envs = [CurvyTronEnv(config) for _ in range(batch_size)]
    row_seeds = np.asarray([seed + row for row in range(batch_size)], dtype=np.int64)
    episode_ordinals = np.zeros(batch_size, dtype=np.int64)

    timers = {
        "reset_autoreset_sec": 0.0,
        "observation_packing_sec": 0.0,
        "row_compaction_sec": 0.0,
        "policy_forward_sec": 0.0,
        "action_scatter_sec": 0.0,
        "env_step_sec": 0.0,
        "rollout_staging_sec": 0.0,
        "artifact_write_sec": 0.0,
    }
    action_latency_samples: list[float] = []
    reset_counts = np.zeros(batch_size, dtype=np.int64)

    def reset_row(row: int) -> None:
        envs[row].reset(seed=int(row_seeds[row] + episode_ordinals[row] * 1_000_003))
        reset_counts[row] += 1

    started = time.perf_counter()
    for row in range(batch_size):
        reset_row(row)
    timers["reset_autoreset_sec"] += time.perf_counter() - started

    obs_dim = int(LIGHTZERO_FLAT_OBSERVATION_SHAPE[0])
    observation = np.zeros((rollout_steps, batch_size, PLAYER_COUNT, obs_dim), dtype=np.float32)
    legal_action_mask = np.zeros(
        (rollout_steps, batch_size, PLAYER_COUNT, ACTION_COUNT),
        dtype=np.bool_,
    )
    live_mask = np.zeros((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.bool_)
    action = np.ones((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.int16)
    action_logprob = np.zeros((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.float32)
    action_probs = np.zeros(
        (rollout_steps, batch_size, PLAYER_COUNT, ACTION_COUNT),
        dtype=np.float32,
    )
    value = np.zeros((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.float32)
    reward = np.zeros((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.float32)
    done = np.zeros((rollout_steps, batch_size), dtype=np.bool_)
    terminated = np.zeros((rollout_steps, batch_size), dtype=np.bool_)
    truncated = np.zeros((rollout_steps, batch_size), dtype=np.bool_)
    final_observation = np.zeros(
        (rollout_steps, batch_size, PLAYER_COUNT, obs_dim),
        dtype=np.float32,
    )
    final_reward_map = np.zeros((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.float32)
    reset_seed = np.zeros((rollout_steps, batch_size), dtype=np.int64)
    episode_id = np.empty((rollout_steps, batch_size), dtype=object)

    needs_reset = np.zeros(batch_size, dtype=np.bool_)
    completed_games = 0
    timeout_count = 0
    active_policy_rows = 0

    loop_started = time.perf_counter()
    model.eval()
    for step_index in range(rollout_steps):
        started = time.perf_counter()
        for row in np.flatnonzero(needs_reset):
            episode_ordinals[row] += 1
            reset_row(int(row))
            needs_reset[row] = False
        timers["reset_autoreset_sec"] += time.perf_counter() - started

        obs_started = time.perf_counter()
        for row, env in enumerate(envs):
            assert env.state is not None
            batch = observe_1v1_egocentric_rays_v0(
                env.state,
                env.config,
                player_ids=env.agents,
                needs_reset=False,
            )
            observation[step_index, row] = batch.observation
            legal_action_mask[step_index, row] = batch.action_mask
            live_mask[step_index, row] = batch.action_mask.any(axis=1)
            reset_seed[step_index, row] = int(row_seeds[row] + episode_ordinals[row] * 1_000_003)
            episode_id[step_index, row] = str(
                (env.last_reset_info or {}).get("episode_id", "unknown")
            )
        timers["observation_packing_sec"] += time.perf_counter() - obs_started

        action_started = time.perf_counter()

        started = time.perf_counter()
        mapping = build_policy_row_mapping(
            observation[step_index],
            live_mask[step_index],
            legal_action_mask[step_index],
            pad_to=batch_size * PLAYER_COUNT,
        )
        active_policy_rows += mapping.active_count
        timers["row_compaction_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        selected, probs, logprobs, values = _sample_tiny_actor_critic_policy(
            model=model,
            observations=mapping.observations,
            legal_action_mask=mapping.legal_action_mask,
            row_mask=mapping.row_mask,
            device=device,
        )
        timers["policy_forward_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        joint_action = policy_rows_to_joint_action(mapping, selected, dtype=np.int16)
        timers["action_scatter_sec"] += time.perf_counter() - started

        action_latency_samples.append(time.perf_counter() - action_started)

        started = time.perf_counter()
        if mapping.active_count:
            active_rows = np.asarray(mapping.row_mask, dtype=np.bool_)
            env_ids = mapping.env_row_id[active_rows]
            player_ids = mapping.player_id[active_rows]
            action[step_index, env_ids, player_ids] = selected[active_rows].astype(np.int16)
            action_logprob[step_index, env_ids, player_ids] = logprobs[active_rows]
            action_probs[step_index, env_ids, player_ids] = probs[active_rows]
            value[step_index, env_ids, player_ids] = values[active_rows]
        timers["rollout_staging_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        for row, env in enumerate(envs):
            assert env.state is not None
            actions = {
                agent: int(joint_action[row, player_index])
                for player_index, agent in enumerate(env.agents)
                if bool(env.state.alive[player_index])
            }
            result = env.step(actions)
            row_done = any(result.terminated.values()) or any(result.truncated.values())
            needs_reset[row] = row_done
        timers["env_step_sec"] += time.perf_counter() - started

        obs_started = time.perf_counter()
        for row, env in enumerate(envs):
            assert env.state is not None
            post = observe_1v1_egocentric_rays_v0(
                env.state,
                env.config,
                player_ids=env.agents,
                needs_reset=bool(needs_reset[row]),
            )
            reward[step_index, row] = post.rewards
            done[step_index, row] = post.done
            terminated[step_index, row] = post.terminated
            truncated[step_index, row] = post.truncated
            if post.done:
                completed_games += 1
                if post.truncated:
                    timeout_count += 1
                final_observation[step_index, row] = post.observation
                final_reward_map[step_index, row] = post.rewards
        timers["observation_packing_sec"] += time.perf_counter() - obs_started

    elapsed_sec = time.perf_counter() - loop_started
    timers["loop_elapsed_sec"] = elapsed_sec

    write_started = time.perf_counter()
    rollout_path = None
    if write_rollout_npz:
        rollout_path = artifact_root / "rollout_buffer.npz"
        np.savez(
            rollout_path,
            observation=observation,
            legal_action_mask=legal_action_mask,
            live_mask=live_mask,
            action=action,
            action_logprob=action_logprob,
            action_probs=action_probs,
            value=value,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            final_observation=final_observation,
            final_reward_map=final_reward_map,
            reset_seed=reset_seed,
            episode_id=episode_id.astype(str),
        )

    report = _build_report(
        batch_size=batch_size,
        rollout_steps=rollout_steps,
        seed=seed,
        artifact_root=artifact_root,
        rollout_path=rollout_path,
        config=config,
        timers=timers,
        elapsed_sec=elapsed_sec,
        action_latency_samples=action_latency_samples,
        completed_games=completed_games,
        timeout_count=timeout_count,
        active_policy_rows=active_policy_rows,
        reset_counts=reset_counts,
        arrays={
            "observation": observation,
            "legal_action_mask": legal_action_mask,
            "live_mask": live_mask,
            "action": action,
            "action_logprob": action_logprob,
            "action_probs": action_probs,
            "value": value,
            "reward": reward,
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
            "final_observation": final_observation,
            "final_reward_map": final_reward_map,
            "reset_seed": reset_seed,
        },
    )
    report["status"] = "ok"
    report["caveats"] = [
        "not a PPO quality result",
        "not a source-fidelity claim",
        "not vector-runtime throughput evidence",
        "single tiny policy-sampled on-policy rollout only",
        "toy curvyzero-v0 scalar envs behind a [B,P] actor-loop shape",
    ]
    report["policy_search"] = {
        **report["policy_search"],
        "policy_kind": "tiny_actor_critic_sampled",
        "opponent_checkpoint_policy": "shared tiny actor-critic controls both players",
        "action_mask_contract_status": "applied to Torch logits before sampling",
    }
    report_path = artifact_root / "report.json"
    report["artifacts"]["report_json"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    timers["artifact_write_sec"] += time.perf_counter() - write_started
    total_elapsed_sec = time.perf_counter() - profile_started
    report["timing_sec"] = _canonical_timing(timers, elapsed_sec=total_elapsed_sec)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _sample_tiny_actor_critic_policy(
    *,
    model: TinyActorCritic,
    observations: np.ndarray,
    legal_action_mask: np.ndarray,
    row_mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = np.ones(row_mask.shape[0], dtype=np.int64)
    probs = np.zeros((row_mask.shape[0], ACTION_COUNT), dtype=np.float32)
    logprobs = np.zeros(row_mask.shape[0], dtype=np.float32)
    values = np.zeros(row_mask.shape[0], dtype=np.float32)

    active_rows = np.asarray(row_mask, dtype=np.bool_)
    if not active_rows.any():
        return selected, probs, logprobs, values

    with torch.no_grad():
        obs = torch.as_tensor(observations[active_rows], dtype=torch.float32, device=device)
        legal = torch.as_tensor(legal_action_mask[active_rows], dtype=torch.bool, device=device)
        logits, active_values = model(obs)
        masked_logits = logits.masked_fill(~legal, -1.0e9)
        distribution = Categorical(logits=masked_logits)
        active_selected = distribution.sample()
        active_logprobs = distribution.log_prob(active_selected)

    selected[active_rows] = active_selected.cpu().numpy().astype(np.int64, copy=False)
    probs[active_rows] = distribution.probs.cpu().numpy().astype(np.float32, copy=False)
    logprobs[active_rows] = active_logprobs.cpu().numpy().astype(np.float32, copy=False)
    values[active_rows] = active_values.cpu().numpy().astype(np.float32, copy=False)
    return selected, probs, logprobs, values


def _validate_rollout_shape(
    *,
    observation_np: np.ndarray,
    legal_action_mask_np: np.ndarray,
    live_mask_np: np.ndarray,
    action_np: np.ndarray,
    old_logprob_np: np.ndarray,
    value_np: np.ndarray,
    reward_np: np.ndarray,
    done_np: np.ndarray,
    expected_batch_size: int,
    expected_rollout_steps: int,
) -> None:
    expected_bp = (expected_rollout_steps, expected_batch_size, PLAYER_COUNT)
    if observation_np.shape[:3] != expected_bp:
        raise ValueError(f"observation must have leading [T,B,P]={expected_bp!r}")
    if legal_action_mask_np.shape != (*expected_bp, ACTION_COUNT):
        raise ValueError("legal_action_mask must have shape [T,B,P,A]")
    for name, array in {
        "live_mask": live_mask_np,
        "action": action_np,
        "action_logprob": old_logprob_np,
        "value": value_np,
        "reward": reward_np,
    }.items():
        if array.shape != expected_bp:
            raise ValueError(f"{name} must have shape [T,B,P]")
    if done_np.shape != (expected_rollout_steps, expected_batch_size):
        raise ValueError("done must have shape [T,B]")


def _compute_gae(
    *,
    reward: torch.Tensor,
    done: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(reward)
    last_gae = torch.zeros_like(reward[0])
    next_value = torch.zeros_like(reward[0])
    for step in reversed(range(reward.shape[0])):
        next_nonterminal = (~done[step]).float()[:, None]
        delta = reward[step] + gamma * next_value * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
        next_value = values[step]
    returns = advantages + values
    return advantages, returns


def _write_skip_report(
    *,
    artifact_root: Path,
    batch_size: int,
    rollout_steps: int,
    seed: int,
    reason: str,
) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_id": SCHEMA_ID,
        "status": "skipped",
        "reason": reason,
        "run": {
            "batch_size": batch_size,
            "player_count": PLAYER_COUNT,
            "rollout_steps": rollout_steps,
            "seed": seed,
        },
        "artifacts": {
            "artifact_root": str(artifact_root),
            "report_json": str(artifact_root / "learner_report.json"),
        },
    }
    Path(report["artifacts"]["report_json"]).write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--rollout-steps", type=_positive_int, default=8)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/private/tmp/curvy-repo-native-ppo-learner-smoke"),
    )
    parser.add_argument("--hidden-dim", type=_positive_int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--update-epochs", type=_positive_int, default=1)
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_learner_smoke(
        batch_size=int(args.batch_size),
        rollout_steps=int(args.rollout_steps),
        seed=int(args.seed),
        artifact_root=args.artifact_root,
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_coef=float(args.clip_coef),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        update_epochs=int(args.update_epochs),
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    artifacts = report["artifacts"]
    if report["status"] == "skipped":
        print(f"repo_native_ppo_learner_smoke skipped reason={report['reason']}")
        return
    metrics = report["metrics"]
    print(
        "repo_native_ppo_learner_smoke "
        f"B={report['run']['batch_size']} T={report['run']['rollout_steps']} "
        f"active_policy_rows={metrics['active_policy_rows']} "
        f"loss={metrics['loss']:.6f} "
        f"entropy={metrics['entropy']:.6f} "
        f"masked_action_violations={metrics['masked_action_violation_count']} "
        f"artifact_root={artifacts['artifact_root']}"
    )


if __name__ == "__main__":
    main()
