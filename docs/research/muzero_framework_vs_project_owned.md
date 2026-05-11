# LightZero-First MuZero Adapter Plan

Research snapshot: 2026-05-09.

Current direction: LightZero-first for the next real dummy Pong MuZero attempt.
Do not start by writing a project-owned JAX/Mctx trainer. Keep Mctx as
fallback/comparison only if LightZero cannot run the custom env cleanly, loses
required metadata, or later blocks CurvyTron ownership.

## Short Answer

Build a `DummyPongLightZeroEnv` adapter and two Modal smokes:

0. Feature-fit audit gate: expose reset/step, observation shape, legal actions,
   reward/info telemetry, trainer entrypoint fit, checkpoint discovery, and
   independent CurvyZero scorecard path before training.
1. Config/import smoke: import LightZero and the adapter, patch a tiny MuZero
   config, create/reset/step the custom env, and capture the config without
   training.
2. Tiny train smoke: call LightZero's own MuZero trainer on the custom env,
   scan logs/checkpoints, and mirror CurvyZero scorecard telemetry.

Feature-fit command after implementation:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --mode feature-fit \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0
```

Config/import command after implementation:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0
```

Tiny train command after implementation:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0 \
  --opponent-policy random_uniform \
  --max-env-step 64 \
  --max-train-iter 2 \
  --num-simulations 2 \
  --batch-size 8 \
  --update-per-collect 1 \
  --n-evaluator-episode 1
```

The train command must call LightZero's `train_muzero` on our custom env. It is
not a dry config smoke, not stock Atari Pong, and not a project-owned trainer.

## Feature-Gap Audit

Run this checklist in order. Do not start the tiny trainer until the first five
items pass. Do not trust a trainer result unless telemetry, checkpoints, and
Modal artifacts pass too.

| Feature | Required feature | Early test | Stop condition |
| --- | --- | --- | --- |
| Custom env API | LightZero can load a CurvyZero-owned env through DI-engine `BaseEnv` registration or a Gym wrapper. `reset()` returns `{"observation", "action_mask", "to_play"}` and `step()` returns `BaseEnvTimestep`. | Config/import smoke imports `curvyzero.training.lightzero_dummy_pong_env`, registers `dummy_pong_lightzero`, creates one env, resets, steps one ego action, and returns JSON with shape, action count, reward, done, and `info.curvyzero_pong`. | Stop before training if env registration requires a LightZero fork, if reset/step cannot return the LightZero dict, or if the env silently falls back to stock CartPole/Pong. |
| Ego wrapper for simultaneous Pong/Curvy | LightZero controls one ego action; wrapper supplies opponent action and logs the joint action. | Fixed-seed direct `PongEnv` episode and wrapper episode produce matching terminal outcome for a scripted ego/opponent trace. | Stop if opponent actions are hidden, nondeterministic without seed records, or simultaneous actions only work through joint-action search. |
| Tabular input | First smoke uses `PongObservation` as a 10-float `model_type="mlp"` input. | Config/import smoke reports `feature_mode=tabular_ego`, `observation_shape=10`, `action_space_size=3`; one step returns finite `np.float32`. | Stop if the CartPole MLP config cannot be patched to `observation_shape=10` and `action_space_size=3` without model errors. |
| Visual/raster input | Raster is later but must look feasible: dummy Pong `9x15` grid can be flattened for MLP or promoted to channel-first image for conv. | After tabular passes, run config/import with `feature_mode=raster_flat` only; no trainer required. | Stop raster work if LightZero image models are tightly tied to Atari wrappers, frame stacks, or fixed 64/84 image assumptions. Continue tabular trainer instead. |
| Architecture flexibility | MuZero MLP config can change observation shape, action count, simulation count, batch size, and env type. | Patched config JSON shows only expected fields changed from CartPole; dry import validates model surface. | Stop if changing basic model/env fields requires broad LightZero source edits. |
| Reward/loss shaping | Environment reward stays honest `+1/-1/0`; shaped loss-delay return is telemetry or an explicit later training experiment, not a silent replacement. | Episode info contains both `score_return` and `shaped_loss_delay_return`; config says which reward goes to LightZero. | Stop if LightZero training forces reward normalization or rewriting that hides the real score reward from logs. |
| Scorecard telemetry | Every episode can emit wins/losses/timeouts, survival stats, score return, shaped return, action histograms, opponent id, seed, and trace hash. | Config/import smoke writes one `episodes.jsonl` row. Tiny train smoke writes aggregate `summary.json`. | Stop if collector/evaluator drops final `info` and no sidecar path can recover it. |
| Checkpoints | LightZero writes `.pth.tar` checkpoints and we can mirror/manifest them in `curvyzero-runs`. | Tiny train smoke finds at least one checkpoint, TensorBoard/log files, and writes `lightzero_artifacts_manifest.json`. | Stop trusting the run if no checkpoint appears or artifact paths cannot be copied/referenced from the Modal attempt. |
| Independent scorecard | CurvyZero can evaluate the learned LightZero checkpoint outside LightZero's own evaluator or clearly schedule that eval as the next command. | Feature-fit audit names the scorecard function/command. Tiny train smoke runs or schedules it after checkpoint discovery. | Stop before claiming learning evidence if only LightZero evaluator output exists and no CurvyZero scorecard path is available. |
| Modal image/runtime | Pinned LightZero image builds and runs within a small CPU Modal job. | Config/import smoke reports `LightZero`, `DI-engine`, `torch`, Python, and platform versions. Tiny train stays inside timeout. | Stop if dependency install or import drift requires unpinned broad packages, AutoROM, Atari ROM work, or GPU before dummy Pong proves the path. |
| Examples | Stock CartPole remains rerunnable; stock Atari Pong remains blocked by ROM; custom dummy Pong is the main example now. | Keep CartPole progression as sanity reference, but the first custom command must load `dummy_pong_lightzero`. | Stop using stock examples as CurvyZero evidence if the command did not load the custom dummy Pong env. |
| Gumbel MuZero | Useful after plain MuZero starts; not required for first trust. | Config-only switch to LightZero Gumbel policy/model path after plain MuZero checkpoint exists. | Stop if adding Gumbel changes the custom env wrapper or loses telemetry. Keep plain MuZero. |
| EfficientZero | Later sample-efficiency comparison, not first dummy Pong proof. | Config-only feasibility read after MuZero custom-env train passes. | Stop if EfficientZero pulls in segment/reanalyze complexity before custom MuZero works. |
| Stochastic MuZero | Later only for real chance events. | Docs/config read only; no implementation until deterministic custom-env MuZero has a measured stochastic failure. | Stop if stochastic support adds chance-node targets before deterministic Pong works. |

Plain recommendation: the first proof is not "can LightZero solve Pong." It is
"can LightZero run its real MuZero trainer on our custom env while keeping
enough CurvyZero metadata to judge the run."

Hard stop conditions for LightZero-first:

- Custom dummy Pong cannot be registered/created/reset/stepped without a
  LightZero or DI-engine fork.
- The adapter cannot preserve seed, joint action, opponent policy, terminal
  rewards, and trace hash.
- Final episode `info` or a sidecar cannot produce scorecard telemetry.
- The tiny trainer does not produce a checkpoint/log artifact under brutal
  caps.
- Modal dependency/runtime friction grows beyond the pinned LightZero image
  plus minimal env dependencies.
- The wrapper must model simultaneous play as joint-action search before
  ego-vs-random works.
- The integration code becomes larger than a tiny project-owned Mctx trainer
  would be.

## Why LightZero First

LightZero is the only complete MuZero trainer already proven in this repo's
Modal setup. Stock CartPole MuZero ran and produced learner/evaluator signals,
TensorBoard events, and `.pth.tar` checkpoints.

Using LightZero first lets us reuse:

- trainer entrypoint;
- collector and evaluator loop;
- replay and learner/update machinery;
- MCTS integration;
- PyTorch model/update path;
- trainer logs and checkpoints.

Mctx remains important, but it is search-only. A project-owned Mctx trainer
would still require CurvyZero to write env batching, replay, target building,
model code, optimizer updates, checkpointing, eval, and Modal run management.
Do not pay that implementation bill until LightZero has failed a small custom
env smoke for a concrete reason.

## Exact Adapter Interface

LightZero custom envs should use DI-engine `BaseEnv` style registration:

```python
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY

@ENV_REGISTRY.register("dummy_pong_lightzero")
class DummyPongLightZeroEnv(BaseEnv):
    def __init__(self, cfg=None): ...
    def reset(self): ...
    def step(self, action): ...

    @property
    def observation_space(self): ...

    @property
    def action_space(self): ...
```

`reset()` returns:

```python
{
    "observation": ego_obs,      # np.float32 shape (10,)
    "action_mask": action_mask,  # np.ones(3, dtype=np.int8)
    "to_play": -1,
}
```

`step(action)` returns:

```python
BaseEnvTimestep(
    {
        "observation": next_ego_obs,
        "action_mask": np.ones(3, dtype=np.int8),
        "to_play": -1,
    },
    reward,  # float ego reward
    done,    # bool terminated or truncated
    info,    # scorecard and trace metadata
)
```

Spaces:

```python
observation_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(10,),
    dtype=np.float32,
)
action_space = gym.spaces.Discrete(3)
```

Use `to_play=-1`. Do not use LightZero board-game self-play mode for the first
dummy Pong run.

## Dummy Pong Wrapper Shape

LightZero controls one ego player. Dummy Pong still steps both players.

Wrapper state:

- `env: PongEnv`;
- `ego_agent: "player_0"`;
- `opponent_agent: "player_1"`;
- `opponent_policy_id: "random_uniform"` first, then `lagged_track_ball_1`;
- `episode_seed`, `episode_index`, `step_index`;
- `last_observations`;
- `episode_return`;
- `action_trace`.

Step policy:

```python
joint_action = {
    "player_0": int(lightzero_action),
    "player_1": opponent_policy_action,
}
```

Reward is `float(step.rewards["player_0"])`. Done is
`step.terminated or step.truncated`. The wrapper must preserve enough `info` to
replay or diagnose the episode outside LightZero.

## First Observation

Use tabular ego features first. This is closest to the proven CartPole MLP
shape and avoids making visual modeling the first integration blocker.

```text
ego_paddle_y / height
opponent_paddle_y / height
ego_paddle_x / width
opponent_paddle_x / width
ball_dx_forward / width
ball_dy_from_ego_center / height
ball_vx_forward
ball_vy
ball_y / height
step / max_steps
```

Use `np.float32`, shape `(10,)`, and schema
`dummy_pong_lightzero_tabular_v0`. Keep raster as logged/debug data at first.
Try raster/conv only after the LightZero trainer path works.

## LightZero Config Shape

Start from the stock CartPole MuZero config and patch it:

```python
main_config.exp_name = "/tmp/curvyzero-lightzero-dummy-pong/seed-0"
main_config.env.env_id = "DummyPongLightZero-v0"
main_config.env.stop_value = 1.0
main_config.env.continuous = False
main_config.env.manually_discretization = False
main_config.env.collector_env_num = 1
main_config.env.evaluator_env_num = 1
main_config.env.n_evaluator_episode = 1
main_config.env.manager.shared_memory = False

main_config.policy.cuda = False
main_config.policy.env_type = "not_board_games"
main_config.policy.model.model_type = "mlp"
main_config.policy.model.observation_shape = 10
main_config.policy.model.action_space_size = 3
main_config.policy.collector_env_num = 1
main_config.policy.evaluator_env_num = 1
main_config.policy.n_episode = 1
main_config.policy.num_simulations = 2
main_config.policy.batch_size = 8
main_config.policy.update_per_collect = 1
main_config.policy.eval_freq = 1
main_config.policy.replay_buffer_size = 256
```

Create config:

```python
create_config = EasyDict(
    env=dict(
        type="dummy_pong_lightzero",
        import_names=["curvyzero.training.lightzero_dummy_pong_env"],
    ),
    env_manager=dict(type="subprocess"),
    policy=dict(type="muzero", import_names=["lzero.policy.muzero"]),
)
```

Call:

```python
train_muzero(
    [main_config, create_config],
    seed=seed,
    model_path=None,
    max_env_step=max_env_step,
)
```

Use the same stdout/stderr capture, timeout, and artifact scan pattern as
`src/curvyzero/infra/modal/lightzero_cartpole_tiny_train_smoke.py`.

## Modal Image

Use the same pinned image family as the working CartPole smoke:

```python
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "LightZero==0.2.0",
)
```

Do not add AutoROM or Atari ROM handling. This lane is custom dummy Pong, not
stock Atari Pong.

## Preserve Scorecard Telemetry

LightZero's native logs are not enough. The wrapper and Modal function must
mirror CurvyZero scorecard fields.

Every final `info` should include:

```python
info["eval_episode_return"] = episode_return
info["curvyzero_pong"] = {
    "schema": "curvyzero_lightzero_dummy_pong_episode/v1",
    "episode_seed": episode_seed,
    "ego_agent": "player_0",
    "opponent_agent": "player_1",
    "opponent_policy_id": opponent_policy_id,
    "steps": steps,
    "winner": winner,
    "truncated": truncated,
    "terminated": terminated,
    "score_return": episode_return,
    "shaped_loss_delay_return": shaped_loss_delay_return,
    "survival_fraction": steps / max_steps,
    "action_counts_by_agent": action_counts_by_agent,
    "last_hit": last_hit,
    "final_rewards": final_rewards,
    "trace_hash": trace_hash,
}
```

Minimum summary fields:

- wins/losses/timeouts;
- mean, median, and p90 survival steps;
- truncation rate;
- score return mean/std;
- shaped loss-delay return mean/std;
- action histogram for ego and opponent;
- opponent policy id;
- seed range;
- LightZero checkpoint paths;
- LightZero log and TensorBoard event paths;
- whether this command actually called `train_muzero`.

## Modal Artifacts

Write LightZero's own files under `/tmp`, then mirror useful summaries into
the CurvyZero Volume layout:

```text
/runs/training/lightzero-dummy-pong/<run_id>/
  run.json
  latest_attempt.json
  attempts/<attempt_id>/
    attempt.json
    config.json
    command.json
    stdout_tail.txt
    stderr_tail.txt
    train/
      summary.json
      episodes.jsonl
      lightzero_artifacts_manifest.json
      lightzero_training_signals.json
  checkpoints/
    lightzero/
      ckpt_best.pth.tar
      iteration_*.pth.tar
      manifest.json
```

Keep LightZero checkpoints in LightZero format for the first smoke. The run
label must say `LightZero custom-env MuZero`, not `project-owned MuZero/Mctx`.

## First Implementation Slice

1. Add `src/curvyzero/training/lightzero_dummy_pong_env.py`.
2. Add
   `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`.
3. Add `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`.
4. Reuse `LightZero==0.2.0`.
5. Reuse `run_management.py` for run ids, attempt manifests, Volume commits,
   and latest pointer.
6. Start with the feature-fit audit gate.
7. Then run config/import only.
8. Train with `opponent_policy=random_uniform`.
9. Then try `opponent_policy=lagged_track_ball_1`.
10. Only then try larger caps or raster/conv observations.

## Pass/Fail

Feature-fit audit passes if reset/step works, observation shape/schema are
explicit, legal actions are represented or deliberately unnecessary,
reward/info telemetry is visible, the trainer entrypoint can target the custom
env, checkpoint patterns are known, and an independent CurvyZero scorecard path
exists.

Config/import smoke passes if LightZero/DI-engine can import, see the custom
dummy Pong env, create/reset/step it, preserve `A=3`, observation shape, and
legal-action surface, and capture the patched MuZero config without calling the
trainer.

Tiny train smoke passes if LightZero's MuZero trainer returns `ok: true`, emits
learner/evaluator signals and at least one checkpoint, stays inside caps,
discovers the checkpoint path, preserves enough metadata to map back to
CurvyZero seeds/actions/rewards, reports dummy Pong score plus survival
telemetry, and runs or schedules an independent CurvyZero scorecard.

Fallback to project-owned JAX/Mctx only if LightZero cannot call a real
trainer, loses required metadata, or takes more code than the owned smoke would
likely take. Record the exact blocker first.

## Sources

- Local: `docs/decisions/0005-main-pong-repository-library-choice.md`,
  `docs/research/muzero_reference_examples.md`,
  `docs/research/muzero_repo_baseline_options.md`,
  `docs/design/muzero_modal_architecture.md`, and
  `src/curvyzero/infra/modal/lightzero_cartpole_tiny_train_smoke.py`.
- LightZero custom environment docs:
  https://opendilab.github.io/LightZero/tutorials/envs/customize_envs.html
- LightZero CartPole MuZero config:
  https://github.com/opendilab/LightZero/blob/main/zoo/classic_control/cartpole/config/cartpole_muzero_config.py
- Mctx PyPI docs, fallback search dependency:
  https://pypi.org/project/mctx/
