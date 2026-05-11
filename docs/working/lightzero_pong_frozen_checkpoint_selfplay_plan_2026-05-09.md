# LightZero Pong Frozen-Checkpoint Self-Play Plan - 2026-05-09

Role: implementation scout only. No pytest and no code changes in this pass.

## Short Answer

The smallest honest next step is learner vs frozen older checkpoint.

Keep the learner exactly where it is: LightZero controls one ego paddle through
`DummyPongLightZeroEnv`. Replace the scripted opponent with a frozen LightZero
checkpoint policy loaded once when the env is constructed. Do not call Modal
from env steps. Do not create a second learner. Do not move to full
multi-agent search yet.

Plain name:

```text
LightZero learner vs frozen LightZero checkpoint opponent
```

This is a staged frozen-opponent step, not full simultaneous multiplayer
self-play. It is still the right bridge after a trusted LightZero run.

MuZero/LightZero data comes from repeated environment interaction. Scale data
with more actors, episodes, and steps only after checkpoint curves are honest.
Every run must report wins, survival steps, shaped score, actions, seeds, and
artifact refs.

## Current Code Shape

`DummyPongLightZeroEnv` is a single-ego wrapper around simultaneous dummy Pong.
LightZero passes one action. The env builds the joint action by asking an
opponent policy for the other paddle action.

Today `_make_opponent_policy()` supports only:

- `random_uniform`
- `track_ball`
- `lagged_track_ball_1`

The independent eval path already has useful checkpoint policy adapters:

- direct policy-head greedy:
  `load_lightzero_policy_head_greedy_checkpoint()`
- eval-mode MCTS:
  `load_lightzero_mcts_eval_mode_checkpoint()`

Those adapters live in `src/curvyzero/training/lightzero_dummy_pong_policy.py`
and already implement the common policy protocol:

```text
reset(episode_seed, agent)
action(observation, raster_grid, agent) -> action_id
```

That is exactly the protocol `DummyPongLightZeroEnv` needs for an opponent.

## Env Support

Add a checkpoint-backed opponent branch to `DummyPongLightZeroEnv`.

Recommended config fields:

```python
{
    "opponent_policy": "lightzero_mcts_checkpoint",
    "opponent_checkpoint_path": "/runs/training/lightzero-dummy-pong/.../iteration_16.pth.tar",
    "opponent_checkpoint_label": "post_seed_fix_iter16",
    "opponent_checkpoint_adapter": "mcts_eval_mode",
    "opponent_checkpoint_num_simulations": 2,
    "opponent_checkpoint_state_key": "model",
}
```

Keep `opponent_policy` as the selector. Use separate keys for the path and
adapter settings so the env config stays readable in LightZero summaries.

Implementation outline:

1. In `__init__`, read the checkpoint fields and keep them on `self`.
2. In `_make_opponent_policy()`, add:
   - `lightzero_policy_head_checkpoint`
   - `lightzero_mcts_checkpoint`
3. Load the frozen policy once in `_make_opponent_policy()`.
4. On every episode reset, call the policy's `reset()`.
5. On every step, call the frozen policy's `action()` exactly like scripted
   policies.

The env should not resolve Volume refs by itself. It should receive a real file
path that exists inside the container.

## Opponent Inference Choice

### Direct policy-head greedy

Simplicity: best.

Cost: cheapest. One `initial_inference()` call per env step.

Correctness: weakest. It bypasses `MuZeroPolicy.eval_mode.forward`, does not use
MCTS, and has known argmax-collapse risk when logits are tied or weak. It is
good for loader smoke and cheap debugging, but it is not the main training
opponent if we want parity with scorecards.

Use it for:

- dry env support smoke;
- fast action-collapse diagnosis;
- maybe a tiny first run if MCTS overhead blocks collection.

Do not use it as the main claim.

### MCTS eval-mode

Simplicity: medium. The adapter already exists, but it constructs a full
`MuZeroPolicy` and calls `eval_mode.forward()`.

Cost: higher. It runs MCTS inside every opponent action. With one collector env
and low `num_simulations`, this is still acceptable for the first bounded run.

Correctness: best available. It matches the independent MCTS scorecard boundary
and uses LightZero's eval-mode policy path.

Use this as the recommended first real frozen-checkpoint opponent:

```text
opponent_checkpoint_adapter = "mcts_eval_mode"
opponent_checkpoint_num_simulations = 2 or 4
```

Keep simulations lower than the learner's train/eval scorecard if needed. This
run is about plumbing and direction, not strength.

### Cheap exported policy

Simplicity: worst for the next step because it requires a new exporter and a
new artifact contract.

Cost: best once built. It could run as a small TorchScript/NPZ/ONNX policy or a
plain exported logits model.

Correctness: mixed. It is cheap and stable, but no longer exactly LightZero
eval-mode unless the export contract is very clear.

Do not build this first. Revisit it only after MCTS checkpoint opponents are
working and env-step cost becomes the bottleneck.

## Recommended Choice

Use MCTS eval-mode for the first frozen-checkpoint opponent run.

Use direct policy-head greedy as a fallback smoke only. Do not add an exported
policy yet.

## Modal Wrapper Path

The Modal wrapper should resolve `ref:` or `volume:` checkpoint inputs before
calling `patched_dummy_pong_configs()`.

Recommended new train wrapper args:

```text
--opponent-policy lightzero_mcts_checkpoint
--opponent-checkpoint ref:training/lightzero-dummy-pong/<PARENT_RUN>/checkpoints/lightzero/iteration_16.pth.tar
--opponent-checkpoint-label parent_iter16
--opponent-checkpoint-num-simulations 2
--ego-agent player_0
```

Inside the Modal function:

1. Convert the checkpoint ref to a local path under `/runs`.
2. Verify the file exists.
3. Compute bytes and sha256 before training starts.
4. Pass the resolved local path into `patched_dummy_pong_configs()`.
5. Write the original ref and resolved file summary into `command.json`,
   `config.json`, and train `summary.json`.

The env config should contain:

```python
"opponent_checkpoint_path": "/runs/...",
"opponent_checkpoint_source_ref": "training/lightzero-dummy-pong/...",
"opponent_checkpoint_sha256": "...",
```

The checkpoint path is what the env uses. The source ref and hash are telemetry.

## Config Patch

Extend `patched_dummy_pong_configs()` to accept optional opponent checkpoint
fields and copy them into `main_config["env"]`.

Do not overload `model_path`. That is LightZero learner state. The opponent is
part of the environment config, not the learner initialization path.

Proposed optional parameters:

```python
opponent_checkpoint_path: str | None = None
opponent_checkpoint_label: str | None = None
opponent_checkpoint_adapter: str | None = None
opponent_checkpoint_num_simulations: int | None = None
opponent_checkpoint_sha256: str | None = None
opponent_checkpoint_source_ref: str | None = None
ego_agent: str = "player_0"
```

Also expose these in `_extract_surface()` so the summary shows the full
training opponent.

## No Modal Calls In Env Steps

The env step must be pure in-container Python:

```text
LightZero learner action
-> DummyPongLightZeroEnv.step(action)
-> frozen opponent policy.action(...)
-> PongEnv.step(joint_action)
```

Bad shape:

```text
env.step()
-> modal.Function.remote()
-> checkpoint inference elsewhere
```

That would be slow, brittle, and would violate the hot-loop locality rule.

The only Modal work should happen before training starts:

- mount `curvyzero-runs`;
- resolve the checkpoint file into `/runs`;
- load it in the same container process as the env;
- keep it in memory for the life of the env instance.

## Telemetry

Every terminal env row should identify both sides clearly.

Add these fields under `curvyzero_pong`:

```text
learner_policy_kind = "lightzero_train_muzero"
learner_checkpoint_ref = null or resumed checkpoint ref
learner_checkpoint_sha256 = null or hash
learner_model_key = "model"
opponent_policy_id = "lightzero_mcts_checkpoint"
opponent_checkpoint_label = "parent_iter16"
opponent_checkpoint_path = "/runs/..."
opponent_checkpoint_source_ref = "training/lightzero-dummy-pong/..."
opponent_checkpoint_sha256 = "..."
opponent_checkpoint_adapter_schema_id = "curvyzero_lightzero_dummy_pong_mcts_eval_mode/v0"
opponent_checkpoint_adapter = "mcts_eval_mode"
opponent_model_key = "model"
ego_agent = "player_0"
opponent_agent = "player_1"
feature_mode = "tabular_ego"
feature_schema_id = "..."
model_vs_target_key = "model"
```

Also keep the existing:

- episode seed;
- action counts by agent;
- reward and shaped loss-delay return;
- trace hash;
- winner;
- last hit;
- truncation status.

For the training summary, record:

- learner run id and attempt id;
- current learner checkpoint refs copied at the end;
- frozen opponent checkpoint ref/path/hash;
- whether the opponent used policy-head or MCTS;
- opponent `num_simulations`;
- seat assignment;
- feature mode;
- model key loaded from the checkpoint.

`model_vs_target_key` should be boring and explicit. Use `model` for the first
run. Keep `target_model` as a diagnostics control only.

## First Self-Play-ish Run

Do this only after the post-seed-fix trust check passes enough to provide a
usable parent checkpoint:

1. Train the post-seed-fix run from
   `docs/working/lightzero_pong_post_seed_fix_run_plan_2026-05-09.md`.
2. Score its latest checkpoint with the independent MCTS scorecard.
3. If it loads, has varied seeds, and is not action-collapsed, select that
   checkpoint as `parent`.
4. Run a tiny learner-vs-parent attempt.

Suggested first command shape after implementation:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --opponent-policy lightzero_mcts_checkpoint \
  --opponent-checkpoint ref:training/lightzero-dummy-pong/<PARENT_RUN>/checkpoints/lightzero/iteration_16.pth.tar \
  --opponent-checkpoint-label parent_iter16 \
  --opponent-checkpoint-num-simulations 2 \
  --ego-agent player_0 \
  --max-env-step 512 \
  --max-train-iter 8 \
  --num-simulations 4 \
  --batch-size 16 \
  --update-per-collect 1 \
  --n-evaluator-episode 4 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --seed 23
```

This is intentionally smaller than the post-seed-fix trust run. It should prove
that the learner can collect against a frozen checkpoint opponent and write
honest telemetry.

Then score the new learner checkpoint against:

- `random_uniform`;
- `lagged_track_ball_1`;
- `track_ball`;
- frozen parent checkpoint.

The parent row should be paired-seat in the independent scorecard. The training
run itself can start with `ego_agent=player_0`; add a second tiny run with
`ego_agent=player_1` only after the player-0 path works.

## Pass/Fail Read

Pass the first frozen-checkpoint training smoke if:

- the env loads the frozen checkpoint once per env instance;
- no Modal call occurs from env reset or step;
- `episodes.jsonl` records the opponent checkpoint ref/path/hash;
- action histograms are present for learner and frozen opponent;
- a learner checkpoint is written and mirrored;
- the independent scorecard can evaluate learner vs parent.

Fail it if:

- env steps try to resolve refs or call Modal;
- the opponent checkpoint path is missing from telemetry;
- the opponent silently falls back to scripted policy;
- action counts collapse without being visible;
- parent and learner checkpoint ids cannot be distinguished.

## Multiplayer And CurvyTron Fit

This is a bridge, not the final form.

For CurvyTron-style multiplayer, the long-term target is multiple policies
acting in the same world with explicit seats, shared rules, and clear value
semantics. The current LightZero wrapper still has a single learner action and
an env-owned opponent action.

Frozen-checkpoint opponent support helps anyway because it establishes:

- policy identity per seat;
- checkpoint provenance per player;
- parent-vs-child comparison checks;
- in-container inference locality;
- paired-seat evaluation habits;
- telemetry that can later scale from two-player Pong to multiplayer rounds.

Do not jump straight to full joint-action MuZero. First make one frozen parent
opponent work, then paired seats and honest checkpoint curves, then consider a
multiplayer env wrapper. Manual promotions/generations are not the core plan.

## Tiny Helper Candidate

There is duplicated ref/path resolution logic in the LightZero policy-head and
MCTS Modal scoreboards. The train wrapper will need the same behavior for
`--opponent-checkpoint`.

A tiny helper probably belongs in `src/curvyzero/infra/modal/run_management.py`
or a small LightZero Modal utility:

```text
resolve_mounted_ref_or_path(path_text, mount, remote_root) -> path + source metadata
```

Do not implement it during the post-seed-fix run. Ask before adding it, because
it touches shared Modal plumbing.

## Implementation Update - 2026-05-09

Small support has now been added, without starting a train.

- `run_management.py` owns shared Modal ref/path helpers:
  `explicit_volume_ref()`, `resolve_mounted_ref_or_path()`, and
  `file_summary_any_mount()`. The policy-head and MCTS scorecard wrappers now
  use this shared resolver instead of carrying duplicate checkpoint resolver
  logic.
- `DummyPongLightZeroEnv` accepts frozen opponent checkpoint config fields and
  can construct either `lightzero_policy_head_checkpoint` or
  `lightzero_mcts_checkpoint` opponents. The env receives a resolved local file
  path; it does not resolve refs or call Modal from reset/step. The checkpoint
  policy is loaded once per env instance and reset per episode.
- Env terminal telemetry now records learner kind plus opponent checkpoint
  label, resolved path, source ref, hash, bytes, adapter, adapter schema, state
  key, seat assignment, and reconstruction opponent policy when a checkpoint
  opponent is used.
- `patched_dummy_pong_configs()` exposes the checkpoint and `ego_agent` fields
  in the env config and config surface.
- `lightzero_dummy_pong_train_attempt` and the shared tiny-train runner accept
  `--opponent-checkpoint`, resolve it under `/runs` or `/repo` before config
  construction, verify the file exists, compute file summary/hash, and write
  the resolved input into command/config/summary artifacts.

Deferred:

- No train was run.
- No pytest was run.
- MCTS checkpoint opponents are implemented as an option, but the first run
  should keep `--opponent-checkpoint-num-simulations` very small because this
  calls LightZero eval-mode MCTS inside env action selection.

## Smoke Update - 2026-05-09

Frozen-checkpoint opponent training is runnable now.

The first actual smoke used the MCTS eval-mode frozen checkpoint opponent with
the deliberately poor but real parent checkpoint:

```text
ref:training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_0.pth.tar
```

Command shape:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt \
  --mode train \
  --run-id frozen-selfplay-smoke-20260509 \
  --attempt-id attempt-mcts-opp-iter0-smoke-2 \
  --opponent-policy lightzero_mcts_checkpoint \
  --opponent-checkpoint ref:training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_0.pth.tar \
  --opponent-checkpoint-label post_deep_seed_iter0 \
  --opponent-checkpoint-adapter mcts_eval_mode \
  --opponent-checkpoint-num-simulations 2 \
  --max-env-step 128 \
  --max-train-iter 2 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --num-simulations 4 \
  --batch-size 8 \
  --update-per-collect 1 \
  --n-episode 1 \
  --game-segment-length 50
```

Result:

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-CnPtTlNv3qLWBTnNnB7V62
status: completed
ok: true
problems: []
called_train_muzero: true
summary: training/lightzero-dummy-pong/frozen-selfplay-smoke-20260509/attempts/attempt-mcts-opp-iter0-smoke-2/train/summary.json
episodes: training/lightzero-dummy-pong/frozen-selfplay-smoke-20260509/attempts/attempt-mcts-opp-iter0-smoke-2/train/episodes.jsonl
```

Trainer-side scorecard:

```text
episodes: 5
wins / losses / timeouts: 4 / 1 / 0
survival_steps mean / median / p90 / max: 10.2 / 8.0 / 14.6 / 19.0
score_return mean: 0.6
shaped_loss_delay_return mean: 0.60625
learner_control_kinds: [live]
opponent_control_kinds: [frozen_checkpoint]
opponent_policy_ids: [lightzero_mcts_checkpoint]
opponent_is_frozen_checkpoint: true
```

Action counts:

```text
player_0: up 11, stay 16, down 24
player_1: up 51, stay 0, down 0
```

Frozen opponent ref reported by the scorecard:

```text
label: post_deep_seed_iter0
adapter: mcts_eval_mode
adapter_schema_id: curvyzero_lightzero_dummy_pong_mcts_eval_mode/v0
num_simulations: 2
source_ref: training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_0.pth.tar
sha256: 4b20241909346a52334d25d2fa4adc91349a5cc7314bf8c8dd7ce9bd8fae493e
state_key: model
```

Small reporting fix made during the smoke:

- terminal env rows now include `learner_control_kind=live` and
  `opponent_control_kind=scripted|frozen_checkpoint`;
- `summarize_episode_rows()` now reports learner/opponent control kinds,
  opponent policy ids, `opponent_is_frozen_checkpoint`, and frozen checkpoint
  refs in the scorecard.

This smoke does not prove policy quality. The frozen opponent selected `up` on
every recorded step, so the next meaningful run should use a better parent
checkpoint, likely `iteration_16.pth.tar`, and then run the independent paired
MCTS scorecard.
