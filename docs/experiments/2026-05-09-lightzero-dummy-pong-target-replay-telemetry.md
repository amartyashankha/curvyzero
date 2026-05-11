# 2026-05-09 LightZero Dummy Pong Target Replay Telemetry

## Goal

Future custom dummy Pong LightZero runs should persist enough collection-target
telemetry to inspect what the policy was trained on. Existing run artifacts had
terminal env rows and checkpoints, but not LightZero `GameSegment` target data.
This note is only about the project-owned custom dummy Pong Modal wrapper, not
official Atari Pong.

Official Atari Pong follows LightZero's stock environment/config path. The
missing telemetry gap here is narrower: our CurvyZero Modal artifact mirror did
not persist the collector's `GameSegment` targets for custom dummy Pong runs,
even though LightZero keeps those targets in memory while training.

## Implementation

The Modal dummy Pong trainer now mirrors a compact JSONL sidecar from
`MuZeroCollector.collect(...)` returns:

- `train/lightzero-dummy-pong/.../target_replay_steps.jsonl`
- `train/lightzero-dummy-pong/.../target_replay_summary.json`

Each target row has:

- `collect_call_index`, `collect_train_iter`, `global_episode_index`,
  `global_step_index`, `segment_index`, and `step_index_in_segment`
- executed `action_segment` plus `action_label`
- normalized `child_visit_segment` and `visit_count_distribution`
- scalar `reward` and `root_value`
- terminal `done` when the returned segment metadata says this is the final
  step of a completed episode
- `target_config` containing `num_simulations`, reward/value support ranges,
  feature mode/shape, reset profile/pressure/max steps, ego agent, opponent
  policy, and opponent checkpoint metadata when configured

Truncation remains in the env terminal sidecar:

- `train/lightzero-dummy-pong/.../episodes.jsonl`

LightZero `GameSegment` does not expose `truncated` per transition, so target
rows set `truncated: null` and point readers to `episodes.jsonl` for terminal
`done`/`terminated`/`truncated` truth.

## Method Choice

Direct replay-buffer mirroring after `train_muzero` is awkward because the
entrypoint owns the replay buffer and only returns the final policy. The
smallest robust method is a scoped wrapper around `MuZeroCollector.collect`.
It serializes the exact `GameSegment`s returned to `train_muzero` before they
are pushed into `MuZeroGameBuffer`, then restores the original collector method
when the train call exits.

## Smoke

No long train and no pytest. Validation used local compilation:

```sh
python -m py_compile src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py
```

The first tiny Modal train smoke reached `train_muzero` but wrote an empty
target replay sidecar because the collector wrapper only recognized list
returns. The wrapper was adjusted to normalize list-or-tuple collector returns,
then recompiled with the command above.

Tiny Modal command:

```sh
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke \
  --mode train \
  --pong-reset-profile contact_pressure \
  --max-env-step 16 \
  --pong-episode-max-steps 16 \
  --max-train-iter 1 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --num-simulations 2 \
  --batch-size 1 \
  --update-per-collect 1 \
  --n-episode 1 \
  --game-segment-length 4 \
  --random-collect-episode-num 1
```

Modal run:

- App run: `ap-F3hnQpD5yQu5E4POOa015y`
- Run id: `lz-dpong-20260509T183337Z-1e205ad09f28`
- Attempt id: `attempt-20260509T183337Z-3fdf4a789850`
- Summary ref:
  `training/lightzero-dummy-pong/lz-dpong-20260509T183337Z-1e205ad09f28/attempts/attempt-20260509T183337Z-3fdf4a789850/train/summary.json`
- Target rows ref:
  `training/lightzero-dummy-pong/lz-dpong-20260509T183337Z-1e205ad09f28/attempts/attempt-20260509T183337Z-3fdf4a789850/train/target_replay_steps.jsonl`
- Target summary ref:
  `training/lightzero-dummy-pong/lz-dpong-20260509T183337Z-1e205ad09f28/attempts/attempt-20260509T183337Z-3fdf4a789850/train/target_replay_summary.json`

Result: the Modal command was not blocked and target telemetry was written, but
the full LightZero train attempt status is `failed` after collection:

```text
LightZero train_muzero failed: ValueError: 'a' and 'p' must have same size
```

Traceback tail points at LightZero replay-buffer sampling:

```text
/usr/local/lib/python3.11/site-packages/lzero/mcts/buffer/game_buffer.py:128
batch_index_list = np.random.choice(num_of_transitions, batch_size, p=probs, replace=False)
ValueError: 'a' and 'p' must have same size
```

Target replay summary:

```json
{
  "rows": 28,
  "episodes": 2,
  "collect_calls": 1,
  "terminal_steps": 2,
  "action_counts": {
    "up": 5,
    "stay": 7,
    "down": 16
  },
  "has_child_visit_segment": true,
  "has_reward": true
}
```

Child-visit distribution counts:

```json
{
  "[0.5, 0.5, 0.0]": 6,
  "[0.5, 0.0, 0.5]": 14,
  "[0.0, 0.5, 0.5]": 8
}
```

Reward counts:

```json
{
  "0.0": 26,
  "1.0": 2
}
```

First rows, trimmed to the fields this smoke cared about:

```jsonl
{"global_step_index": 0, "global_episode_index": 0, "action_segment": 0, "action_label": "up", "child_visit_segment": [0.5, 0.5, 0.0], "reward": 0.0, "root_value": 3.4439482988091186e-05, "done": false}
{"global_step_index": 1, "global_episode_index": 0, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.5, 0.0, 0.5], "reward": 0.0, "root_value": 3.4439482988091186e-05, "done": false}
{"global_step_index": 2, "global_episode_index": 0, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.5, 0.0, 0.5], "reward": 0.0, "root_value": 3.4439482988091186e-05, "done": false}
{"global_step_index": 3, "global_episode_index": 0, "action_segment": 1, "action_label": "stay", "child_visit_segment": [0.5, 0.5, 0.0], "reward": 0.0, "root_value": 3.4439482988091186e-05, "done": false}
{"global_step_index": 4, "global_episode_index": 0, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.0, 0.5, 0.5], "reward": 0.0, "root_value": 3.4439482988091186e-05, "done": false}
{"global_step_index": 5, "global_episode_index": 0, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.5, 0.0, 0.5], "reward": 0.0, "root_value": 3.4439482988091186e-05, "done": false}
```

Terminal target rows:

```jsonl
{"global_step_index": 23, "global_episode_index": 0, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.5, 0.0, 0.5], "reward": 1.0, "done": true, "segment_metadata": {"done": true, "priorities": null, "unroll_plus_td_steps": 10}}
{"global_step_index": 27, "global_episode_index": 1, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.5, 0.0, 0.5], "reward": 1.0, "done": true, "segment_metadata": {"done": true, "priorities": null, "unroll_plus_td_steps": 10}}
```

Config snapshot from row 0:

```json
{
  "algorithm": "LightZero MuZero",
  "num_simulations": 2,
  "feature": {
    "env": "dummy_pong_lag1",
    "feature_mode": "tabular_ego",
    "observation_shape": 10,
    "action_space_size": 3
  },
  "reset": {
    "pong_reset_profile": "contact_pressure",
    "pong_reset_pressure_agent": "ego",
    "pong_episode_max_steps": 16
  },
  "opponent": {
    "opponent_policy": "random_uniform",
    "ego_agent": "player_0",
    "opponent_checkpoint": null
  }
}
```

The env-side terminal sidecar for the same attempt had two terminal rows:

```jsonl
{"episode_index": 1, "episode_seed": 1, "score_return": -1.0, "terminated": true, "truncated": false, "winner": "player_1", "pong_reset_profile": "contact_pressure", "pong_reset_pressure_agent": "player_0"}
{"episode_index": 0, "episode_seed": 0, "score_return": 1.0, "terminated": true, "truncated": false, "winner": "player_0", "pong_reset_profile": "contact_pressure", "pong_reset_pressure_agent": "player_0"}
```

No quality claim: this only verifies that the custom dummy Pong Modal wrapper
can mirror compact `GameSegment` target rows into
`target_replay_steps.jsonl`/`target_replay_summary.json` before the tiny
LightZero replay-buffer sampler fails.

## Safe-Segment Rerun

Reran the tiny Modal telemetry smoke with a safe segment length so it would not
hit the previous replay sampling mismatch. The exact `batch-size 1` rerun with
`--game-segment-length 16` avoided the prior `'a' and 'p'` sampling error, but
failed later in the learner with:

```text
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 128])
```

The bounded completion rerun kept the same tiny collection/train shape but used
`--batch-size 2`:

```sh
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke \
  --mode train \
  --pong-reset-profile contact_pressure \
  --max-env-step 16 \
  --pong-episode-max-steps 16 \
  --max-train-iter 1 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --num-simulations 2 \
  --batch-size 2 \
  --update-per-collect 1 \
  --n-episode 1 \
  --game-segment-length 16 \
  --random-collect-episode-num 1
```

Modal run:

- App run: `ap-rdvkRpLGRYedx39SggsVvm`
- Run id: `lz-dpong-20260509T184118Z-bf893dffebfd`
- Attempt id: `attempt-20260509T184118Z-933639b09f7c`
- Status: `completed`
- Summary ref:
  `training/lightzero-dummy-pong/lz-dpong-20260509T184118Z-bf893dffebfd/attempts/attempt-20260509T184118Z-933639b09f7c/train/summary.json`
- Target rows ref:
  `training/lightzero-dummy-pong/lz-dpong-20260509T184118Z-bf893dffebfd/attempts/attempt-20260509T184118Z-933639b09f7c/train/target_replay_steps.jsonl`
- Target summary ref:
  `training/lightzero-dummy-pong/lz-dpong-20260509T184118Z-bf893dffebfd/attempts/attempt-20260509T184118Z-933639b09f7c/train/target_replay_summary.json`

Fetched and read `target_replay_summary.json` plus
`target_replay_steps.jsonl`. Summary:

```json
{
  "rows": 16,
  "episodes": 1,
  "collect_calls": 1,
  "terminal_steps": 1,
  "action_counts": {
    "up": 3,
    "stay": 4,
    "down": 9
  },
  "has_child_visit_segment": true,
  "has_reward": true
}
```

Child-visit distribution counts:

```json
{
  "[0.5, 0.5, 0.0]": 4,
  "[0.5, 0.0, 0.5]": 9,
  "[0.0, 0.5, 0.5]": 3
}
```

Reward counts:

```json
{
  "0.0": 15,
  "1.0": 1
}
```

First rows, trimmed to the fields this smoke cared about:

```jsonl
{"global_step_index": 0, "global_episode_index": 0, "action_segment": 0, "action_label": "up", "child_visit_segment": [0.5, 0.5, 0.0], "reward": 0.0, "root_value": 3.4439482988091186e-05, "done": false}
{"global_step_index": 1, "global_episode_index": 0, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.5, 0.0, 0.5], "reward": 0.0, "root_value": 3.4439482988091186e-05, "done": false}
{"global_step_index": 2, "global_episode_index": 0, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.5, 0.0, 0.5], "reward": 0.0, "root_value": 3.4439482988091186e-05, "done": false}
{"global_step_index": 3, "global_episode_index": 0, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.0, 0.5, 0.5], "reward": 0.0, "root_value": 3.4439482988091186e-05, "done": false}
{"global_step_index": 4, "global_episode_index": 0, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.5, 0.0, 0.5], "reward": 0.0, "root_value": 3.4439482988091186e-05, "done": false}
{"global_step_index": 5, "global_episode_index": 0, "action_segment": 1, "action_label": "stay", "child_visit_segment": [0.5, 0.5, 0.0], "reward": 0.0, "root_value": 3.4439482988091186e-05, "done": false}
```

Terminal target row:

```jsonl
{"global_step_index": 15, "global_episode_index": 0, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.5, 0.0, 0.5], "reward": 1.0, "done": true, "segment_metadata": {"done": true, "priorities": null, "unroll_plus_td_steps": 10}}
```

No quality claim: this support-lane smoke only verifies that the tiny custom
dummy Pong Modal run can complete with `game_segment_length=16`, preserve the
target replay sidecars, and avoid the previous replay sampling error.
