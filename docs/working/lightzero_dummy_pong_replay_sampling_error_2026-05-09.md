# LightZero Dummy Pong Replay Sampling Error

Date: 2026-05-09

## Question

The custom dummy Pong target-telemetry smoke reached LightZero collection, wrote
target replay sidecars, then failed in replay sampling:

```text
ValueError: 'a' and 'p' must have same size
```

This note checks whether the failure is from an undersized tiny trainer config
or from the telemetry wrapper changing collector output shape.

## Source Inspection

LightZero v0.2.0 `train_muzero` only samples when replay has more transitions
than the configured batch size:

```python
if replay_buffer.get_num_of_transitions() > batch_size:
    train_data = replay_buffer.sample(batch_size, policy)
```

The failing line is in `GameBuffer._sample_orig_data`:

```python
num_of_transitions = self.get_num_of_transitions()
probs = self.game_pos_priorities ** self._alpha + 1e-6
probs /= probs.sum()
batch_index_list = np.random.choice(num_of_transitions, batch_size, p=probs, replace=False)
```

NumPy raises this exact error when `len(probs) != num_of_transitions`.

The length mismatch can be created earlier in `GameBuffer._push_game_segment`.
For non-terminal segments, LightZero computes:

```python
valid_len = data_length - meta['unroll_plus_td_steps']
```

With `priorities is None`, it then appends priority entries using
`range(valid_len, data_length)`. If `valid_len` is negative, this creates more
priority entries than lookup entries. The transition lookup still receives only
`data_length` entries, so later `np.random.choice(a=num_of_transitions, p=probs)`
sees different sizes.

The collector sets:

```python
self.unroll_plus_td_steps = self.policy_config.num_unroll_steps + self.policy_config.td_steps
```

For MuZero v0.2.0, the implicit defaults are `num_unroll_steps=5` and
`td_steps=5`, so `unroll_plus_td_steps=10` unless patched.

## Failing Smoke Config

The failing telemetry smoke used:

```text
--max-env-step 16
--pong-episode-max-steps 16
--max-train-iter 1
--batch-size 1
--update-per-collect 1
--n-episode 1
--game-segment-length 4
--random-collect-episode-num 1
```

The important part is `game_segment_length=4` with implicit
`num_unroll_steps=5` and `td_steps=5`.

For any non-terminal full segment in that run:

```text
data_length = 4
unroll_plus_td_steps = 10
valid_len = -6
```

That is enough to make `game_pos_priorities` longer than
`game_segment_game_pos_look_up`, which explains the later replay sampling
exception.

## Telemetry Wrapper Check

The scoped telemetry wrapper around `MuZeroCollector.collect` calls the original
collector first, normalizes the returned `(segments, metadata)` for JSONL
mirroring, writes sidecar rows, and returns the original result unchanged.

The sidecar evidence also looks structurally coherent:

- `target_replay_summary.json` reported 28 rows, 2 episodes, and 1 collect call.
- terminal rows carried metadata like
  `{"done": true, "priorities": null, "unroll_plus_td_steps": 10}`.
- rows had action ids, action labels, child visit distributions, rewards, and
  root values.

So the wrapper appears to have exposed a bad tiny-config replay state rather
than changing the collector output shape.

## Cause Hypothesis

Primary cause: the smoke configured `game_segment_length` too small for
LightZero's default MuZero target window.

This is not principally caused by:

- `max_env_step`: it limited the run, but did not create the probability-vector
  mismatch.
- `max_train_iter`: it only allowed the learner to reach one sample attempt.
- `batch_size`: `batch_size=1` made sampling happen as soon as replay had more
  than one transition; increasing it could hide the issue by skipping learning,
  but would not fix replay bookkeeping.
- `update_per_collect`: it controlled how many sample attempts were made after
  collection, not the mismatch itself.
- replay buffer capacity: the run was nowhere near capacity.
- telemetry wrapper shape: source inspection shows it returns the original
  collector result unchanged after mirroring.

## Cheapest Fix

For future custom dummy Pong telemetry smokes, do not use
`--game-segment-length 4` with default MuZero target knobs.

Use one of:

```text
--game-segment-length 10
```

or a little roomier:

```text
--game-segment-length 16
```

or omit the override and keep the project default `DEFAULT_GAME_SEGMENT_LENGTH =
50`.

If a deliberately tiny segment length is required, also make the target window
smaller, for example:

```text
--game-segment-length 4 --num-unroll-steps 1 --td-steps 1
```

That variant is a mechanical telemetry smoke only; it changes target semantics
and should not be treated as a learning-quality configuration.

## Safe-Segment Smoke Result

Reran the custom dummy Pong Modal telemetry smoke with
`--game-segment-length 16`. The exact `batch-size 1` variant avoided the
previous replay sampling mismatch, but failed later with a batchnorm single-row
learner error. A still-tiny `--batch-size 2` rerun completed:

```text
run_id=lz-dpong-20260509T184118Z-bf893dffebfd
attempt_id=attempt-20260509T184118Z-933639b09f7c
status=completed
summary_ref=training/lightzero-dummy-pong/lz-dpong-20260509T184118Z-bf893dffebfd/attempts/attempt-20260509T184118Z-933639b09f7c/train/summary.json
target_replay_steps=training/lightzero-dummy-pong/lz-dpong-20260509T184118Z-bf893dffebfd/attempts/attempt-20260509T184118Z-933639b09f7c/train/target_replay_steps.jsonl
target_replay_summary=training/lightzero-dummy-pong/lz-dpong-20260509T184118Z-bf893dffebfd/attempts/attempt-20260509T184118Z-933639b09f7c/train/target_replay_summary.json
```

Fetched and read the target replay sidecars. `target_replay_summary.json`
reported 16 rows, 1 episode, 1 collect call, 1 terminal step, and action counts
`up=3`, `stay=4`, `down=9`. The logged child-visit target distributions were:

```json
{
  "[0.5, 0.5, 0.0]": 4,
  "[0.5, 0.0, 0.5]": 9,
  "[0.0, 0.5, 0.5]": 3
}
```

Reward counts were `0.0=15` and `1.0=1`. The terminal target row was:

```jsonl
{"global_step_index": 15, "global_episode_index": 0, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.5, 0.0, 0.5], "reward": 1.0, "done": true, "segment_metadata": {"done": true, "priorities": null, "unroll_plus_td_steps": 10}}
```

No quality claim: this was only a bounded telemetry support smoke.

## Follow-Up

No pytest was run for this investigation. A possible future guard would be a
pre-train config validation warning for custom dummy Pong when a multi-segment
episode can produce non-terminal segments with
`game_segment_length < num_unroll_steps + td_steps`.
