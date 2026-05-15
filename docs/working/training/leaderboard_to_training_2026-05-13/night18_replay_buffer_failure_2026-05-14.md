# Night18 Replay Buffer Failure - 2026-05-14

## Short Answer

The fixed 18-run batch is hitting a LightZero replay-buffer invariant failure.
The failing line samples transition indices with:

```python
np.random.choice(num_of_transitions, batch_size, p=probs, replace=False)
```

NumPy raises `ValueError: 'a' and 'p' must have same size` when the integer
range size, here `num_of_transitions`, does not match the length of `probs`.
So the replay buffer has one count of stored transitions, but its priority
vector has a different length.

This does not look reward-variant-specific. I found the same failure in sparse
rows and in a survival-plus-bonus-no-outcome row. The reward variant changes
reward values and support settings, but the stack fails before reward-specific
target math can explain the size mismatch.

Most likely root cause: LightZero's replay-buffer priority bookkeeping is
getting out of sync with its transition lookup after `push_game_segments`, or
later when old segments are cleared/re-indexed. Our current audit hooks do not
capture the needed replay-buffer lengths at the failing moment.

## Batch And Settings

Manifest:

`artifacts/local/curvytron_tonight18_manifests/curvy-night18-top10fallback-fixed-20260514a/curvy-night18-top10fallback-fixed-20260514a.json`

Matrix:

- `matrix_name`: `curvy-night18-top10fallback-fixed-20260514a`
- generated at `2026-05-14T09:36:13.517200+00:00`
- run prefix: `curvy-n18fb`
- 18 rows: 3 reward variants x 3 opponent recipes x 2 noise modes

Important fixed knobs:

- `batch_size=32`
- `collector_env_num=256`
- `n_episode=256` through each row command
- `max_env_step=30000000`
- `max_train_iter=300000`
- `source_max_steps=65536`
- `num_simulations=8`
- `save_ckpt_after_iter=10000`

Local launcher details:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  sets `LIGHTZERO_VERSION = "0.2.0"`.
- In the same file, `_lightzero_target_config_for_reward` sets
  `discount_factor=1.0` and `td_steps=int(source_max_steps)` for the
  source-state fixed-opponent env. In this batch that means `td_steps=65536`.
- `_build_visual_survival_configs` patches the LightZero policy with
  `collector_env_num`, `n_episode`, `batch_size`, and the target config.
- Recorded segment metadata therefore has `unroll_plus_td_steps=65541`
  (`td_steps + unroll_steps`, apparently 65536 + 5).

The large `td_steps` is worth watching for memory/target cost, but it is not by
itself the direct NumPy error. The direct error is a replay transition count
versus priority-vector length mismatch.

## Confirmed Failed Rows

Remote latest-attempt and summary snippets show these fixed-prefix failures:

| Row | Reward variant | Run | Failure timing |
| --- | --- | --- | --- |
| r003 | `sparse_outcome` | `curvy-n18fb-sparse-blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35-clean-s1977851466` | failed after `12640.669227` train seconds; latest checkpoint `iteration_130000.pth.tar` |
| r004 | `sparse_outcome` | `curvy-n18fb-sparse-blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35-so10rep10-s965175368` | failed after `2109.07676` train seconds; latest checkpoint `iteration_20000.pth.tar` |
| r011 | `survival_plus_bonus_no_outcome` | `curvy-n18fb-survbonusnoout-blank20-wall5-rank1_75-clean-s791537956` | failed after `112.378284` train seconds; no iteration checkpoint, only `ckpt_best.pth.tar` |

All three have:

```text
LightZero train_muzero failed: ValueError: 'a' and 'p' must have same size
```

The traceback tail is the same:

```text
game_buffer_muzero.py, sample
game_buffer_muzero.py, _make_batch
game_buffer.py, _sample_orig_data
np.random.choice(num_of_transitions, batch_size, p=probs, replace=False)
ValueError: 'a' and 'p' must have same size
```

The r011 latest attempt started at `2026-05-14T09:38:10.111950Z` and ended at
`2026-05-14T09:40:25.360049Z`.

Sparse latest attempts also show r001, r002, r005, and r006 still `running` at
the time of this check, while r003 and r004 were `failed`.

## What The Current Audit Shows

For r011, target/replay audit counts were:

```json
{
  "collector_collect_calls": 1,
  "game_segments_seen": 258,
  "game_segments_recorded": 4,
  "replay_push_calls": 1,
  "replay_sample_calls": 0,
  "replay_samples_recorded": 0
}
```

The first four recorded r011 segment action lengths were:

```text
13, 33, 34, 35
```

Each recorded segment had metadata:

```json
{
  "priorities": null,
  "unroll_plus_td_steps": 65541
}
```

For r003, the same audit saw 14 collector calls, 14 replay pushes, and 131369
successful replay sample calls before the failing sample. For r004 it saw 4
collector calls, 4 replay pushes, and 28129 successful replay sample calls.

This means the same invariant can break immediately after the first random
collect, as in r011, or much later after many samples and several replay pushes,
as in r003/r004.

One caution: the audit note `replay push saw 2 game segments` is probably not
trustworthy as a segment count. The wrapper currently records `len(args[0])`
after `push_game_segments` returns. If LightZero passes a tuple like
`(game_segments, metadata)`, that note reports the tuple length, not the number
of game segments. The collector-side count, for example r011's 258 segments, is
more useful.

## What The Current Audit Misses

The target audit wraps:

- `Collector.collect`
- `GameBuffer.push_game_segments`
- replay-buffer `sample`

But `sample` is recorded only after the original LightZero `sample` call
returns. In this failure, LightZero raises inside `_sample_orig_data`, so the
sample wrapper never records the failing state.

The relevant local code is `_install_lightzero_target_audit` in
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`.
`record_replay_push` also only records the apparent length of `args[0]`; it does
not inspect the replay buffer object itself.

The audit also does not snapshot these replay-buffer fields before sampling:

- `get_num_of_transitions()`
- `len(game_pos_priorities)`
- `len(game_segment_game_pos_look_up)`
- number of stored game segments
- action-length sum across stored segments
- priority min/max/nan/inf checks

Those are the fields needed to prove exactly which internal list drifted.

## Resume Is Unlikely

The confirmed failed summaries have `auto_resume.found=false`.

Also, local resume state currently records replay-buffer metadata only:

```json
{
  "raw_game_segments_saved": false
}
```

`_restore_lightzero_replay_buffer_state` returns false unless raw game segments
were saved. So random collect should not be skipped for these fresh attempts.
The r011 failure happened without an iteration checkpoint and without a resume
source, which points away from resume restore as the cause.

The relevant local functions are `_lightzero_replay_buffer_state`,
`_restore_lightzero_replay_buffer_state`, and the `random_collect` wrapper in
the same launcher file.

## Likely Root Cause

The replay buffer has these conceptual pieces:

- a stored set of game segments
- a lookup from transition index to `(game_segment, position)`
- a priority/probability entry per sampleable transition

The crash means those pieces are not the same length at sample time.

Most likely, after a `push_game_segments` call, or after replay-buffer cleanup,
one of these happens:

- the transition lookup is updated but `game_pos_priorities` is not updated for
  every transition
- `game_pos_priorities` is updated but stale transition lookup entries remain
- empty or unusual short game segments are counted differently by the lookup and
  the priority vector
- priority metadata is absent (`priorities: null`) and LightZero's default
  priority creation path does not always create one priority per transition

The data argues for a global replay-buffer bookkeeping bug, not a sparse-only or
survival-only reward bug:

- r003 and r004 are `sparse_outcome`.
- r011 is `survival_plus_bonus_no_outcome`.
- all fail in the same LightZero replay sampling line.
- r011 fails before learning has really started; r003 fails after
  `iteration_130000`.

Reward variant may change how quickly the bug appears by changing episode
lengths or buffer turnover, but it is probably not the direct cause.

## Minimal Pin-Down Patch

Add a passive replay-buffer invariant snapshot in the existing target audit.
Do not silently fix the buffer first.

Best hook points:

1. Immediately after `push_game_segments` returns.
2. Immediately before `sample` calls the original LightZero method.
3. In a `try/except` around the original `sample`, so failures still record the
   same snapshot before re-raising.

Suggested snapshot fields:

```python
def _audit_replay_buffer_invariant(buffer):
    get_num = getattr(buffer, "get_num_of_transitions", None)
    num_transitions = get_num() if callable(get_num) else None
    priorities = getattr(buffer, "game_pos_priorities", None)
    lookup = getattr(buffer, "game_segment_game_pos_look_up", None)
    segments = getattr(buffer, "game_segment_buffer", None)
    return {
        "num_transitions": num_transitions,
        "game_pos_priorities_len": len(priorities) if priorities is not None else None,
        "game_segment_game_pos_lookup_len": len(lookup) if lookup is not None else None,
        "game_segment_buffer_len": len(segments) if segments is not None else None,
        "segment_action_len_sum": sum(
            len(getattr(segment, "action_segment", [])) for segment in (segments or [])
        ),
    }
```

Then record a `replay_invariant_errors` item when:

```python
num_transitions != game_pos_priorities_len
```

or:

```python
num_transitions != game_segment_game_pos_lookup_len
```

This would turn the current opaque NumPy error into a direct statement such as:

```text
replay invariant mismatch before sample:
num_transitions=12345
game_pos_priorities_len=12312
game_segment_game_pos_lookup_len=12345
```

That is enough to know whether the priority list or lookup list is wrong.

## Minimal Test

A narrow unit test can cover the local wrapper without running training:

1. Build a fake replay buffer with:
   - `get_num_of_transitions()` returning `3`
   - `game_pos_priorities=[1.0, 1.0]`
   - `game_segment_game_pos_look_up=[(0, 0), (0, 1), (0, 2)]`
   - one fake segment with three actions
2. Call the new invariant snapshot.
3. Assert it reports the mismatch before any call to `np.random.choice`.

A better integration smoke, when it is safe to launch a new run, is a tiny
profile/train job with:

- `collector_env_num=4`
- `n_episode=4`
- `batch_size=4`
- small `source_max_steps`
- target audit enabled

The smoke should stop after the first replay push/sample and assert:

```text
get_num_of_transitions()
== len(game_pos_priorities)
== len(game_segment_game_pos_look_up)
```

Do not use the current 18-run jobs for this. They are already running or failed,
and this investigation did not relaunch or kill anything.

## Patch Direction After Pin-Down

First patch should be diagnostic: record the invariant and fail with a clear
message if it is broken.

Only after the broken field is known should we consider a defensive repair. If
the lookup length is correct and only `game_pos_priorities` is wrong, a safe
repair may be to rebuild uniform priorities with one entry per transition before
sampling. But that changes replay behavior, so it should be explicit and covered
by a test.
