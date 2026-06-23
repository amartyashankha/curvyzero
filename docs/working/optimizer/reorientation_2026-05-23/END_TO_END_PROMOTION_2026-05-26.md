# End-To-End Promotion Plan

Date: 2026-05-26

## Plain State

The fast resident GPU path is not wired into stock Coach training yet.

There are two different paths:

- Stock Coach path: `lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  calls LightZero `train_muzero`.
- Fast compact path: `source_state_batched_observation_boundary_profile.py`
  calls `run_hybrid_observation_profile`.

The compact path is fast because it keeps rollout/search/replay accounting in
arrays and can keep observation frames resident on the GPU. The stock path still
turns collection into scalar observations, Python dictionaries, `GameSegment`
objects, and `MuZeroGameBuffer` samples.

## Stock Boundary

Stock LightZero does this:

```text
env.ready_obs
-> MuZeroCollector builds stacked scalar observations
-> policy._forward_collect runs search and returns Python dict rows
-> env.step(actions)
-> GameSegment append/search stats
-> MuZeroGameBuffer.push_game_segments
-> MuZeroGameBuffer.sample builds current_batch/target_batch
-> BaseLearner.train
```

The current batched profile env-manager bridge reaches `train_muzero`, but it
still returns scalar LightZero timesteps. It is useful evidence, not the resident
compact path.

## Promotion Choice

Trying to hide the resident compact path inside stock `train_muzero` means
patching or replacing the collector, collect policy/search, and possibly the
buffer/sample boundary. That is possible, but brittle.

The cleaner next candidate is a separately named compact-owned training profile:

```text
compact actors
-> resident GPU observation stack
-> compact Torch search
-> compact replay rows
-> compact sampler/target builder
-> real or explicitly mocked learner/RND edge
```

Until it has a real learner/target/checkpoint/eval loop, it must say
`calls_train_muzero=false`.

## Missing Honest-Training Pieces

- durable replay buffer, not just an in-memory profile ring;
- MuZero unroll targets, value/reward transforms, discounting, and terminal
  final-observation support;
- real learner update with model/optimizer state, not the tiny timing probe;
- RND as actual reward/target input if enabled, not only an RND-style probe;
- checkpoint/resume/eval/GIF/tournament observability boundaries;
- policy-version handoff from learner back to search.

## Current CPU Crossings To Kill

The target is still: keep the hot path compact/resident and sync only tiny
action/log/checkpoint data at coarse cadence.

As of 2026-05-26, one hidden crossing is fixed locally: resident replay samples
can feed the learner gate as `CompactResidentSampleBatchV1` without copying
observation tensors through NumPy or host arrays.

The next crossing is also fixed locally for the resident compact profile lane:
search replay targets can flush as a device payload and commit as
`CompactDeviceReplayIndexRowsV1`, so visit policy/root value/count tensors do
not pass through the old NumPy replay-payload/index-row builders. The matching
H100 profile row (`optimizer-device-replay-sample-gate-20260526`) confirms
committed replay payload D2H is `0` on the B256/A8/sim8 sample-gate denominator.

The compact learner crossing now has a local strict implementation too:
`compact_muzero` consumes resident tensors plus explicit committed successor
targets. It is real optimizer work, but still profile-only and not stock
`train_muzero`.

The first H100 Modal proof of that strict learner crossing passed:
`optimizer-compact-muzero-gate-smoke-20260526`, row `001`. It reported
`compact_muzero`, `real_muzero_update=true`, `toy_probe=false`, explicit
successor targets, next action masks present, learner input H2D `0`, and
committed replay payload D2H `0`.

Remaining crossings:

- CPU CurvyTron physics still owns `env.step`; selected actions must be copied
  back as a tiny `[B,P]` action array until physics moves.
- Checkpoint/eval/tournament still belong to the stock path unless explicitly
  wired to the compact candidate.

## Compact Learner Edge, 2026-05-26

Added a first fail-closed learner edge:

- code: `src/curvyzero/training/compact_muzero_learner.py`;
- tests: `tests/test_compact_muzero_learner.py`;
- local proof: focused compact MuZero and hybrid-profile tests pass.

What it proves:

- resident compact samples can feed a one-step MuZero-shaped tensor update
  without calling `policy.learn_mode.forward`;
- the update runs model `initial_inference`, model `recurrent_inference`,
  support transforms, policy/value/reward losses, backward, gradient clipping,
  and optimizer step;
- host fallback is rejected when resident samples are required.

What it does not prove yet:

- it is not stock `train_muzero`;
- it is not wired into checkpoint/eval/tournament;
- it is not a stock LightZero learner parity proof.

The important contract point is the next-step target. The strict mode requires
`next_policy_target` and `next_root_value` from committed successor rows.
Reusing the current targets is allowed only through the explicitly named
profile-only mode `repeat_current_targets_profile_only`. Do not hide that
shortcut in a training recommendation.

## Immediate Profile Gate

Run from the same repo state:

1. A stock `train_muzero` profile row with current RND/reward/observation knobs.
2. The best compact resident trainer-like profile row.

Report them as different currencies:

- stock row: real LightZero profile, `called_train_muzero=true`;
- compact row: compact trainer-like profile, `calls_train_muzero=false`.

The result should answer:

- how much stock training currently costs;
- how much the compact candidate costs when it includes search feedback,
  replay sampling, learner/RND probe, and no scalar timesteps;
- how much headroom remains before the compact path is worth promoting.

Do not call the compact row a Coach speedup until it has the missing
honest-training pieces above.
