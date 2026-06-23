# Orchestration

Created: 2026-05-19.

## Operating Principle

Keep the first integration boring. The first goal is not better learning. The
first goal is proving that Curvy observations can flow through a LightZero RND
reward-model path in the same trainer process while changing no target rewards.

## Same-Process Pattern

Run RND inside the same Modal trainer function and the same Python process as
the LightZero trainer.

Expected topology:

```text
Modal train function
  -> build Curvy/LightZero config
  -> select one LightZero entrypoint
  -> install hooks against that selected entrypoint
  -> train_muzero or train_muzero_with_reward_model
  -> replay samples feed Curvy RND adapter
  -> RND metrics/state use existing checkpoint/progress plumbing
```

This avoids a distributed mutable predictor, device-placement drift, and replay
ownership ambiguity. Env workers keep producing ordinary game segments. Eval and
tournament jobs remain extrinsic-only and never update RND.

## Entry Point Selection

Select one callable before hook installation:

```python
entrypoint_name = "train_muzero_with_reward_model" if rnd_enabled else "train_muzero"
train_entrypoint = getattr(entry_module, entrypoint_name)
```

Every hook installer must patch the selected callable, and the final training
call must call that same selected callable.

Do not patch `train_muzero.__globals__` while calling
`train_muzero_with_reward_model`.

Current RND hook implementation:

```text
selected entrypoint globals["RNDRewardModel"] = CurvyRNDRewardModel
lzero.reward_model.rnd_reward_model.RNDRewardModel = CurvyRNDRewardModel when importable
restore both after trainer exit
```

This makes the dependency explicit: if upstream LightZero stops resolving
`RNDRewardModel` through those symbols, the meter path should fail in smoke
before any nonzero reward run is allowed.

## Atomic RND Bundle

RND mode must switch these together:

- trainer entrypoint: `train_muzero_with_reward_model`;
- `cfg.reward_model`;
- `policy.use_rnd_model`;
- `policy.use_momentum_representation_network`;
- target update flags for the intrinsic reward model;
- Curvy RND input adapter config;
- exploration metadata/config hash.

If any piece is missing, fail before training starts.

## First Input Contract

Default input:

```text
feature_source: policy_gray64_latest/v0
shape: (N, 1, 64, 64)
layout: NCHW
dtype: float32
range: [0.0, 1.0]
source: latest channel from the existing policy obs stack
```

The adapter derives batch and unroll lengths from LightZero tensors. It must not
inherit upstream hardcoded reshape assumptions.

## State Ownership

- The env owns game state and extrinsic reward.
- The LightZero trainer owns replay, learner updates, and target reward
  construction.
- The Curvy RND wrapper owns predictor/target/optimizer/normalizer state.
- Existing checkpoint/resume plumbing will own persistence later.
- Current `rnd_meter_v0` is diagnostic/non-resumable.
- Tournament/eval owns extrinsic scoring only and may record exploration hashes
  for provenance.

## 2026-05-20 Cadence Update

Do not run serious RND with `rnd_update_per_collect=1`. That value was only
useful as a tiny smoke. The current code default is `100`, and positive-weight
RND should compare `50`, `100`, and policy `update_per_collect` parity before
any claim about whether curiosity helps or hurts.

Every RND run must report:

- collect/train/estimate call counts;
- `train_cnt_rnd / estimate_cnt_rnd`;
- small-buffer skip count;
- raw RND MSE percentiles before normalization;
- normalized intrinsic percentiles after normalization;
- intrinsic/extrinsic target ratio when weight is nonzero.

The current normalization is LightZero-style per-estimate-batch min/max. That
is acceptable for compatibility and meter smokes, but it is not the same thing
as OpenAI's running intrinsic reward normalization. Label it plainly as
batch-relative until a running normalizer is implemented.

Preferred positive-RND contract: use running standard deviation over the
unnormalized RND MSE stream, divide raw MSE by that scale, clip to an explicit
cap, then multiply by weight. Do not subtract a running mean for the reward
itself because novelty should not become negative. Positive `rnd_replay_target_v0`
stays blocked until this normalizer and its checkpoint/resume state are owned.

## Critique Rhythm

Use parallel critiques to stress the plan, then collapse them into one changed
task board. Avoid creating one doc per concern unless the work has actually
grown enough to need it.

Critique lanes that are useful here:

- minimalism: what can be deleted or deferred;
- orchestration: process boundaries, hook order, resume ownership;
- robustness: tests, fail-closed behavior, metric-only gates;
- input choice: RND feature source and failure modes;
- documentation: whether the packet is still navigable.

## Anti-Patterns

- Building a generic extension framework before the first RND canary.
- Adding env reward edits for true RND.
- Running one RND predictor per env worker.
- Running an RND reward service in a second Modal function.
- Storing mutable predictor state in Modal Dict/Queue.
- Letting checkpoint eval or tournament mutate RND state.
- Hiding intrinsic reward inside `reward_variant`.
- Enabling nonzero intrinsic reward before the meter-only path proves neutral.
