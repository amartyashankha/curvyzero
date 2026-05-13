# Frozen / Recent-Checkpoint Opponent Route

Purpose: evaluate whether we can stay close to stock LightZero by training one
ego policy against a recent frozen opponent, instead of trying to force true
same-tick live self-play immediately.

## Plain Idea

Stock LightZero can train one controlled player cleanly:

```text
policy chooses player 0 action
env supplies player 1 action
env advances one real tick
LightZero stores one normal replay transition
```

The opponent does not have to be a hand-coded straight policy forever. It could
be a frozen checkpoint from a recent run, or a small pool of recent checkpoints.

That is not exact live current-policy self-play. But it may be a practical and
much cleaner route because it keeps `train_muzero`, `GameSegment`, and
`MuZeroGameBuffer` intact.

## What It Can Prove

- The CurvyTron visual env can learn through the stock LightZero loop.
- Checkpoints can improve against named opponents.
- A recent-opponent schedule may produce a useful curriculum or league-like
  training loop.

## What It Cannot Prove By Itself

- It does not prove both players used the exact same latest weights in the same
  physical tick.
- It does not prove full competitive self-play unless the opponent refresh
  policy is explicit and evaluated against more than the training opponent.
- It can overfit to one frozen opponent, so eval panels matter.

## Why It May Be The Right Near-Term Route

The true two-seat path needs a replay bridge. Until that bridge exists, recent
frozen opponents may give us learning signal while keeping the trusted stock
trainer.

This should be labeled honestly:

```text
stock train_muzero + ego player + recent frozen opponent
```

not:

```text
true same-current-policy two-seat self-play
```

## Research Questions

- Does the Modal stock `mode=train` path pass frozen checkpoint config through
  cleanly for `source_state_fixed_opponent`?
- Does the canary prove strict checkpoint load and `called_train_muzero=true`?
- How often should the opponent refresh?
- Should training use one recent opponent or a small pool?
- What held-out opponent panel prevents overfitting claims?

## Current Findings

Frozen checkpoint plumbing exists, including in the source-state local env.
The Modal stock train path has now resolved and passed a real checkpoint
cleanly in a small CPU canary.

Existing pieces:

- a checkpoint opponent provider can load a frozen `MuZeroPolicy` and pick
  legal actions through `eval_mode.forward`;
- the source-state local env already has a test proving it can ask a frozen
  opponent provider for player-1 actions and step one real tick;
- older single-ego wrappers also have frozen opponent hooks;
- two-seat `mixpast` has static frozen-opponent mixing, but it is on the custom
  adapter path and does not call stock `train_muzero`.

Local patch status:

- `source_state_fixed_opponent` now supports
  `opponent_policy_kind=frozen_lightzero_checkpoint` in the env step path;
- the Modal train gate now allows this opponent kind for
  `source_state_fixed_opponent`;
- telemetry can expose opponent checkpoint/provider strict-load metadata;
- focused local tests pass for env frozen-opponent action wiring and readiness
  gate metadata.

Proved with a real checkpoint:

- `mode=train`, `env_variant=source_state_fixed_opponent`, and
  `opponent_policy_kind=frozen_lightzero_checkpoint` call stock
  `train_muzero`;
- checkpoint resolution passes a strict file path/ref into the env config;
- no fallback policy is used if the checkpoint is bad.
- the same stock shape can run on an L4 GPU when the env manager is in-process
  (`env_manager_type=base`).

Evidence:

```text
run_id=stock-frozen-canary-source-state-s304-20260512
attempt_id=trainmuzero-frozen-denseiter32-tiny-wait-cpu-s304
summary_ref=training/lightzero-curvytron-visual-survival/stock-frozen-canary-source-state-s304-20260512/attempts/trainmuzero-frozen-denseiter32-tiny-wait-cpu-s304/train/summary.json
ok=true
called_train_muzero=true
trainer_entrypoint=lzero.entry.train_muzero
opponent_provider_load_ok=true
opponent_provider_load_strict=true

run_id=stock-frozen-gpu-base-canary-source-state-s304-20260512b
attempt_id=waited-gpu-base-single-env-frozen-ckpt-canary-20260512b
summary_ref=training/lightzero-curvytron-visual-survival/stock-frozen-gpu-base-canary-source-state-s304-20260512b/attempts/waited-gpu-base-single-env-frozen-ckpt-canary-20260512b/train/summary.json
ok=true
called_train_muzero=true
torch_cuda_available=true
device=NVIDIA L4
opponent_provider_load_ok=true
opponent_provider_load_strict=true
```

Important caveat: the GPU proof used `env_manager_type=base` with one
collector/evaluator env. That is a plumbing proof, not a throughput or learning
claim.

Likely cause: GPU compute sets the learner policy to CUDA and also derives
`opponent_use_cuda=true` for the env-owned frozen opponent. With
`env_manager_type=subprocess`, the frozen opponent then touches CUDA inside env
worker subprocesses. Avoid `subprocess + frozen checkpoint opponent + GPU`
until the opponent device can be decoupled from learner device.

Known GPU-safe tiny canary shape:

```text
--mode train
--compute gpu-l4-t4
--env-variant source_state_fixed_opponent
--opponent-policy-kind frozen_lightzero_checkpoint
--env-manager-type base
--collector-env-num 1
--evaluator-env-num 1
--num-simulations 8
--batch-size 16
--max-train-iter 32
--lightzero-eval-freq 0
```

## Minimal Implementation Candidate

1. Keep the training shape stock:

   ```text
   LightZero action -> ego player
   frozen provider action -> opponent
   one real CurvyTron tick
   stock GameSegment/GameBuffer
   ```

2. Run the next small learning canary proving:

   - enough iterations to produce an actual learning curve;
   - eval compares against the training opponent and at least one held-out
     opponent;
   - survival and sparse outcome curves are separate.

3. Only after that, design a recent-checkpoint refresh rule or checkpoint pool.

## Claim Label

Use:

```text
stock train_muzero ego-vs-recent-frozen-opponent
```

Do not use:

```text
true live two-seat self-play
```
