# Stock Path RND Reorientation

Status: active worldview update while optimizer work is in flight. This note is
about the original/stock-ish LightZero path, not the compact optimized trainer.

## Short Answer

Yes: the believable place to run RND is the original stock-ish LightZero path.
That is already where the current RND implementation is wired.

Do not add RND to the compact optimized trainer yet. The compact reward/RND
contract is intentionally extrinsic-only and rejects enabled exploration bonus
configs. That fence is useful because compact is still a speed R&D lane, not the
canonical learning path.

## What "Original Path + RND" Means

This is not byte-for-byte vanilla `train_muzero`. RND-enabled rows switch from:

```text
lzero.entry.train_muzero
```

to:

```text
lzero.entry.train_muzero_with_reward_model
```

while keeping the normal LightZero collector, replay, learner, checkpoint,
progress, eval, and GIF machinery. The trainer also patches LightZero's
`RNDRewardModel` to `CurvyRNDRewardModel` so the reward-model path can consume
Curvy source-state visual observations.

That is the right kind of deviation: it is stock LightZero reward-model
plumbing, not the compact owner/search/replay trainer.

## Current Code Shape

Key files:

- `src/curvyzero/training/exploration_bonus.py`
- `src/curvyzero/training/lightzero_config_builder.py`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `scripts/build_curvytron_rnd_blank_sweep_manifest.py`
- `src/curvyzero/training/compact_reward_rnd_contract.py`

Implemented modes:

| Mode | Entrypoint | Target reward | Role |
| --- | --- | --- | --- |
| `none` | `train_muzero` | extrinsic only | stock baseline |
| `rnd_meter_v0` | `train_muzero_with_reward_model` | unchanged | passive RND instrumentation control |
| `rnd_replay_target_v0` | `train_muzero_with_reward_model` | extrinsic plus intrinsic RND | positive experiment |

The RND adapter:

- extracts the latest gray64 policy frame from LightZero replay batches
- trains a predictor against a frozen target network
- writes `rnd_reward_model_metrics_latest.json`
- appends `rnd_reward_model_metrics.jsonl`
- tracks predictor/target hashes
- records whether positive weight changed target rewards
- exposes `state_dict` and `load_state_dict`

The stock trainer wrapper:

- chooses the LightZero entrypoint from the exploration-bonus spec
- writes RND metrics paths into `main_config.reward_model`
- patches LightZero `RNDRewardModel` to Curvy's adapter for the process
- validates required RND metrics when requested
- writes an RND metrics scan sidecar into the run artifacts

The compact contract:

- rejects enabled exploration bonus configs
- records `calls_train_muzero=false`
- records no RND update, reward-target, checkpoint-state, or promotion claim

## Why This Is More Believable Than Compact

For canonical learning experiments, the stock-ish path already owns:

- the current manifest-builder to grouped-submitter to Modal-trainer chain
- `source_state_fixed_opponent` as the controlled default environment lane
- stock collect/search with LightZero CTree
- LightZero checkpoint save/load shape
- background eval and GIF pollers
- tournament/checkpoint consumption path
- existing reward-axis configs
- old r18fresh/CZ26 comparability
- learner/replay/search semantics already used by prior training evidence

Compact currently owns speed evidence, not learning-quality evidence. It should
not carry RND until it proves normal-death correctness, stock checkpoint export,
resume/load, eval/GIF/tournament compatibility, and learning-quality parity.

Important caveat: this is believable fixed-opponent or frozen/mixture-opponent
training. It is not yet a trusted two-seat current-policy self-play claim.
Background GIFs are useful qualitative artifacts, but they are source-state
canvas captures rather than browser-pixel-fidelity proof.

Also keep non-stock collect/search experiments out of canonical learning
claims. `direct_ctree_gpu_latent` and non-LightZero CTree variants are optimizer
or profile lanes unless separately promoted.

## What Is Not Yet Proven

The stock-path RND implementation is believable enough to test, but not proven
enough to promote.

Open risks:

- the current local tests prove config selection, fake-entrypoint plumbing, and
  metrics validation, not a real LightZero collector/replay/learner RND pass
- the reward-model installation is monkey-patch based; if LightZero changes how
  it resolves `RNDRewardModel`, the Curvy adapter could be bypassed
- positive RND has not shown retained extrinsic policy quality
- prior RND sweeps were diagnostic, one-seed, blank-canvas, no-tournament runs
- high positive weights can be confounded with target support effects, because
  support metadata is adjusted for the bounded intrinsic term
- per-batch min/max RND normalization is a compatibility canary, not a proven
  global novelty scale
- RND replay buffer contents are not a durable promotion-grade resume contract
- `state_dict` exists, but end-to-end reward-model checkpoint/resume round-trip
  still needs explicit proof
- meter mode must behave like stock before positive rows are trusted

## Broad Sweep Rule

Given the available H100 budget, do not make RND a slow serial program. The
default posture is broad, embarrassingly parallel exploration with controls and
replicas. The meter/positive canaries below are launch-health sentinels; they
should either be embedded in the broad manifest or run as a same-hour preflight
with the broad sweep ready to go.

## Smallest Believable Health Gate

Before trusting a 45-row positive sweep, the cleanest health gate is:

1. Run a stock-path RND meter canary with normal learner batch size.
2. Require metrics on `rnd_meter_v0`.
3. Prove the entrypoint is `lzero.entry.train_muzero_with_reward_model`.
4. Prove predictor updates and target hashes stay frozen.
5. Prove target rewards are unchanged at weight `0.0`.
6. Produce at least one checkpoint, eval poller artifact, and GIF artifact.
7. Run matched `none` control with the same seed/horizon.

Then include or immediately run a tiny positive `rnd_replay_target_v0` canary:

1. Use a low weight such as `0.003` or `0.01`.
2. Require metrics.
3. Prove `last_target_reward_changed=true`.
4. Prove target reward delta is finite and nonzero.
5. Confirm support metadata is recorded so support saturation can be separated
   from exploration signal.

The prepared 45-row blank sweep should not be blocked on a long serial read. If
the meter/positive health rows are clean, keep the wide sweep moving. If low
positive weights win there, the next RND bridge should be a static exact-ref
opponent curriculum, not a live leaderboard/refresh loop.

## Interpretation Rule

An RND success in the original path means:

```text
stock-ish LightZero + Curvy RND deserves expansion
```

It does not mean:

```text
compact optimized trainer supports RND
```

and it does not mean:

```text
RND is a production reward
```

Promotion requires survival AUC/best/retention against both stock and meter
controls, then fixed-opponent transfer or later tournament exposure.
