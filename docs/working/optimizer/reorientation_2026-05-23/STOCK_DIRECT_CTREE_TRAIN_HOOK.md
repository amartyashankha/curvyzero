# Stock Direct CTree Train Hook

Date: 2026-05-23

## Plain State

We now have a narrow real training-loop patch:

```text
stock LightZero train_muzero
-> MuZeroPolicy._forward_collect
-> direct_ctree_gpu_latent hook
-> stock collector/replay/target/learner continue unchanged
```

This is not compact ownership. It is a practical search-path patch inside the
trusted stock trainer.

## What Changed

- `mode="train"` may request `collect_search_backend="direct_ctree_gpu_latent"`.
- Train mode still requires GPU compute.
- Train mode uses only LightZero CTree for this hook.
- `flat_a3` remains profile-only.
- Fallback is fail-closed in train mode. If the hook cannot actually run, the
  run raises or the summary becomes not-ok.
- The summary now includes `collect_search_backend_proof`, and compact output
  mirrors the same proof under `search_backend_proof`.
- Train proof now requires direct calls, output rows, observed requested CTree
  backend, recurrent inference calls, and model-output D2H bytes when
  simulations are positive.
- `phase_profile.timers_sec.train_muzero_wall_sec` is recorded for train canaries
  too, not only profile rows.
- The local tests now cover the success path, missing proof, CPU reject,
  dry-mode reject, non-LightZero CTree reject, hidden fallback reject, real
  trainer-boundary hook consumption, and direct backend plus required RND
  metrics.

## What This Can Speed Up

This can speed up the stock collect/search path while leaving replay and learner
semantics alone. Based on previous matched profile rows, the realistic ceiling is
roughly `1.2x-1.5x` for the stock training loop. It is not the 5x-10x compact
architecture.

## Required Proof Before Coach Recommendation

A train run with this hook must show:

- `called_train_muzero=true`;
- `command.collect_search_backend=direct_ctree_gpu_latent`;
- `command.collect_search_backend_fallback_policy=fail_closed_when_non_stock`;
- `collect_search_backend_proof.direct_ctree_gpu_latent_calls > 0`;
- `collect_search_backend_proof.fallback_calls == 0`;
- `collect_search_backend_proof.output_rows > 0`;
- `collect_search_backend_proof.recurrent_inference_calls > 0` when
  `num_simulations > 0`;
- `collect_search_backend_proof.model_output_d2h_bytes > 0` when
  `num_simulations > 0`;
- `collect_search_backend_proof.observed_collect_search_backends` contains
  `direct_ctree_gpu_latent`;
- `collect_search_backend_proof.observed_collect_search_ctree_backends`
  contains `lightzero`;
- no `flat_a3` in train mode;
- normal stock replay/target/learner path remains in place.
- current RND/reward settings are either enabled exactly as intended or named
  as off for a controlled ablation.

## Immediate Canary

Run a capped stock-trainer canary before any overnight recommendation:

```text
mode=train
compute=gpu-l4-t4 or gpu-h100-cpu40
collect_search_backend=direct_ctree_gpu_latent
collect_search_ctree_backend=lightzero
RND/reward/noise/death/checkpoint knobs copied from the current Coach baseline
small max_train_iter/max_env_step
enough warmup to avoid reading cold GPU timing as steady state
```

Kill it if any of these happen:

- fallback calls are nonzero;
- direct calls are zero;
- `train_muzero` was not called;
- replay/RND/checkpoint fields are missing for a run that claims to be a train
  canary;
- speed metadata does not name its denominator.

## Canary Results

2026-05-23 local/Modal proof status:

| row | compute | shape | result | read |
| --- | --- | --- | ---: | --- |
| `opt-lane1-direct-rnd-canary-20260523a` | L4/T4 | C8/N8/sim8, 16 learner calls | `ok=true`, `35.15` steps/sec | direct hook works end-to-end with required RND metrics |
| `opt-lane1-stock-rnd-canary-20260523a` | L4/T4 | same | `ok=true`, `33.12` steps/sec | tiny smoke showed `1.06x` direct/stock |
| `opt-lane1-h100-stock-rnd-warm-20260523a` | H100 | C256/N256/sim8, 40 learner calls | `ok=true`, `121.90` steps/sec | warmer stock baseline |
| `opt-lane1-h100-direct-rnd-warm-20260523a` | H100 | same | `ok=true`, `117.83` steps/sec | direct hook proof passed but wall speed was `0.97x` stock |

Direct H100 proof fields:

```text
direct_ctree_gpu_latent_calls = 512
fallback_calls = 0
output_rows = 38640
observed backend = direct_ctree_gpu_latent
observed ctree = lightzero
recurrent_inference_calls = 4096
model_output_d2h_bytes = 6182400
RND required = true
RND collect/train/estimate all positive
```

Plain read: Lane 1 is implemented and real, but this A/B does not justify
recommending it as the next overnight speed setting. In the warmer H100 run,
direct reduces the collect/search sub-bucket but does not reduce total
`train_muzero_wall`. The env/collection wall dominates, and exact trajectory
timing is not controlled tightly enough to treat the sub-bucket win as a
training-speed win.

## What Is Still Not Done

The larger architecture bet is still separate:

```text
CompactRootBatchV1
-> real two-phase fixed-shape MCTS owner
-> action-only hot path
-> delayed replay payload flush
-> compact replay/sample bridge
```

That path is still profile-side. The current fixed-shape owner is first-legal
and not a real MCTS replacement.
