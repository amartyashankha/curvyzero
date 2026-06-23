# Current Code Dataflow Map

Date: 2026-05-21

Purpose: one plain map of what runs where. This separates actual Coach
training from stock full-loop profiles and profile-only optimizer probes.

## One-Sentence Truth

Actual Coach training still enters stock LightZero `train_muzero` through
scalar CurvyTron env rows. The fast H100/direct-CTree numbers are profile-only
evidence that a better boundary may exist; they are not promoted training.

## Production Coach Training Path

Entrypoint:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- local Modal entrypoint is around line `13107`;
- actual train path loads `lzero.entry`, selects `train_muzero`, and calls it
  around line `5725`.

Config:

- `src/curvyzero/training/lightzero_config_builder.py`;
- default CurvyTron env variant is `source_state_fixed_opponent`;
- config patches stock Atari MuZero-style settings such as `policy.type`,
  `env_manager.type`, `collector_env_num`, `num_simulations`, and learner
  `batch_size`.

Plain flow:

```text
Modal launcher
-> build CurvyTron LightZero config
-> register CurvyTron env
-> call lzero.entry.train_muzero
-> stock collector/search/replay/learner
-> checkpoint/eval/GIF sidecars as configured
```

GPU meaning in this path:

```text
compute=gpu-* means LightZero policy/model/search/learner CUDA.
It does not mean batched CurvyTron observation rendering on GPU.
```

## Stock Env Scalarization

The stock env wrapper is scalar at the source:

- `_new_env()` creates `VectorMultiplayerEnv(batch_size=1, player_count=2)` in
  `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
  around line `1036`.
- Each step builds one joint action, reads scalar done/reward bookkeeping, and
  returns one local timestep.
- `_lightzero_observation()` copies the stack and builds NumPy action mask,
  scalar `to_play=-1`, and scalar timestep metadata around line `1227`.
- `_update_stack()` normalizes/rolls host stacks around line `1256`.
- the registered env converts the local row to `BaseEnvTimestep` around line
  `2678`.

Plain flow:

```text
one LightZero env row
-> one CurvyTron VectorMultiplayerEnv(batch_size=1)
-> host NumPy [4,64,64] observation
-> BaseEnvTimestep
-> stock LightZero env manager
```

This is why actual production training does not get the giant profile-only
batch-resident speed by default.

## Stock Full-Loop Profile Path

The stock profile mode can still call `train_muzero`, but it can install
profile-only hooks and profile-only env managers.

Use this path when the question is:

```text
What happens to the real stock LightZero loop if a controlled profile setting
changes?
```

Examples:

- CPU oracle vs batched profile manager vs zero observation;
- no-RND vs RND meter;
- no-death speed rows vs normal-death guardrails;
- collector/env-manager topology.

Currency:

```text
env steps/sec or checkpoint/profile iteration timing
```

Not currency:

```text
roots/sec from a standalone boundary probe
```

## Profile-Only Hybrid Boundary Path

Entrypoint:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`;
- local profile CLI around line `6962`;
- profile-only metadata around line `2313` says
  `profile_only=True`, `calls_train_muzero=False`, and
  `trainer_defaults_changed=False`.

Plain flow:

```text
CurvyTron batch [B,2]
-> compact state
-> profile GPU renderer / batched stack
-> [B,2,4,64,64] uint8 stack
-> optional scalar LightZero edge
-> profile-only consumer
```

This path is allowed to answer architecture questions. It is not allowed to
become Coach launch advice until it reconnects to training and passes semantic
gates.

## Public LightZero Consumer Probe

This probe feeds the pre-scalar `[B,2,4,64,64]` stack to actual
`MuZeroPolicy.collect_mode.forward`.

Important code:

- it builds a batched root tensor from the pre-scalar stack;
- it passes `ready_env_id`, `action_mask`, and `to_play=-1`;
- it calls `policy.collect_mode.forward(...)` around line `4645`.

What it told us:

```text
The neural net root pass is fast.
The public collect/search/output wrapper is much slower than the model-only
path.
```

## Direct CTree Arrays Probe

This is the current active optimizer speed probe.

Plain flow:

```text
pre-scalar uint8 [B,2,4,64,64]
-> real LightZero model initial_inference
-> real LightZero CTree MCTS search
-> compact arrays out
```

Important code:

- direct probe function starts around
  `source_state_batched_observation_boundary_profile.py:4958`;
- model root pass happens around line `5000`;
- latent roots and policy logits are copied to CPU NumPy around lines
  `5020-5021`;
- CTree roots/search happen around line `5036`.

What it told us:

```text
Removing public collect_mode.forward while keeping real model + real CTree MCTS
is about 1.8x faster than the stock facade in the latest P2 refresh.
```

What it did not prove:

```text
It did not prove training speed.
It did not prove full semantic parity.
It did not prove a 5x-10x production path by itself.
```

## RND Path

RND is a separate axis, not a renderer or direct-CTree claim.

Important code:

- `src/curvyzero/training/exploration_bonus.py`;
- RND predictor/target can move to a configured device;
- training and estimate tensors move through `.to(self.device)`;
- estimate output comes back to CPU NumPy.

Policy:

```text
Measure RND as no-RND vs rnd_meter_v0 vs positive-RND separately.
Do not fold RND rows into observation, renderer, or MCTS boundary claims.
```

## Current Bottleneck Model

Most likely current walls:

1. Scalar env-manager boundary in production stock training.
2. Public LightZero collect/search/output wrapper.
3. Direct CTree CPU/NumPy boundary and root/search prep.
4. Host observation/stack movement when the stock scalar env path is used.
5. RND overhead when enabled.

Not the main current wall:

```text
single-frame rendering by itself
```

That does not mean renderer work was useless. It means the wall moved after
the renderer/env fixes and after the denominator changed to full loop/search.

## Promotion Rules

Before direct CTree or any compact batch boundary can affect Coach training:

1. P0 forced cases must pass: legal masks, illegal visit mass, single legal
   action, masked preference, clear preference, `to_play=-1`, output schema,
   and target-row compatibility.
2. Add the missing forced cases: support transform, no-noise eval mode, and
   root-noise collect mode.
3. P1 statistical stock-facade vs direct-CTree comparison must pass across many
   roots/seeds.
4. P2 profile rows must stay favorable on current code.
5. A matched full-loop profile must show that the boundary matters once it is
   connected to stock `train_muzero` or an explicitly documented replacement.

Until then:

```text
direct_ctree_arrays is profile-only.
```
