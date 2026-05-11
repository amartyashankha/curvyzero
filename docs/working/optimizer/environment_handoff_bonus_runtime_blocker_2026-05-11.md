# Environment Handoff: Natural Bonus Runtime Blocker

Date: 2026-05-11

Status: historical/resolved blocker note. Environment has since implemented the
missing source-default bonus effects. Optimizer then found and fixed one
optimizer-side capacity issue for long no-death profiles: the vector natural
bonus placement retry slab was raised from `16` to `256` because scalar source
placement retries until it finds a free spot.

Current profile status:

```text
run_id=opt-source-state-nodeath-profile-c16-sim16-s1181-posfix
attempt_id=profile-nodeath-c16-sim16-steps240-posfix
ok=true
learner_train_calls=5
replay_sample_calls=5
collector_collect_calls=1

run_id=opt-source-state-nodeath-profile-c32-sim16-s1181-posfix
attempt_id=profile-nodeath-c32-sim16-steps240-posfix
ok=true
learner_train_calls=5
replay_sample_calls=5
collector_collect_calls=1
```

Do not use this document as the current blocker list. It is kept as the repro
history for the old `BonusAllColor` / `BonusSelfMaster` gap.

## Copyable Handoff

Historical handoff: Optimizer reran the CurvyTron source-state visual LightZero profile with
profile-only death suppression:

```text
env_variant=source_state_fixed_opponent
underlying_env_class=VectorMultiplayerEnv
runtime_env_impl_id=curvyzero_vector_multiplayer_metadata_natural_bonus_spawn/v0
visual_surface=source_state_visual_tensor
death_mode=profile_no_death
disable_death_for_profile=true
compute=gpu-l4-t4
collector_env_num=16
n_episode=16
num_simulations=16
source_max_steps=1000
decision_ms=300
```

The route is correct, but the run fails before learner/replay because the fast
runtime can naturally spawn a bonus it cannot apply when caught:

```text
RuntimeError: unsupported caught bonus type code 11;
only table-backed runtime bonus effects are supported
```

Type code `11` is `BonusAllColor`. It is in the source default bonus set, and
the current source-state LightZero wrapper constructs `VectorMultiplayerEnv`
with `natural_bonus_spawn=True`, so this is not a random bad config. It is the
current source-default runtime surface reaching an unimplemented catch effect.

Please do not "fix" this by narrowing `natural_bonus_type_codes` unless the
intent is to publish a different partial surface. For the current source-default
surface, Environment should either:

1. implement the missing fast-runtime catch/effect/expiry semantics for
   `BonusAllColor` and the nearby source-default gap `BonusSelfMaster`; or
2. explicitly downgrade the environment contract and say the current natural
   source-default runtime surface is still partial and cannot be used for the
   full long-survival profile.

Minimum tests before Optimizer reruns the profile:

- natural spawn/catch for `BonusAllColor`;
- `BonusAllColor` expiry/restore of colors;
- `BonusSelfMaster` catch/expiry;
- a no-death source-state profile smoke that runs long enough to catch naturally
  spawned source-default bonuses without unsupported-type failure.

After that, Optimizer will rerun the same c16/sim16 no-death profile and report
the real bottleneck breakdown.

## Evidence

Two profile reruns failed the same way:

```text
run_id=opt-source-state-nodeath-profile-c8-sim8-s1160
attempt_id=profile-nodeath-c8-sim8-steps96
error=unsupported caught bonus type code 11

run_id=opt-source-state-nodeath-profile-c16-sim16-s1161
attempt_id=profile-nodeath-c16-sim16-steps1000
error=unsupported caught bonus type code 11
```

The longer c16 run had these useful pre-failure facts:

```text
ok=false
called_train_muzero=true
training_readiness_gate.ok=true
learner_train_calls=0
replay_sample_calls=0
collector_collect_calls=0
mcts_search_calls=140
mcts_search_simulation_budget_sum=2240
env_step_calls=139
max_gpu_util_percent=12
```

Useful partial timing, not a valid full-loop profile:

```text
train_muzero wall:              11.35s
policy_forward_eval:             6.98s
MCTS search:                     5.55s
model recurrent inference:       3.47s
env.step:                        0.386s
vector runtime step_many:        0.166s
render gray64:                   0.042s
stack update:                    0.050s
LightZero obs pack:              0.054s
```

The timing above is only evidence for route, pre-failure work, and the blocker.
It is not the requested steady-state training-loop profile because learner and
replay were never reached.

## Code Pointers

- `src/curvyzero/env/vector_runtime.py`: `BONUS_TYPE_ALL_COLOR = 11`.
- `src/curvyzero/env/vector_runtime.py`: `_catch_bonus_batched(...)` raises when
  a caught bonus is not in `BONUS_RUNTIME_EFFECT_BY_TYPE`.
- `src/curvyzero/env/vector_runtime.py`: `BONUS_RUNTIME_EFFECT_BY_TYPE` currently
  omits `BONUS_TYPE_SELF_MASTER` and `BONUS_TYPE_ALL_COLOR`.
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`:
  the source-state visual wrapper constructs `VectorMultiplayerEnv` with
  `natural_bonus_spawn=True`.
- `src/curvyzero/env/source_env.py`: scalar source reconstruction already models
  `BonusAllColor` as an all-alive-avatar color rotation with expiry restore, and
  models `BonusSelfMaster` as self invincibility plus printing stop.
