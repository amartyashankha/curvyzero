# CurvyTron Game Mechanics Gates - 2026-06-23

Status: source-fidelity checklist for new reward, RND, PPO/Puffer,
macro-action, planner, or MuZero branches. No jobs were launched from this note.

## Purpose

CurvyTron is not just a three-action toy. Any branch that changes environment
runtime, control cadence, observation, reward, planner rollout, or self-play
topology must preserve or explicitly relabel these game mechanics.

If a branch fails these gates, its result may still be a useful diagnostic, but
it is not source-faithful CurvyTron learning evidence.

## Required Gates

| Gate | What Must Hold | Why It Matters |
| --- | --- | --- |
| Action semantics | Action ids remain `0/1/2 = left/straight/right`; live players have all three legal actions; dead players have empty masks. | PPO, search, replay, and opponent policies all depend on stable action meaning. |
| Source timing | Physics remains 60 Hz or the branch explicitly declares a cadence change; macro-action/action-repeat rows report source frames requested and executed and stop early on terminal/cap. | Action-repeat can change reward scale, latency, and denominator. |
| Kinematics | Position/heading updates and elapsed-ms stepping match source scenarios. | A planner over curved motion is only meaningful if the curve dynamics are right. |
| Wall/border | Normal wall death, same-frame draw, survivor scoring, borderless wrapping, edge/corner behavior, and wrap-followup collision semantics hold. | Border logic is a major tactical constraint and bonus effect. |
| Trail/body collisions | Own-body latency, old-body metadata, same-frame point materialization, hit-owner attribution, head/head or body ordering, and death/score order hold. | Enclosure and cut strategies are trail-topology strategies. |
| Print/hole cadence | Delayed print start, print-to-hole and hole-to-print boundaries, random cadence, stop-on-death, and natural hole crossing hold. | Gaps are real strategic openings; tick-based approximations can lie. |
| Bonuses | Default bonus set, natural spawn RNG/order/retry/cap, catch thresholds, stack add/remove, death-before-catch, borderless restore, and game-clear behavior hold or are declared unsupported. | Bonuses change speed, radius, controls, border topology, and trails. |
| Observation and terminal surface | Stacked gray64 shape, raw render parity, final observation/action mask, terminal snapshots, and no render mutation hold. | MuZero, RND, replay, and eval all consume these surfaces. |
| Reward accounting | Sparse outcome, dense survival, same-step bonus reward, scaled terminal outcome, trainer reward, source reward map, and final reward map remain separate. | RND or shaping must not blur the evidence ledger. |
| Opponent contracts | Fixed checkpoint, blank-canvas noop, immortal diagnostics, proactive wall-avoidant, and mixture-on-reset modes are labeled and not called true self-play. | Current Wave A is fixed-opponent/mixture training, not current-policy competitive self-play. |
| Seat perspective | Learner seat, controlled-player render perspective, reward player, opponent mask, and telemetry work for player 0 and player 1. | Two-player PPO/Puffer/self-play can silently train one seat to help the other. |
| Joint-action caveat | The 9-action centralized joint-action env remains diagnostic unless a true two-seat competitive self-play contract is added. | Centralized joint control is not a proof of simultaneous-game self-play. |
| Lifecycle | Round warmup/warmdown, next-round/match-end, present/absent players, and source lifecycle fixtures are covered before claiming multiplayer fidelity. | Mechanics canaries are not enough for full game-loop promotion. |

## Branch-Specific Checks

Macro-action or action-repeat branches must:

- replay source frames internally, not jump to macro endpoints
- record requested and executed source ticks
- report reward accumulation over substeps
- label throughput and quality denominators separately from raw-tick rows

PPO/Puffer or self-play branches must:

- declare behavior-policy contract
- declare reward and value perspective
- record mask/logprob convention
- report entropy/action collapse by player slot
- preserve opponent pool or frozen checkpoint semantics for self-play claims

Planner branches must:

- say whether the planner uses public observation, source state, or privileged
  hidden state
- report policy-only versus planned-action lift on fixed states
- report latency distribution and source-frame replay budget
- use imitation/reanalysis/search targets rather than pretending planner actions
  are vanilla PPO samples

RND branches must:

- keep RND metrics separate from trainer reward components
- prove `rnd_meter_v0` target rewards are unchanged
- prove positive rows change target rewards only through the declared intrinsic
  path
- check novelty does not reward cosmetic visual change, bonus chasing, or dying
  in unusual ways without extrinsic survival gain

## Minimal Local Gate

Use this as the minimum mechanics/fidelity slice before promoting a new
environment, control cadence, planner rollout model, or PPO/Puffer substrate:

```bash
uv run pytest \
  tests/test_env_*.py \
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py \
  tests/test_curvyzero_source_state_visual_turn_commit_lightzero_env.py \
  tests/test_source_state_visual_survival_learner_seat_regression.py \
  tests/test_lightzero_source_state_wrapper_product_fidelity.py
```

Broaden beyond this for lifecycle or tournament claims.

## References

- Source reference summary: `docs/sources/curvytron_reference.md`
- Source mining notes: `docs/research/curvytron_reference_notes.md`
- Runtime constants and bonus effects: `src/curvyzero/env/vector_runtime.py`
- Trainer action/observation/reward contract:
  `src/curvyzero/env/trainer_contract.py`
- Source-state LightZero wrapper:
  `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- Scenario fixtures: `scenarios/environment/`
