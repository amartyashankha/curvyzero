# Remaining Environment Gap Catalog

Status: practical remaining-gap list
Date: 2026-05-09
Owner: working memory

## Short Status

One runtime is under hardening: `VectorMultiplayerEnv`. `CurvyTronSourceEnv`
and the source JS oracle are proof tools. Strict `VectorTrainerEnv1v1NoBonus` is
proof/profiling only, not the destination.
Restricted wrappers are temporary explicit profile configs; the reconstruction
path remains source-default CurvyTron behavior in `VectorMultiplayerEnv`.

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

Green now:

- 18 lifecycle fixtures have direct JS/Python parity. This includes the focused
  4P all-present all-dead warmdown/next-round fixture, the focused 4P survivor
  next-round fixture, the focused 3P present/non-present survivor-scoring
  fixture, and the focused 3P all-present multi-round match-end fixture.
- The forced immediate `BonusGameClear` clear fixture is green.
- The default multi-type bonus weight/type RNG proof is green.
- The live movement event trace
  `source_live_movement_event_trace_2p_no_bonus_multistep` is green.
- These are narrow source-fidelity proofs, not a production training
  environment claim.

Still open:

- The 2026-05-10 prioritized multiplayer target list lives in
  [multiplayer_env_gap_targets_2026-05-10.md](multiplayer_env_gap_targets_2026-05-10.md).
- Broader 4P match lifecycle beyond the focused all-dead and survivor
  next-round proofs.
- Broader present, non-present, and leave variants.
- Longer source-env vs JS rollouts beyond the promoted live movement event
  trace.
- Bonus caps, natural `BonusGameClear` probability/selection, broader
  probability behavior, and other effects.
- Observation, replay, terminal/final observation, and trainer-facing final
  info contracts.
- Fast runtime parity for the source-backed rules above. A metadata-only
  `VectorMultiplayerEnv` public surface exists, but it is not learned
  trainer observation, replay, or broad lifecycle parity.

## Current Green Slice

Lifecycle parity is promoted for exactly these 18 fixture shapes:

- Three 2P core lifecycle fixtures: warmup/start, next-round continuation, and
  one heading-rejection retry.
- Focused 3P first-round spawn order.
- Focused 3P warmup and delayed PrintManager start order/random calls.
- Focused 4P first-round spawn order/RNG labels.
- Focused 4P all-present all-dead warmdown/next-round continuation.
- Focused 4P survivor warmdown/next-round continuation.
- Focused 3P first-round present/absent `onRoundNew()`.
- Focused 3P present/absent survivor scoring through `round:end`.
- Focused 3P present/absent warmdown/next-round continuation.
- Focused 2P max-score match end.
- Focused all-present 3P `max_score: 2` match end.
- Focused 3P tie-at-max-score continuation.
- Focused 3P all-present multi-round match end.
- Focused 3P all-dead warmdown/next-round continuation.
- Focused 3P survivor scoring through `round:end`.
- Focused 3P survivor warmdown/next-round continuation.

Bonus parity is promoted for the narrow `BonusSelfSmall` catch/no-catch,
spawn/type/position RNG, default multi-type weight/type RNG, game-world retry,
and expiry/restore slices, plus one forced immediate `BonusGameClear` clear
slice. This does not prove catch/effects for newly selectable spawned types or
the natural `BonusGameClear` probability path.

## Remaining Gap Table

| Gap | What is known | Practical next move | Training blocker |
| --- | --- | --- | --- |
| Broader 4P lifecycle | 4P first-round spawn order/RNG, all-dead next-round, and survivor next-round are pinned narrowly. | Add one 4P lifecycle fixture only when it proves a rule not already covered by 2P/3P. Start with a broader match-end or present/alive shape. | yes |
| Present/non-present/leave variants | One 3P first-round present/absent case and one warmdown/next-round continuation are pinned. | Add focused variants for non-present after start, present/alive leave, leave during warmdown, and score/timer interactions. Keep each claim separate. | yes |
| Longer source-env vs JS rollouts | Source-env has narrow direct checks, local scout timing, and one promoted live movement event trace, but long behavior is not broadly compared. | Run longer no-bonus and bonus-disabled source-env vs original-JS traces with event/state checkpoints. | yes |
| Bonus caps/probability | Narrow `BonusSelfSmall`, default multi-type weight/type RNG, and forced `BonusGameClear` proofs are green. | Add cap behavior, natural `BonusGameClear` probability/selection, and probability changes by alive/present ratio. | indirect unless bonuses enabled |
| Other bonus effects | Radius restore and immediate clear are pinned narrowly. | Add one fixture per effect: speed, slow, inverse, straight angle, borderless, color, printing, radius collision beyond restore, stack math, expiry ordering, and death interactions. | indirect unless bonuses enabled |
| Observations and final obs | Debug/analytic packers exist, but trainer-facing contracts are not source-backed enough. | Define observation rows from promoted states, then add terminal info, final observation, legal masks, rewards, and padding behavior. | yes |
| Replay | Debug/sample chunks exist, not production replay. | Define chunk schema, episode ids, reset seed/source, done/terminated/truncated fields, final obs policy, event/state refs, manifests, and schema/rules hash rejection. | yes |
| Fast runtime parity | `vector_reset.py`, `vector_spawn.py`, and `vector_lifecycle.py` are reset/spawn/timer-boundary pieces, not full lifecycle. | Port one promoted source claim at a time into the fast path, compare against JS/source-env traces, and keep unsupported cases explicit. | yes |
| Self-play and LightZero | Actor bridge, Modal, Mctx, and adapter smokes are plumbing/runtime evidence. | Keep integration behind source-backed reset/step, observation/reward, replay, final obs, and policy row mapping contracts. | yes |
| Wire/pixels | Source state and source event order remain the authority. | Wait for stable state/event traces, then add one compressed wire fixture. Browser pixels come last. | no for training |

## Suggested Work Order

1. Keep the 18 lifecycle fixtures and forced `BonusGameClear` fixture as
   regression guards.
2. Add one broader lifecycle claim at a time: 4P, present/non-present, or leave.
3. Run longer source-env vs JS rollouts before promoting fast runtime behavior.
4. Expand bonuses in small source claims: caps/probability first, then one
   effect per fixture.
5. Define observation, replay, and final-observation contracts before claiming
   trainer readiness.
6. Move promoted source behavior into the fast runtime only after the source
   claim has an event/state trace.

## Promotion Rule

For every new gap closure:

1. Name the source claim in one sentence.
2. Pin original JS behavior first.
3. Promote Python/source-env parity only for that exact claim.
4. Record random calls, event order, and final snapshot expectations.
5. Add fast-runtime support only after the source claim is promoted.
6. Keep unsupported variants in this catalog.
