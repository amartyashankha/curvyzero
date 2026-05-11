# Aggressive Environment Gap Execution Packet

Status: docs-only execution packet
Date: 2026-05-09
Owner: environment gap wave

This is the short execution map for moving faster without inventing proof.

## Current Truth

- Full training-ready CurvyTron environment: not done.
- Promoted source lifecycle fixtures: 28 pinned lifecycle fixtures including
  `source_lifecycle_spawn_rng_4p_next_round`,
  `source_lifecycle_survivor_score_4p_next_round`,
  `source_lifecycle_present_absent_3p_survivor_score_round_end`, and
  `source_lifecycle_multi_round_match_end_3p` when focused tests pass.
- `2P` means two players. `3P` means three players. `4P` means four players.
- `vector_spawn.py` covers first-round spawn facts for 2P, 3P, and 4P. It is
  not full lifecycle.
- `vector_reset.py` is only a reset boundary.
- `vector_lifecycle.py` is only reset plus first-round spawn composition, with
  optional 1v1 delayed-start metadata while still `full_lifecycle=false`.
- JS reuse now has a persistent scalar Node worker with Python wrapper/tests.
  It is an oracle/backstop, not the final fast batched training env.
- Speed, Modal, JAX, Mctx, and actor-bridge work are runtime or shape evidence
  until they run the real env loop with real reset, observation, reward, and
  replay contracts.

## Why 3P Is Not Proven By 2P

2P is the smallest competitive case: one opponent, fewer scoring branches, and
fewer spawn/RNG branches.

3P is not just "2P plus one more player." More players change source-visible
order and branches:

- Spawn order is reverse player order, so one extra player changes RNG call
  order.
- A player can be absent while other players still spawn.
- Absent players can affect death lists and round bookkeeping.
- RNG calls can be skipped for absent players.
- Scoring, winner selection, all-dead cases, tie-at-max continuation, and
  multi-round matches have more branches.
- Timer and PrintManager behavior can interact with more avatars.

Rule: do not infer 3P or 4P lifecycle from 2P fixtures. Promote only the exact
player-count shape that a JS/Python source fixture proves.

## Blockers In Priority Order

| Priority | Blocker | Current proof | Missing |
| --- | --- | --- | --- |
| P0 | Source lifecycle beyond the pinned slice | 28 pinned lifecycle fixtures including `source_lifecycle_spawn_rng_4p_next_round`, `source_lifecycle_survivor_score_4p_next_round`, `source_lifecycle_present_absent_3p_survivor_score_round_end`, and `source_lifecycle_multi_round_match_end_3p` are promoted when focused tests pass, including focused 4P all-present all-dead warmdown/next-round, focused 4P survivor next-round, focused 3P present/absent survivor scoring, focused 3P present/absent warmdown/next-round, focused 3P all-dead warmdown/next-round, focused 3P survivor-scoring `round:end`, focused 3P survivor warmdown/next-round, focused 3P match-end fixtures, focused 3P tie-at-max continuation, and focused 3P all-present multi-round match-end. | Broader 4P match lifecycle and broader present/non-present lifecycle beyond the focused current cases. |
| P0 | Reset, timers, autoreset, final observation | `vector_reset.py` snapshots/copies selected rows and stamps reset metadata. | Natural reset templates, timer scheduling, public autoreset order, terminal snapshot policy, final observation preservation, horizon truncation, and row-local seed history. |
| P0 | Full vector lifecycle | `vector_spawn.py` covers first-round spawn; `vector_lifecycle.py` composes reset plus spawn and can stamp optional 1v1 delayed-start metadata. | Timer advancement, `game:start`, PrintManager start, warmdown, next-round, scoring, terminal rows, rewards, final observations, replay-safe events, and world-body insertion. |
| P0 | Trainer observation/reward/info | Debug packers and a first trainer contract exist. | Source-backed observation fixtures, legal masks for live/dead/terminal/padded rows, sparse reward wired to terminal rows, done/terminated/truncated info, and final reward maps. |
| P0 | Production replay | Debug/sample chunks exist. | Writer/reader for production shards, episode id, reset seed/source, done/terminated/truncated, final-observation policy, event/state refs, manifest, compaction, and schema/rules hash rejection. |
| P1 | Row-local RNG | Source fixtures label selected lifecycle calls; vector rows carry narrow tape metadata. | Production RNG arrays, cursor/exhaustion rules, call-site labels, reset integration, PrintManager integration, and bonus integration. |
| P1 | Bonuses | Narrow active `BonusSelfSmall` catch/no-catch/death-order, natural one-type spawn/type/position RNG, one default multi-type weight/type RNG proof selecting `BonusAllColor`, one game-world spawn retry, one `BonusSelfSmall` timed expiry/restore slice, and one forced `BonusGameClear` immediate clear slice are source-promoted. | Caps, effects for spawned multi-type bonuses, speed/radius/inverse effects, natural `BonusGameClear` probability/selection, borderless expiry, broader stack math/expiry ordering, color/property events, vector/runtime support, and broader death interactions. |
| P1 | B>1 and mixed player counts | Fixture-backed fixed-P speed slices exist. | Row-local timer/RNG independence, fixed-P grouping or padded mixed-P policy, overflow to truncation, and no-event/debug-event training policy. |
| P1 | Real actor loop and self-play speed | Fixture-cycled bridge, synthetic search, and debug replay staging exist. | Real reset/autoreset, real obs/reward, real policy/search boundary, replay chunks, completed games/min, and p95/p99 action latency. |
| P2 | LightZero adapter | Local no-train smoke is plumbing. | Real registration, reset/step semantics, observation dicts, masks, scalar reward, done/truncated info, final observation, and eval return through the real runtime. |
| P2 | Persistent JS env option | Long-lived Node worker exists for scalar reset/step/snapshot with stable source VM and Python wrapper/tests. | Throughput decision, trainer adapter, and batched/runtime strategy. |
| P3 | Wire and browser pixels | Source state/event work is the authority. | Compressed wire event fixture first, browser pixels last. |

## Parallel Work Packages

Plan in parallel. Serialize edits to shared runners, vector state, and training
interfaces.

| Package | Owner | Files owned | Depends on | Done when |
| --- | --- | --- | --- | --- |
| A. 3P lifecycle continuation | Source lifecycle owner | New `scenarios/environment/source_lifecycle_*_3p*.json`; `tools/reference_oracle/lifecycle_oracle.js`; `src/curvyzero/fidelity/source_runners.py`; `tests/test_source_lifecycle_runner.py` | Current 16-fixture baseline including `source_lifecycle_spawn_rng_4p_next_round` and `source_lifecycle_multi_round_match_end_3p`. | One broader present/non-present variant has JS oracle output and Python parity. |
| B. Present/non-present lifecycle | Source lifecycle owner | New present/non-present scenario id; same runner/test files as A | The exact focused next-round continuation is now pinned; keep broader variants separate. | One additional focused timer, score, or broader next-round present/non-present rule is promoted. |
| C. Match-end stress | Source lifecycle owner | `source_lifecycle_multi_round_match_end_3p.json`; same runner/test files as A | Focused all-present multi-round match-end is now pinned; keep broader variants separate. | Done for one all-present 3P path; next match rule starts as a new source claim. |
| D. 4P lifecycle scout | Source lifecycle owner | New `source_lifecycle_*_4p*.json`; same runner/test files as A | Only start when the rule is broader than first-round spawn. | One 4P lifecycle rule is promoted or explicitly parked. |
| E. Row-local RNG contract | RNG owner | `docs/working/environment/vector_state_schema.md`; `docs/working/environment/reset_timer_autoreset_plan_2026-05-09.md`; later vector RNG tests | Source call labels from lifecycle and bonus fixtures. | Tape/state/cursor, exhaustion, call labels, reset behavior, and replay metadata are defined. |
| F. Reset/timer/autoreset | Reset owner | `src/curvyzero/env/vector_reset.py`; reset/timer docs; reset tests | Lifecycle source claims, RNG contract, replay terminal policy. | Public reset/autoreset tests preserve terminal transition first, then reset done rows with new episode metadata. |
| G. Vector lifecycle | Vector owner | `src/curvyzero/env/vector_spawn.py`; `src/curvyzero/env/vector_lifecycle.py`; `src/curvyzero/env/trace_compare.py`; vector comparator tests | A, B, C, or D must promote a source claim first. | One promoted lifecycle claim is reproduced in vector rows and unsupported cases stay explicit. |
| H. Trainer obs/reward/info | Trainer interface owner | `docs/working/environment/trainer_observation_reward_contract_v0_2026-05-09.md`; `src/curvyzero/env/trainer_observation.py`; `src/curvyzero/env/trainer_contract.py`; trainer tests | Trusted source states and terminal policy from F. | Source-backed obs fixtures, legal masks, sparse reward, terminal info, and final reward maps are tested. |
| I. Production replay | Replay owner | `docs/working/environment/replay_terminal_seed_contract_2026-05-09.md`; replay writer/reader modules; replay tests | F for terminal/reset order; H for obs/reward schemas. | Production shard tests reject schema/rules mismatches and preserve episode/reset/final-observation metadata. |
| J. Bonuses | Bonus source owner | New `source_bonus_*` scenarios; source runner/test updates; RNG docs | E before vector/reset integration. | Continue from the promoted active `BonusSelfSmall` catch/no-catch/death-order, one-type spawn RNG, default multi-type weight/type RNG, game-world retry, and expiry/restore slices; split caps/probability, other effects, other bonus types, and vector/runtime work into separate claims. |
| K. B>1 and speed lane | Performance owner | Batch benchmark scripts; actor bridge scripts; `docs/working/environment/selfplay_speed_lane_2026-05-09.md` | Current fixture slices now; F, H, I, and real policy/search for production claims. | Reports completed-games proxy, p95/p99 action latency, env/obs/search/replay shares, and exact fixture coverage. |
| L. LightZero adapter | Training boundary owner | LightZero smoke/adapter files; policy row mapping tests | F, H, I, and policy/search boundary. | Real runtime reset/step smoke passes with masks, reward, done/truncated, terminal info, and final observation. |
| M. Persistent JS env probe | JS reuse owner | `tools/js_reuse_probe/`; `src/curvyzero/fidelity/js_reuse_probe.py`; probe docs/tests | Independent prototype unless chosen as the env path. | Long-lived worker supports reset/step/advance/snapshot with deterministic RNG/timers. |
| N. Wire/pixels | Wire/render owner | Wire fixture docs/tests; later browser checks | Stable state/event traces. | One compressed wire fixture lands; browser pixels stay last. |

## First 48 Hours

1. Launch A, B, C, E, F, H, I, J, and K in parallel.
2. Keep D optional until someone names a 4P rule beyond first-round spawn.
3. Let G wait for the next source lifecycle claim, except for unsupported-case
   guards and API shape.
4. Let L prepare interfaces only. Do not call it real training.
5. Let M harden only the scalar persistent-worker/oracle question.
6. Keep N parked unless state/event traces are stable.

## Shared File Locks

Only one owner edits each of these at a time:

- `tools/reference_oracle/scenario_runner.js`
- `tools/reference_oracle/lifecycle_oracle.js`
- `src/curvyzero/fidelity/source_runners.py`
- `src/curvyzero/env/trace_compare.py`
- `src/curvyzero/env/vector_spawn.py`
- `src/curvyzero/env/vector_reset.py`
- `src/curvyzero/env/vector_lifecycle.py`
- production vector state/event-row modules
- production trainer observation/reward modules
- production replay modules

Usually safe in parallel:

- New scenario files with unique ids.
- Docs under `docs/working/environment/`.
- Expected event tables for one named claim.
- Artifact inspection outside the repo.
- Regression runs after shared edits settle.

## Promotion Rule

1. Name the source claim in one sentence.
2. List source files or source-map notes read.
3. Add one deterministic fixture id.
4. Pin original JS behavior first.
5. Promote Python parity only for that same claim.
6. Widen vector support only after source parity lands.
7. Record unsupported cases in plain language.
8. Keep speed, Modal, JS reuse, and self-play labels separate from fidelity
   labels.

## No-Proof Zone

Do not write these claims until the matching proof exists:

- "Full CurvyTron training env is ready."
- "3P lifecycle is covered because 2P lifecycle is covered."
- "4P lifecycle is covered because 4P first-round spawn is covered."
- "`vector_spawn.py` is full lifecycle."
- "`vector_reset.py` is autoreset."
- "`vector_lifecycle.py` is production lifecycle."
- "JS reuse worker is the final fast batched training env."
- "Modal/JAX/Mctx speed proves source fidelity."
