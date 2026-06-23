# Subagent Next Falsification Grid

Date: 2026-05-21

Scope: experiment-design critique only. Profile rows only. Do not edit
production code, change trainer defaults, launch live training, or promote a
Coach recommendation from these rows.

## Questions The Grid Must Answer

1. Does resident batching still have headroom after adding real
   policy/search/replay-shaped pressure, or was the `~13.8k` H100 roots/s
   canary mostly a synthetic stack-probe win?
2. How much of the resident win survives the scalar LightZero edge?
   Current canary read: H100 B512/A16/sim8 scalar-off `~13.8k` roots/s,
   scalar-on `~6.5k`; L4 scalar-off `~9.0k`, scalar-on `~4.2k`.
3. Is normal death/autoreset semantics a correctness blocker before any stock
   boundary integration?
4. Does RND/latest-frame accounting add a new wall, or only a measurable edge
   tax?
5. Is the next investment a real stock-boundary/custom-collector integration,
   or should resident batching stay in the profile lane?

## Smallest First Rows

Run these in order. Stop early on invalid telemetry, hidden fallback, or a
clear throughput fail.

| row | compute | shape | death | consumer | scalar edge | purpose |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | H100 | B512/A16/sim8 | no-death | existing resident stack probe | off | repeat the known `~13.8k` harness anchor |
| 2 | H100 | B512/A16/sim8 | no-death | existing resident stack probe | on | repeat the known `~6.5k` scalar-tax anchor |
| 3 | H100 | B512/A16/sim8 | no-death | real policy/search/replay-shaped profile consumer | off | core falsifier: does residency survive real pressure? |
| 4 | H100 | B512/A16/sim8 | no-death | same as row 3 | on | price the unavoidable stock-boundary scalar edge |
| 5 | L4/T4 | B512/A16/sim8 | no-death | same as row 3 | off | hardware sanity: does the cheaper GPU preserve the win? |
| 6 | L4/T4 | B512/A16/sim8 | no-death | same as row 3 | on | hardware scalar-edge sanity |
| 7 | H100 | B256/A8/sim8 | normal death | same as row 3 | off | terminal/autoreset/final-observation semantic gate |
| 8 | H100 | B256/A8/sim8 | normal death | same as row 3 | on | scalar terminal payload gate |

Do not add B1024, C768, RND, or live `train_muzero` rows until rows 1-8 are
valid and interpreted.

## Death Mode Rules

Use no-death for speed/Amdahl rows. Rows 1-6 should be long-survival no-death
so trail growth, stack update, policy/search pressure, and scalar materializing
are measured without reset churn hiding the wall.

Use normal death for correctness rows. Rows 7-8 only need to prove terminal
semantics: partial live masks, terminal `final_observation`, autoreset order,
row/player mapping, and scalar terminal timestep shape. Do not use normal-death
throughput as the main speed claim unless it is repeated after the semantic
gate passes.

RND rows come later. First RND row should be no-death `rnd_meter_v0` at the
best H100 shape to price latest-frame/update overhead. Then run one normal-death
RND row only to verify terminal latest-frame and reset accounting.

## Required Telemetry

Every row must emit these fields before it can be trusted:

- run identity: `schema_id`, `impl_id`, `profile_only`, `calls_train_muzero`,
  `trainer_defaults_changed`, `touches_live_runs`, run id, git SHA, compute;
- shape: batch size, actor count, player count, simulations, measured steps,
  warmup steps, death mode, scalar-edge mode, stack dtype, device-latest flag;
- backend identity: observation mode, renderer backend name, fail-closed GPU
  backend proof, no hidden CPU fallback flag;
- denominators: physical rows, scalar roots, env/source steps, ready count,
  live row count, terminal count, autoreset count, search/root call count,
  replay-shaped write/sample count, learner-shaped batch count if present;
- throughput: wall seconds, physical rows/sec, scalar roots/sec, and effective
  scalar timesteps/sec using one explicit denominator;
- timings: env step, compact merge, observation total, renderer total, renderer
  device, stack update, H2D, D2H/readback, normalize, model/policy forward,
  search loop, replay-shaped pack/write, scalar materialization, host sync
  count, parent send/receive if actors are involved;
- bytes: compact payload bytes/step, compact payload bytes/timestep, rendered
  stack bytes/step, scalar materialized payload bytes, input dtype and input
  bytes to the policy/search consumer;
- semantics: row-major mapping evidence, `policy_env_id`, `policy_env_row`,
  `policy_player`, action mask shape, latest-frame shape, final observation
  presence, terminal-before-autoreset proof;
- GPU: max utilization, max memory, device name;
- RND-only when present: update cadence, predictor/target calls, latest-frame
  source, reward-unchanged meter proof, RND timing.

## Pass / Fail Gates

Harness repeat passes if rows 1-2 are within `15%` of the known H100 canary:
scalar-off at least `11.7k` roots/s and scalar-on at least `5.5k` roots/s.
If not, stop and debug measurement drift before interpreting new rows.

Resident real-pressure passes if row 3 keeps at least `50%` of the repeated
scalar-off canary and is at least `3x` faster than the matched stock C512/sim8
zero-observation ceiling. If row 3 falls below `2x` stock zero-observation,
do not invest in integration.

Scalar edge passes if row 4 keeps at least `45%` of row 3 and remains at least
`2x` faster than matched stock real batched-GPU observation. If scalar
materialization consumes more than half the wall and there is no credible plan
to delay it past policy/search/replay work, pause integration.

Hardware sanity passes if L4 rows 5-6 keep the same qualitative ordering:
resident real-pressure scalar-off beats stock zero-observation by at least
`2x`, and scalar-on remains clearly above stock real observation. If H100 passes
but L4 collapses, integration is H100-only until proven otherwise.

Death semantics pass only if rows 7-8 show correct terminal final observations,
row masks, autoreset order, and scalar terminal payloads with no hidden fallback.
Any missing final observation or row/player mismatch is a hard fail regardless
of speed.

Invest in a real stock-boundary/custom-collector prototype only if all of these
are true:

- rows 1-8 are valid with complete telemetry;
- row 3 clears the resident real-pressure gate;
- row 4 clears the scalar-edge gate or the design can keep scalarization outside
  the hot policy/search/replay loop;
- normal-death semantics pass;
- the result is reported separately from actual Coach training speed.

If any hard gate fails, keep resident batching as a profile-only research lane
and spend the next cycle on the measured failing bucket, not on live training.
