# 2026-05-09 Vector Actor-Loop Bridge

## Question

Can the current fixture-seeded vector path run a first local bridge that looks
like an actor loop: batched env step, debug obs/reward packing, policy/search
shaped action selection, and replay chunk staging?

This is a runnable bridge benchmark, not a final training benchmark.

## Script

Script:

```sh
python3 scripts/benchmark_vector_actor_loop_bridge.py \
  --batch-sizes 128 \
  --event-modes debug-event no-event \
  --repeat 250 \
  --warmup 25 \
  --rollout-steps 4 \
  --hidden-dim 32 \
  --simulations 4 \
  --chunk-steps 32 \
  --body-capacity 4 \
  --format plain
```

What it does:

- Seeds the current supported fixture rows.
- Runs source/common-trace state and fixed event-row preflight once per fixture.
- Runs B>1 batch-state/event preflight for each fixed shape group.
- Groups rows as `P=2,K=4` and `P=3,K=4`.
- Repeats local rollout blocks:
  `reset fixture batch once -> vector step -> debug pack -> synthetic policy/search -> action encode -> next vector step`.
- Uses fixture source moves only for step 0 of each rollout block. Later steps
  feed back synthetic selected actions as source moves `-1/0/1`.
- Packs the existing debug `obs[B,P,9]`, `reward[B,P]`, masks, legal action
  mask, and ego ids.
- Runs a synthetic NumPy root model plus a small recurrent/search-shaped fake
  visit loop. This is intentionally not MCTS.
- Stages obs, reward, action ids, fake action weights, fake root values, done,
  and ego masks into a fixed in-memory replay chunk ring. The script now reports
  replay chunk bytes and shapes in the JSON sample, and chunk bytes in plain
  output.
- Can run `--sample-only` to return the fixed-shape final-step actor bridge
  interface without replay timing. That helper exposes `obs`, `reward`,
  `done`, `truncated`, `ego_mask`, `ego_row_id`, `ego_env_id`,
  `ego_player_id`, and `legal_action_mask` from real vector env steps.
- Can run `--event-modes debug-event no-event` to compare the bridge with debug
  event rows emitted or skipped inside the env step. Normal debug-event source
  and B>1 event preflight still runs; the no-event path adds a state-only
  preflight before timing.
- In no-event mode, the debug obs/reward packer is given the state without event
  arrays, so its existing alive-transition fallback replaces die-event reward
  detection. The benchmark checks this keeps the no-event bridge sample checksum
  aligned with debug-event mode for the focused test case.

Timer buckets:

- `reset_sec`: copy fixture initial arrays into the mutable B-row state once per
  rollout block.
- `prev_snapshot_sec`: copy the minimal previous alive/score/round-score arrays
  needed by the debug reward delta packer before synthetic-feedback steps.
- `env_step_sec`: one in-place vector batch step.
- `debug_pack_sec`: debug obs/reward/mask/ego-id packing.
- `policy_sec`: synthetic root model, fake recurrent/search loop, and action
  selection.
- `action_encode_sec`: selected action ids to `-1/0/1` source moves.
- `replay_sec`: fixed in-memory replay chunk staging.
- `overhead_sec`: loop wall time outside the measured buckets.

## Result

Local runtime labels from the script: macOS arm64, Python 3.11.14, NumPy 2.4.0.

Preflight passed: 8 fixtures passed, 0 failed, 0 unsupported. Batch state and
event preflight passed for both fixed-shape groups. The explicit no-event
state-only preflight also passed for both groups.

Command profile: `repeat=250`, `warmup=25`, `rollout_steps=4`,
`hidden_dim=32`, `synthetic_sims=4`, `chunk_steps=32`,
`event_modes=debug-event,no-event`.

Each timed batch row below is four consecutive vector steps inside a reset block.
At `B=128`, that means 1,000 env step calls, 128,000 env rows, and 256k or 384k
ego rows depending on player count.

| Group | Mode | B | Env rows | Ego rows | Elapsed | Env rows/sec | Ego rows/sec | Env step | Synthetic policy | Replay stage | Top bucket | Top env phase |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `P=2,K=4` | debug-event | 128 | 128,000 | 256,000 | 0.527991s | 242,428.6 | 484,857.2 | 0.235300s | 0.240967s | 0.007450s | `env_step` | `event_emit_sec:0.089580s` |
| `P=2,K=4` | no-event | 128 | 128,000 | 256,000 | 0.425357s | 300,923.8 | 601,847.7 | 0.140443s | 0.239502s | 0.007667s | `policy_search` | `movement_sec:0.028058s` |
| `P=3,K=4` | debug-event | 128 | 128,000 | 384,000 | 0.565989s | 226,152.9 | 678,458.6 | 0.232671s | 0.276348s | 0.008095s | `policy_search` | `event_emit_sec:0.058158s` |
| `P=3,K=4` | no-event | 128 | 128,000 | 384,000 | 0.467054s | 274,058.5 | 822,175.5 | 0.149663s | 0.269833s | 0.007658s | `policy_search` | `movement_sec:0.042024s` |

| Group | Debug env step | No-event env step | Debug minus no-event | Env-step cost share | Env-step speedup | Total-loop speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `P=2,K=4` | 0.235300s | 0.140443s | 0.094857s | 40.3% | 1.675x | 1.241x |
| `P=3,K=4` | 0.232671s | 0.149663s | 0.083008s | 35.7% | 1.555x | 1.212x |

The useful change: the bridge now has a benchmark-local continuous block shape.
It still resets from fixtures, but it can take several vector steps on copied
rows and feed the synthetic action choices into later steps. That is closer to
self-play plumbing than the first reset-plus-one-step bridge.

## Fixed-Shape Sample Helper

Follow-up smoke command:

```sh
python3 scripts/benchmark_vector_actor_loop_bridge.py \
  --sample-only \
  --batch-sizes 4 \
  --event-modes debug-event \
  --rollout-steps 2 \
  --hidden-dim 32 \
  --simulations 4 \
  --body-capacity 4 \
  --player-count 2 \
  --format json
```

Result: passed locally.

- Source preflight: `8` pass, `0` fail, `0` unsupported.
- Selected group: `P2_K4`.
- Final sample source kind: `synthetic_feedback_moves` because
  `rollout_steps=2`; step 0 used fixture source moves.
- `obs`: `[4, 2, 9]`, `float32`.
- `reward`: `[4, 2]`, `float32`.
- `done`: `[4]`, `bool`; `2` rows done.
- `truncated`: `[4]`, `bool`; `0` rows truncated.
- `ego_mask`: `[4, 2]`, `bool`; `4` live ego rows.
- `ego_row_id`, `ego_env_id`, `ego_player_id`: `[4, 2]`.
- `legal_action_mask`: `[4, 2, 3]`, `bool`; `12` true entries.
- Reward death evidence: `event_rows`.
- Actor rollout time inside the helper: `0.0007661670097149909s`.

This gives the Modal/JAX side a small, concrete interface to consume without
pretending the actor loop is production self-play.

Simple conclusion: debug event logging costs 0.095s of 0.235s in the
`P=2,K=4` env-step bucket and 0.083s of 0.233s in the `P=3,K=4` env-step
bucket for this B=128 bridge run. Removing debug events improves the env-step
bucket by 1.56x to 1.68x, but total-loop throughput improves only 1.21x to
1.24x because synthetic policy/search, debug packing, replay staging, and
non-event env state updates remain.

## Honest Gaps

- The loop is fixture-reset rollout-block timing, not a real reset/autoreset
  contract.
- The obs/reward path is the debug packer:
  `curvyzero_debug_global_player_obs/v0` and
  `curvyzero_debug_score_round_delta_death_penalty/v0`.
- The policy/search section is synthetic NumPy work. It is not learned policy,
  MCTS, Mctx, JAX, GPU, or device transfer.
- Replay is fixed in-memory chunk staging only. There is no final replay schema,
  writer, learner handoff, compression, or storage path.
- Source fidelity preflight covers the supported fixture source moves and B>1
  batch equivalence for those fixtures. Timed synthetic policy moves are not
  source-compared.
- `P=2` and `P=3` are still separate fixed-shape groups, not one padded mixed-P
  production batch.
- PrintManager/trail-gap semantics and broader source mechanics are still
  production gates.

## Next Useful Moves

1. Decide which debug event rows must remain on the production hot path, then
   optimize or move the rest out of the actor-loop step.
2. Replace fixture reset blocks with a real fixed-shape reset/autoreset path.
3. Keep the debug packer until the trainer-facing observation/reward contract is
   ready, then swap it behind this benchmark.
4. Replace the synthetic policy/search bucket with the real policy/search bridge
   and keep the same bucket labels for before/after comparison.
5. Turn replay staging into the first real chunk schema once the action/reward
   contract stops moving.
