# Subagent Stock Profile Harvest

Date: 2026-05-12

Scope: stock LightZero `lzero.entry.train_muzero`,
`env_variant=source_state_fixed_opponent`,
`opponent_policy_kind=frozen_lightzero_checkpoint`,
`opponent_use_cuda=false`. This is profiler synthesis only, with no learning
quality claims.

Source: fresh read-only Modal volume downloads via
`scripts/summarize_curvytron_lightzero_profiles.py`, into
`/private/tmp/curvytron_lightzero_profile_harvest_20260512`. Missing requested
refs: none.

## Requested Rows

| Label | Exact run:attempt | OK/fail | Compute | Manager | Collectors | Render | No-death | B | Sims | Steps | Wall | Collect | MCTS | Policy fwd collect | Policy fwd eval | Learner | Steps/s |
| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base no-death browser | `opt-stock-frozen-l4-base-b16-sim8-nodeath-browser-s304-20260512h:profile-l4-base-b16-sim8-nodeath-browser` | ok | `gpu-l4-t4` | base | 1 | `browser_lines` | yes | 16 | 8 | 256 | 73.73s | 68.26s | 8.42s | 6.57s | 5.28s | 1.39s | 3.47 |
| base no-death body circles | `opt-stock-frozen-l4-base-b16-sim8-nodeath-bodycircles-s304-20260512h:profile-l4-base-b16-sim8-nodeath-bodycircles` | ok | `gpu-l4-t4` | base | 1 | `body_circles_fast` | yes | 16 | 8 | 256 | 36.31s | 30.43s | 8.50s | 6.70s | 5.21s | 1.49s | 7.05 |
| C32 sim8 subprocess | `opt-stock-frozen-l4cpu40-subproc-c32-b16-sim8-s304-20260512h:profile-l4cpu40-subproc-c32-b16-sim8-nep32` | ok | `gpu-l4-t4-cpu40` | subprocess | 32 | `browser_lines` | no | 16 | 8 | 314 | 13.89s | 4.78s | 0.36s | 1.49s | - | 1.67s | 22.61 |
| C16 sim16 subprocess | `opt-stock-frozen-l4cpu40-subproc-c16-b16-sim16-s304-20260512h:profile-l4cpu40-subproc-c16-b16-sim16-nep16` | ok | `gpu-l4-t4-cpu40` | subprocess | 16 | `browser_lines` | no | 16 | 16 | 153 | 14.14s | 5.40s | 0.75s | 2.10s | - | 2.09s | 10.82 |

Notes:

- `Policy fwd eval` is absent on the subprocess rows because stock LightZero
  eval was skipped in these profile runs.
- The base rows expose env/render internals. Subprocess rows do not expose
  worker-side env/render/opponent timing buckets in the harvested summary.

Base-manager env attribution:

| Label | Env step | Stack update | RGB render | Vector step | Model initial | Model recurrent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base no-death browser | 61.51s | 51.01s | 50.37s | 4.74s | 2.58s | 5.95s |
| base no-death body circles | 23.58s | 13.19s | 12.52s | 4.67s | 2.57s | 6.00s |

## Prior Completed Context

Rows below are the known completed stock-frozen rows from
`stock_frozen_optimizer_pivot_2026-05-12.md`, re-read with exact attempt IDs.

| Exact run:attempt | Manager | Collectors | Sims | Steps | Wall | Collect | MCTS | Policy fwd collect | Policy fwd eval | Learner | Steps/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `opt-stock-frozen-l4-base-b16-sim8-s304-20260512d:profile-l4-base-b16-sim8` | base | 1 | 8 | 33 | 10.79s | 4.63s | 1.20s | 2.46s | 0.76s | 1.67s | 3.06 |
| `opt-stock-frozen-l4-subproc-c2-b16-sim8-s304-20260512e:profile-l4-subproc-c2-b16-sim8-nep2` | subprocess | 2 | 8 | 22 | 9.90s | 2.84s | 0.34s | 1.46s | - | 2.08s | 2.22 |
| `opt-stock-frozen-l4-subproc-c4-b16-sim8-s304-20260512f:profile-l4-subproc-c4-b16-sim8-nep4` | subprocess | 4 | 8 | 38 | 9.94s | 3.00s | 0.37s | 1.58s | - | 1.87s | 3.82 |
| `opt-stock-frozen-l4cpu40-subproc-c8-b16-sim8-s304-20260512f:profile-l4cpu40-subproc-c8-b16-sim8-nep8` | subprocess | 8 | 8 | 87 | 12.63s | 4.56s | 0.39s | 1.67s | - | 2.11s | 6.89 |
| `opt-stock-frozen-l4cpu40-subproc-c16-b16-sim8-s304-20260512g:profile-l4cpu40-subproc-c16-b16-sim8-nep16` | subprocess | 16 | 8 | 159 | 13.80s | 5.15s | 0.42s | 1.78s | - | 2.06s | 11.52 |

## Plain Amdahl Read

The no-death browser-lines base row is collector/env dominated. Wall is
`73.73s`, collect is `68.26s`, and the exposed base-manager `env_step` bucket is
`61.51s`; within that, stack/render is about `51s`. Named MCTS is only `8.42s`
and learner is `1.39s`, so a pure search or learner speedup cannot move most of
that row.

The matched no-death `body_circles_fast` row cuts wall to `36.31s` mainly by
shrinking stack/render from about `51s` to about `13s`. MCTS and model-forward
time are essentially unchanged. This is a render-cost ablation, not a learning
quality result.

The C32 subprocess sim8 row is the fastest harvested row at `22.61` steps/s.
Visible named MCTS is tiny (`0.36s` of `13.89s` wall); the remaining visible
time is collect, policy-forward collect, learner, and one-time setup/init
overhead, while subprocess hides the env/render/opponent split.

The C16 subprocess sim16 row raises MCTS from the prior C16 sim8 row
(`0.42s -> 0.75s`) but total wall barely changes (`13.80s -> 14.14s`), and
steps/s is close to the prior C16 row (`11.52 -> 10.82`). At this profile size,
doubling simulations is visible but still not the dominant wall-time term.
