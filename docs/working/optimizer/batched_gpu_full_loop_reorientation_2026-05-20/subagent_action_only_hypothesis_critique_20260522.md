# Action-Only / Deferred-Payload Hypothesis Critique, 2026-05-22

Status: docs-only hostile critique. I inspected
`src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`,
`tests/test_mctx_synthetic_benchmark_legality.py`, and the current
`batched_gpu_full_loop_reorientation_2026-05-20` working notes. I did not edit
source and did not touch live Coach training.

## Plain Verdict

The action-only/deferred-payload idea is plausible as a ceiling probe, but the
current `closed_loop_action_only_profile` number is not replay-valid.

In the benchmark it explicitly requires `closed_loop_replay_index=False` and
raises if replay indexing is on. It reads selected actions, checks active action
legality, steps the CPU env, and skips action-weight/root-value materialization,
`CompactSearchResultV1` validation, and `CompactReplayIndexRowsV1` construction.

That means an action-only speedup can be one of three very different things:

1. Real overlap: selected action is the only payload needed before CPU env step,
   while visit policy/root value can be produced or copied later.
2. Sync relocation: action-only moved a JAX wait out of the measured bucket, and
   the wait returns when any later consumer asks for the policy/value payload.
3. Deletion: the profile simply stopped paying work required by replay and
   target rows.

Only case 1 is a real optimizer opportunity. The current action-only switch does
not distinguish those cases.

## What The Current Benchmark Actually Proves

Relevant benchmark behavior:

- Full closed loop blocks search on `action_weights`, then reads actions,
  action weights, extracts root values, validates `CompactSearchResultV1`, builds
  a CPU joint action, steps the env, and optionally builds replay-index rows.
- Action-only closed loop blocks search on `action`, reads actions, scatters
  active selected actions into the joint action, checks those active actions
  against the legal mask, steps the env, and skips the replay-valid payload.
- The benchmark marks this honestly: action-only profile "skips
  root-value/policy materialization and is not a replay-valid lane."
- `tests/test_mctx_synthetic_benchmark_legality.py` covers MCTX legality helper
  behavior, resident latest-frame shape/dtype, root row-major order, and bucket
  breakdown. It does not test the closed-loop action-only branch, deferred
  payload identity, or replay materialization.

The broader compact replay contract requires more than actions. The index-only
hot path deliberately skips observation tensors, but it still stores
`policy_target` and `root_value`. The replay index builder also checks
root identity, row/player mapping, selected action matching `next_joint_action`,
terminal final-observation masks, and final reward mapping. Action-only has not
paid or proven those obligations.

## Hostile Failure Modes

1. **The speedup is fake because replay data was deleted, not deferred.**
   If the row omits `visit_policy`, `root_value`, compact search validation, and
   replay-index construction, it is an action service ceiling, not a collection
   loop.

2. **The wait moves instead of disappearing.**
   Blocking on `loop_output.action` may let `search_sec` look smaller than
   blocking on `action_weights`, but a later `np.asarray(action_weights)` or
   root-value extraction can absorb the same device work. The only trustworthy
   metric is total wall after the deferred payload is actually consumed.

3. **Replay semantics break under non-prefix active roots.**
   Deferred payloads must carry `root_index`, `env_row`, `player`,
   `policy_env_id`, and compacted policy-row identity. A flat payload array
   matched only by active-output order will eventually corrupt rows when active
   roots are sparse.

4. **Terminal/autoreset rows lose final data.**
   Replay rows need the search at record `k`, selected action into `k+1`, next
   reward/done, final reward, and final observation before autoreset. Deferring
   payloads while advancing manager state is unsafe unless final masks and
   record ids are frozen with the search result.

5. **No-copy root batches can alias mutable state.**
   The best denominator uses `compact_root_copy_observation=False`. If deferred
   validation keeps views into a compact batch while the manager advances, a
   later payload drain may validate against changed or stale root sidecars.

6. **Inactive default actions leak into env behavior.**
   Action-only fills the joint action with a default and overwrites active seats.
   Without replay rows, it only proves active selected actions are legal. It
   does not prove inactive/default actions are ignored across mixed live/dead
   rows, terminal rows, or autoreset rows.

7. **Root values may not be cheaply recoverable later.**
   `_extract_mctx_root_values` is best-effort across MCTX output shapes. Full
   mode fails if root values are missing. Action-only hides that failure until a
   replay-valid consumer appears.

8. **Action weights are not just diagnostics.**
   They are the policy target. Skipping them changes the learner target surface,
   not just a logging path. A "deferred payload" design must still prove
   normalized legal visit policy for the exact roots that drove the transition.

9. **The toy MCTX model can underprice payload semantics.**
   The benchmark uses a synthetic JAX MuZero model, not current LightZero
   PyTorch weights, heads, support transforms, noise schedule, or RND/replay
   consumers. Action-only numbers are profile-loop evidence only.

10. **Backlog memory pressure is unmeasured.**
    A real deferred design must retain output buffers, root sidecars, record
    ids, and next-step sidecars until payload drain. The current action-only
    loop discards the expensive obligations instead of queueing them.

11. **Timing labels become misleading.**
    If deferred action weights/root values are drained after `env_step_sec`, the
    work may land in residual, next-step input prep, or the next search wait.
    Bucket wins are meaningless without `measured_bucket_sum_sec`,
    `residual_sec`, and total roots/sec.

12. **Partial rows can become sample-visible.**
    If replay storage receives action/reward/done before policy target/root
    value, the sampler must not expose the row. The current benchmark has no
    partial-row readiness or commit protocol.

## Minimal Validation Rows

These are the smallest rows needed before treating action-only as more than a
ceiling.

| Row | Shape | Switches | Purpose | Pass condition |
| --- | --- | --- | --- | --- |
| A0 full baseline | Current best H100 denominator, B1024/P2, loop24, sim16 and sim32 | resident GPU stack, `compact_root_copy_observation=False`, replay-index on, action-only off | Replay-valid denominator | Total roots/sec, bucket fractions, residual, active roots, completed steps recorded |
| A1 no replay index | Same as A0 | replay-index off, action-only off | Isolate replay-index cost only | Delta from A0 is small and matches `replay_index_sec` |
| A2 action-only ceiling | Same as A0 | replay-index off, action-only on | Price selected-action-only lower bound | Report as ceiling only, never replay-valid speed |
| A3 immediate payload drain | Same as A2 | after env step, read action weights and root values from the same MCTX output; validate search result; no replay index | Separate deletion from deferral | Total wall stays close to A2 if deferral is real; otherwise the A2 win collapses |
| A4 deferred replay drain | Same as A2 | build replay-index rows one step later from saved root batch, payload, next reward/done/final masks | Prove replay-valid deferred commit | Matches A0 replay rows exactly on actions, policy target, root value, reward/final reward, done, ids |
| A5 backlog stress | Same as A4 | drain every N steps, N in {2, 4, 8} | Price queue memory and delayed sync | No memory blowup, no residual spike, no stale payload ids |
| A6 terminal companion | Smaller normal-death/autoreset row, B64 or B256 | A4 switches, terminal/final reward forced | Catch final-observation and final-reward bugs | Materialized target rows match compact replay contract |
| A7 non-prefix active roots | Small local or Modal canary | mixed active/inactive roots | Catch sparse active-root mapping bugs | Compact root rows, policy rows, env rows, players, actions all match |
| A8 sync attribution | Same as A0/A2/A3 | explicit resident sync on and off | Detect wait relocation | Total wall improves, not just bucket movement |

Kill rule: if A3 or A4 gives back most of A2's gain, action-only was deletion
or sync relocation. Do not continue optimizing it as a speed lane.

Keep rule: action-only is interesting only if A4 is replay-valid and remains
materially closer to A2 than A0 on total closed-loop roots/sec, with no residual
or memory backlog hiding the cost.

## Minimal Local Tests

Add these before any trainer-facing recommendation:

- A unit test that `closed_loop_action_only_profile=True` rejects replay-index
  mode and reports `action_only_profile=True`.
- A deferred-payload replay test: save a root batch/search output, step with
  selected actions, then build `CompactReplayIndexRowsV1` after the step and
  materialize target rows. Compare to the immediate full path.
- A non-prefix active-root test where active roots are `[1, 3]`, not a prefix,
  and inactive default joint actions cannot affect replay rows.
- A terminal/autoreset test with `next_final_reward_map != next_reward` and
  required `next_final_observation_row_mask`.
- A no-copy alias test: use `copy_observation=False`, advance the manager, then
  prove saved root sidecars and deferred payload identity still describe the
  original record.

## Recommendation

Keep action-only as a hostile ceiling/profiler switch. It is useful for pricing
the maximum possible benefit of reading only selected actions before the CPU env
step.

Do not quote action-only roots/sec as compact replay throughput, Coach speed, or
MCTX loop speedup. The first legitimate claim is:

```text
closed compact loop with selected-action-first env step and deferred but
replay-valid visit-policy/root-value commit
```

Until A4 passes on the same denominator, the safest interpretation is blunt:
the action-only speedup may just be the cost of the replay payload we stopped
paying.
