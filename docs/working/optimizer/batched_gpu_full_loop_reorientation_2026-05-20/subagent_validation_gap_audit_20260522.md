# Subagent Validation Gap Audit - 2026-05-22

Scope: profile-only CurvyTron compact visual plus MCTX loop validation. I
reviewed the compact replay tests, MCTX synthetic legality tests, compact
root/replay builders, and nearby hybrid replay-proof tests. I did not touch
production trainer code, live runs, checkpoints, or Modal jobs.

Files changed in this pass:

- `tests/test_mctx_synthetic_benchmark_legality.py`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_validation_gap_audit_20260522.md`

## Short Read

The current tests are good local contract tests. They prove that compact
target rows, compact replay index rows, active-root filtering, final
observation before autoreset, RND latest-frame extraction, and non-prefix
active roots can match the object-row path in synthetic chunks.

They do not yet prove that a faster closed-loop profile row is replay-valid
end to end. In particular, the deferred and overlap payload profile flags are
currently action-loop profiles: the benchmark explicitly requires
`closed_loop_replay_index=False` for action-only, deferred-payload, and
overlap-payload modes, and those modes do not build replay rows on the loop
edge. Treat their speed numbers as "selected-action loop" measurements unless a
separate flush/materialization parity test proves the delayed payload produces
the same replay rows.

I added one small safe test for the new direct root-node value extractor. It
checks that `search_tree.node_values[:,0]` is used as the root-value payload
and that payload materialization counts both action weights and root values.

## Existing Coverage That Looks Solid

- `tests/test_compact_search_replay_contract.py` covers the strongest local
  replay contracts:
  - two-record final observation before autoreset and RND latest-frame
    extraction;
  - `CompactRootBatchV1 -> CompactSearchResultV1 -> CompactReplayChunkV1`
    materialized row parity;
  - `CompactReplayIndexRowsV1` without observation materialization, then
    materialization back to target rows;
  - stale env/player/policy id and stale legal-mask failures on the index-row
    path;
  - non-prefix active roots mapping compact root rows to compacted policy rows;
  - deferred payload rows matching immediate rows for a non-prefix active-root
    fixture;
  - optional no-copy root observations for the profile hot path.
- `tests/test_mctx_synthetic_benchmark_legality.py` covers active-only
  legality summaries, resident stack dtype/shape, row-major compact visual
  root order, closed-loop timing breakdown labels, and now direct root-value
  extraction from MCTX-like output.
- `tests/test_source_state_hybrid_observation_profile.py` has useful
  profile-level checks for compact service replay proof: search actions drive
  the next env step, direct arrays and array-ceiling arrays are accepted, warmup
  seeding is separated, and missing warmup is rejected.
- `tests/test_source_state_batched_observation_boundary_profile.py` proves a
  fake direct-CTree compact output can feed checked target rows and preserves
  actions, policy targets, and root values through the materialized
  `CompactReplayChunkV1` validation edge.

## Main Replay-Validity Gaps

### 1. No closed-loop replay-valid equivalence canary

The compact bridge has row-level parity, and the hybrid profile proof checks
that selected search actions are used for the next step, but there is no single
test that runs:

```text
root batch[k]
-> search result[k]
-> env step with selected actions[k]
-> compact replay index rows[k]
-> materialized target rows[k]
-> compare with immediate/object target rows[k]
```

over multiple records. That is the missing "speed claim is replay-valid" gate.
It should include at least one live row, one terminal/final-observation row, and
one non-prefix active-root row.

### 2. Deferred and overlap payload modes are intentionally not replay rows

`mctx_synthetic_benchmark.py` rejects deferred/overlap payload profiling when
`closed_loop_replay_index=True`. The code byte-counts or waits for delayed
`action_weights` and root values, but it does not attach those payloads to
reward/done/final-observation sidecars and materialize rows.

Before claiming replay-valid speed for those modes, add a test that stages
payloads for several records, flushes them later, builds
`CompactReplayIndexRowsV1`, materializes rows, and compares them with the
immediate replay-valid path.

### 3. Overlap needs record-id ordering tests

The overlap profile uses a future to materialize payloads while the env step is
running. A future-order bug could attach record `k+1` root values or visit
policies to record `k` rewards. Add sentinels that encode `record_index`,
`env_row`, and `player` into visit policy/root value, complete futures out of
order, and require commit by explicit record id rather than completion order.

### 4. Root-value extraction needs active-root sentinel coverage

The added unit test checks direct `node_values[:,0]` extraction. The next test
should use inactive roots with deliberately poisonous root values and prove
that `root_values[active_indices]` is the only payload passed into
`validate_compact_search_result_v1`.

### 5. Policy env ids are mostly tested as `arange`

The builders allow unique `policy_env_id` values independent of compact root
row order, but most fixtures use `np.arange(root_count)`. Add a non-identity
unique id fixture, then require it to round-trip through root batch, search
result, index rows, source refs, and materialized target rows. Also make stale
`policy_env_id` fail on every path that claims identity-sidecar validation; the
index-row path already has this guard.

### 6. Resident-vs-host stack equivalence stops before replay rows

The stack tests cover FIFO order, uint8 storage, host/device latest-frame order,
and RND latest extraction. They do not yet run the same selected actions through
host-stack and resident-stack compact root/replay construction and compare the
resulting root observations, replay index rows, and materialized rows.

This is the observation-stack gate needed before saying resident visual stack
speedups preserve replay semantics.

### 7. Profile flag semantics need a small local guard

The benchmark runtime has the important checks: action-only/deferred/overlap
flags are mutually exclusive, and each requires `closed_loop_replay_index=False`.
Those checks live inside the full benchmark path. Extracting or wrapping that
validation into a tiny pure helper would let unit tests pin the semantics:

- replay-valid mode may build replay index rows;
- action-only mode is explicitly not replay-valid;
- deferred/overlap modes are explicitly payload-profile-only until a flush
  parity test exists;
- summaries should carry that label so plots cannot accidentally compare
  action-only rows as replay-valid rows.

## Recommended P0 Tests

1. `test_closed_compact_loop_index_rows_materialize_same_as_immediate_rows`

   Use a two- or three-record local chunk. Build root batches, validate fake
   search results with sentinel visit policies/root values, step with selected
   actions, build index rows, materialize them, and compare to
   `build_compact_target_rows_from_search_arrays_v0` and object rows.

2. `test_deferred_payload_flush_rows_match_replay_valid_loop`

   Stage selected actions immediately, delay visit-policy/root-value payloads,
   flush after `K` records, then compare index rows and materialized target rows
   with the immediate replay-valid mode. Include terminal final observations.

3. `test_overlap_payload_commit_uses_record_identity`

   Complete payload futures in reversed order. Require the replay builder to
   attach payloads by `(record_index, compact_root_row, policy_env_id)`.

4. `test_non_identity_policy_env_ids_round_trip_compact_replay`

   Use unique ids like `[10, 11, 20, 21]` instead of `arange`. Verify source
   refs and stale-id rejection on both index-row and materialized validation
   paths.

5. `test_resident_and_host_stack_sources_build_same_replay_rows`

   With fixed actions and identical initial state, compare host compact visual
   root batches against resident compact visual root batches at the replay-row
   materialization edge, not only at latest-frame extraction.

## Claim Boundary

Replay-valid compact speed claims should require:

- selected actions used for the env step are the same actions stored in replay;
- visit policy and root value payloads are attached to the same active roots;
- `policy_env_id`, `compact_root_row`, `env_row`, `player`, and `policy_row`
  survive compaction/defer/overlap;
- reward/done/final-observation sidecars come from the following record;
- terminal next observations use final observations before autoreset;
- materialized compact rows equal the existing object-row target builder;
- observation stacks match across host/resident source modes where equivalence
  is claimed.

Until those pass, the safer label for action-only/deferred/overlap rows is
profile-only loop throughput, not replay-valid trainer throughput.
