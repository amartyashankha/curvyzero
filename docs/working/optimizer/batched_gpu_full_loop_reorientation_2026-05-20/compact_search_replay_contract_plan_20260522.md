# Compact Search Replay Contract Plan

Date: 2026-05-22

Status: active optimizer design note. Profile-only until the parity tests below
exist and pass.

## Plain Goal

We need one explicit contract for the fast lane:

```text
compact CurvyTron roots -> compact search result -> compact replay/target rows
```

Do not hide this as a quiet replacement for stock LightZero collect dicts. If
we bypass stock dicts and GameSegments, this is a replacement bridge and must
prove that it trains on the same facts.

## Contract Sketch

`CompactRootBatchV1`:

- `observation`: `[B,P,4,64,64]`, player perspective, newest frame last.
- `action_mask`: `[B,P,3]`, strictly binary.
- `target_reward`: `[B*P,1]`.
- `done_root`: `[B*P]`, repeated from row-level done.
- `policy_env_row`, `policy_player`, `policy_env_id`: `[B*P]`.
- `to_play`: `[B*P]`, currently fixed-opponent `-1` only.
- `active_root_mask`: `[B*P]`, not done and at least one legal action.
- terminal/autoreset/final-observation sidecars.
- metadata: observation schema, reward schema/support, RND mode, perspective,
  death/autoreset policy, and search/trainer identity.

`CompactSearchResultV1`:

- `root_index`, `env_row`, `player`: `[N]`.
- `selected_action`: `[N]`.
- `visit_policy`: `[N,3]`, normalized, zero illegal mass.
- `raw_visit_counts`: `[N,3]` if available.
- `searched_value`, `predicted_value`, `predicted_policy_logits`.
- audit fields: implementation, model version, simulations, noise, epsilon,
  temperature, fallback count.

`CompactReplayChunkV1`:

- Time-major arrays over records.
- Search at record `k` consumes `observation[k]`.
- Selected action must match the stored action for the next transition.
- Reward, done, and next observation come from `k+1`.
- Terminal next observation must use final observation, not autoreset
  observation.

## Already Proven

- `HybridCompactBatch` carries most root sidecars.
- The direct-CTree sidecar validates row-major ids, `done_root`, `to_play`,
  active roots, target reward, terminal/autoreset masks, and final-observation
  sidecars before search.
- Compact search arrays can become checked target rows through the existing
  `PolicyRowRecordV0` bridge.
- RND latest-frame extraction now slices the latest channel before
  normalization and enforces `[B*P,1]` target reward shape.
- Closed compact local rows now reach about `57.9k-68.4k` timesteps/sec on
  B512-B4096 shapes, so the compact sidecar itself is no longer the toy wall.

## Missing Proofs

P0 before trusting a compact replay bridge:

- Compact arrays to target rows without allocating `PolicyRowRecordV0`, compared
  exactly against the current target-row builder. First proof landed
  2026-05-22 for live rows and terminal/final-observation rows.
- Two-record and three-record replay chunks, not only one-step batches.
- Mixed live plus terminal batch: terminal roots skipped, live roots searched,
  final observation used before autoreset.
- RND latest-frame sentinel in the same compact path.
- Row/order sentinel with non-identity ids and both players.
- Perspective sentinel: player 0 and player 1 observations must not swap.
- Reward support/schema guard.
- Current-policy two-seat `to_play` must be rejected clearly or tested
  separately from fixed-opponent `-1`.

## Next Executable Proof

Add a local test module:

```text
tests/test_compact_search_replay_contract.py
```

It should build the same deterministic fixture through two paths:

```text
existing record/object path -> current target rows
compact root/search/replay arrays -> compact target rows
```

Assert actions, masks, visits, root values, rewards, final rewards, done flags,
next observations, row/player ids, and `to_play` match exactly. Include one
live row, one terminal/autoreset row, and an RND latest-frame sentinel.

2026-05-22 implementation update:

```text
build_compact_target_rows_from_search_arrays_v0(...)
```

now builds target rows directly from compact search arrays, without allocating
`PolicyRowRecordV0` objects. Focused tests compare this direct compact path to
the existing object path on a live row and a terminal row that must use
`final_observation` before autoreset.

Remaining proof gaps:

- compact replay writer and sampler, not only target-row construction;
- promotion out of private helper imports into a public low-level target-row
  builder contract;
- current-policy two-seat `to_play` support, if/when this moves beyond the
  fixed-opponent `to_play=-1` lane.

2026-05-22 hardening update:

```text
tests/test_compact_search_replay_contract.py
```

now covers:

- two-record mixed live/terminal rows;
- terminal final-observation-before-autoreset semantics;
- three-record chunks using `record_index=1`;
- non-prefix active roots such as compact roots `[1, 3]`;
- compact root row versus compacted replay `policy_row` mapping;
- non-identity `policy_env_id` provenance;
- RND latest-frame row order;
- player-perspective swap rejection.

Bug fixed:

```text
compact_root_row != replay policy_row
```

The bridge now preserves both identities. `compact_root_row` indexes the flat
compact `[B*P]` sidecar; `policy_row` indexes the replay chunk's compacted
live-policy rows.

## Architecture Read

Thin array-native CTree is probably useful but capped. It may give a bridge
class improvement, not a 10x shift. The larger speed path is still:

```text
many compact roots -> batched search/model service -> compact replay chunks
```

LightZero compatibility should become a validation/output adapter, not the
main hot data path.
