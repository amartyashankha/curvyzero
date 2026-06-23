# Compact Ownership Implementation Ladder

Date: 2026-05-23

Scope: Lane 2 implementation plan only. No live Coach runs, Modal runs,
checkpoints, evals, GIFs, tournaments, or volumes were touched.

## Lane Split

Lane 1 is the practical train-facing patch:

```text
stock LightZero train_muzero
-> MuZeroPolicy._forward_collect
-> direct_ctree_gpu_latent collect/search hook
-> stock replay/target/learner remain unchanged
```

This is the right short path for real training-loop speed. It should be
fail-closed in `mode=train`, use real LightZero CTree, and prove
`direct_ctree_gpu_latent_calls > 0` with `fallback_calls == 0`. Its plausible
ceiling is roughly `1.2x-1.5x`. It is not compact ownership.

Lane 2 is the architecture bet:

```text
CompactRootBatchV1
-> real two-phase fixed-shape MCTS owner
-> selected actions drive env
-> delayed replay payload flush
-> compact replay index rows
-> compact sampler / learner-edge materialization
```

Lane 2 is why we have not safely implemented "the whole thing" in one pass:
it changes who owns the training facts. A fast row can be wrong while every
tensor shape is valid. The hard part is not only speed. It is preserving root
identity, player perspective, legal masks, terminal final observations, RND
latest frames, replay visibility, and learner targets across a new dataflow.

## Smallest Real End-To-End Slice

Build one local/profile-only compact loop that is real where it matters:

```text
HybridCompactBatch sequence
-> CompactRolloutSlab
-> FixedShapeMCTSSearchOwnerV0, new file
-> run_action_step returns selected_action only
-> CPU env applies selected joint action
-> flush_replay_payload on commit
-> CompactReplayIndexRowsV1
-> compact sampler materializes one learner-shaped batch
```

Required source surfaces:

- `src/curvyzero/training/compact_search_service.py`
  - `CompactSearchTwoPhaseServiceV1`
  - `CompactSearchActionStepV1`
  - `CompactSearchReplayPayloadV1`
  - `validate_compact_search_two_phase_payload_v1`
- `src/curvyzero/training/compact_rollout_slab.py`
  - `CompactRolloutSlab.step`
  - `_commit_previous`
  - selected-action feedback checks
- `src/curvyzero/training/compact_policy_row_bridge.py`
  - `CompactRootBatchV1`
  - `CompactSearchResultV1`
  - `CompactReplayIndexRowsV1`
  - compact index-row materialization and sampling helpers
- New: `src/curvyzero/training/fixed_shape_mcts_search_owner.py`
  - do not mutate the existing first-legal owner into an ambiguous backend
- Existing comparator/reference:
  - direct CTree compact service
  - `CompactSearchComparatorServiceV1`
  - MCTX/JAX only as a ceiling or algorithm-change comparator

The current `FixedShapeBatchedSearchOwnerV0` is useful as a two-phase contract
proof, but it is first-legal and reports `actual_search_simulations == 0`. It
must not appear in a real MCTS speed claim.

## Implementation Ladder

### L2.0: Freeze Labels And Denominators

Goal: make it impossible to confuse Lane 1, Lane 2, and fake ceilings.

Patch shape:

- Add or enforce metadata fields for every compact row:
  - `profile_only`
  - `trainer_ready`
  - `calls_train_muzero`
  - `not_lightzero_ctree`
  - `algorithm_change`
  - `actual_search_simulations`
  - `first_legal_policy`
  - `fallback_count`
- Summaries must state whether speed is:
  - real Coach training speed;
  - stock `train_muzero` profile speed;
  - compact/profile-only optimizer speed.

Gate:

- Any compact row missing these fields is not promotion eligible.
- Any row with `first_legal_policy=true` is a boundary/profile row only.

### L2.1: Real Fixed-Shape MCTS Owner, Eager First

Goal: replace first-legal with actual search before optimizing it.

Patch shape:

- Create `FixedShapeMCTSSearchOwnerV0`.
- Fixed capacity: `R = B * P`, `A = 3`, fixed `S`.
- Preallocate dense tensors/arrays:
  - visits `[R, S + 1, 3]`
  - value sums `[R, S + 1, 3]`
  - priors `[R, S + 1, 3]`
  - rewards `[R, S + 1, 3]`
  - child ids `[R, S + 1, 3]`
  - latent pool `[R, S + 1, ...]`
  - path node/action/active masks
- Use real model `initial_inference` and `recurrent_inference`.
- Implement masked PUCT select, expand, and backup.
- Start eager and preallocated. Try `torch.compile` or CUDA graphs only after
  eager correctness and baseline speed are understood.

Gate:

- `actual_search_simulations == requested_simulations`.
- `first_legal_policy == false`.
- Selected action comes from visit counts, not from the first legal action.
- Illegal action mass is exactly zero.
- Inactive roots are masked and cannot leak poison values.
- No CTree calls in this owner.

Kill:

- If the owner cannot run real simulations without position-based attachment,
  stop before profiling.
- If it silently falls back to first-legal, it is not Lane 2.

### L2.2: Two-Phase Hot Path

Goal: make the env-critical path action-only.

Patch shape:

- `run_action_step(root_batch)` performs real search and returns:
  - stable root identity;
  - `selected_action`;
  - replay payload handle.
- `flush_replay_payload(handle)` returns:
  - `visit_policy`;
  - `root_value`;
  - raw visit counts;
  - predicted value/logits if available.
- Replay payloads stay hidden until flushed and validated.

Gate:

- `CompactRolloutSlab.step` sees `action_step` and no full `search_result`.
- `_commit_previous` flushes exactly one replay payload per committed search.
- Reusing, dropping, or swapping a handle is a hard error.
- Action D2H bytes and replay payload D2H bytes are reported separately.

Kill:

- If replay payload work is merely moved off the timer and omitted from the
  denominator, the speed row is invalid.

### L2.3: Closed-Loop Action Feedback

Goal: prove the selected actions actually cause the next transition.

Patch shape:

- Build roots from compact env batch `k`.
- Search selects active-root actions.
- Convert to dense joint action `[B, P]`.
- Step the env.
- Commit replay rows using compact batch `k+1`.

Gate:

- `selected_action[k, env_row, player] == joint_action[k+1, env_row, player]`.
- Non-prefix active roots pass.
- Non-identity `policy_env_id` passes.
- Player 0 and player 1 both pass.
- Stale or out-of-order search result attachment fails.

Kill:

- Any candidate that only returns plausible actions but does not prove env
  application stays profile-only.

### L2.4: Compact Replay Visibility

Goal: rows become sample-visible only after all training facts exist.

Patch shape:

- Store compact index rows first, not full Python target rows.
- Hide incomplete rows behind a payload gate.
- Materialize learner-shaped tensors only at sampler or validation edge.
- Include terminal/final-observation and RND sidecars in row completeness.

Gate:

- Incomplete rows are hidden from sampler.
- A row becomes visible only after action, reward, done, visit policy, root
  value, final-observation sidecars, and RND sidecars are complete.
- `CompactReplayIndexRowsV1` materializes the same target rows as the trusted
  immediate path.
- Terminal next observation uses `final_observation`, not autoreset observation.

Kill:

- If full observation rows or stock `PolicyRowRecordV0` objects are rebuilt in
  the collect hot path, do not claim compact ownership speed.

### L2.5: Direct CTree Comparator Gate

Goal: compare against the real semantic oracle before believing speed.

Patch shape:

- Run deterministic no-noise fixtures through:
  - direct CTree compact service;
  - new fixed-shape MCTS owner.
- Use `CompactSearchComparatorServiceV1` or a small comparator script.

Gate:

- Single-legal fixtures match exactly.
- Clear-preference fixtures match selected actions.
- Root ids, env rows, players, and policy ids match.
- Visit distributions have zero illegal mass and bounded L1 distance.
- Root values have declared tolerance.
- Noisy/tie-heavy fixtures are labelled statistical, not exact.

Kill:

- If parity cannot be defined, label the owner as algorithm divergence.
- If divergence is acceptable, make it explicit; do not call it a CTree
  replacement.

### L2.6: RND And Terminal Canaries

Goal: prove compact ownership does not corrupt exploration or terminal targets.

Patch shape:

- Add mixed live/terminal compact batches.
- Include autoreset.
- Include both players.
- Include `rnd_meter_v0` with zero weight first.
- Later include positive RND only with explicit reward schema.

Gate:

- Terminal roots are not searched.
- Live roots are still searched.
- Terminal final observation is stable even if resident/live buffers mutate.
- RND latest frame equals the policy latest frame for the same
  `(env_row, player)`.
- Meter-only RND does not change target rewards.
- RND metrics and cadence fields are present when RND is enabled.

Kill:

- Any RND row without identity sidecars is not a correctness row.
- Any positive RND row without explicit reward schema is an algorithm-change row.

### L2.7: Same-Denominator Compact Profile

Goal: measure speed only after correctness counters are clean.

Rows:

```text
direct_ctree_gpu_latent compact slab
new fixed_shape_mcts_search_owner
service_tax_probe ceiling, labelled not MCTS
mock_search_service ceiling, labelled fake search
MCTX/JAX ceiling, labelled algorithm-change unless model/replay parity is real
```

Required counters:

- `actual_search_simulations`
- `ctree_calls`
- `python_root_lists_built`
- `python_simulation_payloads_built`
- `per_sim_d2h_bytes`
- `action_d2h_bytes`
- `replay_payload_d2h_bytes`
- `python_rows_materialized`
- `rnd_materialized_rows`
- `compact_replay_rows_visible_before_payload`
- `fallback_count`
- `illegal_action_count`
- `identity_mismatch_count`
- `stale_payload_count`

Gate:

- Forbidden counters are zero.
- Same shape, same `R`, same `S`, same masks, same death/RND mode.
- Speed includes action readback, replay payload flush, compact replay commit,
  sampler edge, and any learner-edge materialization included in the claim.
- Continue only if the fixed-shape owner beats direct CTree compact by at least
  `25-30%` after gates are clean.

Kill:

- If the only win is from omitting payloads the learner still needs, kill the
  speed claim.
- If speed is below the threshold but tests are useful, keep the tests and
  demote the backend.

### L2.8: Learner-Edge Attachment

Goal: prove compact samples can feed the learner boundary without stock hot-path
object fanout.

Patch shape:

- Sample compact replay groups.
- Materialize one learner-shaped batch.
- Compare tensor shapes and target values to the trusted target-row builder.
- Do not claim stock `train_muzero` integration yet unless it actually calls the
  real learner path with these samples.

Gate:

- Sampled batch has expected observation/action/reward/policy/value shapes.
- Target rows match the oracle on deterministic fixtures.
- Learner-visible samples never include incomplete rows.
- Materialization cost is in the denominator.

Kill:

- If stock replay/target objects must be rebuilt for every collected row, Lane 2
  has not escaped the old boundary.

### L2.9: Coach Candidate Gate

Goal: decide whether Lane 2 can leave profile/local status.

Required before any Coach recommendation:

- Lane 1 remains the trusted practical train path.
- Lane 2 has a real MCTS owner with nonzero actual simulations.
- Compact replay/sample parity passes.
- RND and terminal gates pass.
- No profile-only or fake-search labels are present in the promoted row.
- A small learning proof shows comparable target rows and learner consumption.
- The result summary says exactly which speed currency is being reported.

Coach-facing kill:

- Any fallback in a promoted row.
- Any missing identity, final-observation, RND, or sample-visibility proof.
- Any attempt to report compact roots/sec as real Coach learner throughput.

## What Must Not Be Faked

- Real MCTS:
  - no first-legal;
  - no mock/service-tax as a backend claim;
  - no precomputed recurrent payloads in speed rows;
  - `actual_search_simulations > 0`.
- Action feedback:
  - selected actions must be the next env joint actions.
- Replay:
  - visit policy, root value, reward, done, final observation, and RND sidecars
    must exist before sample visibility.
- Identity:
  - attach by stable root/env/player/policy ids, not array position.
- Terminal semantics:
  - final observation before autoreset, never reset frame as terminal next obs.
- RND:
  - meter mode does not alter rewards;
  - positive RND is explicit reward-schema change.
- Denominator:
  - include action D2H, replay payload flush, compact commit, sampler/materializer
    edge, and learner-edge cost if claimed.
- Labels:
  - MCTX/JAX shadow, compact Torch, service-tax, and mock rows stay labelled until
    their exact semantic lane is proven or accepted as an algorithm change.

## Parallel Work

These can run in parallel because they touch different risk surfaces:

- Lane 1 train-hook landing:
  - fail-closed `direct_ctree_gpu_latent` in stock `train_muzero`;
  - proof fields and matched train/profile rows.
- Lane 2 MCTS owner:
  - new fixed-shape eager MCTS owner and direct CTree comparator fixtures.
- Replay visibility:
  - payload gate, stale-handle tests, compact sampler hiding incomplete rows.
- Resident observation canaries:
  - latest stack parity, terminal final-observation copy, both-player perspective.
- RND canaries:
  - latest-frame identity, meter-neutral rewards, cadence counters.
- Profiling/summarizer hardening:
  - required counters, forbidden counters, speed-currency labels.
- MCTX/JAX ceiling:
  - keep as a separate comparator and memory/scaling reference, not a silent
    replacement for LightZero CTree.

Do not run these in parallel as one giant train rewrite. Lane 2 should earn each
boundary before it becomes trainer-facing.

## Bottom Line

Lane 1 is being landed because it is practical, real-search, and stock-trainer
compatible.

Lane 2 is not implemented end to end yet because the real deliverable is not
"faster search" in isolation. It is a compact owner that preserves the whole
training fact chain:

```text
observation k
-> real search action k
-> env transition k+1
-> replay row k
-> learner-visible sample
```

The smallest honest next slice is a real two-phase fixed-shape MCTS owner inside
`CompactRolloutSlab`, followed by replay visibility and sampler gates. Only then
do speed numbers become architecture evidence instead of another attractive
profile mirage.
