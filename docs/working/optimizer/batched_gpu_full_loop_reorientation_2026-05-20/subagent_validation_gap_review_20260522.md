# Subagent Validation Gap Review - 2026-05-22

Scope: validation/test critique for the CurvyTron optimizer lane, focused on
`direct_ctree_gpu_latent` after the scenario compare modes and train-facing hook
schema smoke. I read the requested docs, script, tests, and trainer hook source.
I did not edit production code, run live jobs, touch checkpoints, or launch
Modal work.

Reviewed anchors:

- `direct_ctree_promotion_gates_20260522.md`
- `boundary_test_matrix_20260521.md`
- `scripts/compare_curvytron_direct_ctree_stock.py`
- `tests/test_lightzero_phase_profiler.py`
- `tests/test_source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`

## Short Answer

The new scenario gates are necessary but not sufficient. They are good at
falsifying action-mask bugs in the profile-side compact-array path, especially
`single_legal_cycle` and `mixed_legal_cycle`. They do not fully validate the
train-facing `_forward_collect` hook because that hook has a separate output
assembler, a separate all-actions-legal fast path, and returns raw
legal-action visit lists for the stock collector rather than normalized
full-action compact arrays.

The hook schema smoke is a useful first tripwire. It proves install/restore,
one all-legal fast-path output shape, ready-env-id preservation, and the main
counters. It is not enough for Coach-facing confidence. A shape-correct output
dict can still change training targets if `action` is a legal-list index instead
of a full action id, if `visit_count_distributions` are normalized full-action
probabilities instead of raw legal-action visit counts, or if stock replay/target
construction consumes the fields differently than the smoke assumes.

Before deeper architecture experiments, add small local tests around the
train-facing hook and replay/target consumer contract. Do not make ordinary
neutral/tie-heavy MCTS action equality an exact gate. Keep exact tests for
forced masks, clear preferences, schema, fail-closed validation, row/env-id
mapping, and same-model values/logits. Keep stochastic collect rows and wall
clock improvements statistical.

## Question 1: Are Scenario Gates Enough For Masks?

No, not by themselves.

What they cover well:

- `all_legal` hits the vectorized all-actions-legal path and is the right place
  to require deterministic selector parity and zero illegal actions.
- `single_legal_cycle` is the cleanest exact gate for full-action-id mapping.
  Any legal-list-index/full-action-id confusion should surface immediately.
- `mixed_legal_cycle` exercises rows with one, two, and three legal actions, so
  it is the right forced-case gate for compact legal-list ordering.
- `random` is useful as a broader distributional smoke, not as an exact parity
  proof.

What remains uncovered or under-covered:

- The compare script drives
  `_LightZeroCollectForwardStackProbe` from
  `source_state_batched_observation_boundary_profile.py`, not the train-facing
  `_install_lightzero_collect_search_backend_hook(...)` implementation in
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py`.
- The sidecar compact arrays expand visits to full action width and normalize
  them. The train hook must return stock-shaped raw legal-action visit counts.
  A bug can pass compact-array checks and still poison policy targets.
- The compare summary reports illegal decoded actions, but it does not expose an
  explicit `illegal_visit_mass` metric by scenario. In mixed-mask rows, that
  should be a first-class gate, not an inference from visit L1.
- Fractional and zero masks are covered locally for the sidecar probe, but the
  train-facing hook needs its own fail-closed tests.
- The all-actions-legal fast path bypasses stock `select_action(...)`. It needs
  direct parity checks for deterministic collect and distributional checks for
  stochastic collect.

Conclusion: keep the scenario gates, but classify them as profile-boundary
gates until the same mask scenarios are exercised through the train hook output
contract.

## Question 2: Is The Hook Schema Smoke Enough?

No. It is the right first test, but it is still a smoke.

Current value of
`test_direct_ctree_collect_search_hook_matches_stock_collect_output_schema`:

- proves the hook can install and restore without leaving
  `MuZeroPolicy._forward_collect` patched;
- proves unsupported modes can fall back to stock and count fallback calls;
- proves a fake all-legal CUDA-latent collect path returns the same outer keys
  and field names as stock;
- proves the all-actions-legal output fast path can match one deterministic
  fake stock selection with `epsilon=0`;
- proves the compact profile output surfaces the direct-hook counters and
  semantic identity fields.

Missing proof:

- mixed legal masks through the train hook slow output path;
- single-legal forced rows through the train hook;
- raw legal-action visit-count length and ordering as consumed by stock
  collectors;
- explicit illegal output/action validation in the hook result, not just
  sidecar telemetry;
- stochastic collect behavior when `eps_greedy=False` or `epsilon>0`;
- parity after the output dict is consumed by stock collector/GameSegment or the
  target audit path;
- terminal/live batches where zero-mask terminal roots are filtered upstream but
  terminal timesteps and final observations still reach replay;
- a real GPU-latent helper canary proving path/batch latent indexing and
  reward/value/logit readback order.

The stronger test is hook-vs-stock parity at the consumer boundary: call stock
and direct with the same fake model, fake roots, same `ready_env_id`, and the
same masks, then compare both the returned dict and the stock collector/replay
fields derived from it.

## Question 3: Exact Small Tests To Add Next

### P0. Train Hook Mixed-Mask Raw-Visit Contract

Add a unit test beside `test_direct_ctree_collect_search_hook_matches_stock_collect_output_schema`.
Use fake roots with nontrivial legal masks such as:

```text
row0 mask [1, 0, 1], root visits [2, 7] -> action must be full id 2
row1 mask [0, 1, 0], root visits [9]    -> action must be full id 1
row2 mask [1, 1, 0], root visits [5, 1] -> action must be full id 0
```

Pass criteria:

- `collect_search_backend_output_fast_path_calls` stays zero;
- `action` is a full action id and is legal under that row mask;
- `visit_count_distributions` is the raw legal-action list returned by roots,
  not full-width and not normalized;
- `predicted_value`, `predicted_policy_logits`, `searched_value`, entropy, and
  `ready_env_id` mapping match the stock fake output;
- mismatched visit-list length versus legal-action count fails closed.

This is the smallest test that catches the most dangerous policy-target bug.

### P0. Train Hook Single-Legal Exact Gate

Use one legal action per row, with each row forcing a different action. Require
exact action equality, exact one legal visit list per row, and no fallback.

This should be separate from mixed masks because it gives a very crisp diagnosis:
any selected action other than the sole legal full action is wrong.

### P0. Train Hook Fail-Closed Mask Validation

Add direct hook tests for:

- fractional masks reject before model/search;
- all-zero masks reject before model/search;
- `action_mask` length mismatch rejects;
- `ready_env_id` length mismatch rejects;
- unsupported LightZero modes still fall back to stock and increment
  `collect_search_backend_fallback_calls`.

The sidecar probe has some of this coverage. The train hook should have its own
because it is the path that now affects the `train_muzero` denominator.

### P0. Hook Output To Stock Collector/Replay Canary

Add a tiny local consumer test that feeds stock and direct hook outputs into the
same stock collector/GameSegment path, or into a minimal adapter that calls the
same `store_search_stats` contract.

Compare:

- `action`;
- raw `visit_count_distributions`;
- searched value/root value;
- predicted value and policy logits if carried by the segment/audit path;
- previous action mask and `to_play`;
- reward/done/final-observation metadata for a terminal row and a live row.

The point is not to run a trainer. The point is to prove that a stock-shaped dict
really becomes the same replay/target material, not merely that it has the same
keys.

### P0. GPU-Latent Helper Row/Column Sentinel

Add a fake `tree_muzero.batch_traverse` plus fake model canary for
`_direct_ctree_gpu_latent_search_for_collect(...)`.

Use latent path indices, batch indices, and last actions that are deliberately
not identity ordered. Make the fake recurrent model return reward/value/logit
sentinels that encode `(path_index, batch_index, action)`. Assert
`batch_backpropagate(...)` receives the expected per-root reward/value/logit
rows.

This catches the silent class of bug where search still runs, but the leaf
latent gather or readback row order is transposed.

### P1. All-Actions-Legal Fast-Path Selector Parity

Keep the current all-legal schema smoke, then add:

- deterministic `eps_greedy=True, epsilon=0` exact parity over several visit
  matrices, including ties with a documented tie policy;
- stochastic `eps_greedy=False` distributional parity over many seeds;
- `epsilon>0` distributional parity for epsilon exploration.

Do not require exact per-seed parity for stochastic rows if vectorized random
draws consume RNG differently than stock. Require close empirical action
frequencies, zero illegal actions, finite entropies, and stock-compatible output
fields.

### P1. Compare-Script Gate Tightening

For `scripts/compare_curvytron_direct_ctree_stock.py`, add or require:

- a top-level illegal visit mass summary for stock and candidate under each
  mask scenario;
- a small unit test for `_action_mask(...)` scenarios so the forced masks cannot
  drift;
- recommended command presets for exact forced gates versus statistical random
  gates.

Suggested classification:

```text
single_legal_cycle: strict exact
mixed_legal_cycle with clear preferences: strict exact for actions and illegal mass
all_legal deterministic clear preference: exact actions/values/logits
random/tie-heavy/root-noise: statistical visit/action agreement
```

### P1. Profile Summary Promotion Gate

Extend summary/compact-output tests so a direct row cannot be promoted unless:

- `called_train_muzero == true`;
- `collect_search_backend == direct_ctree_gpu_latent`;
- `collect_search_backend_fallback_calls == 0`;
- direct hook call count and output row count are positive and internally
  consistent;
- GPU-latent path was actually enabled where required;
- eval/checkpoint/GIF sidecars are disabled or explicitly accounted for;
- semantic identity includes death mode, RND mode, env-step source, observation
  contract, scalar materialization, `to_play`, mask semantics, and CPU-tree
  inclusion.

This is the guard against turning a fast under-specified profile row into a
Coach recommendation.

## Question 4: What Stays Statistical?

Keep exact gates for deterministic semantics:

- full-action-id mapping under forced masks;
- zero illegal actions and zero illegal visit mass;
- fail-closed fractional and empty masks;
- output schema, ready-env-id order, row/player order, and `to_play`;
- raw legal-action visit-list length/order for stock collector output;
- same-model predicted values and policy logits;
- terminal/live row accounting and final-observation shape/presence;
- profile attestation fields and fallback counters.

Keep these statistical:

- action agreement on neutral or tie-heavy MCTS rows, because stock CTree can
  allocate equal visits differently and tie-breaking is not a training semantic;
- visit distribution agreement under root Dirichlet noise, stochastic action
  selection, or epsilon exploration, because random draw ordering can differ
  between vectorized and per-row assembly while preserving the same policy;
- full-loop throughput, because GPU scheduling, Python overhead, data loading,
  and profiler noise require matched repeats rather than exact equality;
- RND learning strength, because predictor loss/novelty is inherently
  distributional. Meter-mode safety remains exact where it should be exact:
  target network unchanged and target rewards unchanged.

The practical rule is: exact where the training meaning is discrete or
schema-shaped; statistical where the underlying algorithm is stochastic or
system timing is noisy.

## Top Gaps Before Coach-Facing Advice

1. The train hook lacks mixed-mask and single-legal exact parity tests.
2. The hook schema smoke does not prove raw legal-action visit counts survive
   into stock replay/target material.
3. The all-actions-legal output fast path lacks stochastic parity coverage.
4. The GPU-latent helper lacks a row/column sentinel for latent gather and
   reward/value/logit unpack order.
5. Promotion still needs a summary gate that rejects under-attested direct rows
   before they become recommendations.

## Recommendation

Do the small P0 tests above before any deeper architecture experiment. The next
architecture lane can still be array-native CTree, compiled dense search, or a
search service, but those experiments should inherit a train-facing parity
surface that already proves masks, output meaning, replay fields, and
attestation. Otherwise the optimizer can get faster while silently optimizing a
different training problem.
