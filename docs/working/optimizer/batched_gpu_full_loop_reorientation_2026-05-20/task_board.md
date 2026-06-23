# Task Board

Date: 2026-05-20

## 2026-05-23h Reorientation Board

- [x] Re-read the current optimizer docs and Coach reset docs. Current center:
  keep the trusted Coach training path untouched; use profile-only optimizer
  gates to prove a faster search/dataflow backend.
- [x] Record the plain current worldview in
  `reorientation_now_20260523h.md`.
- [ ] Sidecar checkpoint inventory: find a current immutable
  `iteration_N.pth.tar` ref for the LightZero PyTorch -> JAX checkpoint parity
  probe.
- [ ] Sidecar rabbit-hole critique: check whether the JAX/MCTX bridge remains
  the right next gate or should be killed.
- [x] If a checkpoint ref is found, run the Modal checkpoint parity probe.
  Current immutable `iteration_260000` probe loaded strictly and consumed all
  inference weights, but raw latent tensors failed strict tolerance. This is a
  partial diagnostic result, not a pass.
- [ ] Finish the checkpoint parity diagnostics: scalar support values and
  recurrent-from-PyTorch-latent comparisons.
  2026-05-23h: recurrent-from-PyTorch-latent says dynamics/prediction are
  mostly aligned; root representation drift remains. Practical `5e-3` checkpoint
  gates are running on CZ26 `iteration_260000` and r18fresh champion
  `iteration_250000`.
- [ ] If checkpoint parity passes, wire the real JAX shadow model into
  `MctxCompactSearchServiceV1` behind `CompactSearchServiceV1`, still
  profile-only.
  2026-05-23i: first bridge slice landed. `MctxCompactSearchServiceV1` accepts
  an optional real `JaxMuZeroShadowModel`; the Modal hybrid profile can mount
  the runs volume and load an immutable checkpoint into that backend. Local
  ruff and focused tests passed. Tiny L4 real-checkpoint compact-slab smoke
  passed. H100 matched rows now show real-checkpoint MCTX beating direct CTree
  by `1.36x` at B64/sim8, `2.37x` at B512/sim8 with scalar rows on, `1.58x`
  at B512/sim8 with scalar rows off, and `2.20x` at B1024/sim8 with scalar rows
  off. This remains profile-only.
- [x] Fold the B1024 scalar-off pair into the real-checkpoint MCTX read.
  Result: MCTX real checkpoint `19,334` steps/sec vs direct CTree `8,792`
  steps/sec on the same H100 profile-only denominator.
- [ ] Add search-impact parity: same compact roots through direct CTree and
  real-checkpoint MCTX, compare selected actions, visits, and root values.
  2026-05-23j partial: `CompactSearchComparatorServiceV1` landed locally and
  tiny H100 sim2 smoke passed identity (`identity_match=true`) with zero scalar
  timestep materialization. It also found a semantic warning: only `2/4`
  selected actions matched, visit L1 mean was `1.46`, and root-value abs diff
  mean was `57.4`. Larger sim8 rows repeated the warning: action match was
  `34.4%` vs `direct_ctree_gpu_latent` and `37.5%` vs `direct_ctree_arrays`,
  with visit L1 mean around `1.28` and root-value abs diff mean around `15.97`.
  Pre-search predicted-value/logit telemetry then showed the model/input bridge
  is close: policy-logit mean/max diff `0.0000084/0.0000203`, predicted-value
  mean/max diff `0.0090/0.0219`. Current read: the remaining mismatch is search
  semantics/value backup, not root order or checkpoint loading.
- [ ] Design the compact replay/trainer boundary that avoids scalar LightZero
  timestep rows on the hot path. The scalar edge has already been priced; the
  next proof is compact-in, compact-stored, sampled-batch-out with zero
  `MockBaseEnvTimestep` rows.
- [ ] Keep live Coach training runs read-only from this optimizer lane.

## 2026-05-23c Immediate Board

- [x] Fix compact slab summary currency before scaling. Validation subagent
  caught that the manifest summary could divide all roots by the last slab
  search call. The runner now uses aggregate `compact_rollout_slab_sec` for
  `probe_total_sec`; last-call fields are explicit `compact_rollout_slab_last_*`.
  The profile loop now emits `compact_rollout_slab_telemetry_totals` for new
  rows.
- [x] Monitor active H100 slab wave:
  `opt-compact-slab-h100-main-20260523d`,
  `opt-compact-slab-h100-direct-arrays-controls-20260523d`,
  `opt-compact-slab-h100-service-tax-20260523d`,
  `opt-compact-slab-h100-mock-ceiling-20260523d`,
  `opt-compact-slab-h100-actor-tax-a8-20260523d`, and
  `opt-compact-slab-h100-actor-tax-a32-20260523d`.
  Do not promote any row to Coach; these are profile-only denominator rows.
  Result: best LightZero CTree baseline/control row from that old slab wave is
  `direct_ctree_gpu_latent` B1024/A16/sim8 at `8291` steps/sec; B1024/A16/sim16
  is `6992`; B1024/A16/sim32 is `5314`. This is not the current best optimizer
  row. Latest clean current row is real-checkpoint MCTX B1024/A16/sim8
  scalar-off: `19,334` vs direct CTree `8,792`, speedup `2.20x`.
  Detailed summary:
  `compact_slab_h100_profile_summary_20260523d.md`.
- [ ] Next optimizer move: attack the search/control backend behind
  `CompactSearchServiceV1`. Current ceilings suggest roughly `1.7x-2.8x`
  remaining in this slab shape from service-tax/mock rows, not a clean 10x from
  wrapper cleanup alone.
- [x] Add a first profile-only MCTX backend behind `CompactSearchServiceV1`.
  `MctxCompactSearchServiceV1` now runs through the compact slab, returns a
  checked `CompactSearchResultV1`, stages selected actions into the next env
  step, and commits compact replay-index rows. Tiny H100 Modal smoke passed with
  `ok=true`, `compact_rollout_slab_total_roots=16`,
  `compact_rollout_slab_committed_index_row_count=16`,
  `search_impl=mctx_compact_search_service_profile_only_v0`, and
  `calls_train_muzero=false`.
- [x] Run the real MCTX comparator grid on H100 B512/B1024, sim16/sim32, and
  compare against `direct_ctree_gpu_latent`.
  Result: MCTX/JAX profile-only compact slab clearly beat direct CTree on this
  denominator: B512/sim16 `16,250` vs `6,522` steps/sec (`2.49x`),
  B512/sim32 `14,306` vs `4,177` (`3.43x`), B1024/sim16 `20,557` vs `6,992`
  (`2.94x`), B1024/sim32 `16,255` vs `5,314` (`3.06x`). Keep this lane, but
  keep it profile-only.
- [ ] Run the next MCTX scaling/robustness grid before any architecture claim:
  H100 and L4, B512/B1024/B2048, sim8/sim16/sim32/sim64, same compact slab,
  no scalar materialization, same profile-only labels.
- [x] Add the small MCTX validation slice before overreading the speed rows.
  Focused validation now checks stronger profile-only labels, no-legal-action
  rejection, inactive-root rejection under fixed-shape mode, illegal
  `raw_visit_counts`, and MCTX timing promotion into slab summary fields.
  Local validation: ruff clean; focused tests `13 passed`.
- [ ] Fold the MCTX sidecar critiques into the next decision. Current sidecar
  read: MCTX is a real speed signal, but the exact multiplier needs matched
  direct baselines; the training bridge starts with PyTorch-to-JAX model parity,
  not with a Coach run.
- [x] Run a matched 40/10 direct baseline for the MCTX rows. Result: MCTX still
  wins by `2.36x-3.04x` on H100 B512/B1024 sim16/sim32 and `3.14x-5.55x` on
  L4 B512/B1024 sim16/sim32.
- [ ] Run the stricter H100 80/20 direct-vs-MCTX comparison now in flight:
  `opt-mctx-strict-h100-8020-20260523f` and
  `opt-direct-strict-h100-8020-20260523f`.
- [ ] Decide the next implementation shape after the wider grid:
  either deeper MCTX/JAX ownership, a Torch-to-JAX/model bridge spike, or a
  different compiled fixed-shape search backend behind `CompactSearchServiceV1`.
- [x] Start the Torch-to-JAX model bridge spike. Added a profile-only JAX
  shadow-model parity module, script, Modal wrapper, and local tests. Fresh
  LightZero MuZero model parity passes on Modal L4 GPU with explicit GPU
  tolerance `5e-4`; all required inference keys are consumed. This does not
  call MCTX or `train_muzero`. Status doc:
  `lightzero_jax_shadow_parity_status_20260523g.md`.
- [ ] Run the checkpoint parity wrapper on a current immutable
  `iteration_N.pth.tar` checkpoint. An old documented checkpoint ref was not
  present in the current runs volume, so no checkpoint parity claim exists yet.
- [ ] If current checkpoint parity passes, wire the real JAX shadow model into
  `MctxCompactSearchServiceV1` behind `CompactSearchServiceV1`, still
  profile-only.
- [x] Reorient at the right level. Current read: take a bigger optimizer move,
  but do not promote a new Coach backend yet.
- [x] Record the new plain-language world model in `world_model.md`.
- [x] Fold validation-ladder and MCTX comparator reports into the active
  decision: both support profile-only gates before Coach-facing claims.
- [x] Fold fixed-A3 CTree report. Important correction: flat-A3 already exists
  in the stock `train_muzero` profile shell, but the matched evidence
  `516.55` vs `509.69` steps/sec caps it for the tested denominator. It is not
  the next 5x lane.
- [x] Fold profile-tooling report. Next optimizer sidecar wave should use
  `scripts/build_curvytron_hybrid_observation_profile_grid.py` plus
  `scripts/run_curvytron_hybrid_observation_profile_manifest.py`; stock
  full-loop profiles and MCTX sidecars are separate currencies unless we build
  a unifying runner.
- [ ] Next concrete experiment candidate: stronger fixed-shape search/dataflow
  backend behind `CompactSearchServiceV1`, compared against direct CTree and
  service-tax/mock on the same H100 denominator.
- [ ] Keep MCTX as a side comparator, not a Coach path, unless there is an
  explicit framework migration plan.
- [ ] Keep live Coach training runs read-only from this optimizer lane.

## 2026-05-23b Strategy Reset Board

- [x] Reorient around the actual question: are we ready for a big move or a
  full strategy reset? Current read: make a bigger optimizer move in the
  profile lane, but do not promote it to Coach. New note:
  `strategy_reorientation_20260523b.md`.
- [x] Dispatch parallel critique/research wave:
  current bottleneck/Amdahl, external fast-RL patterns, validation ladder, and
  big architecture options.
- [ ] Fold the 2026-05-23b subagent reports into this board and
  `world_model.md`.
- [ ] Choose the next concrete implementation after the reports return.
  Default bias: keep `CompactSearchServiceV1`, kill eager compact Torch as the
  main lane, and test array-native/fixed-A CTree or compiled/fused fixed-shape
  search behind the same contract.
- [ ] Before any new speed claim, run or build a same-denominator proof that
  labels the row as one of: Coach training, stock full-loop profile, or
  optimizer probe.

## 2026-05-23 Dataflow / Architecture Board

- [x] Current parallel architecture critique wave:
  Popper = device-resident/search-service architecture,
  Ohm = non-search materialization audit,
  Parfit = external high-throughput RL pattern research,
  Carver = failure-mode validation. Fold results into
  `full_iteration_dataflow_designs_20260523.md` before choosing the next
  implementation lane.
  2026-05-23 Popper result folded: leading target is CPU batched env plus
  device-resident observation/search service, selected-action-only critical
  sync, delayed replay/RND payload flush, and stock LightZero objects only at
  validation/debug/sample edges.
  2026-05-23 Ohm/Parfit/Carver folded: after search improves, scalar timestep
  splitting, full target/replay rows, RND per-frame materialization, and
  root-observation copies become the next wall; external fast-RL systems point
  to static slabs, shared buffers, batched inference/search, and delayed sync;
  every aggressive path must prove the full
  `observation -> search -> action -> env -> replay -> sample` chain.
- [x] Current parallel wave 2:
  Anscombe = concrete code insertion points,
  Russell = architecture candidates and falsifiers,
  Fermat = validation ladder,
  Turing = external systems follow-up. Fold these before choosing a larger
  implementation lane.
  2026-05-23 folded. Main read: first insertion point is the profile manager
  after compact state/root sidecars exist and before scalar LightZero timestep
  materialization; the next implementation lane should be a named compact slab
  plus two-phase compact search-service ownership, with validation gates for
  out-of-order service ids, incomplete-row visibility, mixed terminal/live
  rows, RND meter metrics, and summary promotion locks.
- [x] Start the full iteration dataflow working-memory doc:
  `full_iteration_dataflow_designs_20260523.md`.
- [x] Fold the design-matrix sidecar into the planning doc. Current read:
  compact service boundaries are valuable, but a real multi-x win needs
  compiled/fused search, array-native CTree, MCTX/JAX comparator, or larger
  native buffer ownership. Eager Torch search is profile-only.
- [x] Fold the validation sidecar into the planning doc. Current read: lock
  the denominator before any speed claim; no-noise parity gates and RND/player/
  terminal checks come before trainer-facing promotion.
- [x] Fold the sync/data movement sidecar into the planning doc. Current read:
  one selected-action readback is acceptable while env mechanics are CPU; replay
  payload readback, full observation copies, root copies, Python row/list
  materialization, and RND materialization must be measured separately.
- [x] Fold the final full-dataflow map sidecar into the planning doc and world
  model. Current read: at B512/P2, stack payloads are tens of MiB while search
  outputs are tiny; the remaining trap is stock replay/learner/RND pulling the
  compact path back into full observation materialization.
- [x] Add or verify telemetry for the next sync ledger:
  `obs_h2d_bytes`, `mask_h2d_bytes`, `action_d2h_bytes`,
  `replay_payload_d2h_bytes`, `root_observation_copy_bytes`,
  `python_rows_materialized`, `rnd_materialized_rows`, and
  `resident_obs_reused`.
  2026-05-23 partial: compact Torch search-service profile path now emits the
  corresponding `lightzero_array_ceiling_*` ledger fields; the hybrid profile
  result now aggregates them under `batched_stack_probe_ledger_totals`; direct
  CTree profile path emits matching `lightzero_mcts_arrays_boundary_*` ledger
  fields; the durable manifest compact summary surfaces the common columns.
  Remaining follow-up, tracked separately: add action-critical wait vs
  replay-payload readback timing if we keep pushing this backend.
- [ ] Add action-critical wait vs replay-payload readback timing if the next
  service/backend lane needs it.
- [ ] Choose the next same-denominator experiment wave. First choice should be
  small and honest: direct CTree vs compact Torch service vs service-tax,
  root-noise off, sim16/sim32, RND meter compatibility, and one input-handoff
  ceiling row if the denominator still points there.
  2026-05-23 done for direct CTree, compact Torch, service-tax, and mock at
  H100/L4, sim16/sim32, root_noise=0.0. H100 result: compact Torch loses
  against direct CTree; service-tax and mock still show headroom. Next wave
  should not polish eager Torch search.
- [ ] Next implementation lane: profile-only compact slab / two-phase search
  service insertion around `HybridBatchedObservationProfileManager.step`.
  Preserve the existing compact contracts, keep live Coach training untouched,
  and start with validation for selected-action feedback plus delayed payload
  sample-visibility.
  2026-05-23 first slice landed: `CompactSearchActionStepV1` and
  `CompactSearchReplayPayloadV1` split a `CompactSearchResultV1` into
  action-critical and replay-critical halves. The validator fails closed if
  delayed replay payload ids no longer match the selected-action step. Local
  validation: compact replay/search contract file, compact Torch service file,
  manifest runner file, and focused boundary compact-service tests passed.
  Remaining: connect this two-phase contract into the profile manager/slab path
  and add incomplete-row sample-visibility tests.
  2026-05-23 follow-up: `CompactSearchPayloadGateV1` now keeps an action step
  sample-invisible until its delayed replay payload is attached and
  identity-checked. It allows out-of-order payload completion by handle, but
  rejects missing, duplicate, stale, or mismatched handles/ids. Local validation:
  `34 passed` for compact search/replay + compact Torch + manifest runner, and
  focused boundary compact-service tests `3 passed`.
  2026-05-23 second follow-up: the profile replay proof now stages the next env
  joint action from `CompactSearchActionStepV1` instead of full search arrays,
  then flushes `CompactSearchReplayPayloadV1` separately through the payload
  gate. Focused source-state compact replay proof tests: `4 passed`.
  2026-05-23 slab slice: added `CompactRolloutSlab` as a profile-only owner in
  `src/curvyzero/training/compact_rollout_slab.py`. It stages selected actions,
  requires the next compact batch to apply them, then commits previous search
  rows into `CompactReplayIndexRowsV1`. It is not wired into Coach or Modal.
  Validation: ruff clean, main focused compact bundle `74 passed`, focused
  boundary compact-service tests `3 passed`.
  2026-05-23 opt-in manager wiring: `HybridBatchedObservationProfileManager`
  now accepts `compact_rollout_slab=None`. When supplied, it builds the compact
  batch before scalar materialization, calls the slab, records timing, and
  returns slab telemetry/step on `HybridObservationProfileStep`. Defaults are
  unchanged. Validation: ruff clean, focused compact/slab/two-phase/boundary
  bundle `16 passed, 174 deselected`; full local profile/contract files passed:
  `38 passed`, `115 passed`, and `22 passed`.
  2026-05-23c Modal/tooling wiring: the hybrid boundary profile entrypoint now
  exposes `hybrid_compact_rollout_slab_probe`, the grid builder emits
  `--hybrid-compact-rollout-slab-probe`, compact JSON keeps slab fields, and
  the durable manifest summary reports slab calls/roots/committed rows. Guard:
  slab and old `compact_service_replay_proof` are mutually exclusive so search
  is not double-run. Validation: ruff clean; focused local compact/builder/
  boundary/runner bundle `12 passed`; tiny L4 Modal smoke returned `ok=true`
  with `compact_rollout_slab_enabled=true`, `calls=1`, `total_roots=4`,
  `committed_index_row_count=4`, and `search_impl=mock_search_service`.
  2026-05-23d sample-gate wiring: the hybrid profile core and Modal wrapper now
  expose `hybrid_compact_rollout_slab_sample_gate`. This is still profile-only,
  but it proves the compact slab can go
  `CompactReplayIndexRowsV1 -> target rows -> sample batch` without
  materializing scalar `BaseEnvTimestep` rows. Guard: it requires
  `hybrid_compact_rollout_slab_probe` and
  `--no-hybrid-materialize-scalar-timestep`. Local validation: ruff clean;
  focused profile core, boundary, grid-builder, and manifest-runner tests
  passed (`9 passed` in the broad local gate). Tiny L4 Modal smoke
  `ap-g8C6fnvHtHym386bAq7odu` passed with scalar timesteps off, sample gate on,
  index/target/sample rows all `8`, and mock `BaseEnvTimestep` rows `0`.
  2026-05-23e sample-gate cadence fix: added sample-gate batch-size and
  interval knobs so profile rows can distinguish exhaustive replay-contract
  proof from realistic learner-sample cadence. H100 matched rows:
  direct-CTree B64/A4/sim8 baseline `3660` steps/sec; sample every step/all rows
  `1064`; sample every step/batch64 `1532`; sample once per 20 opportunities
  with batch64 `3428`. Read: the sample edge is a small tax at realistic
  cadence and still avoids scalar `BaseEnvTimestep` rows.
- [x] Add a promotion-lock summary guard for profile-only hybrid profile rows.
  The manifest runner now prints `profile_only`, `calls_train_muzero`,
  `touches_live_runs`, `promotion_eligible=false`, and
  `promotion_blocker=profile_only_boundary_probe` in compact summaries. This
  protects service-tax/mock/compact-Torch rows from being read as Coach launch
  advice.
- [ ] Do not touch live Coach training runs from this optimizer lane.

## 2026-05-22 Current Optimizer Board

- [x] Fix the expensive MCTX root-value extraction path. `_extract_mctx_root_values`
  now reads the root node value directly before falling back to
  `search_tree.summary()`. This changed root-value extraction from about
  `0.26-0.31s` to about `0.014-0.024s` on the H100 B1024/P2 loop24 shape.
- [x] Re-run replay-valid rows after the direct-root fix. Result:
  sim16 replay-valid `54,977` roots/sec with replay-index cost `0.010s`;
  sim32 replay-valid `38,122` roots/sec with replay-index cost `0.012s`.
  Root payload extraction is no longer the main wall.
- [x] Kill serial deferred payload flushing as the current P0. It was a good
  diagnostic, but after direct-root extraction the deferred flush is only about
  `0.017-0.021s`; optimizing it cannot produce the next material win.
- [x] Kill Python-thread payload overlap as a current recommendation. It hid
  wait time but inflated env/render time through contention, so it stays a
  diagnostic only.
- [x] Add replay-parity coverage for the deferred-payload concept:
  `test_deferred_search_payload_rows_match_immediate_rows_for_non_prefix_roots`
  proves delayed payload attachment can match immediate compact replay rows for
  non-prefix root ids.
- [x] Re-run longer loop48 replay-on/replay-off rows after the direct-root fix
  to stabilize the denominator and avoid overreading loop24 variance.
- [x] Re-run loop96 sim16 replay-on/off as a stability check after the loop48
  sim16 mismatch. Result: replay-on `50,617`, replay-off `53,579` roots/sec.
  This confirms replay row construction is not the major wall.
- [x] Run a fresh current-code H100 refresh matrix on the replay-valid compact
  closed-loop denominator. Latest loop96 rows: refresh-on sim16 `53.9k`,
  refresh-on sim32 `36.4k`, refresh-off sim16 `91.8k`, refresh-off sim32
  `57.9k` roots/sec. Refresh deletion is still a real `1.6-1.7x` ceiling, but
  not the 5-10x path. At sim32, search is already `28.6%` of refresh-on and
  `49.2%` of refresh-off wall.
- [ ] Next P0 architecture target: env/observation/search-input handoff. Split
  and attack compact render-state work, delta pack, H2D/update, resident
  stack/root ownership, and public packaging. At sim32+, keep search scaling in
  the same picture because search is already a large bucket.
- [x] Fold Bernoulli's full dataflow/sync note into the main world model. Key
  read: selected actions are tiny and mandatory; latest frames, resident stack,
  root observation copies, visual trail copies, and repeated compact/delta/H2D
  handoff are the objects that matter.
- [x] Fold Locke's validation audit into the main world model. Key read:
  current row-level compact replay tests are strong, but future
  deferred/overlap/action-only speed rows need an end-to-end multi-record
  flush/materialization parity canary before they become replay-valid claims.
- [x] Fold Goodall's external dataflow research into the main world model. Key
  read: Puffer/MCTX/MiniZero/OpenSpiel/EfficientZero patterns all point toward
  contiguous ownership, batched search/model work, and scalar objects only at
  validation/logging edges.
- [ ] Add a small pure profile-shape guard test if the runtime flag validation
  stays buried inside the full Modal benchmark path: action-only/deferred/overlap
  must remain mutually exclusive and replay-index-off until replay parity lands.
- [ ] Add the P0 multi-record replay-valid canary before promoting any
  deferred/overlap path: selected actions, visit policies, root values,
  sidecars, final observations, materialized rows, and non-prefix root ids must
  match the immediate replay-valid path.
- [ ] Choose the next implementation canary for compact state/search-input
  ownership. Best current candidate: actor/env emits compact render deltas or
  a compact state owner updates in place so `_persistent_compact_state_from_production`
  and `_persistent_delta_state` stop rebuilding/scanning as much every step.
- [x] Add async H2D behavior to the existing persistent-renderer async
  device-only profile flag. Result: loop96 sim16 `50.4k -> 53.2k` roots/sec
  and sim32 `40.0k -> 41.9k`. This is useful but small; keep it opt-in.
- [x] Add `compact_batch_build_sec` and `batched_stack_probe_wall_sec`
  telemetry. Fresh loop48 row showed compact batch build is only `0.0016s`
  over 48 steps, so this is not the current Amdahl wall.
- [ ] Next P0 implementation candidate: compact render delta ownership. The
  current loop48/loop96 leaves keep paying production-to-compact, delta pack,
  H2D, and public packaging every step. The canary should bypass or shrink
  `_persistent_compact_state_from_production` and `_persistent_delta_state` for
  the current native visual-trail state while keeping the same rendered/search
  observation contract.
- [x] Run a current-code refresh-on versus refresh-off ceiling retest on the
  current borrowed/resident/no-copy denominator. Result: refresh-on sim16
  `62.7k`, refresh-off sim16 `98.5k`; refresh-on sim32 `49.1k`, refresh-off
  sim32 `74.9k`. Observation refresh deletion is therefore a real but bounded
  `~1.5-1.6x` ceiling on this profile shape, not a 5-10x path by itself.
- [x] Close the latest dataflow/sync/design subagents and record their docs:
  `subagent_full_dataflow_map_20260522.md`,
  `subagent_gpu_sync_model_20260522.md`,
  `subagent_architecture_design_critiques_20260522.md`, and
  `subagent_direct_visual_delta_canary_critique_20260522.md`.
- [ ] Next P0 after the ceiling retest: run a fresh compact search-service
  ceiling/control wave on current code, using existing profile-only sidecar
  modes where possible: mock search service, service-tax probe, and current
  real-search denominator. The goal is to price whether compact-buffer/search
  ownership has enough headroom beyond the `~1.5x` refresh-only ceiling.
- [x] Run that compact service sidecar wave on the current B512/A16 H100 shape.
  Result: `mock_search_service` `17,711.9` steps/sec, `service_tax_probe`
  `12,461.6`, and clean `direct_ctree_gpu_latent` `7,155.7`. The fake-search
  ceiling is `2.48x` over direct; the real-model/no-CTree service-tax row is
  `1.74x` over direct. This keeps compact search-service ownership alive as
  the next major optimizer lane, but it is not a trainer-facing claim and not
  a standalone 10x proof.
- [ ] P0 validation before any trainer-facing compact service claim: add a
  closed compact-loop parity test proving that selected actions, visit policies,
  root values, replay rows, RND inputs/rewards, terminal final observations,
  and player views stay attached to the same record as the trusted path.
- [ ] P0 design next: define `CompactSearchServiceV1` as the single boundary,
  first wrapping the current direct CTree backend, then adding one fixed-shape
  real search backend behind that same API. Keep stock LightZero scalar objects
  as validation/debug edges, not the hot-path owner.
- [x] Write the main-thread synthesis for the next lane:
  `next_compact_service_synthesis_20260522.md`. Current recommendation:
  compact service boundary plus closed-loop replay parity first; renderer-only
  work is not the next main patch.
- [x] Beauvoir sidecar: implement the first closed compact-loop replay parity
  test in `tests/test_compact_search_replay_contract.py`. Local rerun passed:
  `1 passed, 7 deselected`.
- [x] Huygens sidecar: map the clean source location and minimal API for
  `CompactSearchServiceV1`. Recommendation adopted: new
  `src/curvyzero/training/compact_search_service.py`.
- [x] Add the minimal `CompactSearchServiceV1` protocol and fake-service
  replay-index materialization test. Validation passed: ruff, py_compile, and
  focused pytest `2 passed, 7 deselected`.
- [x] Add the first profile-owned direct CTree adapter shape:
  `_LightZeroCollectForwardCompactSearchService`. It implements the
  `CompactSearchServiceV1` boundary over the existing direct CTree compact-array
  probe and has a unit test preserving non-prefix roots and non-identity
  `policy_env_id` values. Validation passed: boundary adapter pytest
  `1 passed, 105 deselected`.
- [x] Add the matching array-ceiling adapter shape:
  `_LightZeroArrayCeilingCompactSearchService`. It lets mock/service-tax style
  compact probes speak the same `CompactSearchServiceV1` boundary. Validation
  passed: adapter pytest `2 passed, 105 deselected`.
- [x] Run full local contract files after the compact service adapter:
  `tests/test_compact_search_replay_contract.py` -> `9 passed`;
  `tests/test_source_state_batched_observation_boundary_profile.py` ->
  `107 passed`.
- [x] Add `compact_search_result_v1_from_arrays` so arrays from an already-run
  direct/mock/service-tax probe can validate into `CompactSearchResultV1`
  without double-running search.
- [x] Refactor the compact replay proof to use the same arrays-to-result helper.
  Validation passed:
  `tests/test_compact_search_replay_contract.py` -> `10 passed`;
  `tests/test_source_state_batched_observation_boundary_profile.py` ->
  `108 passed`;
  `tests/test_source_state_hybrid_observation_profile.py` -> `35 passed`.
- [x] Add explicit action-feedback verification telemetry to compact service
  replay proof: expected search-selected joint-action checksum must equal the
  applied next-step joint-action checksum. Focused proof tests and full hybrid
  profile tests passed.
- [x] Add `run_compact_batch` to the mock/service-tax array-ceiling probe. It
  now validates produced arrays into `CompactSearchResultV1` and emits common
  `compact_service_*` telemetry without a second probe call. Validation passed:
  `tests/test_source_state_batched_observation_boundary_profile.py` ->
  `108 passed`.
- [x] Close and fold the 2026-05-23 full-dataflow critique wave into the main
  world model:
  `subagent_full_iteration_dataflow_critique_20260523.md`,
  `subagent_sync_budget_and_experiments_20260523.md`,
  `subagent_architecture_designs_2x_5x_10x_20260523.md`, and
  `subagent_optimizer_validation_gate_audit_20260523.md`.
- [x] Next P0 experiment: rerun the direct/mock/service-tax H100 profile rows
  on the current code and compare the common compact service telemetry fields.
  Durable 2026-05-23 row set:
  `direct_ctree_gpu_latent` `5,965.1` steps/sec,
  `service_tax_probe` `11,854.6`, and `mock_search_service` `14,970.0`.
  Ratios: service-tax/direct `1.99x`, mock/direct `2.51x`. Results live under
  `artifacts/local/curvytron_hybrid_observation_profile_results/rerun_current_compact_*_20260523`.
- [x] First P0 validation extension: compact replay proof now verifies identity
  sidecars and RND latest-frame extraction at the same compact roots as the
  search result. It emits
  `compact_service_replay_identity_feedback_verified`,
  `compact_service_replay_rnd_latest_verified`, and checksums for compact root
  rows, player, policy env id, and RND latest frames. Local validation:
  `tests/test_source_state_hybrid_observation_profile.py -k compact_service_replay_proof`
  -> `4 passed`; `tests/test_compact_search_replay_contract.py` plus hybrid
  profile full files -> `46 passed`.
- [x] Extend the compact proof through actual RND reward-model
  collect/train/estimate behavior and terminal final-observation cases in one
  combined trainer-facing canary. The new proof feeds materialized compact
  replay observations into `CurvyRNDRewardModel`, verifies predictor weights
  move while the target hash stays fixed, checks that positive intrinsic reward
  changes the target reward, and confirms terminal rows use final observation
  instead of the latest live observation. Validation passed with focused ruff
  and the compact/hybrid/RND test set: `180 passed`.
- [x] Add the first adversarial compact-result identity gate for future
  service backends. Swapped-player results and duplicate/missing root ids now
  fail before replay writing. Validation: compact replay contract file
  `12 passed`.
- [x] Prove compact index rows compose into the repo learner-facing sample
  batch. `test_compact_index_rows_materialized_sample_batch_matches_immediate_rows`
  now compares `CompactReplayIndexRowsV1 -> materialized rows -> sample batch`
  against trusted immediate rows under the same sample seed. Focused validation:
  `2 passed`; ruff clean.
- [x] Prove the same compact record identity through the public stock LightZero
  sampler edge. The opt-in local gate pushes compact-materialized rows into
  real `MuZeroGameBuffer`, calls public `buffer.sample(...)`, maps sampled
  transition ids back to compact row ids, and compares sampled observations,
  policy targets, reward targets, and zero-model value targets. Validation:
  `tests/test_compact_search_replay_contract.py -k public_sample` -> `1 passed`;
  full compact contract file -> `15 passed`.
- [x] Add the first stock LightZero native target-hook parity gate. Compact
  index rows now materialize into source-state target rows, build real
  LightZero `GameSegment`s when `lzero` is installed, push them into
  `MuZeroGameBuffer`, and compare stock reward/value/policy target hooks
  against the materialized compact rows. This is stronger than the repo-local
  sample batch test; the public sampler edge is covered by the following opt-in
  canary.
- [ ] Next P0 implementation: prototype one fixed-shape backend behind
  `CompactSearchServiceV1`. Current recommendation is the fixed-`A=3`
  compatibility backend first if we need the lowest-risk denominator cleanup,
  then a fixed-shape Torch device-tree backend as the first serious speed
  attempt. See `subagent_fixed_shape_search_designs_20260523.md`.
- [x] Wire existing `dense_torch_mcts` profile probe through the same
  `CompactSearchServiceV1` result path. The array-ceiling probe now stores
  compact search arrays for dense Torch modes, `run_compact_batch()` validates
  them into `CompactSearchResultV1`, and the grid builder allows dense Torch
  compact replay rows. This is profile-only and not trainer-facing.
- [x] Add profile-only persistent compact render-state buffer as a diagnostic.
  It makes renderer `production_to_compact_sec` report `0.0` and passes local
  focused tests, but the H100 A/B is mixed because parent-side compact trail
  writes are expensive: B1024 sim16 improved `26.3k -> 35.8k` roots/sec while
  B1024 sim32 regressed `29.5k -> 25.3k`. Keep it opt-in/profile-only.
- [x] Fold Ramanujan's compact-buffer critique into the active world model.
  The parent compact buffer copies full accumulated trail state from actor/env
  state every actor step, so it moves work out of renderer conversion and into
  actor render-state writes. Next real canary should use env-emitted render
  deltas or actor-owned compact state, not another parent-side full copy.
- [ ] Next P0 render-state architecture question: avoid both conversion and
  parent-side full trail copying. Investigate actor-owned compact x/y state,
  renderer-side direct `visual_trail_pos` consumption, or env/runtime trail
  event deltas. Do not recommend the parent compact-buffer path as the main
  speed setting unless a newer same-denominator profile proves otherwise.
- [ ] Keep direct visual-delta as a small P1 canary unless the service ceiling
  says observation handoff remains the largest unresolved wall. Maxwell's
  critique expects only `1.05x-1.12x` typical upside and hard ceiling near
  `1.2x`.
- [x] Add `closed_loop_deferred_payload_profile` as a profile-only canary. It
  reads only selected actions during the env/search loop, then flushes
  visit-policy/root-value payloads after the loop and reports that flush
  separately.
- [x] Run the matched H100 payload split matrix. Result: action-only is much
  faster (`68.8k` sim16, `53.3k` sim32), but deferred serial flush gives the
  time back (`42.9k` sim16, `39.3k` sim32). Conclusion: simple deferral is not
  the win; the next target is overlapped or resident payload ownership.
- [x] Add a local replay-parity test for the deferred-payload concept before
  any trainer-facing path exists. It must compare actions, policy target, root
  value, reward, done, final reward, env rows, and players against the current
  immediate compact replay builder.
- [x] Run the overlap search payload canary. Result: not recommended. The wait
  moved out of the explicit payload bucket, but total wall got worse through
  JAX/env/render contention.
- [x] Retest the current borrowed/resident compact visual denominator after the
  newest sync patches. Latest clean read: explicit resident sync on gives
  sim16 `45.4k` roots/sec and sim32 `35.7k`; explicit resident sync off gives
  sim16 `50.4k` and sim32 `43.3k`.
- [x] Run async internal renderer device-only sync as a falsifier. Result:
  sim16 `50.2k` roots/sec and sim32 `40.9k`, which does not beat the simpler
  resident-sync-off row. Keep this as a profile-only canary, not a promoted
  speed mode.
- [x] Update the active recommendation: best current profile-only row is
  `borrow_single_actor_render_state=True`, resident GPU stack, no root
  observation copy, replay-index on, and explicit resident-stack sync off. Do
  not turn on the async renderer flag for the next profile batch unless the
  goal is attribution.
- [x] Add guarded vectorized `_persistent_delta_state` fast path, then A/B it
  under a profile flag. Local boundary tests passed (`105 passed`). Current
  A/B says sim16 exact -> vectorized `51.9k -> 53.1k`, but sim32 exact ->
  vectorized `45.1k -> 37.9k`. Keep vectorized opt-in only; exact delta pack
  remains the default/recommended path.
- [ ] Split the remaining borrowed-row wall: `observation_sec`,
  `renderer_render_sec`, delta-pack, H2D, resident stack update, search, and
  unlabeled residual need a cleaner exclusive profile before the next bigger
  rewrite.
- [ ] Prototype the next resident compact state owner. Borrowed actor state
  proved the ownership hypothesis, but the renderer still rebuilds compact
  state, delta-packs it, and transfers compose/delta data every step.
- [ ] P0 next architecture canary: bypass `_persistent_compact_state_from_production`
  and `_persistent_delta_state` in the hot profile path by having the
  actor/env produce compact render deltas or by maintaining a compact state
  owner beside the actor. The cheap falsifier is total roots/sec on the same
  H100 B1024/P2 loop24 borrowed/resident/sync-off denominator, not a local
  timer-only win.
- [ ] P0 measurement before/with that canary: make the remaining step buckets
  exclusive enough to separate CPU compact-state build, CPU delta pack, H2D,
  persistent update/draw, resident stack shift, root sidecar build, search, and
  residual glue.
- [x] Reorient after the fast-visual/no-copy rows. Current read: actual game
  mechanics are not the obvious wall; `env_step_sec` mostly means render-state
  write, observation/stack update, and handoff into the next search call.
- [x] Write the next patch note:
  [subagent_next_state_ownership_patch_20260522.md](subagent_next_state_ownership_patch_20260522.md).
  Recommendation: a profile-only compact render-state owner, with no trainer or
  MuZero semantic changes.
- [x] Add the reproducibility grid for the latest mechanics vs observation vs
  search split:
  [subagent_next_profile_grid_20260522.md](subagent_next_profile_grid_20260522.md).
  It uses the current H100 B1024/P2 actor_count=1 fast visual/no-copy
  denominator and keeps the rows profile-only.
- [x] Make resident-stack refresh-off controls runnable in the profile-only
  benchmark by using a zero resident stack when observation refresh is
  intentionally disabled.
- [x] Run the matched resident-stack/no-copy/fast-visual grid again after the
  newest rendering/environment changes: H100 B1024/P2 sim16/sim32, host stack
  versus resident stack, closed-loop replay-index on, observation refresh on.
  Use this to decide whether resident stack is now a real win or only valid
  plumbing.
- [x] Add closed-loop env-step timing split through the compact hybrid actor
  path. The first smoke proved the timers are visible.
- [x] Correct the Amdahl read: in the timing-split smoke, actual env runtime was
  tiny, while actor render-state write and observation/renderer/stack work
  dominated the env_step_sec bucket.
- [x] Patch the persistent GPU profile path to copy only renderer-required state
  keys into parent native render buffers. CPU oracle and generic renderer paths
  still receive full state.
- [x] Validate focused local tests after that patch:
  `ruff` passed, `native_actor_buffer or timing` passed, and compact/MCTX
  focused tests passed.
- [x] Collect the matched H100 row after the render-state filter:
  B1024/P2/sim16/loop16/native, body_capacity=4096, hidden_dim=64,
  max_depth=16, rollout_steps=4. Run `ap-FVVPGkA3oKem8yPvnetWdL`: `15,906`
  closed-loop roots/sec, env `73.4%`, search `5.3%`.
- [x] Decide the render-state filter outcome: it is useful instrumentation
  cleanup, but not a major speed win. Actor render-state write is visible
  (`0.368s`), yet observation/stack (`0.988s`) and production->compact
  (`0.517s`) still dominate.
- [ ] Attack resident compact observation next: avoid mandatory host frame
  readback and host stack update in the profile hot loop, then feed MCTX from
  the resident device stack.
- [x] Add resident compact observation validation guards and rerun the matched
  H100 grid. Result: host stack + replay rows was fastest in this row
  (`20.7k` active roots/sec). Resident GPU stack is valid profile plumbing but
  not a current speed recommendation.
- [x] Kill replay-index as a primary target. Replay-index was about `0.3%` of
  wall in the fastest matched row and effectively zero when disabled.
- [ ] Run the observation-refresh-off ceiling. This is a profile-only falsifier
  that keeps compact env/search/replay shape but skips render-state write and
  observation refresh so we can price the maximum possible win from deleting
  the current env/observation handoff wall.
- [x] Record the observation-refresh-off ceiling:
  sim16 `20.7k -> 48.6k` active roots/sec (`2.35x`), sim32
  `17.9k -> 32.1k` (`1.80x`). This proves the wall is real but not 10x by
  itself.
- [x] Add profile-only no-copy root observation option for
  `CompactRootBatchV1`. Default behavior still copies. The hot profile path can
  now request an observation view and focused tests prove the contract.
- [ ] Run no-copy root-batch rows with observation refresh on and off. If this
  moves the ceiling materially, keep pushing root-batch/sidecar ownership. If
  not, shift to persistent compact render-state ownership.
- [x] Implement and profile the visual-trail fast compact-state adapter for the
  persistent GPU renderer. It cut production-to-compact from roughly
  `0.37-0.52s` to about `0.054-0.057s` on matched sim16 rows.
- [x] Confirm combined refresh-on win: fast visual state + no-copy root batch
  reached `26.6k` active roots/sec versus prior `20.7k`, about `1.29x`.
- [x] Retest resident GPU stack after root-copy is gone. Result: resident now
  wins this profile denominator. Sim16 host no-copy `26.6k` -> resident
  `31.6k`; sim32 host no-copy `21.2k` -> resident `28.9k`.
- [x] Repeat the resident-stack grid with the latest renderer/environment and
  grouped telemetry. Loop24 H100 rows: sim16 host `23.1k`, resident `30.3k`;
  sim32 host `19.5k`, resident `26.8k`; refresh-off ceiling `57.9k`.
- [x] Confirm the latest policy surface was not dropped: current trusted
  policy observation is still `browser_lines + simple_symbols`; GPU persistent
  framebuffer rows are profile-only evidence for this surface, not the old
  `body_circles_fast` lane.
- [x] Write the compact one-iteration dataflow map:
  [subagent_full_iteration_dataflow_20260522.md](subagent_full_iteration_dataflow_20260522.md).
  It records stock/trusted versus profile-only compact dataflow, measured
  timing buckets, CPU/GPU residency, copy/materialization points, ten design
  alternatives, and the next three profile experiments.
- [x] Add grouped compact stdout. Fresh refresh-on rows show actual game
  mechanics are only about `8-11%` of `env_step_sec`; observation/search-input
  handoff is about `76-80%`; raw GPU draw is about `5-7ms`.
- [x] Audit `env_step_sec` into mechanics versus observation/search-input
  handoff. Result: actual `actor_env_runtime_sec` is small; the current wall is
  render-state write plus observation/stack ownership. See
  `subagent_mechanics_vs_observation_audit_20260522.md`.
- [x] Split `actor_render_state_write_sec` by array family. Result: in the
  fresh B1024/P2 sim16 rows, almost all of the render-state write is
  `visual_trail_*` copy (`0.244s/0.245s` resident, `0.321s/0.323s` host).
- [x] Run the profile-only borrowed single-actor render-state falsifier. Result:
  host sim16 copied `21.3k` -> borrowed `35.7k` roots/sec (`1.67x`);
  resident sim16 copied `32.3k` -> borrowed `44.8k` (`1.39x`);
  resident sim32 copied `34.6k` -> borrowed `36.1k` (`1.04x`).
- [x] Record the post-borrow Amdahl pivot. Result: the parent visual-trail copy
  was real, but after it is removed the sim32 row is mostly search/control and
  residual handoff, not that copy.
- [ ] Next P0: split the remaining borrowed/resident refresh-on path into
  exclusive buckets: renderer production-to-compact, delta pack, H2D,
  persistent update, raw draw, resident stack update, public packaging, search
  control, and unlabeled residual. Current renderer leaf timers are still
  partly nested.
- [ ] Next P0 falsifier: add a profile-only borrowed+resident row with deferred
  or consolidated synchronization around renderer/resident stack update, then
  compare total roots/sec. Kill it if wait time simply moves into search.
- [ ] Split resident-mode `observation_sec` into exclusive renderer call,
  delta pack, H2D, device update/render, final/autoreset handling, and resident
  stack update. Current leaf timers are useful but still partially nested.
- [ ] Keep any new speed claim labeled as profile-only until the stock/training
  denominator is explicitly measured.

## 2026-05-22 Fast Falsifier Reset

- [x] Patch the batched vector ray observer capacity-scan bug. It now slices
  trail arrays to `max(body_write_cursor)` before ray casting. Local B512
  vector actor-loop timing improved from about `456` ego decisions/sec to
  about `20k` ego decisions/sec on the same short diagnostic shape. This is a
  vector-sample fix, not a stock visual training speed claim.
- [ ] Re-run the real-vector MCTX sample after the ray-trim patch on Modal if
  the vector lane becomes relevant again. Expected change: host observation
  setup should collapse from tens of seconds to sub-second or low-single-digit
  seconds for short cursors.
- [x] Add and run the real compact visual-root MCTX gate:
  `curvytron_hybrid_compact_visual_sample`. It now validates
  `HybridCompactBatch -> CompactRootBatchV1 -> MCTX -> CompactSearchResultV1`
  on real `[B,2,4,64,64]` observations.
- [x] Run CPU-oracle versus persistent-GPU compact visual rows. Result:
  persistent GPU makes the B256 compact visual gate about `2.1x` faster on the
  fresh-boundary roots/sec denominator and cuts last-step render from about
  `3.91s` to `0.014s`.
- [ ] Next architecture gate: selected MCTX actions must drive the next
  compact env step and produce `CompactReplayIndexRowsV1` without scalar
  LightZero timestep materialization. Do this as profile-only first.
- [x] Record the failure pattern explicitly:
  [reorientation_20260522_fast_falsifiers.md](reorientation_20260522_fast_falsifiers.md).
  Flat-A3 is valid but did not move the matched full-loop denominator; do not
  keep polishing it as the main lane.
- [x] Run local compact topology controls:
  native-vector zero-observation control and closed-compact-consumer arrays
  row, both with native actor buffer and no scalar timestep materialization.
- [x] Run the sidecar minimal wave only if the local controls stay sane:
  mock search-service ceiling, precomputed recurrent falsifier, and compact
  replay proof row.
- [ ] Promote no speed claim unless the row identifies its currency:
  actual training, stock full-loop profile, or profile-only boundary probe.
- [x] Verify closed compact search/replay ownership before any new CTree
  micro-optimization. Result: the existing `hybrid_compact_service_replay_proof`
  path already makes compact search-selected actions drive the next env step,
  writes `CompactReplayIndexRowsV1`, and reports public LightZero output bytes
  as zero.
- [ ] Next implementation target: search-service boundary itself. Compact
  replay/index rows are cheap enough; do not add more replay machinery until
  root prep, per-simulation CPU/list CTree/control, recurrent-output handling,
  and actor/env/observation scheduling are split again.
- [x] Run sim-scaling and sequential sim16 repeat after the compact proof.
  Corrected result: precomputed recurrent is about `1.25x` over direct at
  B512/A16/sim16, so recurrent/model output matters but is not the sole wall.

## 2026-05-21 Current Focus

- [x] Add whole-loop denominator ledger so optimizer claims separate actual
  Coach training speed, stock full-loop profile speed, and profile-only
  boundary probe speed.
- [x] Add current code dataflow map so production Coach training, stock
  full-loop profiles, profile-only hybrid probes, public collect-forward, and
  direct CTree arrays are not conflated.
- [x] Add profile-only persistent GPU policy-space framebuffer backend
  `jax_gpu_persistent_policy_framebuffer_profile`.
- [x] Validate focused local tests for backend fail-closed config and
  incremental same-owner trail connection.
- [x] Collect first H100 surface rows. B512/100 improved from about `81ms` to
  about `45ms`; renderer bucket improved from about `39ms` to about `10ms`.
- [x] Collect long B512/500 row. Render is now about `16ms`; env step and stack
  update are each about `39ms`.
- [x] Patch env collision broad-phase to scan only live
  `body_write_cursor` prefixes instead of full allocated `body_capacity`.
- [x] Collect the post-patch B512/500 H100 persistent row. Env step dropped
  from about `38.95ms` to about `13.75ms`; total surface step dropped from
  about `78.94ms` to about `65.70ms`.
- [x] Fix collision hit-owner attribution to respect `body_write_cursor` too;
  stale active slots beyond cursor can no longer steal ownership after a prefix
  hit.
- [x] Decide immediate stack lane from evidence: do not do a tactical host-only
  ring buffer yet, because it likely moves the copy into policy materialization.
- [x] Park the stack/device handoff lane unless fresh rows show H2D or stack
  movement is again a meaningful wall. Current evidence says the bigger wall is
  the LightZero/search boundary, not a host-only ring buffer.
- [x] Add `device_stack_handoff_plan_20260521.md`: first probe should use the
  hybrid profile canary's pre-scalar `HybridBatchedStackProbe`, not the trainer
  surface public contract.
- [x] Run first H100 hybrid persistent+uint8+pre-scalar probe with scalar
  materialization disabled; compare against materialized/float32 rows.
- [x] Run first H100 hybrid persistent+uint8+pre-scalar probe grid. Results:
  `uint8` without scalar materialization `~16.3k` scalar steps/sec; `uint8`
  with scalar materialization `~9.8k`; `float32` without scalar materialization
  `~9.2k`.
- [x] Define the next real-consumer probe: a batched uint8 stack that is
  normalized and consumed on GPU by a search/model-shaped probe, with scalar
  LightZero payload only measured as an optional edge cost.
- [x] Add explicit device-latest profile flag and run first H100 row. It cut
  probe H2D (`~0.178s` -> `~0.042s`) but reduced throughput (`~16.3k` ->
  `~11.6k`) because host stack update still happened and a second device stack
  was added.
- [x] Park the device-latest host-stack bypass unless the next denominator
  ledger says stack movement is back on the hot path. The first device-latest
  probe kept two stacks and regressed, so it is not the active lane.
- [ ] Run the cleaner RND cadence pair as a separate profile axis when needed:
  no-RND vs `rnd_meter_v0`, then cadence `10` and `100`. Do not mix RND rows
  into renderer or direct-CTree speed claims.
- [x] Add profile-only surface divergence canary. It compares candidate
  renderer-backed stacks against `CpuOracleBatchedObservationRenderer` over
  rollout steps, reports aggregate mismatch fraction, and stays off by default.
- [x] Run first CPU-vs-CPU divergence control. It passed exact over reset plus
  four steps, proving the canary plumbing is aligned.
- [x] Run first persistent GPU `direct_gray64` divergence smoke. It passed a
  loose tolerance over 32 no-death steps; it was not exact (`max_abs_diff=61`,
  about `0.49%` value mismatches).
- [x] Finish long no-death and timeout/autoreset divergence rows. L4/T4
  timeout/autoreset passed with terminal rows and `0.17%` mismatch fraction;
  L4/T4 256-step no-death passed with `2.71%` mismatch fraction and no render
  truncation.
- [ ] Background fidelity only: add a visual/sample artifact or
  connected-component diff summary if observation parity becomes a launch
  blocker again.
- [ ] Run matched full-loop A/B/C only after the active boundary is
  semantically credible: CPU oracle vs batched profile manager vs zero
  observation, with RND measured separately.
- [x] Run fresh current L4/C256/batch64/sim8 no-death profile anchor grid
  against the latest code:
  `opt-current-l4-c256-ab-rnd-20260521a`. Result: batched GPU no-RND reached
  about `942.63` steps/s versus subprocess CPU-oracle `641.10` steps/s
  (`~1.47x`). RND plus batched GPU regressed to `618.82` steps/s with GPU max
  `97%`; zero-observation rows did not form a clean ceiling (`769.33` no-RND),
  so repeat before claiming a stable Amdahl bound.
- [x] Run fresh search/hardware sweep:
  `opt-current-search-hw-sweep-20260521a` plus H100 retry
  `opt-current-search-h100-sweep-20260521b`. Result: H100 helps at sim4, but
  sim8/sim16 are dominated by collect/search/manager pressure. Zero-observation
  rows are faster but still not free, so isolated render work is no longer the
  main 5-10x lane.
- [x] Add compute alias protection to `scripts/build_curvytron_profile_grid.py`;
  `gpu-h100` now canonicalizes to `gpu-h100-cpu40`.
- [x] Add profile-only split timers for scalar bridge and surface packaging.
  Focused tests pass; next tiny Modal profile smoke should confirm compact
  output fields.
- [x] Run split-timer C256/C512 real-vs-zero rows:
  `opt-split-timer-hw-c256-c512-20260521a`. C512/sim8 zero-vs-real headroom
  is about `1.17x` on both L4 and H100. Bridge/surface package sub-buckets are
  not the main wall; policy collect/MCTS plus manager/render/stack dominate.
- [x] Draft the resident chunk canary plan: compact state batch -> GPU
  observation -> batched policy/search pressure -> scalar materialization only
  at the edge.
- [x] Use worker feedback to decide whether to implement the resident chunk
  canary as a tiny new script/module or reuse the existing hybrid observation
  profile code. Decision: reuse the existing hybrid observation profile code;
  the canary already exists.
- [x] Run first sim8 resident canary repeat:
  H100 scalar-off `~13.8k` roots/sec, H100 scalar-on `~6.5k`, L4/T4 scalar-off
  `~9.0k`, L4/T4 scalar-on `~4.2k`, H100 device-latest scalar-off `~9.8k`.
- [ ] Design the next compact arrays/replay prototype only after direct CTree
  parity and statistical comparison pass. Goal: keep the resident uint8 batch
  alive through real policy/search/replay-shaped work without public-wrapper
  fanout.
- [x] Record the 2026-05-22 promotion-gate update for direct CTree. Sim8 local
  compare passed strict exact checks for both direct arrays and GPU-latent;
  sim16 neutral/tie-heavy strict action/visit equality failed while values,
  logits, illegal-action checks, and GPU-latent activation stayed clean. Plain
  read: exact neutral parity is still the wrong gate; forced/schema/statistical
  gates are the right path.
- [x] Record RND/death guardrails for any future resident-batch recommendation:
  matched no-RND, matched `rnd_meter_v0`, normal-death/autoreset, long
  no-death, and real policy/search-pressure rows before Coach launch advice.
- [x] Audit where the batch dies today. Result: stock/env-manager boundaries
  still scalarize to env-id keyed `BaseEnvTimestep` objects before LightZero
  policy/search gets them, even though policy/search can batch roots after that.
- [x] Implement a profile-only resident replay/search probe in the hybrid Modal
  path. It keeps trainer defaults untouched and can now measure scalar
  materialization as an optional edge cost.
- [x] Run resident probe smoke and medium H100/L4 scalar-off rows. Medium
  B512/A16/sim8 results: H100 `~10.98k` scalar roots/sec, L4/T4 `~5.84k`.
- [x] Run resident probe scalar-on rows on the same medium shape to price the
  LightZero scalar-timestep edge on top of replay/search-shaped pressure.
- [x] Record scalar-edge read: H100 drops to `~7.62k` roots/sec and L4/T4 drops
  to `~4.13k`; scalarization costs `~1.96s` / `~3.68s` over the measured
  B512/A16/sim8 window.
- [x] Build the next "real consumer" canary. Done via actual
  `MuZeroPolicy.collect_mode.forward`, direct CTree arrays, and array-ceiling
  probes from the pre-scalar `[B,2,4,64,64]` stack.
- [x] Add byte/count instrumentation around materialization boundaries so
  scalar edge rows report payload size, object fanout, and final-observation
  bytes instead of just wall time.
- [x] Add byte/count instrumentation around mock collector scalar
  materialization boundaries: flat obs bytes, action-mask bytes, reward/done
  bytes, final-observation bytes, info count, and materialized timestep count.
- [x] Run H100 B1024 resident scale check. Result: scalar-off stays flat
  around `~11.07k` roots/sec and scalar-on drops to `~6.75k`; B512 remains the
  better default resident-probe shape for now.
- [x] Decide whether a tiny JAX/MCTX toy spike is worth building now. Park it:
  direct CTree arrays is the cleaner near-term lane because it still uses real
  LightZero model and real CTree MCTS.
- [x] Implement the next profile-only real-consumer canary:
  `MuZeroPolicy.collect_mode.forward` from the pre-scalar `[B,2,4,64,64]`
  stack, explicitly labeled as `lightzero_collect_forward_search_cpu_tree`.
- [x] Implement the profile-only real-consumer canary:
  `--hybrid-lightzero-collect-forward-probe`. It builds a scratch CurvyTron
  MuZero policy, flattens pre-scalar `[B,2,4,64,64]` into `[B*2,4,64,64]`,
  passes real action masks/player ids to `policy.collect_mode.forward`, decodes
  every root, records illegal-action checks, and keeps scalar timestep
  materialization optional.
- [x] Run first remote LightZero collect-forward smoke:
  L4/T4, B4/A2/sim1, two measured steps. Result: `ok=true`,
  `materialized_timestep_count=0`, `calls_train_muzero=false`,
  `lightzero_root_count=8`, `lightzero_policy_device=cuda:0`,
  `lightzero_illegal_action_count=0`, consumer semantics
  `lightzero_collect_forward_search_cpu_tree`.
- [x] Run real-consumer medium rows: H100/L4, B512/A16/sim8, scalar edge off/on,
  then compare against synthetic resident rows. Result: H100 scalar-off
  `2669.32` roots/sec, H100 scalar-on `2100.31`, L4 scalar-off `2159.35`,
  L4 scalar-on `2053.57`. Real collect-forward preserves the pre-scalar batch
  but is only `24-50%` of the synthetic resident probe throughput.
- [x] Apply real-consumer critique fixes: fail closed for `uint8` +
  `direct_gray64` + persistent GPU profile backend, use fixed-opponent
  `to_play=-1`, filter all-zero action-mask roots, and label collect-forward
  timing as CPU-tree-inclusive.
- [x] Add the next split canary: real MuZero model initial inference only over
  the same pre-scalar batch, separate from LightZero collect-forward/CPU tree
  search. Local fake-policy tests passed, public Modal entrypoint routing was
  added, and tiny remote smoke passed.
- [x] Run corrected split rows with fail-closed `uint8/direct_gray64/persistent`
  contract, fixed-opponent `to_play=-1`, and zero-mask filtering. Result:
  H100 initial inference `9238.85` roots/sec versus H100 collect-forward
  sim8 `2693.10`; L4 initial inference `6790.63` versus L4 collect-forward
  sim8 `1381.35`.
- [x] Record the new Amdahl read: render is not the main wall in corrected
  real-consumer rows; public LightZero collect/search is the next wall.
- [x] Add first direct-CTree parity/debug gate. Stock-facade decoding now maps
  legal-action-only visit vectors back to full action ids, zeroes illegal visit
  mass in full visit vectors, and real CPU LightZero checks compare
  stock-facade versus direct CTree searched values and legal normalized visits
  for sim1/sim2/sim8.
- [x] Add a biased-logit real-policy canary. When action 1 is made the clear
  winner, stock facade and direct CTree both choose action 1 as the top action
  at sim8.
- [x] Add deterministic legal-mask canaries. Single-legal-action rows produce
  exact one-hot visits in stock and direct; masked biased-logit rows choose the
  best legal fallback and keep zero illegal visit mass.
- [x] Tighten stock-facade compact debug output. The facade now decodes
  predicted values and policy logits when LightZero returns them, including
  legal-action-length logits mapped back to full action ids. Focused tests and
  ruff passed.
- [x] Add the binary-mask precondition for LightZero collect/direct probes.
  Stock LightZero treats legal actions as `x == 1`, so profile rows now reject
  fractional action masks instead of treating any positive value as legal.
- [x] Add the compact MCTS visit-policy target-row canary. The checked
  multiplayer target bridge now accepts non-one-hot visit distributions as
  `policy_target`, preserves root value/source metadata, and still enforces
  action legality/alignment. Validation: target-row suite `14 passed`;
  focused ruff passed.
- [x] Diagnose the exact neutral action/visit parity blocker. Fixed
  Python/NumPy/Torch seeds do not make stock LightZero CTree repeat identical
  neutral/tie-heavy visit allocations even stock-vs-stock, so exact neutral
  visit parity is a bad production gate.
- [x] Turn the parity diagnosis into a promotion contract:
  `direct_ctree_promotion_contract_20260521.md`. Current gate: exact
  forced-mask/clear-preference/schema/target-row checks plus
  stochastic/statistical collect-row comparison plus matched full-loop profile.
  Until this contract passes, `direct_ctree_arrays` remains profile-only.
- [x] Decide whether to prototype a deeper search lane after the split:
  profile-only direct CTree arrays is the current lane. Custom collector,
  MCTX/JAX, and topology tuning stay parked until direct parity/statistical
  gates and a matched full-loop profile justify them.
- [x] Pin the boundary test matrix in code/docs. Subagent wave completed:
  coverage map, GPU observation test implementation, LightZero boundary
  semantics critique, and RND/reset/death critique.
- [x] Add opt-in native actor buffer for the profile-only hybrid manager.
  Actors can write reward/done/episode/alive/action-mask/action fields directly
  into parent-owned compact arrays for zero-observation rows.
- [x] Widen the pre-scalar compact consumer contract from obs+mask to
  `HybridCompactBatch`. The new sidecar includes row/player ids, target reward,
  done roots, terminal/autoreset masks, final observation, episode metadata,
  alive flags, and joint action. Legacy two-argument probes still work.
- [x] Validate the widened sidecar locally:
  `uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py`
  passed with `19 passed`.
- [x] Re-run matched local B512/A16 zero-observation rows after the sidecar
  widening. Result: payload+merge `~22.1k` timesteps/sec, native actor buffer
  `~30.5k`, about `1.38x` on this local denominator.
- [x] Build the next sidecar consumer proof for replay/target compatibility:
  active-root ordered `selected_action`, `visit_policy`, and `root_value`
  arrays from `HybridCompactBatch` can become checked `PolicyRowRecordV0`
  records and pass through `build_source_state_multiplayer_target_rows_v0`.
  Validation: `tests/test_multiplayer_source_state_target_rows.py` ->
  `18 passed`.
- [ ] When useful, combine real direct-CTree compact output with the compact
  target-row adapter in one matched profile row. Do not call this native
  LightZero replay or Coach advice until matched full-loop A/B passes.
- [x] Add the local combined edge proof:
  `HybridCompactBatch -> direct CTree compact profile hook -> compact
  action/visit/value arrays -> compact target-row adapter -> checked
  source-state target rows`. Validation:
  `tests/test_source_state_batched_observation_boundary_profile.py`
  plus `tests/test_multiplayer_source_state_target_rows.py` -> `112 passed`.
- [x] Wire the profile-only direct CTree compact search output into
  `CompactRootBatchV1` and `CompactSearchResultV1` validation inside
  `_LightZeroCollectForwardStackProbe.run_compact_batch`. This proves the
  root/result half of the compact service contract without changing trainer
  behavior. Validation: focused ruff passed; focused pytest
  `3 passed, 99 deselected`.
- [x] Add a profile-only helper that carries replay chunk/record index through
  the compact service edge:
  `run_compact_batch_with_replay_chunk(...)`. It validates
  `CompactReplayChunkV1` and target rows from the real direct CTree compact
  search output. Validation: focused ruff passed; focused pytest
  `2 passed, 96 deselected`.
- [ ] Run the next aggressive falsifier: same-denominator profile comparing
  current direct CTree compact output against the closed compact service proof
  with RND latest-frame input and replay target-row materialization. Kill the
  lane if it cannot plausibly produce a `3x` class profile win over current
  direct.
- [x] Before that falsifier, fix the profile denominator: the current hybrid
  profile loop uses random actions to step the env, then profiles search on the
  resulting observation. A valid replay-target loop needs search-selected
  actions to drive the next env step. The new
  `hybrid_compact_service_replay_proof` profile flag does this and then builds
  the two-record `CompactReplayChunkV1` proof.
- [ ] Run the Modal H100 no-RND proof row:
  `direct_ctree_gpu_latent`, uint8 stack, no scalar timestep materialization,
  compact replay proof on, enough warmup/measured steps for stable timing.
  Compare against the current best direct CTree row on the same denominator.
- [x] Add explicit compact sidecar fields for fixed-opponent search semantics:
  `to_play[M]` and `active_root_mask[M]`. Current convention is
  fixed-opponent `to_play=-1`; active roots are legal-mask roots that are not
  done roots.
- [x] Wire the profile-only real LightZero direct-CTree boundary to consume
  `HybridCompactBatch` through
  `_LightZeroCollectForwardStackProbe.run_compact_batch`. It validates
  row-major row/player ids, `to_play`, `target_reward`, and active-root masks,
  then zeroes inactive-root legal masks before calling the existing direct
  CTree arrays path.
- [x] Validate the compact LightZero sidecar hook locally:
  `uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_boundary_profile.py -k "hybrid_profile or compact_batch or direct_ctree_returns_compact_arrays"`
  -> `11 passed, 100 deselected`; full boundary suite
  `tests/test_source_state_batched_observation_boundary_profile.py` ->
  `92 passed`.
- [x] Record the fail-closed Modal smoke result. The first remote smoke used
  the old renderer backend and failed with the intended guard:
  direct MCTS arrays require
  `observation_renderer_backend=jax_gpu_persistent_policy_framebuffer_profile`.
- [x] Finish the corrected Modal compact direct-CTree smoke and record whether
  the real remote path reports `lightzero_compact_batch_contract` plus zero
  illegal decoded actions. Corrected run `ap-RztU5jMKmKBpXuaDY3vZB0` passed
  with `compact_row_player_sidecar_v1`, active-root telemetry, no scalar
  timestep materialization, and zero illegal decoded actions.
- [x] Harden compact sidecar validation after critique: compact action masks
  are now checked as binary before bool coercion, and the direct hook validates
  `done_root`, terminal/autoreset masks, final-observation mask, and sidecar
  shapes before search. Focused malformed-sidecar tests passed.
- [x] Add the next correctness proof for RND: consume `HybridCompactBatch`
  sidecars in an RND-meter-shaped probe and verify reward/done/final/autoreset
  rows survive without scalar timestep materialization. The proof now includes
  `extract_policy_gray64_latest_for_rnd_from_compact_observation`, uint8
  compact-stack normalization tests, and a hybrid profile probe that extracts
  RND latest frames from `HybridCompactBatch` while
  `materialize_scalar_timestep=false`.
- [x] Add a cheap row-ownership/fill-mask guard for the native actor-buffer
  path before using it as evidence for subprocess/shared-slab designs.
- [x] Add two focused local boundary tests:
  initial-inference zero-action-mask filtering and row-selective persistent
  cursor-regression reset. Validation:
  `uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py`
  -> `61 passed`; `uv run ruff check tests/test_source_state_batched_observation_boundary_profile.py`
  -> passed.
- [x] Add compact self-audit fields for the train-facing
  `collect_search_backend=direct_ctree_gpu_latent` hook: direct-hook call
  count, fallback count, output rows, CTree traverse/backprop counts,
  recurrent calls, recurrent batch mean, and model-output D2H bytes/timers.
  Validation: `py_compile` passed, focused phase-profiler tests passed
  (`8 passed`), and focused ruff passed with the existing launcher F401 ignore.
- [ ] Run a sharper full-loop H100 A/B with sim16 and sparse env telemetry.
  The first sim8/C64 A/B proved the hook works structurally but did not produce
  wall-clock speedup; the next pair must make search a larger fraction of the
  denominator before deciding whether this hook is worth extending.
- [ ] Run a more stable no-RND search/backend grid before touching RND:
  C64 stock/direct and C128 stock/direct, H100, sim16, sparse telemetry,
  three learner calls. This prices whether the direct hook needs larger root
  batches and whether the current wall moves with collector size.
- [x] First sparse-telemetry search/backend wave completed. Result:
  direct wins in C64/sim16 (`387 -> 456` steps/sec quick row; `205 -> 421`
  steps/sec three-learner row), but does not win in C64/sim8 (`495 -> 477`)
  and loses in C128/sim16 (`478 -> 397`). C64/sim16 is promising but needs a
  repeat because the stable stock row looked anomalously slow.
- [ ] Repeat C64/sim16 three-learner stock/direct with the fallback-zero
  compact count fix in place. If this repeat confirms a real win, optimize the
  remaining direct-hook buckets next: output assembly, model-output
  D2H/listifying, and CTree Python/list boundary.
- [x] Implement profile-only `dense_torch_mcts_compile_spike` in the
  array-ceiling sidecar. The mode compiles only fixed-shape dense selection and
  backup helpers, keeps recurrent inference eager, and reports explicit
  compile-enabled versus fallback telemetry.
- [x] Update `scripts/build_curvytron_hybrid_observation_profile_grid.py` so
  the standard grid builder can emit `dense_torch_mcts_compile_spike` rows and
  records the correct array-ceiling input mode in its fixed denominator.
- [ ] Run the H100 compile-spike falsifier before investing more in dense
  Torch search: B512/A16, sim8 and sim16, root-noise0, all actions legal,
  60 measured / 15 warmup. Compare against `direct_ctree_gpu_latent` and
  `recurrent_toy` on the same denominator. Count only rows with
  `lightzero_array_ceiling_compile_enabled == 1.0` as compile evidence.
- [x] Run combined focused validation after main-thread and subagent test edits:
  `uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_mock_collector.py tests/test_exploration_bonus.py tests/test_vector_reset.py tests/test_vector_autoreset.py`
  -> `142 passed, 6 skipped`; matching ruff command -> passed.
- [x] Reconcile validation gate status after doc review: stock normal-death
  and stock `rnd_meter_v0` gates already passed as profile gates. Remaining
  P0 work is speed-row semantic attestation, LightZero decode edge cases, and
  deeper collect/search split. Positive RND remains blocked on novelty
  normalization/resume semantics, not on meter-mode overhead plumbing.
- [x] Add summary-side speed-row attestation. The optimizer profile summarizer
  now separates `render_mode` from `render_sec`, flags missing semantic
  identity fields, and supports `--require-attestation`. Focused tests pass.
- [x] Add LightZero decode edge-case tests for string-keyed policy outputs,
  list outputs, nested root wrappers, missing actions, and illegal decoded
  actions.
- [x] Add producer-side compact semantic identity fields for dtype, scalar
  materialization, `to_play`, zero-mask/action-mask semantics, and consumer
  semantics. New compact outputs now include
  `semantic_identity.schema_id=curvyzero_optimizer_profile_semantic_identity/v0`.
- [x] Verify the producer-side `semantic_identity` block on a fresh Modal
  profile artifact. `opt-semantic-identity-smoke-20260521a` passes
  `--require-attestation` after fixing the local runner's nested-JSON parser
  and pinning stock trainer Torch to `2.8.0`.
- [x] Re-run the current H100/L4 split refresh. Fresh rows confirm the same
  read: H100 initial inference `~9466` roots/sec versus H100 collect-forward
  sim8 `~2304` roots/sec; cheap-pool/T4 collect-forward was `~1250` roots/sec.
  The next wall is public LightZero collect/search/output, not root inference.
- [x] Add first internal collect-forward model-call timers to the hybrid probe;
  local tests pass. Next remote row should split `collect_mode.forward` into
  model-call time versus non-model tree/output time.
- [x] Run first H100 internal collect-forward timer row. Result: over the
  measured B512/A16/sim8 row, collect-forward spent about `69.8s`, but timed
  model initial/recurrent calls were only about `2.7s`. The current wall is
  non-model LightZero tree/search/output handling.
- [x] Decide the next search-boundary experiment: add deeper LightZero MCTS
  timing hooks first. Subagent critiques agreed this is the fastest way to
  split the `~67s` non-model collect-forward residual before any replacement
  architecture.
- [x] Add profile-only deeper search timers to the hybrid boundary probe:
  `policy._mcts_collect.search`, ctree `batch_traverse`, ctree
  `batch_backpropagate`, MCTS non-model residual, and outside-MCTS residual.
  The hybrid timing aggregator now records these fields across measured steps.
  Validation: `87 passed`; focused ruff passed.
- [x] Collect the first H100/L4 deeper-search split rows and update the
  experiment log. Result: H100 collect-forward `35.36s`, model calls `1.81s`,
  MCTS search `10.97s`, raw ctree traverse/backprop `0.98s`, outside-MCTS
  residual `24.40s`.
- [x] Implement and validate the profile-only LightZero array-ceiling probe.
  Modes: `policy_arrays` and `recurrent_toy`. Validation:
  `91 passed`; focused ruff passed.
- [x] Collect first H100/L4 array-ceiling rows. Result: H100
  `policy_arrays` `9957.97` roots/sec, H100 `recurrent_toy` `8681.01`
  roots/sec, L4 `policy_arrays` `5589.96`, L4 `recurrent_toy` `5030.25`.
  The H100 recurrent toy is about `3.37x` faster than public MCTS collect
  while still doing 8 real recurrent model calls.
- [x] Partially advance the real arrays-in / arrays-out MCTS boundary with a
  profile-only `direct_ctree_arrays` probe. It calls real LightZero model +
  real CTree MCTS and returns compact arrays without public
  `collect_mode.forward`. It is not trainer advice.
- [ ] Finish parity gates for the direct arrays boundary. Required gates:
  fixed-seed comparison to LightZero for legal masks, to_play, root noise,
  temperature, value/reward support transforms, visit counts, decoded actions,
  and output schema.
- [x] Add a first P1 statistical comparison tool:
  `scripts/compare_curvytron_direct_ctree_stock.py`. It compares stock facade
  vs direct CTree over many small CPU LightZero batches and reports action
  agreement, visit-distribution L1, searched-value deltas, and illegal-action
  counts. This is validation tooling, not trainer code.
- [x] Run first P1 statistical comparison: sim4/root-noise-on over 4 seeds and
  8 roots per seed produced exact action/visit/value agreement and zero illegal
  actions. Focused direct-CTree/real-policy pytest slice passed: `8 passed`.
- [x] Run stronger P1 statistical comparison: sim8/root-noise-on over 8 seeds
  and 8 roots per seed also produced exact action/visit/value agreement and
  zero illegal actions.
- [x] Re-run focused boundary validation after the compare-tool addition:
  `tests/test_source_state_batched_observation_boundary_profile.py` -> `83
  passed`; ruff on the compare script and boundary tests -> passed; compare
  script py_compile -> passed.
- [x] Implement the first profile-only MCTS arrays-boundary facade. It still
  calls stock LightZero `collect_mode.forward`, then decodes compact arrays.
  This is a bridge for validation and future replacement, not a speed win yet.
  Validation: focused tests `94 passed`; focused ruff and py_compile passed.
- [x] Add the optional local LightZero dependency extra and verify imports:
  `LightZero==0.2.0`, `torch==2.8.0`, and `cloudpickle>=3`.
- [x] Run a remote Modal smoke for
  `--hybrid-lightzero-mcts-arrays-boundary-probe`. Run
  `ap-Amg22e2oRyJHZMNqPy9god` passed on L4/T4 with
  `semantics=stock_lightzero_mcts_arrays_facade`, `policy_device=cuda:0`,
  `action_shape=[8]`, `visit_shape=[8,3]`, `searched_value_shape=[8]`,
  `compact_output_bytes=192`, `public_output_bytes=1086`, and zero illegal
  decoded actions. This is a wiring proof, not a throughput claim.
- [x] Run medium B512/A16/sim8 arrays-boundary facade rows on L4/T4 and H100.
  L4/T4 run `ap-5COErgYQQR2Gb9IsShxOnY` reached `1421.28` scalar roots/sec;
  H100 run `ap-4KPxuHpOOw4AfgD7rrlqIu` reached `2319.65` scalar roots/sec.
  Both kept `trainer_defaults_changed=false`, `touches_live_runs=false`, and
  zero illegal decoded actions. This confirms the facade is wired at useful
  scale; it is not a speed win yet because it still calls the public LightZero
  MCTS branch.
- [x] Add local tests for direct CTree arrays: zero-mask filtering, full action
  id mapping, `to_play=-1`, `model.eval()` before root inference, compact array
  output, and the all-actions-legal output fast path.
- [x] Run direct CTree arrays Modal smoke:
  `ap-KIPWDDxQNinfSTjOZkI0rP` passed on L4/T4 with real LightZero
  `MuZeroPolicy`, CUDA model/search, real CTree MCTS, compact arrays out, and
  zero illegal actions.
- [x] Run direct CTree arrays medium rows. H100 direct before the output fast
  path reached `2806.64` roots/sec (`ap-XEoAIwCpbbQTuFLmSnjvwY`) versus the
  matched stock facade `2419.81` (`ap-HJk70PQP2iLAvA7mxxn99u`). L4/T4 direct
  reached `1460.41` roots/sec (`ap-5OB4ye6HKiGfPQ3UjP221v`).
- [x] Remove the measured direct-output assembly wall for the common
  all-actions-legal profile shape. H100 output assembly dropped from `4.709s`
  to `0.027s` over 25 measured steps, and throughput rose to `3859.44`
  roots/sec in run `ap-XEB8GF9B2Gw5V600QVtu10`.
- [x] Carry the input-mode split into the direct CTree arrays boundary. Current
  short H100 rows: host uint8 `5247.95` roots/sec
  (`ap-DoCqvAulFMhZyoAcownQmn`), pinned uint8 `4678.23`
  (`ap-APSw7b1ZSJjSSuPtGEHO3w`), resident reuse ceiling `5820.96`
  (`ap-KCtqhJDwTuLptLKd4XSv38`). Pinned lowered H2D but did not win total wall
  in the matched short row; resident reuse is an upper-bound only.
- [x] Run longer same-shape input-mode repeats to separate H2D improvement from
  short-row variance. H100 `60` measured / `15` warmup rows: host uint8
  `4111.80` roots/sec (`ap-QPLEHOs3dGrcs2tlRpbMge`), pinned uint8 `4513.15`
  (`ap-5F1tMU2HiuHXDcu4O1tGkw`), resident reuse ceiling `5537.40`
  (`ap-wsKyodSayU2KGsTgKKpAqc`). Pinned is now a modest stable win; resident
  reuse is still stale-input-only.
- [x] Add transfer/freshness accounting for direct CTree input modes. Future
  rows report `input_freshness`, transfer bytes, and model-output D2H
  time/bytes; old Modal rows predate some of those fields.
- [x] Re-run the same H100 B512/A16/sim8 `60` measured / `15` warmup shape
  with current telemetry fields. Result: stock facade `2473.11` roots/sec,
  direct fresh host `4564.03`, direct pinned `4113.52`, and resident stale
  ceiling `4884.69`. Fresh direct CTree is about `1.85x` over the stock facade.
  Pinned still cuts H2D, but it did not win total wall in this refresh.
  Resident reuse shows only small extra ceiling, so input transfer is not the
  main remaining wall.
- [ ] Next compact-MCTS target: split/reduce the remaining direct path wall
  after input-copy is priced. In the latest short H100 rows, the big buckets
  are still MCTS search, root prep, model calls, observation/stack, and runtime
  variance around CTree, not compact output assembly.
- [x] Test whether the array-ceiling toy's H2D copy can be removed by keeping
  the pre-scalar stack/model input resident. H100 toy rows spend about
  `2.4s` in H2D inside the array-ceiling bucket.
  Result: H100/B512/A16/sim8 `host_uint8` reached `10086.23` roots/sec,
  `host_uint8_pinned` reached `12295.15`, corrected `host_float32` reached
  `9641.80`, and `resident_torch_reuse` reached `14414.56`. The useful
  lesson is pinned `uint8` or resident input; host-side float preprocessing is
  worse once counted correctly.
- [x] Carry the pinned/resident-input lesson into the compact
  arrays-in/arrays-out MCTS boundary plan. The input split is profile-only and
  does not change trainer defaults; longer repeat rows are running to separate
  real transfer improvement from short-row variance.
- [x] Run matched P2 same-shape refresh for the active decision:
  stock facade, direct CTree host uint8, direct CTree pinned uint8, and
  resident stale-input ceiling on H100 B512/A16/sim8 with 60 measured and
  15 warmup steps. Result: stock facade `2670.68` roots/sec, direct host
  `4764.06`, direct pinned `3689.15`, resident stale ceiling `3069.08`.
  Direct host is about `1.78x` over stock. Pinned/resident input is not the
  active win.
- [x] Add and run pure-policy collect wrapper probe. Result: H100 pure-policy
  collect reached `6286.61` roots/sec versus MCTS collect `2572.12` roots/sec.
  Pure-policy collect-forward took `4.88s`; MCTS collect-forward took `35.36s`.
  The MCTS branch is the slowdown, but raw ctree traverse/backprop was only
  `0.98s`, so the next target is MCTS-branch setup/conversion/result handling.
- [x] Build a profile-only replacement-ceiling toy for MCTS-branch
  representation/output: batched roots in, compact arrays out, no per-root
  Python dict/list fanout. Use this to estimate the practical ceiling before
  touching production training.
- [x] First toy modes should be `policy_arrays` and `recurrent_toy`. Both must
  reuse the same pre-scalar `[B,2,4,64,64]` stack, real scratch MuZero policy,
  zero-mask filtering, fixed-opponent `to_play=-1` convention, CUDA sync, output
  checksums, and illegal-action checks. Neither may call `collect_mode.forward`
  or `_mcts_collect.search`.
- [x] Compare first H100/L4 array-ceiling rows against initial-only,
  pure-policy collect, and MCTS collect. Rows:
  `ap-IubMKmFoUag2Fq2alyd2ZU`, `ap-sIbE42nQBd7vTSi1zWvXnf`,
  `ap-0vgrMYtrZCzD21rXCmDbDX`, `ap-WdE4qyUHLyPoaU1KyYPtO9`. Result: the
  ceiling is high enough to justify designing a real arrays-in / arrays-out
  MCTS boundary, with validation before any trainer change.

## Now

- [x] Build the first local profile-only scalar-action bridge that preserves batched GPU
  observations through `SourceStateMultiplayerTrainerSurface` and scalarizes
  only at the LightZero collection boundary. It now accepts scalar env-id
  actions, commits one batched joint CurvyTron step, and returns scalar
  timesteps keyed by env id.
- [x] Extend that bridge into the first base-manager-shaped local profile
  canary. `BatchedLightZeroProfileEnvManager` now exposes `env_num`,
  `ready_obs`, `reset`, `step`, `seed`, `close`, and `last_reset_info`.
- [x] Wire the manager-shaped bridge into the Modal/profile canary so it runs
  on H100 with the real direct GPU renderer. First B64 row passed with
  `env_num=128`, `scalar_env_instances_created=0`, and median manager step
  about `0.0236s`.
- [x] Collect wider/RND manager-facade rows to see whether payload or RND
  becomes the next local wall.
- [x] Prewarm dynamic JAX render widths before measured manager-facade steps so
  first-use JIT compilation does not pollute p95 timing.
- [x] Build the first profile-only stock LightZero manager adapter canary. This
  must call `train_muzero`, preserve one batched CurvyTron surface internally,
  and report that scalar env instances were not created. First successful
  stock-boundary row: `opt-batched-stock-canary-20260520a/envmgr-b16-sim2e`,
  `called_train_muzero=true`, `env_steps_collected=16384`,
  `mcts_search_calls=1024`, `replay_sample_calls=1`, `learner_train_calls=1`,
  about `150.19 steps/s`.
- [x] Run the first base-manager full-loop A/B: CPU oracle versus batched
  `direct_gray64` GPU facade, same topology and workload counts. C16 batched
  GPU was about `150.19 steps/s`; C16 base CPU-oracle was about `98.01`
  steps/s.
- [x] Run the first larger batched-GPU topology probe to see whether C64/root
  batching improves on the C16 canary. C64 batched GPU improved to about
  `416.89 steps/s`.
- [x] Run the matched production-like subprocess CPU-oracle C64 control. It
  reached about `883.03 steps/s`, so it still beats the current batched GPU
  manager by about `2.12x`.
- [x] Add direct batched-manager `step`/`reset` timing inside the stock profile
  output, then rerun C64. The rerun shows manager step took about `109.44s`,
  with renderer/stack work about `92-94s` of a `151.50s` wall.
- [x] Collect detached 2026-05-20 reruns:
  `opt-batched-stock-detached-c80-20260520a/envmgr-c80-sim2-timed` and
  `opt-stock-cpuoracle-detached-subproc-c128-20260520a/sim2-control`. C80
  batched GPU reached about `513 steps/s`; C128 subprocess CPU-oracle reached
  about `940 steps/s`.
- [x] Patch the stock batched GPU profile hook to skip all-width dynamic JAX
  prewarm. Tiny C2 smoke passed, and reset dropped from about `19s` to about
  `3.5s`.
- [x] Run small no-prewarm dynamic stock rows. C16/C32 at 64 source steps both
  passed, so the hook itself is alive.
- [x] Finish the first dynamic-width failure-boundary probe. C64 at 256, 512,
  and 1024 source steps all passed synchronously. The earlier C64/C96
  profile-spawn failures now look like spawn/readback/artifact fragility, not a
  hard runtime failure.
- [x] Collect wider synchronous dynamic-width rows. C96/1024 reached about
  `759 steps/s`; C128/1024 reached about `879 steps/s`.
- [x] Run a step-back critique pass before more code changes. Current concern:
  batched GPU is close to subprocess CPU-oracle only after C128, because the
  one-process batched manager loses the subprocess lane's CPU parallelism.
- [x] Add a zero-observation stock manager profile row. C64 and C128 both
  passed through stock `train_muzero` with real env stepping, MCTS, replay, and
  one learner call, but zero-filled observations. C64 reached about `1259.60`
  steps/s and C128 reached about `1557.42` steps/s.
- [ ] Record renderer time, surface non-render time, payload bytes, pickle
  time, learner/search counts, env steps, wall clock, and GPU utilization for
  every bridge row.
- [ ] Keep duplicate-seed/runtime variance as a background diagnostic. Use it
  only to check comparison anchors, not as the main optimization lane.
- [ ] Keep RND cadence as a separate profile axis. Compare no-RND, RND meter,
  cadence, and CPU-vs-CUDA rows without mixing those conclusions into renderer
  claims.

## Next

- [ ] Finish vector-facade semantic gates: terminal `final_observation` through
  the manager facade, RND latest-frame extraction through the manager facade,
  and then a Modal H100 manager row. Local missing/extra action rejection,
  invalid action rejection, terminal `done=true` timestep before autoreset, and
  neighboring-row stability already pass.
- [ ] Add fail-closed bridge metadata: exact renderer backend name, profile-only
  run id, no hidden CPU fallback, and explicit non-default backend reporting.
- [ ] If the base-manager bridge passes, repeat with subprocess workers to
  measure payload/process overhead.
- [ ] Decide whether the next real batched GPU lane is a multi-worker/batched
  manager, custom collector boundary, or smaller CPU-oracle optimization. The
  current one-process batched manager is not launch advice.
- [ ] If payload/process overhead dominates, evaluate low-risk payload slimming:
  fewer info fields, no terminal zero arrays when none are terminal, and
  uint8/shared-layout feasibility.
- [ ] If search dominates after the bridge passes, move the next lane to
  MCTS/root batching or topology tuning.
- [ ] Explain the C64/512 oddity: `learner_train_sec` jumped to about `21s`.
  Decide whether this is LightZero target/segment-length behavior, one-off
  runtime noise, or a broken dynamic profile row.
- [ ] If dynamic-width C64 fails or stays slow at 1024 source steps, stop
  treating dynamic render width as a simple win. Compare one stable render
  shape, a deliberately capped/truncated shape with fidelity checks, and a
  multi-worker batched GPU manager that keeps subprocess-style parallelism.
- [ ] Finish analyzing the fresh matched rows. C128 CPU-oracle subprocess
  completed at about `857.48 steps/s`; C128 real batched-GPU render completed
  at about `978.96 steps/s`; C128 zero-observation completed at about
  `1557.42 steps/s`. C256 real batched-GPU render completed at about
  `1193.48 steps/s`; C256 zero-observation completed at about `1748.18`
  steps/s. C512 zero-observation completed at about `1805.22 steps/s`.
  C256 CPU-oracle completed at about `722.43 steps/s`; C512 real batched-GPU
  render completed at about `1352.47 steps/s`.
- [x] Finish analyzing the post-patch batched manager rows. The C256
  real-render row completed at about `1096.18 steps/s`, and the C256
  zero-observation row completed at about `1735.15 steps/s`. The C512
  real-render row completed at about `1439.84 steps/s`, improving over the
  prior C512 anchor (`1352.47`) by about `6.5%`. Treat the patch as a small
  C512 win and noisy/neutral at C256, not a 10x-class change.
- [ ] Consider a multi-worker batched GPU manager only after the zero-observation
  row shows there is enough non-render headroom. The goal would be to preserve
  subprocess-style parallelism while batching each worker's local rows.
- [ ] Keep the backend identity language strict. In batched profile rows,
  `env_manager_type=curvyzero_batched_profile` is the real GPU-observation lane;
  `policy_observation_backend=cpu_oracle` can still appear in command JSON
  because it is the scalar wrapper config field.
- [ ] Reprioritize after the C512 post-patch row: renderer/stack work still
  matters, but at C512 the zero-observation ceiling is only about `1.25x` above
  real render. Next high-leverage work should either reduce manager/policy/search
  overhead or prototype a hybrid actor-parallel plus batched-GPU-render manager.
- [x] Collect corrected `20260521b` gate grid:
  C512 no-RND repeat, C512 RND update10, C512 RND update100, C256 normal-death,
  and C768 real-vs-zero. The first `20260521a` attempt failed because the
  profile-grid builder did not pass an even `--evaluator-env-num`; the builder
  now emits that flag. Final read: RND meter passes as an overhead/safety row;
  C768 does not produce a clean scaling win; C512 remains the cleaner saturation
  anchor.
- [x] Collect corrected normal-death gate `20260521d`. The `20260521b`
  normal-death row failed after 44 manager steps because dynamic GPU render
  only accepted full row-major requests. The renderer now gathers partial
  row/player requests after full-batch render; local focused tests pass. The
  `20260521c` row then failed because LightZero omitted a complete physical row
  from the action dict. The bridge now accepts complete-row omission while still
  rejecting half-row omission. `20260521d` passed through stock
  `train_muzero` with normal death/autoreset: `ok=true`, `36014` raw env steps,
  `485.01 steps/s`, `333` MCTS searches, one replay sample, and one learner
  call.
- [x] Interpret C512 RND gate rows. `rnd_meter_v0` update10 and update100 both
  passed with predictor changed, target frozen, and reward unchanged. They are
  overhead rows only; the reward-model path uses MCTS-root fallback for the
  denominator because raw compact env steps still report zero.
- [ ] Rerun the stock H100 C512/sim4 CPU-oracle no-RND anchor only when a bridge
  comparison needs a fresh baseline.
- [ ] Add same-seed reset/action tracing only if repeated anchor rows disagree
  in workload counts or reset behavior.
- [ ] Plan the next architecture probe from the new Amdahl read. Renderer-only
  work cannot deliver 10x at C512 because real render is already within about
  `1.25x` of the stable zero-observation ceiling. The next meaningful probe
  should combine actor parallelism with batched GPU rendering, or directly
  reduce policy/search/manager scalar overhead.
- [x] Scaffold the first profile-only hybrid actor plus central
  zero-observation harness. It does not call `train_muzero`, change defaults,
  or touch live runs. Local tests pass and the compact CLI is now usable for
  sweeps.
- [x] Run the first local hybrid zero-observation topology probes. B64/A4,
  B256/A8, and B512/A16 reached roughly `15.4k`, `21.6k`, and `24.9k` scalar
  timesteps/sec respectively. This proves topology headroom only; it excludes
  policy/search/replay/learner/RND/real render/IPC.
- [x] Add a renderer-backed hybrid mode seam that replaces the zero stack with
  an injected renderer. Keep it profile-only and no-training. Local CPU-oracle
  smoke and sentinel row/player-order tests pass.
- [x] Build the Modal/profile-only wrapper that injects the real dynamic JAX
  renderer into the hybrid seam. This stays outside the training module so
  Modal internals do not leak into training code.
- [x] Run first Modal hybrid GPU rows. B256/A8 reached about `4496` scalar
  steps/s and B512/A16 reached about `5447` scalar steps/s, profile-only,
  no-training, no live-run changes. These rows prove the architecture probe is
  alive, not trainer speed.
- [x] Run paired B512/A16 compact-payload pickle row. Compact payload pickle is
  tiny (`~21 bytes` per scalar timestep and about `0.0019s` over 20 steps);
  serialization of compact actor metadata is not the current wall in-process.
- [x] Run B1024/A16 scaling row. It reached about `6662` scalar steps/s, so
  wider batch still helps in the no-training harness. Observation remains the
  largest bucket and scalar materialization is starting to show.
- [x] Add terminal/final-observation semantics to the hybrid renderer-backed
  profile before any trainer bridge. Local sentinel tests and a CPU-oracle
  `max_ticks=1` smoke pass. A large GPU terminal/autoreset row is still a
  future profile check, not a blocker for no-death speed rows.
- [x] Preserve action masks through the profile scalarization seam. The scalar
  timestep materializer now accepts `[B,P,3]` or `[B*P,3]`, and the hybrid
  manager passes the merged actor masks instead of silently using all-true masks.
- [x] Remove the avoidable full-stack copy on non-terminal hybrid rows. Terminal
  rows still snapshot before autoreset; no-death profile rows now avoid copying
  the whole `[B,P,4,64,64]` float32 stack before scalarization.
- [x] Quantify the no-copy patch with fresh H100 B512/B1024 profile-only rows.
  B1024 no-probe improved from about `6662` to `9495` scalar steps/s.
- [x] Run B2048 no-probe and sim16 rows. B2048 no-probe reached about `11892`
  scalar steps/s; sim16 reached about `10536`.
- [x] Add fine-grain stack timers: `stack_shift_sec` and
  `stack_latest_update_sec`.
- [x] Use fresh H100 rows with fine-grain stack timers to split observation into
  render/transfer versus CPU stack update.
- [x] Try B4096 no-probe. It did not clearly beat B2048; current useful width
  for this in-process profile is probably B2048-ish.
- [x] Profile uint8 stack storage at B1024/B2048 and B2048 sim16. Result:
  stack-shift time drops, but CPU scalar materialization grows enough that this
  is not a current throughput win. It is evidence for a future device-resident
  handoff, not a trainer recommendation.
- [x] Run a small Modal GPU terminal/autoreset row with compact output to prove
  stale gap strings are gone and terminal counts survive the Modal path.
- [ ] Build the subprocess/IPC hybrid variant. The current actor count only
  partitions rows in process; it does not measure real actor fan-in or IPC.
- [ ] Add a subprocess/IPC variant of the hybrid harness after the in-process
  real-render version wins. The goal is to measure whether real actor
  fan-in/fan-out erases the local topology headroom.
- [ ] Measure a policy/search pressure probe after hybrid real render. The next
  gate is not only env/render speed; it is whether batched roots can feed
  model/search-like GPU work without scalarizing too early. This probe is
  synthetic and must not be reported as LightZero MCTS.
- [x] Add a profile-only device-resident handoff probe. It consumes the
  batched `[B,2,4,64,64]` stack before scalar LightZero row materialization,
  normalize on accelerator when needed, and report timings separately from the
  existing host-scalarized synthetic probe. Local tests pass; H100 comparison
  rows are running.
- [x] Interpret the pre-scalarization H100 comparison:
  float32/uint8 crossed with scalar-materialization on/off at B2048/A32,
  sim16 synthetic batched-stack pressure. This is a profile-only falsifier for
  device/batched handoff, not a trainer recommendation.
- [x] Correct the profile wrapper default to dynamic trail slots. Fixed
  full-slot rendering was a bad optimizer default and about `10x` slower on
  short trajectories; the Coach training launcher already used dynamic slots.
- [x] Finish the trajectory-length ladder with dynamic slots. B512 20/100/200/500
  rows show render becomes dominant as survival length grows; the 500-step row
  spends about `96%` of wall in observation and about `94%` in renderer render.
- [ ] Background only: reprioritize renderer work from the length ladder if a
  fresh Amdahl row re-shows observation dominance. Candidate lanes remain
  cheaper long-trail representation, incremental/dirty long-trail update, or a
  lower-fidelity observation that preserves enough game signal. Current main
  lane is direct CTree fixed-seed parity and remaining direct-path split.
- [x] Run the existing local incremental trail-layer prototype. It is exact on
  its parity cases but not fast enough in Python: about `0.31x/0.43x/0.78x`
  at 100/200/500 trail points and `1.25x` at 1000.
- [ ] Measure the existing integrated CPU dirty-block cache on current
  trajectory profiles, because the prototype result does not necessarily equal
  the production dirty cache path.
- [x] Measure the current local fixed-opponent CPU dirty-cache path. It stays
  near `425-433` env steps/s at 100/500/1000 no-death steps and has no
  fallbacks after cold start. Render is still about `77%` of local wall.
- [ ] Translate the dirty-cache cost-model lesson into the next Modal/profile
  gate: either profile current dirty cache inside a batched manager-like shape
  or prototype a device/batched dirty-update renderer.
- [ ] Add an isolated Modal synthetic benchmark for persistent policy-space
  framebuffer versus stateless redraw. It should not touch trainer defaults or
  live runs. Required metrics: device update/redraw, readback, end-to-end,
  throughput slope across steps, and parity against the stateless synthetic
  target.
- [x] Implement and smoke the isolated Modal synthetic benchmark. H100 B128/S64
  exact parity passed; persistent updates were `3.67x` faster end-to-end with
  readback and `7.86x` faster device-side.
- [ ] Collect larger rows for the persistent framebuffer benchmark: H100
  B512/S512, H100 B2048/S256, and L4/T4 B512/S512 are running.
- [x] Collect larger persistent framebuffer rows. Readback-included speedups:
  H100 B512/S512 `8.48x`, H100 B2048/S256 `5.06x`, L4 B512/S512 `10.86x`;
  all parity checks exact.
- [x] Finish follow-up rows: H100 B512/S512 without readback gave `38.57x`
  total speedup; H100 B512/S256 target `128x128` gave `4.60x` readback-included
  total speedup.
- [ ] Design the real profile-only persistent renderer boundary. Required
  inputs: append slots from `visual_trail_*`, reset/cursor-regression handling,
  game-clear handling, bonus/head overlays, controlled-player views, exact
  telemetry, and a stateless-reference parity sampler.
- [x] Fold in the independent local toybench. It showed direct-64x64
  append-only speedups from `10x` to `210x` versus replaying all old trail
  pixels, with exact final-frame parity.
- [x] Re-run focused validation after the benchmark/tooling changes:
  py_compile passed and targeted pytest returned `57 passed`.

## Done / Current Facts

- [x] `direct_gray64` is the fastest current profile-only surface candidate,
  but surface-only wins are not launch advice.
- [x] The renderer-backed trainer surface can run explicit GPU candidates
  without hidden CPU fallback.
- [x] Host-copy fixes moved the local wall away from renderer-only work; payload
  and non-render surface costs now matter.
- [x] RND cadence microprofiles show RND cost is separable from renderer cost.
  Treat cadence as a training/profile knob.
- [x] Repeated stock H100 rows showed runtime variance despite matching workload
  counts. Use conservative anchors and keep variance work diagnostic.
- [x] Duplicate-seed H100 C512/sim4 diagnostic finished with identical
  workloads between about `883` and `1081 steps/s`. Useful as anchor noise, not
  a new optimization lane.
- [x] Focused local bridge/surface/boundary tests passed: `70 passed, 2 skipped` after
  adding scalar env-id mapping, joint-action commit, manager-shaped `ready_obs`
  and `step`, missing/extra action rejection, invalid-action rejection, and
  manager close/reset guards. This includes a terminal `max_ticks=1` gate that
  keeps final observations before autoreset.
- [x] H100 manager facade B256 no-RND passed with `env_num=512`, median manager
  step about `0.0806s`, and payload bytes about `33.6MB`.
- [x] H100 manager facade B128 CUDA RND update10 passed; RND train median was
  about `0.253s`, much larger than the direct manager surface. Treat RND
  cadence as a separate wall.
- [x] Prewarmed H100 manager facade B128 no-RND passed with median/p95 manager
  step about `0.0438s` / `0.0557s`. Prewarming removed the earlier dynamic JAX
  first-use p95 spikes.
- [x] Prewarmed H100 manager facade B256 no-RND passed with median/p95 manager
  step about `0.0755s` / `0.0983s`; payload pickle stayed visible at about
  `0.0344s` for `33.6MB`.
- [x] Prewarmed H100 manager facade B128 CUDA RND update10 passed with
  median/p95 manager step about `0.0427s` / `0.0493s`, but RND predictor
  training remained about `0.253s` median. This is the current obvious RND wall
  if update10 is used.
- [x] Stock-boundary batched GPU manager canary passed at C16/sim2/no-RND:
  `train_muzero` ran through collector, MCTS, replay sample, and one learner
  call. This proves integration, not speed. First throughput was about
  `150.19 steps/s`, with collection dominating wall time.
- [x] Matched C64 comparison says current production-like subprocess CPU-oracle
  is still faster than the profile-only batched GPU manager: about `883.03`
  steps/s versus `416.89` steps/s.
- [x] Timed C64 batched GPU rerun says the one-process batched manager is still
  observation/render-stack dominated: manager step about `109.44s`; renderer
  aggregate about `92.02s`; device render about `82.28s`; policy forward about
  `27.67s`; MCTS about `8.23s`.

## Not Now

- Do not change live training runs.
- Do not promote batched GPU observations to trainer, tournament, or checkpoint
  defaults.
- Do not promote scalar `policy_observation_backend=jax_gpu`.
- Do not present duplicate-seed variance as the main optimizer lane.
- Do not combine RND cadence conclusions with renderer/full-loop bridge claims.
- Do not recommend direct GPU for Coach until the vector/full-loop bridge has a
  passed full-loop profile.
- Do not recommend current one-process batched GPU manager for Coach training
  just because the renderer itself is fast. The full-loop C64 control says the
  subprocess CPU-oracle path is still faster today.
- Modal run pattern for this launcher is explicit: use
  `-m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train::main`.
  The module alone is ambiguous because it has multiple functions and local
  entrypoints.

## Active: Search Boundary Escape

- [x] Re-read current LightZero CTree boundary and upstream source. Finding:
  LightZero already uses C++ for `batch_traverse` and `batch_backpropagate`, but
  `MuZeroMCTSCtree.search` is still a Python loop that gathers latent states,
  copies recurrent outputs to CPU, converts arrays to lists, and calls the C++
  kernels once per simulation.
- [x] Create the escape-plan doc:
  `search_boundary_escape_plan_20260521.md`.
- [x] Record the tactical GPU-latent CTree lane:
  `gpu_search_fix_ladder_20260521.md`. This is profile-only and keeps latent
  states on GPU inside the LightZero CTree loop, while CTree selection/backprop
  remains CPU.
- [x] Sidecar: GPU-latent CTree feasibility. Read-only result says the local
  `direct_ctree_gpu_latent` lane is the smallest concrete prototype and should
  be tested before a bigger rewrite.
- [x] Sidecar: C++/Cython array-native boundary feasibility. Read-only result:
  LightZero already has C++ CTree kernels; the remaining feasible Cython win is
  array-native root/traverse/backprop/output APIs and less Python/list fanout,
  not "turn CTree on."
- [x] Sidecar: accelerator-native MCTS / MCTX feasibility. Read-only result:
  MCTX is the clean device-resident shape, but it is a new JAX lane rather than
  a LightZero patch.
- [x] Sidecar: MiniZero/KataGo/OpenSpiel batched actor architecture. Read-only
  result: useful architecture pattern, not immediate dependency.
- [x] Launch matched profile wave for `stock_facade`,
  `direct_ctree_arrays`, and `direct_ctree_gpu_latent` on H100
  B512/A16/sim8/60 measured/15 warmup, plus an L4/T4
  `direct_ctree_gpu_latent` sanity row.
- [x] Validate `direct_ctree_gpu_latent` locally against stock/direct arrays.
  Focused tests passed, and the comparison tool returned exact agreement on
  8 seeds at sim8/root-noise-on.
- [x] Harden the stock/direct comparison helper with CUDA reporting, strict
  gates, predicted-value diffs, and policy-logit diffs.
- [ ] Finish a tiny Modal CUDA parity canary with debug arrays included. It
  should compare stock/direct/GPU-latent at roots `<=16`, sim8, same seed, and
  should prove actual CUDA behavior rather than only CPU tensors.
- [x] Add the fixed-denominator manifest preset for the boundary comparison.
  Use `--next-direct-ctree-comparison-preset`.
- [x] Replace GPU-latent list/group gather with a preallocated GPU latent pool.
- [x] Collect post-pool sim8 repeat and H100 sim16 stock/direct/GPU-latent rows.
  Result: sim8 post-pool was roughly neutral; sim16 GPU-latent was about
  `2.8x` over stock facade and about `1.6x` over direct CTree arrays.
- [ ] After sidecars return, choose the first implementation lane:
  if GPU-latent CTree wins materially, finish its gates; if it barely moves,
  prototype dense Torch MCTS and/or Cython array-native CTree instead of
  polishing partial glue.
- [ ] Run the next big-swing proof. Current candidates:
  dense GPU MCTS for fixed `A=3`, or Cython array-native CTree boundary.
  The goal is to remove the remaining CPU/list/search boundary, not render.
- [x] Implement the first profile-only dense GPU MCTS proof
  (`dense_torch_mcts`) for fixed `A=3`.
- [x] Run the first matched H100 dense row. Result: v0 worked but was only
  about `1.07x` faster than the same-run `direct_ctree_gpu_latent` row
  (`6418` vs `6001` scalar roots/sec) and remained below the recurrent-toy
  ceiling (`7467` scalar roots/sec).
- [x] Apply the first dense-mode overhead cleanup: remove inner-loop `.item()`
  CPU sync checks, preallocate path history tensors, and remove per-depth
  bootstrap cloning.
- [x] Reprofile cleaned `dense_torch_mcts` on the same H100 B512/A16/sim8
  denominator. Result: `7720` scalar roots/sec, about `1.30x` over same-run
  `direct_ctree_gpu_latent` and about `0.85x` of the same-run recurrent-toy
  ceiling.
- [x] Apply second dense-mode cleanup: skip per-recurrent CUDA sync in dense
  profile mode and avoid the host boolean-filter copy when all roots have legal
  actions.
- [x] Reprofile second-cleanup dense mode at H100 B512/A16/sim8 and sim16.
  Result: sim8 nudged to `7969` scalar roots/sec, but sim16 fell to `4135`
  scalar roots/sec. The eager dense tree scales poorly as depth grows.
- [x] Collect same-denominator sim16 `direct_ctree_gpu_latent` and
  `recurrent_toy` rows before making the next lane decision. Result:
  `direct_ctree_gpu_latent` reached `5010` scalar roots/sec and
  `recurrent_toy` reached `9134`, confirming dense eager Torch was the wrong
  sim16 shape before the fixed-shape rewrite.
- [x] Patch dense Torch MCTS to use fixed-shape all-root masked
  selection/expansion/backprop instead of dynamic boolean-indexed slices in the
  common all-roots path. Local ruff, py_compile, and focused array-ceiling
  tests passed.
- [x] Collect fixed-shape dense H100 sim8/sim16 repeats and compare against
  same-denominator GPU-latent CTree and recurrent-toy controls. Result after
  semantic fixes: sim8 `8288` roots/sec, sim16 `4294` roots/sec.
- [x] Decide next lane from the fixed-shape rows. Decision: keep
  `direct_ctree_gpu_latent` as the practical LightZero-shaped baseline; do not
  keep polishing eager dense Torch for sim16. If dense GPU search continues, it
  needs a compiled/fused fixed-shape pass or a different batched-search
  topology.
- [x] Apply dense semantic fixes found by sidecar critique: reward+discount
  backup, legal-only root-noise mixing, meaningful zero-visit fallback, skip
  unused policy-array work in dense mode, and binary-mask validation.
- [x] Validate the dense semantic fixes locally:
  `uv run ruff check ...` passed, `uv run python -m py_compile ...` passed, and
  focused pytest `5 passed, 81 deselected`.
- [ ] Next if staying local: run one tiny compiled/fused dense search spike
  only if it is fixed-shape and bounded. Otherwise move to array-native CTree
  API design.
- [ ] Next if preparing Coach-facing advice: full-loop profile only the best
  practical path, with RND/death/checkpoint knobs matched to the current Coach
  lane. Do not use profile-only dense rows as training-speed claims.
- [x] Collect sim32 scaling rows for `direct_ctree_gpu_latent`,
  `dense_torch_mcts`, and `recurrent_toy` on the fresh H100 denominator.
  Result: GPU-latent CTree `4127` roots/sec, dense Torch `2007`, recurrent-toy
  ceiling `6162`. Deeper search makes eager dense Torch worse.
- [x] Add profile-only CPU-heavy LightZero boundary compute route:
  `gpu-h100-cpu64`. Modal rejects `cpu=128`; the legal maximum for this
  function request is 64 cores. This route is only for testing whether more CPU
  helps CTree/search; it does not affect trainers or live runs.
- [x] Collect CPU-scaling falsifier rows: H100+64 CPU
  `direct_ctree_gpu_latent` sim16, plus H100+64 CPU stock facade sim16.
  Result: both got slower (`direct_ctree_gpu_latent` `6145 -> 5119`
  roots/sec; stock facade `2094 -> 1776`). More CPU cores are not the fix for
  this profile shape.
- [x] Reorient from the CPU falsifier. Treat CPU64 as negative/noisy evidence,
  not a new lane. The real fix is still the search boundary shape, and the
  best next experiment is to connect `direct_ctree_gpu_latent` to the real
  `train_muzero` denominator.
- [x] Add the first train-facing profile hook:
  `collect_search_backend=direct_ctree_gpu_latent`. It patches
  `MuZeroPolicy._forward_collect` only inside the stock `train_muzero` profile
  run, keeps stock collector/replay/target/learner ownership, returns stock
  per-env collect dicts, and is rejected outside `mode=profile`.
- [x] Validate the hook locally. `py_compile` passed; focused phase-profiler
  tests passed (`8 passed`); `ruff --ignore F401` passed for the touched files
  because the launcher currently has pre-existing unused imports.
- [x] Run a tiny remote full-loop smoke with
  `collect_search_backend=direct_ctree_gpu_latent`: `called_train_muzero=true`,
  one learner train call, zero illegal actions, and no replay/target schema
  break.
- [x] Run the matched full-loop A/B: stock collect/search versus
  `direct_ctree_gpu_latent` collect-search hook, with the same H100 profile
  settings, death/RND/checkpoint/eval/GIF choices, and at least one learner
  train call.
- [x] Repeat the suspicious C64/sim16 three-learner row. Corrected read:
  stock repeated at `445.19` steps/sec and direct repeated at `438.56`
  steps/sec. Direct reduced the search buckets but did not win full-loop wall
  time. The earlier apparent `2x` result was an anomalously slow stock row, not
  a stable speedup.
- [x] Decide the next search-boundary patch. Current highest-signal target is
  the direct hook's remaining CPU/list/output boundary: model-output
  D2H/listifying (`3.41s`), per-env stock output assembly (`2.87s`), and the
  CTree Python/list API. Do not chase CPU-count tuning unless a fresh profile
  shows a different CPU-parallel section. Decision after the next wave: output
  assembly was worth fixing; D2H packing was only hygiene; the strategic patch
  is array-native CTree or compiled/fused batched search.
- [x] Implement the first low-risk output assembly patch. It adds a
  profile-only all-actions-legal fast path inside the direct collect hook,
  preserves the stock per-env dict return shape, and records
  `collect_search_backend_output_fast_path_calls`.
- [x] Validate the output fast path locally. `py_compile` passed, focused
  phase-profiler tests passed (`8 passed`), and ruff passed with the same
  pre-existing F401 ignore.
- [x] Repeat matched H100 C64/sim16/3-learner stock/direct rows after the
  output fast path. Result: stock `433.17` steps/sec; direct output-fast
  `566.19` steps/sec. This is a real profile-loop win in this denominator
  (`~1.31x`), with `fast_path_calls=256`, fallback `0`, and output assembly
  `0.077s`.
- [x] Reorient after the output fast path. Output assembly is no longer the
  wall. Next candidate walls: model-output D2H/list conversion (`2.47s`),
  recurrent inference (`4.28s`), MCTS/search (`8.06s`), and the stock
  collector/env/replay shell around them. CPU count remains parked. Follow-up:
  listify was measured at only about `0.08s`; packed transfer did not create a
  material full-loop speedup.
- [x] Run the same H100 C64/sim16/3-learner A/B with `rnd_meter_v0` enabled.
  Purpose: check whether the output-fast win survives the current reward-model
  entrypoint and RND meter plumbing.
  Result: stock `342.33` steps/sec; direct output-fast `410.55` steps/sec
  (`~1.20x`). RND adds independent overhead (`rnd_train_with_data ~3.5s`,
  `rnd_state_hash ~3.0s`), so the search hook helps but cannot solve the whole
  current trainer path.
- [x] Run a longer no-RND H100 C64/sim16 A/B with 10 learner calls. Purpose:
  check whether the `~1.31x` output-fast win survives a slightly steadier
  train_muzero profile denominator. Result: direct still looked faster, but the
  stock row was unusually slow, so do not quote the lopsided `2.47x` as the
  stable claim. The defensible no-RND claim remains the matched 3-learner
  repeat: `433 -> 566` steps/sec.
- [x] Remove the RND diagnostic hash wall. Predictor/target state hashes now
  run once before and once after the RND update batch, not around each of the
  100 updates. Local RND tests passed. Matched RND profile after the fix:
  stock `351.02` steps/sec, direct `448.52` steps/sec; `rnd_train_with_data`
  fell from about `3.5s` to about `0.6s`.
- [x] Mirror packed recurrent model-output transfer in both the train-facing
  direct hook and the boundary sidecar. Local compile, focused pytest, and ruff
  passed. Result: safe, but only small throughput movement; not the main lane.
- [ ] Next implementation lane: design the smallest semantics-preserving
  array-native CTree fixed-`A=3` boundary, or a bounded compiled/fused batched
  search spike. The test must compare against the current best profile-loop
  denominator, not only profile-only roots/sec.
- [x] Add the train-facing hook feasibility note:
  `direct_ctree_gpu_latent_forward_collect_hook_feasibility_20260522.md`.
  Current read: do not chase CPU64. The next real optimizer move is a
  profile-only `_forward_collect` hook that returns stock LightZero collector
  fields while keeping root latents on GPU during CTree search. Importing the
  Modal profile module into the trainer is the wrong shape; start with a small
  reversible profile hook or factor a pure helper after the full-loop A/B wins.

## 2026-05-22 Radical Architecture Reorientation Tasks

- [x] Reframe the active optimizer lane after the latest profile rows:
  `direct_ctree_gpu_latent` plus output-fast is a real `~1.28-1.31x`
  train-profile win, but not a 5-10x architecture.
- [x] Record the external architecture refresh: OpenSpiel, MiniZero, KataGo,
  and MCTX all point toward batched inference/search ownership rather than tiny
  scalar wrapper cleanup.
- [x] Launch parallel critique/research lanes for external architecture,
  current hot boundary, compile-spike feasibility, and validation gaps.
- [x] Add the stronger train-facing hook parity gate: actual installed hook
  versus stock collect output over forced masks, clear preference, and
  no-noise deterministic rows. Current coverage: mixed masks, single-legal
  exact rows, full action ids, raw legal-action visit lists, and fail-closed
  bad masks. Keep ordinary tie-heavy visits statistical.
- [x] Add hook-output-shaped compact target material canary: active-root
  `selected_action[N]`, `visit_policy[N,3]`, and `root_value[N]` are checked
  against compact sidecars before target-row materialization.
- [x] Add stochastic selector distributional parity for the all-actions-legal
  output fast path.
- [ ] Add GPU-latent row/column sentinel for latent gather and
  reward/value/logit unpack order.
- [x] Add summary-side gate that rejects under-attested direct rows before
  Coach-facing recommendations.
- [x] Implement and explicitly falsify the profile-only
  `dense_torch_mcts_compile_spike` mode. Fixed shape first: all roots legal,
  `A=3`, root-noise `0.0`, recurrent inference eager, compile/fuse only the
  pure tensor search/update shell if feasible. H100 result: sim8 won
  (`10298` roots/sec), but sim16 failed the practical gate (`4873` roots/sec
  versus `6154` for `direct_ctree_gpu_latent`).
- [x] If the compile spike does not beat `direct_ctree_gpu_latent` at sim16 on
  the matched H100 denominator, stop polishing dense eager Torch and move to
  array-native fixed-A=3 CTree. Decision: stop polishing this exact compiled
  helper. Keep it as profile-only evidence.
- [x] Design the mock search-service ceiling. Purpose: prove whether removing
  CTree/object packaging entirely would expose enough full-loop headroom to
  justify the bigger MiniZero/KataGo-shaped rewrite. Plan doc:
  `mock_search_service_ceiling_plan_20260522.md`.
- [x] Implement the first profile-only `mock_search_service` ceiling mode in
  the hybrid observation sidecar. It reuses real batched CurvyTron
  observations, real legal masks, real scratch MuZero `initial_inference`, and
  returns compact legal action/visit/value arrays while reporting zero real
  CTree calls and zero recurrent rollout calls. Validation: focused boundary
  tests `11 passed`, grid-builder tests `2 passed`, ruff and py_compile
  passed.
- [x] Run the first H100 mock search-service ceiling wave against
  `direct_ctree_gpu_latent` and `recurrent_toy` on the same B512/A16/sim16
  denominator.
- [x] Add a durable runner for hybrid boundary profile manifests:
  `scripts/run_curvytron_hybrid_observation_profile_manifest.py`. Reason:
  raw detached Modal sessions and app logs can lose the JSON result after
  compaction. The runner executes blocking profile-only commands, captures
  stdout, parses the compact JSON, and writes per-row result files locally.
- [x] Finish the durable H100 falsifier rerun launched through that runner:
  `opt-hybrid-durable-mock-h100-20260522a`,
  `opt-hybrid-durable-direct-h100-20260522a`, and
  `opt-hybrid-durable-recurrent-h100-20260522a`.
  Result: `mock_search_service` sim16 `11648.29` roots/sec,
  `direct_ctree_gpu_latent` sim16 `5303.97`, `recurrent_toy` sim16 `8512.57`.
  Plain read: compact search-service shape is about `2.20x` over current
  direct in this profile, useful but not a standalone 10x proof.
- [x] Fold PufferLib into the external architecture critique. Initial read:
  its speed story is contiguous buffers, no dynamic allocation, async env
  stepping, pinned transfer, CUDA graph replay, and native env writes straight
  into training buffers. That supports a bigger vector/search-service rewrite,
  not another scalar wrapper patch. Fresh repo inspection found the concrete
  pattern in `StaticVec`: flat obs/action/reward/done/mask buffers, env-owned
  buffer slices, pinned host/device copies, per-buffer streams, and one-allocation
  tensor ownership.
- [x] Start the native/vector buffer architecture plan:
  `native_vector_buffer_architecture_plan_20260522.md`.
- [x] Add the first local Puffer-style native-vector boundary probe to
  `scripts/profile_hybrid_batched_observation_manager.py` and record the
  result. B512/A16/steps100/zero-observation/uint8/no-pickle:
  no-scalar + native probe `23515` timesteps/sec, scalar-only `18604`,
  scalar + native probe `17380`. Scalar materialization costs about `2.07s`
  over `102400` timesteps; native compact probe costs about `0.62s`;
  actor_step_wall is about `3.42s` in the no-scalar row. Read: object edge
  matters, but actor/env scheduling is now visible too.
- [x] Run a CPU-oracle comparator for the same local topology question. Result:
  B128/A8 cpu-oracle rows were both about `193` timesteps/sec because
  renderer/observation consumed about `52s` of `53s`. Read: CPU-oracle is not
  useful for judging the Puffer-style scalar edge; it is a renderer-wall
  control.
- [x] Implement the first opt-in native actor-buffer falsifier in
  `src/curvyzero/training/source_state_hybrid_observation_profile.py` and
  expose it through `scripts/profile_hybrid_batched_observation_manager.py
  --native-actor-buffer`. It is profile-only and zero-observation-only. Matched
  local B512/A16 result: old payload path `40477` timesteps/sec; native actor
  buffer `67890` timesteps/sec, about `1.68x`. Focused tests: `17 passed`;
  ruff and py_compile passed.
- [x] Add compact search-output-to-target-array validation so closed compact
  consumers do not need to allocate `PolicyRowRecordV0` objects in the hot
  path. Local compact target arrays dropped the target edge from millisecond
  class to about `0.1-0.3ms` per B512-B2048 compact step.
- [x] Fix compact RND latest-frame extraction to slice the latest channel
  before normalization. Before the fix, the compact helper normalized the full
  `[B,P,4,64,64]` stack and then kept only `[B*P,1,64,64]`. Validation:
  compact RND tests `3 passed`; focused optimizer boundary suite
  `158 passed, 2 warnings`.
- [x] Run closed compact consumer local scaling rows. Best current row:
  B2048/A16/uint8/no-scalar/native-actor-buffer/arrays `71605` timesteps/sec
  versus native-vector mock ceiling `80443`. Read: compact sidecar/RND/target
  overhead is no longer the local toy-denominator wall.
- [x] Clean up the train-facing profile-only `direct_ctree_gpu_latent`
  all-actions-legal root-prep path: one rectangular mask parse, shared
  legal-action pattern, batched Dirichlet root noise, and no public output
  contract change. Focused hook tests passed.
- [x] Clean up RND meter zero-weight estimate bookkeeping: no target-reward
  deep copy and no delta array work when `intrinsic_reward_weight == 0.0`;
  reward-neutral meter semantics are unchanged.
- [x] Rerun the combined focused optimizer validation after the latest direct
  hook and RND edits:
  `tests/test_source_state_hybrid_observation_profile.py`,
  `tests/test_source_state_batched_observation_boundary_profile.py`,
  `tests/test_exploration_bonus.py`,
  `tests/test_multiplayer_source_state_target_rows.py`, and
  `tests/test_lightzero_phase_profiler.py` -> `174 passed, 2 warnings`.
- [x] Add the mock search-service public-output edge switch:
  `--hybrid-lightzero-mock-service-materialize-public-output`. It materializes
  LightZero-shaped collect dicts from compact mock arrays and reports count,
  bytes, seconds, and checksum. Focused tests and grid-builder checks passed.
- [x] Record the explicit CompactSearchReplayV1 design note:
  `compact_search_replay_contract_plan_20260522.md`.
- [x] Run local closed-compact scale checks beyond the old B2048 row. B4096/A16
  reached about `66.8k` timesteps/sec and B4096/A8 about `68.4k`; B2048/A32
  stayed worse. Read: bigger batches can help, but more actor partitions are
  not automatically better.
- [x] Run fresh H100 mock/direct/recurrent sidecar comparison on current code:
  mock compact `8543.81`, mock public edge `8285.40`, direct CTree GPU-latent
  `5382.86`, recurrent toy `9068.59` steps/sec. Read: compact search-service
  headroom is real but currently about `1.6x`, not a search-only 10x proof.
- [x] Add `build_compact_target_rows_from_search_arrays_v0(...)` and focused
  tests showing compact search arrays can produce the same target rows as the
  existing object bridge on live and terminal/final-observation rows.
- [x] Run the broader focused optimizer validation after the latest edits:
  `193 passed, 2 warnings`; touched-file ruff passed.
- [x] Reconcile the side-agent critique. Current answer to "are we aggressive
  enough?" is no if we keep polishing wrappers. The next 5-10x candidate is
  compact batch ownership through actor/search/RND/target/replay, with scalar
  LightZero objects only at compatibility edges.
- [ ] Build the first native/vector buffer falsifier or delegate it as a
  bounded prototype: seeded compact CurvyTron batch state in, contiguous
  obs/mask/reward/done buffers out, scalar parity checks on a small subset.
- [x] Refresh the array-native fixed-A=3 CTree feasibility critique after the
  compile-spike result. Decide whether it is a conservative 1.5-2x bridge or
  a distraction from the search-service lane. Current critique: feasible, but
  likely `1.1x-1.4x` full-loop over current direct hook unless a micro-canary
  proves list/vector boundary overhead is worse than current timers suggest.
- [ ] Draft the MiniZero/KataGo-style batched search-service contract around
  the fixed-opponent lane: compact `obs_uint8[N,4,64,64]`, `legal_mask[N,3]`
  in; compact `action[N]`, `visits[N,3]`, `root_value[N]` out; stock replay
  materialized only at the compatibility edge.
- [ ] Extend the compact replay contract proof beyond the first direct builder:
  two/three-record chunks, RND latest-frame sentinel, non-identity row ids, and
  explicit player-perspective swap sentinel.
- [x] Extend the compact replay contract proof. Added a dedicated contract test
  covering two-record terminal/final-observation rows, three-record
  `record_index=1`, non-prefix active roots, RND latest-frame order,
  non-identity `policy_env_id` provenance, and player-perspective swap
  rejection.
- [x] Fix compact-root/replay-policy-row identity drift. Compact root rows are
  now distinct from compacted replay `policy_row`; source refs carry both
  `compact_root_row` and replay `policy_row`.
- [x] Add and run the no-model LightZero CTree list-ABI falsifier:
  `scripts/benchmark_lightzero_ctree_no_model.py`. First read: CTree-list is
  `12x-42x` slower than a simple vectorized fake-flat update on the local
  no-model denominator, but raw CTree alone is still far faster than the current
  full-loop sidecar wall.
- [x] Rerun the no-model CTree falsifier on H100 with visible JSON output.
  Representative H100 rows: CTree-list `0.45M-0.76M` nodes/sec, CUDA payload
  `0.51M-0.68M`, fake-flat `13.9M-19.4M`. Read: CTree/list is real overhead,
  but the bigger wall is still the model/search/replay boundary around CTree.
- [x] Run another web/subagent research wave on GPU/parallel MCTS and fast
  self-play systems. Notes added:
  `gpu_parallel_mcts_research_synthesis_20260522.md`,
  `subagent_mctx_gpu_search_research_20260522.md`,
  `subagent_gpu_mcts_implementation_patterns_20260522.md`, and
  `subagent_fast_rl_architecture_patterns_20260522.md`.
- [ ] Next aggressive implementation lane: prototype a compact search/replay
  consumer that avoids public per-env collect output fanout. Kill condition:
  if it cannot beat current direct train-profile speed by a `3x` class margin
  in a fair profile denominator, stop polishing this wrapper and move to the
  search-service/native-buffer architecture.
- [x] Implement the next P0 falsifier: precomputed recurrent-output direct
  search. Same root shapes as direct CTree, but replace model recurrent calls
  with resident tensors so we can separate recurrent launch/D2H cost from
  tree/list cost. It is profile-only, explicit opt-in, excluded from the
  default direct comparison preset, and labels logical vs actual model evals.
- [x] Run the small H100 precomputed recurrent smoke:
  B64/A8/sim8, 16 measured, 4 warmup. Result:
  direct `2357.77` roots/sec; precomputed recurrent `3745.00` roots/sec.
  Read: recurrent/output handling matters, but this is not a standalone 10x
  lane.
- [x] Finish the large H100 precomputed recurrent pair:
  B512/A16/sim16, 60 measured, 15 warmup. Result:
  direct `4920.30` roots/sec; precomputed recurrent `6771.37` roots/sec.
  Read: about `1.38x`; recurrent/output handling is meaningful but not enough
  for the 5-10x target by itself.
- [x] Interpret the large precomputed recurrent
  row in Amdahl terms. If removing recurrent calls is still only a 1-2x local
  split, stop treating recurrent inference as the main blocker and move to the
  compact search/replay service contract.
- [x] Draft `CompactRootBatchV1`, `CompactSearchResultV1`, and
  `CompactReplayChunkV1` as the explicit next architecture contract. This is
  the MiniZero/KataGo/Puffer-shaped lane, with stock LightZero objects as
  validation edges only. See
  `compact_search_replay_service_contract_20260522.md`.
- [x] Implement the first local compact service contract slice:
  `build_compact_root_batch_v1`,
  `validate_compact_search_result_v1`, and
  `build_compact_replay_chunk_v1_from_search_result`. Validation:
  `tests/test_compact_search_replay_contract.py` -> `4 passed`; compact
  replay/target focused suite -> `16 passed`.
- [x] Wire the profile-only direct CTree compact output into
  `CompactSearchResultV1` in the hybrid boundary path, then measure a closed
  compact service proof row. Result: the first materialized target-row proof was
  correctly falsified as too slow (`~52-54s` proof time over `61440` roots).
  The fixed `CompactReplayIndexRowsV1` proof avoids observation/next-observation
  materialization and drops proof time to `~0.18-0.19s` over the same roots.
  Sim8 H100 row improved `5634 -> 6193` steps/sec; sim16 stayed parity
  `4815 -> 4797`. This is a replay-writer fix, not a 3x search-service win.
- [x] Tighten `CompactReplayIndexRowsV1` after validation critique. Added stale
  policy-id/root-batch rejection, terminal final-observation mask rejection,
  final-reward assertions, non-prefix index-row coverage, and
  `materialize_compact_target_rows_from_index_rows_v1(...)` so the index-only
  hot path can be rebuilt into checked target rows at the sampler/validation
  edge. Focused ruff passed; focused compact-index tests passed.
- [x] Refresh the H100 no-model CTree falsifier after the compact replay fix:
  `ap-9hEH4WJk4kprHGTpcEiPte`. Current rows: CTree-list about
  `0.51M-0.94M` nodes/sec, CUDA payload about `0.58M-0.82M`, fake-flat about
  `16M-22.6M`. Read: the list ABI is a real target, but the trainer-facing
  10x still needs a search-service/replay-boundary architecture.
- [x] Decide and implement the smallest vendored CTree spike:
  `src/curvyzero/vendor/lightzero_ctree_a3` with `batch_backpropagate_flat_a3`
  and a `ctree-flat-a3` no-model benchmark backend. The first exact-parity
  failure was a bad gate: stock CTree reseeds and randomly breaks near-ties
  inside traverse. Added deterministic tie-breaking in the vendored module for
  parity checks only. Local deterministic vendored-list vs flat-A3 parity
  passes for roots=64, sim1/2/4/8, legal profiles all3 and mixed_2of3. After
  switching the flat path back to the fixed-A=3 `expand_a3(...)` overload,
  local roots=1024/sim16 no-model speedup is about `2.02x` all3 and `1.75x`
  mixed_2of3.
- [x] Finish H100 Modal no-model flat-A3 gate:
  run `ctree-list,ctree-flat-a3` at roots=1024/sim16/iters100/warmup10 with
  deterministic parity check. Final expand-A3 path passed exact parity and
  produced about `1.69x` all3 and `1.66x` mixed_2of3 on the H100 no-model
  denominator.
- [ ] Wire `ctree-flat-a3` into the profile-only `direct_ctree_gpu_latent`
  train/profile hook as an explicit opt-in tree backend, then run matched
  full-loop rows before recommending it to Coach.
- [x] Add the train-facing flat-A3 wiring skeleton:
  `collect_search_ctree_backend=flat_a3` now routes through the profile-only
  `direct_ctree_gpu_latent` hook and records compact proof fields. The profile
  grid builder emits both `--collect-search-backend` and
  `--collect-search-ctree-backend`; the summarizer rejects flat-A3 rows unless
  the runtime profiler actually observed `flat_a3` and the flat payload timer.
  Stock/live image coupling was reduced by moving the Cython build to isolated
  CPU40 optimizer images for `gpu-l4-t4-cpu40` and `gpu-h100-cpu40`.
- [x] Finish the first train-facing flat-A3 Modal smoke:
  `opt-flat-a3-smoke-20260522a` failed before search because Python imported
  the unbuilt `/root/curvyzero` package path. Fixed the import path preference.
  `opt-flat-a3-smoke-20260522b / smoke-flat-a3-sim2-c64-steps128` passed:
  `called_train_muzero=true`, one learner train call, zero fallbacks, `16384`
  output rows, and `search_backend_proof` observed `flat_a3`.
- [x] Run matched short full-loop A/B:
  `opt-flat-a3-ab-20260522a`, C64/sim16/3 learner, direct LightZero CTree vs
  flat-A3 CTree, same H100 profile settings. Result: direct LightZero CTree
  `516.55 steps/sec`; flat-A3 `509.69 steps/sec`. Flat-A3 is valid but did not
  win the full-loop denominator in this row. Keep it as a proof/falsifier, not
  as a Coach recommendation.
- [x] Stop dense Torch polishing after the fresh H100 search-boundary wave.
  Dense eager was competitive only at sim16 and lost badly at sim32; compile
  spike was worse. Keep those rows as evidence, not as the main plan.
- [x] Implement the closed compact search-service falsifier:
  `service_tax_probe` now pays real input/model/recurrent/fake-search/readback
  tax and can feed `CompactReplayIndexRowsV1` through the compact replay proof.
  Mock/service-tax compact arrays carry explicit `search_impl`, source,
  requested simulations, and actual search simulations.
- [x] Fix compact replay proof accounting. Warmup-seeded proofs are now
  reported separately from measured-search proofs, and ambiguous direct +
  compact array sources now fail instead of silently choosing one.
- [x] Finish the current closed compact-service H100 rerun:
  `opt-closed-service-direct-20260522b`,
  `opt-closed-service-mock-20260522b`, and
  `opt-closed-service-tax-20260522b`. Summarize from aggregate
  `compact.timings`, not from last-step summary fields.
- [x] Patch the hybrid profile runner compact summary so future stdout uses
  aggregate timings instead of last-step probe telemetry.
- [ ] Launch one higher-stability repeat only if needed: same direct/mock/tax
  shape, but longer measured/warmup rows. Current evidence is already enough
  to deprioritize wrapper-only work: service-tax is 2.10x at sim16 but only
  1.29x at sim32.
- [ ] Next P0 research/prototype sidecar: MCTX visual-root toy, scratch-only.
  Shape: `[B,2,4,64,64] -> R=B*2 -> tiny JAX CNN -> mctx.gumbel_muzero_policy`.
  This is not a trainer migration; it tests device-resident search viability.
- [x] Add the H100 `curvytron_visual_root` scratch route to
  `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`, including selected
  action legality checks and a simple search + steady H2D + policy-output D2H
  timing currency. Local ruff and py_compile pass.
- [x] Finish the live H100 MCTX visual-root rows launched on 2026-05-22:
  B64/sim16, B512/sim16, B512/sim32, and heavier B512/sim32. Use these only as
  a scratch search-architecture gate, not as Coach training advice. Guarded
  B512 rows reached `88.5k` fresh-boundary roots/sec at sim16 and
  `45.5k-47.5k` at sim32 with legal actions and compact action/visit/value
  samples present.
- [ ] Next P0 architecture spike: compact replay writer over arrays, not row
  objects, with a small parity adapter that can materialize stock target rows
  for validation only.
- [x] Move from synthetic visual-root MCTX to a real compact CurvyTron boundary
  sample. H100 B512/P2 with real renderer-backed HybridCompactBatch roots
  produced compact selected_action/visit_policy/root_value arrays, stepped the
  compact env once, and validated `CompactReplayIndexRowsV1`. Fresh-boundary
  throughput: sim16 `124,090` roots/sec, sim32 `51,454` roots/sec.
- [x] Run scale/pressure real compact MCTX wave. B512/B1024 and heavier toy
  visual rows all passed legal-action, compact search, and compact replay-index
  validation. Search-boundary throughput stayed high, but the next env
  step/replay-index edge cost hundreds of milliseconds, so the next Amdahl wall
  is closed-loop compact env/observation/replay synchronization.
- [ ] Next P0 search spike: current-model realism. Decide whether to port enough
  of the actual MuZero visual/recurrent model into the device-resident search
  path or prove a PyTorch-to-JAX bridge does not erase the margin. Keep this
  profile-only until learner/RND/replay edges are checked.
- [ ] Next P0 learner edge: compact replay/RND/learner sampler proof. Show that
  `CompactReplayIndexRowsV1` can feed target materialization, RND latest-frame
  input, and learner batches without scalar LightZero timestep objects entering
  the measured hot path.
- [x] Next P0 closed-loop denominator: implement/profile a repeated compact
  loop over multiple decisions:
  compact obs -> MCTX search -> selected joint actions -> compact env step ->
  compact replay/RND/target edge -> repeat. Report aggregate bucket timings so
  Amdahl points at the real next wall. Result: repeated H100 rows passed, and
  the wall moved to `env_step_sec`, not search.
- [x] Implement renderer-backed native actor-buffer support in the profile-only
  hybrid manager and run H100 closed-loop rows. Result: B512/sim16 improved
  `5.79k -> 6.82k` active roots/sec, B1024/sim16 improved
  `6.25k -> 8.92k`. Keep it, but do not treat it as the big win.
- [ ] Next P0 env-step split: break `env_step_sec` into actor physics/state
  update, renderer render, stack update, host/device movement, and compact-batch
  packaging on the repeated closed-loop denominator. Current native rows still
  spend about `74-81%` of closed-loop wall in `env_step_sec`.
- [ ] Next P0 observation architecture falsifier: test a device-resident or
  lower-copy compact observation path where the renderer/stack output stays in
  the same form that MCTX/search consumes. Keep this profile-only and validate
  against the current strict payload path.
- [x] Implement and test the first state-ownership canary:
  `borrow_single_actor_render_state` for profile-only native_actor_buffer,
  actor_count=1, renderer-backed refresh-on rows. Local ruff/py_compile and
  focused tests passed. Fresh H100 rows: host sim16 `26.6k -> 32.8k`, resident
  sim16 `32.9k -> 48.6k`, resident sim32 `24.0k -> 36.0k`. Keep the canary.
- [x] Tighten borrowed render-state to preserve the persistent renderer
  key-filter contract. Retest passed locally. Fresh H100 key-filtered rows:
  resident sim16 copied `34.1k`, borrowed `51.8k` (`1.52x`); no-refresh sim16
  ceiling `61.9k`; B2048 borrowed sim16 `52.3k`, so larger B did not materially
  improve per-root throughput.
- [ ] Next P0 after borrowed state: test resident-stack sync removal or sampled
  host mirror. The copied render-state wall is gone in the single-actor profile
  case; remaining hot leaves are renderer delta pack/H2D/update, observation
  stack ownership, public packaging, and search as sim count rises.
- [ ] Next P0 after latest Amdahl shift: split/attack the remaining refresh-on
  gap, but keep scope honest. At sim16, borrowed resident is already within
  about `1.2x` of the no-refresh ceiling; at sim32, search is roughly `30%` of
  measured wall. The next experiments should include one observation-handoff
  falsifier and one search/service-boundary falsifier, plus normal-death/RND
  validation.
- [ ] Add compact replay terminal/truncation adversarial test:
  `done=True`, `truncated=True`, `terminated=False`, and
  `final_reward_map != reward`; prove `CompactReplayIndexRowsV1` preserves the
  exact values.
- [ ] Add resident-claim telemetry guard. Any future lane labeled resident must
  expose whether full-frame D2H, `.cpu().numpy()`, or scalar timestep
  materialization happened inside the measured loop.
- [ ] Reconcile policy identity contract across compact bridge and LightZero
  compact batch consumers: either globally unique `policy_env_id` is accepted
  everywhere, or the docs/tests say the ID is local row-major only.
- [ ] Add scalar-vs-compact RND latest-frame property test for uint8 and float
  stacks, including terminal final-observation rows after autoreset.

## 2026-05-22 Current Optimizer Board

- [x] Add fine-grain renderer subtelemetry to the closed compact MCTX loop:
  production-to-compact, delta pack, host-to-device, persistent update,
  device-to-host, and stack-update buckets now flow into the repeated-loop
  aggregate.
- [x] Fix the live-prefix waste in persistent compact render state conversion.
  `_persistent_compact_state_from_production(...)` now trims trail slots to the
  live visual/body cursor prefix instead of copying the full
  `body_capacity=4096` allocation every step.
- [x] Validate the live-prefix trim locally. Focused ruff, py_compile, and
  compact boundary tests passed.
- [x] Run post-trim H100 closed-loop rows. Best matched B1024/sim16/loop16
  reached `15.26k` active roots/sec; longer B1024/sim16/loop32 settled around
  `12.38k`; B2048/sim16/loop16 reached `13.55k`.
- [x] Run actor-count falsifier. B1024/sim16/loop16 was fastest at
  `actor_count=1` (`16.42k`) and slower at `4` (`13.15k`) and `16` (`11.92k`).
  Do not pursue more in-process sharding as the next main lane.
- [ ] Next P0: split the remaining `env_step_sec` bucket inside the current
  repeated closed-loop denominator. Separate actor physics/runtime, public
  package/action-mask work, observation/stack update, production-to-compact,
  host/device transfers, and any Python packing.
- [ ] Next P0: prototype a profile-only resident compact observation loop where
  the latest stack remains in the layout/device placement consumed by MCTX. The
  validation edge should still materialize the strict host compact payload, but
  the measured loop should avoid GPU render -> host stack -> JAX device_put
  bounce when possible.
- [ ] Next P1: rerun B1024/sim32 with one more long repeat if the sim32 result
  becomes decision-relevant. Current read is enough: deeper search raises the
  search bucket to about `10%`, but env/observation remains the wall.

## 2026-05-23 Compact Search Service Board

- [x] Run same-denominator H100 compact search grid:
  direct CTree, dense Torch, service-tax, and mock at sim16/sim32.
  Results are under
  `artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-*-20260523b`.
- [x] Record the grid in `experiment_log.md`, `world_model.md`, and
  `next_phase_optimizer_synthesis_20260523.md`.
- [x] Decide dense Torch status. Result: keep research-only. It is 1.24x
  measured over direct at sim16, but 0.95x at sim32 and not LightZero CTree
  semantics.
- [ ] Next P0: make the compact search service path trainer-facing enough to
  run a matched stock-vs-candidate smoke. This means selected search actions
  must drive the next env step, compact replay rows must attach RND/latest
  frames and player identity, and learner-facing samples must match the trusted
  immediate path.
- [x] Add the smallest public `MuZeroGameBuffer.sample(...)` parity gate. The
  stock LightZero target-hook parity and public sampler edge are both now
  covered by opt-in local tests when `lzero` is installed.
- [ ] Next P0 speed slice: design one fixed-shape/device-resident search
  prototype behind `CompactSearchServiceV1`. Compare it against direct CTree
  and service-tax on the same B512/A16 compact replay denominator.
- [x] Add a comparable MCTX/JAX profile-only service row without promoting it
  as LightZero. `mctx_synthetic_benchmark.py` now emits
  `compact_search_service_profile` with backend
  `mctx_hybrid_compact_visual_search_service` and explicit `profile_only`,
  `not_lightzero_ctree`, and `not_train_muzero` flags. Validation:
  `tests/test_mctx_synthetic_benchmark_legality.py` plus compact replay
  contract -> `22 passed`.
- [x] Add the first small fixed-shape Torch helper module for the next backend
  lane. `src/curvyzero/training/compact_torch_search_service.py` is
  profile-only: it records compile eligibility, fixed-shape mask checks,
  explicit `not_lightzero_ctree` labels, and tiny select/backup helper tests.
  It is not wired into Coach training. It is now wired into the Modal
  profile-only launcher as `compact_torch_search_service`. Focused
  validation: ruff clean, `tests/test_compact_torch_search_service.py` ->
  `9 passed`.
- [x] Fold the compiled-backend "fast but wrong" critique into the smoke plan
  and current audit. Required future gates now include active-root order,
  legal-mask polarity, direct-CTree toy PUCT/backup parity, seeded root-noise
  behavior, recurrent latent/action shape, stale tensor reuse, compile-cache
  signatures, and timing with explicit sync/readback.
- [x] Add the compact Torch backend integration plan:
  `compact_torch_backend_integration_plan_20260523.md`. Current rule:
  direct CTree remains the semantic oracle; the Torch helper is only a staging
  surface; a candidate backend must pass closed-loop replay/RND/player/sampler
  gates before any H100 speed row matters.
- [x] Fold the integration-map sidecar into the plan. The real candidate should
  be `CompactTorchSearchServiceV1` in
  `src/curvyzero/training/compact_torch_search_service.py`; `service.run`
  should select active roots, run one model/search pass, then validate through
  `compact_search_result_v1_from_arrays(...)`. The profile module should only
  wire and measure it.
- [x] Implement the first local `CompactTorchSearchServiceV1` candidate. It is
  still profile-only, but it now owns one model/search pass behind the compact
  service boundary instead of relying on the Modal probe wrapper.
- [x] Add a closed-loop Torch service smoke. It applies the service-selected
  joint actions to the next hybrid env step and proves compact index rows
  materialize to the trusted immediate target rows. Focused validation:
  `1 passed`.
- [x] Wire explicit profile-only array-ceiling mode:
  `compact_torch_search_service`. This mode calls
  `CompactTorchSearchServiceV1.run(root_batch)` directly and stores compact
  arrays/results for the existing proof path; it does not call the old
  `run(observation, action_mask)` wrapper. Focused validation:
  `2 passed, 111 deselected`.
- [x] Add the grid/manifest builder allow-list for
  `compact_torch_search_service` compact replay rows. Focused validation:
  `3 passed, 19 deselected`.
- [x] Run remote H100 compact Torch service smokes through the durable manifest
  runner. Same B512/A16/sim16 denominator, compact replay proof on:
  `direct_ctree_gpu_latent` `4,966` steps/sec, `service_tax_probe` `5,853`,
  first `compact_torch_search_service` `5,140`. Timing-split rerun:
  `compact_torch_search_service` `5,575` steps/sec with `0.271s` initial model
  and `4.250s` tree/recurrent loop. No-noise pair:
  direct CTree `3,955`, compact Torch `5,704`. Read: service boundary works,
  but eager Torch tree is not a large win.
- [x] Add phase-honest compact Torch telemetry and runner summaries. Compact
  Torch mode now reports tensor prepare, initial inference, tree/recurrent loop,
  readback, root-noise weight, and compile status. Manifest builder now prints
  the durable runner command and labels `commands.sh` as debug/raw commands.
- [x] Focused validation after telemetry/tooling patch:
  ruff clean, compact-service focused tests `11 passed`, grid-builder focused
  tests `16 passed`, manifest runner tests `4 passed`, Modal boundary wiring
  test `1 passed`.

## 2026-05-23 Latest Local Validation

- [x] Python lint on the compact search, MCTX, boundary, grid, and hybrid proof
  files: passed.
- [x] `tests/test_compact_torch_search_service.py`
  `tests/test_mctx_synthetic_benchmark_legality.py`
  `tests/test_compact_search_replay_contract.py`
  `tests/test_source_state_batched_observation_boundary_profile.py`: `144 passed`.
- [x] `tests/test_source_state_hybrid_observation_profile.py`
  `tests/test_curvytron_hybrid_observation_profile_grid_builder.py`: `59 passed`.
- [x] Write the next smoke plan:
  `compact_service_next_smoke_plan_20260523.md`.
- [x] Add the first real-backend compact service closed-loop smoke. It uses the
  real direct CTree compact service, turns selected actions into the next
  hybrid env step, and compares compact deferred rows plus learner sample
  batches against the trusted immediate path. Focused validation: `1 passed`.
- [x] Run the minimal fast-but-wrong gate before candidate-backend work:
  `compact_service or direct_ctree or dense_torch or single_legal or
  biased_logits` across compact replay and boundary profile tests. Result:
  `14 passed, 112 deselected`.
