# Environment Current Queue - 2026-05-13

Status: working queue, not a completion claim.
Owner surface: Environment docs/process.

Use this as the short operating queue for the current environment
reconstruction push. Deeper evidence stays in focused notes; this page only
records how the work is sequenced and what proof is needed next.

North star: faithful multiplayer CurvyTron environment first, then
speed/training integration. LightZero/training plumbing is only a guarded
downstream interface over the reconstructed environment, not the center of the
work. Do not divert into speed rabbit holes unless the measurement directly
protects source-fidelity reconstruction.

Guardrail: project-only helpers such as `profile_no_death`, no-death/profile
modes, optimizer modes, and training-helper modes are valid project features.
Preserve them through trainer/replay/target surfaces with explicit metadata,
but do not cite those rows as original CurvyTron/source-fidelity behavior.

## Working Rhythm

- Main thread: plan, delegate, orchestrate, synthesize, and decide. Keep the
  route to faithful multiplayer `VectorMultiplayerEnv` behavior clear and
  reorder the queue when evidence changes risk.
- Subagents: bounded source reads, focused tests, narrow docs updates, and
  small experiments. Each handoff should name the question, files or commands,
  finding, evidence, remaining risk, and next step.
- Docs: working memory, current queue, evidence, conclusions, and gaps. Do not
  turn docs into broad taxonomies or pass-count dashboards.

Active reorientation threads:

- Completed full-game multiplayer gap audit: lifecycle, presence/leave, scoring,
  match-end, replay/final observations, bonus stack/death stress, and 3P/4P
  breadth.
- Completed controls fidelity audit: source-frame control delivery,
  held/released inputs, terminal padding, touch/gamepad input, and
  browser/transport semantics.
- Completed renderer/fast-path boundary audit: keep source-state/native
  fidelity claims separate from optimized or approximate render paths.
- Docs/orchestration rhythm update: docs stay working memory; main thread
  plans/delegates/orchestrates; subagents handle bounded audits, tests, and
  docs.

## Current Snapshot

Done:

- `SourceStateMultiplayerTrainerSurface` emits per-seat source-state visual
  stacks, live-seat policy rows, masks, survival-plus-bonus rewards, terminal
  visual final observations, and honest render/source metadata.
- `SourceStateMultiplayerTrainerReplayRecorder` stores copied in-memory replay
  arrays over time, including terminal visual final observations and variable
  live-policy rows.
- `SourceStateMultiplayerTargetRowsV0` now builds repo-owned target rows from
  trainer replay arrays plus policy-row records. Focused tests cover
  reset-to-step alignment, terminal final observations/rewards, P=4 row
  mapping, `to_play=-1`, copied arrays, invalid policy/action rejection, and
  no-death/death-immunity metadata preservation.
- `SourceStateMultiplayerSampleBatchV0` now builds deterministic sample
  batches on top of target rows through
  `build_source_state_multiplayer_sample_batch_v0`. Focused target-row/sample
  batch tests reported `12 passed` locally per worker.
- Fake/injected native `GameSegment` mapping from
  `SourceStateMultiplayerTargetRowsV0` is implemented in
  `src/curvyzero/training/multiplayer_source_state_native_bridge.py`, with
  focused tests in `tests/test_multiplayer_source_state_native_bridge.py`. It
  is injection-only, does not import LightZero, preserves project-only mode
  metadata, and keeps native/LightZero/training/buffer/learner claims false.
- The separate opt-in real-LightZero construction helper exists as
  construction smoke only. It does not prove `MuZeroGameBuffer` sampled-target
  parity, learner updates, evaluation quality, or true multiplayer self-play.
- Focused controls, product-route, bonus-default, and hit-owner proofs are
  current enough to support the training surface, with gaps below.

Current focus:

- Environment Reconstruction is back on multiplayer fidelity, controls tail
  proof, renderer/fast-path boundaries, and docs/orchestration rhythm. Treat
  the completed audits as inputs to the next implementation/test pass.

Next:

- Convert the completed full-game multiplayer, controls, and renderer boundary
  audits into focused tests and fixes against `VectorMultiplayerEnv` before
  treating any speed/training task as primary.
- Treat Hume's opt-in real-LightZero construction helper as done but
  construction-smoke only. Real `MuZeroGameBuffer` sampled reward, value,
  policy, action, mask, observation, and `to_play` parity remains unproven.
  Keep native integration, learner updates, eval quality, and true multiplayer
  self-play false-claimed until those surfaces are actually tested.

## Remaining Proof Queue

1. Target-row adapter, deterministic sample batches, and fake/injected native
   `GameSegment` mapping for source-state multiplayer training. Done for the
   repo-owned v0 row, sample-batch, and injection-only bridge contracts.
   Remaining downstream follow-up is real LightZero buffer sampled parity.
2. Controls source-frame fidelity.
   Latest proof covers original JS keyboard reduction/server move delivery, 2P
   public action-to-native-control mapping, held controls across
   `decision_source_frames`, release-to-straight, invalid/live action errors,
   inactive noops, terminal-padding noops under `decision_source_frames`, 3P/4P
   public one-frame trajectory parity, 4P held-control parity, and terminal
   early-stop through both direct runtime and the LightZero-facing wrapper.
   Remaining controls work is touch/gamepad input, real transport/browser
   integration if needed, and wider wrapper/replay propagation.
3. End-to-end 2P product route.
   Latest direct `VectorMultiplayerEnv` proof covers raw RGB -> gray64, seeded
   `BonusGameClear`, stale trail/body clear, live ticks, terminal wall death,
   rewards, final observation masks, and metadata replay. Latest
   LightZero-facing wrapper proof covers scalar joint-action decoding, raw RGB
   -> gray64 stack, held source frames, terminal final observation, rewards,
   masks, and native sidecars. New `SourceStateMultiplayerTrainerSurface`
   proof covers per-seat source-state visual stacks, live-seat policy-row
   mapping, survival-plus-bonus rewards, render-mode guards, and terminal
   visual final observations over `VectorMultiplayerEnv`. New in-memory replay
   proof stores those trainer arrays over time. New target-row and sample-batch
   proof validates replay -> target transition rows -> deterministic sample
   batches. New fake/injected native bridge proof maps those rows to injected
   `GameSegment`-like objects without importing LightZero or changing native
   claims. New public `VectorMultiplayerEnv` lifecycle proof covers P=3/P=4
   match-mode rows where one row starts the next round and one row ends the
   match after warmdown. New trainer/replay lifecycle proof carries that shape
   through `SourceStateMultiplayerTrainerSurface.advance_warmdown(...)` and
   `SourceStateMultiplayerTrainerReplayRecorder`: final visual rows, final
   rewards, masks, copied arrays, and variable live-policy rows are preserved.
   Remaining work is durable artifact plumbing and downstream real LightZero
   buffer sampled parity.
4. Bonus probability and source defaults.
   Latest source/default fix treats all non-`BonusGameClear` source-default
   bonuses as probability `1`, matching the JS `BaseBonus.getProbability`
   behavior. Latest public/runtime proof pins corrected default boundary
   draws, RNG labels/cursors, next-delay scheduling, spawned position, and the
   full 12-item source-default set. New focused 2P `BonusSelfFast`
   stack/death proof catches three speed bonuses, kills the boosted player on
   the normal wall, and proves death clears the active stack while later source
   timeout callbacks only emit inert stack-removal events. New same-step timer
   proof pins the narrower ordering where one `BonusSelfFast` expiry drains
   before the wall-death update: speed is restored to `16`, movement uses that
   speed, p0 dies on the normal wall, p1 scores, and the stack stays empty.
   New trainer/replay proof carries both terminal cases through
   `SourceStateMultiplayerTrainerSurface` and
   `SourceStateMultiplayerTrainerReplayRecorder`: terminal visual final
   observation rows, final reward maps, death metadata, winner/loser facts,
   step counters, and compact bonus audit metadata survive into replay
   records. This is source-runner/public-vector plus trainer/replay
   preservation, not browser event-loop or pixel proof. New 4P target-filter
   proof pins the source/public rule that enemy bonuses affect only other alive
   avatars, all-avatar bonuses affect only alive avatars, absent seats are not
   targeted because they are not alive, and game bonuses still apply to global
   game state. New source-backed 4P terminal proof covers `BonusEnemySlow`: a
   JS oracle fixture and public vector mirror now pin p0 catching the enemy
   bonus, p1/p2/p3 receiving slowed stack entries, those targets wall-dying
   before expiry, death clearing their stack rows without restoring dead-player
   speed, and p0 winning the round. The matching trainer/replay proof preserves
   final visual rows, final reward rows, death order, winner/loser facts, step
   counters, and compact bonus metadata.
   Remaining work is broader retry/RNG stress and other stack/death cases
   without narrowing the default bonus set.
5. Hit-owner ordering.
   Latest runtime fix scans source-compatible body-hit corner islands and newest
   bodies first. Latest stress tests cover 4P newest-owner overlap, 4P corner
   island order, 3P own-body latency, and 4P two-victim hit-owner metadata.
   New focused propagation proof carries a 3P terminal body-hit case and a 4P
   nonterminal two-victim case through public env, trainer surface, replay
   records, and debug die events. New raw JS oracle fixtures now pin the exact
   3P terminal and 4P nonterminal stress shapes, and public
   `VectorMultiplayerEnv` mirrors them from fixture-seeded state. Remaining
   work is broader collision edges beyond those two promoted shapes.
6. Wider multiplayer.
   Public 3P/4P lifecycle now has a focused mixed-row match-mode proof for
   reset/warmup, round win, warmdown, next-round, match-end, masks, rewards,
   and public final rows. Trainer/replay now has the matching focused proof
   for the same shape, including legal/live policy rows and terminal visual
   final rows. Public env plus trainer/replay now also have a focused P=3/P=4
   presence/leave proof for mixed active-row and staged-warmdown removal:
   present/alive masks, absent action slots, warmdown next-round carryover,
   trainer live-policy rows, and replay array storage. Focused source-backed
   public proofs also exist for `source_lifecycle_remove_avatar_to_single_present_3p.json`
   and `source_lifecycle_remove_avatar_during_warmdown_3p.json`. Remaining
   work is broader leave variants, more 3P/4P bonus stack/death combinations
   beyond the focused 4P `BonusEnemySlow` terminal replay proof, and later
   browser/canvas pixel parity.
7. Native LightZero bridge after environment reconstruction and target
   rows/sample batches.
   Fake/injected native `GameSegment` mapping from
   `SourceStateMultiplayerTargetRowsV0` is done and injection-only. Hume's
   separate opt-in real-LightZero construction helper is also done, but it is
   still construction-smoke only. Real `MuZeroGameBuffer` sampled reward, value,
   policy, action, mask, observation, and `to_play` parity remains unproven.
   This remains downstream interface work, not the main reconstruction job.

## Queue Discipline

- Keep at most one main active implementation lane and one docs cleanup lane.
- Call a gap closed only for the surface actually tested: source truth,
  product runtime, trainer/replay, or renderer.
- Record blocked items with the missing proof, not just the symptom.
- Keep old toy/debug paths as historical smoke evidence unless a focused note
  promotes one for a narrow proof surface.
- Preserve no-death/profile/training-helper additions as project features, but
  do not cite them as source-fidelity proof.
- Keep optimized and approximate render paths explicitly labeled and separate
  from engine rule fidelity.
