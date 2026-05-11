# CurvyTron Switch Readiness - 2026-05-10

Short answer: CurvyTron adapter/trainer plumbing can run in parallel now that
normal LightZero Atari Pong has repeated late-checkpoint stock-survival signal.
Do not wait for perfect Pong, but do not move any quality claim before the
survival evidence is clean.

## Current Read

- Main proof lane is normal LightZero Atari Pong.
- Old seed `1` and seed `3` later checkpoints improved versus same-run
  `iteration_0`.
- Later normal runs now include multiple stock-survival lifts. Read those as
  control-lane health, not as solved Pong.
- Many early `1000`-`3000` rows are flat and should not be treated as failure.
- For the current run read, use
  `docs/working/coach_north_star_2026-05-10.md`; this note only preserves the
  switch gate logic.
- CurvyTron is the broader goal. Repo-native CurvyTron `[B,P]` remains a
  separate architecture probe; LightZero stays the serious replication/control
  lane.

## Enough To Start CurvyTron Adapter Work In Parallel

Practical threshold:

- At least two normal Pong runs, from independent seeds or a seed plus a repeat
  run, show late-checkpoint improvement over same-run `iteration_0`.
- The improvement appears in stock evaluator fields, not only manual rollout.
- Prefer checkpoints at `8000+`, `10000+`, or later, not only `1000`-`3000`.
- Stock return and score are secondary. They should be reported, but the gate
  is stock evaluator steps survived versus same-run `iteration_0`.
- Each claim has a clear artifact path, eval id, checkpoint id, seed, and eval
  seed.
- The result does not depend on custom dummy Pong or shaped reward.

This is enough to begin CurvyTron adapter work in parallel because it says the
normal visual LightZero control lane can learn something real sometimes.

## Still Blocks Strong Claims

- Survival durability is not fully proven yet. We need late checkpoints that
  keep beating baseline across runs and eval starts, not one lucky row.
- We still need broader repeat evidence if one or two runs look strong while
  others stay flat or fall back.
- Early flat rows make the timing question unresolved: a bad `1000` or `3000`
  row may simply be too early.
- Manual survival and stock evaluator telemetry can disagree, so stock steps,
  stock episode length, stock reward counts, stock action histogram, and return
  must be clearly separated, with stock steps first.
- Each useful row needs a clean saved artifact: manifest, summary TSV, run id,
  attempt id, checkpoint path, eval settings, seed, eval seed, max steps, and
  whether `update_per_collect=None` was preserved.
- Keep the compact latest-checkpoint table in the North Star current before
  citing this gate.

## CurvyTron Prep That Can Start Now

- Write the visual environment contract: observation shape, frame stack, dtype,
  reward, done, truncation, info fields, and render source.
- Treat the first CurvyTron reward as survival time: give reward for staying
  alive longer, and end the episode when the controlled player dies. Do not
  start with a separate win condition unless later game design requires it.
- Make reset and seed behavior explicit and testable.
- Add a discrete ego action wrapper for the controlled player.
- Decide whether `action_mask` is needed. If legal moves can change, include it.
- Set `to_play=-1` unless the LightZero path needs a different fixed value.
- Log full joint actions: ego action, opponent actions, scripted/random policy
  source, and player ids.
- Log episode metadata: seed, map/settings, player count, max steps, terminal
  reason, reward totals, survival time, and winner if available.
- Add comparable timing/profile metadata so CurvyTron rows can be compared with
  Pong rows: hardware label, worker counts, env count, frame stack, image size,
  checkpoint cadence, train step cap, and eval cap.
- Keep repo-native CurvyTron `[B,P]` architecture probing separate from the
  LightZero visual adapter.

## Do Not Move Over Yet

- Do not use shaped reward as proof that the stock lane works.
- Do not use custom dummy Pong assumptions as CurvyTron design truth.
- Do not use manual rollout as stock proof.
- Do not copy Pong-specific action-map assumptions into CurvyTron without a
  written action contract.
- Do not copy Pong's score/win reward into CurvyTron by default. The simple
  first objective is survival length.
- Do not declare CurvyTron training ready until reset/seed, observation, action,
  reward, done/truncation, logging, and eval metadata are pinned down.

## Next Actions

- Keep the compact comparison table current from the North Star: baseline,
  latest useful late checkpoints, and status labels such as `stock signal`,
  `flat`, `regression`, or `missing`.
- Check whether stock survival improves across more than one late checkpoint
  and multiple eval starts. Report return/score after survival, not as the gate.
- Save artifact paths beside every claim.
- Continue CurvyTron visual survival plumbing and checkpointed evals in
  parallel, but keep the quality claim separate from the Pong control claim.
- Prefer frozen-checkpoint opponent profiling over more long fixed-straight
  opponent runs.
- Keep shaped reward and custom dummy Pong in side-lane notes only.
