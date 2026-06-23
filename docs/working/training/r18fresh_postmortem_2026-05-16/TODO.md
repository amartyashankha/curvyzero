# TODO

## Active

- [x] Compute per-run matched-grid AUC, best-so-far, and retention for the r18fresh 18-run batch.
- [x] Record first-pass tournament analysis across checkpoints/runs, including head-to-head patterns and regressions.
- [x] Explain why the tournament website can show 500+ rows while trainer-facing selection is top-100 active rows.
- [x] Pin current raw active top-10 checkpoint refs for next-batch bootstrap.
- [x] Lock next-batch initial policy seed: every new Grid A/Grid B/canary
  trainer starts from the old overnight leaderboard rank-1 checkpoint, not
  raw top-10 or deduped top-10.
- [ ] Verify the pinned rank-1 checkpoint ref exists in the active v2 training
  volume immediately before launch.
- [x] Establish concise current naming conventions for Grid A, Grid B, canary
  rows, reward/noise/immortality tags, and slot recipe codes.
- [x] Add shared current naming helper and make the E2E canary builder use
  `cz26c-r001-out100-n0-imm0-b50r1` by default.
- [ ] Update manifest builders so new current-code rows use the `cz26*`
  naming convention instead of historical `tonight18`/`r18fresh` labels.
- [ ] Keep exact slot counts, percentages, seed refs, and refresh settings as
  structured manifest fields; do not encode all of them in run IDs.
- [x] Correct stale seeding-doc wording that claimed current CZ26 rating had
  advanced to `round-000033`; current CZ26 latest rating is still
  `round-000015` / `919`.
- [ ] Preserve raw/deduped top candidates as audit/opponent-seeding material
  only; do not use them as mixed initial-policy seeds for the next batch.
- [ ] Build a one-row-per-checkpoint lineage table joining checkpoint write,
  intake, tournament status, export generation, assignment SHA, trainer apply,
  provider load, survival, reward, and action mix.
- [x] Write the observability plan that defines the lineage table, game-batch card,
  export ledger, operator health readout, and next-batch proof gates.
- [ ] Implement best-effort `lineage_events.jsonl` emission at checkpoint,
  intake, rating, leaderboard publish, assignment write, pointer rewrite, and
  trainer-consumption boundary transitions.
- [x] Add the shared `lineage_events.jsonl` helper and unit tests.
- [ ] Add or run a compact tournament game-batch readout: game-batch id, raw rows,
  active rows, new checkpoints, pair count, completed/failed games, stable flag,
  max delta, latest pointer, and export age.
- [ ] Add or run an export/assignment ledger: export generation, selected refs,
  assignment shas, trainer runs expected, trainer runs applied, first iteration
  applied, and provider-load failures.
- [x] Add a deployed-function trainer proof path for CZ26 assignment
  consumption.
- [ ] Rerun trainer proof after the next assignment refresh and record whether
  generation-4/latest target application rises above `48/136`.
- [ ] Investigate the two repeated `kept_previous` JSONDecodeError trainer rows:
  `cz26a-r001...` and `cz26a-r020...`.
- [x] Make `trainer-proof` omit or cap per-run rows by default; summary output
  is enough for normal loop monitoring.
- [ ] Wait/recheck active internal game-batch artifact `round-000033`; if it
  publishes a rating beyond `919`, refresh assignments and rerun trainer proof.
- [ ] Add an operator-facing health strip or CLI equivalent for tournament id,
  rating id, current internal game-batch, active cap, active row count, newest
  checkpoint age, max checkpoint iteration, and latest trainer export
  generation.
- [x] Fix local `drain-if-ready` lease so a returned drain with a live rating
  call id still blocks duplicate drain attempts during the scheduling window.
- [ ] Compare tournament rank against eval survival per checkpoint/run.
- [x] Compare tournament rank against own reward and plus-outcome residual by checkpoint/run.
- [x] Run existing `analyze_curvytron_eval_curves.py` on a cleaned eval-status JSON snapshot.
- [x] Add a high-level trend summary across survival, own reward, tournament, and speed.
- [x] Add a slot-population deep dive with exact historical recipes and matched
  same-reward/noise comparisons.
- [x] Pull first eval snapshot for selected `curvy-ownlatest-staticmix-20260516b` control rows.
- [ ] Decide whether bonus reward needs a focused diagnostic or should stay secondary.
- [ ] Integrate in-flight parallel lane findings into the trend docs.
- [ ] Re-pull the own-latest control after it has more checkpoints.
- [x] Run fast local validation sweep from `TESTING_AND_GAPS.md`.
- [x] Run targeted full-loop component tests from `TESTING_AND_GAPS.md`.
- [ ] Decide whether/when to stop the 18-run batch.
- [ ] Keep `ownlatest` `20260516b` control running while analysis proceeds.
- [ ] Redesign the next batch based on measured failure modes and tournament evidence.
- [x] Implement explicit terminal outcome coefficient `alpha` so the next batch
  can test `0.0`, `0.33`, `0.67`, and `1.0` instead of only no-outcome versus
  current plus-outcome.
- [x] Ensure the new full plus-outcome reward cannot make total episode return
  negative; terminal outcome is now scaled by accumulated non-outcome training
  reward for that player, not raw source tick count.
- [ ] Present the 4/5/6-slot recipe scope options and their exact meanings to
  the user; do not silently choose scope.
- [ ] Implement leaderboard-opponent immortal probability as an explicit knob
  instead of relying on separate rank1-immortal slot names.
- [ ] Replace or extend `scripts/build_curvytron_tonight18_manifest.py` so it
  can emit the locked Grid A 96-row manifest and Grid B 40-row manifest. The
  current script still emits the historical 18-row matrix.
- [ ] Do not launch a "Grid A" or "Grid B" batch from the current
  `build_curvytron_tonight18_manifest.py` output; it is still the old
  r18fresh-style manifest shape.
- [ ] Track the two-grid design: Grid A broad 24-per-slot-recipe matrix; Grid B
  slot-focused grid with alpha around 0.5, bonus on, clean/p10 noise candidates,
  and explicit leaderboard-immortality settings.
- [ ] Track opponent refresh cadence as a side lane. Current accepted default
  is periodic refresh every `2000` learner iterations; checkpoint-boundary
  refresh is a later control idea, not the current launch requirement.
- [ ] Before manifest generation, resolve the arithmetic/name for the complex
  rank-spread recipe: either `blank20-wall20-r1_30-r2_20-r3_5-r4_5` or a
  different explicitly normalized split.
- [x] Lock opponent recipe count contract: author recipes as 64-slot bags,
  repeat 4x across 256 collector envs, shuffle deterministically, and keep
  learner `batch_size=64` unchanged.
- [ ] Manifest generation must record both intended percentages and final
  64-slot counts.
- [ ] Keep bonus reward on in next-grid reward recipes; treat any bonus-scale
  bump as a separate side question, not a silent default.
- [ ] Present Grid A/Grid B recommendations clearly before launch: Grid A broad
  robustness over 4 alpha values, 3 noise settings, 2 immortal settings, and
  selected slot recipes; Grid B slot-focused with alpha around 0.5, bonus on,
  clean/p10, immortal p0/p10, and wider slot families.
- [x] Record the latest grid refinement: Grid A is the 96-run broad mixed-recipe
  cross; Grid B is the 40-run slot-focused grid including pure blank,
  pure wall-avoidant, pure rank1, user-proposed coarse mixtures, and the anchor
  mixed recipes.
- [x] Lock next design scope: Grid A stays 96 runs; Grid B stays 40 runs.
  Ladder and pure/coarse recipes stay in Grid B for this design.
- [ ] Cleanup stale apps/artifacts after analysis is complete.
- [ ] Add a local synthetic full-loop test if code work resumes.
- [x] Write optimizer handoff for H100/L4 speed, current observation path, and
  float32 batched GPU boundary status.
- [x] Fold fresh L4/H100 current-surface profile into current launch defaults.
- [x] Make L4/C256/N256/batch64/sim8 the manifest-builder default.
- [x] Make the direct trainer Modal entrypoint defaults use the shared broad
  L4 lane instead of tiny smoke defaults.
- [x] Sweep high-risk docs for stale H100/batch32/current-lane claims and either
  mark them historical or point them at `CURRENT_LAUNCH_DEFAULTS.md`.
- [x] Close or reuse stale subagent lanes before spawning more; agent saturation
  is now part of the orchestration risk.
- [x] Create a compact lane/task board with priorities, owners, next actions,
  artifact predicates, and integration targets.

## Decision Log Seeds

- Stop/continue threshold for the 18-run batch: stop only after top candidates
  and matched-grid findings are preserved.
- Control health criteria for `ownlatest` `20260516b`: keep until its learning
  curve and assignment-consumption proof are documented separately.
- Next-batch axes to preserve: plus-outcome, sparse diagnostic, `blank20-wall5`
  recipe, clean/so10rep10 pairing if deliberately paired.
- Next-batch axes to remove or redesign: latest-only promotion and mixed
  initial-policy seeds. Raw/deduped top candidates are audit/opponent material;
  initial policy seeding is locked to the old-overnight rank-1 checkpoint.
- Next-batch launch gate: no large batch until the full-loop lineage readout can
  prove at least one fresh checkpoint went from trainer write to trainer
  opponent load on the same code path.
