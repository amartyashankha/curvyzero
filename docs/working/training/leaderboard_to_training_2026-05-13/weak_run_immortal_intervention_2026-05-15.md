# Weak-Run Immortal Intervention - 2026-05-15

Read-only historical audit. No code or live Modal control object was changed by
this pass.

Current restart decision: do **not** apply this weak-run-only intervention.
Instead, use launch-wide recipes where blank/hard-coded sentinel slots are
always immortal, checkpoint slots are mostly mortal with small explicit
immortal slices, and total immortal exposure is generally `20-30%`.

## Request

Historical request: for the five `v2real18` runs with weakest
survival/progress, raise combined blank-canvas/no-op plus immortal/invincible
opponent exposure aggressively while preserving some leaderboard-checkpoint
exposure. That request is now superseded for the restart lane.

## Existing Weak-Row Identification

Existing docs already identify the five rows in
`docs/working/training/leaderboard_to_training_2026-05-13/TRAINING_CONTROL.md`
under "Weak-Run Immortal Intervention".

Evidence recorded there: a fresh eval-summary read over the 18 submitted
`curvy-n18conn-*` run ids picked the five weakest by `latest - first` survival,
with `best - first` used as a secondary sanity check. The same request is also
tracked in `TODO.md`, `NOW.md`, `orchestration_2026-05-15.md`, and
`FULL_LOOP_PROOF.md`.

| Weak run label | Evidence: latest - first | Evidence: best - first | Current combined blank/immortal exposure |
| --- | ---: | ---: | ---: |
| `survbonusnoout-blank5-wall5-rank2_25-rank1_65-so10rep10` | `-60.0` | `+0.0` | `10%` |
| `survbonusnoout-blank5-wall5-rank2_25-rank1_65-clean` | `-12.0` | `+94.8` | `10%` |
| `survbonusout-blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35-so10rep10` | `-11.8` | `+23.3` | `15%` |
| `survbonusnoout-blank20-wall5-rank1_75-clean` | `-5.3` | `+14.0` | `25%` |
| `survbonusnoout-blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35-clean` | `-4.3` | `+60.0` | `15%` |

The row labels above map to these submitted live run ids in
`artifacts/local/curvytron_tonight18_manifests/curvy-v2real18-20260515a/curvy-v2real18-20260515a.json`:

| Row | Live run id | Current recipe |
| --- | --- | --- |
| `r008` | `curvy-v2real18-survbonusnoout-blank5-wall5-rank2_25-rank1_65-so10rep10-s780992354` | `blank5-wall5-rank2_25-rank1_65` |
| `r007` | `curvy-v2real18-survbonusnoout-blank5-wall5-rank2_25-rank1_65-clean-s621181572` | `blank5-wall5-rank2_25-rank1_65` |
| `r016` | `curvy-v2real18-survbonusout-blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35-so10rep-5f00e0213c` | `blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35` |
| `r011` | `curvy-v2real18-survbonusnoout-blank20-wall5-rank1_75-clean-s79346742` | `blank20-wall5-rank1_75` |
| `r009` | `curvy-v2real18-survbonusnoout-blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35-clean-s177727378` | `blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35` |

## Current Assignment Probabilities

The initial assignments are local artifacts under
`artifacts/local/curvytron_tonight18_manifests/curvy-v2real18-20260515a/assignments/`.
The later live refresh wrote the same recipes under
`artifacts/local/curvytron_tonight18_manifests/curvy-v2real18-refresh-r1-20260515a/assignments/`.

Current live control-volume pointers, read back from
`curvyzero-curvytron-control-v2`, point to the refreshed r1 assignments:

| Recipe | Current live assignment name | Current live assignment sha |
| --- | --- | --- |
| `blank5-wall5-rank2_25-rank1_65` | `curvy-v2real18-refresh-r1-20260515a-blank5-wall5-rank2_25-rank1_65` | `4db8fe399ce6d423f50cb30d8269c2d18bbf1b7025f8c40ffb8163972604fb5a` |
| `blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35` | `curvy-v2real18-refresh-r1-20260515a-blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35` | `9717c8b00d1e4a030026ca4188611f04d961b6d6a6f477f8758f11489d8f8d45` |
| `blank20-wall5-rank1_75` | `curvy-v2real18-refresh-r1-20260515a-blank20-wall5-rank1_75` | `e348714b7c960ea62423fd5a8cedaf20427778f764957a4142d5968bc2080f36` |

Current recipe probabilities:

| Recipe | Current slots |
| --- | --- |
| `blank5-wall5-rank2_25-rank1_65` | `blank 5%`, `wall_avoidant_immortal 5%`, `rank2 25%`, `rank1 65%` |
| `blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35` | `blank 10%`, `wall_avoidant_immortal 5%`, `rank4 10%`, `rank3 20%`, `rank2 20%`, `rank1 35%` |
| `blank20-wall5-rank1_75` | `blank 20%`, `wall_avoidant_immortal 5%`, `rank1 75%` |

## Was The Intervention Applied?

No evidence found that the requested five-row-only high-pressure
blank/immortal intervention was applied.

What did happen:

- Initial `curvy-v2real18-20260515a` launch wrote three control-volume
  assignments and three per-recipe refresh pointers before spawning rows.
- Later `curvy-v2real18-refresh-r1-20260515a/refresh-publish.json` shows
  `assignment_write_count=3`, `refresh_pointer_write_count=3`, and
  `publish_assignments_only=true`.
- Live control-volume reads confirm the three existing recipe pointers now point
  to the r1 assignments and shas above.

Why that is not this intervention:

- The r1 assignments keep the same `blank5`, `blank10`, and `blank20` recipes:
  combined blank/immortal exposure remains `10%`, `15%`, or `25%`.
- The r1 pointer update was per recipe and selected all 18 rows, not only the
  five weak rows.
- I found no local artifact or live pointer whose audit reason names the
  weak-run intervention, and no row-scoped pointer for only `r007`, `r008`,
  `r009`, `r011`, and `r016`.

No Modal Dict mutation was found for this intervention. The active control
surface here is the control-volume `refresh_pointer.json` files, not a live
Dict recipe.

## Safe Writer Path

The safe writer is the existing assignment artifact writer:

- Modal function:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py::lightzero_curvytron_write_opponent_assignment_artifacts`
- Underlying function:
  `_write_opponent_assignment_artifacts`
- Existing grouped submitter wrapper:
  `scripts/submit_curvytron_survivaldiag_manifest.py --publish-assignments-only`
- Existing command helper used by promotion:
  `scripts/promote_curvytron_rating_round.py::build_write_assignment_command`
  plus `write_refresh_pointer_command`

For a live intervention, assignment JSON must be validated with
`parse_opponent_assignment_snapshot`, hashed with
`canonical_assignment_json_sha256`, written to `target_volume="control"`, and
then pointed to by a `curvyzero_opponent_assignment_refresh_pointer/v0` JSON
containing both `assignment_ref` and `assignment_sha256`.

Do not use `scripts/materialize_curvytron_leaderboard_assignment.py` as the
primary writer for this specific intervention. It is the leaderboard
materializer for normal `stable_slots_v1`/`top_slots_v0` assignments, not a
row-scoped custom high-pressure diagnostic recipe writer for already-running
rows.

## Minimal Safe Apply Path

Do not overwrite the current three live recipe pointers if the requirement is
"only the weak rows." Existing rows share recipe pointers:

- `blank5-wall5-rank2_25-rank1_65` is used by 6 rows, not only weak `r007` and
  `r008`.
- `blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35` is used by 6 rows, not
  only weak `r009` and `r016`.
- `blank20-wall5-rank1_75` is used by 6 rows, not only weak `r011`.

Therefore a shared-pointer mutation would violate the user constraint and
change non-weak rows.

Exact minimal safe path if we decide to apply:

1. Freeze the weak row ids and current assignment shas:
   `r007/r008 -> 4db8fe399ce6d423f50cb30d8269c2d18bbf1b7025f8c40ffb8163972604fb5a`,
   `r009/r016 -> 9717c8b00d1e4a030026ca4188611f04d961b6d6a6f477f8758f11489d8f8d45`,
   `r011 -> e348714b7c960ea62423fd5a8cedaf20427778f764957a4142d5968bc2080f36`.
2. Create new immutable assignment JSONs with unique assignment ids per weak
   row or per weak-row group, not per old shared recipe. This is historical
   guidance only; current restart recipes should use launch-wide `20-30%`
   total immortal exposure instead of a row-scoped high-pressure intervention.
3. Write those assignments to the control Volume with
   `lightzero_curvytron_write_opponent_assignment_artifacts`, using a new
   assignment bank run id such as
   `curvy-v2real18-weakimmortal-20260515a-assignments`.
4. Do not reuse the existing shared refresh pointer paths. Either:
   - relaunch/resume only the five weak rows with row-specific
     `opponent_assignment_ref` and row-specific
     `opponent_assignment_refresh_ref`; or
   - first add/deploy a row-scoped control-pointer mechanism, then point only
     those five row pointers to the new assignment refs.
5. Verify before trainer uptake:
   - each new assignment parses;
   - each new assignment sha matches the pointer sha;
   - no non-weak row references the new pointer path;
   - the old three recipe pointers are unchanged unless intentionally replacing
     all rows using that recipe.
6. Verify after trainer uptake:
   - each affected row writes `opponent_assignment_refresh_events.jsonl` with
     `decision=applied` and the new sha;
   - later `env_steps.jsonl` rows carry the same
     `opponent_assignment_sha256`;
   - `opponent_provider_load_ok=true` for frozen checkpoint slots;
   - non-weak rows continue to report their old assignment shas.

If only the currently deployed row config is available, the minimal safe action
is to wait or relaunch the five weak rows. The current shared pointers cannot
target exactly five rows.

## Risks

- Shared pointer blast radius: overwriting any current recipe pointer affects
  all rows using that recipe, not just the weak rows.
- Stale leaderboard refs: the refreshed assignments point to a later r1
  assignment bank. Any new custom assignment must preserve exact immutable
  checkpoint refs and avoid mutable `latest`/`ckpt_best` refs.
- Trainer uptake proof: writing a control-volume pointer is not enough. The
  trainer must emit an applied refresh event and later env telemetry with the
  new sha.
- Perspective/render fix risk: current docs flag tournament observation
  parity/perspective issues. Applying the intervention before the perspective
  fix can produce hard-to-interpret learning signal.
- Diagnostic semantics: public `opponent_immortal=true` and blank-canvas/no-op
  are intentionally dirty training controls, not source-faithful leaderboard
  players. Env-bound `opponent_death_mode=immortal` is derived runtime plumbing.

## Recommendation

Wait for the perspective/render parity fix and the row-scoped pointer shape
before applying to the live batch.

Apply now only if the operator accepts a relaunch/resume of exactly the five
weak rows with new row-specific assignment and refresh refs. Do not mutate the
three existing shared recipe pointers for this intervention.
