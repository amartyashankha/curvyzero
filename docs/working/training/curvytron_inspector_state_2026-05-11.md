# CurvyTron Inspector State - 2026-05-11

Purpose: current working memory for the CurvyTron inspector.

## Mission

Build observability for CurvyTron training. The first useful tool should read an
eval manifest and optional train summary, then write a compact report.

The report should be blunt. It should say what the artifacts prove, what they
suggest, and what they cannot explain.

## Current Known Inputs

Local eval manifests:

- `artifacts/local/curvytron-eval-manifests/s100_fixed8_manifest.json`
- `artifacts/local/curvytron-eval-manifests/s100_fixed64_manifest.json`
- `artifacts/local/curvytron-eval-manifests/s101_fixed64_manifest.json`
- `artifacts/local/curvytron-eval-manifests/s102_fixed64_manifest.json`
- `artifacts/local/curvytron-eval-manifests/s93_fixed16_manifest.json`
- `artifacts/local/curvytron-eval-manifests/s93_matched16_manifest.json`

Local train summaries:

- `artifacts/local/curvytron-eval-manifests/s100_train_summary.json`
- `artifacts/local/curvytron-eval-manifests/s93_train_summary.json`

## Current Facts

- The local eval manifests have checkpoint rows with survival steps, cap, action
  histograms, terminal reason, strict-load status, and artifact refs.
- The local aggregate rows sometimes use `ok`, not `ok_count`.
- The local eval terminal reasons are mostly `survivor_win` and sometimes
  `unknown`. That is not a real death cause.
- The local train summaries can reveal whether a run was fixed-opponent or
  frozen-opponent, and whether it is only debug fidelity.
- Fixed-opponent evals are useful, but they are not proof of self-play learning.

## Missing

- Death cause: wall, own trail, opponent trail, draw, or timeout.
- Player that died and what it hit.
- Short traces near death for representative bad episodes.
- Baseline comparisons at the exact same timestep and settings.
- A local report that joins manifest data and train summary data in one place.

## Implemented

The first local inspector exists at
`src/curvyzero/training/curvytron_inspector.py`.

Inputs:

- eval manifest path
- optional train summary path

Outputs:

- `report.json`
- `report.md`

Current behavior:

- reads survival by checkpoint
- uses paired first/latest seed deltas when possible
- keeps `survival_read` separate from `learning_claim`
- blocks learning claims when evidence is narrow or missing key fields
- writes a coach-facing `coach_next_move`
- warns on missing death cause, action collapse, non-strict loads, mixed caps,
  fixed/frozen opponent limitations, missing train summary, and missing baseline
  panel

## Current Critique

The first report must avoid fake learning stories.

Acceptance checks now needed:

- same seed set for every checkpoint
- same cap and eval settings
- strict checkpoint loads
- paired first/latest seed deltas when table rows are available
- action collapse warning blocks a strong learning read
- `survivor_win` is treated as a round outcome, not a death cause
- fixed and frozen opponent evals are labeled plainly
- missing baselines keep the learning claim incomplete

Subagents currently own: none.

## Real Report Results

Generated first reports under `artifacts/local/curvytron-inspector/`.

Current reads:

- `s100_fixed8`: survival got worse versus the first checkpoint. It had action
  collapse warnings, fixed-opponent caveats, debug-only training summary, and no
  death cause. Coach next move: stop or avoid scaling this run; inspect death
  causes before more training.
- `s100_fixed64`: survival was flat versus the first checkpoint. It had action
  collapse warnings, fixed-opponent caveats, debug-only training summary, and no
  death cause. Coach next move: do not scale from this run; inspect shortest
  deaths and compare baselines.
- `s93_matched16`: survival got worse versus the first checkpoint. It used a
  frozen opponent, had action-collapse warnings, and no death cause. Coach next
  move: stop or avoid scaling this run; inspect death causes before more
  training.
- `s101_fixed64`: survival improved in the fixed-opponent panel, but there was
  no train summary, action-collapse warning, and no death cause. This should be
  treated as a narrow survival read, not a learning claim. Coach next move: keep
  it as narrow survival evidence and clear blockers.
- `s102_fixed64`: survival was flat versus the first checkpoint. It had
  fixed-opponent caveats, no train summary, action-collapse warnings, and no
  death cause. Coach next move: do not scale from this run; inspect shortest
  deaths and compare baselines.
- `s93_fixed16`: survival got worse versus the first checkpoint. It had
  frozen-opponent caveats, debug-only training summary, action-collapse warnings,
  and no death cause. Coach next move: stop or avoid scaling this run; inspect
  death causes before more training.

Coach-level insight from first-pass manifest reads:

- Manifest rows alone cannot answer wall vs own trail vs opponent trail. They
  mostly say `survivor_win`, which is only a round outcome.
- Nested per-episode artifacts for `s101_fixed64` and `s102_fixed64` do store
  full ego action sequences. Those episodes can be replayed locally.
- None is clean self-play learning evidence.
- One fixed-opponent panel (`s101_fixed64`) shows a narrow survival lift, but it
  is blocked by action collapse, missing train summary, missing baseline panel,
  and missing death cause in the manifest-level report.
- The other panels are flat or worse.
- The practical next move is to use replay/fresh instrumentation to explain the
  deaths, then add baseline comparison before scaling.

## Consumer Angle

The main consumer is a coach or training operator.

They need the inspector report to answer:

- can I trust this curve?
- did survival move?
- is the result blocked by action collapse, missing death cause, or narrow
  opponent setup?
- should I keep training, rerun inspection, compare baselines, or add
  instrumentation?

The report should therefore include one plain `coach_next_move` near the top.
The coach should not have to assemble the next action from separate warning
sections.

This is now implemented in the first local inspector report.

Current claim rule:

- `survival_read` describes only the comparable survival panel.
- `learning_claim` is blocked by missing baselines, missing death cause, narrow
  opponent setup, missing train summary, train-summary flags, action collapse, or
  comparability problems.
- `coach_next_move` turns the result into a plain training decision.

## Next Observability Gap

Death cause was the next highest-value improvement.

Current status:

- Old local eval manifests still cannot answer why episodes died. They only have
  round outcomes like `survivor_win`.
- New runtime deaths now record `death_cause` and `death_hit_owner` when the
  state supplies those arrays.
- Wall deaths record `wall`.
- Body deaths record `own_trail`, `opponent_trail`, or `body_unknown`.
- Public multiplayer env info exposes `death_cause`, `death_cause_name`, and
  `death_hit_owner`.
- Trainer env info now exposes the same death-cause fields.
- Visual survival eval now hoists those fields into episode and manifest rows.
- The inspector now prefers `death_cause_name` and shows death-cause counts in
  the Markdown survival table.

Live probe result after the patch:

```text
death_player: [[1, -1]]
death_cause_name: [["wall", "none"]]
death_hit_owner: [[-1, -1]]
terminal_reason_name: ["round_survivor_win"]
```

That means future reports can distinguish "round ended with a survivor" from
"player 1 died by hitting the wall."

## Old Artifact Replay Result

The first claim that old artifacts could not answer death cause was too broad.
Correct split:

- A manifest row alone is too thin.
- A nested per-episode JSON with ordered `episode.actions` can be replayed when
  the opponent is fixed-straight.
- The replay must compare its final `trace_hash` with the stored final telemetry
  hash before we trust it.

New local helper:

```bash
uv run python -m curvyzero.training.curvytron_visual_survival_replay_inspector \
  artifacts/local/curvytron-eval-manifests/s102_fixed64/s102_fixed64/iteration_0_steps1024_seed1297473639/curvytron_visual_survival_eval_iteration_0_steps1024_seed1297473639_20260510T203821Z.json
```

The main inspector now also does this automatically when:

- `--eval-manifest` points at a local manifest, and
- the manifest rows have `artifact_ref` values, and
- the matching nested files already exist locally.

It does not fetch Modal files or rerun checkpoints.

Actual answer for the shortest old `s102` examples:

- `iteration_0`, `iteration_256`, and `iteration_384` on seed `1297473639`
  replay exactly.
- Final trace hash: `30c77a4dedae3f35`, matched in all three.
- `player_0` died.
- Cause: wall.
- Action pattern: left on all 33 decisions.
- Opponent: fixed straight.

Aggregate replay over the local nested fixed-opponent `s101` and `s102`
episode artifacts:

- 759 replayed.
- 759 final trace hashes matched.
- 759 deaths were wall deaths.
- First death player: `player_1` in 548 episodes, `player_0` in 211 episodes.
- Shortest episodes were `player_0` wall deaths after 33 to 41 decisions,
  usually with all-left or left-heavy action collapse.

Local `s93` nested files are present but currently invalid/empty JSON in this
checkout. The new replay helper reports this as `invalid_json` instead of
crashing.

Plain read: many old short episodes were not mysterious. The learned ego policy
often kept turning left, curved into the wall, and lost to a fixed-straight
opponent. The old manifest hid that because it did not copy death fields into
the table.

Updated report outputs:

- `artifacts/local/curvytron-inspector/curvytron-visual-survival-player-aware-fixed-s102-sim8-131072/s102_fixed64/report.md`
  now shows `wall:381`, with 381/381 replay hashes matched.
- `artifacts/local/curvytron-inspector/curvytron-visual-survival-player-aware-fixed-s101-262144/s101_fixed64/report.md`
  now shows `wall:378`, with 378/378 replay hashes matched.

Main risks:

- Body-hit owner order may differ from source semantics if multiple bodies
  overlap.
- Multiplayer fallback death appends must keep player and cause arrays aligned.
- Non-runtime lifecycle paths that synthesize `death_player` directly should not
  invent a death cause. They currently leave cause as `none` or fallback
  `body_unknown`.
