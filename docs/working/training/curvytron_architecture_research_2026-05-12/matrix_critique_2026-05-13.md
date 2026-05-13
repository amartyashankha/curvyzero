# Matrix Critique - 2026-05-13

Purpose: critique the next CurvyTron survival diagnostic matrix against the
raw user instructions recovered from local Codex transcript state.

## Raw Sources Used

- `~/.codex/history.jsonl`, session `019e09f3-29f3-70d2-bcf6-434ef853f539`,
  records `ts=1778641930`, `1778643468`, `1778645656`, and duplicate
  `1778645673`.
- `~/.codex/sessions/2026/05/08/rollout-2026-05-08T19-36-31-019e09f3-29f3-70d2-bcf6-434ef853f539.jsonl`,
  the direct parent rollout for that session. Matching user-message records
  include the repeated-seed/opponent-design instruction near line 645, the
  matrix-scale refinement reminder near line 649, and the "recover raw
  instructions" directive near line 665.
- `~/.codex/sessions/2026/05/13/rollout-2026-05-13T01-15-28-019e1fc2-eb26-7443-a6da-4a050de6a5f5.jsonl`,
  parent session metadata for `019e09f3-29f3-70d2-bcf6-434ef853f539` and
  preserved user-message records including the matrix refinement reminders.
- Current planning docs in this folder, especially
  `current_source_of_truth.md`, `user_priority_snapshot.md`,
  `aggressive_matrix_scale_plan.md`, `next_matrix_manifest_design.md`,
  `launch_gate_checklist.md`, `v1d_axis_projection.md`,
  `blank_canvas_noop_opponent_lane.md`, `opponent_diagnostic_design.md`,
  `scripted_wall_avoidant_opponent_baseline_2026-05-13.md`, and
  `prelaunch_validation_audit_2026-05-13.md`.

## Raw User Priorities

- First understand old runs. Project v1d along reward, opponent, render,
  search, collector, learner batch, stochasticity, and cap using score/outcome
  curves. Do not use old survival curves as the training-objective read.
- The next objective is survival learning from visual input, with
  `survival_plus_bonus_no_outcome`. Outcome is eval telemetry only.
- Blank canvas/no-op is the clean first wall-avoidance anchor.
- Repeated seed/copy groups are not optional for important stochastic/random
  rows. About five copies is the default when a cell could become a claim.
- Opponent lanes should be separated: blank canvas, passive invincible,
  random-init/checkpoint frozen, ancestor checkpoint, and scripted wall-avoidant
  answer different questions.
- Render rows should be matched fast/browser pairs with the same logical
  seed/copy. They are diagnostics, not extra independent evidence.
- Stochasticity should be a real ladder, more aggressive than v1d's tiny
  `0.05`, with exact semantics recorded.
- Learner/search/batch axes should be projected from old score curves and then
  kept mostly as sentinels unless they show a clear bottleneck.
- Scale can be aggressive, but not as a blind Cartesian product.

## Launch Prerequisites

We are not close to launch until these are true. These are not matrix design
taste calls; they are gates.

| Gate | Why it blocks |
| --- | --- |
| Old-run score-curve projection is accepted | Otherwise we are still guessing which knobs are worth holding fixed. |
| Reward path is exact | `survival_plus_bonus_no_outcome` must be the trainer reward, not just a row label. |
| Blank-canvas contract is proven | Player 1 must be inert, hidden, non-colliding, reward-ignored, and unable to write trails or catch bonuses. |
| High-cap trainer canary passes | `source_max_steps=65536` must work in the actual stock trainer path. |
| Eval/status carries survivaldiag fields | Need reward components, bonus counts, terminal causes, action histograms/entropy, eval/GIF health. |
| Manifest separates seeds | Need `training_seed`, `reset_seed`, `opponent_policy_seed`, `opponent_behavior_seed`, `eval_seed`, and `copy_id`. |
| Stochasticity semantics are canaried | Need to know affected actor, probability, override rule, and train/eval scope. |
| Opponent lane is wired before inclusion | Scripted/random/checkpoint lanes need first-class IDs, immutable refs/seeds, and smoke tests before rows. |

## Matrix Design Choices

Assuming gates pass, the matrix should be staged. The current docs are aligned
in direction, but the first wave should be more focused on the raw priorities:
blank canvas, repeats, stochasticity, render pairs, and only small sentinels for
compute knobs.

Prune or delay:

- A 12-row first-wave reward ablation is too large. The user was opinionated
  that the main reward is survival plus bonus with outcome off. Keep at most
  4-6 survival-only sentinels after the main lane moves.
- Passive immortal should stay tiny unless GIF/manual inspection shows useful
  trail pressure. Current passive behavior can go out of bounds and is not a
  clean opponent family.
- Do not include scripted wall-avoidant rows in a launch manifest until the
  policy is wired into the stock trainer path with tests and e2e canaries.
- Do not include random-init frozen rows until checkpoint identity and
  opponent policy seed are immutable in the manifest.
- Do not spend first-wave rows on sim16/C64/B64 beyond one matched-pair
  sentinel on the strongest cell.

Expand or sharpen:

- Spend first-wave scale on blank-canvas repeats and a clean stochasticity
  ladder, because this directly tests whether visual wall survival can move.
- Separate repeat purpose. For blank canvas, say whether copies vary
  `training_seed`, `reset_seed`, or both. For random opponents, vary
  `opponent_policy_seed`. For stochastic opponent behavior, vary
  `opponent_behavior_seed`.
- Keep fixed/old/mid/recent frozen rows as minimal controls under the new
  reward, not as success claims.
- Add a stop/go rule after the clean lane: if blank canvas cannot improve
  survival without action collapse, richer opponent blocks are much harder to
  interpret.

## Opponent Lane Order

| Lane | First use | Copy rule |
| --- | --- | --- |
| Blank canvas/no-op | Anchor lane for wall avoidance. | Repeat hard: render pairs across stochasticity and 4-5 copies. |
| Normal/fixed/ancestor checkpoint | Controls under new reward. | Repeat lightly: 1-2 copies. |
| Passive invincible | Dirty canary/trail-generator only. | Minimal rows until manual inspection shows it is useful. |
| Scripted wall-avoidant | Second-wave core if wired and canaried. | Repeat hard if it becomes a claim; include behavior seeds for stochastic variants. |
| Random-init frozen checkpoint | Later variance/opponent-family lane. | Repeat opponent policy seeds; refs must be immutable. |

## Sharper Matrix Shape

Status update: this section is a critique sketch, not the current generated
manifest. The current dry-run review shape is 50 executable rows: 4 exact
preflight, 32 blank-canvas core, 8 blank-canvas extra repeats, 4 passive
dirty-control, and 2 sim16 sentinel rows, plus 10 gated specs.

First serious design target, not a launch-ready manifest, about 48-64 rows
after prerequisites pass:

| Block | Rows | Shape |
| --- | ---: | --- |
| Blank deterministic anchor | 8 | `2 renders x 4 copies`, no stochasticity |
| Blank stochastic ladder | 24 | `2 renders x 3 nonzero levels x 4 copies` |
| Blank extra repeats | 8-16 | best 1-2 levels, more copies |
| Survival-only ablation | 4 | strongest blank cell, matched renders, 2 copies |
| Normal/frozen controls | 4-6 | fixed/old/recent, matched renders, 1 copy |
| Passive immortal dirty canary | 2-4 | matched renders, minimal copies |
| Compute sentinel | 2 | sim16 matched render pair on strongest blank cell |

Second wave, about 100 rows total:

- extend blank-canvas best cells to about five or more true copies;
- add scripted wall-avoidant core only if first-class trainer plumbing exists:
  `2 renders x 2-3 stochasticity settings x 4-5 copies`;
- add a few ancestor controls only if they answer a clear question;
- keep C64/B64 as tiny sentinels, not a grid.

Third wave, 200+ rows:

- add random-init frozen opponent families only after immutable checkpoint
  seeding is solved;
- add scripted variants such as `lazy_weave` and `jitter_force_field` only if
  probes and trainer canaries show distinct, stable behavior;
- spend most extra rows on confirmation repeats of top cells.

## Short Critique

The current proposed matrix is directionally right, but it still risks sounding
launch-near. It should be treated as a design scaffold. The immediate work is
to finish old-run score projection, prove the blank-canvas/reward/readout
contracts, and make the manifest seed model explicit. Only then should row
counts matter.
