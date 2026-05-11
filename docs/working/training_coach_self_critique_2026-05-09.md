# Training Coach Self-Critique

Date: 2026-05-09

Scope: hard critique of the training coach lane after the first dummy survival,
Tiny Line Duel, Modal Volume, and Mctx planning passes. This is a correction
memo, not a new roadmap.

## Current Truth / No More Pretending

- We validated stock LightZero CartPole MuZero progression.
- We validated an Mctx search benchmark.
- We validated a CEM-v2 Pong baseline.
- We validated a raster-only MLP Pong baseline.
- We have not run an actual project-owned MuZero/Mctx train loop for Pong.
- We have not run an actual project-owned MuZero/Mctx train loop for Curvy.
- CEM-v2 and the MLP are baselines and scaffolding only. They are not MuZero
  progress.
- The next main lane is LightZero-first: adapt dummy Pong as a custom env and
  run a capped LightZero MuZero trainer on Modal. Project-owned Mctx is fallback
  if LightZero fails or hides required telemetry/artifacts.

Prevention rules:

- Prove the target is scoreable before scaling it.
- Keep baselines separate from MuZero.
- Name the algorithm in every experiment title, command, and summary.
- Distinguish stock LightZero MuZero from project-owned MuZero/Mctx.
- Do not describe CEM, imitation, or MLP results as MuZero progress.

## Bottom Line

The lane is usefully building runnable scaffolds, but the docs are close to
over-reading toy results. The strongest correction is this: the current dummy
survival win is a planner-prior/checkpoint-selection signal, not evidence that
the learner has discovered robust survival. Keep the scaffolds, but tighten the
claims and stop scaling until the learner/planner failure mode is understood.

Post-CEM-v2 update: the same critique now applies to Pong. The coach judged
progress by wins too much, did not prove early enough that the target was
scoreable, and spent runs against an impossible hard gate. Default
`track_ball` was not a valid win-pressure target from normal resets; it was a
survival/tie floor. The correction was the survival audit, the exact
beatability probe, the target ladder, and then the CEM-v2 score-pressure
monitor against `lagged_track_ball_1`.

New working rule: before scaling a training target, first prove the target is
scoreable and define the fine-grained metric that will show improvement before
the win count moves.

## Corrections To Current Claims

- "Positive learned-checkpoint signal" is true only in a narrow sense. The
  safety-aware planner patch hard-coded immediate-clearance avoidance and safer
  tie-breaking; iteration-2 and iteration-4 then matched `one_step_safe` on 10
  diagnostic replay episodes. That is useful, but it mostly proves the planner
  prior and eval/sweep plumbing.
- Best-checkpoint selection is currently a diagnostic, not a success criterion
  by itself. Selecting the best checkpoint on one tiny reused eval list can
  overfit those starts as easily as it can reveal real improvement.
- The dummy survival task is now too easy for the scripted safety heuristic and
  too brittle for the tabular learner. It should remain a plumbing/regression
  harness, not become the main evidence for MuZero readiness.
- Tiny Line Duel proves simultaneous-step and ego-replay shape, but the all-draw
  and weak learned-checkpoint results mean it has not yet proven useful
  multiplayer learning pressure.
- Modal Volume persistence is not run management. The current smoke proves
  durable artifact writes and refs/hashes, not run ids, attempts, latest
  pointers, resume, concurrency, or cleanup semantics.
- Mctx is still a search-library spike, not a training plan. The docs are right
  to prefer a synthetic benchmark first; do not imply Mctx will fix learning
  until the dummy loop has honest targets and evaluation.

## Overfitting Risks

- Single-seed diagnostic eval is doing too much narrative work. Keep it for
  regression, but require a fresh recorded multi-start eval wave before calling
  any checkpoint better.
- The safety planner can mask lack of learning. Add action/value diagnostics
  before celebrating survival: chosen action histograms, zero-clearance action
  rate, planner score components, and whether policy behavior changes without
  the safety tie-break.
- Toy scaffolds may optimize the artifact loop rather than the game. Every new
  scaffold should answer a missing CurvyTron-shaped question: simultaneous
  deaths, seat symmetry, trail geometry, opponent policy metadata, or replay
  target construction.
- "MuZero-shaped" names can create false confidence. A tabular NumPy loop with
  shallow planning is valuable, but it is not evidence about latent dynamics,
  batched search, or neural target learning.

## Pace Check

- Toward Modal: keep moving, but only at coarse job boundaries. Next Modal work
  should be run/attempt ids plus resume/latest semantics, or a quarantined Mctx
  import/benchmark. Do not build orchestration around a learner that still
  degrades locally.
- Toward Mctx: move in parallel, not as the blocking main thread. A CPU import
  smoke and synthetic fixed-shape benchmark are appropriate while the local
  learner is inspected. Real env rollouts inside Mctx should wait.
- Toward environment fidelity: do not block dummy-loop maturation on source
  fidelity, but do keep the adapter metadata pressure alive. Training replay
  without rules/observation/reward/action schema ids is future debt.

## Parallelize Next

- Worker A: inspect dummy survival degradation after iteration 4. Produce a
  short note with replay composition, exploration crash rate, value/action
  tables, and why later checkpoints lose the safety behavior.
- Worker B: harden checkpoint selection. Add or document held-out eval seeds,
  minimum episode counts, and a best-checkpoint manifest that records selection
  metric and eval config.
- Worker C: improve Tiny Line Duel evaluation pressure without new game
  scaffolds. Focus on seat symmetry, non-mirror starts, random/sticky/scripted
  baselines, and collision/death metadata.
- Worker D: run the Mctx spike only as dependency and synthetic-shape evidence:
  CPU import first, then Modal GPU benchmark if the pins are clean.
- Worker E: define the smallest run artifact schema extension: run id, attempt
  id, parent checkpoint, latest pointer, resume source, and config hash.

## Explicitly Defer

- Bigger dummy survival runs before the degradation diagnosis.
- Treating best fixed-seed survival as a learning milestone.
- Full Modal orchestration, queues, multi-node training, or GPU training loops.
- Bigger LightZero runs before the dummy Pong custom-env config and tiny train
  smokes pass.
- Joint-action search, leagues/Elo, checkpoint pools, and 3+ player training.
- New Pong-like scaffolds. Tiny Line Duel already occupies the "second toy"
  slot; improve or retire it before adding another toy.

## Immediate Agenda Correction

Earlier correction: do not open new toy environments when Tiny Line Duel already
covers the CurvyTron-shaped multiplayer scaffold. After the CEM-v2 recovery,
Pong is active again because it is the smallest visual score-pressure lane, not
because the old impossible `track_ball` gate is valid.

## CEM-v2 Recovery Agenda

- Treat CEM-v2 as a Modal-backed geometry baseline, not a serious MuZero path.
- Treat the raster-only MLP as a supervised baseline, not MuZero progress.
- Make the next main lane a real LightZero custom dummy Pong MuZero smoke, or
  say plainly which fallback/baseline lane ran.
- If running another visual policy against `lagged_track_ball_1`, name it as a
  baseline and use score wins as the hard metric with survival/loss-delay only
  as supporting telemetry.
- Keep LightZero Pong blocked at the ROM/license gate.
- Keep Mctx isolated as a benchmark/search spike until it has a clear training
  interface reason to touch replay or environments.
