# Coach Operating Patterns

Purpose: keep the overnight operating style explicit so the main thread can
review it without relying on the user to repeat reminders.

## Role

The main thread is the coach/orchestrator. It should:

- keep the high-level picture clear;
- plan a few steps ahead;
- delegate bounded work to subagents;
- send follow-ups when the worldview changes;
- merge results into docs;
- decide gates and launches;
- keep explanations simple.

The main thread should not sit idle while a subagent, eval, or training run is
running. It should work on docs, analysis, tooling, canary planning, matrix
design, or follow-ups.

## Parallelism

Default to parallel work when tasks are independent.

Basic sanity checks can run on the main thread, especially when they are quick
or when the next step depends on their result. Longer validation should fan out:
delegate artifact inspection, matrix critique, docs synthesis, and independent
canaries when they do not touch the same files or Modal app state.

When code is stable and remaining work is mostly gating, run independent gates
in parallel. If a speculative lane fails, drop or kill the rows that depended on
that lane rather than making unrelated lanes wait. Only serialize steps where
the output changes the actual command set, such as the final manifest after a
pass/drop decision.

Hard rule: a planned expansion is not a duplicate, but unreadable names and app
spray are launch blockers. Large batches should have one clear matrix name, one
readable run prefix, and one grouped submit artifact.

Hard rule: large CurvyTron batches submit into one deployed Modal app. Do not
run one `modal run` command per row. The grouped submitter must spawn the
checkpoint poller first and the train function second for each row.

Hard rule: if the user asks for a hold before launch, finish prep, then sleep
at the launch boundary. Do not sneak in a launch during the hold.

Parallel stage pattern: if steps 1-5 can all be started and only step 3 later
breaks, start them. If step 3 fails, kill or ignore step 3 and anything that
depended on it. Keep step 1, step 2, step 4, and step 5 moving if they are still
valid.

Use subagents for:

- code changes with clear ownership;
- read-only critiques;
- docs cleanup;
- matrix design;
- tooling;
- bug searches;
- historical paper-trail checks.

Keep the active lanes in [delegation_log.md](delegation_log.md). Do not launch
so many lanes that nobody knows what is running. Reuse existing agents when the
thread limit is reached.

## Follow-Ups

Follow-ups are part of orchestration.

Send a follow-up when:

- the main hypothesis changes;
- user priorities clarify an axis;
- a worker may be proceeding under stale assumptions;
- a blocker appears;
- a result from one lane changes another lane.

Good follow-ups are narrow: one correction, one priority, or one concrete
question.

## Docs As Working Memory

Use docs to avoid losing the plot.

Active docs:

- [current_source_of_truth.md](current_source_of_truth.md)
- [instruction_digest_2026-05-13.md](instruction_digest_2026-05-13.md)
- [user_priority_snapshot.md](user_priority_snapshot.md)
- [todo.md](todo.md)
- [hypotheses_and_evidence.md](hypotheses_and_evidence.md)
- [v1d_axis_projection.md](v1d_axis_projection.md)
- [next_overnight_matrix_plan.md](next_overnight_matrix_plan.md)
- [launch_gate_checklist.md](launch_gate_checklist.md)
- [delegation_log.md](delegation_log.md)
- [eval_curve_tooling_plan.md](eval_curve_tooling_plan.md)
- [archive_and_stale_docs.md](archive_and_stale_docs.md)

Older docs are evidence, not automatically current truth. Add stale warnings
instead of deleting history during active investigation.

## Self-Critical Loop

Periodically stop and ask:

- Are we still trying to prove CurvyTron can learn visual survival, or are we
  polishing side tooling?
- Is this lane connected to a launch gate, a run-matrix decision, or a known
  bug?
- Are we confusing training reward with eval metrics like outcome, survival,
  GIF appearance, or action distribution?
- Are we adding a new knob because it answers a hypothesis, or because the
  matrix is becoming knob soup?
- Are we using the trusted stock LightZero lane unless there is a clear reason
  not to?
- Did a subagent result change the source of truth, the todo list, or the
  launch gates?
- Do the docs still say what is actually true after the latest code/test
  result?
- If context feels compressed or contradictory, did we inspect raw local Codex
  session JSONL before trusting derived docs?

If the answer is unclear, re-read
[current_source_of_truth.md](current_source_of_truth.md),
[todo.md](todo.md), [launch_gate_checklist.md](launch_gate_checklist.md), and
[delegation_log.md](delegation_log.md) before doing more work.

## Current CurvyTron Understanding

- Trusted learning lane is stock LightZero `train_muzero` through `--mode train`.
- Old custom `two-seat-selfplay` is historical/untrusted for learning claims.
- v1d showed fixed/old opponents can produce score movement.
- v1d showed recent/mid frozen opponents can be score-saturated immediately.
- For old v1d analysis, use corrected outcome/score curves.
- For next survivaldiag runs, use survival/reward curves as primary.
- Outcome remains useful as an eval/readout metric only; outcome reward should
  be off/zero in the next diagnostic lane.

## Next Experiment Direction

The next serious batch should test whether stock LightZero can learn visual
CurvyTron survival when weak-opponent death cannot saturate reward.

Core priorities:

- use survival-first plus bonus reward;
- set episode cap high, likely `65536`;
- sweep stochasticity meaningfully;
- run matched fast/browser render rows for serious settings;
- keep search/collector/learner batch sweeps small;
- run at least 12 hours only after launch gates pass.

Current correction: the ugly 50-row batch and the broken 300a launch are
historical. The fixed 300b batch is healthy background training. The first clean
mixture batch, `curvy-mix2-clean-20260513a`, is healthy-ish and should be used
for cadence, GIF, and early learning readout. The current new wave is
`curvy-mix3-currentckpt-20260513a`, launched through the single app using
recent/mid/old checkpoints from 300b.

Recent-checkpoint mixture opponents are no longer only a feature-prep lane. The
trusted stock path now has static weighted episode-level mixtures. It still does
not have live rolling same-run self-play. Say the opponent source plainly:
mix3 uses frozen refs from `curvy-survive-bonus-large-20260513b`.

## Do Not Forget

- Do not confuse outcome score from old runs with survival reward for new runs.
- Do not sweep episode cap again unless there is a real technical reason.
- Do not treat recent frozen opponents as good curriculum by default.
- Do not overread one best checkpoint if the curve collapses later.
- Do not drop possible late bloomers too aggressively.
- Do not let GIF/eval plumbing failures masquerade as learning failures.
- Do not let old docs steer a new launch.
- Do not mistake a passing dry gate for a learning claim.
- Do not mistake a spawned Modal call or a poller file for a running trainer.
  Verify sampled rows have trainer files such as `attempt.json`,
  `status_heartbeat.json`, and eventually `progress_latest.json` or
  checkpoints.
- Do not mistake launch order for speed. If fast-render rows are submitted
  before browser rows, early `iteration_0` counts are startup evidence, not a
  clean render-fidelity comparison.
- Do not launch accidental duplicates. Do launch planned independent expansion
  waves with new names and new seeds when their own checks are clear.
- Re-read this file, `orchestration_plan.md`, `todo.md`, and
  `current_source_of_truth.md` whenever work resumes or the thread feels stuck.

## Communication

Use simple language.

When reporting status, answer:

- what is running;
- what has returned;
- what changed;
- what is blocked;
- what happens next.

Avoid vague terms like "setting" or "signal" without saying which metric:
outcome, survival, reward, action distribution, or eval health.
