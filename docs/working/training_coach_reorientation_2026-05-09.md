# Training Coach Reorientation - 2026-05-09

Scope: focused review of recent training docs plus local Codex state summaries.
This is a correction note for future agents. It does not copy raw private logs.

## What I Inspected

Training docs:

- `docs/working/training_loop_agenda.md`
- `docs/working/training_coach_packet.md`
- `docs/working/training_coach_self_critique_2026-05-09.md`
- `docs/working/dummy_survival_degradation_diagnosis_2026-05-09.md`
- `docs/research/training_evaluation.md`
- `docs/design/training_architecture.md`
- `docs/design/training_eval_protocol.md`
- `docs/runbooks/training_smokes.md`

Instruction and memory summaries:

- `docs/working/user_instruction_packet.md`
- `docs/working/user_message_memory_review.md`

Local Codex state, only as metadata:

- `/Users/shankha/.codex/state_5.sqlite`: table list and thread counts only.
  It currently has 2,987 threads, 173 with `cwd=/Users/shankha/curvy`.
- `/Users/shankha/.codex/logs_2.sqlite`: table list and row count only. It
  currently has 1,387,586 log rows. I did not inspect or copy raw log text.
- `/Users/shankha/.codex/logs_2.sqlite` and prior repo notes agree that this is
  mostly telemetry/tool trace data, not the best source for user intent.

## Actual North Star

The project-wide north star is still faithful CurvyTron reconstruction before
serious training claims.

The training-lane north star is narrower:

Build the smallest honest training loop that can show whether a policy improved
for the right reason:

```text
task -> rollout/self-play -> saved rows -> update -> checkpoint -> eval
-> plain summary
```

The key word is honest. The loop must separate:

- learned behavior;
- scripted safety rules;
- planner-only behavior;
- fixed-seed luck;
- best-checkpoint selection luck.

Modal is the real execution target for serious runs. Local is for tiny debug
only. Mctx, LightZero, survival score, and extra toy games are tools; they are
not the north star.

## Where The Coach Drifted

The coach drifted in five ways:

1. Modal wording was muddled. Modal should be the serious-run target, but Modal
   run-management plumbing is not the learning strategy.
2. Survival got too much narrative weight. The survival task is useful, but the
   current success is explained by safety planner behavior and checkpoint
   selection, not clean learning.
3. The fixed seed-123 survival eval did too much work. It is a monitor/debug
   split now, not a held-out proof.
4. The docs accumulated too many parallel fronts: survival fixes, Tiny Line
   Duel, Pong, Modal storage, Mctx, LightZero, run schemas, and source fidelity.
   These are not all equal next steps.
5. "MuZero-shaped" wording risked false confidence. The current loops are
   useful plumbing, but they are tabular dummy learners with simple planning.

The user's recurring instructions cut against this drift:

- use simple language;
- keep the main thread clear;
- preserve durable docs;
- prefer source and command output over memory;
- be skeptical of clean results;
- do not broaden work without need;
- avoid stale front-door docs.

## Survival Decision

Survival should be a side diagnostic, not the core path right now.

Keep it for:

- checking that train/eval/checkpoint files still work;
- detecting planner/model value bugs;
- testing selection and held-out eval mechanics;
- catching regressions in action histograms and crash causes.

Pause:

- bigger survival runs;
- survival as a learning milestone;
- survival-driven Modal orchestration;
- claims that the learner discovered robust survival.

The reason is clear from the diagnosis note: later checkpoints learn a
crash-heavy, non-positive value landscape. Unknown or less-updated actions can
look better than safer known actions. The untrained planner survives because all
model scores tie at zero and the safety rule wins the tie.

## Why Pong Should Be Emphasized Now

Pong should be emphasized now only as the next learning-facing toy, not as a
replacement for CurvyTron fidelity work.

It helps because:

- survival is too easy for `one_step_safe` and too misleading as evidence;
- Pong already has a tiny raster observation path in `frames.jsonl`;
- raster observations are closer to the future CNN/MuZero path than the current
  tabular survival keys;
- Pong is two-player but easier to read than CurvyTron-style simultaneous trail
  collisions;
- fixed Pong policies can give simple baselines without pretending to be source
  faithful CurvyTron.

Use Pong to test whether the learning stack can consume visual observations and
produce honest eval artifacts. Do not use Pong to answer CurvyTron rule,
collision, trail, or multiplayer scoring questions.

Tiny Line Duel still matters as the CurvyTron-shaped two-player scaffold. Pong
matters because it is the cleaner visual-observation learning probe.

Update after the self-play critique: the local Pong self-play loop exists, but
gen2 lost to its parent and won 0 games against `track_ball`. Treat the loop as
a hypothesis/scaffold, not the source of truth. The next decision is whether to
repair that crude trainer or switch to a known simple baseline/curriculum, with
fixed-baseline evals first.

## Top 5 Next Actions

1. Finish the Pong critique decision.
   Choose repair-self-play or a simpler known baseline/curriculum. Keep
   `random_uniform`, `track_ball`, parent/previous, and selected-best eval rows
   before any stronger claim.

2. Freeze survival as a regression diagnostic.
   Keep the current eval, sweep, and held-out tools. Run them after planner or
   trainer changes, but stop scaling survival for quality claims.

3. Keep Tiny Line Duel as the CurvyTron-shaped multiplayer check.
   Improve paired-seat pressure and reset variety only if current rows are too
   deterministic to judge trainer changes.

4. Tighten the front-door training docs.
   Make `training_coach_packet.md`, `training_loop_agenda.md`,
   `training_smokes.md`, and `training_eval_protocol.md` agree on the same
   story: survival diagnostic, Pong visual-learning probe, Tiny Line Duel
   multiplayer scaffold, Modal as whole-job support.

5. Keep Modal as the serious-run target, but delay larger Modal and Mctx work
   until the learner path is chosen. Modal run/attempt ids and Mctx synthetic
   benchmarks are still useful, but they should not outrank the next learning
   signal.

## What Future Agents Should Read First

Read in this order:

1. `docs/working/training_coach_reorientation_2026-05-09.md`
2. `docs/working/training_coach_packet.md`
3. `docs/design/training_eval_protocol.md`
4. `docs/runbooks/training_smokes.md`
5. `docs/working/dummy_survival_degradation_diagnosis_2026-05-09.md`
6. `docs/working/user_instruction_packet.md`

If the task touches source fidelity, read the environment docs too:

- `docs/design/environment/README.md`
- `docs/design/environment/reconstruction_workflow.md`
- `docs/design/environment/training_interface_contract.md`

## Short Reset For The Coach

Do not chase every scaffold.

Current training question:

```text
Can we produce a small, visual, checkpointed learning run whose eval separates
the learner from scripts, planner rules, and seed luck?
```

Best next bet: Pong visual-learning probe, but not more generations by default.

Keep survival as the warning light. Keep Tiny Line Duel as the multiplayer
shape check. Keep Modal outside the hot loop, but use it as the durable place
for serious train/eval runs.
