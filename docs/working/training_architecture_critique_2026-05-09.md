# Training Architecture Critique - 2026-05-09

Role: training architecture critic.

Scope: review Coach work and Optimizer docs at system level. No code changes,
no pytest, and no checkpoint archaeology. This note asks why the current stack
has not produced reliable learning, meaning a repeatable checkpoint curve that
improves under independent heldout eval for the right reason.

## Sources Reviewed

Primary Coach docs:

- `docs/working/training_coach_handoff_2026-05-09.md`
- `docs/working/training_coach_self_critique_2026-05-09.md`
- `docs/working/training_coach_reorientation_2026-05-09.md`
- `docs/working/training_coach_packet.md`
- `docs/working/training_state_index_2026-05-09.md`
- `docs/working/training_process_critique_2026-05-09.md`
- `docs/working/training_lessons_for_curvytron_2026-05-09.md`
- `docs/design/training_architecture.md`
- `docs/design/training_eval_protocol.md`

Primary Optimizer docs:

- `docs/working/optimizer/current_status_2026-05-09.md`
- `docs/working/optimizer/setup_synthesis_2026-05-09.md`
- `docs/working/optimizer/actor_loop_architecture_2026-05-09.md`
- `docs/working/optimizer/world_model_2026-05-09.md`
- `docs/working/optimizer/measurement_plan_2026-05-09.md`
- `docs/working/optimizer/blockers_2026-05-09.md`
- `docs/working/optimizer/profiling_log_2026-05-09.md`

Targeted LightZero/Pong diagnostics:

- `docs/working/lightzero_dummy_pong_root_cause_red_team_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_support_scale_fix_plan_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_observation_model_critique_2026-05-09.md`
- `docs/working/lightzero_muzero_target_semantics_2026-05-09.md`
- `docs/working/lightzero_official_atari_collapse_investigation_2026-05-09.md`
- `docs/working/lightzero_pong_setup_critique_wave2_2026-05-09.md`
- `docs/working/pong_two_lane_worldview_2026-05-09.md`

## Blunt Thesis

The repo has not shown reliable learning because it still does not have one
boring, repo-owned, full actor loop that can learn a small competitive task
under transparent rollout, update, checkpoint, and eval rules.

Instead, the work has repeatedly asked LightZero to serve three roles at once:

- outside control for stock tasks;
- bridge adapter for custom games;
- de facto architecture for future CurvyTron training.

That is too much weight for a library integration that is still running
off-recipe, tiny-scale settings on custom sparse tasks whose root targets,
support scale, and independent eval behavior are not yet healthy.

The Coach lane has built useful scaffolding: Modal jobs, scorecards, checkpoint
mirroring, target sidecars, official controls, dummy Pong, baseline learners,
and enough diagnostics to stop several bad branches. But scaffolding has been
mistaken too often for learning evidence. The Optimizer lane says the missing
piece plainly: there is no production-relevant actor loop with real env
reset/autoreset, trainer observations, policy/search, replay or rollout
handoff, learner update, checkpoint publish, evaluator, and profiler all
running together.

## Architectural Diagnosis

The current setup can answer "can this thing execute?" much better than "is the
policy improving for a valid reason?"

Execution is real:

- stock LightZero controls run;
- official Atari Pong mechanics run;
- custom dummy Pong can train through LightZero and be scored independently;
- Modal can run whole train/eval jobs and persist artifacts;
- scorecards and sidecars can expose action collapse.

Reliable learning is not real yet:

- official Atari Pong has mechanical passes and small/unstable signals, but the
  reproduced settings remain far from stock recipes;
- custom dummy Pong repeatedly fails heldout checkpoint-curve gates;
- trainer-side action diversity does not survive independent deterministic
  MCTS eval;
- CurvyTron has no full trainer loop;
- there is no repo-native PPO/CleanRL-style baseline to tell whether the
  environment contract is learnable without MuZero target complexity.

The failure mode is therefore architectural, not just a bad checkpoint. The
system lacks a simple learning spine that can isolate framework bugs from task
bugs from evaluation bugs.

## Correction: Not "Just A Reference"

Do not demote LightZero to "just a reference" in the lazy sense.

If LightZero has not been convincingly replicated on a task it is supposed to
handle, then the project has not yet calibrated its main MuZero control. Moving
on as if LightZero is solved would be another version of the same mistake:
counting mechanical execution as learning evidence.

Replication remains necessary because it would prove several things that custom
dummy Pong cannot prove:

- the pinned LightZero install, Modal image, dependencies, and trainer entry
  points can reproduce a known working learning pattern;
- the official observation/model/action stack can produce a checkpoint curve
  under settings close enough to upstream recipes;
- the eval path, checkpoint loading, action meanings, frame stacking, and
  no-fallback policy execution are trustworthy;
- the team knows what "healthy MuZero under this library" looks like before
  adapting it to a custom game.

But replication would not prove the CurvyZero architecture:

- it would not prove the custom simultaneous environment contract;
- it would not prove CurvyTron source fidelity;
- it would not prove hidden-opponent or checkpoint-opponent handling;
- it would not prove project-owned replay, rollout buffers, artifact schemas,
  or actor-loop timing;
- it would not prove that a single-ego LightZero wrapper is the right final
  shape for multiplayer CurvyTron.

So the right stance is parallel, not dismissive. Keep LightZero replication as
a serious calibration lane. In parallel, build the repo-native lane because it
answers questions LightZero replication cannot answer. The repo-native
simultaneous `[B,P]` PPO/CleanRL-style actor loop is a CurvyTron architecture
lane, not a replacement for LightZero replication. It must not pretend
LightZero is solved; it should treat LightZero as an unresolved control whose
eventual success or failure will sharpen the architecture decision.

## Ranked Hypotheses

### 1. The main framework fit is wrong for the current question

Likelihood: very high.

LightZero is useful, but it is a poor first microscope for custom CurvyTron
architecture. It hides or owns too much of the collection/search/replay/target
path exactly where the project needs visibility. The custom dummy Pong wrapper
turns a simultaneous two-player game into a single-ego fixed-action MDP with
the opponent hidden inside `env.step()`. That is a reasonable bridge, but it is
not the future CurvyTron actor loop.

This mismatch makes failures hard to classify. A collapse can be a weak game,
bad target distribution, wrong support scale, low simulations, replay sampling
shape, manual eval drift, framework assumption, or real non-learnability. The
docs have had to build many sidecars just to see what a repo-native loop would
have exposed by default.

Falsifier:

- Build or run a transparent repo-native PPO/IPPO baseline on the same
  trainer-facing observation/action/reward contract.
- It must include rollout buffer, learner update, checkpointing, independent
  scorecard, action histograms, terminal causes, and fixed baselines.
- If this simple runner also cannot improve on scoreable dummy Pong or a
  minimal CurvyTron 1v1 slice, the issue is likely environment/objective
  quality, not LightZero fit.
- If PPO learns while LightZero does not, LightZero should remain a control and
  target-audit lane, not the main architecture.

### 2. The LightZero controls are too small and too off-recipe to be meaningful learning tests

Likelihood: very high.

Official Atari and custom dummy Pong have both run at smoke or diagnostic
scales. That is fine for plumbing, but bad for learning conclusions. The
official Atari notes show large gaps from stock settings: far fewer env steps,
fewer collectors, lower simulations, smaller batch sizes, shorter segments, and
manual eval caps. Custom dummy Pong is even further from an official working
Pong pattern: tabular or tiny raster MLP, three actions, hidden scripted
opponent, low simulation counts, small replay/update budgets, and custom
scorecards.

The right conclusion is not "just train longer." Several same-premise rungs
already failed. The point is sharper: smoke-scale LightZero cannot adjudicate
whether the architecture learns. It can only prove the plumbing still moves.
That does not make LightZero optional or "done"; it means the replication lane
has not yet reached the quality bar needed to use it as calibration.

Falsifier:

- For official Atari, run one bounded reproduction that is close enough to the
  stock recipe to produce a real checkpoint curve: substantially higher
  simulations, batch/update scale, collectors, env steps, and stock evaluator
  parity.
- For custom dummy Pong, first prove target quality on fixed scoreable states
  at meaningful simulation counts before any larger run.
- If a near-recipe official run still fails while stock evaluator parity and
  action meanings are verified, LightZero reproduction itself is suspect.
- If target-quality probes pass and a modest custom run still fails, move to
  objective/env quality and learner architecture hypotheses.

### 3. Custom dummy Pong target quality is bad, so MuZero is learning poor labels

Likelihood: very high.

The strongest custom dummy Pong lesson is that executed actions are not policy
targets. LightZero trains the policy head toward MCTS root visit
distributions. Exploration can execute `down`, while the stored visit target
still gives `down` zero or tiny mass. That explains why trainer-side behavior
can look alive while heldout eval collapses.

Sparse rewards make this worse. If value estimates remain near flat, MCTS roots
do not become useful labels. More replay of weak roots can reinforce collapse
instead of solving it.

Falsifier:

- On 24 to 64 known scoreable contact/angle states, log oracle-best action,
  executed action, root visits, normalized policy target, root value, policy
  logits, terminal outcome, and tie rate.
- Sweep simulations such as 2, 8, 16, 25, and 50.
- This hypothesis is falsified if meaningful simulations assign
  state-dependent mass to oracle-good actions and values rank eventual outcomes,
  yet training still fails to learn those targets.
- It is confirmed if known good actions get zero or near-zero target mass, or
  higher simulations only make a wrong action more confident.

### 4. The reward/value support scale is likely miscalibrated for tiny sparse custom rewards

Likelihood: high.

Custom dummy Pong and early CurvyTron rewards are small: mostly `-1`, `0`, and
`+1`. A broad LightZero categorical support inherited from general configs can
make tiny targets poorly calibrated. The support-scale fix plan identifies the
decisive pinned-LightZero fields: `support_scale`,
`reward_support_size`, and `value_support_size`. Logging requested ranges is
not enough if the compiled model still uses a broad default.

This can flatten value learning, which then flattens MCTS roots, which then
poisons policy targets.

Falsifier:

- For every custom LightZero run, log requested and compiled support fields in
  the train summary.
- Inspect raw reward/value targets before transform, categorical mass after
  transform, clipping rate, and inverse-transformed model predictions on
  curated win/loss states.
- Run a one-batch or tiny curated overfit with the actual compiled support.
- This hypothesis is falsified if compiled supports match task scale, targets
  have variance, predictions separate win/loss states, and roots are still bad.

### 5. The custom environment target may be low quality or too biased for learning claims

Likelihood: high.

Dummy Pong is useful as a microscope, but it has become an unstable target for
claims. Some default reset/opponent combinations are not scoreable in the way
early win gates assumed. Contact-pressure states are scoreable for some
opponents but not for others. `track_ball` is often a survival/tie floor rather
than a score target. This means the learner may be trained or judged on
distributions where the intended improvement is either rare, ambiguous, or not
reflected by the chosen gate.

For CurvyTron, this risk grows. If the custom env target is not source-faithful
enough, not scoreable enough, or not instrumented enough, training failures will
be attributed to algorithms when the target was bad.

Falsifier:

- Before scaling any custom task, publish a scoreability table: reset profile,
  opponent, seed split, oracle/simple policy outcomes, action sensitivity, and
  terminal-cause distribution.
- A target earns training budget only if a non-learning oracle or simple policy
  can improve the exact scorecard metric under heldout seeds.
- This hypothesis is falsified for a task when scoreability is proven, baseline
  ladders are sane, and independent eval still shows no learning under a
  transparent learner.

### 6. The missing repo-native PPO baseline is blocking diagnosis

Likelihood: high.

The project is jumping between CEM, imitation, MLP baselines, LightZero MuZero,
Mctx search spikes, and Modal plumbing without the simplest modern RL control:
a transparent PPO/IPPO runner over the repo's own simultaneous `[B, P]` shape.

PPO is not the final ambition. It is the diagnostic baseline that answers:

- can the observation/reward/action contract be learned at all;
- is the environment fast enough for rollout-based learning;
- do scorecards and eval gates behave sensibly;
- where does time go in the real actor loop.

Without PPO, every LightZero failure is overloaded with MuZero-specific
questions about search targets, support transforms, replay, reanalysis, and
framework assumptions.

This is not an argument to abandon LightZero. It is an argument to stop making
LightZero carry the entire diagnosis. A PPO/IPPO lane can proceed in parallel
because it tests repo-owned environment, rollout, eval, and profiling contracts.
It should report against LightZero, not replace the need to replicate it.

Falsifier:

- Implement or run a minimal PPO/IPPO loop with the Optimizer actor-loop shape:
  `reset_many`, `step_many`, ego-row compaction, batched policy forward,
  rollout buffer, GAE or simple returns, update, checkpoint, scorecard.
- If PPO cannot beat random or scripted baselines on a proven-scoreable toy,
  the environment/objective/eval contract is suspect.
- If PPO learns, MuZero/LightZero becomes the suspect rather than the game.

### 7. There is no full actor loop, so speed and learning claims are both partial

Likelihood: high.

Optimizer docs are clear: current speed evidence is mostly fixture rows,
debug/no-event splits, synthetic policy/search timing, and in-memory replay.
There is no production-relevant CurvyTron loop with reset/autoreset, final
observations, trainer observations, real policy/search, replay or learner
handoff, learner update, checkpoint publication, evaluator, and profiler.

That absence matters for learning, not just performance. A real loop imposes
contracts: final observations, rollout alignment, opponent metadata, policy
staleness, batch freshness, replay schemas, and eval cadence. Until those are
real, the project cannot tell whether the learning system is broken or simply
not assembled.

Falsifier:

- Produce one end-to-end actor-loop report with useful games/minute, env
  steps/sec, ego rows/sec, replay rows/sec, p50/p95/p99 action latency, actor
  idle, learner idle, and policy staleness.
- The report must use production-like observation/reward/final-observation
  paths and real model/search timing.
- This hypothesis is falsified only when such a loop exists and failures remain
  despite correct rollout/update/eval wiring.

### 8. Observability is rich around failures but still not native to the training loop

Likelihood: medium-high.

The repo has many good diagnostics now: sidecars, scorecards, target logs,
action histograms, survival telemetry, support audits, and Modal artifacts.
But much of this was added reactively after failures. It is not yet a single
required schema for every serious run.

That leads to repeated ambiguity:

- trainer-side collection versus independent eval;
- executed action versus policy target;
- requested config versus compiled config;
- latest versus selected-best;
- monitor versus heldout;
- manual evaluator versus stock evaluator;
- training horizon versus episode horizon.

Reliable learning needs these distinctions by construction, not after-the-fact
forensics.

Falsifier:

- Define one mandatory train/eval artifact schema for serious runs and reject
  runs missing it.
- Required fields include lane label, run/attempt/checkpoint refs, observation
  schema, reward schema, action schema, support scale, opponent policy,
  reset profile, seed split, target-policy samples, action histograms,
  terminal causes, no-fallback status, and evaluator path.
- This hypothesis is falsified if all serious runs carry this schema and the
  remaining failures are still algorithmic, not interpretive.

### 9. Eval gates exist, but they are not yet decisive enough to prevent bad campaigns

Likelihood: medium-high.

The eval docs are good. The behavior has been improving. But the project still
spends attention on branches after their stop signal is already visible:
same sparse dummy Pong, update/replay-only, simple exploration, contact-pressure
campaigns, raster-flat quality claims, survival scaling, and checkpoint
forensics.

The gate problem is not just metrics. It is decision discipline. A run should
change a lane decision or not be run.

Falsifier:

- For every proposed training run, require lane, decision, predeclared stop
  rule, baseline rows, and heldout confirmation path.
- Stop rules should close branches automatically unless a root-cause hypothesis
  changes.
- This hypothesis is falsified when repeated same-premise runs disappear and
  every experiment either promotes, stops, or falsifies a ranked hypothesis.

### 10. Visual learning claims are premature because the custom visual path lacks history

Likelihood: medium.

The docs already say this, but it matters architecturally. `tabular_ego` is a
reasonable debug lane. `raster_flat` is a single tiny frame flattened into an
MLP; it lacks velocity/history and is not comparable to official stacked-frame
conv Atari Pong. CurvyTron will be even more history-sensitive because heading,
trail gaps, elapsed state, and opponents matter.

If visual claims are made from single-frame flat rasters, the system may blame
MuZero when the observation is simply insufficient.

Falsifier:

- Run a visual bridge with stacked frames or explicit temporal channels,
  channelized semantics, schema ids, and matched checkpoint loading.
- Compare against tabular and single-frame controls on the same scoreable
  evals.
- This hypothesis is falsified if a proper history-bearing visual path still
  collapses while tabular and PPO baselines also fail.

## What To Stop Saying

- Stop saying Modal progress as if it is learning progress. Modal is execution
  substrate.
- Stop saying trainer-side action diversity as if it is learned policy
  diversity. The policy target is the root visit distribution.
- Stop saying dummy Pong visual results as if they are official Atari parity.
- Stop saying a tiny LightZero smoke falsifies MuZero or proves it should
  scale.
- Stop treating checkpoint archaeology as the main lane. It is a tool for
  deciding lanes.

## What To Do Next

1. Make repo-native PPO/IPPO the first architecture diagnostic.
   It should use the real simultaneous actor-loop shape and the same eval
   protocol. It is not a final agent; it is the cleanest learnability probe.

2. Keep LightZero in two bounded roles.
   Use official Atari as a serious replication/control lane, not as a solved
   reference. Use custom dummy Pong for target-sidecar and adapter diagnostics.
   Do not let either define CurvyTron architecture by accident.

3. Prove custom target quality before more custom MuZero scale.
   Fixed scoreable states, root visits, values, oracle action labels, support
   scale, and simulation sweeps should pass before training budget.

4. Build the full actor-loop report.
   The Optimizer loop should include real env, obs packing, policy/search,
   replay/rollout, learner, checkpoint, evaluator, and profiler. Fixture speed
   is not enough.

5. Turn eval gates into branch decisions.
   Every training run should have a lane, a decision, a stop rule, fixed
   baselines, and heldout confirmation. Runs that only add another ambiguous
   checkpoint row should not run.

## Bottom Line

The system is not reliably learning because it still lacks the simplest
diagnostic spine: a transparent repo-owned actor loop with a transparent
baseline learner and mandatory eval artifacts.

LightZero remains useful, but right now it is carrying too much architecture
risk. Custom dummy Pong remains useful, but right now it is more microscope than
milestone. Modal remains useful, but it is not evidence of learning.

The next credible learning claim should come from a full loop that can say, in
plain terms:

```text
same task, same split, same baselines, same evaluator:
the checkpoint improved, the action policy changed for the right states,
the target/reward/support data are sane, and heldout confirmed it.
```

Until then, the honest status is: good scaffolding, weak learning proof.
