# LightZero CurvyTron Wrapper Verbose Brief - 2026-05-09

Owner: verbose brief writer for the next agent/optimizer.

Scope: this is a plain-language handoff. It does not add code, does not run
pytest, and does not make a new training claim.

## Short Version

LightZero is real. It has a full MuZero-family trainer: model, policy, MCTS,
collector, evaluator, replay, learner, configs, logging, and checkpoints. We
have already made it run on stock controls and on a custom dummy Pong wrapper.

LightZero can also do self-play. This brief should not be read as saying
MuZero or LightZero lacks self-play. The question is narrower: does the
available LightZero self-play API match CurvyTron's simultaneous-tick,
multi-player mechanics, or do we need a project wrapper that translates those
mechanics honestly into a LightZero-compatible single-ego or row-based view?

Single-ego wrappers are also real. The current dummy Pong wrapper proves that
LightZero can control one ego player while the project wrapper supplies the
other player action and returns a LightZero-shaped observation:

```text
{
  "observation": one ego observation,
  "action_mask": one ego action mask,
  "to_play": -1
}
```

That pattern can probably be used for a first CurvyTron bridge too.

The current problem is not "does LightZero have MuZero?" It does.

The current problem is:

```text
How do we make a CurvyTron LightZero wrapper that is honest, observable, and
scalable, instead of a wrapper that merely runs?
```

For CurvyTron, "runs" is too weak. The wrapper must preserve simultaneous
game semantics, expose enough telemetry to audit the learned target, avoid
lying about environment reward and policy quality, and not destroy throughput
with debug payloads once the system moves from diagnosis to scale.

## The Important Distinction

There are two separate facts that should not be mixed:

1. LightZero contains a full MuZero implementation.
2. CurvyTron still needs a truthful adapter and evaluation contract.

Fact 1 is basically settled. Local notes already confirm LightZero can create a
MuZero policy, run collection, train, evaluate, write `.pth.tar` checkpoints,
strict-load matched checkpoints, and expose MuZero pieces such as
representation, dynamics, prediction, policy, value, and reward heads.

Fact 2 is not settled. CurvyTron is not a normal single-agent Gym game and not
a turn-based board game. It is simultaneous multiplayer. The project-owned
environment shape wants live players, joint actions, per-player observations,
per-player rewards, terminal/truncation metadata, and seat/perspective
discipline.

LightZero does not automatically own that exact CurvyTron shape. It owns real
MuZero training and real self-play patterns. CurvyZero still owns the
translation between CurvyTron's simultaneous tick mechanics and whichever
LightZero interface we choose.

## Why We Still Own Mechanics If LightZero Has Self-Play

This is the likely source of confusion, so be explicit.

LightZero can perform self-play. In board-game-style environments, LightZero
can use `to_play` and a self-play mode where the policy acts for the side whose
turn it is. That matches games where one legal move happens at a time and the
environment switches the current player after each move.

CurvyTron is different. A CurvyTron tick is simultaneous:

```text
all live players choose actions for tick t
the environment applies the joint action once
movement, trail growth, collisions, deaths, and rewards are resolved together
```

That is why "LightZero has self-play" does not end the wrapper question. The
future agent must check whether a LightZero self-play path can represent one
CurvyTron tick without changing the game. If it cannot, then CurvyZero must own
the mechanics around LightZero:

- collecting all player actions before one env step;
- preserving the joint action in telemetry;
- assigning per-player reward and terminal outcomes after the simultaneous
  resolution;
- choosing whether LightZero sees one ego row, many ego rows, or a custom
  multi-player adapter;
- making opponent policy identity explicit when other players are not directly
  controlled by the current LightZero policy call.

Owning mechanics does not mean rewriting MuZero. It means refusing to let an
adapter quietly change CurvyTron into a different game just because a framework
self-play API expects a different turn structure.

## What The Dummy Pong Work Proved

The dummy Pong work proved useful plumbing, not final learning quality.

Proven:

- A custom project env can be wrapped as a DI-engine/LightZero env.
- LightZero can control one ego action per step.
- The wrapper can supply the opponent action internally.
- The wrapper can build a joint action for the underlying two-player game.
- The wrapper can return `observation`, `action_mask`, and `to_play=-1`.
- LightZero can call real `train_muzero` on this custom setup.
- Training runs can write LightZero checkpoints and project summaries.
- Matched checkpoints can be strict-loaded for independent evaluation.
- Independent scorecards can be run outside the trainer.
- Target sidecars can expose MuZero root visit targets instead of forcing us to
  infer training labels from executed actions.
- Modal can run whole train/eval jobs and mirror useful artifacts into the
  `curvyzero-runs` volume.

That is meaningful. It means the external MuZero lane is not imaginary.

But the dummy Pong work also proved why mere plumbing is not enough.

Not proven:

- It has not proven reliable dummy Pong policy improvement.
- It has not proven that trainer-side returns are trustworthy by themselves.
- It has not proven that action diversity during collection means the learned
  policy target is healthy.
- It has not proven that the current single-ego dummy Pong wrapper is true live
  learner-vs-learner self-play. That is a statement about our wrapper, not a
  statement that LightZero lacks self-play.
- It has not proven a scalable CurvyTron actor loop.
- It has not proven a CurvyTron wrapper that preserves all multiplayer
  metadata needed for replay and audit.

The important negative evidence is action collapse. Several learned dummy Pong
checkpoints could be loaded and evaluated, but independent scorecards often
showed zero `down`, near single-action behavior, or poor held-out results. That
does not mean LightZero is fake. It means the custom setup can produce real
checkpoints whose quality is not yet trustworthy.

## What The Target-Sidecar Work Proved

The target-sidecar work is one of the biggest lessons for CurvyTron.

MuZero does not train the policy head toward the raw exploratory action that
was finally executed in the environment. It trains toward the MCTS root visit
distribution.

So an action histogram alone can mislead us.

For dummy Pong, a target replay smoke wrote rows that separated:

- executed action;
- child visit distribution / policy target;
- reward;
- done flag;
- config snapshot;
- episode and step identity.

That proved the right kind of observability is possible. It also exposed the
next missing piece: the sidecar still needs oracle or diagnostic labels so we
can ask whether the useful action got target mass.

For CurvyTron, this lesson matters even more. A CurvyTron wrapper can collect
millions of steps and still be unhelpful if MCTS systematically assigns little
or no target mass to survival-critical turns. The future agent should not ask
only "did the policy ever turn left/right/straight?" It should ask "did the
root target put mass on the action that the local diagnostic says matters in
this state?"

## What A Single-Ego CurvyTron Wrapper Would Mean

The likely first LightZero CurvyTron wrapper should be single-ego.

In plain terms:

- LightZero sees one ego row.
- LightZero chooses one ego action from the fixed action space.
- The wrapper supplies other players' actions from explicit opponent policies.
- The underlying CurvyTron env steps once with the full joint action.
- The wrapper returns the next ego observation, ego reward, done/truncated, and
  info.

This is similar to dummy Pong.

It is also deliberately not the same thing as proving that LightZero's native
self-play API handles CurvyTron simultaneous ticks. The wrapper hides the other
players behind opponent adapters. That can be okay for an initial bridge if the
metadata is honest.

The wrapper must record that setup clearly:

- ego player id;
- opponent policy ids;
- opponent checkpoint refs when used;
- joint action taken at each tick;
- all player terminal outcomes;
- observation schema id/hash;
- action schema id/hash;
- reward schema id/hash;
- ruleset id/hash;
- seed;
- episode id;
- reset profile or spawn profile;
- LightZero config snapshot;
- compiled support-scale fields;
- MCTS settings;
- checkpoint lineage.

Without that metadata, a run can be fast and still not interpretable.

## What Is Still Missing For CurvyTron

CurvyTron is missing the proof that the wrapper is honest.

Honest means:

- The training reward is the true sparse round outcome unless an experiment is
  explicitly marked as objective-changing.
- Survival, clearance, terminal cause, and timeout are telemetry unless a
  separate reward schema says otherwise.
- The wrapper does not silently turn simultaneous multiplayer into a false
  turn-based game just to fit a self-play API.
- `to_play=-1` is used only when LightZero is being treated as a single-agent
  ego controller.
- The opponent behavior is explicit, reproducible, and logged.
- Seat/perspective effects are evaluated, not hand-waved.
- A checkpoint is not called "best" unless the selection rule is visible.

CurvyTron is missing the proof that the wrapper is observable.

Observable means:

- Scorecards are independent of trainer logs.
- Target sidecars can show root visit distributions.
- Action histograms are reported for left/straight/right.
- Survival distributions are reported, not just win/loss.
- Terminal causes are reported.
- Truncation is separated from game termination.
- Per-player outcome data survives the wrapper.
- The future agent can reconstruct what the wrapper did on a row without
  reading the whole training process from memory.

CurvyTron is missing the proof that the wrapper is scalable.

Scalable means:

- The core actor loop can run without row-level JSON on every tick.
- Heavy sidecars can be sampled, gated, or run in diagnostic modes.
- Modal boundaries are whole jobs or sweeps, not per-step calls.
- Opponent inference does not accidentally become the dominant cost.
- Checkpoint-opponent loading is amortized.
- The optimizer can measure environment step, observation packing, policy or
  search, opponent action selection, action unmap, replay staging, reset, actor
  idle, learner idle, and policy staleness in one profile.

The missing point is not just speed. The missing point is speed after truth is
preserved.

## Why CurvyTron Is Harder Than Dummy Pong

Dummy Pong is a useful bridge because it is small and has clear actions.
CurvyTron is harder for several reasons:

- The environment is simultaneous by nature.
- Players leave trails, so history and terminal geometry matter.
- Survival may improve before wins appear.
- Multiple players make opponent policy identity more important.
- Seat and spawn distributions can change the difficulty of the same policy.
- Observation schema mistakes can look like learning failures.
- Reward-shaping mistakes can look like progress.
- Joint-action search becomes expensive quickly.

The safe first MuZero shape is probably not joint-action MCTS over every
player. For `N` players and three actions each, naive joint branching is
`3^N`. The safer first bridge is focal-agent MCTS: search the ego action space
and sample or script the other players from explicit policies.

That is a compromise, but it is an honest compromise if logged correctly.

## Exact Questions For A Future Agent

The next agent should answer these questions before recommending scale.

1. What exact CurvyTron adapter shape is being proposed?

Expected answer format:

```text
LightZero sees: one ego row / many rows / something else
Underlying env steps: one joint action per tick / something else
Opponent actions come from: random / scripted / frozen checkpoint / live policy
to_play value: -1 / player id / other
Reward returned to LightZero: true sparse outcome / shaped / other schema
```

2. Does the wrapper preserve simultaneous semantics?

The future agent should show where the ego action and opponent actions become
one joint action before the env step. If the wrapper pretends simultaneous play
is turn-based, it must say so and justify the approximation.

3. If using LightZero self-play, which self-play semantics are being used?

The future agent should name the mode instead of saying only "self-play":

- board-game-style alternating `to_play` self-play;
- single-agent rollout where the current policy plays the environment;
- single-ego wrapper with scripted or frozen opponents;
- row-based simultaneous adapter where multiple ego rows are gathered before
  one joint CurvyTron step;
- another custom integration.

The key check is whether one CurvyTron tick is still one simultaneous
CurvyTron tick after the adapter is applied.

4. What observation schema is used?

The current trainer-facing CurvyTron contract points at
`curvyzero_egocentric_rays/v0`, with `float32[106]`, action ids
`0=left`, `1=straight`, `2=right`, and sparse round-outcome reward. A future
wrapper should either use that contract or explicitly declare a new schema id.

5. Is the action mask correct?

The agent should verify:

- live row mask;
- dead row mask;
- terminal padding mask;
- strict left/right rules if relevant;
- LightZero `int8[3]` conversion in the same action order.

6. Is reward honest?

The agent should state whether the reward is:

- true terminal win/loss/draw;
- survival-shaped;
- contact/clearance-shaped;
- discounted;
- transformed for support-scale reasons.

If it is not true sparse round outcome, the run must be labeled as an
objective-changing ablation.

7. What opponent policy is used?

The answer must name the opponent source:

- random;
- scripted;
- frozen checkpoint;
- previous checkpoint pool;
- same live policy;
- external oracle;
- mixed distribution.

It should also state whether opponent identity is included in telemetry and
whether it is included in the learned observation. Fixed opponent per run is
different from mixed opponents inside one run.

8. What does independent eval measure?

Minimum useful scorecard:

- raw return;
- wins/losses/draws;
- survival mean, median, p90, max, and standard deviation;
- truncation rate;
- terminal cause counts;
- action histogram;
- action entropy;
- seat/perspective split;
- baseline comparisons;
- checkpoint lineage.

Trainer logs alone do not answer this question.

9. What target-sidecar fields exist?

Minimum diagnostic sidecar for a small run:

- executed ego action;
- root visit distribution;
- root value if available;
- searched value if available;
- reward;
- done/truncated;
- state hash or observation hash;
- seed;
- episode index;
- tick index;
- ego player id;
- opponent policy ids;
- config snapshot.

Better sidecar:

- local oracle action;
- oracle margin;
- target mass on oracle action;
- whether oracle action is top target;
- disagreement matrix between oracle, executed action, and target top action.

10. Is support scale actually compiled correctly?

The agent should not trust only requested config fields. It should report the
compiled LightZero fields that decide value/reward support size and scale. If
the run uses sparse `-1/0/+1` rewards but the compiled support is still broad
Atari-style support, the run should be marked config-suspect.

11. What is the scale mode?

The agent should label the run:

- import smoke;
- adapter contract smoke;
- target-quality diagnostic;
- small learning probe;
- scorecard probe;
- speed profile;
- serious training run.

Each mode has different telemetry needs. Import smokes can be tiny. Serious
training runs need scorecards and compiled config proof. Speed profiles should
sample or disable high-volume sidecars.

12. What would make the agent stop instead of scale?

Stop conditions should include:

- action mapping mismatch;
- wrong reward schema;
- missing terminal/truncation distinction;
- missing opponent policy telemetry;
- target assigns zero mass to obvious survival-critical actions;
- compiled support scale mismatches the claimed reward scale;
- independent scorecards show collapse while trainer logs look good;
- throughput profile is dominated by debug sidecars rather than real actor
  loop work.

13. What exactly is being claimed at the end?

The future agent should choose one of these claims, not blur them:

- "LightZero imported."
- "Adapter shape works."
- "CurvyTron wrapper can collect episodes."
- "Target sidecar exposes useful root labels."
- "Checkpoint strict-loads."
- "Independent scorecard ran."
- "Checkpoint improved over baselines."
- "Actor loop profile identifies the bottleneck."
- "This is ready for scale."

Most early runs should stop before the last claim.

## How Optimizer Should Use This Brief

Optimizer should use this brief as a boundary document.

The optimizer lane should not decide that LightZero is solved just because a
wrapper can step. It should ask whether the wrapper preserves the contract that
makes profiling meaningful.

Use this rule:

```text
Do not optimize a wrapper whose outputs cannot be independently audited.
```

That does not mean every speed run needs full JSON sidecars. It means the
wrapper must have a diagnostic mode that can prove the same code path is
honest before the speed mode turns heavy logging down.

Recommended optimizer sequence:

1. Confirm the adapter contract.
2. Confirm independent scorecard shape.
3. Confirm target-sidecar diagnostic mode on a tiny run.
4. Confirm compiled LightZero support/action/observation fields.
5. Profile the full actor loop with diagnostics off or sampled.
6. Optimize the largest real bucket.

The profile should include at least:

- environment step;
- observation packing;
- LightZero policy/search;
- opponent policy action selection;
- action unmap;
- replay staging/write;
- reset/autoreset;
- checkpoint/opponent refresh if present;
- actor idle;
- learner idle;
- policy staleness.

Optimizer should not treat source-fidelity or policy-quality claims as its own
deliverable. Those belong to environment and training/coach lanes. Optimizer's
job is to make sure the fast path is measuring the real fast path, not a debug
toy or a hidden invalid adapter.

## Practical Recommendation

Keep LightZero in the system, but keep it contained.

Use it for:

- external MuZero control runs;
- stock Atari/Pong sanity checks;
- custom dummy Pong target audits;
- first single-ego CurvyTron wrapper probes;
- checkpoint and scorecard discipline.

Do not use it as:

- proof that CurvyTron self-play is solved;
- proof that a simultaneous multiplayer contract is unnecessary;
- proof that trainer-side logs equal policy quality;
- a reason to skip repo-owned environment contracts;
- a reason to optimize before target observability exists.

The next good milestone is not "train CurvyTron big." The next good milestone
is:

```text
Run a tiny CurvyTron single-ego LightZero wrapper probe that has honest reward,
explicit opponent policy, preserved joint-action telemetry, independent
scorecard output, and root-target sidecar rows for a small diagnostic slice.
```

After that, Optimizer can profile the same actor-loop shape with heavy
diagnostics reduced. That is the path from "wrapper runs" to "wrapper can be
trusted and scaled."
