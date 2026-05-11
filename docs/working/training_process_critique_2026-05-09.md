# Training Process Critique - 2026-05-09

Role: high-level process critic. This note uses the existing training docs only.
No code changes and no pytest.

## Bottom Line

Yes, we are over-focusing on fine-grained checkpoint analysis now.

The checkpoint work was useful. It found real facts: trainer-side telemetry can
mislead, old seed handling was suspect, dummy Pong checkpoints really collapse
under independent eval, `down` is legal, and weak MCTS root targets can omit the
right action. That work should not be thrown away.

But it should no longer dominate the main thread.

The main thread needs lane decisions, not another pile of per-checkpoint
detective work. Checkpoint forensics should be a worker tool used only when a
lane hits a clear gate. The main thread should stay on reproduction/parity:

- Can stock LightZero Atari Pong produce a small but real learning signal at a
  normal-enough setting?
- Can custom dummy Pong produce sane root targets and independent eval behavior
  before any longer training run?
- Where exactly does custom dummy Pong differ from stock LightZero assumptions?
- Which stopped dummy Pong branches must stay stopped?

## Current Pattern Problem

The work pattern has become too reactive.

One bad checkpoint leads to another probe, then another load check, then another
action histogram, then another target sidecar. Each probe is reasonable in
isolation, but together they pull attention away from the bigger question:
which lane deserves the next serious run?

The result is that the main thread can sound busy while still not making the
core decision. The docs already have enough evidence to stop several branches:
same sparse dummy Pong, update/replay-only dummy Pong, simple exploration-only
dummy Pong, and contact-pressure scaling. Reopening those through more
checkpoint detail is a distraction unless a root-cause lane changes the premise.

Simple rule:

```text
Use checkpoint analysis to decide a lane.
Do not let checkpoint analysis become the lane.
```

## Lanes That Should Dominate

### 1. Official Atari Pong Reproduction

This should be the highest-priority reproduction lane.

Why: it is the stock LightZero visual Pong path. It uses ALE/Gym, Atari
wrappers, frame stacking, the convolutional MuZero model, and the official Pong
action surface. It is the only lane that answers whether LightZero behaves in
its native visual Pong setting.

Current state: mechanics work. The GPU1024 L4 run gave a small real signal:
checkpoints through `iteration_4`, same-cap 256-step eval improved from `-5` to
`-3`, and one `+1` reward appeared. This is not solved Pong, but it is the best
current reproduction signal.

Continue if:

- it runs at a normal-enough search/update scale, preferably moving toward
  `25` simulations before `50`;
- it emits several post-init checkpoints;
- eval uses no fallback actions;
- later checkpoints differ from `iteration_0`;
- nonzero Atari rewards appear under a fair eval cap;
- reports keep saying plainly: this is official Atari Pong, not custom dummy
  Pong or CurvyTron.

Stop or downgrade if:

- it cannot run past toy `2`-simulation smoke settings;
- it only repeats short CPU rungs that prove mechanics again;
- it produces no checkpoint curve;
- it becomes a claim about custom dummy Pong or CurvyTron.

Main-thread decision: this lane gets the next meaningful reproduction budget.
Do not bury it under dummy Pong checkpoint minutiae.

### 2. Custom Dummy Pong Root-Target Parity

This should be the highest-priority custom-env lane, but it is not a scale lane.

Why: custom dummy Pong is the bridge to CurvyTron-owned envs. It tests adapter
contracts, sparse competitive rewards, opponent handling, scorecards, target
logging, and frozen-checkpoint opponent plumbing. It does not test stock visual
Atari parity.

Current state: plumbing works, but learning quality is blocked. Multiple runs
show action collapse. The key fact is target semantics: LightZero trains the
policy toward MCTS root visit distributions, not the exploratory action finally
executed. If a winning `down` state gets visits like `[1, 1, 0]`, more replay
of that target will not teach `down`.

Continue if:

- fixed scoreable states are used, including immediate-win states;
- root visits, policy logits, selected action, value, and oracle-best action are
  recorded at `2`, `8`, `16`, `25`, and maybe `50` simulations;
- at `25` or `50`, the known winning action gets nonzero visit mass and is
  usually ranked near the top;
- support/value scale is verified from the compiled policy, not only requested
  config text;
- the tiny replay sampling error is isolated as plumbing, not confused with
  policy quality.

Stop or do not scale if:

- known winning actions still get zero root visit mass at `25/50`;
- higher simulations only make the wrong action more confident;
- final held-out scorecards still have zero `down`;
- the only positive evidence is trainer-side action diversity;
- the proposal is another longer same-config dummy Pong run.

Main-thread decision: dummy Pong needs target/root parity first. It does not
get another broad training campaign until root targets pass on known states.

### 3. Stock-vs-Custom Discrepancy Map

This lane should stay active as a small critic lane.

Why: official Atari and custom dummy Pong are both called Pong, but they are
different worlds. The useful question is not "which Pong is real?" The useful
question is which assumptions transfer and which do not.

Known big differences:

- stacked visual frames and conv model vs tabular or single-frame flat raster
  MLP;
- official `50`-simulation Atari default vs custom smoke/debug settings;
- stock ALE single-agent env vs custom single-ego wrapper around a two-player
  toy env;
- stock evaluator/logging vs project scorecards and target sidecars;
- Atari reward/wrapper semantics vs sparse toy score reward plus shaped
  telemetry.

Continue if:

- it produces short transfer rules for the main thread;
- it says which official pattern should be copied next;
- it flags which custom result cannot be used as stock Pong evidence.

Stop or shrink if:

- it becomes another source tour;
- it repeats the same difference table without changing a decision;
- it tries to solve learning by documentation alone.

Main-thread decision: use this lane to prevent category errors. Do not let it
become a third main campaign.

### 4. Reporting And Eval Hygiene

This lane should be constant but lightweight.

Why: many past mistakes came from mixing trainer telemetry, independent eval,
different seats, changed horizons, best-checkpoint selection, and run lineage.

Continue if every report includes:

- lane label: official Atari Pong or custom dummy Pong;
- run id, attempt id, checkpoint ref, and eval cap;
- whether actions came from trainer-side collection, trainer evaluator, or
  independent scorecard;
- no-fallback status;
- action histogram, raw score, survival/loss-delay telemetry where relevant;
- stop/continue decision.

Stop if:

- docs copy large logs instead of making a decision;
- score wins are reported without action/survival context;
- trainer-side sidecars are described as final held-out quality.

Main-thread decision: this is the guardrail. It should make decisions clearer,
not add more front doors.

## Lanes That Should Stay Stopped For Now

Same sparse dummy Pong longer runs: stopped. Pure 2x did not improve held-out
survival, shaped return, raw score, or action entropy.

Update/replay-only dummy Pong: stopped. UPC25 did not fix held-out quality.

Simple exploration/data-distribution under the same sparse target: stopped.
Trainer-side diversity improved, but held-out quality did not.

Contact-pressure campaign: stopped as a campaign. The tiny and modest rungs
passed mechanically but did not produce a clean held-out policy-quality win.

Raster-flat as a learning claim: stopped for quality claims. It is a mechanical
bridge only until it has history/velocity or a clearer visual contract.

Survival scaling: side diagnostic only. It can catch regressions, but it should
not be treated as the main learning milestone.

Board-game smokes: controls only. They show LightZero can run sparse examples;
they do not prove Pong, dummy Pong, CurvyTron, or visual learning.

## What To Delegate Next

Delegate official Atari scale to a worker:

- prepare one bounded official Atari run closer to normal LightZero scale;
- prioritize `25` simulations if mechanically feasible;
- keep a fair same-cap baseline;
- report checkpoint curve, action histogram, return, nonzero rewards, and
  no-fallback status;
- do not edit the main orientation docs until the run has a clear readout.

Delegate custom root-target parity to a worker:

- build or run the fixed-state target probe;
- include known scoreable states, especially immediate-win `down` states;
- sweep simulations `2/8/16/25/50`;
- record root visits, logits, selected action, value, oracle action, and tie
  rate;
- return a pass/fail table, not a narrative.

Delegate support/value-scale verification to a worker:

- inspect the compiled LightZero policy/model config for actual support scale;
- compare it to requested reward/value support ranges in summaries;
- report whether support ablations are trustworthy.

Delegate discrepancy-map maintenance to a worker:

- keep one short official-vs-custom transfer note current;
- mark which custom results are plumbing, which are diagnostics, and which are
  learning evidence;
- remove duplicate explanations from new summaries.

Delegate docs hygiene to a worker:

- update the state index and experiment backlog only after lane decisions;
- keep stopped branches visible but closed;
- prevent the main thread from re-litigating old dummy Pong rungs.

## Main Thread Operating Rule

The main thread should ask four questions, in this order:

1. What lane is this?
2. What decision will this run or doc change?
3. What is the stop rule?
4. Who can do the detailed work away from the main thread?

If a proposed task cannot answer those questions, it is probably a distraction.

## Short Reset

Do less checkpoint archaeology in the main thread.

Push official Atari Pong reproduction forward, because it is the real stock
visual control. Keep custom dummy Pong, but make it prove target/root parity
before giving it more scale. Use discrepancy mapping and reporting hygiene as
guardrails. Delegate probes and doc cleanup. Keep the main thread focused on
lane decisions.
