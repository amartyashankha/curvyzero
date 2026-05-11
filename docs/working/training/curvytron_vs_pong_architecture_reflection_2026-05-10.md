# CurvyTron vs Pong Architecture Reflection - 2026-05-10

No pytest was run.

## Short Answer

CurvyTron two-seat is harder than Pong/LightZero because the hard part is not
only "can one policy learn from survival reward?" The hard part is that two live
players act at the same time, share one policy, create each other's training
data, and need a clear rule for how collection, search, replay, update, and
evaluation fit together.

Pong is still useful because it proves the control lane can produce a survival
signal through the LightZero-style path. But that does not prove CurvyTron
two-seat self-play is solved.

## Why The Same Pathways Do Not Just Work

Pong has a simpler control shape:

- one ego policy decision is enough to define the main learning action
- the opponent can be treated as fixed, simple, or outside the core contract
- replay rows mostly mean "what did the ego see, do, and receive?"
- evaluation can ask whether one policy survives longer or plays better

CurvyTron two-seat changes that shape:

- both seats need decisions before the same `env.step`
- each seat's action changes the other seat's future data
- both seats may be rows for one shared policy update
- seat identity must be represented without accidentally creating two policies
- replay must explain which player, env row, iteration, and decision each row
  came from
- evaluation must compare survival over random seed sets, not chase one fixed
  seed

So the old path can carry some pieces, but it does not define the whole game.
Using the same learner call is not the same as having a mature two-seat
self-play system.

## Small Bugs

These are real bugs, but they are not the main architecture gap:

- target and batch shape issues that blocked the learner adapter from using the
  sampled two-seat rows cleanly
- missing or weak replay metadata needed to build discounted survival targets
- summary reporting that made current-iteration replay look accumulated
- evaluation/load wiring that had to prove strict checkpoint loads and sane
  survival numbers

Fixing these made the smoke lane honest. It means the bounded loop can collect
both seats, train the shared policy object, save checkpoints, and evaluate them.
That is a mechanical pass.

## Missing Architecture

These are not tiny bugs. They are missing architecture:

- a real two-seat collector/trainer contract that owns both player decisions
  before each simultaneous step
- a replay design that can accumulate and sample across time without becoming
  only an in-memory smoke table
- actor/learner weight revision rules, refresh cadence, and checkpoint lineage
- clear multiplayer search semantics: one searched ego, sampled opponent,
  joint-action search, or another documented approximation
- promotion gates against baselines, frozen checkpoints, same-run parents,
  held-out seeds, and swapped seats
- evidence that longer training improves survival distribution, not just that
  checkpoints load and numbers are stable

The current bounded two-seat loop is useful because it is small and testable.
But it is not LightZero's normal full trainer, not the upstream collector, not a
distributed self-play lane, and not a final answer for multiplayer learning.

## Why The Custom Adapter Exists

The blunt critique is fair: we chose a custom adapter because CurvyTron
two-seat did not fit the normal single-ego LightZero path cleanly, and we wanted
fast proof that one live policy could control both seats, write both seats to
replay, train on those rows, and produce checkpoints.

That choice was necessary as a smoke bridge. It was not a good final
architecture. The adapter let us see the truth faster, but it also carried
training pieces that LightZero already knows how to do better.

Now the custom lane should shrink. Keep only CurvyTron-specific code:
- simultaneous two-seat action ownership before `env.step`
- player/seat metadata and survival-time reward contract
- multiplayer eval panels with random seed sets and seat swaps

Move these back toward LightZero's normal path where possible:
- learner update scheduling
- replay buffer storage and sampling
- checkpoint management
- collector lifecycle and weight refresh
- priority, batching, logging, and trainer control flow

The adapter answered "is this mechanically possible?" The next architecture
should answer "what is the smallest custom multiplayer layer on top of the
normal trainer?"

## What The Latest Evidence Says

The corrected smoke eval loaded strictly and returned sane values. The larger
target-fixed scale run also trained, changed the model, wrote checkpoints, and
evaluated cleanly.

But the curve stayed flat: selected checkpoints sat around 193 to 197 mean
steps, with the final checkpoint only about 1.3 mean steps above `iteration_0`.

Plain read: the lane is alive, but it has not shown convincing CurvyTron
learning yet.

## Why Pong Still Matters

Pong is the control lane. It keeps us from confusing every CurvyTron failure
with a broken LightZero integration. If Pong can learn through the same broad
infrastructure, then CurvyTron's flat curve is more likely about multiplayer
contract, replay/search design, or scale than about the whole stack being dead.

Pong is not the CurvyTron gate. It is a reference instrument: it says the basic
pipes can carry signal, while CurvyTron tests the two-seat architecture.

## Next Decision Gate

Do not claim CurvyTron learning yet.

The next gate is: run longer corrected CurvyTron two-seat training with the
target-fixed eval path, then evaluate checkpoints on random seed sets and report
survival-time distributions.

Decision rule: if survival distributions improve clearly, scale the bounded
lane and design the durable self-play version. If the curve stays flat, stop
treating this as a bug hunt and decide the collector/search/replay architecture.
If results regress or depend on one seed panel, keep the gate closed.

The choice is whether the smoke lane is enough to scale, or whether CurvyTron
now needs a first-class multiplayer trainer instead of more patches around a
single-player-shaped path.
