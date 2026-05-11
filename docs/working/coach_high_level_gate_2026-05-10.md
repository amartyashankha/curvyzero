# Coach High-Level Gate - 2026-05-10

Short critique for the current LightZero/Pong work. Lead with steps survived.
Do not lead with score.

## Plain Read

Pong is still a control path for CurvyTron, not the goal.
Keep it separate from CurvyTron adapter work: stock visual Pong survival-time
eval is the control lane.

The current runs are meaningful only when they answer this question:

Did a stock-reward visual Pong policy survive more stock-eval steps than its
own `iteration_0`, and did that improvement persist across later checkpoints
and multiple eval starts?

A return change from `-21` to `-20` is not enough. A one-checkpoint bump is not
enough. A single eval start is useful for triage, but not enough for a gate.
The first signal is stock evaluator steps survived. Score is secondary.

## Run Classes

| run class | signal it tests | meaningful now? | critique |
| --- | --- | --- | --- |
| Normal official/control Pong, fresh seeds | Whether stock-reward LightZero visual Pong learns to survive longer from ALE pixels. | Yes. This is the proof lane. | Keep judging by stock steps survived versus same-run `iteration_0`. Early `1000`-`5000` rows are triage only unless the gain persists later. |
| Longer normal runs, 131k/199k caps | Whether weak early survival bumps become sustained late-checkpoint survival gains. | Yes. This is the most important next read. | These runs matter more than more short seed churn. Check later checkpoints before declaring failure. |
| CPU16 versus CPU40 normal training | Whether resource allocation changes throughput or curve timing. | Yes, but only as labeled throughput evidence. | Do not mix CPU class into the learning claim unless the run ids and writeups keep it explicit. CPU class is not a new reward or env condition. |
| H100 normal long runs | Whether extra budget buys a clearer late survival curve. | Yes, but expensive. | Keep one or two as long-curve probes. Do not use H100 one-offs as the main proof if cheaper runs do not reproduce the signal. |
| Survival-shaped Pong | Whether adding per-step survival reward changes exploration or action collapse. | Side-lane only. | This cannot prove stock Pong learning. Useful telemetry, but it must not hide a broken stock-reward setup. |
| Controlled-repeat runs | Whether a result is reproducible under a controlled same start. | Diagnostic only. | Good for debugging sensitivity. Bad as the main generalization story. Training matrices should use varied recorded training seeds. |
| Single-start evals | Fast checkpoint triage. | Useful but weak. | A single eval start can pick candidates. Promotion needs many eval starts. |
| Multi-start evals | Whether the survival gain survives different eval starts. | Required for the next gate. | For each eval wave, sample a fresh pseudo-random eval seed list, record the generator seed and exact list, and use many starts before making a claim. |

## Current Meaning

The meaningful positive signal so far is survival-only: some normal runs survive
more steps at a later checkpoint than at `iteration_0`.

The current weak point is stability. Several runs bump at one checkpoint and
then fall back. Early Wave11 evals at `iteration_1000` for normal seeds `70`-`74`
did not show survival gain: across eval seeds `100`-`115`, stock steps survived
stayed around a mean of `760`. That is a useful negative/flat early read, not a
proof signal. Later `iteration_5000` and `iteration_7000` checkpoints exist and
are now being evaluated.

Stock return and positive reward count usually stay flat. That means the stack
is not yet proven solid.

Shaped runs are not meaningless, but they answer a different question. They can
say whether survival shaping changes behavior. They cannot say that the normal
stock-reward control path is ready.

## Next Passing Gate

Before moving the visual LightZero claim toward CurvyTron:

1. Pick normal stock-reward Pong runs only.
2. For each candidate, compare stock steps survived against that same run's
   `iteration_0`.
3. Require late-checkpoint gains, not just `1000` or one lucky `5000` row.
4. Require the gain to persist across later checkpoints or at least not collapse
   immediately at the next checkpoint.
5. Re-evaluate selected checkpoints with many eval starts.
   Use fresh pseudo-random eval seed lists for new waves, and record the RNG
   seed plus exact list. Fixed lists are replay/debug only.
6. Pass only if at least two normal runs show sustained stock survival gains
   over their own baselines.

The passing sentence should look like this:

Claim: two normal stock-reward Pong runs survived more stock-eval steps than
their own `iteration_0` at late checkpoints, and the gain held across multiple
eval starts.

Non-claim: this does not prove solved Pong, exact upstream reproduction, or
CurvyTron readiness.

## CurvyTron Training Blocker

CurvyTron scalar survival wrapper work is a contract check only. It can verify
the single-ego row shape, action mask, reward/done/info, reset/seed, final
observation, and sidecar metadata. Do not call an adapter smoke a full loop.

The next real CurvyTron training blocker is visual `[4,64,64]` stacking plus a
bounded collect/search/replay/sample/learner profile. That profile must say what
is included in the timed path before it can support a training claim.

## Kill Or Replace

Do not kill fresh normal proof-lane runs just because `1000` or `5000` is flat.
Wait for later checkpoints unless the run is broken.

Do pause new shaped launches until the normal lane has a clearer read. If
capacity is tight, replace shaped long/H100 work with normal stock-reward
long-curve work.

Do replace duplicate same-start seed chasing with varied training seeds. Same
seed repeats are allowed only when the purpose is reproducibility debugging.

Do replace single-start promotion claims with multi-start evals. One eval start
can choose candidates; it should not graduate a run.

## CurvyTron Reminder

CurvyTron should not inherit Pong score logic as its first reward.

First CurvyTron reward: survival time only. Reward the controlled player for
staying alive longer. End the episode when the controlled player dies.
