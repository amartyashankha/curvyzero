# Conceptual Landscape Review - 2026-06-23

Status: critical synthesis. No Modal jobs were launched from this review.

## How To Read This

This note reviews two external-context piles:

- `external/chatgpt/context.md`
- `docs/working/training/curvytron_flash_comparison_2026-06-23/`

Treat both as evidence and critique, not as authority. Current repo contracts,
manifest artifacts, and launch auditors remain the launch source of truth.

Standing decision: this review does not replace the current Wave A plan. It
reinforces the stock-ish LightZero reward/RND campaign, the bestseed non-RND
repair posture, and the rule that Flash/PPO/raw-env evidence must not be mixed
with CurvyZero MuZero full-loop or learning-quality claims.

## Source Fit

The external memo is directionally useful on game shape. CurvyTron really has
short-horizon collision reflexes, long-horizon territory/enclosure pressure,
opponent-action dependence, and hidden or optional stochasticity. Its advice to
separate simulator mechanics, policy learning, search, and planner semantics is
sound.

But it is not a repo audit. Its Puffer/PPO and dense-planner recommendations are
architecture proposals, not evidence that the current source-state LightZero
Wave A should stop. It also makes claims about legacy update order and current
library capabilities that need current-code verification before they become
implementation facts.

The Flash packet is useful in a different way. It gives a recovered PPO/raw-env
control system and a clean denominator vocabulary:

- Flash raw environment throughput is a mechanics ceiling/control.
- Flash PPO profile throughput is a PPO control.
- CurvyZero speed proof is only same-work H100 full-loop evidence.
- CurvyZero reward/RND proof is policy-quality evidence, not raw throughput.

The Flash packet therefore strengthens our claim hygiene. It does not prove that
CurvyZero should port Flash wholesale or replace MuZero/LightZero with PPO today.

## Repo Reality Check

The active believable campaign is not true simultaneous two-seat self-play. The
source-state fixed-opponent env declares `two_seat_self_play=False`, takes an ego
action, supplies the opponent action from a configured opponent policy, and
steps a two-player joint action internally.

The repo also contains a source-state joint-action adapter, but it is explicitly
centralized 9-action joint control and marked not true competitive self-play. It
is a diagnostic/research surface, not the current production training topology.

Current default timing is raw-tick tight:

- `CURVYTRON_DECISION_SOURCE_FRAMES = 1`
- source-state default policy action repeat min/max is `1`
- `CURVYTRON_DEFAULT_NUM_SIMULATIONS = 8`

That means the memo's time-scale critique is relevant. Current Wave A should be
read as reward/RND/cadence evidence under the present one-source-tick contract,
not as evidence that raw-tick search is the final architecture.

## What Changes Now

No launch decision changes. As of the capacity artifacts referenced in
`LAUNCH_QUEUE.md`, the capacity-clear candidate remains the long-tier bestseed
staged profile recorded in that queue and the approval packet. Rerun capacity
before any launch. The staged profile keeps:

- RND `none` and `rnd_meter_v0` controls
- low and medium positive RND weights
- non-RND bestseed static/long-horizon/cadence triad
- the 8h+ cap discipline of 10-20 H100 rows

What changes is the interpretation posture:

- Survival reward can be a useful learning scaffold and a passivity trap.
- RND can improve exploration metrics without improving extrinsic quality.
- Policy-only, PPO, MuZero search, dense planner, and raw environment rows are
  different denominators.
- Search must prove policy improvement at equal GPU-seconds/latency before it
  earns production status.
- Leaderboard/tournament remains later; it should select nonzero checkpoints,
  not drive the first Wave A curriculum.

## Reward And RND Implications

The external memo's warning about dense survival reward is credible. A per-tick
alive helper can reward passive survival, loops, or opponent avoidance. That does
not make the current reward sweep wrong; it means the sweep must be judged by:

- survival AUC
- best-so-far versus latest retention
- action collapse and passive-loop diagnostics
- fixed-opponent/eval health
- eventual tournament exposure after nonzero checkpoints exist

The current RND plan remains the most believable RND implementation:

- source-state visual LightZero path
- `none`, `rnd_meter_v0`, and `rnd_replay_target_v0`
- RND metrics required only for enabled rows
- positive RND judged against stock, meter, and independent non-RND controls
- no compact RND claim

Do not call blank-canvas RND success a game-strength result. A positive blank
result only promotes the next experiment: fixed-opponent RND bridge rows or
replicated low-weight rows with the best non-RND controls alive.

## Search And Planner Implications

The memo's strongest systems point is that raw-tick tree search may be the wrong
accelerator shape. MCTS can be GPU-host-synchronized by bugs, but it can also be
structurally under-batched or irregular even after synchronization bugs are
removed.

Future search work should therefore require profile splits before promotion:

- policy only
- recurrent/model only
- tree only
- full search

Record root batch, simulation count, max depth, kernel count, D2H/H2D copies,
GPU active percentage, traversal depth, allocator churn, and action latency
distribution. A speed row without those labels is support evidence only.

Dense macro-action planners, PPO/Puffer baselines, recurrent structured-state
policies, local crops, reachability auxiliaries, and reanalysis/distillation are
plausible future branches. They are not Wave A blockers. Add them only through
separate manifests and ledgers so their wins do not contaminate reward/RND
claims.

The combined doctrine now lives in
`LONG_TERM_PLANNING_RND_STRATEGY.md`. Use that note to decide how recurrence,
macro-actions, dense planners, and RND should compose before creating new
launch rows.

## Flash Implications

The Flash packet gives useful controls:

- raw H100 grid mechanics around `161.55M env/s`
- raw H100 raycast observation control around `15.65M env/s`
- one diagnostic PPO profile around `438k agent_steps/s`

Those rows are not comparable to the CurvyZero OPT-104 same-work H100 baseline
or the current compact support rows. The actionable lesson is organizational:
keep a first-five-minute map, label denominators, preserve playable/export
feedback loops, and keep raw mechanics under `artifacts/local/flash_controls/`.

The Flash controls are strongest where they are narrow:

- raw benchmark rows reported `grid_overflow=0`
- the raycast row usefully prices observation extraction
- the PPO profile is a one-update diagnostic, not production PPO
- `env_step_s` includes observation generation in that profile
- CUDA graph capture was not an automatic win in the fresh raw grid row

The weak Flash claims should stay weak:

- recovered `H100:8` PPO/DDP speed is a README/worklog claim until rerun with
  DDP summary and rank logs
- checkpoint/export existence is not policy-quality evidence
- accelerated parity is useful but not full exact final-hash parity with the JS
  reference
- "GPU-resident" still permits host-visible done/reset and metric syncs

Do not port Flash wholesale unless a future branch has an explicit objective,
such as a recurrent PPO baseline or playable policy export bridge. Even then,
run it as a separate algorithmic lane, not as a CurvyZero reward/RND result.

## Claims To Verify Before Believing

- Current CurvyZero simultaneous semantics and any legacy update-order mismatch.
- Whether PufferLib's current self-play/runtime claims fit CurvyTron today. The
  current source-read position is in
  `../curvytron_flash_comparison_2026-06-23/PUFFERLIB_STRATEGY.md`, but any
  integration still needs a build/train smoke in the target runtime.
- Whether macro-action/action-repeat control improves quality under the current
  source-state env rather than merely changing the denominator.
- Whether search improves the same checkpoint over policy-only at equal
  GPU-seconds and equal action latency.
- Whether dense survival reward teaches active territory control or passive
  survival loops.
- Whether any RND weight improves extrinsic survival retention after stock and
  meter controls are healthy.

## Operating Decision

For the next launch window, keep the campaign boring and auditable:

1. Do not launch from this conceptual review.
2. Use `LAUNCH_QUEUE.md` and the approval packet for any actual Wave A launch.
3. Prefer the long17 bestseed profile while capacity is tight.
4. Keep RND and non-RND controls together.
5. Interpret early health only as health.
6. Let long rows run long enough before calling no-signal.
7. Preserve best checkpoints separately from latest checkpoints.
8. Treat PPO/Flash/dense-planner work as future separate branches.

This is the clean middle path: aggressive experiments now, but no algorithmic
identity theft between PPO, MuZero, raw env, compact speed, RND, and reward
quality.
