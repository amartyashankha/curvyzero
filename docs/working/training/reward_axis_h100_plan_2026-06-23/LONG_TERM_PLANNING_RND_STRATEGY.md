# Long-Term Planning And RND Strategy - 2026-06-23

Status: operating doctrine. No jobs were launched from this note.

## Short Answer

Long-term planning should enter CurvyTron as a layered program, not as pure
raw-tick MCTS.

The most believable stack is:

1. Train a strong recurrent policy/value system first.
2. Give it observations and auxiliary targets that expose territory, corridors,
   collision horizon, opponent motion, and reachable space.
3. Use action repeats or macro-actions so explicit decisions cover meaningful
   geometry.
4. Add dense GPU-friendly trajectory planning over macro-actions before
   spending more effort on pointer-shaped raw-tick trees.
5. Use search/reanalysis/distillation only where it improves policy quality at
   equal wall-clock and action-latency budgets.

RND belongs beside this as training-only exploration pressure. It should make
data collection visit useful unfamiliar states, not become the game objective.
Every positive-RND row needs stock and `rnd_meter_v0` controls, survival
retention, action-collapse checks, and later fixed-opponent or tournament
transfer before promotion.

## Planning Is More Than Search

For CurvyTron, "long-term planning" has five separable mechanisms:

| Mechanism | Role | Current posture |
| --- | --- | --- |
| Recurrent policy/value | Remember partial observations and encode strategy across seconds. | Needed in PPO/Puffer lane; MuZero already has value targets but current source-state default is still raw-tick tight. |
| Strategic observations | Show global territory, local collision geometry, opponent state, trail age/gaps, and reachable corridors. | Flash/raycast and source-state visual evidence are controls, not final design. |
| Auxiliary targets | Predict time-to-collision, reachable free space, future occupancy, opponent action, and survival horizons. | Prefer as representation learning before dense reward shaping. |
| Macro-actions/action repeat | Make one decision cover 2/4/8/12 physics ticks instead of one tiny arc. | Needs its own manifest/profile; do not mix with current Wave A denominator. |
| Explicit planner | Beam/CEM/MPC/Gumbel/MuZero search over macro-actions or selected states. | Research lane only until it beats policy-only controls at equal GPU-seconds and latency. |

This makes pure raw-tick MCTS a poor default, but not a reason to abandon
planning. Search should return as measured macro-planning, selected-state
reanalysis, or joint-action/Gumbel MuZero after simpler baselines have earned a
comparison.

Current repo reality matters: the believable production-ish learning path is
still source-state fixed-opponent LightZero/MuZero, not true simultaneous
current-policy self-play. The defaults are also raw-tick tight: one source frame
per decision, action repeat `1`, and shallow search. That makes action-repeat
or macro-action sweeps the highest-leverage planning bridge, but any such sweep
changes the denominator and must be labeled separately from Wave A.

## Actual Game Constraints

Do not let generic RL abstractions erase the game. CurvyTron is curved
real-time geometry with persistent topology:

- Source physics targets 60 Hz. Base speed is `16` units/s, angular velocity is
  `2.8 / 1000` radians/ms, avatar radius is `0.6`, and the natural full-turn
  radius is about `5.7` world units.
- A one-tick action changes heading by only about `2.7` degrees and moves about
  `0.27` world units. Raw-tick tree depth is therefore mostly tiny arc
  refinement, not strategic lookahead.
- Trail printing starts after the round begins and holes are distance-based, not
  tick-based. A planner must know whether a future arc leaves collidable trail,
  visual trail, or a gap.
- Collision is endpoint-circle/source-style body lookup, not a swept continuous
  collision test. Macro-actions must internally replay source frames; they
  cannot jump directly to the macro endpoint without changing death semantics.
- Same-avatar self-collision has trail latency. Fresh body points and old body
  points are not equivalent.
- Bonuses can change speed, radius, inverse controls, printing, border wrapping,
  and trail clearing. Natural bonus support is partial/contracted, so bonus
  experiments must say which effect set is actually active.
- Current source-state training observes stacked gray64 visual frames plus masks
  and info, not privileged perfect state. Any planner using hidden state is a
  privileged teacher/evaluator unless the deployed policy receives equivalent
  information.
- Current fixed-opponent training supplies the opponent action from configured
  scripted/frozen/mixture policies. It is not true simultaneous current-policy
  self-play, even though the underlying step applies a joint action.

Game-shaped planning should therefore reason about:

- time-to-wall and time-to-body along feasible curvature
- whether a turn creates or preserves escape corridors
- reachable free space under a Dubins-like turning radius, not ordinary flood
  fill alone
- opponent relative heading, speed, and potential cut lines
- gap/printing state and bonus-induced topology changes
- action latency and how much source-frame replay a planner can afford

The old local toy-v0 simulator is useful for plumbing, not for deciding this
strategy. It has different simplified geometry, action repeat, grid occupancy,
and no-bonus assumptions. Use source-state/vector-runtime contracts and golden
scenarios for planning or PPO/Puffer parity.

Use `CURVYTRON_GAME_MECHANICS_GATES.md` as the promotion checklist for any
branch that changes environment runtime, control cadence, observation, reward,
planner rollout, or self-play topology.

## How RND Fits

Current credible RND path:

- source-state visual LightZero training
- `none`, `rnd_meter_v0`, and `rnd_replay_target_v0`
- feature source `policy_gray64_latest/v0`
- stock LightZero reward-model entrypoint for RND rows
- compact optimized trainer remains no-RND

RND should be used for three questions:

1. Does novelty pressure improve early exploration and survival AUC?
2. Does the effect survive beyond blank-canvas/noop opponents?
3. Does it help policies discover strategic states that a later recurrent
   policy or planner can exploit?

RND should not be used to claim:

- leaderboard strength from a blank-canvas result
- compact-trainer reward support
- policy-quality gain when meter rows diverge from stock
- a planning win when only intrinsic reward metrics improved

RND is most likely to help CurvyTron by forcing visits to unfamiliar trail
topologies, near-wall states, narrow corridors, gap transitions, opponent
approach geometries, and bonus-contact situations. It can also fail by rewarding
mere visual novelty: spinning, chasing bonuses, painting unusual trails, or
dying in new ways. For that reason, RND features should be audited for whether
they see strategically useful novelty rather than frame noise or cosmetic
variation.

## Evaluation, Improvement, Amortization

Every planning or exploration claim should identify which operation it improved:

| Operation | Question | CurvyTron examples | Common self-deception |
| --- | --- | --- | --- |
| Evaluation | How good is this state/action? | Value network, rollout score, MCTS return, time-to-collision estimate. | A better evaluator is claimed as a better policy before action selection or training changes. |
| Improvement | How do we choose better actions now? | PPO update, planner argmax/sample, search policy, action-repeat choice. | Search chooses better actions but the network never learns them. |
| Amortization | How do future decisions become cheap? | Distill planner into policy, train value on reanalysis, checkpoint export. | Expensive planner win is reported as deployable inference without distillation cost. |

PPO is mainly experience plus stable policy improvement. MCTS is mainly
inference-time evaluation and improvement. AlphaZero/MuZero-style loops work
because they cycle evaluation, improvement, and amortization. RND is neither
planning nor improvement by itself; it changes the experience distribution and
must be judged by downstream extrinsic quality.

For CurvyTron, hybrid short-horizon planning is a plausible sweet spot, not a
law. It must earn that label by showing tactical collision/corridor improvement,
then amortizing that improvement into a cheaper policy or durable checkpoint.

## Combined Architecture

The clean combined program is:

1. **Wave A stock-ish LightZero reward/RND**
   Use the prepared bestseed long-tier plan only after rerunning the launch and
   capacity gates. Keep RND low/mid positive weights, stock and meter controls,
   and the non-RND bestseed triad alive. This is the current canonical learning
   campaign.

2. **Policy-first baseline lane**
   Build a recurrent PPO/Puffer-style lane with fixed buffers, frozen-bank
   self-play, action masks, and separate PPO speed/quality ledgers. RND can be
   added there only after the no-RND recurrent baseline learns and exports
   evaluable checkpoints.

3. **Temporal-abstraction lane**
   Sweep action repeat or macro-action durations under matched controls. This
   tests whether the policy needs fewer, more meaningful decisions before we add
   explicit search.

4. **Dense planner lane**
   Compare policy-only, full small-depth enumeration, beam/CEM/MPC, and
   Gumbel/MuZero-style search at equal GPU-seconds and action latency. Planner
   outputs are imitation/reanalysis targets, not PPO old-policy samples.

5. **Transfer and selection**
   Preserve best checkpoints, run fixed-opponent evaluation, then tournament
   exposure after nonzero policies exist. Leaderboard feedback remains late.

## Hybrid Planning Contract

Hybrid short-horizon planning means: search a few meaningful steps, then let a
learned value estimate the tail. For horizon `H`, evaluate each candidate action
by rollout reward plus `gamma^H * V(s_H)`, then either choose the best action or
sample from a temperature-scaled planned policy.

This creates a spectrum:

| Method | Inference | Training contract |
| --- | --- | --- |
| Pure PPO/self-play | Sample or choose from the policy. | PPO/actor-critic on behavior actions and rewards. |
| PPO with value | No inference-time planning. | Value improves advantages, but actions still come from policy. |
| Short rollout planning | Sample/enumerate a few continuations, score with reward plus value bootstrap. | Planner output is a separate behavior/improvement signal. |
| MCTS | Adaptive tree allocates simulations to promising actions. | Search policy/value targets, not vanilla PPO samples. |
| AlphaZero/MuZero | Neural policy/value/model inside tree search. | Train network from search-improved policy/value/reward targets. |

Use PPO/self-play when inference must be cheap, rollouts are plentiful, the
simulator exists but branching is large, and exact equilibrium guarantees are
not required. Use imitation when a planner, expert, or prior checkpoint can
bootstrap a reactive policy. Use hybrid short-horizon planning when pure PPO is
too myopic, pure MCTS is too expensive, the value function is decent, and the
latency budget allows a small local search. Use MCTS/MuZero-style search when
state is sufficiently observable, branching is manageable with priors, tactical
lookahead matters, and inference compute is acceptable.

CurvyTron points toward hybrid planning over macro-actions first: branching is
small per player but raw ticks are too fine, tactical collision lookahead
matters, and the long tail should be carried by a learned value/recurrent
state.

## PPO Plus Planning Rules

Planner hybrids are useful only if the training math is honest:

| Pattern | Allowed interpretation |
| --- | --- |
| Planner actions plus imitation | AlphaZero-style or supervised distillation. Not PPO. |
| PPO plus planner auxiliary loss | PPO remains on behavior-policy samples; planner targets add an auxiliary term. |
| Planner as critic or advantage estimator | Policy improvement / advantage-weighted regression / stronger critic. Requires clear target semantics. |
| Planner actions inside PPO rollout | Dangerous unless the planner distribution and logprob are the recorded behavior policy. |
| MuZero-style learned model plus search | Separate search-target training objective. Not a drop-in PPO variant. |

If actions came from `pi_plan`, the behavior policy is `pi_plan`. Vanilla PPO's
ratio against `pi_old` is wrong unless `pi_plan` is actually the sampled old
policy or its logprob is recorded and used correctly. Search actions can still
be valuable, but the bridge is imitation, reanalysis, off-policy correction, or
a custom search-training objective.

Every PPO/planner/hybrid manifest should declare
`behavior_policy_contract = ppo | imitation | off_policy_mixture | search_training`.
If it cannot declare that field honestly, it is a research scratchpad, not a
believable training run.

## Two-Player And Imperfect-Information Pitfalls

CurvyTron is not a clean alternating perfect-information board game. Even when
the simulator state is known to the trainer, policies may observe partial visual
state and opponents act simultaneously. Guard against:

- wrong reward sign: train each action from the acting player's perspective
- value-perspective bugs: know whether `V(s)` is current-player, ego-player, or
  fixed-player value
- forgotten-strategy overfit: use frozen checkpoints, opponent pools, and
  exposure matrices
- illegal-action/logprob mismatch: masks must be applied consistently for old
  and new policy probabilities
- deterministic collapse: keep entropy/diversity checks for mixed-strategy
  settings
- information leakage: do not let planners use hidden state that the deployed
  policy will not have unless the result is explicitly privileged training or
  imitation data
- search-improvement confusion: a planner choosing a better move is not network
  learning until distillation, value learning, policy gradient, or replay
  training happens

## Experiment Orchestration

Run the program as embarrassingly parallel lanes with one central captain
ledger:

| Lane | Owner question | Parallel work units | Captain gate |
| --- | --- | --- | --- |
| Wave A reward/RND | Which reward/RND settings produce retained extrinsic quality? | RND stock/meter/positive rows, non-RND triad, cadence/support rows. | Packet audit, capacity tier, healthy controls, and horizon readout. |
| Checkpoint anchors | What is the best seed/opponent source we can actually use? | Eval-best audit, tournament-best audit, Modal existence audit, immutable-ref report. | One seed policy and one opponent-ref file per launch family. |
| PPO/Puffer baseline | Can a recurrent model-free lane learn faster or cleaner? | ABI spike, parity traces, raw env benchmark, one-update profile, export/eval bridge. | No production spend until build/parity/profile are artifacted. |
| Temporal abstraction | Do fewer, more meaningful decisions help? | Action-repeat 2/4/8/12 sweeps and macro-action reward-accounting tests. | Label denominator change; compare quality and latency, not env/s only. |
| Dense planner/search | Does planning improve decisions enough to pay for itself? | Policy-only, model-only, tree-only, beam/CEM/MPC, Gumbel/MuZero selected-state profiles. | Equal GPU-seconds/action-latency comparison and profile split. |
| Monitoring/forensics | Are runs healthy and are claims honest? | Status snapshots, RND metrics, eval curves, action histograms, GIF review, artifact integrity. | Stop/pivot only by `MONITORING_SIGNALS.md` and `CONTINGENCY_PLANS.md`. |

External agents should receive bounded packets:

- exact lane question
- files/artifacts to read
- rows or run ids in scope
- what not to infer
- expected output format
- whether they may edit files

The captain keeps the cross-lane ledger, chooses launch priorities, enforces
H100 tier limits, and prevents one lane's attractive result from closing another
lane's question. If agents disagree, resolve by source contract, manifest,
artifact, and fresh audit, in that order.

## Parallelism At Every Level

Default to embarrassing parallelism when the lane has a credible launch surface:

1. **Across hypotheses**
   Keep reward/RND, non-RND reward controls, checkpoint-anchor audits,
   temporal-abstraction smokes, bounded PPO/Puffer baseline work, and planner
   profiling moving independently.

2. **Inside each hypothesis**
   Use rows for replicas, low/medium/high weights, stock/meter controls, reward
   arms, opponent recipes, and cadence knobs. Prefer one audited manifest over a
   serial chain of handcrafted one-offs.

3. **Across time horizons**
   Read health in the first 30 minutes while longer rows continue. Do not let a
   short health canary become an accidental serialization point for a whole
   campaign.

4. **Across agents**
   Delegate bounded questions: RND metrics, checkpoint anchors, planner profile
   design, Puffer feasibility, tournament exposure, and fake-progress critique.
   Each agent should return artifacts, references, and caveats, not a global
   decision.

5. **Across monitoring**
   Prepare status, eval, RND metrics, action-collapse, artifact-integrity, and
   capacity checks as independent readouts. One dashboard should never be the
   only view of a run.

The constraint is interpretability. Parallel experiments are useful only when
each row has a named control, a first decision horizon, and a stop/pivot rule.
For long runs, parallelism narrows by tier: 2-8 hour runs stay at or below 40
H100s, and 8h+ runs stay at 10-20 H100s unless the operator explicitly changes
the campaign shape.

Red-team rule: the safest successful Wave A claim is deliberately small. Say
"this seed/curriculum/eval stack shows retained survival lift versus stock and
meter and deserves a fixed-opponent bridge," not "we found the reward" or "RND
works." Survival AUC, bestseed initialization, and long17 capacity shaping are
all useful but easy to overread.

## Initial Experiments

Do not start by rewriting the architecture. Start with isolated rows:

| Experiment | Purpose | First useful signal |
| --- | --- | --- |
| Current long17 bestseed Wave A | Stock-ish reward/RND evidence with 8h+ cap discipline. | Health in 30 min; useful read 100k-170k; retention 240k-300k. |
| RND meter plus low-positive bridge | Prove RND metrics, target reward mutation, and low-weight behavior. | Metrics in 30 min; stock/meter parity and low-weight AUC by 50k-170k. |
| Action-repeat/macro smoke | Test whether 2/4/8 tick decisions improve quality or speed without search. | Same-work caveat immediately; survival/AUC after enough checkpoints. |
| Recurrent PPO/Puffer feasibility spike | Establish a serious model-free baseline independent of MuZero. | Build/parity first; raw env/rollout/update split; policy beats random/scripted. |
| Dense planner selected-state benchmark | Price beam/CEM/MPC versus policy-only on fixed state batches. | Latency distribution, GPU util, and chosen-action value lift. |

The action-repeat and PPO/Puffer rows are separate denominators. They should
not change the interpretation of Wave A unless they are explicitly designed as
bridge experiments.

## Signals

Planning is working only if:

- policy-only and planner rows are compared at equal GPU-seconds or action
  latency
- selected actions improve fixed-state survival/value estimates
- eval survival AUC or head-to-head outcomes improve after distillation
- planner latency distribution fits the real-time control budget
- the win persists without reading raw trainer reward across variants

RND is working only if:

- `rnd_meter_v0` behaves like stock
- low positive weights beat both stock and meter on survival AUC or best-so-far
- latest remains close enough to best, not just a one-off spike
- predictor loss and intrinsic scale are finite and nontrivial
- action histograms do not collapse into novelty chasing
- fixed-opponent extension preserves the blank-canvas gain

## Failure Modes And Contingencies

| Failure | Read | Contingency |
| --- | --- | --- |
| RND metrics improve but survival is flat | Curiosity is measuring novelty, not useful exploration. | Keep meter/stock, lower weight, or move RND to auxiliary diagnostics only. |
| High RND weights win early | Intrinsic scale may swamp extrinsic objective. | Prefer low weights; require retention and GIF/action checks. |
| Meter diverges from stock | RND instrumentation is not passive. | Treat all positive rows as uninterpretable until fixed. |
| Macro-action rows look faster | Denominator changed. | Compare quality and action latency, not env/s only. |
| Planner beats policy on selected states but not training | Distillation/reanalysis path is weak. | Use planner as eval/control or improve imitation targets before online use. |
| Raw-tick MCTS remains slow after sync fixes | Sequential tree structure is the bottleneck. | Shift effort to macro/dense planning. |
| PPO/Puffer learns fast but transfers poorly | Baseline exploited simplified reward/opponents. | Add frozen opponents, eval exports, and tournament only after nonzero checkpoints. |

## Documentation Links

Use these docs as the active map:

- RND implementation and gates: `RND_LANE.md`
- Source-fidelity gate checklist: `CURVYTRON_GAME_MECHANICS_GATES.md`
- Long-horizon static curriculum and RND bridge: `LONG_HORIZON_CURRICULUM.md`
- Reward variants: `REWARD_INVENTORY.md`
- Run signals and timing: `MONITORING_SIGNALS.md`
- Capacity and stop rules: `OPERATING_PATTERNS.md`
- Critical external-context review: `CONCEPTUAL_LANDSCAPE_REVIEW_2026-06-23.md`
- Puffer/PPO gap: `../curvytron_flash_comparison_2026-06-23/PUFFERLIB_STRATEGY.md`
- Flash/raw-env controls:
  `../curvytron_flash_comparison_2026-06-23/COMPARISON_MATRIX.md`

The operating rule is simple: use many H100s when the lanes are interpretable,
but keep the claims narrow. Curiosity, recurrence, macro-actions, PPO, and
search can all help, but only if their evidence stays labeled.
