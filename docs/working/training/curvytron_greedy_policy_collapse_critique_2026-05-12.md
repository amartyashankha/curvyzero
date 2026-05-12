# CurvyTron Greedy Policy Collapse Critique - 2026-05-12

## Facts

- Training collection is not the same path as the GIF trace. The trainer defaults
  to `collect` action selection with `temperature=1.0`, `epsilon=0.25`,
  no-op-skip disabled, and observation noise `0.10`
  (`curvytron_two_seat_lightzero_train_smoke.py:115-123,166-179`).
- The collection path calls `MuZeroPolicy.collect_mode.forward` with temperature
  and epsilon, then records fresh-policy action counts by player
  (`curvytron_two_seat_lightzero_train_smoke.py:1472-1481,1902-1984`).
- Skipped policy chances execute NOOP and do not create replay rows/reward targets
  (`curvytron_two_seat_lightzero_train_smoke.py:1556-1603,1707-1721,1871-1886`).
- The GIF/self-play capture path is deterministic: it calls
  `eval_mod._policy_eval_action(...)` each scalar turn, steps the turn-commit env,
  and writes scalar/joint action traces
  (`lightzero_curvyzero_stacked_debug_visual_survival_train.py:6379-6411,6546-6549`).
- The GIF eval helper calls `MuZeroPolicy.eval_mode.forward` and extracts one
  selected action from the output
  (`lightzero_curvytron_visual_survival_eval.py:539-563`).
- Old GIF traces were also mostly one action, but not the same action:
  old `iteration_50` looked like mostly `2`, old `iteration_4350` like mostly
  `1`, and new `iteration_1`/`4` like all `0`. So this should not be framed as
  definitely a new GIF stochasticity bug.
- Existing action-collapse warnings in GIF telemetry only flag the extreme case
  where all observed actions are identical
  (`lightzero_curvyzero_stacked_debug_visual_survival_train.py:6757-6764`).
- Current reward rows are `alive + bonus pickup + terminal outcome`; the code logs
  each component separately
  (`curvytron_two_seat_lightzero_train_smoke.py:1646-1681,1722-1745`).

## Read

Greedy GIF action strings are a bad standalone health metric. They are useful as
a smoke alarm, but they compress too much: deterministic eval, one selected action,
no temperature/epsilon, no margin, and no proof that the selected action was a
strong preference rather than a weak argmax.

The old/new contrast makes the main question sharper. The visible greedy policy
has often been low-entropy; the dominant action changed from mostly `2` or `1` to
`0` under a much heavier new config. That is concerning, but it does not prove the
new trainer is collapsing during collection.

The overnight gate should be fresh collection decisions, not GIF strings and not
raw physical action counts. Fresh policy decisions are the rows where the policy
actually chose an action and replay gets a target. Physical actions include no-op
skip ticks; since skipped policy chances send NOOP (`1`), physical counts can look
skewed even when fresh decisions are less skewed. If fresh policy decisions by
player are broad and reward/terminal metrics are sane, greedy eval collapse is a
readout risk. If fresh decisions also become one-action, it is training collapse
or collection-policy collapse.

## Reward Risks

- The alive helper is small but dense. It mostly rewards "keep not dying" and may
  not distinguish good steering early unless terminal outcomes arrive often enough.
- `discount=1.0` makes long-horizon credit blunt. The terminal scaling keeps wins
  and losses on survival scale, but early replay can still look like many similar
  tiny-positive rows.
- Bonus pickup reward is immediate and small. It can bias local chasing, but it
  does not obviously explain action `0` collapse by itself.
- No-op skips staying out of replay rows is cleaner than training on skipped ticks.
  But physical action histograms will include those skip NOOPs, so they should not
  be used as the primary collapse metric. Watch `policy_selected_action`,
  `executed_action`, and fresh-decision row counts separately.

## Checks Before Overnight

Emit or inspect these at every progress write and every checkpoint eval:

- Fresh policy-decision action histogram by player, with row counts, top-action
  fraction, and entropy. This is the primary collapse gate.
- Physical action histogram by player, separately labeled as physical/executed
  actions and interpreted with no-op skip counts. Do not fail the run on physical
  NOOP skew alone.
- Deterministic eval action histogram plus margin data. If greedy eval is one
  action but margins are tiny or visit distributions are tied, call it weak
  argmax readout, not proven training collapse.
- Stochastic collect-mode replay of the same checkpoint on the same seeds. If
  collect-mode replay is diverse while deterministic eval is one-action, overnight
  risk is mostly observability/readout.
- For sampled eval turns: selected action, legal mask, visit-count distribution,
  root prior/policy logits if available, root value, and margin between best and
  second-best action.
- Same initial observations through `eval_mode.forward` and `collect_mode.forward`
  with `temperature=1.0`, `epsilon=0.25`, to separate eval tie-breaking from model
  preference.
- Reward component sums by action and by player: alive helper, bonus pickup,
  sparse outcome, terminal outcome, terminal causes, and episode length.
- Seed sweep for the first checkpoint. If all seeds and both players choose one
  greedy action with large margins, treat it as dangerous even if collection is
  still noisy.

## Launch Recommendation

Use B32 for the overnight canary, not B64. B64 is not clearly worth the wall time
while the only visible checkpoint health signal is greedy and low-entropy. Prefer
faster checkpoints and faster feedback: B32, sim8, collect64, updates4,
accumulated replay, learner sample 128, normal death, background eval/GIF on.
Move back to B64 only after the first few checkpoints show non-collapsed fresh
collection decisions and either non-collapsed eval or clearly weak/tied eval
margins.

## Small Changes

- Doc-only change for now: keep this note as the pre-overnight guardrail.
- Small code change worth doing next, if time permits: add fresh-decision
  top-action fraction/entropy to trainer progress, separately add physical-action
  top fraction/entropy, and include compact visit-count distributions/margins for
  the first N GIF scalar actions.
- Do not change reward or architecture before the canary unless the new checks show
  collection collapse too.

## Causal Critic Addendum

Facts first:

- Action ids are `0=left`, `1=straight`, `2=right`; `NOOP_ACTION_ID` is `1`, so
  skipped ticks look like straight actions in physical histograms.
- Current timing canaries showed varied fresh collect decisions. That is the
  strongest evidence against immediate collection-path one-action collapse.
- Physical counts being NOOP-heavy is expected when policy no-op skips are
  enabled because skipped policy chances send action `1` and do not create
  replay rows. The baseline launch should keep that skip knob off; stochastic
  variants can turn it on only when we are explicitly testing robustness.
- GIF greedy traces being mostly one action in older runs means deterministic
  low-entropy eval is a recurring readout/problem, not proof of a newly introduced
  stochasticity bug.
- The two-seat trainer maps active player rows back to joint actions with legal
  masks, records `policy_selected_action` and `executed_action`, and stores reward
  components per fresh decision row.

Likely causes:

- Deterministic greedy eval plus tie-breaking or tiny margins. If eval chooses
  argmax from weak near-tied outputs, it can show one repeated action even while
  collect mode remains varied.
- Weak early policy under low-simulation MCTS. With few simulations and immature
  value/policy heads, root visit counts can be dominated by priors, noise, or
  deterministic ordering rather than real steering knowledge.
- Reward is a weak early steering teacher. Alive reward mostly says "do not die";
  terminal outcome is delayed; bonus pickup is small. This can learn "keep doing
  whatever survived briefly" before it learns when to switch turns.
- Action semantics amplify collapse visually. Constant `0` or `2` creates a loop
  until self-collision; constant `1` is also common in physical logs because it is
  both straight and the skip NOOP.
- Self-play can reinforce bad conventions early. One shared weak policy controls
  both seats, so a degenerate maneuver can shape both players' data until varied
  collection or terminal losses break it.

Less likely from current evidence, but still test:

- Policy no-op skips as the cause of greedy GIF collapse. Skips explain physical
  `1` skew, but GIF scalar actions are fresh eval decisions.
- Reset randomness as the main cause. Per-row reset seeds are generated from the
  env RNG, and current fresh canaries varied; still, a fixed GIF seed can hide
  seed sensitivity.
- Player-perspective mapping as the main cause. The code has a perspective schema
  and probe showing player frames differ, but a sign/action inversion would still
  look like "always turn the wrong way" and deserves one targeted check.
- Visual input blankness. The stack is validated and GIF frames are source-state
  RGB captures, but visual/channel corruption would also produce near-constant
  policy outputs, so this is a cheap smoke check.

Fast separator tests before overnight:

- Run the same checkpoint/seed through eval mode and collect mode. If eval is
  one-action and collect is diverse, the problem is greedy readout/tie-breaking.
- For the first 50-100 GIF scalar turns, log selected action, legal mask, visit
  distribution, top-two margin, root prior/logits if present, and root value.
  Tiny margins mean weak argmax; large margins mean real model preference.
- Disable policy no-op skips for a tiny collect smoke
  (`policy_action_repeat_min=max=1`, extra probability `0`). Physical and fresh
  histograms should then agree closely.
- Sweep 5-10 reset seeds for the same early checkpoint. Collapse on every seed
  and both players is worse than one bad visual trace.
- Force scripted action probes from identical starts: all-left, all-straight,
  all-right, alternating left/right. Confirm terminal reasons and survival lengths
  match expected action semantics.
- Feed player 0 and player 1 perspective observations from the same env row
  through eval. A mirrored situation should not require the same global turn if
  perspective/action semantics are correct.
- Save a few raw observation min/max/nonzero/channel summaries next to the GIF.
  This separates blank/constant visual input from bad policy preference.

Minimal launch changes:

- Do not rewrite reward, architecture, or self-play today.
- Keep the baseline run clean: no policy no-op skip. Use stochastic variants
  deliberately, not as the default.
- Gate launch on fresh collect-decision entropy/top-action fraction, not physical
  NOOP-heavy counts or GIF strings alone.
- Add or inspect compact eval margin/visit telemetry for early GIF turns if the
  hook is quick; otherwise run the eval-vs-collect and no-skip smokes manually.
- If fresh collect decisions are also over 95% one action for either player, pause
  the overnight and inspect reward/action/perspective before scaling.
