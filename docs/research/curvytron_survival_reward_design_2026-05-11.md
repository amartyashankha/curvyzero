# CurvyTron Survival Reward Design - 2026-05-11

## Short Answer

Immediate note: shared `+1 per survived step` is acceptable as a short-term
diagnostic while the training plumbing is still being proven. It is not the
main blocker today. When using it, log shaped survival and sparse outcome
separately so a long loss is not mistaken for a real win-rate improvement.

Do not make `+1 per survived step` plus `+episode_length` winner bonus the
default objective. It is simple, but it makes both agents share a large survival
reward and makes reward scale grow with horizon. A long loss can beat a short
win under that objective:

```text
1/step for both, +T winner bonus
win at T=40  -> winner return 80
loss at T=100 -> loser return 100
```

That is the wrong default for "survive longer than the opponent." Use a
competitive outcome-dominant reward first. If sparse reward is too hard, test a
small bounded survival auxiliary that cannot outrank winning.

## Evidence

AlphaZero trains value against game outcome: `-1` loss, `0` draw, `+1` win, not
a hand-shaped living score. MuZero adds a reward head, but it predicts the
observed environment reward stream plus value and policy targets; changing the
reward stream changes the objective, not just the logging.

Multi-agent reward structure is behavior structure. Shared rewards promote
cooperation; zero-sum rewards promote direct competition. Pong studies that vary
the reward scheme show the same game can become competitive or cooperative, and
cooperative Pong agents learn to keep the ball alive and delay serving when play
can only make things worse.

Potential-based shaping is the clean theoretical exception, but a flat living
bonus is not obviously `gamma * Phi(s') - Phi(s)`. Multi-agent extensions exist,
but they still require a real potential-style construction. "Alive this step"
does not get policy-invariance for free.

## Reward Variants To Test

### 1. Sparse Zero-Sum Outcome

Default and promotion metric:

```text
both alive:              0, 0
ego survives, opp dies: +1, -1
ego dies, opp survives: -1, +1
same-tick draw:          0, 0
time-limit truncation:   0, 0  # logged separately
```

This is boring in the best way. It matches the game objective, keeps reward
scale stable, and makes self-play comparisons cleaner.

### 2. Outcome Plus Small Normalized Alive Auxiliary

Use only if sparse learning is too slow:

```text
per decision while alive: +epsilon / max_decisions
terminal:                +1 / -1 / 0
epsilon:                 0.05 to 0.25
```

No episode-length winner bonus. The entire survival bonus over a full episode is
bounded below the terminal payoff, so a long loss cannot become better than any
win. This is general-sum and can still encourage stalling, so evaluate and select
on variant 1 metrics.

### 3. Sparse Outcome Plus Anti-Stall Time Cost

Use if policies learn to circle forever or timeout farm:

```text
per decision while both alive: -epsilon / max_decisions for both
terminal:                     +1 / -1 / 0
epsilon:                      0.02 to 0.10
```

This asks for decisive wins, not just survival. It can over-encourage risky
contact or suicide if too large, so keep it tiny and compare terminal causes.

## Risks And Footguns

- Shared living reward changes the game from competitive toward cooperative
  survival.
- Length-scaled winner bonuses change value scale across `decision_ms`,
  `max_ticks`, and curriculum settings.
- If unnormalized returns exceed MuZero support or local scalar assumptions,
  reward/value heads can spend capacity on scale rather than ranking actions.
- Survival shaping can make "lose later" look like progress while win rate is
  flat.
- Time-limit truncations must not look like successful survival unless the eval
  metric says so explicitly.
- Reward changes must get new reward schema ids; old and new replay should not
  be mixed silently.

## Metrics To Log

- reward schema id, `epsilon`, `max_decisions`, `decision_ms`, `max_ticks`
- true sparse outcome return even for shaped runs
- shaped return separately from true return
- win/loss/draw/truncation rates, seat-swapped
- survival decisions and physical time: mean, median, p90, max
- terminal reason: wall, own trail, opponent trail/body/head, draw, timeout
- winner id, loser id, same-tick death flag
- action histogram, top-action fraction, per-seat entropy
- episode length conditioned on win, loss, draw, timeout
- value/reward target min, max, mean, std, and support clipping rate
- policy-vs-baseline panels against random, straight, wall-avoid, and earlier
  checkpoints

## Recommendation

Start with variant 1. If it is too sparse, run variant 2 as a labeled ablation
with `epsilon <= 0.25`, but keep promotion on sparse outcome metrics. Keep
variant 3 ready for timeout farming. Do not test the proposed unbounded
`+1/step + T winner bonus` as a serious candidate unless the goal is explicitly
to study the failure mode.

## Sources

- AlphaZero paper: https://arxiv.org/abs/1712.01815
- MuZero paper: https://arxiv.org/abs/1911.08265
- AlphaGo Zero Nature page: https://www.nature.com/articles/nature24270
- Multi-agent reward taxonomy survey:
  https://link.springer.com/article/10.1007/s10462-021-09996-w
- Multiagent Pong reward-structure study:
  https://doi.org/10.1371/journal.pone.0172395
- Potential-based reward shaping:
  https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html
- Multi-agent potential-based shaping:
  https://pure.york.ac.uk/portal/en/publications/theoretical-considerations-of-potential-based-reward-shaping-for-/
- ALE multi-agent Pong reward/timer docs:
  https://ale.farama.org/multi-agent-environments/pong/
- Reward misspecification example:
  https://openai.com/index/faulty-reward-functions/
