# Hypotheses And Evidence

Purpose: make the scientific loop explicit. Hypotheses can be wrong; the point
is to track what evidence changed our mind.

## H1: Recent/mid frozen opponents are too weak or too deterministic to provide useful reward.

Status: currently favored.

Evidence:

- Corrected v1d outcome curves show many recent/mid frozen rows already at
  mostly or fully `win` from the first checkpoint.
- Those same rows stay near the survival floor around `8` steps.
- Fixed and old-opponent rows often start with losses and later move toward
  wins, so the status/eval pipeline can see outcome movement when it exists.

Implication:

- Do not build the next batch around normal recent/mid frozen opponents.
- Add immortal/no-trail or otherwise non-ending diagnostic opponents.

## H2: Survival-first reward is the right next diagnostic objective.

Status: favored for the next batch.

Evidence:

- Outcome can saturate while behavior remains bad.
- The goal of the next diagnostic lane is wall/trail avoidance from visual
  input, not beating a weak opponent.

Implication:

- Train with survival plus bonus pickup reward.
- Keep outcome as an eval metric only; outcome reward should be off/zero for
  the next diagnostic lane.

## H3: Fixed-straight and old frozen opponents still have value as controls.

Status: supported.

Evidence:

- Fixed and old rows showed outcome movement and some survival movement.
- They are useful low-bar controls for whether the stock loop can learn
  anything under a given reward/render/stochasticity setup.

Implication:

- Keep a small number of fixed/old rows in the next matrix.
- Do not confuse them with the main success condition.

## H4: Episode cap should not be swept in the next batch.

Status: favored.

Evidence:

- v1d `256` vs `1024` did not explain the early failure.
- The agents are dying far below either cap.

Implication:

- Set a high cap, e.g. `65536`, and treat longer episodes as success.

## H5: Render fidelity still needs matched rows.

Status: open.

Evidence:

- v1d browser sentinels did not wildly disagree with fast render.
- The optimizer has changed render performance, and final training likely cares
  about the visual surface.

Implication:

- Do not broadly sweep render, but for important rows run fast and browser
  matched pairs.

## H6: Stochasticity may matter and was underswept.

Status: open but important.

Evidence:

- v1d only had tiny straight override checks.
- The next survival-first lane may benefit from more starting-state and control
  variation.

Implication:

- Sweep stochasticity levels meaningfully in the next matrix.
- Keep no-stochasticity controls.

## H7: Search/collector/learner knobs are secondary until the objective/opponent is fixed.

Status: favored.

Evidence:

- v1d sim16, C64, B64 did not rescue recent/mid frozen rows.

Implication:

- Use a small projection-informed sweep, not a huge Cartesian explosion over
  compute knobs.

## H8: Passive immortal opponent is not enough by itself.

Status: supported.

Evidence:

- `opponent_death_mode=immortal` only suppresses player 1 death.
- Player 1 can continue out of bounds and still leave collidable trail/body
  points.
- Player 0 can still die on player 1's trail.

Implication:

- Treat passive immortal as a canary or building block, not the main design.
- Compare it with cleaner opponent designs: no-trail blank, reflecting wall
  behavior, or scripted wall avoidance.

## H9: Random opponent copies need explicit opponent randomness.

Status: open.

Evidence:

- Stock source-state training currently exposes fixed-straight and frozen
  LightZero-checkpoint opponents.
- A random learned frozen opponent is not currently a first-class setting.

Implication:

- Decide whether random opponents come from generated random checkpoints or a
  new explicit random-policy opponent kind.
- Repeated copies should vary the relevant opponent seed, not just the eval
  seed.

## H10: The desired survival-plus-bonus/no-outcome reward needs support-size review.

Status: active implementation gate.

Evidence:

- A `survival_plus_bonus_no_outcome` path now appears in the fixed-opponent
  env/trainer code.
- Bonus pickup count exists in telemetry as `bonus_catch_count_step`.
- The remaining risk is not only reward semantics. With a high cap, value and
  reward support sizes can become too large unless the LightZero target/model
  support configuration is capped or separated safely.

Implication:

- Verify tests for same-step bonus reward and outcome exclusion.
- Verify practical model support sizes at `source_max_steps=65536`.
- Keep sparse outcome logged as telemetry, not trainer reward, in this lane.

## H11: Proactive wall avoidance is the first scripted baseline.

Status: supported by probe data.

Evidence:

- Proactive force-field with margin `20` stayed in bounds for `0/384` OOB
  across three 128-start, 1024-step real-env probes with normal trail writing.
- Its action mix was about `13.6%` left, `72.9%` straight, `13.6%` right.
- Contact-only reflection failed `128/128` OOB; pure reflected heading failed
  `59/64` OOB in a shorter run.
- Inward-biased and rollout-style reflection variants can be improved, but they
  are turn-heavy and less clean.

Implication:

- Use proactive force-field margin `20` as the first integration candidate for
  scripted wall-avoidant rows.
- Keep reflection/rollout variants as later scripted-variant rows only if they
  answer a specific pressure question.

## H12: A large matrix should be organized around staged repeat groups.

Status: favored.

Evidence:

- The main uncertainty is seed/opponent-family stability, not whether we can
  enumerate more knobs.
- Fast/browser render pairs are diagnostic pairs, not independent evidence.

Implication:

- Scale to 50/100/200+ rows only when the added rows answer a named question.
- Spend extra rows on repeated copies of important cells and confirmation
  blocks, not a full reward x opponent x render x stochasticity x compute
  product.
