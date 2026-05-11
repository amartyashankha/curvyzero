# LightZero Pong Train-Longer Vs Bug Hypotheses - 2026-05-09

## Short Answer

The current failures are not well explained by "just train longer" alone.

Sparse reward and short runs are real problems, but the strongest recent facts
point to a weak training/eval setup:

- collection can show varied actions because it uses noise, sampling, and
  optional epsilon;
- independent eval is deterministic, uses no root noise, and breaks tied visit
  counts with `argmax`;
- recent MCTS roots have tiny policy differences and frequent tied visits;
- several checkpoints collapse to exactly one eval action, but the collapsed
  action changes by run: all `up`, all `stay`, or all `down`;
- baselines use all three actions, and the adapter still maps
  `0=up`, `1=stay`, `2=down`.

So this does not look like a simple action-map or environment contract bug.
It also does not look like a run where the same config merely needs more wall
clock. The best read is: the setup is underpowered and poorly configured for
this Pong task, and sparse reward may need a curriculum or dense auxiliary
target before scaling has a fair chance.

## Evidence Read

Recent docs read:

- `docs/experiments/2026-05-09-lightzero-dummy-pong-horizon-fixed-probe.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-lagged-opponent-smoke.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-lag1-shaped-knob-run.md`
- `docs/working/lightzero_pong_eval_action_collapse_debug_2026-05-09.md`
- `docs/working/lightzero_source_setup_audit_2026-05-09.md`

Key facts:

- Horizon-fixed probe: horizon wiring now works. Train budget was `1024`,
  episode cap was `120`, and trainer-side survival reached exactly `120`.
  Independent eval still collapsed to all `up`: `[570, 0, 0]`.
- Lagged-opponent smoke: tiny training against `lagged_track_ball_1` produced
  better small scorecard counts than the earlier random-opponent run, but eval
  collapsed to all `stay`: paired `[0, 10269, 0]`, player0-only
  `[0, 2296, 0]`.
- Lag1 shaped knob run: larger knob run collected varied train actions
  `[97, 85, 292]`, but train score was negative and small eval collapsed to
  all `down`: `[0, 0, 335]`.
- Eval debug: for one collapsed checkpoint, 24 first-N rows all chose `stay`
  because every row had visits `[2, 3, 3]`. Deterministic `argmax` picked the
  lower tied action even though logits weakly preferred `down`.
- Source setup audit: `to_play=-1`, fixed action space, all-ones action mask,
  and non-board-game env type look coherent for an ego-vs-scripted wrapper.
  Bigger suspects are weak exploration, tiny collection, low update/replay,
  low simulations, and unrecorded or inherited config defaults.

## Hypothesis Table

| Hypothesis | Current Fit | Smallest Discriminating Experiment | Confirming Result | Refuting Result |
| --- | --- | --- | --- | --- |
| Needs longer | Weak to medium. Runs are short, but the 4096/64 shaped-knob run did not improve train score and still collapsed in eval. Longer alone is not the leading explanation. | Hold the exact best current config fixed and run only a 2x or 4x budget increase, with checkpoints scored every fixed interval. | Scorecard wins, shaped return, survival, and eval action diversity improve steadily without changing exploration, replay, or simulations. | Longer run keeps one-action eval collapse, flat or worse score, and tiny/tied root visits. |
| Needs more simulations | Medium. Eval debug directly shows 8-sim roots with tied visit counts. This can create fake-looking collapse. | Same checkpoint, same first 24 observations, sweep `num_simulations` across `8, 16, 32, 64`; log logits, visits, action, and tiny score rows. | Visit ties mostly disappear, action choice follows state, and small scorecard improves. | More sims only makes the same single action more certain, or score stays bad despite fewer ties. |
| Needs more updates/replay | Medium to high. Source audit says current runs are far below official CartPole reuse: tiny `n_episode`, `batch_size`, and `update_per_collect`. | Same env-step budget, compare current config against `n_episode=8`, `batch_size=32`, `update_per_collect=8` or higher, with random warmup held constant. | Better policy logits, lower value/policy loss noise, improved heldout score, and less eval collapse at the same env steps. | More replay/update overfits the same weak trajectories and does not improve heldout score or root separation. |
| Sparse reward needs curriculum or dense auxiliary target | Medium. Terminal reward is sparse and scripted `track_ball` exposes a real skill gap. But current failures also appear before the learner has stable action/state signal. | Train a tiny auxiliary/curriculum lane: keep game reward honest, but add loss-delay/survival target or staged opponent curriculum; score against the same heldout opponents. | Early checkpoints learn survival/action geometry faster, beat random/lagged more reliably, and still transfer to sparse-score eval. | Dense/curriculum lane improves auxiliary metrics only, while scorecard and action diversity remain bad. |
| Eval deterministic tie artifact | High for the action-collapse symptom. It directly explains train/eval action mismatch and all-`stay` debug rows. It does not by itself explain poor learning. | First-N collect-vs-eval diagnostic on the same observations: call collect mode and eval mode, log visits and actions. Also test randomized tie-break only for diagnosis. | Collect samples varied actions from similar visits; eval chooses the deterministic max/tie action; randomized tie-break changes action histogram without changing model. | Eval collapse occurs with strong untied visits and the same action remains dominant under randomized tie-break. |
| Config bug or bad config | High for "bad config", low for "hard action/env bug". The source audit found coherent action mask, `to_play`, and fixed action space, but weak exploration/data/replay and inherited defaults are major suspects. | One explicit config-hygiene A/B: record `action_type`, `game_segment_length`, `td_steps`, `num_unroll_steps`, random warmup, epsilon, temperature, supports; then run the recommended exploration/data-volume episode config. | Better action diversity and score after only config/data-volume changes; no need to change env/action mapping. | Fully explicit sane config still collapses with tied roots and poor score, pushing suspicion toward algorithm fit or environment/reward. |
| Environment mismatch | Low to medium. Horizon mismatch was real and fixed. Current action facts do not point to inverted actions or illegal `down`; baselines act coherently. Still, wrapper-vs-direct parity should stay guarded. | Fixed seed trace parity: direct `PongEnv` and `DummyPongLightZeroEnv` replay the same scripted joint actions and compare observations, rewards, dones, and terminal outcome. | Any mismatch in action effect, reward sign, terminal timing, or observation fields. | Exact parity across scripted traces and baseline policies; environment mismatch drops below config/training issues. |

## Ranking

1. **Eval deterministic tie artifact explains the visible action collapse.**
   This is the cleanest answer for why eval can be all one action while train
   telemetry is varied.

2. **Bad config / underpowered setup explains why the roots are weak.**
   The setup uses small simulations, little collection, little replay/update,
   and weak exploration compared with official working examples.

3. **Sparse reward and curriculum remain plausible learning blockers.**
   They matter, but they should not be used to justify simply making the same
   sparse setup longer.

4. **Pure "needs longer" is not the next best bet.**
   Longer may help after fixing exploration, replay, and simulations. Longer
   alone is currently a weak explanation.

5. **Environment/action-map bug is not the leading story.**
   The collapse direction changes by run, baselines cover all actions, and the
   wrapper contract matches the single-agent hidden-opponent pattern.

## Recommended Minimal Sequence

1. Run the no-training simulation-count sweep on existing checkpoints.
   This tells us whether eval collapse is mostly a low-simulation tie artifact.

2. Run one explicit exploration/data-volume config A/B.
   Keep `to_play=-1`, fixed action space, and the same env. Change only
   collection/replay/exploration knobs.

3. If that still fails, run the sparse-reward curriculum or dense auxiliary
   target probe.
   This tests whether Pong geometry is learnable before another long sparse
   run.

4. Keep wrapper trace parity as a cheap guardrail.
   Do not make environment mismatch the main lane unless parity fails.

## Decision

Do not treat the current Pong failures as "probably just too-short training."
The action-collapse debug facts mostly imply deterministic eval over weak,
tied MCTS roots. The source audit mostly implies bad or underpowered config,
not a fatal action-map bug. The next useful work is discriminating config and
simulation probes, then curriculum/dense-target probes if the roots remain
weak.
