# LightZero Pong Contact Curriculum Critique - 2026-05-09

Role: curriculum critique researcher. No pytest.

## Verdict

A scoreable contact/angle curriculum is the right next move after the sparse
Pong lane failed under length, MCTS sims, update/replay, and collection
exploration. But the important word is **scoreable**, not contact.

Implementation update: the first minimal version is now in code as the
explicit opt-in custom dummy Pong reset profile
`pong_reset_profile=contact_pressure`. It is a synthetic near-contact start
profile, clearly labeled as custom dummy Pong curriculum and not stock Atari
Pong replication. `env.step()` still returns only true sparse score reward
`+1/-1/0`. Tiny Modal train `ap-bNRz3Mtil6apjX5w6tNZxa` and matching MCTS
scorecard `ap-XRyCAYWAN7F3ptvRAKRC0x` passed mechanically, but held-out
`iteration_2` still failed quality gates with zero down actions. This is an
implementation smoke, not a promoted curriculum result.

This is a **custom LightZero dummy Pong environment** curriculum, not stock
Atari Pong replication. It borrows the useful LightZero pattern from
bot-mode/sparse final-outcome tasks: a hidden opponent inside the env step,
true terminal score rewards, long-horizon value targets, and enough replay
pressure for sparse outcomes. It does not prove that official ALE/Atari Pong
training has been reproduced, and it should never be reported as such.

The failed rungs changed training volume and exploration while preserving the
same sparse target and mostly the same reset/opponent distribution. They
created some trainer-side action diversity, but heldout MCTS policies still
collapsed and did not improve raw score. That points away from "just more
same data" and toward a distribution problem: the learner is rarely seeing
states where a controllable paddle/contact choice creates a terminal score
difference.

The curriculum should therefore change the initial-state and opponent
distribution while keeping the game reward unchanged:

```text
env.step reward:
  ego scores:       +1
  opponent scores:  -1
  no score event:    0
```

Do not train the MuZero reward head on paddle hits, rally length, survival
bonus, contact quality, or loss-delay by default. Those remain telemetry,
curriculum filters, or explicitly labeled ablations.

Critique-wave incorporation:

- Do not scale the old self-play/generation loop. Generation machinery and
  promotion gates are guardrails, not the next strategy.
- Prove the target is scoreable before training or scaling it.
- The next Pong learner should use a weaker/changed target ladder, with fixed
  baseline eval rows before selection.
- Every run must emit run health: iteration metrics, action histograms by
  seat, entropy/collapse metrics, terminal causes, failure examples, and
  heldout eval after selection.
- CPU/GPU Modal execution and Volume artifacts matter for real runs, but
  remote plumbing is not policy quality.

## Evidence Read

Current stop signals:

- Sparse rung 0 and pure 2x stayed collapsed all-up, with no heldout
  improvement in survival, shaped return, raw score, or action entropy.
- UPC25 higher update/replay produced mildly live trainer telemetry but failed
  heldout; `iteration_50` was still mostly up and `ckpt_best` was all-up.
- UPC25 plus random warmup and epsilon collection diversified collect actions,
  but heldout `iteration_50` still had no stay actions, remained mostly up, and
  did not improve raw score against `lagged_track_ball_1` or `random_uniform`.
- Increasing eval simulations removed some visit ties but made the bad action
  collapse more confident.

Geometry/contact evidence:

- Default `track_ball` is not a valid hard win target from normal resets. The
  exact bounded beatability probe found zero winning traces against
  deterministic `track_ball` under `PongConfig(width=15,height=9,paddle_height=3,max_steps=120)`.
- Contact probes proved that top/center/bottom hits change outgoing `ball_vy`,
  but default geometry and width-9 geometry both produced flat zero
  score-delta returns over the tested short horizons.
- `random_uniform` and `lagged_track_ball_1` are scoreable opponents; default
  `track_ball` is a survival/tie floor, not a win gate.

Reward-shaping rule:

- `docs/research/reward_shaping_for_pong_curvy_muzero.md` is still the right
  guardrail: keep `env.step()` and MuZero reward targets on true sparse score;
  use survival/loss-delay for telemetry, selection tie-breaks, and curriculum
  state choice only.

LightZero pattern match:

- This curriculum matches the custom LightZero env lane: single ego action,
  scripted/frozen opponent folded into `env.step()`, all actions legal, and
  score reward returned from the ego perspective.
- It also matches the sparse final-outcome lesson from bot-mode board-game
  controls: use `td_steps` through the point horizon, `discount_factor=1`, and
  raw terminal outcome as the value target.
- It does not match official Atari Pong replication. Stock Atari Pong has ALE
  observations, Atari wrappers, a different action space, larger visual models,
  frame stacks, and much larger budgets. Keep that lane labeled separately.

## What The Curriculum Should Be

The curriculum should make score-bearing choices common enough for MuZero to
learn from, without redefining what winning means.

Use named reset profiles:

1. `canonical_v0`: current normal reset. Centered paddles, centered ball x,
   seeded ball y, seeded serve direction, seeded vertical velocity. This stays
   the main heldout eval profile.
2. `contact_scoreable_v1`: near-contact starts for training only. States are
   admitted only if a short true-reward rollout shows that legal ego choices
   can change score outcome or avoid a loss.
3. `angle_pressure_v1`: post- or pre-contact pressure starts for training only.
   These emphasize top/center/bottom angle choices, but still admit states by
   true score spread, not by "made an off-center hit".

Exact reset constraints:

- Keep `PongConfig` rules and `PongEnv.step()` physics unchanged.
- Store every curriculum start with `reset_profile`, `state_id`,
  `state_source`, `generator_version`, `base_seed`, `episode_seed`,
  `ego_agent`, `opponent_policy_id`, and whether the state was harvested from a
  legal rollout or synthesized.
- Prefer harvested legal states from canonical rollouts. If a synthetic state
  setter is used, validate it against `PongEnv.step()` parity and mark it
  synthetic in telemetry.
- Mirror states across `player_0` and `player_1` so the learner does not learn
  a seat-specific trick.
- Do not start from terminal, already-scored, unavoidable-score, or
  unavoidable-loss states. The ego must have at least two legal actions whose
  short rollout outcomes differ.
- Keep timeout reward at `0`. Reject a profile if it mainly increases timeouts
  rather than score events.
- Do not mix curriculum starts silently into canonical eval. Training can use a
  curriculum mix; quality claims must include canonical heldout scorecards.

Admission test for a curriculum state:

```text
For each candidate state:
  evaluate legal ego actions or contact targets
  use the chosen opponent policy
  roll out with true sparse env reward for H steps
  admit only if max(return) - min(return) > 0
  record best actions, return spread, terminal causes, and impact offsets
```

For the first version, use `H=48` or the remaining horizon to the 120-step cap.
The earlier contact probes had `score_delta_return_differs_state_count=0`; the
new builder must beat that before any MuZero training run is worth launching.

## Opponent Changes

Do not train or select against default `track_ball` as a win target in this
geometry. It is useful as a full-survival/tie monitor only.

Use an opponent ladder that creates real score pressure:

1. `random_uniform`: dense score events, useful for checking the learner can
   exploit mistakes. Not sufficient alone, because it can teach brittle
   anti-random behavior.
2. `lagged_track_ball_1`: the current best scoreable scripted target. Keep it
   as the first serious training opponent.
3. `lagged_track_ball_2` or `track_ball_with_dropout_p10/p25`: add a slightly
   weaker-but-structured opponent if implemented. The point is to make angle
   choices matter, not to copy a perfect tracker.
4. Frozen weak checkpoints: use only after they are proven scoreable and not
   collapsed. Treat them as staged self-play opponents, not quality proof.
5. `track_ball`: canonical monitor row for survival/tie behavior; never the
   sole promotion gate.

For the first curriculum MuZero run, the safest mix is:

```text
reset mix:
  50% canonical_v0
  50% contact_scoreable_v1

opponent mix:
  70% lagged_track_ball_1
  30% random_uniform
```

If the curriculum-state probe cannot find enough score-spread rows under that
mix, change the opponent ladder before changing reward. A delayed/dropout
tracker is cleaner than a hit bonus.

## Bad Curricula

Bad curricula teach the wrong objective or hide the failure:

- Paddle-hit rewards, off-center-hit rewards, rally-length rewards, or living
  bonuses in `env.step()`. These can make contact or stalling look better than
  scoring.
- Training the MuZero reward head on `shaped_loss_delay_return` while reporting
  raw Pong as if unchanged.
- A near-contact dataset admitted because it has contact, even though all
  legal choices have the same zero score return.
- Starts where the opponent is already doomed or the ego is already doomed.
  That teaches recognition of pre-decided states, not control.
- Starts sampled from impossible geometry without metadata. Synthetic states
  are acceptable only as a labeled curriculum profile and never as canonical
  eval.
- Smaller boards, smaller paddles, faster balls, or altered bounce rules used
  as the promoted task. Those are domain/geometry variants, not the same Pong
  objective. They are fine only after a one-page profile definition and
  separate eval rows.
- Training only against `random_uniform` and claiming general Pong progress.
  Random is a score-event generator, not a strategic ceiling.
- Requiring wins against default `track_ball` from normal resets. The exact
  probe says that target is impossible under the current bound.
- Selecting checkpoints by action entropy, contact count, or survival when raw
  score gets worse.

## Ranked Implementation Plan

1. Build a curriculum-state probe/exporter before training.
   Extend the contact-outcome tooling into a `contact_scoreable_v1` table that
   records states, legal action returns, terminal causes, impact offsets, and
   opponent id. Go only if at least 10/64 sampled states have nonzero true
   score-return spread, and at least two different first actions appear among
   best actions.

2. Add named reset profiles to the LightZero dummy Pong wrapper.
   Keep `canonical_v0` as default. Add a JSONL-backed curriculum profile that
   resets `PongEnv` to admitted states and writes reset metadata into every
   telemetry row. Do not touch `PongEnv.step()` reward.

3. Add a small opponent-ladder surface.
   Reuse `random_uniform` and `lagged_track_ball_1` first. If score-spread rows
   are too rare, add `lagged_track_ball_2` or a dropout tracker as a named
   scripted opponent. Keep `track_ball` as monitor-only.

4. Run one tiny LightZero MuZero curriculum ablation.
   Use the latest sparse settings that were mechanically healthy
   (`td_steps=120`, `discount_factor=1`, fixed 120-step horizon, UPC25 shape),
   but change only reset/opponent mix. Score `iteration_0`, final, and
   `ckpt_best` on both canonical heldout and curriculum heldout.

5. If and only if the tiny run passes, repeat on two seeds.
   Promotion requires a directionally consistent checkpoint curve, not one
   lucky final checkpoint. Then consider frozen weak-checkpoint opponents.

6. If the curriculum probe fails to produce scoreable rows, stop LightZero
   sparse training and change the target generator.
   Next choices are a stronger but labeled geometry profile, a small
   policy-gradient/PPO baseline on the same curriculum, or a project-owned
   planner/distillation path. Do not run the same sparse MuZero shape again.

## Falsification Criteria

These are hard gates. A run that fails any relevant gate is a negative result,
even if trainer-side reward or shaped telemetry looks better.

Pre-training falsifiers:

- `contact_scoreable_v1` has `score_delta_return_differs_state_count=0`, like
  the previous contact probes.
- Fewer than 10/64 sampled curriculum states have true score-return spread.
- Best actions are all the same action, or all admitted states are
  immediate/unavoidable wins or losses.
- The profile cannot produce paired-seat mirrored states.
- More than 25% of admitted states are synthetic without a legal-rollout
  source or transition-parity validation.

Training falsifiers:

- Heldout canonical raw score versus `lagged_track_ball_1` and
  `random_uniform` is flat or worse than `iteration_0`.
- Curriculum heldout improves but canonical heldout regresses by `>=0.125`
  mean raw score against either `lagged_track_ball_1` or `random_uniform`.
- Final or best checkpoint has normalized action entropy below `0.35` across
  `random_uniform + lagged_track_ball_1`, or any action exceeds `90%` of
  actions without state-dependent root evidence.
- Survival/shaped loss-delay improves only through higher timeout rate, defined
  as timeout rate increasing by `>=0.10` absolute without raw score
  improvement.
- The learned checkpoint beats only `random_uniform` while getting worse
  against `lagged_track_ball_1`.
- Results do not reproduce on a second seed.
- The run summary cannot prove `env.step()` reward stayed sparse and unchanged.

Promotion criteria:

- At least two checkpoints after initialization improve raw score by `>=0.125`
  absolute or win rate by `>=10` percentage points on canonical heldout against
  a scoreable opponent.
- No material regression against `random_uniform`.
- `track_ball` row is treated correctly: survival/tie pressure can improve,
  but lack of wins is not a failure under current geometry.
- Telemetry proves the run used true sparse env reward and records reset
  profile, opponent id, seed distribution, survival stats, shaped telemetry,
  raw score, truncation rate, and action histograms.

Decision table:

| Gate | Pass | Fail action |
| --- | --- | --- |
| Curriculum-state probe | Score spread in at least 10/64 states and varied best actions | Do not train; change opponent/profile generator |
| Tiny MuZero ablation | Canonical heldout raw score improves on a scoreable opponent without collapse | Repeat on another seed |
| Second seed | Same qualitative improvement | Consider frozen weak-checkpoint opponent |
| Canonical regression | No material drop versus random or lagged target | Stop; curriculum is overfitting starts |
| Reward integrity | Sparse env reward unchanged and logged | Stop; relabel as objective-changing ablation |

## Sources

- `docs/working/training_coach_handoff_2026-05-09.md`
- `docs/working/training_experiment_backlog.md`
- `docs/working/pong_training_critique_wave_2026-05-09.md`
- `docs/working/training_coach_self_critique_2026-05-09.md`
- `docs/research/reward_shaping_for_pong_curvy_muzero.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-scale-ladder-rung0-rung1.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-upc25-sim8-run.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-upc25-epscollect-run.md`
- `docs/experiments/2026-05-09-dummy-pong-track-ball-beatability-probe.md`
- `docs/experiments/2026-05-09-dummy-pong-contact-outcomes-smoke.md`
- `docs/experiments/2026-05-09-dummy-pong-contact-outcomes-width9-h48.md`
- `src/curvyzero/training/dummy_pong.py`
- `src/curvyzero/training/lightzero_dummy_pong_env.py`
- `src/curvyzero/training/dummy_pong_eval.py`
