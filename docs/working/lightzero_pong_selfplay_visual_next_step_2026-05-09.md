# LightZero Pong Self-Play And Visual Next Step - 2026-05-09

## Plain Answer

The current LightZero dummy Pong training loop is not self-play.

It is single-agent ego training: LightZero controls one paddle, usually
`player_0`, and `DummyPongLightZeroEnv` supplies the other paddle action from a
scripted opponent such as `random_uniform`, `lagged_track_ball_1`, or
`track_ball`.

That run can honestly be called:

```text
LightZero MuZero on dummy Pong, ego vs scripted opponent
```

It should not be called:

```text
MuZero self-play
```

The evidence is in the adapter shape: LightZero receives one action, the env
builds a two-player joint action internally, and the opponent policy id is part
of the env config and episode telemetry.

## Smallest True Self-Play Step

The smallest practical true self-play step is not joint-action MuZero search.
Do not start there.

Use ego-vs-checkpoint self-play first:

1. Train or load a LightZero checkpoint for one ego paddle.
2. Freeze that checkpoint as the opponent policy.
3. Start the next LightZero run with the learner controlling ego and the frozen
   checkpoint controlling the opponent.
4. Alternate seats or run paired seatings so `player_0` quirks do not become
   the result.
5. Promote only if the new checkpoint beats the frozen parent and does not
   regress on fixed baselines.

Plainly, this is:

```text
latest learner vs frozen older checkpoint
```

That is enough to stop confusing fixed-opponent training with self-play. It is
also smaller than two independent LightZero trainers controlling both paddles
inside one env, and much smaller than simultaneous joint-action search.

The next implementation boundary should be:

- add a checkpoint-backed opponent policy for `DummyPongLightZeroEnv`;
- make `opponent_policy` accept something like
  `lightzero_checkpoint:<ref-or-path>`;
- record both policy ids, checkpoint hashes, feature mode, and seat assignment
  in every episode row;
- run paired eval, not just one fixed seat.

Two LightZero-controlled players can come later. Alternating seats should come
immediately, because it is cheap and keeps the read honest.

## Visual Input

`tabular_ego` is a bootstrap. It is useful because it isolates trainer,
checkpoint, MCTS, and scorecard plumbing. It is not the desired visual setup.

Do not move the main learning lane to visual input until the training signal is
less broken. The current LightZero checkpoints still behave effectively
up-only in policy-head and MCTS scorecards. Moving to raster before fixing that
risks hiding a training/control bug behind a new observation change.

Use `raster_flat` next only as a compatibility smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --mode feature-fit \
  --env dummy_pong_lag1 \
  --feature-mode raster_flat \
  --seed 0 \
  --max-env-step 64 \
  --num-simulations 2 \
  --batch-size 8
```

This checks that the custom LightZero env can reset, step, compile config, and
produce observation shape `135` for the flattened `9x15` raster.

Only after that passes, run the smallest raster trainer smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode raster_flat \
  --seed 0 \
  --opponent-policy random_uniform \
  --max-env-step 64 \
  --max-train-iter 2 \
  --num-simulations 2 \
  --batch-size 8 \
  --update-per-collect 1 \
  --n-evaluator-episode 1
```

Treat that as a visual-observation smoke only. It should prove that raster
observations survive the LightZero path and write checkpoints/telemetry. It
should not be used as a quality claim.

## Eval Ladder

Keep the eval ladder exactly boring and explicit:

1. `random_uniform`
   - Sanity floor.

2. `lagged_track_ball_1`
   - First score-pressure target. This is the main early scripted rung.

3. `track_ball`
   - Survival/tie floor. Do not require wins here for early progress.

4. older checkpoint
   - Frozen parent or selected previous checkpoint. This is the first real
     self-play/regression row.

5. current checkpoint
   - Latest candidate, scored against the older checkpoint and the fixed
     baselines.

Every row should include wins/losses, truncations, score return, survival step
stats, shaped loss-delay diagnostics, and action histograms. Constant-action
policies must be visible in the compact summary.

## Blocking Gaps

- The current LightZero train env supports scripted opponents, not checkpoint
  opponents.
- No clean `LightZero checkpoint as opponent policy` adapter exists yet.
- Paired-seat LightZero training/eval is not the default path.
- The current LightZero checkpoint behavior is still effectively up-only; fix
  or explain that before scaling or switching the main lane to raster.
- `raster_flat` exists in the env/config path, but it has not been established
  as the main training signal.
- Full self-play terminology is not clean in docs: ego-vs-scripted, ego-vs
  frozen checkpoint, and two-live-player self-play need separate names.
- Checkpoint promotion rules need to be explicit: beat parent, preserve fixed
  baseline performance, and report action histograms.
- If the long-term target is simultaneous Curvy-style self-play, LightZero's
  single-action wrapper is only a bridge. Joint-action search and multi-player
  value semantics remain future work.

## Next Step

Do this next:

1. Keep `tabular_ego` for the main LightZero debugging lane.
2. Add frozen-checkpoint opponent support to the LightZero dummy Pong adapter.
3. Run `latest learner vs frozen older checkpoint` with paired seats.
4. In parallel, run only the `raster_flat` feature-fit smoke above to prove the
   visual observation path still fits.

That gives a real self-play boundary without making visual input the scapegoat
for the current training-signal problem.
