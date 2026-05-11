# Pong Angle Learning Next Steps - 2026-05-09

Scope: diagnostic angle/contact work for Pong.

Status update: the angle-control and contact-outcome probes have both run. They
are useful observability tools, not the main eval scoreboard. The main Pong
scoreboard is learned checkpoints versus `random_uniform`, `track_ball`, and
later older/best checkpoints.

Second status update: the first short-lookahead replay builder now exists. It
can relabel raster states by score-delta lookahead, and an angle-control
tie-break can create non-`track_ball` labels on equal-return states. The first
trained smoke still got 0 wins against `track_ball`, with only 2/8 truncations,
so this is pressure evidence at best, not a solved policy-improvement step.

Third status update: the larger angle-tie run produced 1,669 labels and 442
targets different from `track_ball`, but all lookahead checkpoints still won
0/64 against `track_ball`, and the selector kept the older imitation epoch-1000
checkpoint. Do not keep scaling this one-step label objective unless a bug is
found.

Fourth status update: a loss-delay target is now implemented for lookahead
labels. It gives a small training-only bonus to candidates that lose later, but
the first smoke with alpha `0.05` and `track_ball` tie-break still produced 0
targets different from `track_ball` and 0/32 learned wins against `track_ball`.
Do not scale that exact setting.

Fifth status update: depth-2 ego-sequence lookahead is now implemented. The
first strict smoke produced 10 non-tied avoided-loss rows, but all target first
actions still matched `track_ball`, so it was not trained.

Current contact-outcome result: top/center/bottom contacts changed outgoing
`ball_vy`, but short score-delta returns stayed flat against `track_ball` in
the default geometry. Do not fit a chooser from this until the contact rows show
score signal.

Sixth status update: the first LightZero dummy Pong contact-pressure reset
curriculum is implemented and ran as a tiny Modal smoke. This is custom dummy
Pong curriculum work, not stock Atari Pong benchmark replication. The opt-in
knobs are `pong_reset_profile=contact_pressure` and
`pong_reset_pressure_agent=ego` for training; matching player-0-only MCTS eval
uses `pong_reset_pressure_agent=player_0`. `env.step()` still returns only the
true sparse score reward `+1/-1/0`. Modal train
`ap-bNRz3Mtil6apjX5w6tNZxa` and scorecard `ap-XRyCAYWAN7F3ptvRAKRC0x` passed
mechanically. Trainer-side telemetry was non-flat, but held-out
`iteration_2` still collapsed with zero down actions and lost to lagged/random;
this is not a policy-quality win.

## Next Proper Run After Modal Wrapper

After `curvyzero.infra.modal.dummy_pong_train_attempt` exists, run one small CPU
Modal train attempt that uses the repaired self-play trainer only as a guarded
baseline, not as a promoted generation:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_train_attempt \
  --games 64 \
  --max-steps 120 \
  --policy random_uniform \
  --epsilon 0.05 \
  --epochs 100 \
  --policy-learning-rate 0.05 \
  --value-learning-rate 0.001 \
  --action-diversity-beta 0.02 \
  --checkpoint-every-epochs 25 \
  --seed 101
```

Then score the periodic checkpoints remotely on a monitor split with the
existing scoreboard wrapper:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --checkpoints ckpt25=ref:training/dummy-pong/<run_id>/attempts/<attempt_id>/train/checkpoints/epoch-000025/checkpoint.npz,ckpt50=ref:training/dummy-pong/<run_id>/attempts/<attempt_id>/train/checkpoints/epoch-000050/checkpoint.npz,ckpt75=ref:training/dummy-pong/<run_id>/attempts/<attempt_id>/train/checkpoints/epoch-000075/checkpoint.npz,ckpt100=ref:training/dummy-pong/<run_id>/attempts/<attempt_id>/train/checkpoints/epoch-000100/checkpoint.npz \
  --episodes 32 \
  --seed 313 \
  --split-id dummy_pong_modal_selfplay_repair_monitor \
  --split-role monitor
```

Progress is a scoreboard row, not an angle row. A useful candidate must beat or
tie its parent/earlier checkpoint, keep `random_uniform` wins respectable, and
show at least one `track_ball` improvement: learned wins, fewer `track_ball`
wins, or more truncations at the same seed count. If the best checkpoint still
wins 0/64 against `track_ball` with no loss reduction, stop and change the
learner/curriculum before adding games.

Required train summary fields for this run: replay rows/games, terminal causes,
action histograms by seat, predicted action histograms, mean policy entropy,
score/shaped return stats, periodic checkpoint refs, and Volume refs. Required
eval rows: baseline sanity (`track_ball` sweeps `random_uniform`,
`track_ball` vs `track_ball` truncates), every checkpoint vs `random_uniform`,
every checkpoint vs `track_ball`, and learned-vs-learned periodic rows.

## Input Signal

Use the angle-control eval to create a tiny contact-outcome dataset:

- raster observation before the controllable paddle contact;
- ego paddle offset at contact: above center, center, or below center;
- outgoing `ball_vy` after contact;
- short score-delta return after the contact, using only `+1`, `0`, or `-1`;
- optional current value prediction for diagnostics, not as the training label.

The learner should consume score-delta returns attached to contact choices. The
contact labels are useful for filtering and debugging, but progress should be
judged by whether choosing an offset improves later score outcomes.

## Smallest Policy Improvement Experiment

Build a tiny one-step contact chooser for states where the ego paddle is about
to hit the ball.

For each eligible state, evaluate three scripted micro-actions: aim for top,
center, or bottom contact. Roll out each choice for a short fixed horizon
against `track_ball`, record the score-delta return, and label the best contact
choice. Train or table-fit a tiny policy head from raster observations to that
best-contact label.

Then run eval against `track_ball` and `random_uniform`:

- default to `track_ball` behavior away from contact;
- near contact, use the learned or table-fit contact chooser;
- compare against pure `track_ball` on the same recorded eval wave.

This is deliberately closer to short search plus supervised distillation than
full RL. It only asks whether off-center contact selection can be learned and
used without changing the reward.

## Avoid For Now

- Full MuZero, MCTS infrastructure, Mctx, or learned dynamics.
- Environment/eval reward shaping for paddle hits, survival, distance, or rally
  length. Training-only loss-delay targets are allowed as a diagnostic, but only
  if they create labels or value targets that move the scoreboard.
- Cloning random-action rows as expert policy targets.
- More one-step angle-tie label scaling in the default geometry.
- Large Modal runs or broad seed sweeps before the local signal is clear.
- Treating value predictions as proof of policy improvement.

## Artifacts To Write

Use one compact output directory per attempt:

- `summary.json`: config, seeds, baseline scores, chooser scores, contact
  counts, off-center contact rate, and win/loss/draw counts.
- `contact_rows.jsonl`: one row per evaluated contact state, with raster or
  raster reference, contact option, chosen label, rollout return, and outcome.
- `policy_checkpoint.npz` or `chooser_table.json`: only if a chooser is fit.
- `eval_episodes.jsonl`: seeded comparison episodes against `track_ball` and
  `random_uniform`.

## Progress

Counts as progress:

- more off-center contacts than pure `track_ball` without a large scoring drop;
- paired-wave improvement against `track_ball`, even if small;
- better score-delta returns on contact states than always-center contact;
- artifacts that make failed contact choices inspectable.
- for future lookahead labels: lower `track_ball` wins, more truncations without
  random-opponent collapse, or actual learned wins against `track_ball`.

Does not count as progress:

- higher value fit with no better policy behavior;
- wins caused only by random opponent mistakes;
- more off-center contacts that immediately lose points;
- a larger training loop that cannot explain which contact choices helped.
