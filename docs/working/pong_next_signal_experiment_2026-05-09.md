# Pong Next Signal Experiment - 2026-05-09

Status: historical diagnostic note. The current active spine is
`docs/working/pong_selfplay_training_plan_2026-05-09.md` plus
`docs/working/pong_training_critique_wave_2026-05-09.md`. Self-play replay plus
shaped episode returns exists locally, but it is under critique after gen2 lost
to the parent and won 0 games against `track_ball`.

Scope: decide why the current dummy Pong learner is not improving, and name the
next smallest experiments that should produce a real learning signal. No code is
implemented here.

## Short Answer

The lack of improvement is probably not from too little training.

The bigger problems are:

- The current labels mostly do not say how to score on `track_ball`.
- The default Pong geometry is too forgiving for one contact or one action to
  create a score difference.

Eval noise exists for small random and learned-vs-learned comparisons, but it
does not explain the main result. The `track_ball` gate has been stable: learned
policies keep winning 0/64 against `track_ball`.

## Evidence

Longer imitation training did change behavior. Epoch 1000 beat epoch 750 on the
heldout direct checkpoint row and survived longer against `track_ball`.
However, it still won 0/64 against `track_ball`.

The larger lookahead angle-tie run also did not fix it:

- 1,669 replay labels.
- 442 targets differed from `track_ball`.
- 1,532 rows had all actions tied and used `angle_control` as a tie-break.
- 58 rows had a unique best score-return action.
- 0 positive-return target rows.
- Every lookahead checkpoint won 0/64 against `track_ball`.
- The selector chose the older imitation checkpoint over every lookahead
  checkpoint.

The first loss-delay lookahead smoke also did not fix it:

- 251 replay labels.
- 0 targets differed from `track_ball`.
- The trained checkpoint won 0/32 against `track_ball`.
- `track_ball` won 27/32, with 5/32 truncations.

This means the loss-delay code path works, but alpha `0.05` with `track_ball`
tie-break mostly teaches another `track_ball` copy.

The contact-outcome smoke showed the geometry issue clearly:

- Top, center, and bottom contacts changed outgoing `ball_vy`.
- But all 12 candidate rows had score-delta return `0.0`.
- All 4 same-state `track_ball` baseline rows also had return `0.0`.
- No sampled state had a score-delta difference between contact choices.

The width-9 contact-outcome probe repeated the same issue:

- 64 controlled near-contact states.
- 192 candidate contact rows.
- Outgoing `ball_vy` differed for all 64 states.
- Every score-delta return was still `0.0`.
- `score_delta_return_differs_state_count` was still `0`.

So the learner can fit labels, and the eval harness can detect wins, but the
current target construction rarely creates useful winning labels.

## Diagnosis

Not training long enough: unlikely. Training for 1000 epochs and larger
lookahead data did not produce wins against `track_ball`.

Bad labels/objective: likely. Plain imitation copies `track_ball`, which cannot
beat itself. All-ego action copying mixes good scripted actions with random
actions. One-step angle-tie labels mostly come from tied score returns, so many
labels are preferences, not evidence that the action scores. Loss-delay labels
with `track_ball` tie-break created no new actions at all.

Geometry too forgiving for `track_ball`: likely. The default width 15,
height 9, paddle height 3 game lets `track_ball` recover from off-center hits.
One contact changes angle, but it usually does not create a score event within
the short horizon. Width 9 alone did not fix this.

Eval noise: not the main issue. Some random-baseline and learned-vs-learned rows
move around by seed, but repeated 0/64 learned wins against `track_ball` is not
a small-seed artifact.

## Historical Experiment 1: Geometry Signal Probe

Goal: find the smallest Pong geometry where contact choice creates score
differences against `track_ball`.

Do this before training another policy. If top, center, and bottom contacts
still all return zero, a learner has no useful score label to learn.

Tiny script change:

- In `scripts/build_dummy_pong_contact_outcomes.py`, add CLI args:
  `--width`, `--height`, and `--paddle-height`.
- Pass them into `build_dummy_pong_contact_outcomes`.
- In `build_dummy_pong_contact_outcomes`, build:
  `PongConfig(width=width, height=height, paddle_height=paddle_height, max_steps=horizon)`.
- Keep default values equal to current behavior: width 15, height 9,
  paddle height 3.
- Keep `_validate_config` strict enough for top/center/bottom contacts. For the
  first pass, use paddle height 3.

Command after that change:

```sh
uv run python -m py_compile scripts/build_dummy_pong_contact_outcomes.py
```

```sh
uv run python scripts/build_dummy_pong_contact_outcomes.py \
  --states 64 \
  --seed 17 \
  --horizon 48 \
  --width 9 \
  --height 9 \
  --paddle-height 3 \
  --output-dir artifacts/local/dummy-pong-contact-outcomes-width9-h48-2026-05-09
```

Success signal:

- `score_delta_return_differs_state_count` is above zero.
- `best_candidate_impact_offset_histogram` is not only `all_tied`.
- Some states show one contact choice scoring or avoiding a loss while another
  choice does not.

If width 9 is too harsh or too noisy, try width 11 with the same command shape.

## Historical Experiment 2: Depth-2 Ego Lookahead

Goal: test whether the failure is only that one immediate action is too short.

The current lookahead tries `up`, `stay`, and `down` for the ego's first action,
then returns to `track_ball`. That may be too weak because aiming for an
off-center contact often needs two ego moves.

Tiny script change:

- In `scripts/build_dummy_pong_lookahead_replay.py`, add
  `--ego-sequence-depth` with default `1`.
- Thread it into `build_dummy_pong_lookahead_replay`.
- In `src/curvyzero/training/dummy_pong_lookahead_replay.py`, when depth is 2,
  evaluate all 9 ego action sequences:
  `(up, up)`, `(up, stay)`, `(up, down)`, `(stay, up)`, and so on.
- Keep the opponent as `track_ball` on both forced steps.
- After the forced ego sequence, roll out both agents with `track_ball` as
  before.
- Use the first action of the best sequence as `target_action_id`.
- Add the chosen sequence and sequence returns to each replay row for debugging.

Command after that change:

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_lookahead_replay.py \
  scripts/build_dummy_pong_lookahead_replay.py
```

```sh
uv run python scripts/build_dummy_pong_lookahead_replay.py \
  --games-per-seat 32 \
  --seed 19 \
  --max-steps 120 \
  --lookahead-steps 48 \
  --ego-sequence-depth 2 \
  --collector-policy random_uniform \
  --output-dir artifacts/local/dummy-pong-lookahead-depth2-strict-g32-h48-2026-05-09
```

Only train if this replay has real signal:

- more than a token number of rows;
- nonzero `return_spread` on a meaningful share of rows;
- at least some positive-return target rows, or clear avoided-loss labels;
- not mostly tied rows.

Training and scoreboard command if the replay passes that check:

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lookahead-depth2-strict-g32-h48-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-lookahead-depth2-policy-e1000-2026-05-09 \
  --seed 0 \
  --epochs 1000 \
  --learning-rate 0.5 \
  --validation-fraction 0.2 \
  --checkpoint-every-epochs 250
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 32 \
  --seed 317 \
  --split-id dummy_pong_lookahead_depth2_monitor \
  --split-role monitor \
  --checkpoint depth2_250=artifacts/local/dummy-pong-lookahead-depth2-policy-e1000-2026-05-09/checkpoints/epoch-000250/checkpoint.npz \
  --checkpoint depth2_500=artifacts/local/dummy-pong-lookahead-depth2-policy-e1000-2026-05-09/checkpoints/epoch-000500/checkpoint.npz \
  --checkpoint depth2_750=artifacts/local/dummy-pong-lookahead-depth2-policy-e1000-2026-05-09/checkpoints/epoch-000750/checkpoint.npz \
  --checkpoint depth2_1000=artifacts/local/dummy-pong-lookahead-depth2-policy-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz \
  --checkpoint imitation1000=artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lookahead-depth2-scoreboard-monitor-2026-05-09
```

Success signal:

- Any learned wins against `track_ball`, even a few.
- Or, weaker but useful: fewer `track_ball` wins than imitation1000 on the same
  scoreboard, without collapsing against `random_uniform`.
- Or, before training: a replay summary showing many untied, score-bearing
  sequence labels.

## Recommendation

Experiment 1 has now run for width 9 and was negative. If we keep testing
geometry, use a stronger change than width 9 alone, such as width 7, faster ball,
or smaller paddle only after the contact probe supports that config.

If the default geometry must stay unchanged, run Experiment 2 next. Depth-2 ego
lookahead is the smallest objective change that tests whether off-center scoring
needs two setup actions rather than one.

Do not keep scaling one-step angle-tie labels unless a bug is found.
Do not keep scaling loss-delay alpha `0.05` with `track_ball` tie-break; it did
not create a new policy target.

Experiment 2 has now run as a strict smoke. It produced 10 non-tied avoided-loss
rows from 65 sampled states, but all target first actions still matched
`track_ball` and all chosen target returns were `0.0`. Do not train that small
strict replay. The next training run needs a stronger signal first: either a
geometry/contact probe with actual score differences, or a target builder that
can produce non-`track_ball` first actions for avoided losses or scoring setups.
