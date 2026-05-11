# LightZero Dummy Pong Seating And Perspective

Date: 2026-05-09

This note explains what seating means in the current dummy Pong LightZero setup.
No pytest was run.

## Short Answer

Current LightZero training controls one ego paddle. With the default config that
ego paddle is `player_0`. The wrapper supplies `player_1` from a fixed opponent
policy such as `random_uniform`.

The independent scorecard is stricter than training. It evaluates the same
checkpoint in paired seats: checkpoint as `player_0` and checkpoint as
`player_1` against the same baseline policy family. That is good for detecting
seat/perspective problems, but it is not the same distribution as the current
training run.

The `tabular_ego` observation is partly ego-normalized. It flips horizontal ball
features into "forward from ego" coordinates, so `player_0` and `player_1` can
share one policy in principle. It is not perfectly seat-canonical because the
absolute paddle x fields remain in the vector. So a policy trained only as
`player_0` should often transfer, but `player_1` performance is still held-out
seat generalization, not a guaranteed training-seat result.

Update after the post-seed-fix `iteration_16` player_0-only MCTS scorecard:
the training-seat control does not rescue the checkpoint. Seat pairing was not
hiding a good player_0 policy. The checkpoint still collapses to no action
index 0 and heavy action index 2.

Player_0-only refs:

- Modal:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-BM3gHko0cO0SbaU2rQwbLl`
- Summary:
  `training/lightzero-dummy-pong/lz-dpong-20260509T153355Z-0ea60caea3e3/attempts/attempt-20260509T153355Z-f981d8701b03/eval/mcts-scoreboard-post-seed-fix-1024x16-iter16-player0-only/summary.json`
- Episodes:
  `training/lightzero-dummy-pong/lz-dpong-20260509T153355Z-0ea60caea3e3/attempts/attempt-20260509T153355Z-f981d8701b03/eval/mcts-scoreboard-post-seed-fix-1024x16-iter16-player0-only/episodes.jsonl`

Rows:

| Opponent | LZ wins | Opp wins | Mean steps | Truncs | LZ actions |
| --- | ---: | ---: | ---: | ---: | --- |
| `lagged_track_ball_1` | 16 | 16 | 10.406 | 0 | `[0,60,273]` |
| `random_uniform` | 15 | 17 | 12.812 | 0 | `[0,97,313]` |
| `track_ball` | 0 | 28 | 142.219 | 4 | `[0,547,4004]` |

## What The Code Does

### Base Pong Observation

`PongEnv.observation(ego_agent)` builds a `PongObservation` for either agent.
For `player_0`, `forward = 1`. For `player_1`, `forward = -1`.

The ego-normalized fields are:

- `ball_dx_forward = (ball_x - ego_paddle_x) * forward`
- `ball_vx_forward = ball_vx * forward`
- `ball_dy_from_ego_center = ball_y - ego_paddle_center_y`
- `ego_paddle_y` and `opponent_paddle_y` are named by role, not by fixed player id

That means "ball in front of me moving toward/away from me" has the same sign
convention for both seats.

The not-fully-normalized fields are:

- `ego_paddle_x`
- `opponent_paddle_x`
- `ball_y`
- `step`

The x fields leak the physical side of the arena. In this simple Pong game that
is probably harmless if the model uses the forward ball features, but it means
the observation is not a perfect proof of seat invariance.

### LightZero Training Wrapper

`DummyPongLightZeroEnv` defaults to:

- `feature_mode = "tabular_ego"`
- `ego_agent = "player_0"`
- `opponent_policy = "random_uniform"`

Its `step(action)` treats the LightZero action as the ego action, asks the fixed
opponent policy for the other paddle's action, then calls the simultaneous
`PongEnv.step(joint_action)`.

Unless `ego_agent` is explicitly overridden in the LightZero env config,
training controls only `player_0`.

### Independent Scorecard

`dummy_pong_eval.run_dummy_pong_eval()` creates paired seatings for each
different-policy matchup:

- checkpoint as `player_0`, baseline as `player_1`
- baseline as `player_0`, checkpoint as `player_1`

For each controlled agent, the policy receives `observations[agent]`. The
current LightZero policy adapters then encode `tabular_ego` from that
observation. They do not hard-code `player_0` at action time.

So yes: current training is `player_0` by default, and current independent eval
also tests the checkpoint as `player_1`.

## Is One Policy Supposed To Work In Both Seats?

Mostly yes, but do not overclaim it.

Reasons it should work:

- The action schema is the same for both seats: `0=up`, `1=stay`, `2=down`.
- Vertical movement has the same meaning for both seats. There is no left/right
  action that would need seat-dependent remapping.
- The most important horizontal ball features are ego-normalized with
  `ball_dx_forward` and `ball_vx_forward`.
- The LightZero adapters encode the observation passed for the controlled
  agent, so when the checkpoint is seated as `player_1`, it receives a
  `player_1` ego observation.

Reasons it is still a generalization test:

- Training currently only samples `player_0` ego episodes by default.
- `tabular_ego` still includes absolute x positions, so the model can learn
  seat-correlated features.
- The scorecard's paired rows aggregate training-seat and held-out-seat
  behavior unless the raw matchup rows are inspected.

Practical read: a healthy learned policy should not collapse when moved to
`player_1`, but a `player_1` failure is not automatically the same as a
training failure. It could be seat generalization, perspective encoding, or a
real policy-quality problem.

## Could A Perspective Bug Make `down` Disappear?

It is possible in principle, but the current evidence points away from a simple
seat/action mapping bug.

Evidence against an action mapping bug:

- Dummy Pong defines `ACTION_LABELS = ("up", "stay", "down")`.
- `PongEnv._move_paddle()` uses `delta = action - 1`, so action `2` moves the
  paddle down for both players.
- `DummyPongLightZeroEnv` uses `Discrete(3)` and an all-ones action mask. It
  does not mask out `down`.
- The independent LightZero adapters return the action id unchanged after
  encoding `tabular_ego`.
- Baseline scorecard rows show nonzero `down` counts, so the env, scorecard, and
  action histogram plumbing can represent and count down actions.
- The longer trainer-side telemetry had some `player_0` down actions
  (`[9539, 800, 170]`), so the training wrapper itself is not physically unable
  to issue action `2`.

Evidence that still keeps this on the watchlist:

- Independent MCTS scorecards for LightZero checkpoints still show zero `down`
  in held-out eval rows, even after horizon fixes.
- The direct policy-head path can collapse tied or weak logits to action `0`
  because argmax chooses the first index.
- MCTS eval-mode also heavily prefers `up`/`stay`, so the issue is not only the
  direct policy-head argmax path.

Practical read: zero `down` should be reported as action collapse or
trainer/eval mismatch until proven otherwise. It is not currently ruled in as a
seat perspective bug.

## Next Scorecard Shape

Implemented option: `run_dummy_pong_eval(..., paired_seats=True)` remains the
default paired-seat scorecard. Passing `paired_seats=False` changes
checkpoint-vs-baseline pair groups to only run:

- checkpoint as `player_0`
- baseline as `player_1`

That is the player-0 training-seat control row for the current LightZero setup.
It deliberately avoids averaging that row with checkpoint-as-`player_1` held-out
seat transfer.

The option is threaded through the local eval/scoreboard scripts with
`--no-paired-seats`, and through the LightZero policy-head and MCTS Modal
scoreboard wrappers as `paired_seats=False`. Summaries now record both
`paired_seats` and an `eval_seating` object; split metadata mirrors the selected
`paired_seat` value when a split is present.

Preferred reporting shape includes both:

1. A `player_0`-only row that matches the current training seat.
2. Paired held-out rows that keep swapping the checkpoint into `player_0` and
   `player_1`.

The `player_0`-only row answers: "Does the exported checkpoint work in the same
seat it trained in?"

The paired rows answer: "Does the same checkpoint survive seat transfer, or is
it using player/side artifacts?"

Those are different questions. Keeping both prevents a real `player_0` learning
signal from being hidden by `player_1` transfer failure, and prevents a
training-seat-only policy from being mistaken for robust learning.

Current read for post-seed-fix `iteration_16`: the player_0-only row answered
the training-seat question, and the answer was still negative.

## Reporting Rules

Docs and scorecards should say:

- Training ego seat: currently `ego_agent=player_0` unless explicitly changed.
- Evaluation seating: paired scorecards test both checkpoint-as-`player_0` and
  checkpoint-as-`player_1`.
- Observation mode: `tabular_ego` is ego-oriented but not perfectly
  seat-invariant because absolute x fields remain.
- Action schema: `0=up`, `1=stay`, `2=down`.
- Action histograms must be reported by policy and, where possible, by seat.
- Do not report only an averaged paired win rate when diagnosing learning.
- Treat `player_0`-only rows as training-seat checks.
- Treat swapped-seat rows as perspective/generalization checks.
- Treat zero `down` as an action-collapse or wiring alarm until action
  histograms show both vertical directions in independent eval.

Plain-language claim to use in reports:

> The current LightZero dummy Pong trainer is a single-ego `player_0` setup.
> The independent scorecard intentionally evaluates the checkpoint in both
> seats. Because `tabular_ego` is mostly but not perfectly seat-normalized,
> paired-seat results are a useful robustness check, while a separate
> `player_0` row is needed to match the training distribution.
