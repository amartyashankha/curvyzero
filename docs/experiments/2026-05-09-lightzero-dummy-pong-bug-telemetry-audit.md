# 2026-05-09 LightZero Dummy Pong Bug And Telemetry Audit

Worker D lane: broad bug and telemetry audit. This note uses local code/docs
first, plus a direct Modal Volume pull of the policy-head scoreboard artifacts.
No pytest was run.

## Bottom Line

The independent LightZero policy-head scoreboard is scoring a constant `up`
policy, not a meaningful learned Pong policy.

This is the highest-priority sanity failure. The checkpoint probe already showed
`policy_logits: [0.0, 0.0, 0.0]` for one default observation. The raw policy-head
scoreboard matchups confirm the behavior across full eval episodes:

| Matchup seat | `lightzero_best` action histogram `[up, stay, down]` |
| --- | ---: |
| `lightzero_best_p0__random_uniform_p1` | `[237, 0, 0]` |
| `random_uniform_p0__lightzero_best_p1` | `[314, 0, 0]` |
| `lightzero_best_p0__lagged_track_ball_1_p1` | `[450, 0, 0]` |
| `lagged_track_ball_1_p0__lightzero_best_p1` | `[428, 0, 0]` |
| `lightzero_best_p0__track_ball_p1` | `[437, 0, 0]` |
| `track_ball_p0__lightzero_best_p1` | `[536, 0, 0]` |

So the policy-head eval results are useful as a negative loader/action-collapse
canary only. They should not be read as evidence that LightZero learned even a
weak Pong policy.

## Evidence Read

- Pulled policy-head scoreboard summary:
  `eval/lightzero-dummy-pong/policy-head-scoreboard-20260509T143955Z/summary.json`.
- Pulled/inspected policy-head episodes:
  `eval/lightzero-dummy-pong/policy-head-scoreboard-20260509T143955Z/episodes.jsonl`.
- Read local train-smoke inspection artifacts under:
  `artifacts/inspections/lightzero-dummy-pong/lz-dpong-20260509T141607Z-3696aa333028/attempt-20260509T141607Z-98662e4917b4/`.
- Read the relevant code:
  `src/curvyzero/training/lightzero_dummy_pong_policy.py`,
  `src/curvyzero/training/lightzero_dummy_pong_env.py`,
  `src/curvyzero/training/dummy_pong_eval.py`, and
  `scripts/run_dummy_pong_lightzero_checkpoint_scoreboard.py`.

Small local command run:

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_eval.py \
  src/curvyzero/training/lightzero_dummy_pong_policy.py \
  src/curvyzero/training/lightzero_dummy_pong_env.py \
  scripts/run_dummy_pong_lightzero_checkpoint_scoreboard.py
```

Result: compile passed.

## Ranked Issues

### 1. Confirmed: greedy policy-head eval is constant `up`

The policy loader does exactly what it says: encode `tabular_ego`, call
`MuZeroModelMLP.initial_inference`, take `argmax(policy_logits)`, and return
that action. There is no action mask issue in the eval path; all actions are
legal. If logits are tied, Torch argmax picks index 0, which maps to `up`.

The probe saw one all-zero logit vector. The scoreboard action histograms show
`[N, 0, 0]` for every LightZero seat and opponent. That means the policy-head
scoreboard is mostly measuring the game behavior of "always up" under different
seats/seeds.

Evidence:

- `lightzero_dummy_pong_policy.py` uses
  `argmax(initial_inference(...).policy_logits)`.
- `docs/experiments/2026-05-09-lightzero-dummy-pong-checkpoint-probe.md`
  reports `policy_logits: [0.0, 0.0, 0.0]`, `action_id: 0`.
- The pulled policy-head summary's raw `matchups` report the six constant
  histograms listed above.

Next falsifier:

```sh
modal run -m curvyzero.infra.modal.lightzero_dummy_pong_checkpoint_probe \
  --checkpoint-ref training/lightzero-dummy-pong/lz-dpong-20260509T141607Z-3696aa333028/checkpoints/lightzero/ckpt_best.pth.tar \
  --probe-grid normal_resets_64
```

That command shape is aspirational: add a probe-grid option first, then log
policy logits and argmax actions over many real reset observations. The pass
condition is not win rate; it is nonzero logit variance and non-constant action
histograms.

### 2. Updated: policy-head scorecard is not MCTS; strict load now has a passing smoke

The current independent scorecard is explicitly `policy-head greedy, no MCTS`.
It does not call `lzero.policy.muzero.MuZeroPolicy.eval_mode.forward`, does not
run tree search, and does not validate LightZero evaluator parity.

The old strict-load failure on dynamics/config variants is no longer the
current blocker for the 512/8 `iteration_8` checkpoint. A newer MCTS loader
smoke passes strict full-model load and one eval-mode forward call.

Evidence:

- Policy metadata says `mcts_parity_claim: false`.
- `scripts/run_dummy_pong_lightzero_checkpoint_scoreboard.py` writes
  `mcts: False`.
- Passing probe:
  `training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/attempts/attempt-20260509T144635Z-ece79bad80d0/probe/lightzero_mcts_loader_smoke_20260509T145607Z.json`.
- Probe details: `ok: true`, `mcts_eval_status: ok`,
  `strict_full_model_load_ok: true`,
  `strict_full_model_load_variant: res_connection_in_dynamics_true`,
  call shape `data [1,10]`, `action_mask [[1,1,1]]`, `to_play [-1]`,
  `ready_env_id [0]`.
- Eval-mode forward returned action `0`, visits `[2,1,1]`, policy logits
  `[0.0170983, 0.00644484, 0.0132326]`, predicted value about `0.0000259`,
  and searched value about `0.000114`.

Next falsifier:

Run a true LightZero eval-mode/MCTS scoreboard on the same seed split. Compare
MCTS actions against the current constant policy-head actions.

### 3. High: the tiny training run is too small to mean anything

The train smoke proves that LightZero can call the custom env and write
checkpoints. It does not prove learning.

The run used `max_train_iter=2`, `max_env_step=64`, `collector_env_num=1`,
`evaluator_env_num=1`, `n_evaluator_episode=1`, `num_simulations=2`,
`batch_size=8`, and `update_per_collect=1`. Logs expose only
`training_iterations: [0]`, checkpoint iterations `[0, 2]`, and five env-side
terminal telemetry rows. The env-side rows show 4 wins / 1 loss against
`random_uniform`, but the independent policy-head eval says the exported
checkpoint is constant `up`.

So trainer-side telemetry and independent eval disagree in the expected way for
a tiny smoke: the former proves plumbing, the latter exposes action collapse.

Next falsifier:

Do not scale until eval is honest. Once the full MCTS/eval-mode scorecard
exists, run a small longer job with enough iterations to show a learning curve,
then score `iteration_0`, an intermediate checkpoint, and latest under the same
independent eval.

### 4. Medium: scoreboard rows hide the action-collapse evidence

The raw `matchups` include `action_histogram_by_policy`; the condensed
`scoreboard_rows` omit it. That is why the headline rows can look merely weak:

- vs `random_uniform`: `21/40`
- vs `lagged_track_ball_1`: `16/40`
- vs `track_ball`: `0/40`

But the raw action histograms show the model always chose `up`. The condensed
scoreboard should carry action histograms, at least for rows involving learned
or LightZero policies.

Evidence:

- `dummy_pong_eval.py` emits `action_histogram_by_policy` in `_summarize_records`.
- `scripts/run_dummy_pong_lightzero_checkpoint_scoreboard.py` builds
  `scoreboard_rows` from `pair_groups`, which do not include action histograms.
- Confirmed with `jq`: every LightZero `scoreboard_rows` entry lacks
  `action_histogram_by_policy`.

Next fix:

Extend `pair_groups` and/or `_scoreboard_rows` to aggregate and expose
`action_histogram_by_policy`. This is observability, not training behavior.

### 5. Medium: same-policy pair groups merge both seats under one label

This is confirmed and should be documented clearly. For same-policy rows such
as `random_uniform_vs_random_uniform`, both seats share one policy id. The
summary then aggregates both players under the same label:

- `random_uniform_p0__random_uniform_p1` had `wins_by_player` of
  `player_0: 14`, `player_1: 6`.
- The same row reports `wins_by_policy: {"random_uniform": 20}`.
- `score_return_stats_by_policy.count` is `40` because both players' returns
  are included across 20 episodes.

This is not a bug for different-policy rows, where labels are distinct. It is a
known interpretation trap for same-policy baseline sanity rows.

Next fix:

Add a small `same_policy_aggregates_both_seats: true` field or split
same-policy display into `wins_by_player` plus policy-level aggregate with a
plain note.

### 6. Medium: train wrapper has a stale independent-scorecard label

The train-smoke summary still says `independent_scorecard.status: blocked`
because at the time the wrapper only had env-side telemetry. A policy-head
greedy scorecard now exists, but it is not an MCTS scorecard and it is currently
constant-action.

This is stale documentation/metadata, not a training failure. The better status
would be something like:

```json
{
  "status": "policy_head_greedy_available_mcts_blocked",
  "policy_head_warning": "constant up in 20260509T143955Z scorecard"
}
```

Next fix:

Update the wrapper summary wording only after deciding whether train attempts
should know about post-hoc eval artifacts. Avoid making training summaries
pretend that MCTS eval exists.

## Env Wrapper Audit

The LightZero env wrapper mostly appears internally consistent:

- Action schema matches dummy Pong: `0=up`, `1=stay`, `2=down`.
- The wrapper validates ego action bounds.
- It supplies the opponent action from `random_uniform`, `track_ball`, or
  `lagged_track_ball_1`.
- It passes a simultaneous joint action to `PongEnv.step`.
- Reward is the ego player's raw dummy Pong reward: `+1`, `0`, or `-1`.
- `done` is `terminated or truncated`.
- `winner` is set only for terminal score events, not truncations.
- `max_steps` is passed into `PongConfig`.
- It records action counts, an action trace, trace hash, score return,
  survival fraction, and shaped loss-delay return in `info`.

The suspicious wrapper-level items are weaker than the policy-head collapse:

- `random_action()` reinitializes an RNG from the episode seed on every call,
  so repeated calls within an episode would return the same random action. I did
  not find evidence this path drives the reported policy-head scorecard.
- `to_play` is always `-1`. That may be correct for LightZero's single-agent
  non-board-game path, but it should be checked when implementing real MCTS
  eval.
- The default `DummyPongLightZeroEnv.config.max_steps` is `120`, while the tiny
  trainer used `64`. The artifacts record the actual configured value, so this
  is not itself a bug.

## Baseline Sanity

Baseline rows look sane once same-policy aggregation is understood:

- `track_ball_vs_track_ball` times out every episode, as expected for the
  current survival/tie floor.
- `random_uniform_vs_track_ball` gives `track_ball` `40/40` wins.
- Same-policy rows merge both seats under one label, so
  `random_uniform_vs_random_uniform` reports `random_uniform: 20` wins for 20
  episodes and 40 policy returns.

The important caveat is that policy-head scoreboard rows alone hide the most
important baseline/action fact because they omit action histograms.

## Next Experiments

1. Add a logits-grid checkpoint probe.
   - Run `initial_inference` on many real reset/rollout observations.
   - Log logits, softmax, argmax, margin, and action histogram.
   - Falsifies "policy head is dead/constant" if logits vary and actions are
     not collapsed.

2. Add action histograms to `scoreboard_rows`.
   - Keep raw `matchups`, but make the compact table impossible to misread.
   - This should be a small observability patch with no training behavior
     change.

3. Build full LightZero MCTS/eval-mode scorecard.
   - Resolve the dynamics key/config mismatch.
   - Instantiate `MuZeroPolicy` eval mode and run the same dummy Pong ladder.
   - Falsifies "only policy head is bad" if MCTS also collapses.

4. Score all mirrored checkpoints through the same diagnostic path.
   - Compare `iteration_0`, `iteration_2`, and `ckpt_best`.
   - If all are `[N, 0, 0]`, suspect initialization/load/config.
   - If only `ckpt_best` collapses, suspect checkpoint selection or training.

5. Run a no-training/untrained model control.
   - Same model config, random initialized weights, same argmax path.
   - If untrained and trained both produce all-zero logits or all-up actions,
     the eval reconstruction path is probably wrong.

6. After the above, run a longer tiny LightZero job.
   - Only do this after eval can distinguish action collapse.
   - Track checkpoint progression, logits/action entropy, wins, survival,
     shaped loss-delay, and replay/update counts.

## Code Edits

Follow-up observability patch made after the initial audit:

- `scripts/run_dummy_pong_lightzero_checkpoint_scoreboard.py` now aggregates
  raw `matchups[].action_histogram_by_policy` into compact
  `scoreboard_rows[].action_histogram_by_policy`.
- This does not change eval behavior or training. It only prevents the compact
  scoreboard rows from hiding action collapse.

Verification:

```sh
uv run python -m py_compile scripts/run_dummy_pong_lightzero_checkpoint_scoreboard.py
```

Local smoke against the pulled scorecard summary produced:

```text
lightzero_best_vs_random_uniform: [551, 0, 0]
lightzero_best_vs_lagged_track_ball_1: [878, 0, 0]
lightzero_best_vs_track_ball: [973, 0, 0]
```

## New 512/8 Evidence

A separate main-thread 512/8 LightZero MuZero dummy Pong run completed:

```text
run_id: lz-dpong-20260509T144635Z-eb5a0ed35de0
attempt_id: attempt-20260509T144635Z-ece79bad80d0
summary_ref: training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/attempts/attempt-20260509T144635Z-ece79bad80d0/train/summary.json
```

Trainer-side telemetry reported 42 episodes, 37 wins, 5 losses, no timeouts,
mean survival 9.0476, p90 survival 8.0, shaped mean 0.7633, score mean 0.7619,
and checkpoint iterations `[0, 8]`.

Independent policy-head scoreboard artifacts:

```text
eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-20260509T144736Z/summary.json
eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-20260509T144736Z/episodes.jsonl
```

Important rows:

| Checkpoint | Opponent | LightZero wins | Opponent wins | Truncations | Action histogram `[up, stay, down]` | Shaped mean | Mean steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ckpt_best` | `random_uniform` | 30/64 | 34/64 | 0 | `[864, 0, 0]` | -0.03047 | 13.5 |
| `iteration_8` | `random_uniform` | 31/64 | 33/64 | 0 | `[787, 0, 0]` | -0.00117 | 12.2969 |
| `ckpt_best` | `lagged_track_ball_1` | 21/64 | 35/64 | 8 | `[1595, 0, 0]` | -0.1919 | not recorded here |
| `iteration_8` | `lagged_track_ball_1` | 29/64 | 35/64 | 0 | `[677, 0, 0]` | -0.0691 | not recorded here |
| `ckpt_best` | `track_ball` | 0/64 | 60/64 | 4 | `[1510, 0, 0]` | -0.8704 | not recorded here |
| `iteration_8` | `track_ball` | 0/64 | 53/64 | 11 | `[2217, 0, 0]` | -0.7697 | 34.6406 |

Read:

This confirms the earlier bug audit. Trainer-side env/evaluator wins are not a
policy-quality proof. Independent checkpoint eval says the
exported/reconstructed greedy policy-head is constant-up. A strict-config
direct policy-head rerun after the loader fix kept `load_state_dict` strict
true, but still produced constant-up behavior:

```text
eval_id: policy-head-scoreboard-512x8-strictcfg
modal_url: https://modal.com/apps/modal-labs/shankha-dev/ap-Q7sPmscebJQWisowuweBxV
summary: eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-strictcfg-20260509T145841Z/summary.json
episodes: eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-strictcfg-20260509T145841Z/episodes.jsonl
lagged: action [590,0,0], 13 wins vs 17, 2 truncs, mean_steps 18.4375, shaped -0.0987
random: action [388,0,0], 12 wins vs 20, mean_steps 12.125, shaped -0.2134
track: action [968,0,0], 0 wins vs 28, 4 truncs, mean_steps 30.25, shaped -0.8115
```

The old missing-`cfg.policy.device` MCTS loader failure is stale;
device/action-mask issues were fixed. Do not scale more until the full
MCTS/eval-mode scorecard exists.

MCTS loader smoke on 512/8 `iteration_8` now passes:

```text
training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/attempts/attempt-20260509T144635Z-ece79bad80d0/probe/lightzero_mcts_loader_smoke_20260509T145607Z.json
```

Details: `ok: true`, `mcts_eval_status: ok`,
`strict_full_model_load_ok: true`, strict-load variant
`res_connection_in_dynamics_true`, call shape `data [1,10]`,
`action_mask [[1,1,1]]`, `to_play [-1]`, `ready_env_id [0]`, action `0`,
visits `[2,1,1]`, policy logits `[0.0170983, 0.00644484, 0.0132326]`,
predicted value about `0.0000259`, searched value about `0.000114`.
