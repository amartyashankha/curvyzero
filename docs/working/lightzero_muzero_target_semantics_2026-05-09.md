# LightZero MuZero Target Semantics - 2026-05-09

## Question

Why can the dummy Pong trainer-side telemetry show exploration over all actions
while held-out LightZero MCTS learning/eval uses zero `down`?

No pytest was run. This note is a source read of local project code plus the
local LightZero checkout at `/tmp/lightzero-src`.

## Short Answer

LightZero MuZero policy targets are MCTS root visit distributions, not the raw
exploratory action that was selected and stepped.

The executed action still matters, but for different parts of learning:

- it determines the next observation and reward stored in replay;
- it is used as the action sequence for recurrent dynamics unrolls;
- it is not the supervised policy label.

So trainer-side action diversity can be real while learned/eval action diversity
is absent. Temperature, epsilon, and random warmup can make collection execute
`down`, but the policy head is trained to match `visit_count_distributions`. If
MCTS put little or no visit mass on `down`, the target policy also says little
or nothing about `down`. Held-out eval then removes collect noise/sampling and
does deterministic argmax over eval MCTS visits.

## Source Files Read

LightZero source:

- `/tmp/lightzero-src/lzero/entry/train_muzero.py`
- `/tmp/lightzero-src/lzero/entry/eval_muzero.py`
- `/tmp/lightzero-src/lzero/entry/utils.py`
- `/tmp/lightzero-src/lzero/policy/muzero.py`
- `/tmp/lightzero-src/lzero/policy/random_policy.py`
- `/tmp/lightzero-src/lzero/policy/utils.py`
- `/tmp/lightzero-src/lzero/policy/scaling_transform.py`
- `/tmp/lightzero-src/lzero/worker/muzero_collector.py`
- `/tmp/lightzero-src/lzero/worker/muzero_evaluator.py`
- `/tmp/lightzero-src/lzero/mcts/buffer/game_segment.py`
- `/tmp/lightzero-src/lzero/mcts/buffer/game_buffer.py`
- `/tmp/lightzero-src/lzero/mcts/buffer/game_buffer_muzero.py`
- `/tmp/lightzero-src/zoo/classic_control/cartpole/config/cartpole_muzero_config.py`
- `/tmp/lightzero-src/zoo/atari/config/*muzero*config.py`

Project source/docs:

- `src/curvyzero/training/dummy_pong.py`
- `src/curvyzero/training/dummy_pong_eval.py`
- `src/curvyzero/training/lightzero_dummy_pong_env.py`
- `src/curvyzero/training/lightzero_dummy_pong_policy.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_scoreboard_attempt.py`
- `docs/working/lightzero_pong_eval_action_collapse_debug_2026-05-09.md`
- `docs/working/lightzero_pong_action_collapse_bug_hunt_2026-05-09.md`
- `docs/working/lightzero_pong_scorecard_plan_2026-05-09.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-upc25-epscollect-run.md`
- `docs/working/training_state_index_2026-05-09.md`

## Local Action Contract

Dummy Pong has a fixed three-action map:

```text
0 = up
1 = stay
2 = down
```

`PongEnv._move_paddle` applies `delta = action - 1`, so `2` really moves the
paddle down. `DummyPongLightZeroEnv` exposes a `Discrete(3)` action space,
`legal_actions = [0, 1, 2]`, and returns `action_mask = [1, 1, 1]` with
`to_play = -1`.

The held-out MCTS adapter also passes an all-ones action mask and `to_play=[-1]`
to `MuZeroPolicy.eval_mode.forward`. That makes action-mask/action-map bugs low
probability. Baseline policies also emit all three actions.

## Collect Path

The training loop constructs collect kwargs on each iteration:

- `temperature = visit_count_temperature(...)`;
- `epsilon = epsilon_greedy_fn(collector.envstep)` only if
  `eps_greedy_exploration_in_collect=True`;
- otherwise epsilon is `0.0`.

In `MuZeroPolicy._forward_collect`, LightZero:

1. runs `initial_inference`;
2. builds legal actions from the action mask;
3. adds root Dirichlet noise;
4. runs MCTS;
5. reads `roots.get_distributions()`;
6. chooses the executed action.

Normal collect samples the executed action from the visit counts using
temperature. Epsilon collect first takes deterministic argmax over visits, then
may replace that executed action with a random legal action. In both cases, the
output still contains the original MCTS `visit_count_distributions`.

The collector then sends `output["action"]` to `env.step`, but stores
`output["visit_count_distributions"]` through `GameSegment.store_search_stats`.
`store_search_stats` normalizes those visits into `child_visit_segment`.

## Random Warmup

`random_collect_episode_num` calls `random_collect(...)`, swaps in
`LightZeroRandomPolicy.collect_mode`, collects that many episodes, pushes them
into replay, then restores the real collect policy.

The important detail: `LightZeroRandomPolicy` still runs MCTS and returns real
MCTS visit distributions. Its source explicitly randomizes only `action`; the
other fields are MCTS-derived. Therefore random warmup diversifies visited
states and rewards, but it does not create behavioral-cloning targets for the
random actions.

If random warmup executes `down` but the untrained MCTS visit distribution gives
no mass to `down`, the replay row can contain an executed `down` action and a
policy target with zero `down`.

## Replay And Targets

`MuZeroGameBuffer.sample` returns:

- `current_batch`: observations, executed action unrolls, masks, indices,
  weights;
- `target_batch`: target rewards, target values, target policies.

Reward/value targets are n-step returns: collected rewards up to `td_steps`,
plus a bootstrapped target-model value when the bootstrap state is valid.
`env_type` matters here:

- `not_board_games` uses single-agent reward signs as-is;
- `board_games` applies to-play sign handling for alternating-player games.

Policy targets are MCTS visits:

- with `reanalyze_ratio=0`, non-reanalyzed targets are the stored
  `child_visit_segment`;
- with `reanalyze_ratio>0`, LightZero reruns MCTS with the current target model
  and replaces the policy target with fresh visit distributions.

For dummy Pong, `env_type="not_board_games"` and
`action_type="fixed_action_space"` are the appropriate LightZero branch:
single ego, fixed three-action vector, all actions legal. `action_type` matters
when mapping varied legal-action lists into full policy vectors; it should not
be the reason `down` is absent here.

The MuZero learner then computes:

```text
policy_loss = cross_entropy(policy_logits, target_policy)
```

So the model is explicitly trained toward visit distributions, not raw sampled
actions.

## Eval Path

Eval is deliberately more deterministic than collect:

- `MuZeroPolicy._forward_eval` uses `roots.prepare_no_noise(...)`;
- it runs MCTS with the eval model;
- it calls `select_action(..., temperature=1, deterministic=True)`;
- deterministic selection is `np.argmax(visit_counts)`.

`np.argmax` breaks exact ties by choosing the lowest index among tied actions.
With 3 legal actions and tiny search counts, ties like `[2, 3, 3]` select
`stay`, not `down`; `[3, 3, 2]` selects `up`. That matches the recent debug docs:
low-simulation roots were often broad/tied, and higher eval sims sometimes
changed the collapsed action rather than creating a useful mixed controller.

## What Each Knob Actually Does

`random_collect_episode_num`

Adds initial replay episodes with random executed actions. It changes state and
reward coverage. It does not make policy targets equal random actions.

`epsilon`

Only affects collect-time executed action after search. It changes which
branches of the environment enter replay and can improve value/dynamics data.
It does not directly change the policy target for the root where epsilon fired.

`temperature`

Controls collect-time sampling from visit counts. Higher temperature can execute
more diverse actions from the same root visits. The stored target remains the
visit distribution, not the sampled action.

`num_simulations`

Controls how many MCTS visits exist in collect, reanalysis, and eval. In
training collect, it directly shapes the stored policy targets. In held-out
eval, it shapes the action chosen by no-noise deterministic MCTS. Very low sims
make targets and eval decisions coarse: a useful action can receive exactly
zero target probability, and exact ties become common.

`env_type`

Mostly affects value-target sign/bootstrapping semantics. Dummy Pong should stay
`not_board_games`: it is a single-ego fixed-action wrapper over a two-player
environment. Setting board-game semantics would be a different algorithmic
claim and could corrupt value signs. It is not an imitation-vs-visits switch.

## Why Collect Diversity Can Fail To Become Learned Diversity

Collect diversity has to influence learning indirectly. It must create states
where rewards/value learning make MCTS visit the useful actions later. Several
failure modes block that conversion:

1. The exploratory action is selected after the root search, so the current
   state's policy target still reflects the pre-exploration MCTS visits.
2. Sparse terminal reward plus short/weak value targets can leave MCTS unable
   to distinguish `down` as useful.
3. Low `num_simulations` gives coarse visit vectors; `down` can get zero target
   mass even when it occasionally appears in behavior.
4. Deterministic held-out eval removes root noise, temperature sampling, and
   epsilon, then tie-breaks low-count roots.
5. `reanalyze_ratio=0` freezes the collect-time target visits in replay. If
   those visits underweight `down`, more replay of the same rows does not
   relabel them.

The sparse UPC25 epsilon-collect run is the concrete example: trainer-side
actions improved to `[288, 74, 64]` for `[up, stay, down]`, but held-out MCTS
still collapsed strongly, including `ckpt_best` at `[806, 0, 0]`. That is
exactly what the source semantics allow.

## Plain Interpretation For `down`

Seeing `down` in trainer telemetry proves only that behavior explored `down`.
It does not prove that the policy head was trained to prefer `down`.

For the policy head to learn `down`, MCTS visit distributions must put mass on
`down` in the states sampled for training, or reanalysis must later relabel
those states with `down` mass. Otherwise MuZero can learn a dynamics model that
has seen `down` transitions while its policy head learns a low- or zero-`down`
prior.

## Top 3 Config/Code Tests

1. Target audit: instrument one train run to persist, per root, executed action,
   `visit_count_distributions`, normalized `child_visit_segment`, policy logits,
   and final `target_policy`. Confirm whether rows with executed `down` also
   have target mass on `down`.

2. Collect-vs-eval same-state probe: for the same saved observations and
   checkpoint, call collect forward with configured temperature/epsilon and eval
   forward with no noise. Log selected action, visits, tie status, and logits.
   This should show whether collect samples/overrides actions from nearly the
   same roots that eval deterministically collapses.

3. Training target-quality sweep: train small matched runs that change only
   target-producing knobs, not the eval metric: `num_simulations` 8 vs 25,
   `reanalyze_ratio` 0 vs a small positive value, and explicit sparse-horizon
   settings (`pong_episode_max_steps=120`, `game_segment_length=120`,
   `td_steps=120`, `discount_factor=1`, narrow +/- outcome supports). Gate on
   held-out MCTS action histograms, root tie/margin telemetry, and score/survival
   together, not trainer-side action counts alone.
