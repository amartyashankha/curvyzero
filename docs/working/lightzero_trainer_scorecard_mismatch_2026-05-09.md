# LightZero Trainer/Scorecard Mismatch - 2026-05-09

## Short Read

The 4096/64 LightZero trainer-side `535 wins / 43 losses` result is not a
clean final-checkpoint scorecard. It is an env-side sidecar summary collected
during the LightZero training/eval process. That sidecar is dominated by
repeated seed-2 episodes: 513 of 578 rows used `episode_seed=2`; those rows
were 505/8. The 65 non-seed-2 rows were 30/35.

The independent MCTS scorecard is a fresh paired split. It strict-loads
`iteration_64.pth.tar` from the checkpoint's `model` key, matches the 4096
horizon, calls `MuZeroPolicy.eval_mode.forward`, and still never chooses down.

So the independent fresh paired MCTS scorecard failure is much less
mysterious. The current read is blunt: no reliable baseline improvement yet.
Infrastructure works; learning signal, seed handling, and train/eval split are
not trustworthy yet.

## Patch Note: Seed Telemetry Fix

Implemented after this investigation:

- `DummyPongLightZeroEnv` now defaults `dynamic_seed=True`, so repeated training
  resets advance from the configured base seed instead of replaying the same
  `episode_seed` unless a smoke explicitly sets `dynamic_seed=False`.
- `env.seed(seed)` preserves the configured dynamic/fixed mode; deterministic
  smokes can still call `env.seed(seed, dynamic_seed=False)`.
- `patched_dummy_pong_configs(...)` writes `dynamic_seed=True` into the
  LightZero env config and exposes it in the patched config surface.
- `summarize_episode_rows(...)` now reports compact seed distribution fields:
  unique seed count, most-common seed/count/fraction, top 5 seeds, and a
  dominance warning when one seed accounts for at least half the rows.

Why: the 4096/64 sidecar had 513/578 rows at `episode_seed=2`, and those rows
were 505/8 while the 65 non-seed-2 rows were 30/35. The previous sidecar summary
made that repeated-seed artifact too easy to miss.

Validation, no training run:

```text
python3 -m py_compile src/curvyzero/training/lightzero_dummy_pong_env.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py

uv run --with LightZero==0.2.0 python -c "<reset smoke>"
dynamic: [7, 8, 9, 10, 11]
fixed: [7, 7, 7, 7, 7]
via_seed_preserves_cfg_false: [11, 11, 11]

uv run --extra modal --with LightZero==0.2.0 python -c "<config surface smoke>"
dynamic_seed: True
validation: ok
```

## Patch Note: Deeper Seed Plumbing Fix

The first dynamic-seed fix was not enough. A later post-seed train still had
`episode_seed=3` in 129 of 148 sidecar rows, which means DI-engine/LightZero was
likely calling `env.seed(3, dynamic_seed=False)` and overriding the intended env
config.

Follow-up fix:

- `DummyPongLightZeroEnv` now treats config `dynamic_seed` as authoritative.
  Config/default `dynamic_seed=True` keeps reset seeds advancing even when an
  env-manager calls `env.seed(seed, dynamic_seed=False)`.
- Deliberate deterministic mode still works by setting env config
  `dynamic_seed=False`; env-manager seed-call arguments cannot accidentally turn
  that fixed mode dynamic either.
- Episode telemetry now records `base_seed`, `configured_dynamic_seed`,
  `effective_dynamic_seed`, `seed_call_dynamic_seed_arg`, and `seed_source`.
- The seed dominance warning now also catches the single-seed case when dynamic
  seed mode was effective or unknown, while avoiding false alarms for explicit
  fixed-seed telemetry.
- The config/import smoke includes a seed-policy check for the exact DI-engine
  call shape.

Validation, no training run:

```text
python3 -m py_compile src/curvyzero/training/lightzero_dummy_pong_env.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py

PYTHONPATH=src uv run --with LightZero==0.2.0 python -c "<seed policy smoke>"
dynamic_cfg_env_manager_false:
  configured=True effective=True arg=False
  source=config_dynamic_overrode_env_seed_arg_false
  seeds=[3, 4, 5, 6]
fixed_cfg_env_manager_false:
  configured=False effective=False arg=False
  source=env_seed_arg_matches_config
  seeds=[3, 3, 3, 3]
fixed_cfg_env_manager_true:
  configured=False effective=False arg=True
  source=config_fixed_overrode_env_seed_arg_true
  seeds=[3, 3, 3, 3]
```

## Evidence Read

Run:

```text
run_id: lz-dpong-20260509T151212Z-b95b61de2eb0
attempt_id: attempt-20260509T151212Z-8b9db08f8fcb
checkpoint: iteration_64.pth.tar
sha256: 11a0cc80f797ce8e63150e0a6018efc163b7858bed9efd92b77dda8cadaf95e4
```

Trainer config:

```text
env: dummy_pong_lag1
feature_mode: tabular_ego
opponent_policy: random_uniform
seed: 2
max_env_step: 4096
max_train_iter: 64
num_simulations: 8
batch_size: 32
n_evaluator_episode: 8
collector_env_num: 1
evaluator_env_num: 1
```

Trainer-side sidecar summary:

```text
episodes: 578
wins/losses: 535 / 43
timeouts: 0
player_0 actions [up, stay, down]: [9539, 800, 170]
```

Fresh MCTS checkpoint scorecard:

```text
config.max_steps: 4096
lightzero_eval_config.max_env_step: 4096
checkpoint state_dict_path: model
strict load: true
num_simulations: 8
action_mask: [[1,1,1]]
to_play: [-1]
ready_env_id: [0]
```

Scorecard rows:

```text
vs random_uniform:       wins 13/32, actions [290, 219, 0]
vs lagged_track_ball_1:  wins 11/30 plus 2 trunc, actions [6475, 2045, 0]
vs track_ball:           wins 0/31 plus 1 trunc, actions [3335, 1284, 0]
```

## Ranked Causes

### 1. Trainer-side telemetry is not a held-out final-checkpoint eval

This is the strongest finding.

The trainer `episodes.jsonl` has 578 rows, but 513 rows use `episode_seed=2`.
Only 65 rows use other seeds. Split by seed:

```text
seed=2 rows:
  episodes: 513
  wins/losses: 505 / 8
  player_0 actions: up=9105, stay=548, down=6

non-seed-2 rows:
  episodes: 65
  wins/losses: 30 / 35
  player_0 actions: up=434, stay=252, down=164
```

That means the big `535 / 43` number is mostly "does the changing training
policy beat one repeated random-opponent sequence?" It is not the same question
as the independent scorecard, which evaluates fresh seeds and both seats.

The repeated-seed rows also explain why a mostly-up policy can look great: the
random opponent sequence for seed 2 is easy for this behavior. The scorecard's
fresh split exposes that the final checkpoint does not generalize.

Next fix/probe:

```text
Make trainer/evaluator telemetry label row source: collector vs evaluator.
Force evaluator env seeding to fresh recorded pseudo-random eval seed lists.
Report trainer-side final eval as a separate recorded eval-wave block, not mixed with all sidecar rows.
```

### 2. The independent scorecard loads `model`; `target_model` is very different

`iteration_64.pth.tar` has top-level keys:

```text
model
target_model
optimizer
last_iter = 64
last_step = 869
```

Both `model` and `target_model` have 108 tensors, but they are not close copies.
Every comparable tensor differs. The global mean absolute difference is about
`0.03725`; many batch-norm counters differ by hundreds. `target_model` has
default-looking values in early representation layers:

```text
target_model representation BN weight mean: 1.0
target_model representation BN running_mean mean: 0.0
target_model representation BN running_var mean: 1.0
target_model representation num_batches_tracked: 0
```

The scorecard currently chooses `model`. This is probably the right key for
LightZero eval: the remote source excerpt for `MuZeroPolicy._forward_eval`
calls `self._eval_model.initial_inference(...)`, not the target network. But
because the checkpoint carries a very different `target_model`, this deserves
one explicit control probe.

Next probe:

```text
Run a tiny target_model control:
  load checkpoint["target_model"] into the same eval adapter
  probe logits/actions on real observations
  optionally score only vs random_uniform for a few episodes

Expected: target_model is worse or untrained-looking. If it is better, our loader
is using the wrong key.
```

Also probe the official load path:

```text
Compare:
  policy._model.load_state_dict(checkpoint["model"])
against:
  policy.learn_mode.load_state_dict(torch.load(checkpoint))

Then compare eval_model actions/logits for the same observation.
```

### 3. Action extraction from `eval_mode.forward` looks correct

Remote loader smoke for `iteration_64` returned:

```text
output type: dict
output key: 0
output[0]["action"]: 0
visit_count_distributions: [4, 3, 1]
predicted_policy_logits: [0.1173, -0.0286, -0.1333]
```

LightZero source excerpt says `_forward_eval` fills `output[env_id]` for each
`ready_env_id`. Our extractor checks key `0` and `"0"` before falling back to
plain `"action"` fields. With `ready_env_id=[0]`, this is the right field.

This does not explain the mismatch.

Next probe:

```text
Add optional debug logging of raw eval output for the first N scorecard actions.
This is observability only.
```

### 4. Observation shape/type mostly matches LightZero eval

`DummyPongLightZeroEnv.reset()` returns a dict:

```text
observation: float row
action_mask: int8[3]
to_play: -1
timestep: int
```

LightZero's policy eval method does not receive that full dict directly. The
source excerpt for `_forward_eval` expects:

```text
data tensor: [N, observation_shape]
action_mask: [N, action_space_size]
to_play: list/array
ready_env_id: env ids
```

The independent scorecard passes:

```text
data: torch float32 [1, 10]
action_mask: float32 [[1,1,1]]
to_play: [-1]
ready_env_id: [0]
```

That matches the policy-level eval API. `timestep` was required by the LightZero
evaluator loop earlier, but `_forward_eval` itself does not use it.

Remaining caveat: training only uses LightZero ego `player_0`; the independent
paired scorecard also seats the checkpoint as `player_1`. The bad result is not
only a player-1 issue, though: as player_0 vs random it still went 6/16.

Next probe:

```text
Run a one-seat-only scorecard:
  LightZero as player_0 only
  same recorded eval wave as trainer evaluator for replay
  then a fresh recorded eval wave for the claim
```

### 5. LightZero eval action selection is deterministic

The remote source excerpt shows eval does:

```text
roots.prepare_no_noise(...)
self._mcts_eval.search(...)
select_action(distributions, temperature=1, deterministic=True)
```

So independent eval is not missing stochastic visit sampling for evaluator
mode. Temperature is passed as `1`, but `deterministic=True` means it chooses
the best visit-count action.

This does not explain the mismatch.

### 6. `to_play`, `ready_env_id`, and action mask look right

For this single-agent-vs-scripted-opponent wrapper:

```text
to_play = -1
ready_env_id = [0]
action_mask = all ones
```

All three dummy Pong actions are always legal. The action order remains:

```text
0 = up
1 = stay
2 = down
```

No evidence points to an action-id inversion or action-mask bug.

### 7. Independent env/opponent/horizon is now matched where it matters

Earlier 512/8 scorecards had a horizon mismatch. This 4096/64 scorecard does
not:

```text
training max_env_step: 4096
scorecard config.max_steps: 4096
scorecard lightzero_eval_config.max_env_step: 4096
opponent for checkpoint config reconstruction: random_uniform
```

The main remaining env/protocol mismatch is the seed schedule and row source,
not horizon.

## Most Likely Story

The trainer-side sidecar says the changing policy did well on a repeated
seed-2 random-opponent sequence while training/evaluator telemetry was being
written. That does not prove the final checkpoint improved. The final
checkpoint is still a shallow, biased policy: it mostly chooses up or stay and
never chooses down under deterministic MCTS eval. On a fresh paired scorecard
it fails.

The final checkpoint is not "strong but loaded wrong" yet. It is more likely
"trainer telemetry was optimistic and not held out." The checkpoint-key probe
is still worth doing because `target_model` is visibly different, but it is not
the top explanation.

## Next Concrete Fixes/Probes

1. Fix trainer telemetry semantics.
   - Add row source: collector/evaluator.
   - Add policy/checkpoint identity if LightZero exposes it.
   - Summarize recorded eval-wave rows separately from all env sidecar rows.
   - Warn if one seed dominates the sidecar.

2. Fix evaluator seeding.
   - Ensure LightZero evaluator envs use fresh recorded pseudo-random eval seed lists.
   - Record `episode_seed` distribution in the top-level summary.
   - Treat repeated-seed evaluator wins as a smoke only.
   - Make trainer/eval use a fresh recorded multi-start eval wave before any
     quality claims.

3. Run a trainer-seed reproduction scorecard.
   - Score `iteration_64` as player_0 vs `random_uniform` on the exact seed-2
     setup that dominates trainer telemetry.
   - Then score the same checkpoint on seeds 3..67 and a fresh held-out split.
   - This will separate "memorized seed/opponent sequence" from "loader bug."

4. Run the checkpoint-key control.
   - Compare `model` vs `target_model` logits/actions.
   - Compare `_model.load_state_dict(checkpoint["model"])` vs
     `learn_mode.load_state_dict(full_checkpoint)`.
   - Run at least a small scorecard control if `target_model` is not obviously
     dead.

5. Add first-N action debug rows to the MCTS scorecard.
   - Store raw output action, visit counts, policy logits, player seat, seed,
     and observation summary.
   - Keep this small, because full episode traces can get noisy fast.

No pytest was run for this investigation.

After these fixes, do only a modest rerun. Do not scale the same shape until
held-out seed handling and the `model` vs `target_model` control are clean.
