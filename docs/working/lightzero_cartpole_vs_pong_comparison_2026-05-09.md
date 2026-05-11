# LightZero CartPole Vs Custom Pong Comparison - 2026-05-09

Scope: compare the stock LightZero CartPole Modal lane that produced a clean
progression signal against the custom dummy Pong LightZero lane that still fails
independent checkpoint scorecards. This is an evidence note only. No pytest was
run.

## Executive Read

Stock CartPole worked because it stayed close to a known LightZero example:
stock env type, stock trainer entrypoint, dense per-step reward, small MLP
observation, two actions, and a progression config that used more learner work
per collect plus eval every iteration.

Custom Pong is a much larger protocol jump. It patches the stock CartPole
config into a custom single-ego-vs-scripted-opponent Pong env, uses sparse
terminal score reward, has three actions, encodes a hand-built 10-float ego
state, ties `max_env_step` to the environment episode horizon, and relies on a
custom checkpoint reconstruction path for independent eval. Recent seed
plumbing is much better, and strict MCTS checkpoint loading now passes, but the
held-out MCTS scorecards still show zero `down` actions and no improvement over
random/scripted baselines.

The strongest differences to test next are:

1. `update_per_collect`: CartPole progression used `4`; Pong runs used `1`.
2. Reward density/timing: CartPole gets a reward every alive step; Pong returns
   `0` until terminal `+1/-1` or horizon truncation `0`.
3. Horizon semantics: Pong uses `max_env_step` as both trainer budget and
   `PongConfig.max_steps`, so changing budget also changes the normalized step
   feature and truncation target.
4. Exploration/random collect behavior: Pong's custom `random_action()` had a
   repeated-action bug earlier, and the LightZero random-collect/epsilon surface
   is still not explicitly audited in the run summaries.
5. Eval/sidecar mismatch: CartPole trusted stock evaluator logs; Pong needs an
   independent MCTS scorecard because trainer sidecar rows are not final
   checkpoint quality.

## Source Lanes Read

- CartPole wrapper:
  `src/curvyzero/infra/modal/lightzero_cartpole_tiny_train_smoke.py`
- CartPole result:
  `docs/experiments/2026-05-09-modal-lightzero-cartpole-tiny-train-smoke.md`
- Pong wrappers:
  `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
  `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
  `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py`
  `src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_scoreboard_attempt.py`
- Pong env/eval/checkpoint code:
  `src/curvyzero/training/dummy_pong.py`
  `src/curvyzero/training/lightzero_dummy_pong_env.py`
  `src/curvyzero/training/lightzero_dummy_pong_features.py`
  `src/curvyzero/training/lightzero_dummy_pong_policy.py`
  `src/curvyzero/training/dummy_pong_eval.py`
- Pong result notes:
  `docs/experiments/2026-05-09-lightzero-dummy-pong-implementation-inspection.md`
  `docs/experiments/2026-05-09-lightzero-dummy-pong-longer-run.md`
  `docs/experiments/2026-05-09-lightzero-dummy-pong-mcts-scorecard.md`
  `docs/experiments/2026-05-09-lightzero-dummy-pong-post-seed-fix-run.md`
  `docs/experiments/2026-05-09-lightzero-dummy-pong-post-deep-seed-fix-run.md`
  `docs/working/lightzero_trainer_scorecard_mismatch_2026-05-09.md`
  `docs/working/lightzero_pong_training_config_bug_investigation_2026-05-09.md`
  `docs/working/lightzero_pong_action_collapse_bug_hunt_2026-05-09.md`

## What Worked In CartPole

The successful stock CartPole progression run used LightZero 0.2.0,
DI-engine 0.5.3, torch 2.11.0, and `lzero.entry.train_muzero`. It stayed on
the stock `zoo.classic_control.cartpole.config.cartpole_muzero_config` example
with CPU caps:

```text
env_id: CartPole-v0
env_type: cartpole_lightzero
policy_type: muzero
model_type: mlp
action_space_size: 2
collector_env_num: 1
evaluator_env_num: 1
n_evaluator_episode: 1
n_episode: 1
num_simulations: 5
batch_size: 16
update_per_collect: 4
eval_freq: 1
cuda: false
max_train_iter: 4
max_env_step: 128
```

Result: `ok: true`, trainer returned `MuZeroPolicy`, checkpoint iterations were
`[0, 4]`, final evaluator reward was `33.0`, and learner metrics/checkpoints
were visible. This proves the stock LightZero trainer can progress and write
artifacts on Modal CPU. It does not prove anything about custom Pong quality.

Two important properties of CartPole are easy to miss:

- The environment reward is dense: alive steps produce positive reward. A
  33-step episode gives many nonzero reward targets.
- The stock env/trainer/evaluator/checkpoint path is internally matched. There
  is no custom observation adapter, scripted opponent, independent checkpoint
  loader, or external scorecard required to interpret the basic progression
  signal.

## What Pong Does Differently

Custom dummy Pong starts from the stock CartPole config but patches in a custom
env:

```text
env_id: DummyPongLightZero-v0
env_type: dummy_pong_lightzero
curvyzero_env: dummy_pong_lag1
feature_mode: tabular_ego
opponent_policy: random_uniform
model_type: mlp
observation_shape: 10
action_space_size: 3
collector_env_num: 1
evaluator_env_num: 1
n_evaluator_episode: 1/4/8 depending run
num_simulations: 2/4/8 depending run
batch_size: 8/16/32 depending run
update_per_collect: 1
cuda: false
```

The wrapper exposes a single LightZero-controlled ego paddle. The opponent is
supplied inside the environment by `random_uniform`, `track_ball`,
`lagged_track_ball_1`, or a frozen checkpoint policy. This is not simultaneous
self-play from LightZero's perspective.

The action space is:

```text
0 = up
1 = stay
2 = down
```

The LightZero observation is a dict containing a 10-float `tabular_ego` vector,
an all-ones action mask, `to_play=-1`, and `timestep`. The 10 floats include
paddle positions, ball relative position/velocity, ball row, and
`step / max_steps`.

The reward returned to LightZero is raw score reward only:

```text
nonterminal: 0
ego scores: +1
ego misses: -1
pure truncation: 0
```

`shaped_loss_delay_return` and survival are telemetry only. They are not passed
as env reward.

## Result Contrast

Pong trainer smokes and scaled attempts can call LightZero, write checkpoints,
and mirror artifacts. The problem is quality under independent eval.

Key results:

- Tiny 64/2 run passed as plumbing: 5 terminal sidecar rows, 4 wins / 1 loss,
  checkpoints `ckpt_best`, `iteration_0`, `iteration_2`.
- 512/8 run: trainer-side sidecar looked strong, but corrected MCTS scorecard
  for `iteration_8` remained effectively up-only:
  aggregate actions `[4849, 19, 0]`.
- 4096/64 run: trainer-side sidecar said 535 wins / 43 losses, but the sidecar
  was dominated by repeated seed 2. Fresh MCTS scorecard for `iteration_64`
  lost to random/scripted opponents and chose no `down`:
  vs random `[290, 219, 0]`, vs lagged `[6475, 2045, 0]`,
  vs track `[3335, 1284, 0]`.
- Post deep-seed-fix 1024/16 run: seed diversity finally passed
  (`131` unique seeds in `148` rows, top seed only `2/148`), but independent
  MCTS still failed:
  paired learned rows aggregated `[7285, 4781, 0]`;
  player0-only rows aggregated `[3353, 2073, 0]`.

This points away from "Modal/LightZero cannot run" and toward a training
signal/protocol mismatch: the final checkpoint is not a useful Pong controller
under strict independent MCTS eval.

## Concrete Difference Matrix

| Surface | CartPole working lane | Custom Pong failing lane | Why it matters |
| --- | --- | --- | --- |
| Config source | Stock LightZero CartPole config | CartPole config patched into custom Pong env | Pong may inherit defaults that are merely compatible, not appropriate. |
| Env type | `cartpole_lightzero` | `dummy_pong_lightzero` | Pong adds a custom wrapper, opponent policy, telemetry, and checkpoint adapters. |
| Reward density | Dense per-step alive reward | Sparse terminal score reward; nonterminal 0 | Pong has far fewer reward targets and a harder credit assignment problem. |
| Reward timing | Reward every step until terminal | Reward after joint action; usually only at score/miss | Pong reward arrives after simultaneous ego/opponent transition. |
| Horizon | `max_env_step=128` trainer cap; stock env episode horizon separate | `max_env_step` also becomes `PongConfig.max_steps` and observation normalization denominator | Budget changes alter the task and the `step/max_steps` feature. |
| `update_per_collect` | `4` in progression | `1` in all main Pong attempts | Pong may be under-updating relative to sparse reward and small collection. |
| Eval frequency | Explicit `eval_freq=1` in CartPole progression | Patched to `eval_freq=1`, but train-side sidecar mixes row sources | Pong needs separated collector/evaluator/final-checkpoint semantics. |
| Action space | 2 actions | 3 actions: up/stay/down | Pong has an observed action-collapse failure: zero `down` in MCTS eval. |
| Observation | Stock CartPole MLP shape | Hand-built 10-float ego row or 135 raster flat | Pong depends on adapter correctness and horizon normalization. |
| Opponent | None; single-agent env dynamics | Scripted opponent folded into env transition | Pong's `random_uniform` opponent creates a stochastic game slice and seed-stream issues. |
| Random action/exploration | Stock env/helper path | Custom `random_action()` had a repeated-action bug, now fixed | Random collect/eps behavior still deserves explicit audit in summaries. |
| Seed handling | Stock env manager behavior was enough for smoke | Multiple fixes needed; earlier sidecars dominated by one seed | Bad seed distribution made trainer wins misleading. |
| Checkpoint loading | Trainer returned stock policy and wrote stock checkpoints | Independent eval reconstructs `MuZeroPolicy`, strict-loads `checkpoint["model"]`, calls eval-mode MCTS | More places for config/key/eval mismatch, though strict load now passes. |
| Quality signal | Stock evaluator reward/checkpoint index was enough for progression smoke | Trainer sidecar is not final checkpoint quality; independent MCTS is the quality gate | Pong's trusted scorecard remains negative. |

## Difference Details

### Config Defaults

CartPole progression deliberately changed only a few stock fields and got a
known-good stock result. Pong reuses the CartPole MuZero config template, then
overwrites env type/id, model observation shape, action size, collector/eval
counts, `num_simulations`, batch size, `update_per_collect`, and env-specific
fields. That makes Pong a compatibility port, not a native LightZero example.

Stock Atari Pong dry config exists, but the project did not train it. It showed
the stock Atari Pong path is conv-based with `PongNoFrameskip-v4`,
`atari_lightzero`, action space size `6`, and much larger stock source defaults
such as `num_simulations = 50`, `batch_size = 256`, and
`max_env_step = int(5e5)`. Custom Pong is instead an MLP `action_space_size=3`
toy env based on the CartPole template.

### Horizon

The most dangerous Pong horizon detail is that the same `max_env_step` is used
for multiple meanings:

- trainer `max_env_step` cap passed to `train_muzero`;
- `env.max_steps` passed into `PongConfig`;
- denominator for the tabular feature `step / max_steps`;
- independent checkpoint eval horizon when `lightzero_max_env_step` is passed.

Earlier scorecards had a real mismatch: a 512-step checkpoint was reconstructed
with 64 and played in an independent env with 120. That was patched so
LightZero checkpoint eval uses `PongConfig(max_steps=lightzero_max_env_step)`.
Corrected scorecards still failed, so horizon mismatch is not the whole root
cause, but it remains a sharp footgun.

Test: split "training budget" from "episode horizon" in the Pong wrappers. Keep
episode horizon fixed at 120 or another named profile while varying train
budget, and assert the eval horizon exactly matches the checkpoint's training
horizon/profile.

### Reward Density

CartPole's reward gives the learner a signal at every alive step. Pong gives
mostly zeros, then terminal `+1/-1`, with pure timeouts at `0`. A typical Pong
episode can end in 8-30 steps, so most collected rows carry no immediate
reward. MuZero can learn sparse games, but this custom setup uses tiny budgets
and a fixed scripted opponent; the sparse terminal objective makes the
training signal much weaker than CartPole's smoke.

Test: run a deliberately labeled training-reward ablation, not an env-reward
rewrite. Keep eval on raw score reward, but compare raw sparse reward against a
temporary loss-delay training reward or curriculum contact profile to see if
action `down` ever becomes learnable.

### `update_per_collect`

CartPole progression raised `update_per_collect` to `4`. The Pong attempts kept
`update_per_collect=1`, including 512/8, 4096/64, and post deep-seed-fix
1024/16. With sparse reward and tiny collected batches, one update per collect
may be too little learner pressure. This is one of the cleanest config
differences because it does not require changing the env.

Test: fixed 1024/16 or 2048/32 Pong run with `update_per_collect=4`,
same `num_simulations=8`, same batch size or a small sweep, then score
`iteration_0` and latest under the same held-out MCTS protocol. Pass condition
is not just wins; require nonzero `down` action selection on states where the
ball is below the paddle center.

### Random Collect / Epsilon

Pong has had random/exploration footguns:

- `DummyPongLightZeroEnv.random_action()` previously reseeded on every call,
  so helper calls could repeat inside an episode. It now uses a persistent
  per-episode RNG.
- Earlier trainer sidecars were dominated by one episode seed because
  DI-engine/LightZero seeding overrode the intended dynamic seed mode.
- The run summaries do not clearly expose LightZero's random collect count,
  collector epsilon/exploration mode, replay fill, or action-source identity.

CartPole did not need this scrutiny because the goal was only stock trainer
progression. Pong does need it because action collapse is the failure mode.

Test: add summary fields for random collection, collector action source,
action histogram by collector/evaluator, replay size, and root visit entropy.
Then run the same Pong config with exploration/random-collect explicitly
audited.

### Eval Frequency And Telemetry Semantics

CartPole progression patched `eval_freq=1` and parsed stock evaluator/learner
tables. That is enough for a stock smoke.

Pong also patches `eval_freq=1`, but the env-side `episodes.jsonl` sidecar
mixes LightZero activity and does not label row source as collector vs
evaluator. The 4096/64 sidecar looked excellent mostly because repeated seed-2
rows were easy for the changing policy. The independent MCTS scorecard is the
trusted quality read.

Test: label Pong sidecar rows by source if LightZero exposes it, or write
separate telemetry paths. Summarize final held-out evaluator rows separately
from all collector/evaluator sidecar rows.

### Action Space

CartPole has two actions. Pong has three, and the failing checkpoints
systematically avoid action index `2` under MCTS eval. Baseline policies and
random policies do emit `down`, so the scoreboard can count it and the env can
execute it. Action-id inversion and action mask bugs are low suspicion.

Test: add first-N MCTS debug rows with observation, ball vertical offset,
policy logits, visit counts, selected action, seat, seed, and opponent. The
first thing to inspect is states with `ball_dy_from_ego_center > 0`; a sane
policy should sometimes select `down`.

### Observation Shape

CartPole uses the stock MLP observation shape. Pong uses a custom 10-float row:
paddle y/x values, ego-frame ball dx/dy/vx/vy, ball y, and normalized step.
The same encoder is used by the env and checkpoint adapters, so duplication is
not the current leading bug. But the observation is still more fragile because:

- `step / max_steps` changes when the horizon knob changes;
- training mainly used `ego_agent=player_0`, while paired scorecards also test
  `player_1`;
- physical `ego_paddle_x` and `opponent_paddle_x` leak side even though other
  features are ego-oriented.

Player0-only scorecards still failed, so seating is not the whole cause.

Test: run a real-observation diagnostic grid for player0 only: compare logits
and MCTS visits for mirrored states where only vertical ball offset changes.
If `down` stays suppressed, the learned prior/search is collapsed or the value
model has a sign/protocol issue.

### Environment Reward Timing

Pong reward is emitted after the simultaneous joint action. LightZero chooses
only ego action; the wrapper chooses the opponent action from the current
observation, steps both players, and returns the ego reward from
`pong_step.rewards[ego]`. This timing is internally consistent, but it is a
harder transition model than CartPole because the opponent is hidden inside
the environment transition from LightZero's point of view.

Test: train a deterministic-opponent control against `lagged_track_ball_1`,
then score against that same opponent first. If deterministic-opponent Pong
learns action `down`, the random-opponent transition/reward sparsity is the
main stressor.

### Checkpoint Loading Path

CartPole did not need a separate loader to claim progression. It called stock
`train_muzero`, parsed logs, and scanned stock artifacts.

Pong independent eval reconstructs config from local helper defaults, creates
`MuZeroPolicy`, strict-loads the checkpoint state dict, and calls
`policy.eval_mode.forward(...)` with:

```text
data: torch float32 [1, 10]
action_mask: [[1,1,1]]
to_play: [-1]
ready_env_id: [0]
```

Earlier direct policy-head probing was not MCTS and collapsed tied logits to
action `0`. The MCTS loader path now strict-loads successfully with the
`res_connection_in_dynamics_true` variant. The post deep-seed-fix scorecard
records strict load `ok=true`, no missing or unexpected keys. That lowers
loader suspicion, but it does not eliminate official-load-path differences.

Test: compare the adapter's `policy._model.load_state_dict(checkpoint["model"])`
against LightZero's official `learn_mode.load_state_dict(full_checkpoint)` on
the same observations. Also run the already-suggested `model` vs `target_model`
control, labeled as diagnostic only.

## Ranked Next Config Tests

1. **Pong `update_per_collect=4` control.**
   Keep the post deep-seed-fix shape close to 1024/16, but match CartPole's
   progression `update_per_collect=4`. Score latest and `iteration_0` with the
   same held-out MCTS ladder. This is the cleanest CartPole-vs-Pong config
   delta.

2. **Fixed episode horizon with larger train budget.**
   Decouple `max_env_step` from `PongConfig.max_steps`. For example, keep
   `max_steps=120` or a named `canonical_v0` horizon while training for
   1024/2048/4096 env steps. This tests whether changing the normalized step
   feature and horizon has been destabilizing every scale attempt.

3. **Reward-density ablation.**
   Keep eval raw score. For training only, compare raw sparse score reward
   against a labeled loss-delay or contact-curriculum lane. The purpose is to
   test whether the MuZero setup can learn any vertical control before
   insisting on pure sparse score reward.

4. **Deterministic-opponent control.**
   Train against `lagged_track_ball_1` and first score against
   `lagged_track_ball_1`, player0-only. This removes random opponent dynamics
   as a confounder.

5. **Exploration/random-collect audit run.**
   Do not trust another Pong run until the summary records random collect,
   collector/evaluator action histograms, seed distribution, replay size, and
   row source or an explicit "unknown" field.

6. **Official checkpoint-load parity probe.**
   Compare adapter strict load, official LightZero full-checkpoint load, and
   `target_model` diagnostic actions/logits on the same real observation grid.

## Bottom Line

CartPole validated the stock LightZero/Modal trainer path. Pong validated the
custom-env plumbing and checkpoint-scorecard infrastructure, but not learning.
The failure is now specific: independent strict MCTS eval gets a policy that
does not choose `down` and does not beat baselines. The next best move is not a
bigger run. It is a small set of config-isolation tests, starting with
`update_per_collect=4`, fixed horizon semantics, reward-density ablation, and a
deterministic-opponent control.
