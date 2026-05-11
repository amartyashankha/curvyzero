# LightZero Dummy Pong Root-Cause Red-Team - 2026-05-09

Question: what are we still missing that could explain no learning signal in
custom dummy Pong?

No pytest was run. This is a broad root-cause pass over the current docs and
source, not a new training result.

## Sources Read

- `docs/working/training_state_index_2026-05-09.md`
- `docs/working/training_coach_handoff_2026-05-09.md`
- `docs/working/training_experiment_backlog.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-contact-pressure-curriculum-smoke.md`
- `docs/experiments/2026-05-09-dummy-pong-contact-pressure-scoreability-probe.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-contact-pressure-modest-rung.md`
- `docs/working/lightzero_pong_contact_curriculum_critique_2026-05-09.md`
- `docs/working/lightzero_pong_action_collapse_bug_hunt_2026-05-09.md`
- `docs/working/lightzero_pong_eval_action_collapse_debug_2026-05-09.md`
- `docs/working/lightzero_pong_broad_bug_hunt_2026-05-09.md`
- `docs/working/lightzero_source_setup_audit_2026-05-09.md`
- `docs/working/lightzero_muzero_target_semantics_2026-05-09.md`
- `docs/working/lightzero_pong_setup_critique_wave2_2026-05-09.md`
- `docs/working/lightzero_trainer_scorecard_mismatch_2026-05-09.md`
- `docs/working/lightzero_pong_training_config_bug_investigation_2026-05-09.md`
- `src/curvyzero/training/dummy_pong.py`
- `src/curvyzero/training/lightzero_dummy_pong_env.py`
- `src/curvyzero/training/lightzero_dummy_pong_features.py`
- `src/curvyzero/training/lightzero_dummy_pong_policy.py`
- `src/curvyzero/training/dummy_pong_eval.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
- Local LightZero source under `/tmp/lightzero-src`, especially
  `lzero/policy/muzero.py`, `lzero/policy/utils.py`,
  `lzero/mcts/buffer/game_segment.py`,
  `lzero/mcts/buffer/game_buffer_muzero.py`, and official CartPole/Atari/
  TicTacToe configs.

## Current Read

The plumbing is no longer the main story. Custom dummy Pong can train through
LightZero MuZero on Modal, mirror checkpoints, strict-load checkpoints into the
independent MCTS adapter, run scorecards, preserve seed diversity, and expose
contact-pressure reset metadata.

The persistent symptom is sharper:

- trainer-side rows can contain all three actions;
- independent held-out MCTS often collapses to a narrow action set;
- the collapse direction changes by run (`up`, `stay`, `down`, or no `down`);
- contact-pressure starts are genuinely action-sensitive and scoreable against
  `lagged_track_ball_1` and `stay`;
- the modest contact-pressure rung still had `down=0` in every held-out learned
  row, and final `iteration_3` got worse than initialization on the scoreable
  lagged target.

That pattern points away from a single action-id inversion and toward target
policy/value signal, weak MCTS roots, and remaining eval/load/config
provenance risks.

## Ranked Root Causes And Cheap Falsifiers

### 1. Policy-target semantics are starving the action the behavior executed

Likelihood: high.

LightZero MuZero does not train the policy head to imitate the executed action.
The target policy is the MCTS root visit distribution stored in
`child_visit_segment`. Executed actions matter because they choose the next
state and reward, but random warmup, epsilon override, and collect sampling do
not directly put probability mass on that action.

This explains the biggest confusing fact: trainer telemetry can show `down`,
while held-out eval never chooses `down`. The behavior distribution and policy
target distribution are different objects. The action-collapse debug docs
already show weak/tied roots such as `[2,3,3]`, and deterministic eval selects
the lowest tied index.

Cheapest falsifier:

Instrument one tiny run or inspect one mirrored replay/game segment sample and
dump, for first N transitions:

- observation features;
- executed action;
- root visit distribution stored as the policy target;
- root value;
- reward and terminal outcome;
- scoreability oracle row, when available.

Use scoreable contact-pressure states with known best actions. If executed
`down` appears but `target_policy[down]` is near zero on states where `down` is
oracle-good, this is confirmed. If target policies already put meaningful,
state-dependent mass on all useful actions but eval still collapses, move this
down and inspect load/eval/model capacity.

Very cheap precheck:

Run `scripts/summarize_lightzero_pong_scorecards.py debug-mcts` with
`--decision-mode both --collect-repeats 16` on the same observations. If collect
samples varied actions while eval remains tied/collapsed, that supports the
semantics mismatch but does not prove the stored training targets are good.

### 2. Sparse reward/value signal is too weak for the current tiny MuZero setup

Likelihood: high.

The environment reward is honest sparse score reward: `+1/-1/0`. Survival and
loss-delay are telemetry only. That is correct as a reward contract, but it
means the value and reward heads see a lot of zeros until a point is scored.
Official configs are much larger than our dummy Pong runs: CartPole uses
25 simulations, `n_episode=8`, `batch_size=256`, and `update_per_collect=100`;
Atari Pong uses 50 simulations and much larger collection/replay; sparse board
games use larger update pressure and final-outcome targets.

The evidence is not "just train longer." Pure 2x, UPC25, epsilon/random
warmup, and contact-pressure modest rung all failed held-out gates. The more
specific suspect is that current MCTS roots do not become useful policy
targets before eval determinism exposes them.

Cheapest falsifier:

Run a target-quality probe, not a campaign:

- choose 24 to 64 scoreable contact-pressure states with varied oracle-best
  first actions;
- run current checkpoint MCTS first-N debug at `num_simulations=8,16,25`;
- record whether visits prefer the oracle-good action and whether values rank
  the eventual outcomes.

If higher-quality search over known scoreable states still gives flat or wrong
visits, data volume alone is not enough. If visits become state-dependent and
align with oracle actions, then the training setup likely needs more target
quality, more useful states, or better replay pressure before any larger run.

### 3. Reward support or value scale is wrong enough to flatten search

Likelihood: medium-high.

This is boring and still plausible. LightZero defaults are
`reward_support_range=(-300,301,1)` and `value_support_range=(-300,301,1)`,
which are wildly broad for dummy Pong terminal returns. Sparse probes did set
smaller supports, but not all runs used the same ranges, and contact-pressure
used reward support `[-1,1,1]` plus value support `[-1,1,0.01]`. A bad support
or scalar-transform setup can leave predicted values clustered near zero, which
makes MCTS roots close to uniform even when terminal returns exist.

Cheapest falsifier:

Add a support/target audit to one training summary or replay sample:

- configured reward and value support ranges;
- raw reward targets and raw value targets before scalar/support transform;
- transformed categorical target mass;
- predicted reward/value scalar after inverse transform for sampled states;
- clipping rate and target standard deviation.

Pass condition: terminal `+1/-1` and value targets land in expected support
bins with nontrivial variance, and the trained model's predicted values move
away from zero on scoreable win/loss states. If targets are clipped, nearly
constant, or transformed into uninformative mass, this becomes a top root
cause. A one-batch value overfit on curated win/loss states is the fastest
model-side sanity check.

### 4. Eval/load/config provenance drift is still fooling us sometimes

Likelihood: medium.

Several concrete bugs already came from this class:

- a 512-step checkpoint was scored with mismatched 64/120 horizon values;
- scorecard defaults still make `--max-env-step` look like both eval horizon
  and training budget;
- feature mode must be passed explicitly, especially for `raster_flat`;
- checkpoint refs and Modal artifacts are copied manually in docs/commands;
- `_find_state_dict(...)` chooses a state dict by heuristic and currently
  prefers `model`, while some command surfaces expose `state_key`.

Current failures persisted after the known horizon and strict-load fixes, so
this is not the leading explanation for all no-signal results. But it is still
the cheapest way to accidentally create a false negative.

Cheapest falsifier:

Make one provenance-locked scorecard path that requires a train `summary_ref`
and refuses to run unless these match:

- checkpoint SHA and `last_iter`;
- run id and attempt id;
- feature mode and observation shape;
- `pong_episode_max_steps`;
- reset profile and pressure agent;
- opponent policy;
- reward/value support ranges;
- `num_simulations`;
- selected checkpoint state key;
- adapter load method.

Then run official load parity on the same 24 observations:

- manual `checkpoint["model"]` load into `policy._model`;
- official `policy.learn_mode.load_state_dict(full_checkpoint)`;
- explicit `target_model` control.

If logits, values, visit counts, and actions match between manual and official
`model` load, close this for current checkpoints. If they diverge, stop all
scorecard interpretation until load parity is fixed.

### 5. Low simulation count and deterministic eval tie-breaking distort the symptom

Likelihood: medium for action histograms, lower as the full root cause.

This is real: eval uses no root noise and deterministic `np.argmax` over visit
counts. With 2 or 8 simulations over three legal actions, ties are common and
argmax biases toward lower action ids. The debug rows already showed exact ties
explaining stay-only behavior.

But this is not enough to explain no learning signal by itself. A later sim
sweep removed exact ties at 16/32/64 simulations and simply collapsed the
first-N rows to `down`. More sims can change the collapse direction without
making the controller state-sensitive or good.

Cheapest falsifier:

For each candidate checkpoint, before a full scorecard, run first-N debug rows
at `2,8,16,25,50` sims and report:

- action histogram;
- exact max-tie rate;
- top-2 visit margin;
- policy-logit margin;
- predicted and searched values;
- ball vertical offset;
- eventual terminal outcome if followed for a few steps.

If higher sims produce state-dependent actions and better raw score, low sims
were blocking eval. If higher sims only produce confident one-action collapse,
the root is weak policy/value targets, not eval simulation count.

### 6. The MLP or observation contract cannot represent the needed controller

Likelihood: medium for `raster_flat`, medium-low for current `tabular_ego`.

`tabular_ego` includes paddle positions, ball relative position, ball velocity,
ball row, and normalized step, so it is not obviously blind. Still, the
LightZero MLP policy head is small enough that capacity is worth falsifying.
`raster_flat` is riskier: one current grid frame does not expose velocity the
way official Atari frame stacks do.

Cheapest falsifier:

Do a local supervised oracle overfit, not a MuZero run:

- build rows from the contact-pressure scoreability table and normal-reset
  rollouts;
- label the best first action by true sparse rollout;
- train the same feature encoder and a small MLP with the same hidden width;
- evaluate held-out action accuracy and a direct policy scorecard.

If this cannot overfit or generalize at all, capacity/observation is a real
blocker. If it fits easily and scores, MuZero target/value/search is the issue.
For `raster_flat`, require frame stack or velocity features before making
learning claims.

### 7. Reset/opponent distribution is still biased away from useful score targets

Likelihood: medium.

Default `track_ball` is not a scoreable win target in the current geometry.
The contact-pressure scoreability probe improved the situation: 46/64
`lagged_track_ball_1` groups and 59/64 `stay` groups were scoreable, while
`track_ball` remained 0/64 scoreable. That means reset distribution was a real
problem, but the latest modest rung suggests it is not sufficient by itself:
trainer-side saw all three actions, held-out MCTS still had `down=0`, and final
was worse than init on the scoreable lagged target.

Cheapest falsifier:

Use the scoreability summary as a diagnostic replay fixture, not as the standing
eval panel. Sample rows where each of `up`, `stay`, and `down` is uniquely best,
and run checkpoint debug-MCTS on those exact reset states. If the model/search
does not prefer the known-best action on this curated distribution, reset bias
is no longer the main blocker.
If it does prefer the best action there but fails normal resets, the curriculum
is overfitting or canonical resets are too far from the training distribution.

### 8. Stale checkpoint selection or `ckpt_best` semantics are hiding the curve

Likelihood: medium-low globally, medium for individual reports.

`ckpt_best` is not necessarily the final checkpoint, and in recent runs the
final checkpoint was often worse than initialization while `ckpt_best` showed a
weak improvement or a different collapse. Docs and commands manually copy refs,
so it is easy to score the wrong file or over-read `ckpt_best`.

Cheapest falsifier:

Score every mirrored checkpoint in a run (`iteration_0`, all `iteration_N`,
`ckpt_best`) in one provenance-locked summary with SHA, `last_iter`, and
train summary ref. Do not promote any checkpoint unless at least two
post-init checkpoints improve raw score or survival on the same held-out
split without action collapse.

### 9. Modal artifact mismatch can create false evidence

Likelihood: medium-low, but cheap to close.

There has already been a Modal packaging failure from changing `__pycache__`
during build. Volume refs, attempt refs, local fetched summaries, and docs can
also drift.

Cheapest falsifier:

For every report, include:

- Modal app id;
- run id and attempt id;
- train summary ref;
- eval summary ref;
- checkpoint ref;
- checkpoint SHA and bytes;
- local fetched path, if any.

Run scorecards with `python -B` when packaging scripts, and make summaries
record file summaries for all inputs. If the local file SHA does not match the
Volume checkpoint SHA, throw away the result.

### 10. Action order mismatch is unlikely, but keep the canary

Likelihood: low.

The code and docs agree: `0=up`, `1=stay`, `2=down`, and `_move_paddle()` uses
`action - 1`. Baselines emit all three actions. The collapse direction changes
by run, which argues against one illegal or inverted action.

Cheapest falsifier:

Keep a tiny canary in config/import or scorecard summaries:

```text
action 0 -> paddle y delta -1
action 1 -> paddle y delta  0
action 2 -> paddle y delta +1
random_uniform action histogram has all three actions over a short eval
track_ball chooses 0 above center and 2 below center
```

Only reopen this if a checkpoint's selected action is interpreted differently
between LightZero output and `PongEnv.step()`.

### 11. `to_play` or reward sign is unlikely, but one diagnostic closes it

Likelihood: low.

The current wrapper is single ego vs hidden scripted/frozen opponent, with
`env_type='not_board_games'`, fixed action space, all-ones mask, and
`to_play=-1`. That matches LightZero non-board-game usage. Reward is returned
from the ego perspective.

Cheapest falsifier:

On the same checkpoint and 24 observations, run debug-MCTS with `to_play=-1`
and `to_play=0`. Log logits, values, visits, and actions. If only `to_play=0`
produces sensible state-dependent choices, inspect LightZero value sign
handling. Otherwise close it. Separately keep the reward-sign canary:
player_0 scoring right wall gives player_0 `+1`, player_1 `-1`.

### 12. `state_key` load issue is unlikely for current main results, but not closed

Likelihood: low-medium.

Current scorecards usually load `model`, and strict full-model load passes.
`target_model` controls have looked different and sometimes more random, but
that does not prove they are the right eval key. The unresolved boring issue is
that some command/config surfaces mention `opponent_checkpoint_state_key`, while
the actual loader finds state dicts through `_find_state_dict(...)`.

Cheapest falsifier:

Expose an explicit `state_key` argument in the LightZero checkpoint loader and
score the same observations with:

- `model`;
- `target_model`;
- official full-checkpoint load.

If `model` and official load match, keep `model`. If `target_model` is the only
one that matches LightZero's own evaluator or yields calibrated values, change
the loader and relabel all older scorecards as stale.

## Boring Possibilities Coverage

| Possibility | Current suspicion | Cheapest check |
| --- | --- | --- |
| Wrong reward support | Medium | Dump support ranges, raw targets, transformed categorical mass, clipping rate. |
| Bad value scale | Medium-high | Calibrate predicted values on known win/loss contact-pressure states. |
| Action order mismatch | Low | Keep action delta and baseline action canaries in summaries. |
| Eval config mismatch | Medium | Require train summary ref and assert feature/horizon/support/opponent/reset match. |
| State-key load issue | Low-medium | Compare explicit `model`, `target_model`, and official load on same observations. |
| `to_play` | Low | Debug-MCTS same obs with `to_play=-1` and diagnostic `to_play=0`. |
| Policy target semantics | High | Audit executed actions vs stored child-visit target policies. |
| Too-low sims | Medium | First-N sim sweep with tie rate, visit margin, score outcome. |
| MLP capacity | Medium | Supervised oracle overfit using same features and small MLP width. |
| Observation missing key info | Low-medium tabular, high raster | Oracle/action-label fit; require raster frame stack or velocity. |
| Reset distribution bias | Medium | Scoreability-indexed eval over states with varied unique best actions. |
| Stale checkpoint selection | Medium-low | Score full checkpoint curve with SHA, `last_iter`, and train summary ref. |
| Modal artifact mismatch | Medium-low | Record Volume refs and SHAs for every input/output; use `python -B` for packaging. |

## Next Checks, Ranked

1. Target-policy audit: executed action vs stored child-visit policy target vs
   scoreability oracle action. This is the most direct way to answer why train
   actions diversify while held-out eval collapses.
2. Provenance-locked loader/eval parity: one scorecard path requiring a train
   summary ref, plus manual `model` load vs official `learn_mode.load_state_dict`
   parity on the same observations.
3. Reward/value support and calibration audit: prove `+1/-1/0` targets and
   bootstrapped values are not flattened or clipped by support/scalar handling.
4. Scoreability-indexed debug-MCTS sim sweep: use known unique-best
   contact-pressure states, not arbitrary rollouts, and report visit margins
   and value estimates.
5. Supervised oracle overfit for `tabular_ego` and `raster_flat`: falsify MLP
   capacity and missing-observation concerns before another large MuZero run.

## Top 5 Most Plausible Root Causes

1. Stored policy targets do not put mass on the useful actions, because
   LightZero trains policy on MCTS visits, not executed exploratory actions.
2. Sparse terminal reward plus tiny/current MCTS-value setup yields weak roots
   and near-uniform or unstable search targets.
3. Reward/value support or scalar scale is flattening the value signal enough
   that MCTS cannot distinguish scoreable actions.
4. Eval/load/config provenance drift can still create false negatives unless
   feature mode, horizon, support ranges, state key, checkpoint SHA, and run
   summary are locked together.
5. Observation/model capacity, especially for `raster_flat`, may be too weak
   or missing velocity/history, and should be falsified with an oracle overfit
   before scaling.

## Bottom Line

Do not launch another longer dummy Pong campaign yet. The cheapest useful move
is to inspect what LightZero is actually using as policy and value targets on
scoreable states. If the targets are already good, chase load/config and model
capacity. If the targets are not good, the problem is not that the environment
never executed `down`; it is that the trainer never learned to want `down`.
