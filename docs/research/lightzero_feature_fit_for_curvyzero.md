# LightZero Feature Fit For CurvyZero

Research snapshot: 2026-05-09.

## 2026-05-10 Correction

This note is historical research, not the current front-door priority.
LightZero remains the serious MuZero control/training framework, and official
Atari Pong is the control pattern for the visual stack. For CurvyTron, the
current no-regret work is still repo-native and scalar/ray first:
`float32[106]`, `action_mask`, `to_play=-1`, one ego action, named opponent
policy, final-observation/reward metadata, and full wrapper action logging.
Visual CurvyTron frames are later adapter work. CurvyTron should not use ALE.

## Short Answer

LightZero is the right next practical attempt because it is the only complete
MuZero-family trainer we have already proven on Modal. It can probably get us
to a real dummy Pong MuZero checkpoint faster than writing replay, targets,
model code, optimizer updates, checkpointing, and eval ourselves.

But LightZero is not automatically the right long-term Curvy/MewZero backbone.
It is strongest when CurvyZero can look like a single-agent or simple
Gym/DI-engine env. It is weakest where CurvyZero needs simultaneous multi-agent
self-play, exact replay metadata, custom scorecards, custom losses, unusual
dynamics targets, or total ownership of artifact layout.

So the direction is:

- Try LightZero first for the next real dummy Pong MuZero run.
- Keep Mctx/project-owned training as fallback/comparison if LightZero loses
  semantics, metadata, or flexibility.
- Make the first LightZero smoke expose the risky parts early, not just prove
  that imports work.

## Fit Matrix

| Need | LightZero fit | Risk | Early smoke requirement |
| --- | --- | --- | --- |
| Real MuZero trainer | Strong. CartPole MuZero already ran on Modal. | Proven run is stock CartPole, not dummy Pong or Curvy. | The smoke must call LightZero's trainer on our custom env and write a checkpoint. |
| Deterministic learned dynamics | Strong for standard MuZero. | Model/config assumptions may be harder to bend than expected. | Patch CartPole MLP config to `observation_shape=10`, `action_space_size=3`, tiny caps. |
| Stochastic learned dynamics | Present in LightZero family, but not yet proven for our env. | Stochastic MuZero may be framework-level support without a clean custom-env path for Curvy noise. | Do not start stochastic. Log random streams and chance events so we know when deterministic MuZero fails. |
| Reward shaping | Medium. Env can return any scalar reward. | Custom auxiliary losses or split score/shaped targets may require trainer/policy edits. | Use honest env reward. Report shaped loss-delay telemetry as diagnostics first. |
| Custom loss flexibility | Medium to weak for v0. Config knobs may be enough for standard loss weights; new losses may need policy/model edits. | If MewZero needs custom targets, LightZero may stop being cheap. | First smoke must avoid custom losses. Later smoke should test one harmless loss/weight override before deeper adoption. |
| Visual/raster input | Strong in stock Atari path; medium for custom env. | The current control work has run ALE-backed `PongNoFrameskip-v4`; the risk is not ROM blockage anymore. The risk is accidentally treating Atari/ALE behavior as CurvyTron truth, or letting custom raster shape/config work distract from the scalar/ray CurvyTron contract. | Use official Atari Pong only as a control pattern. For CurvyTron, keep the scalar/ray single-ego adapter boundary first; visual frames come later. |
| Custom architectures | Medium. LightZero has model configs and PyTorch modules. | A truly custom MewZero architecture may require registering/editing model code. | First smoke uses stock MLP. Follow-up should test one custom model registration before relying on custom architecture plans. |
| Custom env wrappers | Medium. DI-engine `BaseEnv` and Gym-like paths exist. | Simultaneous multi-agent envs are not native; wrapper hides opponents inside `step`. | Use `DummyPongLightZeroEnv`: one ego action, scripted opponent action, full `info` sidecar. |
| Simultaneous games | Weak to medium. Ego-wrapper is acceptable for dummy Pong. | Not real simultaneous self-play; opponent choices are outside search. Joint actions explode. | First smoke must label itself ego-vs-scripted, not full Curvy self-play. |
| Self-play | Medium for board/alternating games; weak for Curvy-style simultaneous games. | Board-game `to_play` semantics do not equal CurvyTron. | Do not use board-game self-play mode first. Use `to_play=-1` and scripted opponent. |
| Checkpoints | Strong. LightZero writes `.pth.tar`. | Format is LightZero-owned, not CurvyZero-owned. | Mirror paths/hashes into `curvyzero-runs`; do not convert format in v0. |
| Eval hooks | Medium. LightZero evaluator exists. | Native eval return is not enough for Pong survival telemetry. | Wrapper and Modal function must write wins, survival, truncation, score, shaped return, seed, trace hash. |
| Modal friendliness | Good for CPU smokes. CartPole already ran. | Dependency stack is large; subprocess envs and logs may make artifact copying awkward. | Reuse pinned `LightZero==0.2.0` image and one coarse Modal Function. |
| Examples | Strong for CartPole and Atari-like tasks. | Examples are not CurvyTron semantics. Stock Atari Pong is useful control evidence, but its ALE env, six actions, and Pong reward should not be copied into CurvyTron. | Reuse only the LightZero interface pattern: observation dict, action mask, `to_play=-1`, timestep row, strict eval/reporting. |

## What Would Make LightZero A Bad Choice

LightZero becomes a bad immediate choice if any of these happen in the tiny
dummy Pong smoke:

- It cannot create/reset/step a custom `BaseEnv` without invasive DI-engine or
  LightZero changes.
- It calls stock CartPole or stock Atari Pong by accident instead of the custom
  env.
- It hides seed, action trace, terminal reward, opponent policy, or truncation
  details so the run cannot be replayed or diagnosed.
- It reports only reward means and cannot surface survival/loss-delay telemetry
  without awkward sidecar hacks.
- It forces board-game/alternating-player semantics onto simultaneous Pong.
- The wrapper must encode joint actions as the policy action space before a
  simple ego-vs-scripted run works.
- Config/model changes for `observation_shape=10`, `action_space_size=3`, and
  `model_type=mlp` become harder than writing a tiny project-owned trainer.
- The trainer starts, but artifact mapping is too opaque: checkpoints, logs,
  evaluator output, and config cannot be mirrored into `curvyzero-runs`.
- Pure wrapper or collector overhead is so high that MCTS cost dominates before
  any learning signal exists.
- Custom architecture, custom target, or custom loss needs immediately require
  forking LightZero policy/model code.

If one of these happens, write down the exact blocker and switch to the
Mctx/project-owned fallback with that blocker as the justification.

## Stochasticity And Dynamics

CurvyZero should separate three ideas:

- stochastic environment data;
- stochastic learned dynamics;
- Stochastic MuZero search with explicit chance nodes.

LightZero appears to support Stochastic MuZero as part of its algorithm family,
but that does not mean the first dummy Pong run should use it. The first dummy
Pong ruleset is deterministic enough to test standard MuZero. Robustness noise
such as sticky actions, frozen controls, action noise, image noise, and domain
randomization can be logged as env wrapper metadata first.

Early rule:

- Use deterministic MuZero first.
- Log seeds and random streams in every episode.
- Escalate to Stochastic MuZero only if planning needs to branch over real
  chance events, not merely because training data includes randomized wrappers.

Mctx fallback comparison: Mctx exposes deterministic, Gumbel, and stochastic
search APIs directly, which is cleaner if CurvyZero eventually needs explicit
chance-node modeling. The cost is that Mctx does not provide the trainer.

## Reward And Loss Flexibility

LightZero can consume scalar rewards from the env, so basic reward shaping is
possible at the wrapper boundary. That does not mean we should put every
diagnostic into the reward.

For dummy Pong:

- Environment reward stays honest: win `+1`, loss `-1`, no score `0`.
- Shaped loss-delay return is reported as telemetry and may be used later as a
  training target only if we explicitly choose that experiment.
- Survival variance is diagnostic or checkpoint-selection context, not the
  environment reward.

Risk: if MewZero needs custom losses, auxiliary heads, multi-target values,
representation regularizers, or nonstandard policy targets, LightZero may
require editing policy/model internals. That is a future fork-risk gate.

Smoke to reveal this early:

- First smoke: no custom loss.
- Second or third smoke, if first passes: change one harmless loss/config
  weight or add one extra logged target without editing LightZero internals.
  If that is painful, keep LightZero to standard trainer runs and move custom
  MewZero work toward Mctx/project-owned.

## Visual And Custom Architecture Fit

LightZero has a strong visual story for stock Atari, and the current control
work has run the installed `LightZero==0.2.0` Atari Pong surface through
ALE-backed `PongNoFrameskip-v4`. That proves the control stack can exercise
real Atari/ALE plumbing, but it does not make ALE part of CurvyTron.

For CurvyTron, the useful next interface is repo-native first:
`float32[106]` rays/scalars, `int8[3]` action mask, `to_play=-1`, one ego
action, named opponent policy, and full wrapper action metadata. Visual
CurvyTron should imitate LightZero's frame shape later; it should not depend on
Atari ROMs.

Dummy Pong should remain bridge/debug evidence for custom env mechanics. Its
tabular ego and raster paths isolate framework fit, but they are not CurvyTron
quality evidence.

Sequence:

1. Use official Atari Pong only as the LightZero/ALE control pattern.
2. Keep the CurvyTron scalar/ray single-ego adapter boundary as the next
   no-regret CurvyTron task.
3. Add visual CurvyTron later as one grayscale `(1,64,64)` frame per env step
   and let LightZero stack frames to `(4,64,64)`.
4. Use dummy Pong tabular/raster experiments only to debug custom env plumbing
   and reporting discipline.

Bad sign: if LightZero cannot accept a simple custom observation/action shape
without deep config surgery, it is unlikely to be pleasant for CurvyTron visual
models.

## Simultaneous Games And Self-Play

The first LightZero adapter should be honest:

```text
LightZero controls player_0.
The wrapper controls player_1 with a fixed policy.
The result is ego-vs-scripted MuZero, not full simultaneous self-play.
```

This is acceptable for the next smoke because the urgent question is whether a
complete trainer can run on a CurvyZero-owned env. It is not enough for the
long-term CurvyTron goal.

What to avoid first:

- board-game `self_play_mode`;
- joint-action search;
- vector values for every player;
- checkpoint-pool leagues;
- hidden opponent randomness with no trace.

Smoke requirements:

- `info` must include opponent policy id.
- `info` must include joint action or trace hash.
- evaluator summaries must say ego-vs-which-opponent.
- any later "self-play" claim must name whether the opponent is fixed,
  frozen-checkpoint, same-checkpoint policy-only, or searched.

Mctx fallback comparison: project-owned Mctx can model one ego row per player
or later try stochastic/opponent-conditioned dynamics, but we would write the
trainer ourselves.

## Checkpoint, Eval, And Modal Fit

LightZero's checkpointing is a strength for the immediate lane. We should keep
its `.pth.tar` format in the first smoke and mirror metadata into CurvyZero.

Required Modal artifact mirror:

```text
training/lightzero-dummy-pong/<run_id>/
  run.json
  latest_attempt.json
  attempts/<attempt_id>/
    attempt.json
    config.json
    command.json
    train/
      summary.json
      episodes.jsonl
      lightzero_artifacts_manifest.json
      lightzero_training_signals.json
  checkpoints/lightzero/
    ckpt_best.pth.tar
    iteration_*.pth.tar
    manifest.json
```

The first smoke does not need to convert LightZero checkpoints into a
CurvyZero-owned format. It does need to record enough metadata to know exactly
what checkpoint came from what command, config, seed, opponent, feature mode,
and env schema.

## Exact Smoke That Should Reveal The Risks

Implement this, in order:

1. `src/curvyzero/training/lightzero_dummy_pong_env.py`
   - DI-engine `BaseEnv`;
   - one ego action;
   - scripted opponent;
   - tabular ego observation shape `(10,)`;
   - action mask ones for `A=3`;
   - `to_play=-1`;
   - rich final `info`.

2. `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
   - pinned `LightZero==0.2.0`;
   - import/register env;
   - patch CartPole MuZero config to dummy Pong;
   - create/reset/step fixed seeds;
   - capture config and wrapper summary;
   - no trainer call.

3. `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
   - call LightZero `train_muzero`;
   - `max_env_step=64`;
   - `max_train_iter=2`;
   - `num_simulations=2`;
   - `batch_size=8`;
   - `update_per_collect=1`;
   - one collector env;
   - one evaluator env;
   - one evaluator episode;
   - mirror checkpoints/logs/telemetry.

Run commands:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0 \
  --opponent-policy random_uniform \
  --max-env-step 64 \
  --max-train-iter 2 \
  --num-simulations 2 \
  --batch-size 8 \
  --update-per-collect 1 \
  --n-evaluator-episode 1
```

Pass condition:

- real LightZero trainer called;
- custom dummy Pong env used;
- at least one LightZero checkpoint produced;
- learner/evaluator signals captured;
- wins/losses, survival steps, truncation, score return, shaped loss-delay
  return, seed, opponent policy, and trace hash written;
- artifacts mirrored into `curvyzero-runs`;
- no LightZero/DI-engine fork.

Fail condition:

- any required field is hidden;
- trainer cannot run the custom env;
- custom env requires invasive framework surgery;
- artifact mapping is too opaque;
- simple config/shape changes are brittle;
- runtime overhead is obviously out of proportion for dummy Pong.

## Other Repos Only Where They Reveal Gaps

- Mctx/project-owned fallback: best when CurvyZero needs exact control over
  dynamics, targets, stochastic modeling, replay rows, or checkpoint format.
  Bad for immediate speed because it is not a trainer.
- Muax: could reduce some JAX/Mctx trainer code, but it is not proven here for
  Pong, Modal, or simultaneous games. Revisit only if LightZero fails and
  direct Mctx looks too much.
- muzero-general: useful readable trainer structure, but older dependency and
  orchestration assumptions make it a reference, not the immediate backbone.
- EfficientZero: relevant later for Atari/sample efficiency, too heavy for the
  first dummy Pong custom-env proof.

## Bottom Line

LightZero is a good next attempt, not a permanent marriage. It earns more
scope only by running the custom dummy Pong smoke while preserving CurvyZero's
telemetry and artifacts. If it cannot do that cheaply, Mctx/project-owned
training stops being premature and becomes the justified fallback.

## Links

- `docs/decisions/0005-main-pong-repository-library-choice.md`
- `docs/research/muzero_framework_vs_project_owned.md`
- `docs/research/muzero_repo_baseline_options.md`
- `docs/research/mctx_integration.md`
- `docs/design/muzero_modal_architecture.md`
- `docs/runbooks/training_smokes.md`
