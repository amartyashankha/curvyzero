# 2026-05-09 Modal LightZero Official Atari Pong 4096 Sim10

## Question

Does the staged official Atari Pong rung from
`docs/working/lightzero_official_atari_next_run_plan_2026-05-09.md` produce a
useful no-fallback checkpoint curve when moved closer to LightZero's stock
Atari settings?

This is official LightZero Atari Pong only: stock `PongNoFrameskip-v4` via
`zoo.atari.config.atari_muzero_config` and `lzero.entry.train_muzero`. It is
not custom dummy Pong and not CurvyTron.

## Wrapper Patch

Changed only `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`.

- Raised the cheap-GPU Modal timeout from `12m` to `30m`.
- Raised validation caps to allow:
  `max_env_step=4096`, `max_train_iter=32`, `collector_env_num=2`,
  `evaluator_env_num=1`, `num_simulations=10`, `batch_size=32`,
  `update_per_collect=2`, `max_episode_steps=512`, and
  `game_segment_length=64`.
- Left the official Atari config path, ALE ROM image, and trainer semantics
  unchanged.

No pytest was run. Syntax check passed:

```sh
python -m py_compile src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py
```

## Train Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --compute gpu-l4-t4 --mode train --max-env-step 4096 --max-train-iter 32 --collector-env-num 2 --evaluator-env-num 1 --num-simulations 10 --batch-size 32 --update-per-collect 2 --max-episode-steps 512 --game-segment-length 64 --run-id lz-visual-pong-4096-sim10-s0 --attempt-id train-4096-sim10-b32-env2
```

Modal app: `ap-P1o51a8VOlB5arlSNC8w5l`

## Train Result

```text
run_id: lz-visual-pong-4096-sim10-s0
attempt_id: train-4096-sim10-b32-env2
ok: true
compute: gpu-l4-t4
runtime GPU: NVIDIA L4
CUDA available: true
remote_elapsed_sec: 136.665226
train_elapsed_sec: 125.488984
```

The train mirrored checkpoints only through `iteration_8`, not the hoped-for
`iteration_16`. This still clears the plan's hard checkpoint gate because
`iteration_8` exists.

Mirrored checkpoint refs:

```text
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/ckpt_best.pth.tar
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_0.pth.tar
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_1.pth.tar
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_2.pth.tar
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_3.pth.tar
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_4.pth.tar
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_5.pth.tar
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_6.pth.tar
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_7.pth.tar
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_8.pth.tar
```

Train summary ref:

```text
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/attempts/train-4096-sim10-b32-env2/train/summary.json
```

Trainer-side eval rewards seen in logs: `[-13.0, -13.0, -14.0, -13.0]` under
the 512-step episode cap.

## Eval Commands

Each eval used the same 4096/sim10 config surface, `--max-eval-steps 256`,
`--max-episode-steps 512`, and `--no-allow-model-fallback`.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_0.pth.tar --run-id lz-visual-pong-4096-sim10-s0 --attempt-id train-4096-sim10-b32-env2 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/attempts/train-4096-sim10-b32-env2/eval/iteration_0_sim10_eval256/lightzero_visual_pong_eval_iteration0_sim10_eval256.json --max-env-step 4096 --max-train-iter 32 --collector-env-num 2 --evaluator-env-num 1 --num-simulations 10 --batch-size 32 --update-per-collect 2 --max-episode-steps 512 --game-segment-length 64 --max-eval-steps 256 --step-detail-limit 8 --no-allow-model-fallback

uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_4.pth.tar --run-id lz-visual-pong-4096-sim10-s0 --attempt-id train-4096-sim10-b32-env2 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/attempts/train-4096-sim10-b32-env2/eval/iteration_4_sim10_eval256/lightzero_visual_pong_eval_iteration4_sim10_eval256.json --max-env-step 4096 --max-train-iter 32 --collector-env-num 2 --evaluator-env-num 1 --num-simulations 10 --batch-size 32 --update-per-collect 2 --max-episode-steps 512 --game-segment-length 64 --max-eval-steps 256 --step-detail-limit 8 --no-allow-model-fallback

uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_8.pth.tar --run-id lz-visual-pong-4096-sim10-s0 --attempt-id train-4096-sim10-b32-env2 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/attempts/train-4096-sim10-b32-env2/eval/iteration_8_sim10_eval256/lightzero_visual_pong_eval_iteration8_sim10_eval256.json --max-env-step 4096 --max-train-iter 32 --collector-env-num 2 --evaluator-env-num 1 --num-simulations 10 --batch-size 32 --update-per-collect 2 --max-episode-steps 512 --game-segment-length 64 --max-eval-steps 256 --step-detail-limit 8 --no-allow-model-fallback
```

Eval Modal apps:

```text
iteration_0: ap-pdSW72urgxRgLUnhzPUlQz
iteration_4: ap-zz2sF9NvMc12Rvnlqj3hLD
iteration_8: ap-S6jVQ51hZCrIJnNRMwGaGE
```

## Eval Results

| Checkpoint | Eval cap | Fallback count | Actions | Return | Nonzero reward steps | Terminal/truncation |
| --- | ---: | ---: | --- | ---: | --- | --- |
| `iteration_0` | 256 | 0 | `{0:238, 1:18}` | `-6.0` | `60:-1, 95:-1, 130:-1, 165:-1, 200:-1, 235:-1` | no terminal; stopped by eval cap |
| `iteration_4` | 256 | 0 | `{5:256}` | `-6.0` | `60:-1, 95:-1, 130:-1, 165:-1, 200:-1, 235:-1` | no terminal; stopped by eval cap |
| `iteration_8` | 256 | 0 | `{5:256}` | `-6.0` | `60:-1, 95:-1, 130:-1, 165:-1, 200:-1, 235:-1` | no terminal; stopped by eval cap |

Eval artifact refs:

```text
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/attempts/train-4096-sim10-b32-env2/eval/iteration_0_sim10_eval256/lightzero_visual_pong_eval_iteration0_sim10_eval256.json
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/attempts/train-4096-sim10-b32-env2/eval/iteration_4_sim10_eval256/lightzero_visual_pong_eval_iteration4_sim10_eval256.json
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/attempts/train-4096-sim10-b32-env2/eval/iteration_8_sim10_eval256/lightzero_visual_pong_eval_iteration8_sim10_eval256.json
```

## Eval-Only Parity Probe

Follow-up command, no training:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_8.pth.tar --run-id lz-visual-pong-4096-sim10-s0 --attempt-id train-4096-sim10-b32-env2 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/attempts/train-4096-sim10-b32-env2/eval/iteration_8_parity_probe/lightzero_visual_pong_eval_iteration8_parity_probe_v3.json --max-env-step 4096 --max-train-iter 32 --collector-env-num 2 --evaluator-env-num 1 --num-simulations 10 --batch-size 32 --update-per-collect 2 --max-episode-steps 512 --game-segment-length 64 --max-eval-steps 32 --step-detail-limit 32 --no-allow-model-fallback --run-stock-evaluator
```

Modal app: `ap-Wt8fIl4Z13fsFrfRAUGJ9w`

Wrapper update:

- `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py` now records compact
  MCTS output in step summaries, defaults detail capture to 32 steps, logs reset
  and policy-facing observation shapes, records ALE action meanings when
  accessible, and has a guarded `--run-stock-evaluator` attempt.
- Syntax check passed:
  `python -m py_compile src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py`.
- No pytest was run. No training was run.

Artifact:

```text
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/attempts/train-4096-sim10-b32-env2/eval/iteration_8_parity_probe/lightzero_visual_pong_eval_iteration8_parity_probe_v3.json
sha256: 68a6b914faeb5ac27cfed5da860af6a4bdd0849af83c0a9b3ff87dbdc3403bb0
checkpoint sha256: 7838096303b64ccc743f20bfc8491239888b0ca45230bad0879bbb24b131e44f
```

Runtime action meanings:

```text
0 NOOP
1 FIRE
2 RIGHT
3 LEFT
4 RIGHTFIRE
5 LEFTFIRE
```

The direct LightZero env object did not expose the wrapper chain action-meaning
method (`AtariLightZeroEnv` only surfaced `action_space`, env cfg helpers,
`legal_actions`, and `random_action`), but `gym.make("PongNoFrameskip-v4")`
inside the same Modal image reported the ALE meanings above.

Reset/frame-stack shape:

```text
raw reset observation: [1, 64, 64]
policy-facing observation after manual stack: [4, 64, 64]
action mask: [1, 1, 1, 1, 1, 1]
```

Manual eval-mode result for the first 32 steps:

```text
ok: true
strict checkpoint load: true
model fallback: false
actions: {5: 32}
rewards: {0.0: 32}
searched_value: 0.0 on all 32 steps
predicted_value: 2.586841583251953e-05 on sampled steps
```

Every recorded MCTS root selected action `5`. The visit distribution always put
the dominant mass on action `5`: usually `9/10` visits, sometimes `10/10`.
The policy head strongly preferred action `5` as well: across the 32 logged
steps, action-5 logits stayed around `7.59`, while the next-highest action-4
logit stayed around `0.49`. The full per-step
`predicted_policy_logits`, `visit_count_distributions`, `searched_value`, and
selected action are in the artifact.

Earlier generic DI-engine evaluator attempt:

```text
path: ding.worker.InteractionSerialEvaluator
status: blocked before any recorded eval-mode call
blocker: MuZeroPolicy._forward_eval() missing 1 required positional argument: 'action_mask'
```

I tried the generic DI-engine interaction evaluator with the loaded checkpoint,
one evaluator env, and no training. Construction succeeded after letting the
evaluator launch its own env manager, but all attempted `evaluator.eval(...)`
call signatures failed because the generic DI-engine evaluator passed
observations without LightZero MuZero's required `action_mask`. The later
LightZero `MuZeroEvaluator` rerun below is the true stock-evaluator comparison.

## Interpretation

This rung is an infrastructure pass and a quality/signal fail.

The wrapper cap widening worked, Modal provided an L4, CUDA was available, the
stock ALE Pong trainer ran, and checkpoint mirroring produced `iteration_8`.
The strict eval path also stayed clean: zero model fallback at every curve
point.

But the curve did not move. Every evaluated checkpoint returned `-6.0` over the
256-step eval window with no positive Pong rewards. `iteration_0` used only two
actions, then `iteration_4` and `iteration_8` collapsed to action `5` for every
step. This fails the plan's weak/useful-signal gates and should not trigger the
optional 512-step final eval or a climb to `num_simulations=25`.

The parity probe sharpens that read. Action `5` is real ALE Pong `LEFTFIRE`,
not an invalid action id, and the first 32 eval roots are already dominated by
action `5` in both policy logits and MCTS visits. The manual wrapper's
frame-stack behavior is explicit: raw env reset is one frame, policy input is
the local four-frame stack. The remaining generic evaluator gap is a harness API
gap (`action_mask` collation), not evidence that the collapse is caused by an
Atari action mapping error. A later rerun against LightZero's stock
`lzero.worker.MuZeroEvaluator` closed the parity gap; see below.

## Stock MuZeroEvaluator Parity Rerun

Follow-up command, no training:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_8.pth.tar --run-id lz-visual-pong-4096-sim10-s0 --attempt-id train-4096-sim10-b32-env2 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/attempts/train-4096-sim10-b32-env2/eval/iteration_8_stock_evaluator_parity32/lightzero_visual_pong_eval_iteration8_stock_evaluator_parity32.json --max-env-step 4096 --max-train-iter 32 --collector-env-num 2 --evaluator-env-num 1 --num-simulations 10 --batch-size 32 --update-per-collect 2 --max-episode-steps 512 --game-segment-length 64 --max-eval-steps 32 --step-detail-limit 32 --no-allow-model-fallback --run-stock-evaluator
```

Modal app: `ap-iQWGfUcUzj1N257PuWnzBO`

Artifact:

```text
training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/attempts/train-4096-sim10-b32-env2/eval/iteration_8_stock_evaluator_parity32/lightzero_visual_pong_eval_iteration8_stock_evaluator_parity32.json
sha256: 02a29acb6388e1a2b16aba65c5f074a7bf7dc3ba18927fed787dd7ea33d89140
checkpoint sha256: 7838096303b64ccc743f20bfc8491239888b0ca45230bad0879bbb24b131e44f
```

Result:

```text
ok: true
stock evaluator: lzero.worker.MuZeroEvaluator
strict checkpoint load: true
model fallback: false
manual actions: {5: 32}
stock actions: {5: 32}
actions_match_for_recorded_prefix: true
manual reset observation: [1, 64, 64]
manual policy observation: [4, 64, 64]
stock policy data: [1, 4, 64, 64]
manual action mask: [1, 1, 1, 1, 1, 1]
stock action mask: [[1, 1, 1, 1, 1, 1]]
manual rewards: {0.0: 32}
manual total reward: 0.0
stock eval_episode_return: [0.0]
stock eval_episode_return_mean: 0.0
fallback_step_count: 0
```

Read: the manual eval wrapper and stock LightZero `MuZeroEvaluator` now agree
on action choice, action masks, and policy-facing 64x64 four-frame input for
the first 32 steps. This removes evaluator parity as the next blocker for the
4096/sim10 checkpoint. The checkpoint still collapses to ALE action `5`
(`LEFTFIRE`) and earns no reward in the 32-step parity window.

## Reporting Metadata Requirement

Future LightZero eval parity and next-action reports must include comparable
metadata, even when the run is only a control:

- profile metadata: LightZero/DI-engine/torch/ALE/gym package versions, Modal
  app id, compute class, CUDA/runtime GPU when available, and wrapper/schema id;
- contracts: checkpoint ref and sha, config caps, env id, observation contract,
  action mask contract, evaluator path, strict-load status, and fallback policy;
- seed/reset: seed, dynamic-seed/reset policy, reset observation shape,
  policy-facing observation shape, `to_play`, and initial action mask;
- timing buckets where available: remote elapsed, eval elapsed, env-step timing,
  evaluator timing, and startup/build time if captured by the wrapper or Modal
  logs;
- throughput/latency where available: steps/sec, avg envstep/sec,
  avg time/episode, and per-step env latency if emitted;
- checkpoint ids: run id, attempt id, checkpoint iteration, checkpoint sha, and
  artifact ref/sha;
- clear non-claims: state when the run is parity/infrastructure only, not policy
  quality, not custom dummy Pong, not CurvyTron, and not a final architecture
  decision.

For this parity rerun, captured timing included manual eval elapsed `4.637077s`,
manual steps/sec `6.901544`, stock evaluator `evaluate_time=1.886381s`,
stock `avg_envstep_per_sec=16.963696`, stock `avg_time_per_episode=0.530115`,
and per-step env latency around `0.0013s` to `0.0016s` in the recorded manual
step summaries. LightZero remains a serious replication/control lane, not the
final architecture.

## Next Credible Replication Action

Prefer a current-config from-scratch near-upstream rung before reviving the
older `96x96`/downsample pretrained-eval lane. The 64x64 evaluator parity gap is
closed, so the next useful question is whether a closer-to-upstream stock
training recipe and budget can move the current matched config. Do not report
that rung as solved Atari Pong, dummy Pong progress, CurvyTron progress, or an
architecture choice unless independent score/return/action-diversity evidence
supports it.
