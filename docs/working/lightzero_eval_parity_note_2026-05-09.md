# LightZero Eval Parity Note - 2026-05-09

Scope: official LightZero Atari Pong eval parity only. I did not edit code and
did not run pytest. I read the eval wrapper, the manifest summarizer, the
parallel eval plan, and the fetched local manifest.

## Inputs

- Train run: `lz-visual-pong-exact-installed-0.2.0-s0`
- Train attempt: `train-faithful-short-installed-0.2.0-s0-8192-relpath`
- Train summary ref:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/train/summary.json`
- Eval id: `faithful-short-periodic-custom512-stockeval-s0-8192-relpath`
- Remote manifest ref:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/eval/faithful-short-periodic-custom512-stockeval-s0-8192-relpath/manifest_custom_steps512_seed0_20260509T232832Z.json`
- Local manifest:
  `artifacts/local/lightzero-eval-manifests/faithful-short-periodic-custom512-stockeval-s0-8192-relpath-manifest.json`

## What The Mismatch Means

The mismatch is specifically this flag:

```text
manual_vs_stock.actions_match_for_recorded_prefix = false
```

In `lightzero_pong_eval_smoke.py`, the manual path runs a custom single-env
loop and records its first 32 actions. The stock path runs
`lzero.worker.MuZeroEvaluator` and records the first 32 calls into
`policy.eval_mode.forward`. The flag compares:

```text
manual_actions[:len(stock_actions)] == stock_actions
```

So `false` means the custom manual eval loop is not producing the same opening
action sequence as the official LightZero evaluator, even with the same
checkpoint, seed, config patch surface, strict checkpoint load, and 512-step
episode cap.

This is an eval-harness parity warning. It is not a checkpoint-load failure.

## Observed Result

Manifest summary helper output:

| checkpoint | strict load | fallback | manual/stock match | manual return | stock return | positive rewards | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `iteration_0` | true | false | false | -13 | -13 | 0 | `manual_stock_mismatch` |
| `iteration_3697` | true | false | false | -13 | -8 | 0 | `manual_stock_mismatch` |

More detail:

- Both checkpoints loaded strictly: no missing or unexpected model keys.
- Model fallback was disabled and not used.
- Both manual episodes ran 512 steps and ended by `TimeLimit.truncated`.
- Both manual episodes had 13 negative rewards and no positive rewards.
- The stock evaluator completed for both checkpoints.
- `iteration_0`: stock return matched manual return at -13, but first 32
  actions still differed.
- `iteration_3697`: stock return was better than manual, -8 vs -13, and first
  32 actions differed.

First-prefix examples:

```text
iteration_0 manual: 1,0,4,3,1,3,2,4,2,2,3,1,3,2,1,2,...
iteration_0 stock:  3,1,1,0,0,0,4,2,2,2,0,0,0,2,3,2,...

iteration_3697 manual: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,4,...
iteration_3697 stock:  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,3,...
```

## Likely Causes

Most likely: the custom manual loop is not exactly the same evaluator as
LightZero's stock `MuZeroEvaluator`.

The manual loop builds one raw `AtariEnvLightZero`, resets it, keeps its own
frame stack, and calls:

```text
policy.eval_mode.forward(obs_tensor, action_mask=..., to_play=..., ready_env_id=...)
```

The stock path builds a DI-engine env manager and lets
`lzero.worker.MuZeroEvaluator` drive the environment and policy. Even with the
same seed and same patched config, small differences can change the opening
trajectory:

- Env manager reset or seeding may not be identical to the manual single-env
  reset.
- Stock evaluator may own some state transition details that the manual loop
  approximates by hand.
- Manual frame stacking may differ from the env-manager/evaluator path.
- MuZero eval with MCTS can be sensitive to any tiny observation or RNG
  difference.
- The manual and stock paths both pass timestep and action mask, but they do
  not share the same driving loop.

The current evidence points more at eval harness mismatch than at learned-policy
regression.

## What This Does Not Invalidate

This does not invalidate the faithful-short train artifact by itself.

Still valid:

- The checkpoints exist and load into the LightZero MuZero policy strictly.
- The policy can act in the real ALE-backed Pong env.
- The stock evaluator can run the checkpoints.
- The stock return readout is currently the better score signal.
- The result still says this short train has no positive Pong reward in the
  sampled 512-step evals.

Not valid as a strong claim:

- Do not use the manual loop return as the final score while
  `stock_manual_match=false`.
- Do not interpret manual-vs-stock action differences as a training bug.
- Do not compare checkpoints primarily by manual return if stock return is
  available.

## Fastest Next Probes

1. Treat stock evaluator return as the main score column for this run.
2. Rerun the same two checkpoints with stock evaluator enabled for seeds
   `0,1,2`. This checks whether `iteration_3697` being -8 stock return is a
   real improvement or just seed noise.
3. Add a mid checkpoint if available, but keep it cheap: first/mid/last is
   enough.
4. For parity debugging only, compare stock recorded calls against manual
   recorded steps at the same timestep and action mask. If the first difference
   happens at timestep 0, focus on reset/frame-stack/env-manager differences.
5. If code changes are allowed later, make the manual path use the same
   env-manager/evaluator stepping path, or stop treating manual prefix equality
   as a hard verdict when stock evaluator output is present.

## Recommended Eval Command

Use stock evaluator and keep `--max-episode-steps` equal to the eval step cap.
For the exact two-checkpoint follow-up, run one seed at a time and put the seed
in the eval id:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --parallel \
  --eval-pass custom \
  --eval-id faithful-short-periodic-custom512-stockeval-s1-8192-relpath \
  --checkpoint-refs 'training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/train/lightzero_exp/ckpt/iteration_0.pth.tar,training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/train/lightzero_exp/ckpt/iteration_3697.pth.tar' \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-8192-relpath \
  --seed 1 \
  --max-eval-steps 512 \
  --max-episode-steps 512 \
  --step-detail-limit 8 \
  --max-env-step 4 \
  --max-train-iter 1 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --num-simulations 2 \
  --batch-size 4 \
  --update-per-collect 1 \
  --game-segment-length 16 \
  --no-allow-model-fallback \
  --run-stock-evaluator
```

For seed `2`, change both `--eval-id` and `--seed`.

## Recommended Readout Columns

Use the manifest summary helper because it reads full `results` and includes
`stock_return`:

```sh
uv run python scripts/summarize_lightzero_pong_eval_manifest.py \
  --format tsv \
  --output artifacts/local/lightzero-eval-manifests/<EVAL_ID>.tsv \
  artifacts/local/lightzero-eval-manifests/<MANIFEST_FILE>.json
```

Read these columns first:

```text
checkpoint
checkpoint_ref
strict_load
fallback_used
stock_manual_match
steps_survived
return
stock_return
nonzero_reward_count
positive_reward_count
dominant_action
dominant_action_share
action_entropy
verdict
```

For the coaching decision, rank by `stock_return` first, then by
`positive_reward_count`, then by action collapse signals
`dominant_action_share` and `action_entropy`. Keep `strict_load=true` and
`fallback_used=false` as required validity gates.
