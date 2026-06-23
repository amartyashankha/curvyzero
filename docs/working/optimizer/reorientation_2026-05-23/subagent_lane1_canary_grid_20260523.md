# Lane 1 Train-Facing Hook Canary/Profile Plan

Date: 2026-05-23

Status: command plan only. Do not launch from this doc. Do not touch live run
volumes, checkpoint roots, or Coach run IDs.

## Scope

Launcher:
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`

Lane 1 claim under test:

```text
mode=train
stock LightZero train_muzero
collect_search_backend=direct_ctree_gpu_latent
collect_search_ctree_backend=lightzero
GPU compute only
fallback fail-closed
RND meter required: exploration_bonus_mode=rnd_meter_v0, weight=0, require_rnd_metrics=true
```

Use fresh `lane1-*` run IDs only. Run rows one at a time with
`--wait-for-train`; never use detached/background canaries here. Background eval
and GIF stay off to avoid live-run interference and timing noise.

## Local Preflight

Local/mock only:

```bash
uv run pytest -q -p no:cacheprovider \
  tests/test_lightzero_phase_profiler.py \
  -k "direct_ctree_collect_search_hook or curvytron_train_call_cap"

uv run pytest -q -p no:cacheprovider \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  -k "train_mode_can_install_direct_ctree_gpu_latent_collect_hook or train_mode_direct_ctree_gpu_latent_marks_missing_observation_not_ok or train_mode_direct_ctree_gpu_latent_rejects_non_lightzero_ctree or rnd_meter_mode_selects_reward_model_entrypoint_and_patch"
```

Stop if either fails.

## Optional Profile Rehearsal

This does not prove train attachment. Use it only to review the matched profile
denominator and command expansion before any train canary. These commands should
not launch jobs.

```bash
uv run python scripts/build_curvytron_profile_grid.py \
  --experiment-id lane1-profile-h100-rndmeter-20260523 \
  --family lane1_direct_ctree_profile_rehearsal \
  --run-prefix lane1-prof-h100-rnd-20260523 \
  --attempt-prefix profile \
  --seed 304 \
  --computes h100 \
  --env-manager-types subprocess \
  --collectors 256 \
  --evaluator-env-num 1 \
  --batch-sizes 64 \
  --num-simulations 8 \
  --exploration-bonus-modes rnd_meter_v0 \
  --exploration-bonus-weight 0.0 \
  --exploration-bonus-rnd-batch-size 64 \
  --exploration-bonus-rnd-update-per-collect 100 \
  --source-max-steps 512 \
  --max-train-iter 128 \
  --max-env-step 400000 \
  --save-ckpt-after-iter 999999 \
  --stop-after-learner-train-calls 40 \
  --env-telemetry-stride 256 \
  --reward-variant sparse_outcome \
  --collect-search-backends stock,direct_ctree_gpu_latent \
  --collect-search-ctree-backends lightzero \
  --disable-death-for-profile \
  --detached

uv run python scripts/run_curvytron_optimizer_profile_manifest.py \
  --manifest artifacts/local/curvytron_optimizer_profile_manifests/lane1-profile-h100-rndmeter-20260523.json \
  --dry-run
```

Profile rehearsal success means the manifest has exactly matched stock/direct
rows, RND meter flags, `collect_search_ctree_backend=lightzero`, background
eval/GIF disabled, and compact output. It is not a train proof and should not
touch Modal.

## T0 Direct Smoke

Small train proof, not a speed claim. This should get enough learner work to
exercise collect, replay, RND, checkpoint discovery, and the fail-closed proof.

```bash
uv run --extra modal modal run --quiet \
  -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train::main \
  --mode train \
  --compute gpu-h100-cpu40 \
  --seed 304 \
  --run-id lane1-train-smoke-h100-direct-rnd-s304-20260523 \
  --attempt-id train-smoke-direct-rnd-s304 \
  --env-variant source_state_fixed_opponent \
  --reward-variant sparse_outcome \
  --opponent-policy-kind fixed_straight \
  --collector-env-num 64 \
  --evaluator-env-num 1 \
  --n-episode 64 \
  --n-evaluator-episode 1 \
  --batch-size 32 \
  --num-simulations 4 \
  --source-max-steps 256 \
  --max-train-iter 32 \
  --max-env-step 65536 \
  --save-ckpt-after-iter 8 \
  --stop-after-learner-train-calls 8 \
  --env-telemetry-stride 64 \
  --lightzero-eval-freq 0 \
  --no-background-eval-enabled \
  --no-background-gif-enabled \
  --collect-search-backend direct_ctree_gpu_latent \
  --collect-search-ctree-backend lightzero \
  --exploration-bonus-mode rnd_meter_v0 \
  --exploration-bonus-weight 0.0 \
  --exploration-bonus-rnd-batch-size 64 \
  --exploration-bonus-rnd-update-per-collect 100 \
  --require-rnd-metrics \
  --output-detail compact \
  --wait-for-train | tee /private/tmp/lane1-t0-direct-rnd-s304.json
```

T0 must show:

- `ok=true`, `status=completed`, `mode=train`, `called_train_muzero=true`.
- `trainer_entrypoint=lzero.entry.train_muzero_with_reward_model`.
- `command.collect_search_backend=direct_ctree_gpu_latent`.
- `command.collect_search_ctree_backend=lightzero`.
- `command.collect_search_backend_fallback_policy=fail_closed_when_non_stock`.
- `search_backend_proof.direct_ctree_gpu_latent_calls > 0`.
- `search_backend_proof.fallback_calls == 0`.
- `search_backend_proof.output_rows > 0`.
- `search_backend_proof.observed_collect_search_backends` contains
  `direct_ctree_gpu_latent`.
- `search_backend_proof.observed_collect_search_ctree_backends` contains
  `lightzero`.
- With `num_simulations > 0`:
  `search_backend_proof.recurrent_inference_calls > 0` and
  `search_backend_proof.model_output_d2h_bytes > 0`.
- `counts.learner_train_calls >= 8` and `counts.replay_sample_calls > 0`.
- `rnd_reward_model_metrics.required=true` and RND collect/train/estimate counts
  are positive.
- Background eval/GIF are false.

Stop the whole grid if T0 fails any item.

## T1 Warm H100 A/B

First useful train-mode timing read. Same seed, shape, checkpoint cadence, and
RND settings. Run stock first, then direct. Do not overlap with live H100 work.

Stock:

```bash
uv run --extra modal modal run --quiet \
  -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train::main \
  --mode train \
  --compute gpu-h100-cpu40 \
  --seed 304 \
  --run-id lane1-train-warm-h100-stock-rnd-s304-20260523 \
  --attempt-id train-warm-stock-rnd-s304 \
  --env-variant source_state_fixed_opponent \
  --reward-variant sparse_outcome \
  --opponent-policy-kind fixed_straight \
  --collector-env-num 256 \
  --evaluator-env-num 1 \
  --n-episode 256 \
  --n-evaluator-episode 1 \
  --batch-size 64 \
  --num-simulations 8 \
  --source-max-steps 512 \
  --max-train-iter 128 \
  --max-env-step 400000 \
  --save-ckpt-after-iter 16 \
  --stop-after-learner-train-calls 40 \
  --env-telemetry-stride 256 \
  --lightzero-eval-freq 0 \
  --no-background-eval-enabled \
  --no-background-gif-enabled \
  --collect-search-backend stock \
  --collect-search-ctree-backend lightzero \
  --exploration-bonus-mode rnd_meter_v0 \
  --exploration-bonus-weight 0.0 \
  --exploration-bonus-rnd-batch-size 64 \
  --exploration-bonus-rnd-update-per-collect 100 \
  --require-rnd-metrics \
  --output-detail compact \
  --wait-for-train | tee /private/tmp/lane1-t1-h100-stock-rnd-s304.json
```

Direct:

```bash
uv run --extra modal modal run --quiet \
  -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train::main \
  --mode train \
  --compute gpu-h100-cpu40 \
  --seed 304 \
  --run-id lane1-train-warm-h100-direct-rnd-s304-20260523 \
  --attempt-id train-warm-direct-rnd-s304 \
  --env-variant source_state_fixed_opponent \
  --reward-variant sparse_outcome \
  --opponent-policy-kind fixed_straight \
  --collector-env-num 256 \
  --evaluator-env-num 1 \
  --n-episode 256 \
  --n-evaluator-episode 1 \
  --batch-size 64 \
  --num-simulations 8 \
  --source-max-steps 512 \
  --max-train-iter 128 \
  --max-env-step 400000 \
  --save-ckpt-after-iter 16 \
  --stop-after-learner-train-calls 40 \
  --env-telemetry-stride 256 \
  --lightzero-eval-freq 0 \
  --no-background-eval-enabled \
  --no-background-gif-enabled \
  --collect-search-backend direct_ctree_gpu_latent \
  --collect-search-ctree-backend lightzero \
  --exploration-bonus-mode rnd_meter_v0 \
  --exploration-bonus-weight 0.0 \
  --exploration-bonus-rnd-batch-size 64 \
  --exploration-bonus-rnd-update-per-collect 100 \
  --require-rnd-metrics \
  --output-detail compact \
  --wait-for-train | tee /private/tmp/lane1-t1-h100-direct-rnd-s304.json
```

Optional echo only after T1 passes: rerun the same A/B with seed `305`, changing
only `--seed`, `--run-id`, `--attempt-id`, and output file names.

## Quick Extraction

For each result:

```bash
jq '{
  ok,
  status,
  mode,
  called_train_muzero,
  trainer_entrypoint,
  command: {
    compute: .command.compute,
    collector_env_num: .command.collector_env_num,
    batch_size: .command.batch_size,
    num_simulations: .command.num_simulations,
    collect_search_backend: .command.collect_search_backend,
    collect_search_ctree_backend: .command.collect_search_ctree_backend,
    fallback_policy: .command.collect_search_backend_fallback_policy,
    exploration_bonus: .command.exploration_bonus,
    background_eval_enabled: .command.background_eval_enabled,
    background_gif_enabled: .command.background_gif_enabled
  },
  counts,
  timers_sec,
  derived,
  search_backend_proof,
  rnd_reward_model_metrics
}' /private/tmp/lane1-t1-h100-direct-rnd-s304.json
```

Compare direct vs stock only when both rows have the same shape and
`learner_train_calls` cap:

```text
direct.derived.steps_per_sec / stock.derived.steps_per_sec
stock.timers_sec.train_muzero_wall / direct.timers_sec.train_muzero_wall
```

If `train_muzero_wall` is absent or checkpoint save time dominates, treat T1 as
semantic proof only, not a speed claim.

## Kill Criteria

Stop immediately if any row has:

- `ok=false` or `status!=completed`.
- `called_train_muzero != true`.
- RND row not using `lzero.entry.train_muzero_with_reward_model`.
- Background eval or GIF enabled.
- `counts.learner_train_calls` below the requested cap.
- `counts.replay_sample_calls <= 0`.
- Missing required RND metrics, no RND estimate calls, predictor hash unchanged,
  target hash changed, or `rnd_meter_v0` changes target rewards.

Stop direct rows specifically if:

- fallback calls are nonzero.
- direct calls are zero.
- output rows are zero.
- observed backend/CTree do not include `direct_ctree_gpu_latent` and
  `lightzero`.
- with simulations > 0, recurrent calls or model-output D2H bytes are zero.

Stop speed promotion if:

- H100 direct is slower than stock by more than 5%.
- H100 direct/stock is below `1.10` on the warm pair.
- repeats disagree in direction.

## Success Bar

T0 success: all direct proof fields and RND required metrics pass. No speed
claim.

T1 useful signal: all proof fields pass and direct/stock is at least `1.20` on
both steps/sec and wall-time ratios.

T1 cleanup-only signal: proof passes but speedup is `1.05x-1.20x`; keep the hook
as safe cleanup, but do not sell it as a Coach scheduling win.

Failure signal: proof fails, fallback is observed, RND metrics are not clean, or
direct/stock is below `1.05x`.

## Explicit Non-Goals

- No `flat_a3` in train mode.
- No detached train canaries.
- No reuse of live run IDs.
- No background eval/GIF.
- No compact-ownership claim from this hook.
- No comparison against compact/MCTX profile rows without naming the
  denominator.
