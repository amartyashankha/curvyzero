# LightZero Atari Pong Side-Lane Critic - 2026-05-09

Scope: official Atari-like Pong replication/control lane only. This note uses
local docs and code. No pytest was run. No training code was edited.

## Short Verdict

Yes, the current `lightzero_pong_exact_reproduction.py` lane is now the closest
stock LightZero setup we can reasonably run as a bounded Modal control. It uses
installed `LightZero==0.2.0`, imports `zoo.atari.config.atari_muzero_config`,
keeps the stock Atari env/model/search/batch/collector/evaluator settings, and
calls `lzero.entry.train_muzero`.

But the runs so far are still `faithful-short`, not exact reproduction. The
completed `8192` run and active `32768` run shorten only the trainer env-step
budget, but that one change is enough to make them controls instead of exact
upstream reproduction. The older tiny smoke wrapper is not the closest stock
setup; it patched many training knobs and should stay a smoke/control path.

## Still Drifting From Exact

- Source target is split. Current GitHub LightZero Pong uses about `500000`
  env steps, while the installed `LightZero==0.2.0` package in the Modal image
  exposes `200000`. Pick one target in every claim.
- Current real trains are shortened: `8192` completed and `32768` is active,
  versus installed stock `200000`. That is the main remaining train-side drift
  in the exact-wrapper lane.
- The wrapper still patches `exp_name` and runs train mode from `/runs` so
  artifacts land in the Modal Volume. This is an output-path change, not a
  learning-semantics change, but it is still a patch and should stay visible.
- Eval is not exact upstream scoring. The stock-ish eval is useful, but the
  eval wrapper compiles its own policy config, ran on CPU in the recorded
  `8192` read, and carried wrapper fields like `max_train_iter=1` and
  `update_per_collect=1`.
- Manual and stock eval still disagree on first-prefix actions for the `8192`
  stock-ish read. That makes manual telemetry diagnostic, not the scoring
  authority.
- The score read is thin: one seed, two checkpoints for `8192`, capped
  512-step eval, and weak improvement only. It is not solved Pong.
- Older `ckpt_best` behavior is suspect in the separate 8192/sim25 lane:
  `ckpt_best` looked reset-like. Do not use `ckpt_best` as quality evidence
  until the best-save path is understood for the exact-wrapper lane too.
- The OpenDILab pretrained checkpoint path is still not a valid control unless
  a matching checkpoint/config/model surface is found.

## Next Thing That Reduces Confusion

Do not start a new variant first. Finish or inspect the active
`train-faithful-short-installed-0.2.0-s0-32768-relpath` run, then evaluate only
periodic `iteration_*.pth.tar` checkpoints with strict load and no fallback.

Use the planned stock-ish eval settings:

- `num_simulations=50`
- `collector_env_num=8`
- `evaluator_env_num=3`
- `batch_size=256`
- `game_segment_length=400`
- `max_env_step=200000`
- manual cap and env episode cap both `512`
- `--run-stock-evaluator`
- no model fallback

Rank checkpoints by `stock_return`. Keep manual return, survival steps,
nonzero/positive reward counts, action histogram, dominant-action share,
entropy, and manual/stock match as diagnostics beside the stock result.

This eval would answer the current practical question: did the cleaner
faithful-short stock lane move at all when scaled from `8192` to `32768`, or
was the weak `8192` improvement just noise? If strict/no-fallback eval stays
flat or mismatched, the next decision should be explicit: either pay for an
installed-package exact `200000` run, or pause this lane and investigate eval
parity/best-checkpoint behavior before spending more.

## Do Not Mix With This Lane

- Custom dummy Pong results. That is a bridge/debug lane with a single-ego toy
  adapter, MLP features, different rewards, and target-sidecar telemetry.
- `tabular_ego`, `raster_flat`, or future `raster_stack4_ego` bridge results.
  They are not visual Atari parity.
- Repo-native PPO actor-loop or learner smokes. They are CurvyTron architecture
  probes, not LightZero replication.
- CEM, supervised raster policies, or Mctx search-only benchmarks. Useful
  controls, not MuZero Atari Pong reproduction.
- Old tiny official Atari smoke returns from patched settings such as low sims,
  tiny batches, one evaluator env, short episode caps, forced update counts, or
  frequent checkpointing. They can prove plumbing, not stock learning.
- Pretrained OpenDILab claims until the checkpoint and config surface match and
  strict-load eval succeeds.

## Plain Read

The clean lane exists now. Call it `installed LightZero 0.2.0 faithful-short`
unless it runs the full `200000` package budget. Use stock evaluator return as
the primary score when strict load and no fallback are clean. Use manual
512-step telemetry to explain the score, not to replace it.
