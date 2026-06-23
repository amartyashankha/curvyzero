# TODO

Created: 2026-05-19.

This is the live planning board for exploration-bonus work. Keep it smaller than
the implementation. Promote stable decisions into design docs only after a
working canary exists.

## P0: Current Behavior Stays Put

- [x] Name the disabled contract: `exploration_bonus_mode=none`.
  Owner:
  Proof: current trainer entrypoint, config patch, env reward, target rewards,
  sidecars, and tournament behavior remain baseline-equivalent.
  Next: disabled mode now appears in trainer command/summary metadata; LightZero
  config remains unpatched when disabled.

- [x] Add fail-closed validation for any exploration mode string.
  Owner:
  Proof: unknown mode fails before launching Modal training.
  Next: accepted modes are `none`, `rnd_meter_v0`, and blocked plumbing mode
  `rnd_replay_target_v0`.

- [x] Keep env/game reward extrinsic-only.
  Owner:
  Proof: no edits to `VectorMultiplayerEnv` scoring or tournament scoring for
  RND plumbing.
  Next: document any intrinsic metrics as trainer-only diagnostics.

## P1: Meter-Only RND

- [x] Select the LightZero trainer entrypoint once, up front.
  Owner:
  Proof: `none` calls `train_muzero`; `rnd_meter_v0` calls
  `train_muzero_with_reward_model`; hook patching targets the selected callable.
  Next: entrypoint name is visible in command, summary, and compact output.

- [x] Treat RND config as one atomic bundle.
  Owner:
  Proof: entrypoint, policy/create type, `cfg.reward_model`,
  `policy.use_rnd_model`, Curvy adapter config, and checkpoint schema are
  enabled together or not at all.
  Next: keep builder surface validation pinned to entrypoint, RND policy flag,
  reward model type, and exploration config hash.

- [x] Pin the first RND input to `policy_gray64_latest/v0`.
  Owner:
  Proof: adapter emits `(N,1,64,64)` NCHW `float32` in `[0,1]` from the newest
  frame of the current `(4,64,64)` policy observation stack.
  Next: shape/range/latest-frame extraction tests are in
  `tests/test_exploration_bonus.py`.

- [x] Build the Curvy replay-batch adapter.
  Owner:
  Proof: derives batch and unroll lengths from tensors, never hardcodes `6`,
  never mutates replay data in place, and restores augmented rewards to the
  expected LightZero layout.
  Next: add terminal/final-observation variants after a real LightZero sampled
  batch is inspected.

- [x] Run RND with `weight=0.0`.
  Owner:
  Proof: Modal dry selected `lzero.entry.train_muzero_with_reward_model` with
  no problems; Modal profile
  `rnd-meter-profile-local-20260519c` completed one learner train call with
  `replay_sample_calls=1`, `learner_train_calls=1`, `problems=[]`, and
  `weight=0.0`. The first profile exposed LightZero's real replay sample shape
  `(B,T*C,H,W)`, which is now covered by adapter tests.
  Next: keep positive-weight RND blocked until metrics sidecar and resume
  behavior are explicit.

- [x] Add the minimal metric surface.
  Owner:
  Proof: command/config/surface metadata now includes mode, entrypoint, input
  shape, target reward effect, trainer effect, RND knobs, config hash, and
  runtime metrics refs.
  Next: keep extending the compact JSON sidecar, but the first sidecar now
  proves construction, collect/train/estimate call counts, buffer size,
  predictor/target hashes around training, latest loss, intrinsic mean, and
  exact target-equality status for `weight=0.0`.

- [x] Add a tiny hard gate for real RND trainer proof.
  Owner:
  Proof: `require_rnd_metrics=true` fails the run unless the RND sidecar exists,
  `collect_data`, `train_with_data`, and `estimate` ran, predictor weights
  changed, target weights stayed frozen, and `weight=0.0` target rewards stayed
  unchanged.
  Next: use this only on tiny profile smokes until resume/checkpoint ownership
  is explicit.

## P1.5: Resume And Checkpoint Reality

- [ ] Extend the existing full-resume hook rather than adding a second resume
  system.
  Owner:
  Proof: RND state is captured after the reward model exists and before training
  resumes.
  Next: meter-only RND is currently diagnostic/non-resumable; identify the exact
  LightZero holder field for the reward model before enabling long runs.

- [ ] Store heavy RND state in the resume sidecar.
  Owner:
  Proof: predictor, target, optimizer, observation normalizer, reward
  normalizer, counters, and config hash can restore or fail clearly.
  Next: for the first meter-only patch, decide whether to support resume or mark
  the mode diagnostic/non-resumable.

- [ ] Fingerprint, do not embed, RND state in checkpoint metadata.
  Owner:
  Proof: checkpoint metadata records exploration mode/hash/input spec and state
  hashes without dumping model tensors into JSON.
  Next: keep tournament scoring extrinsic while later stratifying pools by
  exploration hash.

## P2: First Positive-Weight Canary

- [ ] Add or document the actual launch gate for positive RND.
  Owner:
  Proof: `rnd_replay_target_v0` is accepted by the core spec and has unit
  tests for reward mutation, so "blocked" is currently a process/doc posture
  unless the submitting path refuses it. Either add a clear launcher gate or
  rewrite this board to say positive mode is allowed only in named canary
  manifests.
  Next: before any overnight run, confirm the active launcher cannot
  accidentally run positive RND while normalization/resume gates are open.

- [ ] Enable one low-weight RND canary only after P1 is behavior-neutral and
  normalization is decided.
  Owner:
  Proof: blocked. Positive RND plumbing exists, but normalization, resume,
  retained extrinsic quality, and seed robustness remain open.
  Next: keep launched blank-canvas ladders historical/diagnostic only.

- [ ] Pin cap/schedule/support fields before positive RND exists.
  Owner:
  Proof: trainer CLI/Modal kwargs, builder config, checkpoint metadata, and
  docs agree on exact names and defaults for cap, schedule, and decay horizon.
  Next: update `lightzero_target_config_for_reward(..., exploration_bonus=...)`
  or equivalent so requested/effective reward and value support include the
  bounded intrinsic term.

- [ ] Measure retained extrinsic quality.
  Owner:
  Proof: compare best and latest checkpoints by sparse outcome, survival, action
  distribution, tournament rank, intrinsic/extrinsic target ratio, and support
  clipping.
  Next: require at least one seed that does not regress latest-checkpoint
  quality before broad sweeps.

- [ ] Audit RND reward decay semantics separately from speed profiles.
  Owner:
  Proof: compare predictor update count/rate against target estimate calls, and
  decide whether per-batch min/max normalization makes the intrinsic reward
  merely relative within each sampled batch instead of globally decaying as
  observations become familiar.
  Current: code default is now `rnd_update_per_collect=100`, with raw MSE stats
  logged before LightZero-style batch min/max normalization. Optimizer fixed
  the diagnostic hash overhead on 2026-05-22: predictor/target hashes now run
  once before and once after the update batch, not once per update.
  Next: treat running intrinsic normalization as a correctness/research lane for
  positive RND, not an optimizer-overhead or batch-size smoke result.

## Parking Lot

- [ ] Stack4 RND input after latest-frame RND is stable.
- [ ] MuZero-latent RND research path only.
- [ ] Source-state count/coverage bonus as a separate native novelty canary.
- [ ] Generic `TrainingExtension`/`HookBundle` layer.
- [ ] Env-worker RND, separate RND service, Modal Dict predictor state, or eval
  mutation of RND state.
- [ ] Optimizer speed smoke for RND with `batch_size >= 32`; batch size 1 failed
  inside training after reaching `train_muzero_with_reward_model`. Prefer the
  normal learner batch and `rnd_update_per_collect=100` unless intentionally
  ablating cadence.
