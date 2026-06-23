# Critiques

Created: 2026-05-19.

Five parallel critiques were run against the exploration-bonus plan. This file
captures the pressure that changes the plan.

## Critique: The Plan Is Too Framework-Shaped

Verdict: accept.

Risk: building `TrainingExtension`, `HookBundle`, broad config schemas, native
count bonuses, support scaling, and tournament hash changes before a working
RND meter creates architecture without evidence.

Plan change:

- First patch has only `none` and `rnd_meter_v0`.
- Defer generic extension layers.
- Defer native count/coverage bonus.
- Defer nonzero schedules, caps, and support scaling until the positive-weight
  canary.
- Defer tournament pool-hash behavior until an RND-trained checkpoint can
  actually exist.

Decision after first smoke: meter-only RND is diagnostic/non-resumable until a
reward-model state hook is added to existing full-resume plumbing.

## Critique: RND Must Stay In The Trainer Process

Verdict: accept.

Risk: a separate Modal service, env-worker predictor, or mutable Modal Dict adds
state drift, device-placement drift, replay ownership ambiguity, and fragile
resume semantics.

Plan change:

- Run RND in the same Modal trainer function and same Python process as
  LightZero.
- Select `train_muzero` versus `train_muzero_with_reward_model` once before
  hook installation.
- Patch hooks against the selected callable, then call that same callable.
- Extend existing resume/checkpoint hooks instead of adding a hidden second
  resume path.

Verified by dry/profile smoke against installed `LightZero==0.2.0`: the
entrypoint path exists, the selected callable can be patched, and the real replay
sample layout includes the `(B,T*C,H,W)` case now covered by tests.

## Critique: The First RND Input Should Be Latest Gray64

Verdict: accept.

Risk: supporting latest-frame, stack4, latent, and source-state inputs at once
makes shape bugs and experiment interpretation harder.

Plan change:

- Default to `policy_gray64_latest/v0`.
- Adapter emits `(N,1,64,64)` NCHW `float32` in `[0,1]`.
- Source is the newest frame from the existing `(4,64,64)` policy observation
  stack.
- Keep `policy_gray64_stack4/v0` as a later input canary.
- Keep MuZero latent RND as research-only.
- Keep source-state features for a separate count/coverage idea, not default
  RND.

Open question: whether latest-frame temporal blindness is severe enough to move
to stack4 after the first positive-weight canary.

## Critique: Meter-Only Must Be A Hard Gate

Verdict: accept.

Risk: if `weight=0.0` cannot be proven behavior-neutral, any positive-weight
result is uninterpretable.

Plan change:

- E0m requires augmented target rewards to equal extrinsic targets exactly.
- E0m metadata must say `training_effect: reward_target_unchanged` and
  `target_reward_effect: unchanged`.
- RND metrics are allowed at `weight=0.0`; learning targets are not allowed to
  change.
- Positive-weight RND is blocked until E0m passes.

Open question: where to write the exact target-equality proof so it is visible
in both tests and launch summaries.

## Critique: Shape Adapter Is The Fragile Interface

Verdict: accept.

Risk: upstream LightZero RND code assumes a particular target length and flat obs
shape. Curvy observations are stacked image tensors and must not pass through a
hardcoded reshape silently.

Plan change:

- Build a Curvy adapter that derives batch and unroll lengths from the actual
  replay tensors.
- Never hardcode the target-step count.
- Never mutate replay data in place.
- Test latest-frame extraction, range, terminal/final observation handling, and
  target reward identity at `weight=0.0`.

Open question: which exact LightZero batch keys are stable enough to use for
Curvy's installed version.

## Critique: Checkpoint Metadata Should Fingerprint RND, Not Store It

Verdict: accept.

Risk: dumping heavy predictor state into policy metadata makes sidecars noisy,
fragile, and hard to compare. Not recording the state at all makes resume
unsafe.

Plan change:

- Store heavy RND state in the existing resume sidecar.
- Record mode, config hash, input spec, and state hashes in checkpoint metadata.
- Tournament/eval scoring remains extrinsic-only.
- Later tournament stratification can use exploration hash once RND checkpoints
  exist.

Open question: whether metric-only runs should appear in tournament intake at
all, or stay training-diagnostic only.

## Critique: Keep The Documentation Packet Small

Verdict: accept.

Risk: many dated planning files make the work harder to operate than the code
itself.

Plan change:

- Keep one front door: `README.md`.
- Keep one live board: `TODO.md`.
- Keep one process note: `ORCHESTRATION.md`.
- Keep one gate plan: `EXPERIMENT_PLAN.md`.
- Keep one critique synthesis: `CRITIQUES.md`.
- Leave `INTEGRATION_PREP.md` as the interface/spec scratchpad.

Open question: when enough decisions have hardened to promote a stable version
into `docs/design/training/`.
