# Champion Bootstrap Modal Red Team

Date: 2026-05-14

## Purpose

Make the next production-shaped training launch boring and correct.

The target contract is:

```text
trusted final rating
-> immutable public leaderboard snapshot
-> rank-1 champion checkpoint ref
-> stable_slots_v1 assignment
-> fresh trainer starts from champion model weights
-> trainer uses immutable assignment opponents
-> trainer writes new checkpoints
-> intake/tournament/leaderboard continue
```

Do not confuse these links:

- tournament ingesting checkpoints;
- trainer starting from champion weights;
- trainer using leaderboard-derived opponents;
- running trainer refreshing opponents;
- controller launching a fresh next attempt.

## Current Smoke State

Remote champion bootstrap smoke attempts:

- `champion-bootstrap-smoke-20260514a`: failed before training because the
  bootstrap loader treated the policy wrapper as the primary model and found no
  matching state-dict keys.
- `champion-bootstrap-smoke-20260514b`: failed the same way after a first local
  ordering fix.
- `champion-bootstrap-smoke-20260514c`: failed. The trainer reached
  `train_muzero`, but the bootstrap hook saw only a policy wrapper with a
  3-key `state_dict`; the real MuZero model was not visible at the first
  `before_run` load point.
- `champion-bootstrap-smoke-20260514d`: failed earlier in recursive discovery.
  DI-engine autolog objects can raise `KeyError` from `__getattr__`, so plain
  `hasattr(obj, "load_state_dict")` is not safe in this runtime.
- `champion-bootstrap-smoke-20260514e`: next proof lane after replacing module
  discovery with safe attribute access. It failed usefully: the policy wrapper
  state dict is `{model, target_model, optimizer}`. The real LightZero API here
  wants wrapper-level load, not a visible `_model` attribute.
- `champion-bootstrap-smoke-20260514f`: passed. It loaded the rank-1 checkpoint
  before LightZero's original `before_run`, replaced `model` and `target_model`
  weights through the policy wrapper, preserved the fresh optimizer, collected
  64 env steps, and completed one learner train call.

Local tests currently pass after the 20260514f patch:

```text
17 passed, 2 skipped
ruff: all checks passed
```

Remote accepted proof:

```text
champion-bootstrap-smoke-20260514f
ok=true
auto_resume=false
load_phase=before_original_before_run
candidate=policy_wrapper_model_target
loaded_key_count=342
nested model loaded 171/175 keys
nested target_model loaded 171/175 keys
fresh_optimizer_preserved=true
env_steps_collected=64
learner_train_calls=1
```

Deployed trainer app after the accepted proof:

```text
curvyzero-lightzero-curvytron-visual-survival-train
deployed 2026-05-14 after champion-bootstrap-smoke-20260514f
```

The four skipped keys are reward/value support heads:

```text
dynamics_network.fc_reward_head.3.*
prediction_network.fc_value.3.*
```

That is acceptable for this matching-shape canary because the checkpoint and the
smoke reward config use different support shapes. It must remain visible in
summaries.

The skipped tests need torch locally; the remote smoke is the real container
proof for model loading.

## Known Guardrails Already Added

- `initial_policy_checkpoint_ref` rejects mutable refs like `latest` and
  `ckpt_best`.
- Train auto-resume now blocks champion bootstrap instead of silently winning.
- Initial model load happens before LightZero's original `before_run`.
- If the model is not visible before LightZero's original `before_run`, the
  hook records that deferral and retries once after `before_run`. This is only
  allowed for the "no real model visible yet" case; real mismatches still fail.
- Bootstrap rejects "successful" loads into tiny policy wrappers; a
  matching-shape load must hit a model-sized state dict and at least half the
  checkpoint keys.
- Recursive module discovery uses safe attribute access because DI-engine
  wrappers can raise non-`AttributeError` exceptions for missing attributes.
- Policy-wrapper bootstrap preserves the fresh optimizer state while replacing
  only `model` and `target_model` nested weights.
- Grouped submit writes assignment-bank artifacts before spawning train/poller.
- Assignment refs are idempotent: same payload retry is okay, same ref with a
  different payload is a hard error.
- Assignment and initial-checkpoint readers call `runs_volume.reload()` before
  resolving cross-function artifacts, so warm containers do not trust stale
  Volume views.
- Grouped submit rejects train-only initial-checkpoint kwargs in poller kwargs.
- Manifest rows record one shared rank-1 immutable `iteration_N.pth.tar`
  initial checkpoint.

## Modal Facts Checked

Official Modal docs support these contracts:

- `Function.remote()` blocks until the remote function returns; `Function.spawn()`
  submits work and returns a `FunctionCall` without waiting.
  Source: <https://modal.com/docs/reference/modal.Function>.
- `Function.from_name(app, fn)` references a Function from a deployed App by
  name. Source: <https://modal.com/docs/reference/modal.Function>.
- Volume writes must be committed to become visible outside the writer
  container, and a running reader container must call `reload()` to see commits
  made after its Volume was mounted. Source:
  <https://modal.com/docs/guide/volumes>.
- `modal run --detach` keeps an ephemeral app alive if the client disconnects;
  production should prefer deployed apps for clean grouping and persistence.
  Sources: <https://modal.com/docs/reference/cli/run> and
  <https://modal.com/docs/guide/apps>.
- Redeploying an existing app creates a new version; old containers finish
  accepted work while new containers take new work. Source:
  <https://modal.com/docs/guide/managing-deployments>.

## Red-Team Questions

1. Does the trainer load the correct model object, not a wrapper or stale copy?
2. Does loading only matching-shape tensors create a half-compatible policy that
   silently skips important heads?
3. Does the target model need to be copied from the online model after load, or
   loaded independently from the checkpoint?
4. Does LightZero create collect/eval model wrappers before or after
   `before_run`, and do they see the loaded weights?
5. Does same-run auto-resume block only the right cases?
6. Does assignment-bank writing happen exactly once and before training starts?
7. Are Modal `.remote()` assignment writes durable before subsequent `.spawn()`
   train/poller calls?
8. Are local-entrypoint ephemeral Modal apps acceptable for canary smokes, while
   production batch uses the deployed app?
9. Does the next batch consume an immutable launch bundle, or can mutable
   `latest.json` change under us?
10. Are we proving the full loop, or only one link?

## Parallel Lanes

| Lane | Owner | Scope | Output |
| --- | --- | --- | --- |
| Modal primitives | Hume + Helmholtz | `.remote` vs `.spawn`, deployed app lookup, Volume commit/reload, `modal run --detach`, local copy patterns. | Need blocking assignment `.remote`, deployed app submit, reader reload, durable Volume status. |
| Trainer bootstrap | Avicenna | LightZero hook timing, module selection, model/target/wrapper state, auto-resume collision. | Do not accept wrapper/tiny loads; prove canonical model load remotely; add role/digest observability later. |
| Assignment submit | Hooke | Assignment-bank write-before-spawn and grouped submit schema. | Reload before assignment reads; make writes idempotent; avoid orphan pollers; tighten schema over time. |
| Full-loop architecture | Russell | End-to-end controller shape and launch bundle contract. | Use one immutable `launch_bundle.json`; mutable Dict/latest are only hints. |
| Docs consistency | Carver | Find stale claims after the correction. | Patch stale "production-shaped" and "closed-loop" language. |

## Ten Current Bugs / Foot-Guns

1. Bootstrap can appear to work while loading only the policy wrapper.
2. `matching_shape` can create a half-random policy if success means "one key
   loaded"; current guard requires a model-sized load.
3. Loading before `before_run` may be too early for LightZero's real model
   object; current hook retries after `before_run` only for hidden-model cases.
4. Online/target/collect/eval synchronization is still not fully proven by
   digest; remote smoke must at least prove the main model load.
5. Assignment writer retries could mutate an immutable ref; current patch makes
   same-ref/different-payload a hard error.
6. Warm Modal containers can read stale assignment/checkpoint views without
   `runs_volume.reload()`.
7. Poller-before-train can leave an orphan poller if train spawn fails.
8. `latest.json` / Modal Dict pointers can drift between champion and assignment
   reads.
9. The launch bundle contract is still mostly prose, not a required schema.
10. Docs still contain stale phrases that overstate static assignment batches as
    production-shaped.

## Minimal Launch Bundle Direction

Near-term production-shaped launches should consume one immutable bundle, not
separate mutable reads. Minimal fields:

- `schema_id`, `bundle_id`, `created_at`, `controller_id`
- `leaderboard_snapshot_ref`, `leaderboard_snapshot_sha256`
- `rating_context_hash`, `rating_status`
- `champion_checkpoint_ref`, ideally `champion_checkpoint_sha256`
- `assignment_ref`, `assignment_sha256`
- `selection_contract`: `stable_slots_v1`
- `training_contract`: fresh run, model-only champion bootstrap, no auto-resume

The manifest builder should reject missing fields and copy the bundle identity
into every row.

Follow-up from the controller critique: the next production-shaped manifest rows
should inherit only these two training inputs from the bundle:

```text
initial_policy_checkpoint_ref = bundle.champion_checkpoint_ref
opponent_assignment_ref = bundle.assignment_ref
```

and should also record:

```text
launch_bundle_ref
launch_bundle_sha256
```

## Bootstrap Acceptance Bar

Do not accept a remote smoke merely because `train_muzero` starts. The smoke is
accepted only if the summary proves:

- no same-run auto-resume blocked or replaced champion bootstrap;
- a model-sized candidate loaded, not the policy wrapper;
- the load phase is recorded (`before_original_before_run` or
  `after_original_before_run`);
- skipped keys/shape mismatches are visible;
- ideally, online/target/collect/eval role digests are added before declaring
  this production-hardened.

Near-term pragmatic decision: a high-coverage matching-shape model load is good
enough for the next canary. Longer-term, strict load or critical-prefix coverage
should be required for champion bootstrap.

## Assignment Handoff Acceptance Bar

The launch barrier is:

```text
assignment writer .remote() returned
+ writer committed Volume
+ train/poller reload Volume before resolving assignment_ref
+ returned assignment ref/hash match manifest
```

`remote()` alone is not the barrier; `spawn()` is never a write barrier.

## Immediate Rule

When blocked on the main proof, run the smallest honest lane that exercises the
same contract. Do not monitor a lane for a link its manifest did not enable.
