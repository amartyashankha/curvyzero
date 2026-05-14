# Granular Action Cadence Side-Effect Audit

Date: 2026-05-13

## Plain Question

We changed the trusted stock LightZero lane from bundled source ticks to one
policy action per source physics step. The local contract tests pass, but this
is still a large behavior change.

This doc tracks what could break or change downstream.

## Current Known Status

- Local focused tests passed: `161 passed, 1 skipped`.
- Lint passed for touched cadence files.
- `git diff --check` passed.
- A fresh real training-loop smoke has not yet been run after the cadence guard.
  That is an explicit validation gap, not a learning claim.

## Main Side-Effect Questions

1. Does stock `train_muzero` still run end to end with the new one-frame
   cadence and the `decision_ms` guard?
2. Does LightZero step accounting change enough to make `max_env_step`,
   `max_train_iter`, `td_steps`, replay size, or checkpoint cadence misleading?
3. Does `source_max_steps` now mean source physics frames everywhere the trusted
   lane claims it does?
4. Do rewards and value-support scales still make sense when one policy
   transition is now much shorter in game time?
5. Do background eval, GIF generation, summaries, and the website see and
   display the cadence clearly?
6. Do launch manifests, old run manifests, or external scripts still pass stale
   `decision_ms=200/300` into the trusted lane?
7. Do downstream checkpoint consumers, tournaments, opponent selection, or
   policy browsers mix policies trained under different cadence without saying
   so?
8. Are old product-fidelity, checkpoint-tournament, and two-seat lanes clearly
   labeled as outside this trusted cadence contract if they keep bundled steps?

## Parallel Audit Lanes

| Lane | Owner | Focus | Output |
| --- | --- | --- | --- |
| Trainer loop | Raman | LightZero collector/replay/learner/checkpoint effects | `granular_action_cadence_trainer_loop_side_effects_2026-05-13.md` |
| Env semantics | Euclid | rewards, source caps, timers, wall/trail physics | `granular_action_cadence_env_reward_side_effects_2026-05-13.md` |
| Eval/GIF/site | Dirac | background eval, GIF summaries, website visibility | `granular_action_cadence_eval_gif_site_side_effects_2026-05-13.md` |
| Launch/manifests | Dewey | scripts, CLI, Modal app calls, stale values | `granular_action_cadence_launch_manifest_side_effects_2026-05-13.md` |
| Downstream consumers | Lovelace | tournaments, checkpoint browsers, opponent policy use | `granular_action_cadence_downstream_consumers_side_effects_2026-05-13.md` |
| E2E smoke | Darwin | smallest honest post-patch train-loop run | `granular_action_cadence_e2e_smoke_plan_2026-05-13.md` |

## E2E Smoke Plan

Darwin found the smallest honest smoke command. The main thread should run a
fresh waited CPU Modal `--mode train` job with:

- `max_train_iter=1`
- `max_env_step=64`
- `source_max_steps=64`
- one collector env and one evaluator env
- `num_simulations=1`
- `batch_size=4`
- stock LightZero in-loop eval off
- background eval/GIF off
- `wait_for_train=true`

Pass means the real `lzero.entry.train_muzero` path still instantiates the real
source-state visual env, collects, trains once, and writes a checkpoint. It does
not mean the policy learned anything.

## Working Decision

Do not launch serious training runs from this change until:

- one fresh tiny post-patch train-loop smoke completes, or we record a clear
  blocker;
- the parallel side-effect audits are folded into this doc;
- any obvious stale `decision_ms` launch paths are either fixed or explicitly
  labeled outside the trusted lane.

## Findings Folded So Far

### Trainer Loop

Raman's audit says the stock LightZero collector, replay buffer, MCTS/search,
learner, and stock checkpoint save path are still owned by LightZero. The
cadence patch mostly changes the meaning of an env step.

Main side effect: the same `max_env_step`, `source_max_steps`, replay capacity,
and checkpoint iteration now cover less real game time than old bundled-cadence
runs. That is intentional for control, but old and new checkpoints must not be
compared by `iteration_N` alone.

Immediate follow-ups:

- Treat resume from old bundled-cadence runs as mixed-semantics unless cadence
  metadata proves otherwise.
- Run one small background eval/GIF smoke after the pure train smoke, because
  the pure smoke does not prove artifact workers.
- Decide whether `td_steps=source_max_steps` and support scaling need retuning
  before serious learning runs.

### Env And Reward

Euclid's audit says the env contract is now clear: default trusted train means
one policy decision, one opponent decision, one source physics frame, and one
reward tick. Explicit policy repeat is the only allowed way to hold an action.

Main side effect: dense survival reward, `steps_survived`, natural bonus
opportunities, wall/trail timing, and opponent cadence are now source-frame
level. Old bundled runs can look worse or better purely because they reported
different units.

Immediate follow-ups:

- Add or keep artifact fields that show `decision_source_frames`,
  `source_physics_step_ms`, reward variant, repeat fields, and
  `source_max_steps_semantics` together.
- Do not rank old and new checkpoints by raw survival length unless cadence is
  labeled or normalized.
- Add a deterministic one-frame natural-bonus smoke and near-wall/trail timing
  fixtures if this cadence becomes the long-term training surface.

### Downstream Consumers

Lovelace's audit says trusted train/eval artifacts mostly carry cadence now,
but checkpoint consumers are only partly protected. The weak spot is frozen
opponent selection: an old bundled-cadence checkpoint can still be loaded as a
model and used in a new one-frame env. It may load correctly while behaving
differently because it is queried more often.

Tournament code has better protection because it records cadence in runtime
metadata and hashes cadence into rating context. The remaining gap is missing
metadata: if an old checkpoint has no cadence fields, a mixed-cadence comparison
can still be ambiguous.

Immediate follow-ups:

- Treat cadence as a required compatibility field for leaderboard rows and
  future opponent assignment.
- Add `cadence_family`, training cadence, eval cadence, and repeat fields to
  assignment/audit records.
- Default future frozen-opponent selectors should filter to same cadence unless
  a run explicitly opts into mixed-cadence curriculum.
- GIF/browser cards should show cadence so humans do not compare clips under
  different action clocks by accident.

### Eval, GIF, Site, And Checkpoint Display

Dirac's audit says the train command, env config, telemetry, and eval result
rows now contain cadence fields. The weak point is what humans and background
workers see.

Main side effect: background eval/GIF config objects contain cadence, but the
spawned remote eval/GIF functions do not accept cadence fields yet. Today they
match trusted train only because those functions rebuild envs with the same new
default. That is fragile if we later allow explicit repeat/cadence variants or
older checkpoints.

Other display gaps:

- `progress_latest.json` does not show cadence.
- run-status output does not show cadence.
- GIF browser cards say "steps" but not "frames per action".
- mirrored checkpoint paths do not have per-checkpoint cadence metadata.
- tournament discovery can find checkpoint refs without reading cadence.

Immediate follow-ups:

- Pass cadence fields through eval/GIF remote APIs, not just config dicts.
- Add cadence to `progress_latest.json`, run-status rows, GIF summaries, and
  GIF browser cards.
- Add a checkpoint metadata sidecar or index beside mirrored checkpoints.
- Teach checkpoint discovery to warn on mixed or unknown cadence.

### Launch And Manifest

Dewey's audit says the active trusted launch path is mostly protected:
train/dry reject stale bundled `decision_ms`, and the active survivaldiag and
opponent-mixture manifest builders now emit the one-frame value.

Remaining launch risks:

- manifest review command text relies on launcher defaults instead of spelling
  out `--decision-ms`;
- grouped submit validation checks that `decision_ms` exists, but not that it is
  one-frame before spawning pollers/train jobs;
- profile mode can still accept old `decision_ms` metadata even though the env
  config also forces `decision_source_frames=1`;
- historical two-seat reruns can silently inherit the new top-level default if
  not explicitly labeled or pinned;
- tournament defaults still use legacy 12-frame vector cadence.

Immediate follow-ups:

- Add submitter-side stale-`decision_ms` validation before any spawn.
- Make manifest command text cadence-explicit, or mark it as using launcher
  default.
- Decide whether profile should reject stale cadence or clearly normalize and
  label effective cadence.
- Label two-seat and tournament legacy cadence paths so they cannot be mistaken
  for the trusted one-frame stock train lane.

### Post-Patch Smoke Result

Fresh waited CPU Modal smoke:

`curvytron-cadence-e2e-smoke-20260513-202837 / train-smoke-001`

Returned:

- `ok=true`
- `called_train_muzero=true`
- `problems=[]`
- telemetry `row_count=128`

But repeated Volume listing after completion showed only early files:

- `run.json`
- `latest_attempt.json`
- `attempt.json`
- `train/status_heartbeat.json`

Missing from the Volume listing:

- `train/summary.json`
- `train/action_observability.json`
- `train/lightzero_artifacts_manifest.json`
- mirrored `checkpoints/lightzero/iteration_*.pth.tar`

Current interpretation: the real train call ran, but train-mode final artifacts
were not committed or did not become visible. This is now a trainer scaffolding
bug to investigate/fix before claiming full artifact end-to-end success.
