# Modal/RL Debugging Web Research, 2026-05-16

Goal: collect primary-source debugging patterns for the CurvyZero/CurvyTron
trainer -> checkpoint -> subscriber -> tournament -> leaderboard -> trainer
loop, with survival stagnation as the motivating symptom.

## Source Set

Modal official docs:

- Apps and entrypoints: ephemeral apps stop when the caller exits unless run
  with `--detach`; deployed apps persist until stopped.
  <https://modal.com/docs/guide/apps>
- App logs CLI: `modal app logs` supports app id/name, follow, tail, time
  ranges, search, source, function id, function-call id, container id, and
  timestamps. <https://modal.com/docs/reference/cli/app>
- Job processing: deploy a worker, submit with `.spawn()`, return the
  `FunctionCall` id, and poll with `FunctionCall.get(timeout=0)`.
  <https://modal.com/docs/guide/job-queue>
- `modal.Function`: `.spawn()` starts work without waiting and returns a
  pollable `FunctionCall`; `.remote()` waits for the result.
  <https://modal.com/docs/reference/modal.Function>
- Web request timeouts: web functions should finish quickly; long work should
  be spawned and polled by another endpoint.
  <https://modal.com/docs/guide/webhook-timeouts>
- Web functions and URLs: `fastapi_endpoint`, `asgi_app`, `wsgi_app`, and
  `web_server` expose functions over HTTP when served or deployed; URLs can be
  found in CLI/dashboard or via `get_web_url()`.
  <https://modal.com/docs/guide/webhooks>,
  <https://modal.com/docs/guide/webhook-urls>
- Queues: Modal Queues are FIFO communication primitives for active functions;
  they are not durable storage, default partition TTL is 24h after last put,
  and partitions/items have explicit limits.
  <https://modal.com/docs/guide/queues>
- Queue CLI: `modal queue peek` and `modal queue len` are useful
  non-destructive inspection tools; `clear/delete` are destructive.
  <https://modal.com/docs/reference/cli/queue>
- Dicts: Dict entries are persisted but expire after 7 days of inactivity, and
  primitive keys are recommended.
  <https://modal.com/docs/reference/modal.Dict>
- Volumes: readers need `.reload()` to see committed changes from other
  containers; writers need `.commit()` for visibility; background commits
  happen but should not replace deliberate commit points for handoffs.
  <https://modal.com/docs/guide/volumes>
- Volume consistency: avoid concurrent writes to the same file; same-file
  conflict semantics are last-write-wins; reload fails with open files; v2
  improves distinct-file concurrency but same-file concurrent writes are still
  unsafe. <https://modal.com/docs/guide/volumes>
- Long training: make training interruptible/reentrant by checkpointing to a
  Volume, resuming from latest checkpoint, and adding retries.
  <https://modal.com/docs/examples/long-training>
- Preemption/timeouts/retries: long-running functions must tolerate
  interruption, be idempotent, fit within the 24h function timeout, and use
  retries for flaky work.
  <https://modal.com/docs/guide/preemption>,
  <https://modal.com/docs/guide/timeouts>,
  <https://modal.com/docs/guide/retries>
- Troubleshooting: reused containers can preserve local side effects, and
  heartbeat timeouts point to GIL-holding or shutdown stalls.
  <https://modal.com/docs/guide/troubleshooting>
- Async API: `.aio` variants allow parallel Modal calls from `asyncio`, but
  concurrent inputs can create thread/coroutine race hazards depending on sync
  vs async execution.
  <https://modal.com/docs/guide/async>
- Schedules: deploy scheduled functions; `Period` resets on redeploy while
  `Cron` is stable by calendar expression; logs are visible from the Apps UI.
  <https://modal.com/docs/guide/cron>

RL/self-play primary or project-owned sources:

- MuZero paper: MuZero learns reward, policy, and value predictions for
  planning, so debugging must inspect all three heads/targets, not Elo alone.
  <https://arxiv.org/abs/1911.08265>
- AlphaZero paper: tabula-rasa self-play starts from random play, so a perfect
  initial leaderboard is not required; evaluation and checkpoint selection
  still matter. <https://arxiv.org/abs/1712.01815>
- OpenSpiel AlphaZero docs: the actor/learner/evaluator/checkpoint split is a
  useful reference architecture, while noting it is an illustrative
  reimplementation rather than the original DeepMind system.
  <https://openspiel.readthedocs.io/en/latest/alpha_zero.html>
- LightZero config docs: collector/evaluator environment counts, support
  ranges, batch size, replay, and learner settings are explicit tuning and
  debugging knobs.
  <https://opendilab.github.io/LightZero/tutorials/config/config.html>
- LightZero MuZero agent source docs: the loop constructs collector, evaluator,
  replay buffer, learner, then collect -> replay sample -> train; this supports
  inspecting replay ratio and stale opponent data directly.
  <https://opendilab.github.io/LightZero/_modules/lzero/agent/muzero.html>
- Multi-agent replay nonstationarity: changing co-players can make replay
  incompatible with current learning targets; age/fingerprint or decay-style
  fixes are known stabilizers.
  <https://arxiv.org/abs/1702.08887>

## Actionable Modal Debugging Patterns

1. Treat `modal run` as a development surface, not service proof.
   If a command spawns child tournament workers and returns before they finish,
   use a deployed app, wait for child completion, or run the proof with
   `modal run --detach`. Then verify the detached app is stopped when no longer
   needed.

2. Prefer `.spawn()` for background work and `.remote()` for bounded calls.
   A parent that calls child game workers with `.remote()` waits and is simpler
   to reason about; a parent that wants parallel child work should `.spawn()`,
   persist function call ids/status, and reconcile from durable artifacts.

3. Debug by durable artifact advancement, not scheduling logs.
   "Submitted", "spawned", or "round scheduled" is not success. Success means
   the expected Volume files appeared, committed, and are readable after reload:
   checkpoint sidecar, intake manifest, round inputs, completed game summaries,
   `latest.json`, public leaderboard snapshot, assignment/audit, and trainer
   refresh/env telemetry.

4. Make Queue/Dict disposable in every design.
   Queue events should wake reconcilers. Dict keys should cache pointers,
   leases, active manifest keys, or operator intent. Neither should be the only
   record of admitted checkpoints, desired pool version, completed games, or
   published leaderboard identity.

5. Use Volume JSON as the source of truth, but avoid mutable hot files.
   Write immutable per-shard/per-round/per-snapshot files first, commit, then
   move or write one small pointer last. Never let multiple containers write the
   same JSON file concurrently. For v2 Volumes, distinct-file fanout is better,
   but same-file races still need single-writer discipline.

6. Make reload/commit boundaries explicit in readers.
   A long-lived web container, subscriber, or drain may not see another
   container's commit until it reloads. Reload only when no Volume files are
   open. On reload failure, show stale data with a timestamp and retry later
   rather than poisoning progress state.

7. Build status endpoints as pollers, not long requests.
   For tournament controls and diagnostics, use submit/status endpoints:
   `/submit` spawns work and returns an id; `/status/{id}` reads
   `FunctionCall` state plus durable Volume status and returns 202/200 style
   states. Do not make web requests wait for tournament games or trainer
   curves.

8. Log with join keys.
   Every spawned unit should print and persist: app name, function call id,
   run id, attempt id, checkpoint ref, tournament id, rating run id, pool hash,
   shard id, assignment sha, generation, and Volume ref written. Then
   `modal app logs ... --function-call ... --timestamps --show-container-id`
   can be joined to the Volume artifacts.

9. Make leases scoped and expiring.
   A claim for a partial pool must not block a later full pool. Scope leases to
   immutable work identity such as `(rating_run_id, pool_hash, shard_id)` and
   require heartbeat/expiry. Reconciliation should rebuild work from Volume
   manifests if Queue messages vanish.

10. Design trainer work as reentrant.
    The trainer should checkpoint frequently to `curvyzero-runs-v2`, resume from
    the latest valid checkpoint, and make checkpoint writes idempotent. A
    preemption/retry should not double-admit a checkpoint or silently change the
    assignment/reward recipe used for replay.

11. Separate learning health from control-plane health.
    The Modal loop can be healthy while learning regresses. For survival
    stagnation, plot per-checkpoint survival, action distribution, policy/value/
    reward losses, reward/value target support saturation, replay age, and
    opponent assignment generation beside leaderboard rank.

## Mapping To The Current Repo Loop

| Loop stage | Current repo surface | Pattern to apply |
| --- | --- | --- |
| Trainer writes checkpoints | Training app `curvyzero-lightzero-curvytron-visual-survival-train-v2`; runs Volume `curvyzero-runs-v2`; checkpoints under `train/lightzero_exp*/ckpt/iteration_*.pth.tar` | Reentrant checkpointing; commit on checkpoint; sidecar metadata; no live tournament reads during learner updates |
| Checkpoint discovery/subscriber | Intake scanner/ticks, `checkpoint_intake_state` Dict, `checkpoint_intake_queue` Queue | Queue wakes only; durable manifest on tournament Volume is truth; rebuild missing Dict/Queue state from manifest |
| Drain/claim/reconcile | Intake drain and rating claim code in tournament app `curvyzero-checkpoint-tournament-v2` | Scope claims by desired pool hash and claim mode; old partial-pool claim cannot block expanded pool; empty Queue must not imply no desired work |
| Tournament game workers | Tournament Volume `curvyzero-curvytron-tournaments-v2`; spawned game/eval workers | Use deployed or detached parent for work that must outlive local command; use `.spawn()` plus persisted ids for fanout; verify completed summaries, not just round inputs |
| Rating reducer | `latest.json`, per-round `ratings.json`, pair history, scheduler state | Immutable round outputs, single-writer reducer, explicit pool/roster/context hash, fail closed when pool does not match desired manifest |
| Leaderboard publisher | Public snapshot and live Dict pointer `curvyzero-curvytron-opponent-leaderboard-live-v2` | Publish only complete immutable artifacts; write snapshot to Volume and commit before updating Dict pointer; Dict pointer is repairable cache |
| Assignment materializer | `stable_slots_v1`, `assignment.json`, `audit.json`, control Volume `curvyzero-curvytron-control-v2` | Trainer consumes frozen assignment only; assignment sha/generation included in logs and env telemetry; no leaderboard polling inside learner loop |
| Trainer refresh | Run-control pointer at clean collect/restart boundary; default refresh interval `2000` train iterations | Read one small prepared assignment pointer at boundary, verify immutable assignment, then log `decision=applied`; do not live-read Elo, Queue, or slot recipes while learning |
| Web/debug UI | Tournament browser/GIF browser/web endpoints | Use cached/paged Volume reads; explicit reload failures should be visible and non-mutating; long actions should be submit/status jobs |

## Debugging Pass For Survival Stagnation

1. Prove the control plane is not stale:
   - current desired checkpoint count from durable intake manifest;
   - active rating pool count/hash;
   - latest completed `latest.json` round and completed game summary count;
   - published leaderboard snapshot hash;
   - latest assignment sha consumed by each running trainer.

2. Prove the trainer is not reading a moving target:
   - one launch assignment ref;
   - zero live leaderboard/Queue/tournament reads inside learner updates;
   - refresh reads only the control pointer at the clean boundary;
   - replay rows tagged with assignment generation.

3. Prove Volume visibility is not hiding progress:
   - checkpoint sidecars exist beside `.pth.tar` files;
   - subscriber/drain reload before scans and commit after manifest writes;
   - publisher commits snapshot before Dict pointer update;
   - web status shows last successful reload time.

4. Prove late-regression is learning, not measurement:
   - run a frozen-assignment no-refresh control;
   - evaluate best, latest, and tournament-top checkpoints on the same fixed
     seed/opponent grid;
   - compare survival, win/outcome, reward components, Elo, action histogram,
     and policy/value/reward losses.

5. Prove MuZero target health:
   - log reward/value target ranges before categorical support transform;
   - count target values outside configured support;
   - plot replay sample age and opponent assignment generation;
   - compare effective collect/update ratio against intended LightZero config.

## Short Recommendations

- Keep the current "Volume truth, Dict/Queue coordination" boundary. It matches
  Modal's own durability semantics and explains the earlier partial-pool/empty
  Queue failure.
- For any next live proof, use one deployed tournament app or one deliberate
  detached app, not overlapping ephemeral parents.
- Add one operator status bundle that joins Modal app logs, Queue length/peek,
  Dict pointer values, and Volume artifact counts by the same ids/hashes.
- Do not use leaderboard improvement as the only survival-learning gate. Gate on
  fixed-grid survival and action/reward/value health, then let the leaderboard
  decide opponent strength separately.
