# V2 Volume Migration - 2026-05-14

## Why This Exists

The old training Volume is no longer a good control surface for the next real
run. Reads still work, but small writes have started failing with Modal's
`too many layers in volume` error, and the old `curvyzero-runs` Volume is near
its inode limit. That makes pointer writes, assignment writes, and cleanup
fragile.

The next real CurvyTron coach/tournament lane should move fragile checkpoint
and tournament-artifact writes to verified v2 Volumes. Do not infer VolumeFS
version from the name alone; verify with Modal.

| Purpose | Old Volume | New Volume |
| --- | --- | --- |
| training checkpoints, assignments, trainer artifacts | `curvyzero-runs` | `curvyzero-runs-v2` |
| tournament rounds, ratings, public leaderboards, GIFs | old contents in `curvyzero-curvytron-tournaments` | current `curvyzero-curvytron-tournaments` object, verified VolumeFS v2 |

## Current Truth

- 2026-05-15 all-v2 reset: the active lane now uses `-v2` names for every
  durable object, not the prior hybrid state. Current objects are
  `curvyzero-runs-v2`, `curvyzero-curvytron-tournaments-v2`,
  `curvyzero-curvytron-control-v2`,
  `curvyzero-curvytron-checkpoint-intake-v2`,
  `curvyzero-curvytron-checkpoint-events-v2`, and
  `curvyzero-curvytron-opponent-leaderboard-live-v2`. Current apps are
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`,
  `curvyzero-checkpoint-tournament-v2`, and
  `curvyzero-curvytron-gif-browser-v2`.
- The earlier note below about `curvyzero-curvytron-tournaments` being the
  current tournament Volume is historical. It was true for the hybrid lane and
  is no longer the launch contract.
- The old `champ18a` jobs produced useful checkpoints, but they were launched
  with static assignment refs. They do not prove running-trainer refresh.
- A read-only health audit of the 18 old champ rows found all 18 started and
  wrote artifacts, but 10 had failed and 8 were still running. Survival/eval
  mean steps were noisy rather than cleanly rising. Treat those rows as useful
  evidence and seed material, not as the next production lane.
- A relaunch reused the same run/attempt ids. At least one row failed correctly:
  `initial_policy_checkpoint_ref` was set, but same-run auto-resume found an
  existing checkpoint. That is a guard doing its job, not a training signal.
- The next real batch must use fresh run ids and fresh attempt ids.
- The next real batch should use v2 Volumes by default so it does not depend on
  old Volume compaction or cleanup timing.
- 2026-05-15 correction: the current deployed lane does not use separate `-v2`
  app names and does not have `curvyzero-curvytron-control-v2`. Verified
  current objects are: `curvyzero-runs-v2` is actual VolumeFS v2;
  `curvyzero-curvytron-tournaments` is actual VolumeFS v2 despite no suffix;
  `curvyzero-curvytron-control` is actual v1; `curvyzero-runs` is actual v1.
  The shared code now records this as an explicit volume-version map, not a
  suffix rule.
- Two v2 canary launches failed before creating train artifacts because of bad
  operator kwargs, not because of the v2 storage loop:
  `curvy-v2-looplive-proof-20260515a` passed unsupported `wait_for_train`, and
  `curvy-v2-looplive-proof2-20260515a` passed invalid
  `background_eval_launch_kind='none'`. Do not reuse those run ids.
- V2 canary `curvy-v2-looplive-proof3-20260515a` passed the storage refresh
  proof: it wrote checkpoints on `curvyzero-runs-v2`, v2 intake-spawned rating
  `elo-v2-looplive-proof3-r0-20260515a` completed `1` pair / `3` games / `0`
  failures, direct rating also completed on verified-v2
  `curvyzero-curvytron-tournaments`, promotion wrote a control assignment
  with sha
  `adb04ed3905fb9c8984e5e213a9261079f0e4be188315912d12ae5290d55b770`, and the
  same running v2 trainer applied that sha at train iter `1904`.
- Keep old useful jobs alive only while they are still providing signal. Kill
  clearly failed/noisy jobs after an audit, then clean their artifacts.
- Cleanup action taken: stopped stale failed app `ap-5CRdftJaDAF5LUxT9iw2C6`,
  which was repeatedly failing old one-frame tournament games with
  `FileNotFoundError`. Kept the useful old trainer and two useful old tournament
  apps alive.

## Migration Shape

1. Create the v2 Volumes.
2. Patch the CurvyTron coach/tournament/browser/operator surfaces to default to
   v2 Volumes, with environment variables as an escape hatch.
3. Copy only the seed artifacts needed for the next launch from the old runs
   Volume into `curvyzero-runs-v2`.
4. Deploy v2 app names first, so the old deployed lane can keep cooking while
   the v2 lane is validated.
5. Launch a tiny v2 canary that proves:
   - assignment files can be written to v2;
   - the trainer can load the champion bootstrap checkpoint from v2;
   - the trainer writes checkpoints to v2;
   - the v2 tournament can read those checkpoints from v2.
6. Once that passes, launch the fresh real batch with fresh ids.

## Executed So Far

- Created fresh v2 runs Volume:
  - `curvyzero-runs-v2`
- Verified existing tournament artifact Volume:
  - `curvyzero-curvytron-tournaments` is actual VolumeFS v2.
- Current deployed CurvyTron lane defaults:
  - train app: `curvyzero-lightzero-curvytron-visual-survival-train`
  - tournament app: `curvyzero-checkpoint-tournament`
  - GIF browser: `curvyzero-curvytron-gif-browser`
  - checkpoint intake Dict/Queue and live leaderboard Dict retain their current
    non-v2 names unless explicitly changed.
- Deployed the v2 train, tournament, and GIF apps.
- Verified the seven minimal seed refs can be read back from
  `curvyzero-runs-v2`.
- Built fresh-ID manifest:
  `artifacts/local/curvytron_tonight18_manifests/curvy-v2champ18-20260514a/curvy-v2champ18-20260514a.json`.
- Dry-run submit passed:
  `artifacts/local/curvytron_tonight18_manifests/curvy-v2champ18-20260514a/submission-dryrun.json`.
- Full submit completed:
  `artifacts/local/curvytron_tonight18_manifests/curvy-v2champ18-20260514a/submission-full.json`.
- Full submit wrote three v2 assignment artifacts and spawned 18 train calls plus 18 pollers.
- First health pass: all 18 run manifests exist in `curvyzero-runs-v2`; all 18 rows wrote `progress_latest.json` at iteration 0 between 22:30:35Z and 22:31:03Z.
- Second health pass: all 18 rows reached nonzero progress, roughly
  `iteration=10000` through `iteration=30000`. Row `r001` visibly has
  `iteration_0.pth.tar`, `iteration_10000.pth.tar`, and
  `iteration_20000.pth.tar` in `curvyzero-runs-v2`.
- Incident note: the v2 train app stopped showing in `modal app list` after the
  first launch, while v2 tournament and GIF apps remained deployed. Treat the
  first v2 batch as an early plumbing proof unless a later app check shows the
  train app/tasks are still live.
- Relaunch rule: do not reuse `curvy-v2champ18-20260514a` run ids, because
  they already wrote checkpoints. A relaunch should use fresh run/attempt ids
  or explicitly resume without champion bootstrap.
- Fresh relaunch `curvy-v2champ18b-20260514a` was submitted at about
  2026-05-14 21:33 EDT:
  - manifest:
    `artifacts/local/curvytron_tonight18_manifests/curvy-v2champ18b-20260514a/curvy-v2champ18b-20260514a.json`;
  - submission:
    `artifacts/local/curvytron_tonight18_manifests/curvy-v2champ18b-20260514a/submission-full.json`;
  - 18 train calls and 18 pollers spawned on
    `curvyzero-lightzero-curvytron-visual-survival-train-v2`;
  - the v2 train app was visible immediately after submit with 37 tasks;
  - first artifact pass found 18/18 run manifests and 18/18
    `progress_latest.json` refs present; `summary.json` was not expected yet.
- Important correction: `curvy-v2champ18b-20260514a` still uses static
  launch-time immutable assignments. It is useful as a fresh trainer/checkpoint
  producer, but it does not by itself prove the full closed loop back into a
  running trainer.
- Full-loop attempt launched next to v2b:
  - tournament: `curvy-v2champ18-live-20260514a`;
  - rating: `elo-v2champ18-live-20260514a`;
  - app run: detached `curvyzero-checkpoint-tournament-v2`
    `ap-f2KlkPa9UCTjJ0A6NPBkZO`;
  - seed found all 18 exact v2b run ids and queued their first checkpoint refs;
  - rating call: `fc-01KRMMEMG81BEZYVHNJB9T165P`;
  - first progress: 153 pairs, 3213 games, 66 started pairs, estimated 1386
    games seen, zero failed games.
- Parallel small v2 proof lane launched:
  - tournament: `curvy-v2tiny-loop-20260514a`;
  - rating: `elo-v2tiny-loop-20260514a`;
  - first four v2b run ids, 4/4 checkpoint refs discovered and queued;
  - first progress: 6 pairs, 126 games, 5 started pairs, estimated 105 games
    seen, zero failed games.
- Update at 2026-05-14 21:44 EDT:
  - tiny lane finished cleanly: 6/6 pairs, 126/126 games, zero failures,
    `stable=true`, ratings written;
  - tiny publish/materialize/trainer-smoke controller is running with strict
    gates, so this can prove the v2 back half quickly if it passes;
  - full 18-way lane has 153/153 pairs started and lightweight progress has
    seen all 3213 expected games with zero failures, but final reducer artifacts
    are still pending.

## What Is Not Yet Validated

The full feedback loop is not done until all of this is seen in artifacts:

1. v2 trainers write new checkpoints.
2. v2 tournament subscriber/intake discovers those checkpoints.
3. v2 tournament rates those checkpoints as a continuation.
4. v2 tournament publishes a clean public leaderboard snapshot.
5. Coach materializes a new immutable assignment from that leaderboard.
6. A trainer consumes that new assignment, either at launch or at a proven
   refresh boundary.

Current v2 proof now covers trainer checkpoint writes, v2 intake-spawned
tournament rating, assignment materialization, and same-running-trainer
refresh. The earlier stuck read was stale progress.

The tiny v2 lane is now the first back-half candidate. If promotion and smoke
pass, it proves steps 2-6 at small scale. It still does not prove that the
already-running `curvy-v2champ18b-20260514a` trainers refresh, because those
rows were launched without a refresh interval or refresh ref.

Update: both the tiny lane and the full 18-way lane passed promotion and trainer
smoke. The full 18-way public leaderboard snapshot has 18 active rows and 0
provisional rows, and the full assignment was written to `curvyzero-runs-v2`.

Next refresh proof is now the new full batch:

- matrix: `curvy-v2refresh18-20260514a`;
- 18 H100 train rows + 18 pollers submitted through
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`;
- starts from three immutable recipe assignments built from the full 18-way
  leaderboard;
- refresh interval is 50 train iterations;
- shared refresh pointer:
  `training/lightzero-curvytron-visual-survival/v2refresh18-control-20260514a/attempts/try-v2refresh18-control-20260514a/opponents/refresh_pointer.json`;
- current pointer target is the full 18-way assignment:
  `training/lightzero-curvytron-visual-survival/v2champ18-live-assignment-bank-20260514a/attempts/try-v2champ18-live-assignment-bank-20260514a/opponents/assignments/v2champ18-live-r0-assignment-20260514a/assignment.json`.

Do not mark this as a refresh success until
`opponent_assignment_refresh_events.jsonl` contains `decision=applied` for at
least one new row and later `env_steps.jsonl` rows carry the full assignment
hash with `opponent_provider_load_ok=true`.

Update: that trainer-side refresh gate passed for refresh18 row `r001`.
The event applied the full 18-way assignment hash
`d881126f31b726b52a1e932b42b3eb3734acbd0e51faef78a8ef7a8b151155e6`, env
readiness was `ok=true` for 256 envs, and downloaded env telemetry showed
41,040 rows using that refreshed hash.

This is still not the full feedback loop. It proves the running trainer can
consume the current pointer target. The remaining proof is that a later
tournament result publishes a new assignment, the pointer changes, and an
already-running trainer applies that newer assignment.

Refresh18 tournament status:

- `curvy-v2refresh18-live-20260514a / elo-v2refresh18-live-20260514a` is live
  on the v2 tournament Volume.
- Intake initially saw only 8/18 checkpoints because it was started while the
  trainers were still writing `iteration_0`.
- A later intake status/tick repaired the manifest to 18/18 checkpoints with
  zero missing refs.
- `round-000000` rated the first 8 checkpoints: 28 pairs, 588 games, zero
  failures.
- `round-000001` rated the 18-checkpoint all-pairs pool: 153 pairs, 3213
  games, zero failures. The explicit game-summary progress check counted all
  3213 games.
- Promotion of `round-000001` is blocked until the durable `latest.json`
  advances from `round-000000` to `round-000001`. The first promotion attempt
  correctly refused the stale latest pointer.

Update at 2026-05-14 22:10 EDT: this lane is contaminated and must not be used
as promotion proof. Because `latest.json` stayed stale, a continuation reused
`round-000001` and overwrote `input.json`/`progress.json` with a newer
20-checkpoint round, while `ratings.json`/`results.json` still describe the
older 18-checkpoint result. The fix is a stale-latest/round-overwrite guard.
The clean replacement proof lane is
`curvy-v2refresh18-proof-20260514b / elo-v2refresh18-proof-20260514b`, launched
with `modal run --detach`, no continuation, 18 current trainer checkpoints,
all-pairs, 21 games per pair, one-frame, and GIFs on.

The next proof must explicitly cover the missing back half. Do not call v2b a
closed-loop success until artifacts show that one of its checkpoints was
discovered by intake, rated by the v2 tournament, published into a public
leaderboard snapshot, materialized into an immutable assignment, and then
consumed by a trainer at launch or through a recorded refresh event.

## Minimal Seed Copy

The v2 runs Volume starts empty. A v2 trainer cannot load old refs unless those
checkpoint files exist under the same relative refs in `curvyzero-runs-v2`.

Copy only the current trusted source checkpoints and assignments, not whole old
training trees:

- the rank-1 champion used for `initial_policy_checkpoint_ref`;
- the leaderboard checkpoint refs used by tonight's assignment bank;
- any explicitly chosen top checkpoints used for a small fallback arena.

Current minimal seed set for the next v2 canary:

- `curvy-champ18a-assignments/.../blank5-wall5-rank2_25-rank1_65/assignment.json`
- `curvy-champ18a-assignments/.../blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35/assignment.json`
- `curvy-champ18a-assignments/.../blank20-wall5-rank1_75/assignment.json`
- `curvy-n18conn-sparse-blank5-wall5-rank2_25-rank1_65-so10rep10.../iteration_40000.pth.tar`
- `curvy-n18conn-survbonusnoout-blank20-wall5-rank1_75-clean.../iteration_40000.pth.tar`
- `curvy-n18conn-survbonusnoout-blank20-wall5-rank1_75-so10rep10.../iteration_40000.pth.tar`
- `curvy-n18conn-survbonusout-blank5-wall5-rank2_25-rank1_65-clean.../iteration_20000.pth.tar`

## Operating Rule

Do migration work in parallel with monitoring. Do not block every lane on full
cleanup. If a v2 canary can run while cleanup is still auditing old jobs, run
the canary. Promotion stays gated on artifacts, but learning work should stay
parallel.
