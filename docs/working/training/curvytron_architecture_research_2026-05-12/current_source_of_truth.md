# Current Source Of Truth

Date: 2026-05-12/13 overnight

Purpose: keep the active CurvyTron training worldview in one place. Older docs
are useful history, but this is the current decision surface.

## North Star

Train CurvyTron well enough that we trust the training setup, observability, and
analysis loop before returning to richer self-play. The immediate learning
target is not "beat a weak opponent"; it is "learn to survive longer from visual
input."

## Current Read

The stock LightZero path can move some metrics, but the last v1d matrix did not
prove a useful CurvyTron training setup.

Corrected outcome read:

- Fixed-straight and old frozen opponents often started with losses, then moved
  toward wins. That is a real but low-bar signal.
- Recent and mid frozen opponents were often already wins at the first
  checkpoint, while survival stayed around the floor. That means outcome reward
  was already saturated and gave little useful pressure.
- The next serious batch should use survival plus bonus reward. Outcome reward
  should be off/zero for the diagnostic lane and remain an eval metric only.

## Trusted Training Lane

Use stock LightZero `train_muzero` through:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train
```

Avoid the old custom `--mode two-seat-selfplay` lane for learning claims.

## Current Phase

The ugly `survivaldiag-v1b` 50-row batch was stopped and preserved. The first
clean 300-row grouped-app launch failed because trainer calls crashed before
startup from incomplete grouped `train_kwargs`; those broken 300 run roots were
deleted from the Modal volume.

The rescue launch is now `curvy-survive-bonus-large-20260513b`. It was
submitted through one deployed trainer app. The app currently has hundreds of
active tasks, which is the intended shape: one app, many train/poller function
calls.

Fresh all-row read, 2026-05-13 09:30 EDT: all 300
`curvy-survive-bonus-large-20260513b` rows are running and have trainer
heartbeats, readable `progress_latest.json`, checkpoints, eval manifests, and
GIF artifacts. Latest checkpoint distribution is mostly `iteration_75000` to
`iteration_105000`; latest eval mean survival across the 300 rows averages
about 81 steps, with a best sampled latest eval mean of 143.375. Treat this
batch as healthy background training and as a usable frozen-checkpoint source.

Current work is:

- finish the pruned-run cleanup without changing the keep/kill policy. The
  train/poller calls for killed rows are canceled, but Modal volume deletion is
  currently rate-limited;
- monitor the surviving `curvy-survive-bonus-large-20260513b`,
  `curvy-mix2-clean-20260513a`, and `curvy-mix3-currentckpt-20260513a` rows
  after cleanup;
- use `save_ckpt_after_iter=10000` as the current mixture cadence. It gives
  roughly 20-25 minute first-checkpoint gaps on many healthy matched rows.

Run inventory is now tracked separately in
[run_inventory_2026-05-13.md](run_inventory_2026-05-13.md). Use that file for
the current factual catalog of launched batches, stale canaries, preserved
artifacts, launch times, and volume-root counts.

Current named surfaces:

| Surface | Status | What to watch next |
| --- | --- | --- |
| `curvy-survive-bonus-large-20260513b` | healthy 300-row background batch using `save_ckpt_after_iter=15000` | keep as source of frozen checkpoints; sample survival/eval trends, not liveness |
| `curvy-mix2-clean-20260513a` | healthy-ish 156-row mixture batch using `save_ckpt_after_iter=10000` | wait for more `k10`/`k20` eval coverage before ranking recipes |
| `curvy-mix3-currentckpt-20260513a` | newly launched 300-row mixture batch through the single trainer app | verify trainer roots, `iteration_0`, eval/GIF artifacts, and first `k10` checkpoints |

Current naming:

- mix2 clean rows use run prefix `curvy-mix2clean` and attempt prefix
  `try-mix2clean`;
- mix3 current-checkpoint rows use run prefix `curvy-mix3cur` and attempt
  prefix `try-mix3cur`;
- all large batches should submit into the single deployed app
  `curvyzero-lightzero-curvytron-visual-survival-train`.

Run inventory, 2026-05-13 10:25 EDT: the apparent "about 800 runs" count is
real if canaries and preserved old rows are included. Logical launched/tracked
rows are about 821: 300 current survival rows, 156 mix2 rows, 300 mix3 rows, 50
preserved v1b rows, and 15 old/canary mixture rows. The Modal volume currently
shows 713 run directories under `training/lightzero-curvytron-visual-survival`:
300 `curvy-survive-bonus`, 156 `curvy-mix2clean`, 192 `curvy-mix3cur`, 50
preserved `survivaldiag-v1b`, 6 `curvy-mix2b`, 6 old `curvy-mix2`, and 3 older
`curvy-mix` canary roots. The difference is mostly newly submitted mix3 rows
that have not yet created trainer-owned volume roots. Do not interpret 713 as
713 healthy training rows or 821 as 821 current learning claims.

Prune execution, 2026-05-13 afternoon EDT: use
[run_prune_decision_2026-05-13.md](run_prune_decision_2026-05-13.md) as the
cleanup source of truth. FunctionCall cancellation is complete: `1720` killed
train/poller calls canceled, `0` failures. Latest read-only volume check after
small cleanup waves shows `664` run roots: all `212` intended survivors present
and `452` kill-list roots still visible. The remaining artifact cleanup is
mechanical and blocked on Modal volume rate limits / layer settling.
The GIF website marker cleanup has succeeded: all `503` kill-list roots now
have `show_in_gif_browser.flag` absent, while all `212` survivor roots still
have it present. A representative 33-row survivor health sample after marker
cleanup had 33/33 rows running, with trainer heartbeats, train roots, eval
manifests, and GIF artifacts. A single all-212 status read timed out at 300s;
use chunked reads for a full sweep.

Important correction, 2026-05-13 evening EDT: several rows that looked stuck at
`iteration_0` were not truly missing checkpoints. After Modal restarts,
DI-engine `compile_config` can append a timestamp to `cfg.exp_name` when the
configured directory already exists, so LightZero writes checkpoints under
`train/lightzero_exp_YYMMDD_HHMMSS/ckpt` while CurvyZero status/poller/resume
code keeps scanning `train/lightzero_exp/ckpt`. The current investigation doc is
[stale_checkpoint_bug_investigation_2026-05-13.md](stale_checkpoint_bug_investigation_2026-05-13.md).
All six rows that the fixed-path health snapshot showed at `iteration_0` have
now been sampled and fit this timestamped-directory pattern. Do not treat
fixed-path `iteration_0` as proof that training failed until all
`lightzero_exp*` directories have been scanned.

Checkpoint tournament warning: the current Modal tournament discovery helper has
the right broad `train_root.glob("lightzero_exp*/ckpt")` scan when it discovers
from run roots. The danger is any caller or manifest that passes fixed
`train/lightzero_exp/ckpt/...` refs, or uses the trainer's stable mirror/status
output as the only source. Use
[checkpoint_tournament_checkpoint_discovery_handoff_2026-05-13.md](../checkpoint_tournament_checkpoint_discovery_handoff_2026-05-13.md)
for the short external-agent handoff.

Runtime-log split, 2026-05-13: see
[modal_runtime_failure_classes_2026-05-13.md](modal_runtime_failure_classes_2026-05-13.md).
Preemption and DI-engine env reset interruptions are real training-hot-path
interruptions. `DataLossError` and `PytorchStreamReader` failures are mainly
eval/GIF artifact and checkpoint-reader problems. No CUDA OOM or clear learner
update-loop crash was found in that read-only log pass.

Fast-render clarification: the trusted stock `--mode train` source-state path
does not use the old custom/two-seat name `fast_gray64_direct`. Its fast
training render is `source_state_trail_render_mode=body_circles_fast`. The
current preserved run set is not browser-only: it has `107` `body_circles_fast`
roots and `105` `browser_lines` roots. A spot check of remote Modal `run.json`
for a mix3 `rf` row confirmed `config.source_state_trail_render_mode` is
`body_circles_fast`; the paired `rb` row is `browser_lines`.

Opponent-mixture local status: episode-level weighted opponent mixtures are now
wired in the trusted stock train path. Background checkpoint eval/GIF carries
the mixture spec and records the selected component. The manifest builder
exists at `scripts/build_curvytron_opponent_mixture_manifest.py`.

Historical mixture launch artifacts:

- 3-row canary:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix-recent-canary-20260513a.json`
- old 100-row batch draft, now stale because cadence and base-grid shape changed:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix-recent-20260513a.json`

The old dense-run checkpoint refs were cleaned from the volume. The early mix2
mixture manifests used preserved v1b immutable refs:
`iteration_20000.pth.tar` as recent, `iteration_10000.pth.tar` as mid, and
`iteration_0.pth.tar` as old. These refs were checked on the Modal volume.
The current mix3 launched surface does not use those preserved refs; it uses
fresh `curvy-survive-bonus-large-20260513b` checkpoints listed below.

Local gates passed for the first mixture implementation: focused pytest, ruff,
py_compile, `git diff --check`, canary manifest dry-run, and grouped submitter
dry-run. The first three remote canaries were launched before the cadence
correction and failed before trainer startup because the deployed train
function rejected `opponent_mixture_spec`. Treat those canaries as invalid.
Before the corrected canary, redeploy the trainer app and verify the function
call no longer fails on the mixture argument. The next launchable mixture
manifest must use the corrected checkpoint cadence and should include a small
grid over baseline knobs, especially render fidelity.

Corrected `curvy-mix2` artifacts now exist:

- canary:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix2-canary-20260513a.json`
- full review manifest:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix2-20260513a.json`

The full review manifest has 228 rows and is not automatically approved for
launch. The cadence audit recommends `k10` and warns against broad sim16/C64/B64
sentinel rows unless the corrected canary gives a reason to expand beyond the
core sim8/C32/B32 shape.

Corrected canary launch: `curvy-mix2-canary-20260513a` was submitted after the
trainer redeploy. It has six rows and call IDs recorded in
`artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix2-canary-20260513a.grouped_submit_launch.json`.
It failed before `train_muzero`: the command metadata still said
`learner_vs_fixed_straight` while the generated env surface said
`learner_vs_weighted_episode_opponent_mixture`. That was a safety-check
mismatch, not a learning result.

Fix status: local code now computes the opponent-training relation from the
same helper for command metadata and env config, and the source-state readiness
summary accepts episode-level mixture runs. Focused pytest, ruff, py_compile,
and `git diff --check` passed after the patch. The trainer app was redeployed
and a fresh six-row canary was launched as `curvy-mix2-canary-20260513b`
(`curvy-mix2b-*` run ids). Early status shows pollers and train roots exist.
The generated LightZero config for a browser row contains
`opponent_training_relation='learner_vs_weighted_episode_opponent_mixture'`.
Current canary status: all six rows have reached `iteration_10000` with
`progress_latest.json`, raw GIFs, and `collect_t1` GIFs. The `iteration_10000`
GIF summaries are `ok=true` and record selected mixture component fields. The
sampled selections included `blank`, `mid`, and `recent`. The checkpoint gap was
about 31-38 minutes across the six canary rows, so
`save_ckpt_after_iter=10000` is close enough to the user's target range of
roughly one checkpoint every 15-30 minutes for this launch.

The first full mixture launch became the pruned 156-row clean manifest:
`artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix2-clean-20260513a.json`.
It dry-ran through the grouped submitter into one deployed trainer app. Shape:
7 main recipes x paired `body_circles_fast`/`browser_lines` x
`rep0`/`repM`/`repH` x 3 seeds, plus 5 small controls x the same paired base
profiles x 1 seed. It removes passive rows from the first full launch, keeps
sim8, 32 collectors, batch32, checkpoint every 10000 iterations, CurvyZero
eval/GIF on, and stock LightZero eval off. The older 180-row core manifest is a
ceiling/review artifact because it still includes passive dirty controls.

Launch status: `curvy-mix2-clean-20260513a` was submitted at
2026-05-13 07:49 EDT. The grouped submission wrote
`artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix2-clean-20260513a.grouped_submit_launch.json`
with 156 train call IDs and 156 poller call IDs, all targeting the one deployed
trainer app `curvyzero-lightzero-curvytron-visual-survival-train`.

First full status sweep after launch: 156 rows were visible in the status
reader, 56 had `iteration_0`, 83 had a trainer heartbeat, 128 pollers were
running, and 17 still had no train root. Split by render mode, 34/78
`body_circles_fast` rows and 22/78 `browser_lines` rows had `iteration_0`.
This is not yet evidence that browser rendering is the main slowdown. The
manifest launches fast rows before browser rows inside each recipe, and the
sweep happened during staggered startup. Treat this as liveness/startup
evidence until both render modes have reached the first real `k10` checkpoint.

Current render/cadence rule: keep fast and browser rows paired, but do not read
early `iteration_0` counts as speed evidence unless launch order is controlled.
For future manifests, interleave or randomize render order if startup timing is
part of the readout. For the next mixture wave, keep a grid over render fidelity
and a small number of other baseline knobs instead of assuming one baseline is
settled.

Later JSON status read: 25/156 rows had reached `iteration_10000`; 6 matched
fast/browser pairs had both reached `iteration_10000`. Among those six matched
pairs, median browser-minus-fast trainer elapsed time was about `-70` seconds,
meaning browser was not slower in that tiny paired sample. Do not overread the
sign, but it rules out the simple claim that browser rendering alone explains
the slow or missing checkpoints. The larger issue is still staggered startup
plus recipe/repeat variability until more matched pairs mature.

Next JSON status read: 49/156 rows had reached `iteration_10000`, and 15
matched fast/browser pairs were comparable. Median browser-minus-fast trainer
elapsed time was about `+11` seconds, mean about `-24` seconds. Current plain
read: render mode is not the main checkpoint blocker. The rows with no
checkpoints are still mostly late manifest-order recipes and controls, so keep
separating startup/order lag from training speed.

Eval/GIF artifact read: artifacts are appearing normally. In the latest JSON
status, 39 rows had eval manifests and 100 rows had at least one GIF artifact.
`background_poller_completed_count=0` is not an artifact-health failure while
pollers are still running; the poller only fills that completion count when it
exits and joins outstanding jobs. For live artifact health, use
`eval_manifest_count`, `eval_checkpoints`, `gif_artifact_count`, and
`latest_gif_checkpoint`.

Corrected cadence read: the first matched analyzer used current
`progress_latest.elapsed_sec`, which is not a stable time-to-checkpoint once a
row moves past `k10`. The status reader now exports checkpoint file mtimes, and
the analyzer uses `iteration_10000` mtime minus `iteration_0` mtime. On the
latest snapshot, 89/156 rows had reached `iteration_10000`, 38 matched
fast/browser pairs were comparable, and browser rows were about two minutes
slower on this real checkpoint-gap measure:

- fast median `k0 -> k10`: about `1285` sec;
- browser median `k0 -> k10`: about `1395` sec;
- median browser-minus-fast gap: about `117` sec.

Plain read: browser is a little slower, but not the main blocker. Both render
modes are usually producing `k10` in roughly the 20-25 minute band. The bigger
live issue was startup/order/capacity lag; by the latest snapshot 152/156 rows
had trainer status `running`.

Status tooling update: `lightzero_curvytron_run_status.py --output json` now
also exports latest eval fields, checkpoint artifact mtimes, and selected
opponent-mixture component fields from GIF summaries. This is the preferred
source for future monitoring snapshots.

Next-wave launch status: the old draft
`curvy-mix3-nextwave-20260513a` remains a dry-run artifact that used preserved
v1b checkpoint refs. It has been superseded by
`curvy-mix3-currentckpt-20260513a`, generated from the same 300-row next-wave
profile but with frozen opponent refs from the now-healthy
`curvy-survive-bonus-large-20260513b` run:

- recent: `iteration_105000` from
  `curvy-survive-bonus-blank-fast-light-base-r063-s1111121`;
- mid: `iteration_60000` from the same run;
- old: `iteration_0` from the same run.

`curvy-mix3-currentckpt-20260513a` has 300 rows: 180 main rows, 60 controls, and
60 compute probes. It keeps passive out, pairs `body_circles_fast` and
`browser_lines`, alternates fast/browser launch lead, and keeps recent
checkpoint exposure substantial in every main recipe. Dry-run, focused tests,
ruff, format check, and `git diff --check` passed before launch. It was
submitted through the single deployed trainer app at about 2026-05-13
09:31 EDT. Launch artifact:
`artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix3-currentckpt-20260513a.grouped_submit_launch.json`.

Fresh startup read, 2026-05-13 10:13 EDT: `curvy-mix3-currentckpt-20260513a`
is submitted and moving, but not fully mature. Status reader counts: 300 rows,
187 train roots, 180 running pollers, 36 rows with live trainer heartbeat, 35
rows with `progress_latest.json`, 32 rows at `iteration_0`, 3 rows at
`iteration_10000`, 8 rows with eval manifests, and 26 rows with GIF artifacts.
This is a startup/liveness read only, not a learning read and not proof that all
300 rows are actively training yet.

Follow-up read, 2026-05-13 10:22 EDT: mix3 is still moving. Counts are now 190
train roots, 186 running pollers, 37 live trainer heartbeats, 36 progress rows,
33 rows at `iteration_10000`, 1 row at `iteration_20000`, 28 rows with eval
manifests, and 34 rows with GIF artifacts. This confirms forward motion after
the first startup read.

Artifact commit caveat, 2026-05-13 10:10 EDT: the large live batches created a
Modal volume commit storm in checkpoint eval/GIF workers. Old already-spawned
workers can still fail at direct `runs_volume.commit()` calls with
`DataLossError`. Current code now reduces checkpoint eval to one final commit,
uses `commit=False` for the inner seed evals, and wraps trainer/eval/GIF volume
commits in retry/backoff with jitter. The trainer and eval apps were redeployed
around 10:02-10:03 EDT. New logs show retry events with labels like
`checkpoint_selfplay_gif` and `checkpoint_eval_and_inspect`; old-line failures
can continue until pre-redeploy workers drain. Existing long-running pollers
also keep their launch-time code snapshot; a future relaunch or current-code
sidecar/backfill would be needed if old pollers continue to miss artifacts.

Learning-read caveat: current `curvy-mix2-clean` status has many `k10`
checkpoints and `k10` GIF artifacts, but the eval checkpoints visible in the
status snapshot are still `iteration_0` only. Do not rank mixtures by learning
until `iteration_10000` eval manifests appear or another trusted metric is
extracted from trainer/eval artifacts.

Updated learning read: `k10` and `k20` eval manifests are now appearing. Latest
fresh `curvy-mix2-clean` status has 155/156 running rows, 105 rows at
`iteration_10000` or later, 64 rows with eval manifests, and 149 rows with GIF
artifacts. Early survival signal exists: sampled `r50-mid50`, `r50-scr50`,
`r50-blank25-scr25`, and `r50-mid25-old25` rows are above their
`iteration_0` floor by `k10`/`k20`. Do not rank recipes yet, because the eval
coverage is still incomplete, but this is real progress evidence.

Website read, 2026-05-13 09:30 EDT: the GIF browser is deployed at
`https://modal-labs-shankha-dev--curvyzero-curvytron-gif-browser--bada8e.modal.run/`.
An exact API check for a `curvy-mix2-clean` row returned both `raw.gif` and
`collect_t1.gif` at `iteration_10000`. `speed unknown` means the selected row
does not yet have a readable `attempts/*/train/progress_latest.json`; it is not
a GIF failure.

Website update, 2026-05-13 10:17 EDT: the GIF browser was redeployed. A normal
exact API check for
`curvy-mix3cur-r25-blank75-rf-s8-c32-l32-rep0-k10-c1-s2301011` returns both
`raw.gif` and `collect_t1.gif` for `iteration_0` and `iteration_10000`. A
forced `fresh=1` reload can still be slow under the current volume load and
returned one 500 during the check; the non-forced endpoint and run list show
mix3 artifacts.

Scripted-component plumbing check: a raw `r50-scr50` `iteration_10000` GIF
summary was fetched from the Modal volume. It had `ok=true`,
`opponent_mixture_entry_name=scripted`,
`opponent_mixture_age_label=scripted_wall_avoidant`, 68 physical steps, and no
greedy action-collapse warning. This is not a learning claim, but it confirms
the scripted mixture component can be selected and rendered in the checkpoint
GIF path.

New blocker found: current `opponent_death_mode=immortal` only suppresses
player 1 death. It does not keep the opponent inside the arena or remove its
trail. Treat it as a canary or raw ingredient, not as the main opponent design.

Reward/canary gate status: `survival_plus_bonus_no_outcome` is implemented,
locally tested, Modal dry-gated, and tiny e2e-gated. It gives survival plus
same-step bonus pickup and excludes outcome from trainer reward. The first real
canary exposed that LightZero v0.2.0 uses shared `model.support_scale`; after
the fix, reward and value heads are both effective `601` bins from
`support_scale=300`. Tiny stock `train_muzero` canaries now pass for
`blank_canvas_noop/body_circles_fast`, `blank_canvas_noop/browser_lines`, and
`normal/body_circles_fast`.

Control stochasticity status: the stock trainer path now exposes and records
the source-state action-repeat knobs:
`policy_action_repeat_min`, `policy_action_repeat_max`, and
`policy_action_repeat_extra_probability`. This is held-action repeat inside one
LightZero env step, not a separate no-op transition. Local tests cover the env
accounting, and a tiny Modal stock `train_muzero` repeat smoke completed with
`policy_action_repeat_min=1`, `policy_action_repeat_max=3`,
`policy_action_repeat_extra_probability=0.25`, blank canvas, and 46 telemetry
rows.

Status/export status: the local status rollup now preserves and derives richer
survivaldiag fields into `eval_checkpoints` when checkpoint eval manifests have
them: reward totals/components, bonus counts, terminal-cause histograms, action
histograms/entropy, failure rate, and eval health. This has local test coverage.
The live reward/status gate is now cleared for the first-wave families:
strict-stop canaries write completed heartbeat/status, completed poller state,
eval/GIF artifacts, `eval_reward_variant=survival_plus_bonus_no_outcome`,
positive survival reward instead of sparse `-1.0` outcome reward, and explicit
zero-valued bonus fields. Reward components mean trainer reward terms only:
`survival` plus `bonus`. Sparse outcome is separate telemetry.

Latest live canaries:

| Family | Fast | Browser | Launch read |
| --- | --- | --- | --- |
| Blank canvas no-op | passed | passed | main anchor cleared |
| Passive immortal dirty control | passed | passed | plumbing cleared, still conceptually dirty |
| Blank canvas sim16 sentinel | passed | passed | sim16 sentinel cleared |

Live canary stop semantics: for very short train-mode canaries, do not rely on
`max_train_iter` as an exact checkpoint count. LightZero checks that cap after a
collect/update block, and one block can run many learner updates. Use
`--stop-after-learner-train-calls` for strict small canaries. A forced Modal
stop can leave heartbeat/poller JSON saying `running`; check Modal app state
directly when deciding whether anything is still alive.

Future successful runs now write a final `status_heartbeat.json` with
`status=completed` or `failed`; the stale-heartbeat warning mainly applies to
older runs and manually stopped runs.

Manifest safety status: the old
`scripts/build_curvytron_stock_train_manifest.py` generator is historical-only
and now refuses to emit by default. It can only be used with
`--allow-historical-matrix` to inspect old May 12 controls.

Survivaldiag manifest status: `scripts/build_curvytron_survivaldiag_manifest.py`
defaults to `large_ready`, the 300-row shape used by the current 300b launch. It
uses stock `--mode train`, `survival_plus_bonus_no_outcome`, high cap `65536`,
separated seed/copy fields, matched render pairs, reward/opponent/stochasticity
contract fields, and background eval/GIF enabled. Shape:

| Block | Rows |
| --- | ---: |
| Blank canvas all-level repeats | 160 |
| Blank canvas medium/high extra repeats | 40 |
| Passive immortal fixed-straight dirty controls | 40 |
| Blank canvas compute sentinels: search16, collect64, batch64 | 60 |

The generated run names are human-readable, for example
`curvy-survive-bonus-blank-fast-steady-base-r001-s1110011`. The manifest also
stores the exact `train_kwargs` and `poller_kwargs` needed by the grouped
submitter. It still sets `current_launch_approved=false` because generation is
a review artifact; actual launch goes through
`scripts/submit_curvytron_survivaldiag_manifest.py --allow-launch` after the
manual hold. Current default caps are overnight-sized:
`max_train_iter=300000`, `max_env_step=30000000`,
`save_ckpt_after_iter=15000` for the already-running 300-row batch, and
`background_eval_poller_max_runtime_sec=64800`.
The stock train Modal functions have a 16-hour timeout; the checkpoint poller
function has a 20-hour timeout so it can keep watching after a 12-hour-plus
training run.
The launched batch uses short attempt IDs with the `sdv1bh-a` prefix because
run-management IDs must be at most 96 characters.

Current live batch: `curvy-survive-bonus-large-20260513b` is running in the
single deployed trainer app. Submission record:
`artifacts/local/curvytron_survivaldiag_manifests/curvy-survive-bonus-large-20260513b.grouped_submit_launch.json`.
It has 300 train function call IDs and 300 poller function call IDs. Early row
status proves the missing-kwargs launch bug is fixed.

Speed display bug: the web UI shows `speed unknown` when
`attempts/<attempt>/train/progress_latest.json` is absent. Future trainer code
now writes this file on each checkpoint save with `iteration` and `elapsed_sec`.
It also refreshes the file from the LightZero `SaveCkptHook` path and writes
`event="checkpoint"`, so later checkpoints should no longer leave
`progress_latest.json` stuck at iteration 0 after redeploy. Local plumbing test
passed: `45 passed, 1 skipped`.

Launch hygiene note: an attempted dispatch from the old
`survivaldiag-v1-blank-core-*` review commands partially spawned a few detached
apps before the attempt-id length bug was fixed. Those apps were stopped. Do not
use those partial rows for learning claims. The later ugly 50-row `v1b` batch
was also stopped and preserved. The broken `20260513a` 300-row roots were
deleted. The current live survivaldiag batch is
`curvy-survive-bonus-large-20260513b`.

Modal dashboard cleanliness: large launches must use one deployed app plus many
function calls. The grouped submitter mirrors the local entrypoint by spawning
the checkpoint poller first and the train function second. Do not go back to
one `modal run` per row. See
[modal_grouped_app_launcher_followup_2026-05-13.md](modal_grouped_app_launcher_followup_2026-05-13.md).

Manifest/launcher audit status: the default generated commands use accepted
compute `gpu-l4-t4-cpu40`, all emitted flags map to the launcher CLI surface,
background eval/GIF flags are real, and no command uses `two-seat-selfplay`.
Bad manual `--compute` overrides are still not prevalidated by the generator,
so keep review commands generated from known defaults unless adding explicit
validation.

## Next Training Concept

Add diagnostic opponent settings so the learner cannot get easy reward from a
weak opponent dying:

- `opponent_death_mode=immortal`: ego can die; opponent cannot. Current behavior
  is passive death immunity: the opponent can keep moving out of bounds and
  still leaves normal trail/body points.
- `opponent_trail_mode=none`: possible later/canary lane where the opponent
  leaves no collision trail. Do not bundle this with immortal death until the
  separate effects are understood.
- repeated copies: important stochastic/random-opponent rows should run several
  times with different seeds so one lucky seed does not drive the conclusion.
- seed meanings must stay separate: training seed, reset seed, opponent policy
  seed, opponent behavior seed, and eval seed are different axes, not one vague
  `seed`.
- scripted wall-avoidant opponent: possible stronger baseline that turns away
  from walls and creates useful trails.
- random learned frozen opponent: not a first-class source-state opponent yet.
  Prefer immutable random-init or `iteration_0` LightZero checkpoints through
  the existing frozen-checkpoint path, with explicit opponent policy seeds.

Audit result: current `opponent_death_mode=immortal` is not a clean main lane
by itself. It suppresses player 1 death; it does not reflect, stop, remove
trail, or make the opponent avoid walls.

Those gates are now clear for the running first-wave blank-canvas-heavy batch.
For later opponent families, repeat the same rule: no scripted, random, or
checkpoint-opponent rows enter a serious matrix until their exact lane is wired,
manifest-addressable, and canaried.

## Opponent Feature Status

| Feature | Current state | Launch stance |
| --- | --- | --- |
| Blank canvas no-op | Implemented and locally/e2e canaried. Player 1 keeps the two-player shape but is inert, hidden, no-trail, no-collision, and no-bonus. | Main anchor for first survivaldiag matrix. |
| Passive immortal | Implemented as player-1 death suppression. It does not reflect, avoid walls, stop out-of-bounds motion, or remove trail. | Tiny dirty control only. Do not treat as clean opponent design. |
| Opponent no-trail mode | Design exists as a concept, but broad moving no-trail opponent is not implemented as a first-class trainer lane. | Keep out unless separately wired and canaried. Blank canvas already covers the clean no-trail/absent case. |
| Scripted wall-avoidant | Probe/design exists. Best first policy is `proactive_force_field` with safe margin `20`; variants include `lazy_weave`, `jitter_force_field`, and `wall_follower`. These use legal left/straight/right actions and do not bounce or teleport. | Second-wave lane only after stock-trainer wiring and e2e canary. |
| Random learned/frozen | Frozen checkpoint path exists; random-init/iteration-0 checkpoint generation and immutable opponent seed fields are not launch-ready. | Controls only after immutable refs/seeds are in the manifest. |

Do not let the manifest work erase the feature backlog. The first launch can be
blank-canvas-heavy, but the next useful feature lane is probably scripted
wall-avoidant opponent wiring, starting with `proactive_force_field`.
That work is not first-wave launch-blocking: the current policy exists as a
probe/design only and needs first-class source-state trainer wiring, metadata,
tests, and a tiny e2e canary before it can join a matrix.

Feature audit status: the clean first-wave core is implemented and live-tested as
`blank_canvas_noop` + explicit `survival_plus_bonus_no_outcome` + held-action
repeat + eval/GIF/status plumbing. Do not rely on the launcher default
`reward_variant=auto` for this lane; current review commands set the reward
explicitly, and live checkpoint eval defaults to that same training reward
unless explicitly overridden. Moving no-trail, scripted wall-avoidant,
random-init, and ancestor checkpoint families remain separate gated feature
work.

## Next Immediate Lane: Opponent Mixture Batch

The next batch should keep opponent mixture as the main question, but it should
not use one fixed base anymore. Include a compact base grid: paired
`body_circles_fast` and `browser_lines` rows for the core recipes, plus a few
named search/stochasticity probes. The main recipes should use a recent frozen
checkpoint for about 50% of episodes, with the remaining 50% split across older
checkpoints, scripted hand-designed opponents, passive immortal dirty
trail-makers, and blank canvas.

Do not launch this until first-class episode-level mixture support exists in
the trusted stock path and passes local tests plus tiny remote canaries. Full
plan: [opponent_mixture_batch_plan_2026-05-13.md](opponent_mixture_batch_plan_2026-05-13.md).

## Matrix Priorities

| Axis | Current stance |
| --- | --- |
| Reward | Survival plus bonus pickup. Outcome reward off/zero; not a serious next-lane candidate. |
| Opponent | Main clean lane should start with blank canvas/no-op once implemented. Passive immortal is a dirty control. Scripted wall-avoidant is the stronger trail-maker lane once validated. |
| Repeated copies | Important stochastic/random-opponent rows should have repeated seeds, about five copies when the row matters. |
| Episode cap | Do not sweep. Set high, e.g. `65536`. If runs get long, that is the signal. |
| Render | For serious settings, run both `body_circles_fast` and `browser_lines` as matched rows. |
| Stochasticity | Sweep meaningfully, including more aggressive levels than `0.05`. |
| Search / collectors / learner batch | Do not over-sweep until projected from v1d. Use a small focused set. |

## Future Tensor Rule

The next tensor should be assembled as staged blocks, not a single crossed
product and not the old 48-row sketch.

Block order should roughly be: mechanics canaries, blank-canvas wall-avoidance,
fixed immortal trail-maker, random-policy immortal opponent repeat groups,
ancestor-checkpoint controls, stochasticity ladder, tiny reward ablation, and
projection sentinels.

Use about five repeated copies for important random/stochastic opponent rows.
Use fewer copies for ancestor checkpoint controls unless they become the main
claim. Keep render twins matched by logical seed/copy so render differences can
be read cleanly.

Aggressive scale is allowed and expected after gates pass. See
[aggressive_matrix_scale_plan.md](aggressive_matrix_scale_plan.md). Rough
targets: about 50 runs to prove the clean survival lane is alive, about 100
runs to compare opponent families and seed stability, 200+ runs for variance
and confirmation, and about 300 runs only if the extra rows are repeats or
clearly different opponent families.

Do not launch from any historical dry-run generator or old launch shell script.
The next launch surface must be generated from a current survivaldiag manifest
that records reward, opponent, render pairing, stochasticity, and separate
seed/copy fields. The current dry-run generator does this locally, but the
commands are still not launch-approved.

## Old-Run Projection Rule

For the v1d runs, project knobs using the corrected outcome/score curve:
first, best, and latest `win/loss/draw`. Do not use those old survival curves
to choose the next reward objective, because those runs did not train on the
survival-first objective we now care about.

For the next runs, the primary curve should be survival/reward. Outcome remains
a secondary eval metric, not a training reward.

## Active Docs

- [todo.md](todo.md)
- [user_priority_snapshot.md](user_priority_snapshot.md)
- [hypotheses_and_evidence.md](hypotheses_and_evidence.md)
- [v1d_axis_projection.md](v1d_axis_projection.md)
- [v1d_fresh_eval_summary_2026-05-13.md](v1d_fresh_eval_summary_2026-05-13.md)
- [next_overnight_matrix_plan.md](next_overnight_matrix_plan.md)
- [next_matrix_manifest_design.md](next_matrix_manifest_design.md)
- [aggressive_matrix_scale_plan.md](aggressive_matrix_scale_plan.md)
- [opponent_diagnostic_design.md](opponent_diagnostic_design.md)
- [blank_canvas_noop_opponent_lane.md](blank_canvas_noop_opponent_lane.md)
- [scripted_wall_avoidant_opponent_baseline_2026-05-13.md](scripted_wall_avoidant_opponent_baseline_2026-05-13.md)
- [source_state_reward_wiring_audit_2026-05-13.md](source_state_reward_wiring_audit_2026-05-13.md)
- [prelaunch_validation_round3_2026-05-13.md](prelaunch_validation_round3_2026-05-13.md)
- [live_highcap_canary_plan_2026-05-13.md](live_highcap_canary_plan_2026-05-13.md)
- [delegation_log.md](delegation_log.md)
- [eval_curve_tooling_plan.md](eval_curve_tooling_plan.md)
- [operating_patterns.md](operating_patterns.md)
- [archive_and_stale_docs.md](archive_and_stale_docs.md)
- [instruction_digest_2026-05-13.md](instruction_digest_2026-05-13.md)

## Do Not Forget

- We need tooling that turns many run artifacts into comparable eval curves.
- Curves should support different metrics: outcome score now, survival and
  reward later, and possibly weighted combinations later.
- Analysis should be cautious about false negatives: weak or late-learning runs
  should not be thrown away too early.
- The main thread should plan, delegate, and synthesize. Subagents can implement
  tools, inspect code, analyze axes, and clean docs.
- Future tensors should be staged blocks, not one giant crossed product.
