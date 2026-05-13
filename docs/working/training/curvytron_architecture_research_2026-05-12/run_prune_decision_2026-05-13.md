# CurvyTron Run Prune Decision - 2026-05-13

Snapshot: 2026-05-13 late morning EDT.

Purpose: reduce the current CurvyTron run set aggressively while keeping the
surviving rows readable and dependency-safe.

## Decision

Use prune plan:

`artifacts/local/curvytron_pruning/curvytron_prune_plan_20260513c.json`

Keep `212` logical rows and kill `609`.

| Surface | Keep | Kill | Reason |
| --- | ---: | ---: | --- |
| `curvy-mix3-currentckpt-20260513a` | 126 | 174 | Main current mixture surface; keep one best row per unique recipe/base/render/probe config. |
| `curvy-mix2-clean-20260513a` | 52 | 104 | Mature comparison surface using older checkpoint refs; keep one best main row per config plus medium-repeat controls. |
| `curvy-survive-bonus-large-20260513b` | 33 | 267 | Compact survival/source subset; keep unique configs and the exact source checkpoint row used by mix3. |
| `survivaldiag-v1b-20260513h` | 1 | 49 | Keep only the exact checkpoint-source root used by mix2. |
| failed/corrected canaries | 0 | 15 | Superseded by full batches; local manifests preserve the plumbing history. |

This plan also cancels the duplicate failed `curvy-survive-bonus-large-20260513a`
function calls if they still exist. Its volume roots were already deleted.

## Dependency Roots

Never delete these unless their checkpoint files are first copied to a stable
archive and all manifests are updated:

| Root | Needed by |
| --- | --- |
| `curvy-survive-bonus-blank-fast-light-base-r063-s1111121` | `mix3` recent/mid/old checkpoint refs |
| `survivaldiag-v1b-20260513h-001-survbonusnoout-blanknoop-fast-armed-c00-s910001-l4t4c40` | `mix2` recent/mid/old checkpoint refs |

## What The Survivors Measure

| Question | Survivor coverage |
| --- | --- |
| Does blank-canvas survival training work? | compact `300b` survival subset |
| Do render modes differ? | major kept lanes stay paired across `body_circles_fast` and `browser_lines` |
| Does action repeat/noise help? | `rep0`, `repM`, `repH` in mix2/mix3; none/low/medium/high in compact 300b |
| Which opponent mix helps? | mix3 keeps all main recipes and controls |
| Do older checkpoint opponents differ from fresher checkpoint opponents? | mix2 compared with mix3 |
| Do compute knobs matter? | compact search16, collect64, batch64 probes |

Render coverage after prune:

| Surface | `body_circles_fast` | `browser_lines` |
| --- | ---: | ---: |
| compact survival/source subset | `17` | `16` |
| mix2 older-checkpoint mixture | `26` | `26` |
| mix3 current-checkpoint mixture | `63` | `63` |
| v1b dependency root | `1` | `0` |
| total | `107` | `105` |

For the trusted stock source-state path, `body_circles_fast` is the current fast
render lane. `fast_gray64_direct` was an older custom/two-seat lane name, not a
stock `--mode train` render setting.

## Metrics To Read After Prune

Use multiple metrics. Do not rank rows from one number.

| Metric | Meaning |
| --- | --- |
| mean survival steps | main learning signal |
| latest and best survival | distinguishes steady progress from noisy peaks |
| slope over checkpoints | checks whether a row is improving |
| terminal cause | shows wall/trail/other death pattern |
| bonus count/reward | checks exploration pressure |
| action entropy/collapse | catches one-action loops |
| render-paired differences | checks fast-vs-browser fidelity |
| checkpoint cadence and artifact health | tells whether the row is usable |

## Critique

The plan deliberately kills repeated seed copies. This loses variance estimates,
but that is acceptable for this cleanup because the current problem is too many
live jobs and too much artifact clutter. The survivor set keeps breadth across
important axes instead of repeated copies of the same semantic config.

The main risk is deleting a checkpoint-source root. The preserve manifest must
therefore include both dependency roots, and the cleanup dry-run action set must
be checked before destructive deletion.

## Execution Status

Snapshot: 2026-05-13 afternoon EDT.

The kill-list FunctionCalls were canceled successfully:

- cancel manifest:
  `artifacts/local/curvytron_pruning/curvytron_prune_cancel_20260513c.json`
- result:
  `artifacts/local/curvytron_pruning/curvytron_prune_cancel_result_20260513c.json`
- result count: `1720` canceled, `0` failures

The volume cleanup is only partly complete. The mounted cleanup command reported
`550` deleted roots, but a later read still saw those roots. A direct
`modal.Volume.remove_file(..., recursive=True)` pass deleted `47` roots before
Modal started returning `ResourceExhaustedError` / "too many layers in volume."

Latest read-only volume count after waiting:

| Count | Meaning |
| ---: | --- |
| `715` | direct run directories visible under the CurvyTron training task |
| `212` | intended survivor roots present |
| `0` | intended survivor roots missing |
| `503` | kill-list roots still visible |
| `0` | unexpected roots outside preserve/kill manifests |

Current cleanup state: the jobs have been canceled, but artifact deletion must
continue in slow waves after Modal volume rate limits settle. Do not rebuild the
prune plan from this partial artifact state; use the preserve/kill manifests
above as the source of truth.

After an additional cooldown, a single direct delete retry still failed with
`ResourceExhaustedError: too many layers in volume. Please wait and retry.`
Treat further artifact cleanup as wait-and-retry work unless a different Modal
volume maintenance path is chosen.

After the fast-render audit, another read-only check still showed `212`
survivor roots, `503` kill-list roots, and `0` unexpected roots. A second
single-directory delete retry failed with the same `too many layers in volume`
error. Live-job pruning remains complete; stale artifact deletion remains
paused on Modal volume recovery.

Retry 1 after a 10-minute cooldown still showed `212` survivor roots, `503`
kill-list roots, and `0` unexpected roots. A single stale-root delete probe
still failed with `ResourceExhaustedError: too many layers in volume. Please
wait and retry.`

Website marker cleanup: `scripts/cleanup_curvytron_modal_runs.py` now supports
`--purge-unpreserved --markers-only`. The marker cleanup deleted
`show_in_gif_browser.flag` from `447` killed roots; `56` killed roots already
had no marker. Post-check: all `503` killed roots have
`has_picker_marker=false`, and all `212` survivor roots still have
`has_picker_marker=true`. This hides stale killed roots from the GIF website
while full directory deletion waits on Modal volume recovery.

Survivor health spot-check after marker cleanup: a 33-row sample across the
survival, mix2, mix3, and dependency-root buckets had 33/33 rows running, 33/33
trainer heartbeats, 33/33 train roots, and eval/GIF artifacts on every sampled
row. A single all-212 status read hit the status tool's 300s timeout, so future
full checks should split the preserve list into smaller chunks.

Small-wave directory cleanup: mounted one-root and 25-root waves persisted, but
a 50-root mounted wave reported `deleted` without changing the later listing.
Use small waves with a read-only recount after each wave, and stop when a wave
does not show up in the next listing. Current stable recount: `664` direct run
roots, `212` preserved roots, `452` stale roots still visible, and `0` missing
preserved roots. A direct `modal.Volume.remove_file(..., recursive=True)` probe
on the next stale root still returned `ResourceExhaustedError: too many layers
in volume. Please wait and retry.` A later mounted 25-root wave also reported
`deleted` but did not change the immediate recount, so pause directory deletion
until the volume settles again.
