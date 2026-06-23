# Leaderboard-To-Training Archive

Date: 2026-05-13. Current cleanup pass: 2026-05-16.

## Start Here

This directory is now archive/audit trail material.

For general feedback-loop architecture, policy-observation, and observability
contracts, start here:

`../curvytron_feedback_loop/README.md`

For the current r18fresh batch postmortem and evidence trail, start here:

`../r18fresh_postmortem_2026-05-16/README.md`

For current broad launch defaults, start here:

`../r18fresh_postmortem_2026-05-16/CURRENT_LAUNCH_DEFAULTS.md`

Use `CURRENT_TOURNAMENT_PIPELINE_2026-05-16.md` only as the current pipeline
map. Most other files here are historical notes and should not drive new work
unless the new postmortem workspace explicitly references them.

## Current Shape

```text
trainer writes checkpoint
-> scheduled intake subscriber discovers it
-> scheduled intake drain continues the rating run
-> tournament latest feeds the training-candidate refresh controller
-> controller writes immutable assignments and refresh pointers
-> running trainers refresh assignments at coarse train-iteration boundaries
```

The loop is mechanically proven for the `r18fresh` case-study lane through
tournament -> assignment -> trainer consumption at generation 9, generation 10,
and scheduled generation 12. Learning quality is still modest: the batch often
finds better mid-run checkpoints, then regresses at latest checkpoint.

## Archive Rule

Most other dated files in this directory are historical working notes. Use them
for audit trail, bug context, or old command examples only. Do not treat older
proof lanes, non-v2 names, pre-purge refs, or stale arena defaults as current
unless `CURRENT_TOURNAMENT_PIPELINE_2026-05-16.md` explicitly points there.

Useful historical references:

- `NOW.md`: rolling operator log; contains many superseded entries.
- `FULL_LOOP_PROOF.md`: detailed proof trail for canaries and live-loop checks.
- `automation_cleanup_audit_2026-05-16.md`: automation inventory behind the
  current summary.
- `r18fresh_learning_readout_2026-05-16.md`: current learning-quality readout.
- `post_purge_current_truth_2026-05-16.md`: post-purge v2 reset context.

## Current Source Of Truth

Runtime defaults live in `src/curvyzero/contracts/curvytron.py`. Current broad
launch defaults are summarized in
`../r18fresh_postmortem_2026-05-16/CURRENT_LAUNCH_DEFAULTS.md`. General
architecture contracts live in `../curvytron_feedback_loop/`. If this README or
any older note disagrees with those files and
`CURRENT_TOURNAMENT_PIPELINE_2026-05-16.md`, treat the older note as history.
