# DI-engine `compile_config` Timestamped Exp Dir Evidence - 2026-05-13

Scope: read-only verification of the hypothesis that DI-engine `compile_config`
renews `cfg.exp_name` by appending a timestamp when the requested experiment
directory already exists, causing checkpoints to land under sibling
`lightzero_exp_YYMMDD_HHMMSS` directories while CurvyZero progress/poller
artifacts continue to scan the original `lightzero_exp`.

No source files were edited. Modal checks used `modal volume ls` plus
read-only `modal volume get`; local evidence came from
`artifacts/local/curvytron_pruning/status_chunks_20260513e/combined_status.json`
and `status_chunks_20260513f/raw_status_files`.

## Code-Path Anchor

Upstream DI-engine `compile_config` has the relevant behavior: if `save_cfg` is
true, rank is 0, the configured `cfg.exp_name` path already exists, and
`renew_dir` is true, it appends `datetime.datetime.now().strftime("_%y%m%d_%H%M%S")`
to `cfg.exp_name` before making the directory and saving config. Source checked
read-only from upstream `ding/config/config.py`.

This exactly matches the Modal volume shape below: stale rows have one original
`lightzero_exp` plus one or more `lightzero_exp_260513_*` sibling directories.
The healthy comparator has only `lightzero_exp`.

## Evidence Table

| Row | Status | Original `lightzero_exp/ckpt` highest | Timestamped exp dirs and highest checkpoint | `progress_latest` checkpoint / train iter | Poller `seen_count` | Readout |
| --- | --- | ---: | --- | --- | ---: | --- |
| `curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021` | stale | `iteration_0` | `lightzero_exp_260513_121430`: `iteration_110000`; `lightzero_exp_260513_172129`: `iteration_0`; `lightzero_exp_260513_175026`: `iteration_40000` | `iteration_0.pth.tar` / `43017` at `2026-05-13T20:12:16Z` | `0` | Fits strongly. The current resumed container appears to be writing into `_175026`, while progress/poller still report original `lightzero_exp`. |
| `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051` | stale | `iteration_0` | `lightzero_exp_260513_153203`: `iteration_40000`; `lightzero_exp_260513_172427`: `iteration_70000` | `iteration_0.pth.tar` / `78317` at `2026-05-13T20:11:51Z` | `0` | Fits strongly. Checkpoints exist beyond progress train iter cadence, but only in timestamped dirs. |
| `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011` | stale | `iteration_0` | `lightzero_exp_260513_121133`: `iteration_0`; `lightzero_exp_260513_123802`: `iteration_180000` | `iteration_0.pth.tar` / `185627` at `2026-05-13T20:06:59Z` | `1` | Fits, with caveat. The poller saw one checkpoint, likely original `iteration_0`, but did not see the timestamped `iteration_180000` stream. |
| `curvy-mix2clean-r50-blank50-rf-s8-c32-l32-rep0-k10-c2-s2101021` | healthy comparator | `iteration_210000` | none found under attempt `train` root | `iteration_210000.pth.tar` / `216570` at `2026-05-13T20:09:44Z` | `22` | Comparator fits the inverse pattern: no timestamped sibling, original dir advances, progress and poller see the stream. |

## Modal Volume Details

Parent `train` listings:

- Stale `scr50-rb` row contained `lightzero_exp`,
  `lightzero_exp_260513_121430`, `lightzero_exp_260513_172129`, and
  `lightzero_exp_260513_175026`.
- Stale `mix3cur-r40...` row contained `lightzero_exp`,
  `lightzero_exp_260513_153203`, and `lightzero_exp_260513_172427`.
- Stale `scr50-rf-repH` row contained `lightzero_exp`,
  `lightzero_exp_260513_121133`, and `lightzero_exp_260513_123802`.
- Healthy comparator contained only `lightzero_exp`.

Checkpoint listings:

- In all three stale rows, original `lightzero_exp/ckpt` had only
  `iteration_0.pth.tar` plus `ckpt_best.pth.tar`.
- In all three stale rows, timestamped sibling directories had later
  `iteration_*.pth.tar` sequences.
- In the healthy comparator, the original `lightzero_exp/ckpt` had the normal
  sequence from `iteration_0` through `iteration_210000`.

Local artifact cross-check:

- `status_chunks_20260513e/combined_status.json` had the same stale shape:
  the three stale sampled rows reported `latest_checkpoint=iteration_0`, while
  the healthy comparator reported `latest_checkpoint=iteration_190000` at that
  earlier snapshot.
- `status_chunks_20260513f/raw_status_files/*/progress_latest.json` already
  showed stale rows with high `learner_train_iter` but
  `checkpoint_name=iteration_0.pth.tar`.
- `status_chunks_20260513f/raw_status_files/*/checkpoint_eval_poller.json`
  showed `seen_count=0`, `0`, and `1` for the stale rows, consistent with the
  poller not scanning timestamped sibling exp dirs.

## Critical Notes

This is stronger than the earlier "alive but not checkpointing" hypothesis.
The sampled stale rows are checkpointing; they are checkpointing in directories
that the CurvyZero progress writer, status table, and poller do not treat as
the source exp dir.

The evidence does not prove every stale row in the matrix has this cause. It
does prove that the sampled high-iteration stale rows above were misclassified
by artifact discovery.

The `scr50-rf-repH` row is the main caveat: poller `seen_count=1`, not `0`.
That does not contradict the path-mismatch hypothesis because the original
directory contains `iteration_0`; it does mean this row is not a pure "poller
never saw anything" case.

The `scr50-rb` row has multiple timestamped dirs. The older `_121430` dir
reached `iteration_110000`, then later resumed attempts created `_172129` and
`_175026`. This suggests repeated restarts/preemptions can fragment one logical
run across several DI-engine-renewed exp dirs.

## Verdict

Smoking gun confirmed for the sampled stale rows. The original
`lightzero_exp` directories remained stuck at `iteration_0`, while timestamped
`lightzero_exp_260513_*` siblings contained the real checkpoint streams. The
healthy comparator lacked timestamped siblings and its original `lightzero_exp`
advanced normally.

Policy-tournament warning: any agent selecting checkpoints for tournaments,
ratings, or comparator pools must scan every `train/lightzero_exp*/ckpt`
directory under an attempt, not only `train/lightzero_exp/ckpt`. Treat
`lightzero_exp_YYMMDD_HHMMSS` as first-class DI-engine output for resumed
runs, merge candidates across those dirs by checkpoint iteration and mtime, and
record the exact checkpoint ref chosen. A narrow `train/lightzero_exp/ckpt`
scan will silently select stale `iteration_0` policies for the sampled rows
above even though later trained policies exist on the volume.
