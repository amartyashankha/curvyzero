# Checkpoint Discovery Audit - 2026-05-13

Scope: read-only audit after the stale `train/lightzero_exp/ckpt` bug. No
source code was changed. No jobs were killed.

## Short Read

The fixed-path checkpoint status is undercounting real training progress.

Targeted discovery for the six preserved rows that fixed-path status showed at
`iteration_0` is complete: all six have later checkpoints under timestamped
`train/lightzero_exp_260513_*` directories. Main-thread follow-up also verified
that `curvytron_checkpoint_tournament --mode discover` resolves those six via
broad `lightzero_exp*/ckpt` discovery.

A partial broader scan of the 212 preserved rows found more of the same issue:
at least `54` preserved rows have timestamped `lightzero_exp*` directories, and
at least `50` rows have a higher checkpoint under broad discovery than fixed
status reports. Importantly, at least `45` of those are nonzero rows, not just
the known `iteration_0` cases.

This broad count is a lower bound. The all-212 CLI scan hit Modal
`VolumeListFiles` rate limits on `47` train-directory listings, so a final
health table still needs a slower chunked scan.

## What Was Sampled

Source of preserved run IDs:

```text
artifacts/local/curvytron_pruning/curvytron_prune_preserve_20260513c.json
```

Source of fixed-path status snapshot:

```text
artifacts/local/curvytron_pruning/status_chunks_20260513e/combined_status.json
```

Rows in preserved set: `212`.

## Targeted Fixed-Zero Rows

These are the six rows that fixed-path status showed at `iteration_0`. All six
resolve to later checkpoints when scanning all `lightzero_exp*/ckpt` dirs.

| Run | Fixed-path status | Broad discovery |
| --- | ---: | ---: |
| `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011` | `0` | `180000` |
| `curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021` | `0` | at least `110000` |
| `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051` | `0` | `80000` |
| `curvy-mix3cur-r50-blank50-rb-s16-c32-l32-repH-k10-c1-s2302011` | `0` | `100000` |
| `curvy-mix3cur-r75-blank25-rf-s16-c32-l32-repM-k10-c1-s2303011` | `0` | `110000` |
| `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s16-c32-l32-repM-k10-c1-s2306011` | `0` | `50000` |

Plain conclusion: none of the six known fixed-zero rows still looks truly stale
after broad checkpoint discovery.

## Partial 212-Row Broad Scan

The broader scan used fixed status from `combined_status.json`, then listed each
preserved row's train directory and any `lightzero_exp*/ckpt` directories.

Results from the partial scan:

| Quantity | Count | Caveat |
| --- | ---: | --- |
| preserved rows considered | `212` | full preserve manifest |
| train directory listing failures | `47` | Modal CLI rate limits/timeouts; not row failures |
| rows with timestamped `lightzero_exp*` dirs | `54` | lower bound |
| rows where broad highest > fixed status highest | `50` | lower bound |
| fixed-zero rows where broad highest > fixed status highest | `5` | sixth fixed-zero row was also verified separately |
| fixed-nonzero rows where broad highest > fixed status highest | `45` | key new finding |

The apparent `broad_still_zero_count=1` from the partial script is not a real
stale finding. It was the known row
`curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021`, and a direct
follow-up train-directory listing showed timestamped dirs:

```text
lightzero_exp_260513_121430
lightzero_exp_260513_172129
lightzero_exp_260513_175026
```

So there is no confirmed preserved row that remains truly stale after broad
discovery in the evidence collected here.

## Representative Nonzero Undercount Rows

These rows already had nonzero fixed-path checkpoints, but broad discovery found
much later checkpoints elsewhere.

| Run | Fixed-path highest | Broad highest | Timestamped dirs observed |
| --- | ---: | ---: | --- |
| `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-rep0-k10-c3-s2104031` | `20000` | `220000` | `lightzero_exp_260513_125346` |
| `curvy-mix2clean-r50-old50-rf-s8-c32-l32-rep0-k10-c3-s2103031` | `30000` | `200000` | `lightzero_exp_260513_132344`, `lightzero_exp_260513_201021` |
| `curvy-mix2clean-r50-blank25-scr25-rf-s8-c32-l32-rep0-k10-c3-s2106031` | `20000` | `160000` | `lightzero_exp_260513_125339`, `lightzero_exp_260513_150704` |
| `curvy-survive-bonus-blank-browser-medium-base-r180-s1120301` | `135000` | `240000` | `lightzero_exp_260513_150334`, `lightzero_exp_260513_162238` |
| `curvy-mix3cur-r50-scr50-rb-s8-c64-l32-repH-k10-c1-s2304011` | `20000` | `130000` | `lightzero_exp_260513_160242` |
| `curvy-mix3cur-blank100-rb-s8-c32-l32-repH-k10-c2-s2308021` | `30000` | `130000` | `lightzero_exp_260513_164102` |

Plain conclusion: the bug is not limited to `iteration_0`. Any fixed-path
status table can undercount a restarted row, even if it has some nonzero
checkpoints under `train/lightzero_exp/ckpt`.

## Partial Broad Health Table

This is not final because `47` listings failed under Modal rate limits. It is
still useful as a rough lower-bound read.

| Broad highest checkpoint bucket | Rows |
| --- | ---: |
| `>=200k` | `30` |
| `100k-199k` | `122` |
| `50k-99k` | `52` |
| `<50k` | `4` |
| `0` | `1` false/incomplete case, resolved by targeted follow-up |
| missing | `3` incomplete due listing failures |

The preserved set needs a new health table, but it should be generated with a
rate-limited broad scan rather than a single aggressive Modal listing burst.

## Tournament Footgun

The current tournament helper already has the right discovery shape when it is
given a run root:

```text
src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py
train_root.glob("lightzero_exp*/ckpt")
```

The risk is not that helper's broad glob. The risk is any caller, manifest, or
status-derived input that hands the tournament a fixed checkpoint ref such as:

```text
train/lightzero_exp/ckpt/iteration_*.pth.tar
```

Those fixed refs can be stale after DI-engine appends a timestamp to
`cfg.exp_name`. Tournament inputs should be built from broad discovery over run
roots, or they should explicitly include timestamped checkpoint refs.

## Commands Used

Representative read-only commands:

```bash
modal volume ls --json curvyzero-runs \
  training/lightzero-curvytron-visual-survival/<run>/attempts/<attempt>/train

modal volume ls --json curvyzero-runs \
  training/lightzero-curvytron-visual-survival/<run>/attempts/<attempt>/train/lightzero_exp/ckpt

modal volume ls --json curvyzero-runs \
  training/lightzero-curvytron-visual-survival/<run>/attempts/<attempt>/train/lightzero_exp_260513_*/ckpt
```

Local files read:

```bash
python - <<'PY'
import json
from pathlib import Path
preserve = json.loads(Path(
  "artifacts/local/curvytron_pruning/curvytron_prune_preserve_20260513c.json"
).read_text())
status = json.loads(Path(
  "artifacts/local/curvytron_pruning/status_chunks_20260513e/combined_status.json"
).read_text())
print(len(preserve["rows"]), len(status["rows"]))
PY
```

The broad 212 scan also used the same `modal volume ls --json` calls in a local
Python loop over preserved run IDs. That loop was stopped after it produced the
partial counts above because Modal began returning `VolumeListFiles rate limit
exceeded`.

## Next Read-Only Step

Run a slow chunked broad discovery pass for all 212 preserved rows:

- small chunks, for example 10-20 runs at a time;
- sleep between chunks;
- for each row, record every `lightzero_exp*` dir and the highest
  `iteration_*.pth.tar` under every `ckpt` dir;
- compare that to the fixed status snapshot;
- write the final broad health table from that data only.

