# LightZero Signal Reconciliation - 2026-05-10

Purpose: explain why one normal LightZero Atari Pong run, seed 1, no survival
reward shaping looked strongly positive while earlier summaries looked weak or
bad, and record the replication/eval launched from this reconciliation.

## Plain Answer

The positive result is probably real for that one run, but it is not yet a
stable Pong-learning claim.

In simple words: one seed 1 L4/T4 run found a better policy late in training.
That checkpoint survived to the 2048-step eval cap and scored positive under
one stock-return read. Other runs either learned less, peaked earlier, or fell
back by the final checkpoint. So the old bad-looking summaries were not
necessarily wrong; they were mostly looking at weaker seeds, earlier
checkpoints, shorter caps, or final checkpoints that had already regressed.

The right read is:

- real signal in normal LightZero Atari Pong run, seed 1, no survival reward
  shaping;
- not solved Pong;
- not a cross-seed result yet;
- not safe to summarize by only the best checkpoint from each run;
- not explained away by the eval setting drift, because the stricter rerun
  still shows a large gain.

## What Changed Across Runs

Seed changed. Seed 1 on L4/T4 produced the strongest late result. Seed 2 was
mostly weak. Seed 3 had a strong middle checkpoint and then a weaker latest
checkpoint. Seed 0 repeat A had a smaller late bump. Same code, different seed,
can produce a very different checkpoint curve.

Training compute changed. The seed 1 L4/T4 run and seed 1 H100 run are not the
same run. The H100 run peaked around `iteration_10000` and then fell back near
baseline by latest `iteration_17504`. That means faster/different hardware did
not simply make the final checkpoint better.

Training length changed. Repeat A used `32768` max env steps. The matrix runs
used `65536`. Both are still short compared with installed LightZero's stock
Atari Pong config surface at `200000` max env steps. The short runs can show
noisy partial learning without being expected to finish as strong Pong agents.

Checkpoint choice changed. This is the biggest explanation. A run can have a
good middle or late checkpoint and then fall back. Seed 1 L4/T4 is the best
case because both `iteration_16000` and latest `iteration_18459` were strong.
Seed 1 H100 peaked at `iteration_10000` and fell back by latest `iteration_17504`.
Seed 3 L4/T4 peaked at `iteration_16000` and fell back by latest
`iteration_17010`. Seed 2 only had weak bumps.

Eval cap changed. The 512-step eval cap hid survival differences because many
rows survived all 512 steps. The 2048-step cap separated weak from stronger
checkpoints better.

Eval seed and eval settings changed. The main compact evals used the matching
seed for each run. Earlier eval code defaulted to `update_per_collect=1` in the
eval helper even though training used the stock `None` setting. Reruns with
`--update-per-collect -1` make the eval surface closer to the stock training
surface. Those stricter reruns changed magnitudes, but did not erase the
signal.

Timeout changed during this reconciliation. The strong seed 1 L4/T4 stricter
rerun hit the old 480-second Modal eval timeout for `iteration_16000` and
`iteration_18459`. I changed only the eval function timeout in
`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py` from an inline
`8 * 60` to `EVAL_FUNCTION_TIMEOUT_SEC = 16 * 60`, then relaunched the missing
rows. This is a tooling change, not a training or reward change.

Stock telemetry changed. Newer artifacts record stock-path survival, reward
counts, action histogram, terminal/truncation status, and stock return. Older
artifacts often had stock return without stock-path survival details. For rows
where manual and stock actions differ, manual survival and stock return should
be read side by side, not merged into one story.

No survival reward shaping changed these normal runs. The shaped runs are a
separate ablation and are not part of this reconciliation.

## Evidence Table

| Run | Best or latest useful point | Plain result |
| --- | --- | --- |
| normal LightZero Atari Pong run, seed 1, no survival reward shaping, L4/T4, `65536` | latest `iteration_18459` | strongest result. Original compact eval: `2048/2048`, stock return `+13`. Stricter rerun: manual `2048/2048`, stock steps `2048`, stock return `-11`, stock positive rewards `8`. Still far above same-run baseline. |
| normal LightZero Atari Pong run, seed 1, no survival reward shaping, H100, `65536` | `iteration_10000` | real middle bump: `1731/2048`, stock return `-16`; latest `iteration_17504` fell back near baseline at `782/2048`, stock return `-21`. |
| normal LightZero Atari Pong run, seed 2, no survival reward shaping, L4/T4, `65536` | `iteration_10000` or latest `16829` | weak survival bump only: best compact `882/2048`, latest `840/2048`; stock return stayed `-21`. |
| normal LightZero Atari Pong run, seed 3, no survival reward shaping, L4/T4, `65536` | `iteration_16000` | strong middle bump, then partial fallback. Default eval: `1605/2048`, stock return `-18`; stricter rerun: `1172/2048`, stock return `-20`. Latest dropped from default `1236/2048` to stricter `847/2048`, but stock return improved to `-18`. |
| normal LightZero Atari Pong run, seed 0 repeat A, no survival reward shaping, L4/T4, `32768` | latest `iteration_9559` | smaller bump. Default eval: `973/2048`, stock return `-17`; stricter rerun: `909/2048`, stock return `-18`; still no positive manual rewards. |

## Stricter Seed 1 L4/T4 Rerun Launched

Question tested: is the strong normal LightZero Atari Pong run, seed 1, no
survival reward shaping just an eval-setting artifact?

Command launched:

```bash
uv run python scripts/lightzero_live_eval_queue.py --run-id lz-visual-pong-exact-installed-0.2.0-s1 --attempt-id train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-relpath --eval-id drift-s1-l4-stockupc-none-0-16000-18459-stock2048-seed1 --compute gpu-l4-t4 --eval-pass custom --seed 1 --max-eval-steps 2048 --max-episode-steps 2048 --max-env-step 65536 --update-per-collect -1 --selected-iterations 0,16000,18459 --group-size 1 --max-parallel-launches 3 --execute
```

The first launch wrote `iteration_0` and timed out on the two stronger rows at
480 seconds. After the eval-timeout bump, I relaunched the same command with
`--max-parallel-launches 2`; it picked up only `iteration_16000` and
`iteration_18459`.

Local summary:
`artifacts/local/lightzero-eval-manifests/drift-s1-l4-stockupc-none-0-16000-18459-stock2048-seed1/summary_baseline_deltas.tsv`.

Remote eval root:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s1/attempts/train-faithful-short-installed-0.2.0-s1-65536-ckpt1000-spawn-relpath/eval/drift-s1-l4-stockupc-none-0-16000-18459-stock2048-seed1/`.

| Checkpoint | Manual steps | Stock steps | Manual return | Stock return | Stock positive rewards | Strict load | Fallback |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `iteration_0` | `763/2048` | `761` | `-21` | `-21` | `0` | `true` | `false` |
| `iteration_16000` | `1501/2048` | `1671` | `-19` | `-20` | `1` | `true` | `false` |
| `iteration_18459` | `2048/2048` | `2048` | `-3` | `-11` | `8` | `true` | `false` |

Plain read: the strict eval makes the `iteration_18459` score less spectacular
than the earlier `+13` stock return, but the signal is still very real versus
same-run `iteration_0`: it reaches the cap in both manual and stock paths and
gets real positive reward events.

## Is This Cherry-Picked?

For the seed 1 L4/T4 run alone, latest `iteration_18459` is not just a hand
picked middle checkpoint. It was the latest visible checkpoint, and
`iteration_16000` was already strong before it.

Across all runs, only talking about the seed 1 L4/T4 latest checkpoint would be
cherry-picking. The honest cross-run story is mixed: one strong final/latest
run, one H100 run that peaked and fell back, one weak seed 2 run, one seed 3
run that peaked then fell back, and one weaker seed 0 repeat.

## Is This An Eval Artifact?

Not mostly. The stricter rerun kept strict checkpoint load, no fallback,
stock evaluator enabled, matching eval seed `1`, 2048 cap, and
`--update-per-collect -1`. The strong row survived that check.

But the exact number is sensitive. Earlier `iteration_18459` stock return was
`+13`; stricter stock return is `-11`. That is a large difference in magnitude,
so do not claim a precise score from one eval episode. Claim only the robust
direction: the checkpoint is much better than same-run baseline under this
survival-first eval.

## Next Replication

Do not start another short seed-search trainer just to celebrate this row. The
right next replication is already in motion: the longer normal stock/control
H100 wave at `199000` max env steps for seeds `4` and `5`, no survival reward
shaping. Evaluate those with the same strict stock-path telemetry at
`iteration_0`, `iteration_5000`, `iteration_10000`, `iteration_20000`, and
latest/final as checkpoints appear.

If a quick additional eval is wanted, repeat the same stricter 2048-step eval
on seed 1 H100 `iteration_0/10000/16000/17504` with
`--update-per-collect -1` and the longer timeout. That would answer whether the
H100 fallback story survives the same stricter eval surface.

No pytest was run. A syntax check was run:

```bash
uv run python -m py_compile src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py
```
