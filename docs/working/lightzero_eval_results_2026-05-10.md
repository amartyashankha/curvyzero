# LightZero Pong Eval Results Ledger - 2026-05-10

Claim: the recent fetched Pong eval manifests are valid strict-load/no-fallback
readouts, but they do not show stock-evaluator Pong learning yet. Across all
summarized checkpoints, `stock_return` stayed `-21`, stock positive reward count
stayed `0`, and stock negative/nonzero reward counts stayed `21/21`.

Non-claim: this is not a solved-Pong result, not a stable policy result, and not
evidence that the manual rollout path is enough by itself. Manual steps are
listed only as secondary telemetry because the stock evaluator fields are the
first-line readout.

All six requested eval directories fetched from the `curvyzero-runs` Modal
Volume into `artifacts/local/lightzero-eval-manifests/<eval-id>`. No fetch path
was missing.

## Stock-First Summary

| Eval | Checkpoint | Stock steps / cap | Stock return | Stock rewards +/-/nonzero | Stock terminal | Manual steps / cap | Verdict |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| s10 sweep65k L4 | `iteration_0` | `757/2048` | `-21` | `0/21/21` | `done` | `762/2048` | `collapsed_action` |
| s10 sweep65k L4 | `iteration_1000` | `757/2048` | `-21` | `0/21/21` | `done` | `762/2048` | `manual_stock_mismatch` |
| s11 sweep65k L4 | `iteration_0` | `760/2048` | `-21` | `0/21/21` | `done` | `758/2048` | `collapsed_action` |
| s11 sweep65k L4 | `iteration_1000` | `760/2048` | `-21` | `0/21/21` | `done` | `758/2048` | `manual_stock_mismatch` |
| s14 sweep65k L4 CPU16 | `iteration_0` | `758/2048` | `-21` | `0/21/21` | `done` | `761/2048` | `collapsed_action` |
| s14 sweep65k L4 CPU16 | `iteration_1000` | `758/2048` | `-21` | `0/21/21` | `done` | `761/2048` | `manual_stock_mismatch` |
| s14 sweep65k L4 CPU16 | `iteration_2000` | `758/2048` | `-21` | `0/21/21` | `done` | `761/2048` | `collapsed_action` |
| s18 sweep65k H100 CPU16 | `iteration_0` | `759/2048` | `-21` | `0/21/21` | `done` | `762/2048` | `collapsed_action` |
| s18 sweep65k H100 CPU16 | `iteration_1000` | `759/2048` | `-21` | `0/21/21` | `done` | `762/2048` | `collapsed_action` |
| s18 sweep65k H100 CPU16 | `iteration_3000` | `759/2048` | `-21` | `0/21/21` | `done` | `762/2048` | `manual_stock_mismatch` |
| s19 sweep65k H100 CPU16 | `iteration_0` | `761/2048` | `-21` | `0/21/21` | `done` | `757/2048` | `collapsed_action` |
| s19 sweep65k H100 CPU16 | `iteration_1000` | `789/2048` | `-21` | `0/21/21` | `done` | `757/2048` | `manual_stock_mismatch` |
| s19 sweep65k H100 CPU16 | `iteration_2000` | `883/2048` | `-21` | `0/21/21` | `done` | `817/2048` | `manual_stock_mismatch` |
| repeatB s1 L4 | `iteration_0` | `761/2048` | `-21` | `0/21/21` | `done` | `763/2048` | `collapsed_action` |
| repeatB s1 L4 | `iteration_1000` | `761/2048` | `-21` | `0/21/21` | `done` | `763/2048` | `manual_stock_mismatch` |
| repeatB s1 L4 | `iteration_2000` | `789/2048` | `-21` | `0/21/21` | `done` | `763/2048` | `manual_stock_mismatch` |
| repeatB s1 L4 | `iteration_3000` | `761/2048` | `-21` | `0/21/21` | `done` | `824/2048` | `manual_stock_mismatch` |

## Per-Eval Notes

### s10 sweep65k L4

Remote eval:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s10-sweep65k-l4/attempts/train-normal-sweep65k-s10-ckpt1000-spawn-l4-relpath/eval/sweep65k-s10-0-1000-stock2048-seed10`.

Stock evaluator result is flat from `iteration_0` to `iteration_1000`:
`stock_steps_survived=757`, `stock_episode_length=757`,
`stock_return=-21`, and stock rewards `0` positive, `21` negative, `21`
nonzero. Manual rollout also stays flat at `762/2048`.

### s11 sweep65k L4

Remote eval:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s11-sweep65k-l4/attempts/train-normal-sweep65k-s11-ckpt1000-spawn-l4-relpath/eval/sweep65k-s11-0-1000-stock2048-seed11`.

Stock evaluator result is flat from `iteration_0` to `iteration_1000`:
`stock_steps_survived=760`, `stock_episode_length=760`,
`stock_return=-21`, and stock rewards `0` positive, `21` negative, `21`
nonzero. Manual rollout stays flat at `758/2048`.

### s14 sweep65k L4 CPU16

Remote eval:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s14-sweep65k-l4cpu16/attempts/train-normal-sweep65k-s14-ckpt1000-spawn-l4cpu16-relpath/eval/sweep65k-s14-0-1000-2000-stock2048-seed14`.

Stock evaluator result is flat across `iteration_0`, `iteration_1000`, and
`iteration_2000`: `stock_steps_survived=758`,
`stock_episode_length=758`, `stock_return=-21`, and stock rewards `0`
positive, `21` negative, `21` nonzero. Manual rollout stays flat at
`761/2048`.

### s18 sweep65k H100 CPU16

Remote eval:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s18-sweep65k-h100cpu16/attempts/train-normal-sweep65k-s18-ckpt1000-spawn-h100cpu16-relpath/eval/sweep65k-s18-0-1000-2000-3000-stock2048-seed18`.

The fetched directory summarized rows for `iteration_0`, `iteration_1000`, and
`iteration_3000`; no `iteration_2000` row was present in the fetched manifests.
Stock evaluator result is flat on the available rows:
`stock_steps_survived=759`, `stock_episode_length=759`,
`stock_return=-21`, and stock rewards `0` positive, `21` negative, `21`
nonzero. Manual rollout stays flat at `762/2048`.

### s19 sweep65k H100 CPU16

Remote eval:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s19-sweep65k-h100cpu16/attempts/train-normal-sweep65k-s19-ckpt1000-spawn-h100cpu16-relpath/eval/sweep65k-s19-0-1000-2000-stock2048-seed19`.

This is the only sweep65k eval here with a stock step increase. Stock steps go
from `761` at `iteration_0` to `789` at `iteration_1000` and `883` at
`iteration_2000`, a best stock-step delta of `+122`. The important caution is
that `stock_return` remains `-21`, stock positive rewards remain `0`, and stock
negative/nonzero rewards remain `21/21`. Manual rollout improves secondarily
from `757/2048` to `817/2048`.

### repeatB s1 L4

Remote eval:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s1-repeatB-65536-l4/attempts/train-normal-repeatB-s1-65536-ckpt1000-spawn-l4-relpath/eval/repeatB-s1-0-1000-2000-3000-stock2048-seed1`.

Stock steps are mostly flat: `761` at `iteration_0`, `761` at
`iteration_1000`, `789` at `iteration_2000`, and back to `761` at
`iteration_3000`. Stock return remains `-21` on every row, with stock rewards
`0` positive, `21` negative, `21` nonzero. Manual rollout is secondary but does
show a later-step rise to `824/2048` at `iteration_3000`.

## Commands Run

Fetch pattern:

```bash
uv run --extra modal modal volume get curvyzero-runs <remote-eval-dir> artifacts/local/lightzero-eval-manifests --force
```

Summary pattern:

```bash
uv run python scripts/summarize_lightzero_pong_eval_manifest.py --baseline-deltas --format tsv <local-eval-dir>
```

No pytest run.

## CPU40 eval wave harvest - 2026-05-10

Claim: the CPU40 eval wave completed for normal seeds `24`-`27`, shaped seeds
`30`-`33`, and shaped WaveB retry seeds `34`-`37`. It found two useful
stock-survival bumps: normal seed `27` and shaped seed `30`.

All summarized rows report `strict_load=true` and `fallback_used=false`.
Survival is the first readout; return is secondary. The only useful
stock-step bumps were normal seed 27 and shaped seed 30. Neither got a stock
return gain. Return stayed `-21` and stock positive rewards stayed `0` across
this wave.

| Lane | Eval | Fetched rows | Stock steps survived vs `iteration_0` | Stock return |
| --- | --- | --- | --- | --- |
| normal | `normal-s24-0-1000-5000-stock2048-seed24` | `0,1000,5000` | flat: `763 -> 763 -> 763` | all `-21` |
| normal | `normal-s25-0-1000-5000-stock2048-seed25` | `0,1000,5000` | flat: `758 -> 758 -> 758` | all `-21` |
| normal | `normal-s26-0-1000-5000-stock2048-seed26` | `0,1000,5000` | flat: `763 -> 763 -> 763` | all `-21` |
| normal | `normal-s27-0-1000-5000-stock2048-seed27` | `0,1000,5000` | `758 -> 758 -> 848`; best delta `+90` | all `-21` |
| shaped | `shaped-s30-0-1000-5000-stock2048-seed30` | `0,1000,5000` | `763 -> 763 -> 824`; best delta `+61` | all `-21` |
| shaped | `shaped-s31-0-1000-5000-stock2048-seed31` | `0,1000,5000` | flat: `758 -> 758 -> 758` | all `-21` |
| shaped | `shaped-s32-0-1000-5000-stock2048-seed32` | `0,1000,5000` | high start then lower: `849 -> 761 -> 761` | all `-21` |
| shaped | `shaped-s33-0-1000-5000-stock2048-seed33` | `0,1000,5000` | flat: `762 -> 762 -> 762` | all `-21` |
| shaped WaveB | `shaped-s34-0-1000-stock2048-seed34` | `0,1000` | flat: `758 -> 758` | all `-21` |
| shaped WaveB | `shaped-s35-0-1000-stock2048-seed35` | `0,1000` | flat: `760 -> 760` | all `-21` |
| shaped WaveB | `shaped-s36-0-1000-stock2048-seed36` | `0,1000` | flat: `764 -> 764` | all `-21` |
| shaped WaveB | `shaped-s37-0-1000-stock2048-seed37` | `0,1000` | drops: `789 -> 761` | all `-21` |

Plain read: s27 normal and s30 shaped survived longer at a later checkpoint,
but every summarized row still has stock return `-21` and `0` stock positive
rewards. This is survival signal only, not solved Pong. The follow-on ten-run
wave has spawned normal seeds `50`-`57` and shaped telemetry seeds `60` and
`61`; launch details are in
`docs/working/lightzero_10run_launch_wave_2026-05-10.md`.

## Wave10 seed 56 eval

Eval id: `wave10-normal-s56-0-1000-5000-stock2048-seed56`.

Plain read: this is a survival-only bump, not solved Pong. The stock evaluator
survived longer at `iteration_5000`, but `stock_return` stayed `-21` and stock
positive rewards stayed `0` on every row.

| Checkpoint | Stock steps / cap | Stock return | Stock positive rewards | Manual steps / cap | Plain read |
| --- | ---: | ---: | ---: | ---: | --- |
| `iteration_0` | `763/2048` | `-21` | `0` | secondary | dominant action `4` collapsed |
| `iteration_1000` | `763/2048` | `-21` | `0` | secondary | no stock change |
| `iteration_5000` | `823/2048` | `-21` | `0` | `1479/2048` | survival-only stock bump; action mix less collapsed |

Stock steps improved from `763` to `823`, a `+60` survival delta. The manual
rollout reached `1479` steps at `iteration_5000`, but there is stock/manual
mismatch, so the stock evaluator remains the primary readout.

## Second eval wave

Fetched the four requested eval roots from `curvyzero-runs` into
`artifacts/local/lightzero-eval-manifests/<eval-id>` with `--force`. The bad
seed 17 eval id without `b` was ignored; this section uses only
`sweep65k-s17-0-1000-2000-3000-stock2048b-seed17`.

Stock evaluator still does not show Pong learning. Every available row has
`stock_return=-21` and `stock_positive_reward_count=0`.

| Eval | Checkpoint | Stock steps / cap | Stock return | Stock positive rewards | Stock rewards -/nonzero | Manual steps / cap | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| s12 sweep65k L4 | `iteration_1000` | `757/2048` | `-21` | `0` | `21/21` | `761/2048` | `collapsed_action` |
| s12 sweep65k L4 | `iteration_2000` | `757/2048` | `-21` | `0` | `21/21` | `761/2048` | `collapsed_action` |
| s12 sweep65k L4 | `iteration_3000` | `757/2048` | `-21` | `0` | `21/21` | `761/2048` | `collapsed_action` |
| s13 sweep65k L4 | `iteration_0` | `760/2048` | `-21` | `0` | `21/21` | `760/2048` | `collapsed_action` |
| s13 sweep65k L4 | `iteration_1000` | `760/2048` | `-21` | `0` | `21/21` | `760/2048` | `collapsed_action` |
| s13 sweep65k L4 | `iteration_2000` | `760/2048` | `-21` | `0` | `21/21` | `760/2048` | `collapsed_action` |
| s15 sweep65k L4 CPU16 | `iteration_0` | `759/2048` | `-21` | `0` | `21/21` | `762/2048` | `collapsed_action` |
| s15 sweep65k L4 CPU16 | `iteration_1000` | `759/2048` | `-21` | `0` | `21/21` | `762/2048` | `manual_stock_mismatch` |
| s15 sweep65k L4 CPU16 | `iteration_2000` | `759/2048` | `-21` | `0` | `21/21` | `762/2048` | `negative_return` |
| s17 sweep65k L4 CPU16 | `iteration_0` | `760/2048` | `-21` | `0` | `21/21` | `760/2048` | `manual_stock_mismatch` |
| s17 sweep65k L4 CPU16 | `iteration_1000` | `880/2048` | `-21` | `0` | `21/21` | `820/2048` | `collapsed_action` |
| s17 sweep65k L4 CPU16 | `iteration_2000` | `760/2048` | `-21` | `0` | `21/21` | `760/2048` | `negative_return` |

### Second-wave notes

s12 is incomplete. The fetched manifest contains exactly these rows:
`iteration_1000`, `iteration_2000`, and `iteration_3000`. The requested
`iteration_0` row is not present yet, so that row is pending.

s13 is flat on the fetched rows: `iteration_0`, `iteration_1000`, and
`iteration_2000` all have `760` stock steps, `-21` stock return, and `0` stock
positive rewards. No `iteration_3000` row was present in the fetched manifest.

s15 is also flat on stock steps across `iteration_0`, `iteration_1000`, and
`iteration_2000`: all three rows have `759` stock steps, `-21` stock return,
and `0` stock positive rewards. The action histogram changes, but the stock
score does not.

s17 has the only stock-step bump in this wave: `760` at `iteration_0`, `880` at
`iteration_1000`, then back to `760` at `iteration_2000`. The important
caution is that the stock return still stays `-21` and stock positive rewards
stay `0`. No `iteration_3000` row was present in the fetched `stock2048b`
manifest.

## Third eval wave - exact roots requested 2026-05-10

Claim: these seven requested roots exist in the `curvyzero-runs` Modal Volume
and were fetched locally. They are normal LightZero Atari Pong evals. No
survival-shaped run is included in this section.

Claim: the best stock result in this wave is s13 compact at `iteration_5000`:
`stock_steps_survived=866`, `stock_episode_length=866`, `stock_return=-20`,
and stock rewards `1` positive, `21` negative, `22` nonzero. This is one
stock point better than the starting checkpoint, but it still loses the game.

Non-claim: this is not solved Pong, not a stable policy result, and not proof
that the main lane has learned Pong. Most rows still have `stock_return=-21`
and `0` stock positive rewards. Manual rollout steps are listed only as
secondary notes.

| Eval | Checkpoint | Stock steps / episode / cap | Stock return | Stock rewards +/-/nonzero | Manual steps / cap | Plain read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| s16 sweep65k L4 CPU16 | `iteration_0` | `761/761/2048` | `-21` | `0/21/21` | `762/2048` | no stock gain |
| s16 sweep65k L4 CPU16 | `iteration_1000` | `761/761/2048` | `-21` | `0/21/21` | `762/2048` | no stock gain |
| s16 sweep65k L4 CPU16 | `iteration_2000` | `761/761/2048` | `-21` | `0/21/21` | `762/2048` | no stock gain |
| s16 sweep65k L4 CPU16 | `iteration_3000` | `761/761/2048` | `-21` | `0/21/21` | `762/2048` | no stock gain |
| s13 compact L4 | `iteration_0` | `760/760/2048` | `-21` | `0/21/21` | `760/2048` | start row |
| s13 compact L4 | `iteration_1000` | `760/760/2048` | `-21` | `0/21/21` | `760/2048` | no stock gain |
| s13 compact L4 | `iteration_5000` | `866/866/2048` | `-20` | `1/21/22` | `861/2048` | one stock point won, still lost |
| s19 compact H100 CPU16 | `iteration_0` | `761/761/2048` | `-21` | `0/21/21` | `757/2048` | start row |
| s19 compact H100 CPU16 | `iteration_1000` | `821/821/2048` | `-21` | `0/21/21` | `757/2048` | more stock steps, no score gain |
| s19 compact H100 CPU16 | `iteration_5000` | `761/761/2048` | `-21` | `0/21/21` | `817/2048` | manual steps rose, stock score did not |
| long s20 H100 CPU16 | `iteration_0` | `758/758/2048` | `-21` | `0/21/21` | `763/2048` | start row |
| long s20 H100 CPU16 | `iteration_1000` | `758/758/2048` | `-21` | `0/21/21` | `763/2048` | no stock gain |
| long s21 H100 CPU16 | `iteration_0` | `760/760/2048` | `-21` | `0/21/21` | `762/2048` | start row |
| long s21 H100 CPU16 | `iteration_1000` | `760/760/2048` | `-21` | `0/21/21` | `823/2048` | manual steps rose, stock score did not |
| s18 compact H100 CPU16 | `iteration_0` | `759/759/2048` | `-21` | `0/21/21` | `762/2048` | start row |
| s18 compact H100 CPU16 | `iteration_5000` | `759/759/2048` | `-21` | `0/21/21` | `762/2048` | no stock gain |
| s18 compact H100 CPU16 | `iteration_7000` | `838/838/2048` | `-21` | `0/21/21` | `940/2048` | more stock steps, no stock score gain |
| repeatB compact L4 | `iteration_0` | `761/761/2048` | `-21` | `0/21/21` | `763/2048` | start row |
| repeatB compact L4 | `iteration_5000` | `761/761/2048` | `-21` | `0/21/21` | `763/2048` | no stock gain |

### Third-wave artifact notes

All seven requested eval roots were present and fetched:

- `sweep65k-s16-0-1000-2000-3000-stock2048-seed16`
- `sweep65k-s13-compact-0-1k-5k-stock2048-seed13`
- `sweep65k-s19-compact-0-1k-5k-stock2048-seed19`
- `long199k-s20-0-1000-stock2048-seed20`
- `long199k-s21-0-1000-stock2048-seed21`
- `sweep65k-s18-compact-0-5k-7k-stock2048-seed18`
- `repeatB-s1-compact-0-5k-stock2048-seed1`

Each root has local files under
`artifacts/local/lightzero-eval-manifests/<eval-id>`.

For repeatB compact, the directory has both raw eval JSON files:
`iteration_0_custom_steps2048_seed1/...050803Z.json` and
`iteration_5000_custom_steps2048_seed1/...050803Z.json`. The fetched combined
manifest `manifest_custom_steps2048_seed1_20260510T051152Z.json` lists only
`iteration_5000`, so the table above uses
`summary_baseline_deltas_all_iteration_json.tsv` for that eval.

Summary files written:

- `summary_baseline_deltas.tsv` for s16, s13 compact, s19 compact, long s20,
  long s21, s18 compact, and repeatB compact.
- `summary_baseline_deltas_all_iteration_json.tsv` for repeatB compact because
  its combined manifest omitted the `iteration_0` row.

No pytest run.

## Later normal harvest and shaped early harvest - 2026-05-10

Claim: all eight requested roots were present in the `curvyzero-runs` Modal
Volume and were fetched locally. Every summarized row here reports
`strict_load=true` and `fallback_used=false`.

Non-claim: this is not solved Pong and not proof of stable Pong learning.
Normal proof-lane and survival-shaped side-lane rows are separate below.
Manual rollout steps are secondary telemetry only.

### Normal proof lane

Claim: the strongest stock-step row in this later normal batch is s19
`iteration_5000` at `881/2048`, but its stock return is still `-21` with
stock rewards `0/21/21`. The strongest stock-score rows are s13
`iteration_5000` and s18 `iteration_13000`, both at stock return `-20` with
stock rewards `1/21/22`.

| Eval | Checkpoint | Stock steps / episode / cap | Stock return | Stock rewards +/-/nonzero | Manual steps / cap | Plain read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| s13 compact L4 later | `iteration_0` | `760/760/2048` | `-21` | `0/21/21` | `760/2048` | start row |
| s13 compact L4 later | `iteration_5000` | `866/866/2048` | `-20` | `1/21/22` | `861/2048` | one stock point won, still lost |
| s13 compact L4 later | `iteration_8000` | `760/760/2048` | `-21` | `0/21/21` | `760/2048` | gain did not persist |
| s18 compact H100 CPU16 later | `iteration_0` | `759/759/2048` | `-21` | `0/21/21` | `762/2048` | start row |
| s18 compact H100 CPU16 later | `iteration_7000` | `879/879/2048` | `-21` | `0/21/21` | `883/2048` | more stock steps, no score gain |
| s18 compact H100 CPU16 later | `iteration_10000` | `759/759/2048` | `-21` | `0/21/21` | `762/2048` | back to baseline |
| s18 compact H100 CPU16 later | `iteration_13000` | `833/833/2048` | `-20` | `1/21/22` | `835/2048` | one stock point won, still lost |
| s19 compact H100 CPU16 later | `iteration_0` | `761/761/2048` | `-21` | `0/21/21` | `757/2048` | start row |
| s19 compact H100 CPU16 later | `iteration_1000` | `821/821/2048` | `-21` | `0/21/21` | `757/2048` | more stock steps, no score gain |
| s19 compact H100 CPU16 later | `iteration_5000` | `881/881/2048` | `-21` | `0/21/21` | `877/2048` | best stock steps here, no score gain |
| s19 compact H100 CPU16 later | `iteration_10000` | `761/761/2048` | `-21` | `0/21/21` | `757/2048` | back to baseline |
| repeatB s1 compact L4 later | `iteration_0` | `761/761/2048` | `-21` | `0/21/21` | `763/2048` | start row |
| repeatB s1 compact L4 later | `iteration_5000` | `761/761/2048` | `-21` | `0/21/21` | `763/2048` | no stock gain |
| repeatB s1 compact L4 later | `iteration_8000` | `761/761/2048` | `-21` | `0/21/21` | `763/2048` | no stock gain |

### Shaped side lane

Claim: the shaped early rows do not show stock-score improvement. The largest
shaped stock-step count is s32 `iteration_0` at `997/2048`, but every shaped
row has stock return `-21` and stock rewards `0/21/21`.

| Eval | Checkpoint | Stock steps / episode / cap | Stock return | Stock rewards +/-/nonzero | Manual steps / cap | Plain read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| shaped s30 65k | `iteration_0` | `763/763/2048` | `-21` | `0/21/21` | `763/2048` | start row |
| shaped s30 65k | `iteration_1000` | `763/763/2048` | `-21` | `0/21/21` | `763/2048` | no stock gain |
| shaped s31 65k | `iteration_0` | `758/758/2048` | `-21` | `0/21/21` | `760/2048` | start row |
| shaped s31 65k | `iteration_1000` | `758/758/2048` | `-21` | `0/21/21` | `760/2048` | no stock gain |
| shaped s32 199k | `iteration_0` | `997/997/2048` | `-21` | `0/21/21` | `973/2048` | high start steps, no score gain |
| shaped s32 199k | `iteration_1000` | `761/761/2048` | `-21` | `0/21/21` | `758/2048` | lower than start |
| shaped s33 199k | `iteration_0` | `762/762/2048` | `-21` | `0/21/21` | `759/2048` | start row |
| shaped s33 199k | `iteration_1000` | `762/762/2048` | `-21` | `0/21/21` | `759/2048` | no stock gain |

### Later harvest artifact notes

Fetched normal proof-lane roots:

- `sweep65k-s13-compact-0-5k-8k-stock2048-seed13`
- `sweep65k-s18-compact-0-7k-10k-13k-stock2048-seed18`
- `sweep65k-s19-compact-0-1k-5k-10k-stock2048-seed19`
- `repeatB-s1-compact-0-5k-8k-stock2048-seed1`

Fetched shaped side-lane roots:

- `shaped-s30-65k-0-1000-stock2048-seed30`
- `shaped-s31-65k-0-1000-stock2048-seed31`
- `shaped-s32-199k-0-1000-stock2048-seed32`
- `shaped-s33-199k-0-1000-stock2048-seed33`

Summary files written under each local root:

- `summary_baseline_deltas.tsv` for all eight fetched roots.
- `summary_baseline_deltas_all_iteration_json.tsv` for
  `sweep65k-s18-compact-0-7k-10k-13k-stock2048-seed18`.

For s18 later, the fetched directory has per-checkpoint JSON dirs for
`iteration_0`, `iteration_7000`, `iteration_10000`, and `iteration_13000`, but
the combined manifests summarized only three rows and omitted `iteration_0`.
The normal proof-lane table above uses the all-iteration JSON summary for s18.

Pending or missing requested roots: none.

No pytest run.
