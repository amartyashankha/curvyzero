# Tournament Debug - 2026-05-14

Read inputs: `current_state.md`, `operator_runbook.md`, current
`src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`, Modal app/list
history/logs, Modal Dict/Queue, and tournament Volume artifacts. This was a
read-only inspection of remote state: no remote purge, stop, or takeover.

## Arenas That Currently Matter

- `curvy-night18-top10r1-20260514a / elo-night18-top10r1-20260514a`
  - This is the live tonight18 fallback intake. Dict manifest is active and was
    updated at `2026-05-14T14:17:24Z`; scan prefix is `curvy-night18top10r1`.
  - Queue partition
    `q:curvy-night18-top10r1-20:elo-night18-top1:b4c6f255f5` had length `289`
    at check time; total intake Queue length was `5423`.
  - Latest fetched `latest.json`: `round-000004`, `91` rating rows, `51`
    active rows, `300` pairs, `6300` games, `decision_source_frames=1`,
    `stable=false`.
  - Latest fetched `progress.json`: `status=running`,
    `phase=game_map_started`, `round-000005`, `300` pairs, `6300` games,
    `started_pair_count=0`, `completed_game_count=0`, updated
    `2026-05-14T14:23:26Z`, count basis `summary_files`.
  - Obvious artifact concern: the Volume has `round-000000` through
    `round-000007`. `round-000004` and `round-000005` have old
    `ratings.json/results.json`, but their `input.json/progress.json` were
    modified later at `10:22`/`10:23 EDT`. `round-000007` has only
    `input.json/progress.json`. That is live evidence of overlapping or
    rewritten round artifacts, not a clean single-writer continuation.

- `curvy-oneframe-visual-main-20260514a / elo-oneframe-visual-main-20260514a`
  - Active manifest, updated `2026-05-14T14:17:26Z`.
  - Queue partition
    `q:curvy-oneframe-visual-ma:elo-oneframe-vis:8e9444851b` had length `2`.
  - `progress.json`: `status=running`, `phase=game_map_started`,
    `round-000000`, `80200` pairs, `1684200` games, zero counted starts/games,
    updated `2026-05-14T14:02:07Z`.
  - App logs at `10:22 EDT` show successful game workers with
    `ok=true` and battle ids beginning `rate-elo-oneframe-v-r000000...`, so the
    deployed service is doing live game work even though the durable progress
    file is not counting per-game summaries.

- `curvy-oneframe-visual-main-20260514b / elo-oneframe-visual-main-20260514b`
  - Active manifest, updated `2026-05-14T14:17:26Z`.
  - Queue partition
    `q:curvy-oneframe-visual-ma:elo-oneframe-vis:c975b44e95` had length `1`.
  - `progress.json`: `status=running`, `phase=game_map_started`,
    `round-000000`, `36856` pairs, `773976` games, zero counted starts/games,
    updated `2026-05-14T14:16:55Z`.
  - Same log evidence as above: live successful game workers are present, but
    the progress artifact is not useful without game-summary counting.

- `curvy-oneframe-top100-gate-20260514a / elo-oneframe-top100-gate-20260514a`
  - This top100 gate is complete, not currently stuck.
  - `progress.json`: `status=complete`, `phase=reduced`, `round-000000`,
    `1000/1000` pairs, `21000/21000` games, `0` failed games, reduced at
    `2026-05-14T08:54:07Z`.
  - `latest.json`: `100` rating rows, `62` active rows, `1000` pairs,
    `21000` games, `decision_source_frames=1`, `stable=false`.

## Intake / Claim Evidence

- Deployed app `curvyzero-checkpoint-tournament` is deployed as v5 at
  `2026-05-14 05:33:51-04:00`, commit shown by Modal as `df436fb*`.
- Modal app list showed the deployed app with `526` tasks plus several detached
  tournament apps from `03:14` through `05:27 EDT`, each with roughly
  `505-520` tasks.
- Current local claim-key code returns
  `rating_claim:{manifest_key}:mode-{fresh|continue}`. The live Dict contains
  both older pool-suffixed claim keys and current mode-only keys.
- Night18 current mode-only claim:
  `rating_claim:manifest:curvy-night18-top10r1-20260514a:elo-night18-top10r1-20260514a:mode-continue`
  was created `2026-05-14T14:22:56Z`, has `checkpoint_count=198`,
  `event_count=100`, `queue_len_before=0`, `queue_len_after_repair=198`,
  `pool_hash=5af0138374dc0342`, and `repaired_stale_claim=true`.
- Earlier night18 pool-suffixed claim
  `...:pool-eb9c2c1872704001` was created `2026-05-14T14:10:27Z` with
  `checkpoint_count=192`, `event_count=100`, `queue_len_after_repair=192`.

## Live Failures / Warnings

- Recent deployed logs contain many successful game JSON lines with
  `"ok": true`, `error_type: null`, and real `worker_timing`, so the service is
  not globally dead.
- Recent logs also contain repeated Modal cancellation lines such as
  `failed to respond to cancellation for too long: 30 seconds - killing task`,
  plus `Runner failed with exception: Runner disappeared, in-progress inputs
  will be re-scheduled.` These are live service warnings/failures, even though
  game work continues.
- A `Traceback` search found one `KeyboardInterrupt` in importlib at
  `2026-05-14 09:30:13-04:00`.
- Searches for `input was replaced`, `refusing`, and `RuntimeError` in the last
  two hours returned no matching deployed logs.

## Bottom Line

Top100 gate artifacts look complete and usable as health evidence. Visual-main
`a`/`b` are doing live game work, but their progress files are misleading under
per-game summary output. Night18 is the concerning lane: the live intake is
active and claims are being repaired, but rating artifacts show overwritten
round input/progress around already-written ratings/results and a current
progress pointer behind existing later round directories. Do not publish or use
night18 as clean continuation evidence until that artifact mismatch is resolved.
