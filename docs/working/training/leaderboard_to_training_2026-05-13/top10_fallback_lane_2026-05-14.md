# Top-10 Fallback Tournament Lane - 2026-05-14

## Purpose

Side lane for a small active leaderboard while the top-100 gate is debugged.
This lane uses the normal Modal checkpoint tournament app, launched detached so
the rating worker can continue after the local entrypoint exits.

## IDs

- tournament_id: `curvy-oneframe-top10-fallback-20260514a`
- rating_run_id: `elo-oneframe-top10-fallback-20260514a`
- Modal app run: `ap-axtneIsSmTEoFc1iBjByDN`
- rating function call id: `fc-01KRJR5DX1K6AQV49HK66TTDGY`

Duplicate cleanup:

- A second main-thread retry accidentally launched the same IDs as app
  `ap-EECA9fbQ6dDoQR8OSBKRAd` / call `fc-01KRJR6WHR46SAHJKK6H1XRGAW`.
- That was stopped at about 04:05 EDT with
  `uv run --extra modal modal app stop --yes ap-EECA9fbQ6dDoQR8OSBKRAd`.
- Keep only `ap-axtneIsSmTEoFc1iBjByDN` as the intended fallback writer.
- Do not launch another writer for these same tournament/rating IDs unless this
  lane is explicitly abandoned and cleaned up.

## Source

- historical ranked source: `curvytron-latest212-smoke-20260513`
- source ref:
  `tournaments/curvytron/leaderboards/curvytron-latest212-smoke-20260513/latest.json`
- local source copy:
  `/private/tmp/curvy-top10-fallback-source-latest.json`
- source shape: `.rows`, not `.ratings`
- source row count: 212
- selected source refs: 10
- selected files:
  - `/private/tmp/curvy-top10-fallback-refs-lines.txt`
  - `/private/tmp/curvy-top10-fallback-refs.csv`

Top 10 selected by ascending `rank` from `.rows`:

```text
1 ckpt-059-train-lightzero_exp_260513_125142-ckpt-ite-64ad373b
2 ckpt-006-train-lightzero_exp-ckpt-iteration_270000-01cf5f89
3 ckpt-007-train-lightzero_exp-ckpt-iteration_255000-2696f65f
4 ckpt-159-train-lightzero_exp_260513_161440-ckpt-ite-f115e40d
5 ckpt-015-train-lightzero_exp-ckpt-iteration_255000-9ea2e6b4
6 ckpt-131-train-lightzero_exp_260513_164102-ckpt-ite-969d31c2
7 ckpt-005-train-lightzero_exp-ckpt-iteration_240000-4bea7fc3
8 ckpt-030-train-lightzero_exp-ckpt-iteration_225000-6ccc2e34
9 ckpt-082-train-lightzero_exp-ckpt-iteration_160000-29f996dc
10 ckpt-004-train-lightzero_exp-ckpt-iteration_255000-76a5716a
```

## Commands

Check this lane did not already exist locally:

```text
rg -n "curvy-oneframe-top10-fallback-20260514a|elo-oneframe-top10-fallback-20260514a|top10 fallback|top10_fallback" docs . --glob '!node_modules'
```

Check this lane did not already exist on the tournament Volume:

```text
uv run --extra modal modal volume ls --json curvyzero-curvytron-tournaments tournaments/curvytron/curvy-oneframe-top10-fallback-20260514a
```

Fetch the durable latest 212 leaderboard:

```text
uv run --extra modal modal volume get --force curvyzero-curvytron-tournaments tournaments/curvytron/leaderboards/curvytron-latest212-smoke-20260513/latest.json /private/tmp/curvy-top10-fallback-source-latest.json
```

Inspect the JSON shape:

```text
jq '{keys: keys, has_ratings: has("ratings"), has_rows: has("rows"), count: ((.ratings // .rows // [])|length), first_keys: ((.ratings // .rows // [])[0] | keys), leaderboard_id: .leaderboard_id, snapshot_id: .snapshot_id}' /private/tmp/curvy-top10-fallback-source-latest.json
```

Write ranked refs:

```text
jq -r '[.ratings[]? // empty, .rows[]?] | sort_by((.rank // 999999), (-(.rating // 0))) | .[:10][] | .checkpoint_ref' /private/tmp/curvy-top10-fallback-source-latest.json > /private/tmp/curvy-top10-fallback-refs-lines.txt
```

Write comma-separated CLI refs:

```text
jq -r '[.ratings[]? // empty, .rows[]?] | sort_by((.rank // 999999), (-(.rating // 0))) | .[:10] | map(.checkpoint_ref) | join(",")' /private/tmp/curvy-top10-fallback-source-latest.json > /private/tmp/curvy-top10-fallback-refs.csv
```

Verify count and uniqueness:

```text
wc -l /private/tmp/curvy-top10-fallback-refs-lines.txt
sort -u /private/tmp/curvy-top10-fallback-refs-lines.txt | wc -l
```

Estimate the plan:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode estimate --checkpoint-refs "$(cat /private/tmp/curvy-top10-fallback-refs.csv)" --expected-checkpoint-count 10 --pair-selection all_pairs --pairs-per-round 45 --games-per-pair 21 --games-per-shard 1 --max-steps 8000 --num-simulations 8 --decision-source-frames 1 --decision-ms 16.666666666666668 --source-physics-step-ms 16.666666666666668 --policy-mode eval --active-pool-limit 10 --save-gif --gif-sample-games-per-pair 5
```

Launch detached:

```text
uv run --extra modal modal run --detach -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode rating --tournament-id curvy-oneframe-top10-fallback-20260514a --rating-run-id elo-oneframe-top10-fallback-20260514a --checkpoint-refs "$(cat /private/tmp/curvy-top10-fallback-refs.csv)" --expected-checkpoint-count 10 --round-count 1 --pair-selection all_pairs --pairs-per-round 45 --games-per-pair 21 --games-per-shard 1 --max-steps 8000 --num-simulations 8 --decision-source-frames 1 --decision-ms 16.666666666666668 --source-physics-step-ms 16.666666666666668 --policy-mode eval --active-pool-limit 10 --save-gif --gif-sample-games-per-pair 5
```

Verify durable rating files:

```text
uv run --extra modal modal volume ls --json curvyzero-curvytron-tournaments tournaments/curvytron/curvy-oneframe-top10-fallback-20260514a/ratings/elo-oneframe-top10-fallback-20260514a
uv run --extra modal modal volume ls --json curvyzero-curvytron-tournaments tournaments/curvytron/curvy-oneframe-top10-fallback-20260514a/ratings/elo-oneframe-top10-fallback-20260514a/rounds
uv run --extra modal modal volume get --force curvyzero-curvytron-tournaments tournaments/curvytron/curvy-oneframe-top10-fallback-20260514a/ratings/elo-oneframe-top10-fallback-20260514a/progress.json /private/tmp/curvy-top10-fallback-progress.json
uv run --extra modal modal volume get --force curvyzero-curvytron-tournaments tournaments/curvytron/curvy-oneframe-top10-fallback-20260514a/ratings/elo-oneframe-top10-fallback-20260514a/config.json /private/tmp/curvy-top10-fallback-config.json
uv run --extra modal modal volume get --force curvyzero-curvytron-tournaments tournaments/curvytron/curvy-oneframe-top10-fallback-20260514a/ratings/elo-oneframe-top10-fallback-20260514a/latest.json /private/tmp/curvy-top10-fallback-latest.json
```

## State

Estimate output:

- checkpoint_count: 10
- pair_selection: `all_pairs`
- pair_count: 45
- games_per_pair: 21
- game_count: 945
- games_per_shard: 1
- save_gif: true
- gif_sample_games_per_pair: 5
- gif_sample_strategy: `evenly_spaced`
- gif_count: 225

Durable config was present at 2026-05-14 04:03 EDT:

- `config.json`
- `progress.json`
- `provisional_latest.json`
- `rounds/round-000000`

Config verified:

- checkpoint_count: 10
- pair_selection: `all_pairs`
- pairs_per_round: 45
- games_per_pair: 21
- games_per_shard: 1
- policy_mode: `eval`
- num_simulations: 8
- max_steps: 8000
- decision_source_frames: 1
- decision_ms: 16.666666666666668
- source_physics_step_ms: 16.666666666666668
- active_pool_limit: 10
- save_gif: true
- gif_sample_games_per_pair: 5
- gif_sample_strategy: `evenly_spaced`

Important active-row threshold:

- With 10 players and all-pairs, each checkpoint can only play 9 opponents and
  189 games.
- The rating config normalized `placement_min_games` to 189. The effective
  opponent target is `min(placement_min_opponents=20, possible_opponents=9)`,
  so rows can become `active` after the full all-pairs round.
- If this fallback is published as a public leaderboard, use scaled public
  thresholds such as `leaderboard_active_min_valid_games=189` and
  `leaderboard_active_min_distinct_opponents=9`. The top100 defaults are too
  strict for this fallback.

Progress snapshot immediately after launch:

- status: `running`
- phase: `game_map_started`
- pair_count: 45
- game_count: 945
- started_pair_count: 0
- completed_pair_count: 0
- completed_game_count: 0
- latest_ref:
  `tournaments/curvytron/curvy-oneframe-top10-fallback-20260514a/ratings/elo-oneframe-top10-fallback-20260514a/latest.json`

`latest.json` did not exist yet at the first post-launch check.

Later check:

- 45 battle directories existed, covering the full all-pairs schedule.
- `latest.json` was still absent while games were running.
- The fallback is useful only after
  `ratings/elo-oneframe-top10-fallback-20260514a/latest.json` exists and
  `ratings[:4]` are `active`.

Completed round-0 evidence:

- `latest.json` exists.
- `progress.json` says status `complete`, phase `reduced`.
- round 0 completed 45/45 pairs and 945/945 games.
- top 4 rows are `active`, each with 189 games and 9 distinct opponents.
- `stable=false` means Elo still moved more than the convergence threshold; it
  does not mean the rows are provisional. Treat this as usable fallback evidence
  but not final strength truth.

Round-1 continuation:

- A follow-up command with `--continue-from-latest` and empty explicit refs
  created `round-000001`. This is valid because the code restores the checkpoint
  roster from existing `latest.json` when `continue_from_latest=true`.
- round 1 app: `ap-3kIow3FeXD4VVwQmHXraNI`
- round 1 call: `fc-01KRJRGCZB6G7N8KDH6J80G3AY`
- logs showed real `ok=true` game summaries for `round-000001`.
- round 1 finished cleanly at 2026-05-14T08:13Z with 45/45 pairs and 945/945
  games.
- `latest.json` now points at `round-000001`.
- `stable=false` remains true, but the latest is final, not provisional.
- Do not start a third writer for the same IDs unless intentionally continuing.

Current lesson:

- Small detached tournaments work. The top100 issue is not "Modal cannot run
  this"; it is that the rating parent waits for the full game map before
  reducing. The old cheap progress path was also misleading for
  `games_per_shard=1` because it did not count per-game summaries by default.
  The current tree adds an explicit `--progress-count-game-summaries`
  diagnostic switch for this case. Use it on bounded tournaments; do not use it
  as routine top100 web polling.

Pre-launch fallback gate for trainer use:

```text
uv run --extra modal modal volume get --force curvyzero-curvytron-tournaments \
  tournaments/curvytron/curvy-oneframe-top10-fallback-20260514a/ratings/elo-oneframe-top10-fallback-20260514a/latest.json \
  /private/tmp/curvy-oneframe-top10-fallback-20260514a-latest.json

jq -e '
  .rating_spec.decision_source_frames == 1 and
  .rating_spec.games_per_pair == 21 and
  (.ratings | length) >= 4 and
  all(.ratings[:4][]; .status == "active") and
  all(.ratings[:4][]; (.checkpoint_ref | test("iteration_[0-9]+\\.pth\\.tar$"))) and
  all(.ratings[:4][]; ((.checkpoint_ref | contains("latest")) | not) and ((.checkpoint_ref | contains("ckpt_best")) | not))
' /private/tmp/curvy-oneframe-top10-fallback-20260514a-latest.json
```

Fallback manifest built from round-0 latest:

```text
artifacts/local/curvytron_tonight18_manifests/curvy-night18-top10fallback-20260514a/curvy-night18-top10fallback-20260514a.json
```

Dry-run submit of one row passed:

```text
uv run python scripts/submit_curvytron_survivaldiag_manifest.py \
  artifacts/local/curvytron_tonight18_manifests/curvy-night18-top10fallback-20260514a/curvy-night18-top10fallback-20260514a.json \
  --limit 1
```
