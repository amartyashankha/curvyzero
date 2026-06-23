# Next-Batch Seeding

The current tournament is useful and should be preserved. It found strong
mid-run policies even when the training runs regressed later.

## Historical r18fresh Leaderboard Truth

As of direct reads on 2026-05-16 for the old overnight r18fresh leaderboard:

- The historical r18fresh rating latest had advanced to `round-000033`.
  Do not confuse this with current CZ26 live state. Current CZ26 latest rating
  remains `round-000015` / `919` while active internal game-batch artifact
  `round-000033` is still playing.
- Rating rows: `564` total, `100` active, `0` provisional, `464` retired.
- Rating stable: `false`.
- Current rank 1: `ckpt-432-train-lightzero_exp-ckpt-iteration_180000-0ed114de`.
- Current rank 1 run:
  `curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423`.
- Public/trainer snapshot read was `auto-r000032-g22-555c999b`, generation 22,
  with `563` rows and `100` active rows.

The website showing around 500 rows is not the top-100 training source. It is
showing raw rating rows with a page cap. The trainer-facing selection should use
active rows, not all displayed rows.

## Top-10 Candidate Refs From Snapshot `auto-r000032-g22-555c999b`

These are raw active ranks 1-10 from the trainer/public leaderboard snapshot.
They are not deduped by run.

Exact checkpoint refs are preserved in
`TOP10_RAW_REFS_auto-r000032-g22-555c999b.txt`.

| Rank | Rating | Iteration | Checkpoint ID | Run |
| ---: | ---: | ---: | --- | --- |
| 1 | 1632.68 | 180000 | `ckpt-432-train-lightzero_exp-ckpt-iteration_180000-0ed114de` | `curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423` |
| 2 | 1615.02 | 250000 | `ckpt-541-train-lightzero_exp-ckpt-iteration_250000-9db934d8` | `curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423` |
| 3 | 1606.17 | 150000 | `ckpt-313-train-lightzero_exp_260516_110032-ckpt-ite-514e46fe` | `curvy-r18fresh-survbonusout-blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1i-2115bef336` |
| 4 | 1594.84 | 190000 | `ckpt-433-train-lightzero_exp-ckpt-iteration_190000-530e6681` | `curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423` |
| 5 | 1594.36 | 200000 | `ckpt-460-train-lightzero_exp-ckpt-iteration_200000-fd7d2374` | `curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423` |
| 6 | 1588.15 | 260000 | `ckpt-555-train-lightzero_exp-ckpt-iteration_260000-f365c280` | `curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423` |
| 7 | 1587.87 | 240000 | `ckpt-529-train-lightzero_exp-ckpt-iteration_240000-e3badb5a` | `curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423` |
| 8 | 1584.43 | 150000 | `ckpt-321-train-lightzero_exp-ckpt-iteration_150000-a49409d1` | `curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423` |
| 9 | 1581.33 | 170000 | `ckpt-378-train-lightzero_exp-ckpt-iteration_170000-c67f268e` | `curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423` |
| 10 | 1569.82 | 150000 | `ckpt-291-train-lightzero_exp-ckpt-iteration_150000-9f8dd2b1` | `curvy-r18fresh-survbonusout-blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1i-2115bef336` |

## Locked Seeding Decision

For the next Grid A, Grid B, and canary launches, every learner policy should
start from the single top checkpoint from the old overnight r18fresh
leaderboard snapshot. Do not seed different rows from different top-10 entries.
Do not use a deduped top-10 list as initial policy seeds for this launch.

Current pinned launch seed:

```text
rank 1, rating 1632.6800339881258
training/lightzero-curvytron-visual-survival/curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/attempts/try-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/train/lightzero_exp/ckpt/iteration_180000.pth.tar
```

Raw top 10 is still preserved for audit and possible opponent seeding, but it
is not the initial-policy seed plan. The top-1 checkpoint above is the shared
`initial_policy_checkpoint_ref` for every new training run unless the user
explicitly changes this decision.

## Commands

Use deployed status/control tooling first for live CZ26 truth. The raw Volume
commands below are audit/debug fallback for this preserved historical
r18fresh seed snapshot, not the normal live-control workflow.

Count top-100 active rows:

```bash
uv run --extra modal modal volume get curvyzero-curvytron-tournaments-v2 \
  tournaments/curvytron/leaderboards/curvy-r18fresh-live-bounded-dsf1-20260516b-elo-r18fresh-live-bounded-dsf1-20260516b-training/latest.json - \
| sed '/Finished downloading files to local/d' \
| jq '[.rows[] | select(.status=="active" and (.rank|tonumber) <= 100 and (.checkpoint_ref // "") != "")] | {count:length, unique_count:(map(.checkpoint_ref)|unique|length), first_rank:(map(.rank)|min), last_rank:(map(.rank)|max)}'
```

Extract raw active top-10 refs:

```bash
uv run --extra modal modal volume get curvyzero-curvytron-tournaments-v2 \
  tournaments/curvytron/leaderboards/curvy-r18fresh-live-bounded-dsf1-20260516b-elo-r18fresh-live-bounded-dsf1-20260516b-training/latest.json - \
| sed '/Finished downloading files to local/d' \
| jq -r '[.rows[] | select(.status=="active" and (.rank|tonumber) <= 10)] | sort_by(.rank)[] | .checkpoint_ref'
```

## Before Launching Next Batch

- Pin the exact snapshot used for bootstrap.
- Audit every checkpoint ref exists in the active v2 training volume.
- Verify the rank-1 checkpoint ref above exists in the active v2 training
  volume.
- Verify every Grid A/Grid B/canary row has exactly this
  `initial_policy_checkpoint_ref`.
- Document whether rank stability is required. Current rating is useful for
  exploratory seeds, but `stable=false` means it is not final Elo truth.
