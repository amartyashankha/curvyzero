# Tournament Survival Duration Trajectory - 2026-05-16

Active run analyzed: tournament `curvy-r18fresh-live-bounded-dsf1-20260516b`, rating `elo-r18fresh-live-bounded-dsf1-20260516b`.

Source: read directly from Modal volume `curvyzero-curvytron-tournaments-v2`. I did not use local cached copies.

## Where The Duration Fields Live

Rating round snapshots are slim. `rounds/round-*/ratings.json` contains standings and rating deltas, and `rounds/round-*/results.json` contains `pair_rating_results` plus `pair_summary_refs`, but neither carries per-game duration distributions.

The per-game fields are in shard summaries referenced from each pair:

`tournaments/curvytron/<tournament_id>/battles/<battle_id>/shards/shard-000000-games-000000-000020/summary.json`

Each shard `games[]` row has:

- top-level `physical_steps`
- `score.physical_steps`
- `score.max_steps`
- `score.terminal_reason`
- `score.score_reason`
- optional `gif_ref`

Pair `battle.json` has only aggregate `tally.average_physical_steps` in this run because `result_detail_mode` is `shard_tally`.

## Duration Trajectory

Completed rounds available: `round-000000` through `round-000030`. `round-000031` had only `input.json` and `progress.json` at read time, so it was not included. Each completed round had 300 pairs x 21 games = 6,300 games; total analyzed games = 195,300.

| Window | Games | Mean steps | Median | p90 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| r00-r04 | 31,500 | 131.59 | 131 | 213 | 249 | 315.01 | 699 |
| r13-r17 | 31,500 | 139.64 | 131 | 227 | 262 | 333 | 575 |
| r26-r30 | 31,500 | 151.48 | 140 | 251 | 285 | 352 | 610 |

Readout: game duration is increasing, not flat. Round-index correlation is strong for mean (`r=0.935`), p90 (`r=0.914`), p95 (`r=0.896`), and p99 (`r=0.832`). Median is noisier/stickier but still rises (`r=0.718`), from 131 to 143 by r30. The single-game max is noisy (`r=-0.012`) and should not be used as the trend metric.

Representative rounds:

| Round | Mean | Median | p90 | p95 | p99 | Max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 131.16 | 131 | 216 | 252 | 314 | 458 |
| 10 | 137.16 | 131 | 224 | 258.1 | 334 | 530 |
| 20 | 144.49 | 134 | 234 | 268 | 336 | 520 |
| 28 | 152.62 | 142 | 254 | 288 | 353 | 581 |
| 29 | 157.36 | 147 | 260 | 294 | 366 | 610 |
| 30 | 154.61 | 143 | 256 | 288 | 352 | 557 |

## Top-Rank Association

Top-ranked policies are associated with longer tournament games.

Using latest r30 ranks, games involving a top-10 policy averaged 164.32 steps, median 155, versus 138.89 mean and 131 median for games with no latest-top-10 policy. Games involving latest top-50 policies averaged 148.11 steps versus 136.03 otherwise.

Equal-weighted per-policy averages show the same effect: final top-10 policies averaged 166.69 steps across their games, top-50 averaged 153.58, bottom-50 averaged 118.77. Correlation between latest rank number and per-policy average game duration was `-0.700`; lower rank number means better rank, so this is a strong longer-duration/top-rank association.

Using each round's rank at that time is weaker but still positive: round-current top-10 involved games averaged 145.33 steps versus 139.11 otherwise; round-current top-50 involved games averaged 142.22 versus 137.32 otherwise.

## Cap / GIF Truncation Check

No duration looked capped by `max_steps`. The run used `max_steps=1048576`; max observed game length was 779 steps, and `physical_steps >= max_steps` occurred 0 times.

No GIF sampling cap applies here. `save_gif=false` in the rating spec and every analyzed game had `gif_ref=null`; `gif_sample_games_per_pair=5` is present in config but inactive.

Terminal reasons were ordinary round endings: `round_survivor_win` = 193,204 games; `round_all_dead_draw` = 2,096 games.

## Reproduce

List the remote run and rounds:

```bash
modal volume ls --json curvyzero-curvytron-tournaments-v2 \
  /tournaments/curvytron/curvy-r18fresh-live-bounded-dsf1-20260516b/ratings/elo-r18fresh-live-bounded-dsf1-20260516b

modal volume ls --json curvyzero-curvytron-tournaments-v2 \
  /tournaments/curvytron/curvy-r18fresh-live-bounded-dsf1-20260516b/ratings/elo-r18fresh-live-bounded-dsf1-20260516b/rounds
```

Run the analyzer:

```bash
python - <<'PY'
import json, math, statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import modal

VOL='curvyzero-curvytron-tournaments-v2'
T='curvy-r18fresh-live-bounded-dsf1-20260516b'
R='elo-r18fresh-live-bounded-dsf1-20260516b'
ROOT=f'tournaments/curvytron/{T}/ratings/{R}'
SHARD_SUFFIX='shards/shard-000000-games-000000-000020/summary.json'
v=modal.Volume.from_name(VOL)

def read_json(path): return json.loads(b''.join(v.read_file(path)))
def pct(xs,p):
    ys=sorted(xs); k=(len(ys)-1)*p/100; f=math.floor(k); c=math.ceil(k)
    return ys[f] if f==c else ys[f]*(c-k)+ys[c]*(k-f)
def corr(xs,ys):
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(vx*vy)

completed=[]
for i in range(100):
    try:
        res=read_json(f'{ROOT}/rounds/round-{i:06d}/results.json')
        rat=read_json(f'{ROOT}/rounds/round-{i:06d}/ratings.json')
    except Exception:
        continue
    if res.get('pair_summary_refs'):
        completed.append((i,res,rat))

latest_rank={r['checkpoint_id']: int(r['rank']) for r in completed[-1][2]['ratings']}
rank_by_round={i:{r['checkpoint_id']: int(r['rank']) for r in rat['ratings']} for i,_,rat in completed}

def load(task):
    i, pair_ref = task
    sh=read_json(pair_ref.replace('/battle.json', f'/{SHARD_SUFFIX}'))
    return i, sh

tasks=[(i,ref) for i,res,_ in completed for ref in res['pair_summary_refs']]
games=[]
with ThreadPoolExecutor(max_workers=32) as ex:
    for f in as_completed([ex.submit(load,t) for t in tasks]):
        i, sh=f.result()
        for g in sh.get('games', []):
            if not g.get('ok'): continue
            score=g.get('score') or {}
            ids=[p.get('checkpoint_id') for p in g.get('players', []) if isinstance(p,dict)]
            games.append({
                'round': i,
                'steps': int(g.get('physical_steps') or score.get('physical_steps')),
                'max_steps': int(score.get('max_steps')),
                'terminal_reason': score.get('terminal_reason'),
                'ids': ids,
                'round_ranks': [rank_by_round[i].get(pid) for pid in ids],
                'latest_ranks': [latest_rank.get(pid) for pid in ids],
            })

by_round=defaultdict(list)
for g in games: by_round[g['round']].append(g)

print('completed_rounds', [i for i,_,_ in completed], 'games', len(games))
for i in sorted(by_round):
    xs=[g['steps'] for g in by_round[i]]
    print(i, len(xs), round(statistics.fmean(xs),2), statistics.median(xs),
          round(pct(xs,90),1), round(pct(xs,95),1), round(pct(xs,99),1), max(xs))

def group(pred):
    a=[g['steps'] for g in games if pred(g)]
    b=[g['steps'] for g in games if not pred(g)]
    return len(a), round(statistics.fmean(a),3), statistics.median(a), len(b), round(statistics.fmean(b),3), statistics.median(b)

print('latest_top10', group(lambda g:any(r and r<=10 for r in g['latest_ranks'])))
print('latest_top50', group(lambda g:any(r and r<=50 for r in g['latest_ranks'])))
print('round_top10', group(lambda g:any(r and r<=10 for r in g['round_ranks'])))
print('round_top50', group(lambda g:any(r and r<=50 for r in g['round_ranks'])))
print('caps', sum(1 for g in games if g['steps'] >= g['max_steps']))
print('terminal_reasons', Counter(g['terminal_reason'] for g in games).most_common())
PY
```
