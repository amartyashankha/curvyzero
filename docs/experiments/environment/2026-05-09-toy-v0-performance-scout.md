# 2026-05-09 Toy-v0 Performance Scout

## Question

Where does the current simplified `curvyzero-v0` single-env smoke spend time,
and what shape constraints should future vector/JAX/GPU work preserve?

This is not a CurvyTron source-fidelity benchmark. It is a local scout on the
toy-v0 environment and benchmark scaffold.

## Setup

- Machine: local macOS development machine.
- Platform probe: `macOS-15.6-arm64-arm-64bit`, `arm64`.
- Python: `Python 3.11.14`.
- NumPy: `2.4.0`.
- Git revision: unavailable because this checkout has no commit revision
  visible to `git rev-parse --short HEAD`.
- Package path: `PYTHONPATH=src`.
- Benchmark script: `scripts/benchmark_env.py`.
- Env config in benchmark: `CurvyTronConfig(action_repeat=1)`.

## Commands

```sh
python3 --version
python3 -c "import numpy; print(numpy.__version__)"
PYTHONPATH=src python3 -c "from curvyzero.env import CurvyTronConfig, CurvyTronEnv; print(CurvyTronConfig().rules_hash); print(CurvyTronEnv().agents)"
PYTHONPATH=src python3 scripts/benchmark_env.py --episodes 100 --max-steps 500 --format json
PYTHONPATH=src python3 scripts/benchmark_env.py --episodes 1000 --max-steps 500 --format json
PYTHONPATH=src python3 scripts/benchmark_env.py --episodes 1000 --max-steps 2000 --format json
PYTHONPATH=src python3 - <<'PY'
import cProfile
import io
import pstats
from scripts.benchmark_env import run
prof = cProfile.Profile()
prof.enable()
summary = run(seed=0, episodes=500, max_steps=500)
prof.disable()
print({k: summary[k] for k in ('episodes', 'steps', 'elapsed_sec', 'steps_per_sec')})
stream = io.StringIO()
pstats.Stats(prof, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(18)
print(stream.getvalue())
PY
```

## Results

Baseline import/config probe:

```text
Python 3.11.14
NumPy 2.4.0
CurvyTronConfig().rules_hash = b8446844bd278765
CurvyTronConfig(action_repeat=1).rules_hash = d1aab3da8c983fc4
agents = ['player_0', 'player_1']
```

Sequential smoke runs:

| Command shape | Steps | Elapsed sec | Steps/sec | Step sec | Action sec | Reset sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `--episodes 100 --max-steps 500` | 2,370 | 0.0761 | 31,131.4 | 0.0698 | 0.0042 | 0.0016 |
| `--episodes 1000 --max-steps 500` | 23,423 | 0.6699 | 34,962.5 | 0.6202 | 0.0345 | 0.0115 |
| `--episodes 1000 --max-steps 2000` | 23,423 | 0.9480 | 24,707.3 | 0.8344 | 0.0828 | 0.0265 |

The last two runs reached the same step count because episodes usually ended
before 500 steps. The slower `max_steps=2000` run is local noise or measurement
side effect, not a behavior change.

Profiler scout on 500 episodes:

```text
summary: episodes=500, steps=11785, elapsed_sec=0.6264, steps_per_sec=18814.3

11785 calls core.py:65(step)            cumulative 0.580s
11785 calls core.py:96(_physics_tick)   cumulative 0.438s
11785 calls core.py:156(_draw_segments) cumulative 0.327s
23035 calls core.py:163(_mark_segment)  cumulative 0.311s
23035 calls numpy linspace              cumulative 0.094s
79024 calls core.py:169(_mark_cell)     cumulative 0.055s
12285 calls core.py:176(_observations)  cumulative 0.049s
72210 calls core.py:38(agents)          cumulative 0.033s
```

## Interpretation

The current toy-v0 env is already partly array-shaped: positions, headings,
alive flags, death ticks, and occupancy are NumPy arrays. That is good future
prep.

The current hot path is still single-env Python:

- `step` builds dict outputs each call.
- `agents` rebuilds `["player_0", "player_1"]` many times.
- `_draw_segments` loops over players and calls `_mark_segment`.
- `_mark_segment` calls `np.linalg.norm`, `np.linspace`, then marks cells one at
  a time in Python.
- `_observations` concatenates a tiny global vector and copies it once per agent.

The first useful benchmark split should measure trail/occupancy writes,
observation generation, and wrapper/dict overhead separately. Do not optimize the
current toy-v0 implementation as production code. Use these numbers to design
the next benchmark manifest and future state layout.

## JAX And GPU Shape Notes

The external source pass supports the existing project stance:

- JAX `vmap` works best when the transition is pure array code with explicit
  batched axes. Source: [JAX vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)
  and [JAX automatic vectorization](https://docs.jax.dev/en/latest/automatic-vectorization.html).
- JAX `lax.scan` fits fixed-length rollout loops where the carry has the same
  shape and dtype every step. Source: [JAX lax.scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html).
- JAX random code requires explicit keys. Future print gaps, holes, spawn
  variation, and domain variation should carry per-env RNG state instead of
  hidden global random state. Source: [jax.random](https://docs.jax.dev/en/latest/jax.random.html)
  and [JAX pseudorandom numbers](https://docs.jax.dev/en/latest/random-numbers.html).
- Mctx search is JAX-native, JIT-friendly, and batched, but its MuZero
  `recurrent_fn` uses a learned embedding, not the real simulator state. The real
  CurvyTron rollout env does not need to become JAX immediately. Source:
  [Mctx README](https://github.com/google-deepmind/mctx).
- EnvPool is evidence that CPU C++/threadpool batched environment execution can
  be a serious later path if Python stepping bottlenecks. That is a later option,
  not a current rewrite. Sources: [EnvPool paper](https://papers.nips.cc/paper_files/paper/2022/hash/8caaf08e49ddbad6694fae067442ee21-Abstract-Datasets_and_Benchmarks.html)
  and [EnvPool docs](https://envpool.readthedocs.io/).
- gymnax shows the clean JAX env shape: explicit state, params, RNG key, `jit`,
  `vmap`, and `scan`. Use the pattern, not the dependency. Source:
  [gymnax README](https://github.com/RobertTLange/gymnax).
- Brax shows that accelerator simulation can work for the right problem, but its
  README warns that its env side is no longer the direction to copy. Use the
  fixed-shape/JAX pattern, not Brax itself. Source:
  [Brax README](https://github.com/google/brax).

## CurvyTron-Specific Shape Rules To Preserve

- Fixed max players per run profile, with masks for absent/dead players.
- Fixed map/grid size per run profile.
- Structure-of-arrays state for positions, headings, alive flags, scores,
  body/trail counters, print/gap state, and RNG.
- Occupancy grid and/or fixed trail/body buffers with owner, age, and active
  masks, not variable Python lists in the future hot path.
- Two-phase collision/trail writes so same-tick deaths, head-head cases, and
  death trail policy stay explicit.
- Batch-friendly observation generation. Avoid per-agent Python crop/rotation
  work in the eventual backend.
- Equivalence gates before speed claims: source fixtures, common trace diffs,
  config/rules hash, observation schema hash, and benchmark manifest.

## Next Task

Benchmark manifest work is now in `scripts/benchmark_env.py`. Do honest split
instrumentation or isolated microbenchmarks next, not a backend rewrite. The
smallest useful next measurement is a focused benchmark that can time:

- movement only;
- collision lookup/write;
- segment/trail rasterization;
- observation generation;
- dict/wrapper output construction;
- reset/autoreset.
