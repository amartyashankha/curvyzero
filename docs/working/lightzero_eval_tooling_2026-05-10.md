# LightZero Eval Tooling - 2026-05-10

Use `scripts/lightzero_live_eval_queue.py` for live checkpoint polling. It
lists visible `iteration_*.pth.tar` checkpoints in the Modal Volume, checks
which eval output dirs already exist, and prints eval commands only for pending
checkpoint/seed eval jobs.

User-facing complaint this tooling must keep fixed: eval output should be a
compact survival table, not a giant JSON blob that needs interpretation. In
`--summary-only` mode, CurvyTron and Pong eval CLIs print the aggregate
survival TSV under `# aggregate_by_checkpoint`, then per-checkpoint/per-seed
survival rows under `# per_checkpoint_seed_curve`, and then return without
dumping the full result JSON.

Scope: this page is for the stock visual Pong eval control lane. Do not mix its
survival-time proof claims with CurvyTron scalar contract checks or adapter
smokes.

## Clean Operator Path

Use this path for normal Pong eval artifact access. The queue records the eval
seed sampling lines, the fetch command gets one complete eval root once, and
the summarizer reads both root `manifest_*.json` files and raw per-episode
JSONs under `iteration_*` dirs without double-counting the same artifact.

| date | lane | operator row |
| --- | --- | --- |
| 2026-05-10 | Pong stock eval artifact access | Launch with `--eval-seed-count` and record the printed `# eval seed sampler seed:` plus `# eval seeds:` lines; after eval completion run the printed `modal volume get` once for the whole eval root; run the printed summarizer command over `<local-root>/**/*.json`; paste one concise result row into the run note. |
| 2026-05-10 | Eval hot path cleanup | Pong and CurvyTron scoring jobs no longer SHA256-hash the checkpoint in the hot eval path after `torch.load`; artifacts still record checkpoint path, existence, byte size, strict-load status, and per-job output refs. Use the checkpoint probe if a checksum is needed. |
| 2026-05-10 | CurvyTron slim root manifests | CurvyTron visual survival eval root manifests now default to slim mode: the combined `manifest_*.json` keeps summary tables, artifact refs, `results_omitted: true`, and `result_count`, while full per-job JSON artifacts remain unchanged. Use `--no-slim-manifest` only for a debug root manifest that must embed every per-job result object. |
| 2026-05-10 | CurvyTron eval table/batch cleanup | CurvyTron visual survival eval now uses eval-local `batch_size=64` by default instead of inheriting the training batch size. `--summary-only` aggregate rows label `ok_count`, `capped_count`, `failure_count`, and `mean_elapsed_sec`, and per-seed rows include `elapsed_sec` before the artifact ref. |

Launch a serious fresh-seed eval wave:

```sh
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id <run-id> \
  --attempt-id <attempt-id> \
  --eval-id <eval-id> \
  --compute gpu-l4-t4-cpu40 \
  --eval-pass custom \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --step-detail-limit 8 \
  --eval-seed-count 16 \
  --eval-speed-profile serious \
  --update-per-collect -1 \
  --stock-only \
  --optimizer-phase-timing \
  --group-size 4 \
  --max-parallel-launches 64 \
  --slim-manifest \
  --execute
```

Immediately record these two queue output lines in the run note:

```text
# eval seed sampler seed: <sampler-seed>
# eval seeds: <comma-separated eval seeds>
```

After the eval finishes, use the two commands printed by the queue. They should
have this shape:

```sh
uv run --extra modal modal volume get curvyzero-runs \
  training/lightzero-official-visual-pong/<run-id>/attempts/<attempt-id>/eval/<eval-id> \
  artifacts/local/lightzero-eval-manifests/<eval-id> \
  --force

uv run python scripts/summarize_lightzero_pong_eval_manifest.py \
  --survival-curve \
  --survival-aggregate \
  --format tsv \
  'artifacts/local/lightzero-eval-manifests/<eval-id>/**/*.json'
```

Naming rules:

- Eval ids should say the attempt, checkpoint range or `live`,
  `survival2048`, `stockeval`, and whether the seed list is fresh random,
  replayed, or telemetry.
- A completed checkpoint/seed artifact dir is named
  `iteration_<N>_custom_steps2048_seed<S>`.
- Use one local fetch dir per eval id:
  `artifacts/local/lightzero-eval-manifests/<eval-id>`.
- Quote the `**/*.json` summarizer argument. The summarizer expands it itself,
  so this works even when the shell does not expand recursive globs.

Do not do this anymore:

- Do not repeatedly list the Volume while trying to discover individual JSON
  files. Fetch the complete eval root once, then inspect locally.
- Do not fetch partial `iteration_*` roots unless you are debugging a specific
  corrupt artifact.
- Do not treat a missing or incomplete combined manifest as a failed eval. The
  raw per-episode JSONs under `iteration_*` are valid summary input.
- Do not launch a new claim wave against a standing fixed seed list. Use
  `--eval-seed-count`, record the sampler seed and sampled list, and reserve
  `--eval-seeds` for replay/debug.
- Do not quote results from another attempt's `iteration_0` baseline.

## Survival-first 2048 queue

Use a survival-first eval id. Keep the name plain: include the attempt, the
checkpoint range or `live`, `survival2048`, `stockeval`, and the seed.

Default live eval posture:

- `--compute gpu-l4-t4-cpu40`
- `--group-size 4` for multi-seed backlogs; use `--group-size 1` only when earliest first checkpoint signal matters more than launch overhead
- `--max-parallel-launches 64`
- `--eval-seed-count 16` for a fresh pseudo-random eval seed list
- `--eval-seeds <printed-list>` only for replay or manual debugging
- `--slim-manifest` for compact combined manifests; raw per-checkpoint JSONs
  still keep full detail
- strict no-fallback checkpoint load
- stock evaluator on, stock-only triage on by default
- survival time / steps survived first
- serious eval speed profile by default: 50 MCTS simulations per action
- add `--optimizer-phase-timing` when the question is eval runtime breakdown;
  omit it for ordinary claim waves if the extra JSON is not needed

Every new queue-driven eval wave samples a fresh pseudo-random seed list by
default. Record both printed lines: `# eval seed sampler seed:` and
`# eval seeds:`. Do not tune against one standing fixed seed list. Use
`--eval-seed-rng-seed <printed-sampler-seed>` only to reproduce sampling, and
use `--eval-seeds` only when replaying a recorded wave or doing a manual debug
check.

Fixed eval seed lists are replay/debug tools, not the default for new claim
waves.

Dry run:

```sh
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath \
  --eval-id detached-s0-live-survival2048-stockeval-seed0 \
  --compute gpu-l4-t4-cpu40 \
  --eval-pass custom \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --step-detail-limit 8 \
  --update-per-collect -1 \
  --eval-speed-profile serious \
  --stock-only \
  --eval-seed-count 16 \
  --group-size 4 \
  --max-parallel-launches 64
```

The printed eval commands are strict by default:

- `--no-allow-model-fallback`
- `--run-stock-evaluator`
- `--stock-only`
- one Modal app call per pending checkpoint group; missing eval seeds for that group are passed as one `--eval-seeds` list and mapped remotely with `Function.starmap`
- slim combined manifests by default; use raw per-checkpoint artifacts for
  full detail
- duplicate skip based on the expected eval output dir

## Why Evals Take Minutes

The slow path is not stdout or JSON formatting. Stock-only removes the duplicate
manual rollout, but the remaining stock evaluator still does real LightZero
Atari work:

- Modal has to start the app/function container, attach the image, and mount the
  `curvyzero-runs` Volume. Grouping checkpoints and seeds reduces repeated app
  startup, but every remote worker still starts a function container.
- Each checkpoint/seed job loads the checkpoint from the Modal Volume, patches
  and compiles the LightZero config, builds a MuZero policy, strict-loads the
  model, and creates a real ALE-backed Pong env. Grouping seeds inside one
  Modal app call reduces local/app launch overhead, but each checkpoint/seed
  still does this setup in its own remote function call.
- The eval scoring hot path used to load the checkpoint and then SHA256-hash
  the same checkpoint file for artifact metadata. That extra full-file Volume
  read has been removed from Pong and CurvyTron eval jobs; checksum probing is
  still available in the checkpoint probe path.
- Every env step asks MuZero for an action. With `--num-simulations 50`, that is
  50 MCTS simulations per environment step. A 2048-step cap is therefore on the
  order of 102,400 tree-search simulations for one stock episode before
  counting env stepping, model inference, and evaluator overhead.
- The old high-throughput command ran both a manual rollout and the stock
  `lzero.worker.MuZeroEvaluator` rollout. That duplicated the expensive
  policy/env/MCTS path for the same checkpoint/seed. Stock-only removed that
  duplication, but it did not reduce the cost inside the remaining stock
  evaluator episode.
- Each remote job writes a per-checkpoint artifact and commits the Modal Volume.
  The parallel entrypoint then writes a root manifest and commits again.

So "one eval" is closer to "load a model, boot ALE Pong, run a capped MuZero
episode with MCTS, write durable artifacts" than to "read a scalar metric."

Plain timing model:

```text
wall time ~= Modal startup
          + checkpoint/env/policy setup per checkpoint/seed
          + stock_steps_survived * num_simulations * cost_per_MCTS_simulation
          + Volume artifact writes/commits
```

The multiplicative term is the important one. If a checkpoint survives to the
2048-step cap, `--num-simulations 5` is roughly one tenth of the tree-search
work of `--num-simulations 50`. It is a faster telemetry read, not the same
quality gate.

Optimizer handoff: this cleanup removed one redundant checkpoint checksum read
from the hot eval job. The remaining bottleneck is still the per
checkpoint/seed remote job: checkpoint load, config compile, policy/env
construction, ALE boot, and `stock_steps_survived * num_simulations` MuZero
search. The next real optimizer project is to amortize that setup across seeds
or checkpoints inside a long-lived worker, then replace per-job Volume commits
with a batched artifact writer. Keep that separate from readability cleanup
because it changes failure, retry, and artifact-index semantics.

CurvyTron note: Modal rejected a one-function `cpu=64` eval attempt with:
`Function CPU request out of bounds. Must be between 0.125 and 40 cores.` Keep
CurvyTron eval on `gpu-l4-t4-cpu40`; scale by launching many independent
checkpoint/start workers instead of asking one worker for more CPUs.

## Fast Triage Versus Debug Eval

Use stock-only triage for broad eval waves. It skips the duplicate manual
rollout and keeps the stock evaluator episode, so `stock_steps_survived`,
`stock_return`, stock reward counts, strict-load status, artifacts, and
manifests remain the lead readout. The queue helper defaults to this mode and
passes `--stock-only`.

This is the current default because the old high-throughput path was doing both
manual rollout and stock evaluator work for each checkpoint/seed, doubling the
expensive policy/env/MCTS path. Stock-only eval is enough for the queue's
survival-first triage, but stock-only is still slow when the cap and search
budget are high.

Use `--eval-speed-profile telemetry` for cheap live trend checks. It keeps the
same stock-only artifact contract and preserves `stock_steps_survived`, but
defaults to `--num-simulations 5`. Treat those rows as "is the run alive and
moving?" telemetry, not a final claim about policy quality. Use a separate
`--eval-id` containing `telemetry` so duplicate filtering does not mix lower
search artifacts with serious gate artifacts.

Use `--eval-speed-profile serious` for gates and quoted results. It defaults to
`--num-simulations 50`, keeps `--max-eval-steps` and
`--max-episode-steps` matched, and is the setting to use before claiming a
checkpoint survived a target cap.

Use full manual+stock debug eval only when investigating parity, action traces,
manual/stock disagreement, or observation/action-mask details. Disable the fast
path with `--no-stock-only` on `scripts/lightzero_live_eval_queue.py`, or omit
`--stock-only` when calling
`curvyzero.infra.modal.lightzero_pong_eval_smoke` directly. Expect the debug
mode to be much slower because it runs the manual rollout and then runs the
stock evaluator rollout.

What is parallel today:

- Checkpoint fan-out: yes. The queue helper finds pending checkpoints and
  prints one Modal eval command per checkpoint group. The default
  `--group-size 4` reduces repeated Modal app startup and manifest noise for
  multi-seed waves.
- Local Modal call fan-out: yes. With `--execute`, the queue helper keeps up to
  `--max-parallel-launches` local `modal run` commands in flight.
- Remote Modal function fan-out: yes inside each Modal command when a command
  contains multiple checkpoint refs or multiple eval seeds; the eval CLI expands
  the checkpoint x seed grid and submits it with `eval_fn.starmap(...)`, so
  those jobs run as separate Modal function calls.
- Eval starts/seeds: yes through the queue helper. `--eval-seed-count 16`
  samples a fresh seed list and expands every selected checkpoint into separate
  checkpoint/seed eval jobs inside the Modal call for that checkpoint group.
  Use `--eval-seeds` only to replay a recorded list or debug by hand.
- Episodes per checkpoint/seed: no. Each checkpoint/seed job still runs one
  stock evaluator episode in the default stock-only triage mode. Full
  manual+stock debug mode runs one manual eval episode and one stock evaluator
  episode. Use more eval seeds for more starts.

The queue helper keeps its own preflight output compact by default. It prints
counts and the checkpoint range instead of dumping every visible checkpoint or
existing eval dir. Add `--verbose-listings` only when debugging Modal Volume
visibility.

Volume access hygiene: every normal queue run lists the checkpoint dir once and
lists the eval root once for resume-safe duplicate filtering. For a known-new
eval id, add `--skip-eval-root-listing` to save the eval-root `volume ls` call.
Use it only when duplicate skipping is not needed; it intentionally treats every
selected checkpoint/seed pair as pending. Do not use it for reruns, partial
panels, or monitoring an eval id that may already contain artifacts.

The queue helper also passes `--summary-only` to Modal eval commands by
default. In that mode, the Modal eval CLI suppresses the full per-checkpoint
result JSON on stdout and prints only summary JSON: the compact table plus
manifest/artifact refs. Full per-checkpoint eval artifacts are still written.
Use `--no-summary-only` on the queue helper, or omit `--summary-only` when
calling `curvyzero.infra.modal.lightzero_pong_eval_smoke` directly, only when
you intentionally want full stdout dumps containing results/actions/step
records.

The queue helper also passes `--slim-manifest` by default. That keeps combined
root `manifest_*.json` files small: they contain the compact summary table,
artifact refs, and result count, but not the full per-job result payloads. The
raw `iteration_*/*.json` artifacts are still full-detail and remain the source
of truth. Use `--no-slim-manifest` only for a narrow debug run where a single
root manifest must embed every result object.

The queue helper also passes `--quiet-framework-logs` by default. That redirects
Python-level LightZero/Gym stdout and stderr around each checkpoint eval, but it
does not change checkpoint loading, policy actions, rewards, stock evaluator
behavior, result dicts, artifacts, or manifests. Use `--no-quiet-framework-logs`
only when debugging framework logs.

Known eval-log noise: messages like `EOFError`, `invalid load key`, and
`Env 0 reset has exceeded max retries(1)` can appear after the stock evaluator
has already finished an episode and written an `ok: true` artifact. Treat those
as cleanup noise unless the raw per-checkpoint artifact has `ok: false`,
`stock_ok: false`, `strict_load: false`, or `fallback_used: true`.

Combined root manifests are convenience indexes. If a combined manifest is
missing a checkpoint row, fetch the whole eval root and recover from the raw
per-checkpoint files under `iteration_*/*.json`. Those raw artifacts are the
source of truth for completed checkpoint evals.

Repeated evals of the same checkpoint can happen when we reuse an eval id and
launch again for later checkpoints. The survival-curve summary now collapses
those duplicate checkpoint rows to the latest artifact it can identify, and it
adds:

- `duplicate_eval_count`
- `duplicate_stock_steps_survived_values`
- `duplicate_stock_steps_disagree`

If `duplicate_stock_steps_disagree=true`, do not silently quote only one
number. Record the disagreement and treat the run as needing a repeat or
multi-start eval before making a strong claim.

For stock/control reruns, `--update-per-collect -1` restores stock
`update_per_collect=None` in the eval config helper. The queue helper now uses
`-1` by default so this safer stock-like behavior is automatic, but explicit
flags in important commands are still fine.

## Embarrassingly parallel checkpoint evals

`scripts/lightzero_live_eval_queue.py` now supports size-1 or small grouped
Modal calls. The default is `--group-size 4`, which gives a better multi-seed
default because each app call can submit several checkpoint/seed jobs through
remote `Function.starmap` while writing one combined root manifest for the
group. Use `--group-size 1` for tailing a live run when fastest first checkpoint
signal matters more than repeated app startup.

It also supports eval-start fan-out with `--eval-seed-count`. Each new eval
wave should use a fresh pseudo-random seed list. Record the printed sampler
seed and final seed list in the run notes. Output directories already include
the seed, for example
`iteration_<N>_custom_steps2048_seed3`, so duplicate filtering is per
checkpoint/seed pair. Re-running the queue with the same eval id skips any
checkpoint/seed output dir already visible under the eval root. `--limit`
walks checkpoints first, then seeds, so small dry-run previews preserve the
complete seed set for the earliest pending checkpoints.

The helper samples unique integer eval seeds with Python `random.Random`.
If `--eval-seed-rng-seed` is omitted, the helper generates a sampler seed and
prints it. Pass `--eval-seed-rng-seed <int>` to reproduce the same sampled
list. For exact replay later, pass the printed final list back through
`--eval-seeds`.

As of the Modal eval-seed fanout update, multiple eval starts are mapped inside
one Modal app call instead of launching one local `modal run` per seed. The
queue helper groups pending checkpoint/seed pairs by checkpoint and missing
seed set, then prints or executes one Modal command with `--eval-seeds` for
that set. Inside `curvyzero.infra.modal.lightzero_pong_eval_smoke`, the local
entrypoint expands the checkpoint x eval-seed grid and submits the work with
`Function.starmap`.

This keeps the strict/load/artifact contract intact:

- strict no-fallback checkpoint load;
- stock evaluator on;
- one immutable artifact directory per `(checkpoint, eval_seed)`;
- root `manifest_*.json` files that the summarizer can merge;
- `stock_steps_survived` as the lead metric.

Official Modal docs recommend `Function.map` and `Function.starmap` for
independent repeated inputs, with `spawn_map` reserved for background batches
whose results are stored externally. Use blocking `starmap` here because the
local entrypoint writes a combined manifest after collecting per-job results.
Use `spawn_map` only after adding a separate manifest/index writer that does not
depend on returned results.

Modal source pages checked:

- https://modal.com/docs/reference/modal.Function
- https://modal.com/docs/guide/scale
- https://modal.com/docs/guide/volumes
- https://modal.com/docs/guide/resources

Sources checked:

- Modal scaling guide: `Function.map`, `Function.starmap`, pending input
  limits, and map concurrency limits.
- Modal Function reference: `map`, `starmap`, `for_each`, and `spawn_map`
  behavior.
- Modal batch processing guide: `spawn_map` for background jobs stored in a
  Volume/bucket/database.
- Modal Volumes guide/reference: commit/reload semantics and avoid concurrent
  writes to the same file.
- Modal Images guide: image layer cache reuse and rebuild behavior.
- Local examples:
  `/Users/shankha/modal-examples/03_scaling_out/basic_grid_search.py`,
  `/Users/shankha/modal-examples/06_gpu_and_ml/hyperparameter-sweep/hp_sweep_gpt.py`,
  and `/Users/shankha/modal-projects/flash-projects/modal-decagon/training/benchmarks/volume_throughput/archive/exp_large_pipeline.py`.

Toy validation on 2026-05-10 with four half-second Modal sleep jobs:

| pattern | wall time |
| --- | ---: |
| one `modal run` using `Function.map(count=4)` | ~4.1-4.3s |
| one `modal run` using `Function.starmap(count=4)` | ~4.1s |
| four separate `modal run` calls in parallel | ~5s |
| four separate `modal run` calls sequentially | ~19s |

The toy has much less import/checkpoint overhead than LightZero, so treat the
exact seconds as directional only. The important read is that one mapped app
call preserves parallel remote work while trimming repeated local/app startup.

The queue default is aggressive: `--compute gpu-l4-t4-cpu40`,
`--group-size 4`, and `--max-parallel-launches 64`. This changes only compute
allocation and queue/app-call fan-out. Checkpoint selection, loading, seed,
strict no-fallback behavior, stock evaluator execution, result files, and
metric meaning stay the same. The 64-wide ceiling is naturally bounded by
pending work.

CPU8/CPU16 eval guidance is stale. Keep those strings only when naming or
describing old artifacts. CPU64 was also wrong: Modal rejected `cpu=64` because
this workspace caps function CPU at 40 cores. New live eval waves should default
to CPU40 compute and 64-wide launch fan-out unless capacity friction forces a
temporary reduction.

Urgent large eval-wave command shape:

```sh
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id <run-id> \
  --attempt-id <attempt-id> \
  --eval-id <eval-id> \
  --compute gpu-l4-t4-cpu40 \
  --eval-pass custom \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --step-detail-limit 8 \
  --eval-seed-count 16 \
  --eval-speed-profile serious \
  --update-per-collect -1 \
  --stock-only \
  --group-size 4 \
  --max-parallel-launches 64 \
  --slim-manifest \
  --execute
```

With `--group-size 4`, that launches separate Modal eval calls per checkpoint
group and seed set. Each call maps its checkpoint x seed grid inside the remote
app, keeps the strict flags, and writes one root `manifest_*.json` for that
group, plus one per-checkpoint/per-seed artifact dir. The helper prints the
fetch and summary commands after the launch set. The printed summary command
uses `<local-root>/**/*.json`, so it can read root manifests and raw
per-episode JSONs from the same fetched root.

Cheap telemetry command shape:

```sh
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id <run-id> \
  --attempt-id <attempt-id> \
  --eval-id <eval-id>-telemetry \
  --compute gpu-l4-t4-cpu40 \
  --eval-pass custom \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --step-detail-limit 2 \
  --eval-seed-count 4 \
  --eval-speed-profile telemetry \
  --update-per-collect -1 \
  --stock-only \
  --group-size 8 \
  --max-parallel-launches 64 \
  --skip-eval-root-listing \
  --slim-manifest \
  --execute
```

That keeps the survival-time metric shape intact but cuts search work by using
5 simulations/action. Promote only promising checkpoints to the serious
profile before making a strong survival claim.

Omit `--skip-eval-root-listing` when resuming a telemetry eval id that may
already contain completed checkpoint/seed artifact dirs.

Print only the first few pending checkpoint/seed jobs:

```sh
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath \
  --eval-id detached-s0-live-survival2048-stockeval-seed0 \
  --compute gpu-l4-t4-cpu40 \
  --eval-pass custom \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --step-detail-limit 8 \
  --eval-seed-count 16 \
  --update-per-collect -1 \
  --group-size 4 \
  --max-parallel-launches 64 \
  --limit 16
```

Replay or debug one recorded seed list from one local process:

```sh
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath \
  --eval-id detached-s0-live-survival2048-stockeval-seed0 \
  --compute gpu-l4-t4-cpu40 \
  --eval-pass custom \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --step-detail-limit 8 \
  --eval-seeds <recorded-eval-seed-list> \
  --update-per-collect -1 \
  --group-size 4 \
  --max-parallel-launches 64 \
  --slim-manifest \
  --execute
```

Use a small `--group-size` such as 4 or 8 when Modal call overhead and stdout
noise are the bottleneck. Keep it modest for serious 16-seed waves, and keep
`--slim-manifest` on so combined manifests stay compact. Groups may include
multiple eval seeds through `--eval-seed-count` or replayed `--eval-seeds`;
output paths still stay one directory per checkpoint/seed. The helper still
writes commands with `--parallel`, so each Modal call emits a mergeable
manifest under the eval root. After calls finish, fetch the whole eval root and
run the summary command printed by the helper; the summarizer will read all
root `manifest_*.json` files together and can recover from raw per-checkpoint
JSON if a combined manifest misses a row.

Run pending evals with explicit lower fan-out only for small targeted checks,
manual debugging, or capacity friction. Mark that as a temporary capacity
choice, not as the default:

```sh
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath \
  --eval-id detached-s0-live-survival2048-stockeval-seed0 \
  --compute gpu-l4-t4-cpu40 \
  --eval-pass custom \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --step-detail-limit 8 \
  --eval-seeds <recorded-eval-seed-list> \
  --update-per-collect -1 \
  --max-parallel-launches 1 \
  --execute
```

For `iteration_<N>.pth.tar` and eval seed `<S>`, this 2048 recipe expects an
eval dir named:

```text
iteration_<N>_custom_steps2048_seed<S>
```

If that dir is already visible under the chosen eval id, that checkpoint/seed
job is not printed again.

## Fetch and summarize

After evals finish, use the fetch and summary commands printed at the end of
the queue helper output. Fetch the complete eval root once, then summarize the
local copy. The summary helper is:

```sh
uv run python scripts/summarize_lightzero_pong_eval_manifest.py \
  --survival-curve \
  --survival-aggregate \
  --format tsv \
  'artifacts/local/lightzero-eval-manifests/detached-s0-live-survival2048-stockeval-seed0/**/*.json'
```

`--survival-curve --survival-aggregate` now prints the checkpoint aggregate
table first under `# aggregate_by_checkpoint`, then the checkpoint curve under
`# checkpoint_curve`. Multiple eval seeds for the same checkpoint are collapsed
into one checkpoint row using the latest artifact per seed; the row shows
mean/min/max/latest stock survival near the front of the table, followed by the
cap, best/latest flags, seed list, deltas, returns, and artifact checks. The
helper de-duplicates rows with the same `artifact_ref`, so a recursive JSON
input can include both combined manifests and raw per-episode artifacts. Use
this as the first readout after fetching the root. The key columns are:

- `stock_steps_survived`
- `stock_steps_survived_mean`
- `stock_steps_survived_min`
- `stock_steps_survived_max`
- `stock_steps_survived_latest`
- `eval_seed_count`
- `eval_seeds`
- `delta_stock_steps_survived`
- `delta_previous_stock_steps_survived`
- `best_so_far_stock_steps_survived`
- `delta_best_so_far_stock_steps_survived`
- `run_best`
- `latest`
- `stock_survival_fraction`
- `eval_cap_steps`
- `steps_survived`
- `delta_steps_survived`
- `stock_return`
- `delta_stock_return`
- `stock_positive_reward_count`

Use the full manifest table only when debugging details such as action
histograms, load/fallback state, or manual/stock mismatch:

```sh
uv run python scripts/summarize_lightzero_pong_eval_manifest.py \
  --baseline-deltas \
  --sort-checkpoints \
  --format tsv \
  artifacts/local/lightzero-eval-manifests/detached-s0-live-survival2048-stockeval-seed0
```

Survival time / steps survived is the primary Pong signal. Score and return are
secondary context and should not lead the read.

For monitoring, fetch one whole eval root once, then summarize locally. Avoid
looping over individual checkpoint dirs or JSON files with repeated
`modal volume ls` / `modal volume get` calls:

```sh
uv run --extra modal modal volume get curvyzero-runs \
  training/lightzero-official-visual-pong/<run-id>/attempts/<attempt-id>/eval/<eval-id> \
  artifacts/local/lightzero-eval-manifests/<eval-id> \
  --force

uv run python scripts/summarize_lightzero_pong_eval_manifest.py \
  --survival-curve \
  --survival-aggregate \
  --format tsv \
  'artifacts/local/lightzero-eval-manifests/<eval-id>/**/*.json'
```

Do not compare across attempts unless the baseline is clearly documented. The
normal baseline is same-run `iteration_0`.

CurvyTron optimizer correction: scalar survival wrappers are contract checks
only. The next training blocker is visual `[4,64,64]` stacking plus a bounded
collect/search/replay/sample/learner profile. Adapter smoke is plumbing
evidence, not a full loop.

CurvyTron visual survival eval now has a `gpu-l4-t4-cpu40` compute option and
uses it by default. The older `cpu` and `gpu-l4-t4` variants remain available
for narrow smoke/debug checks, but serious multi-seed evals should use the
CPU40 GPU variant unless capacity friction forces a temporary downgrade.
Its combined root manifests are slim by default, matching the queue helper's
normal `--slim-manifest` behavior. Raw per-checkpoint `iteration_*/*.json`
artifacts still carry full result detail; use `--no-slim-manifest` for the rare
debug run that needs every per-job result embedded in the root manifest.

CurvyTron eval has its own default `batch_size=64` now. This is only the eval
policy/config batch size passed into the LightZero policy setup; it does not
change training defaults. Operators can still pass `--batch-size` explicitly
for smoke/debug parity. The compact stdout tables now make counts explicit:
aggregate rows use `ok_count`, `capped_count`, `failure_count`, and
`mean_elapsed_sec`, while per-checkpoint/per-seed rows keep artifact refs but
put `elapsed_sec` before the long path.

Active board pointer:
[docs/working/training_coach_active_board_2026-05-10.md](training_coach_active_board_2026-05-10.md).
