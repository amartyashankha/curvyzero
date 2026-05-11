# LightZero Eval Optimizer Handoff - 2026-05-10

Coach has already handled the simple eval-speed and artifact-access fixes:

- Stock-only eval path for the serious gate.
- Lightweight telemetry profile mode.
- Skip broad eval-root listing.
- One-root fetch/summarize pattern for artifacts.

Optimizer should own the deeper speed work now. The current evidence is useful but slow: full Wave11 serious stock-only rand16 results showed a survival lift in 3/5 runs, but evals still take minutes.

Exact asks:

- Profile MCTS-per-step cost and identify whether search, env stepping, checkpoint load, or artifact movement dominates wall time.
- Check whether model/env setup can be reused across seeds and checkpoints instead of cold-starting each eval.
- Check whether multiple seeds can run inside one warm Modal container without losing useful parallelism.
- Decide whether search sims can be reduced for telemetry runs while keeping the serious gate at 50 sims.
- Decide whether artifact root manifests should always be written so fetch/summarize does not need directory discovery.
- Evaluate whether eval should become one Modal map over checkpoint/seed with one manifest bundle at the end.

Ownership split: Coach owns signal interpretation, gate definitions, and simple operational cleanup. Optimizer owns runtime profiling, container reuse, eval parallelism shape, artifact manifest design, and the next concrete implementation plan for faster eval throughput.

Update: optional eval phase timing is now implemented behind
`--optimizer-phase-timing` on both `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py`
and `scripts/lightzero_live_eval_queue.py`. Defaults are unchanged. When enabled,
result JSON includes `optimizer_phase_timing_sec` buckets for checkpoint load,
config setup, manual policy/env setup and rollout when used, stock evaluator
setup/eval, artifact write, and Volume commit.

Worker B cleanup, 2026-05-10:

- Pong and CurvyTron `--summary-only` output now stops after compact survival
  TSV tables and artifact refs. This directly addresses the operator complaint:
  eval output should be a compact survival table, not a giant blob that needs
  interpretation.
- CurvyTron visual survival eval has a `gpu-l4-t4-cpu40` function/compute
  choice in `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py`
  and defaults to that compute posture for serious evals.
- The remaining speed bottleneck is architectural: each checkpoint/seed remote
  worker still reloads the checkpoint from the Modal Volume, builds the
  LightZero policy/env, runs MCTS, writes its own artifact, and commits the
  Volume. Bigger work belongs here: profile whether a checkpoint-batch worker,
  warm-container model/env reuse, or a separate manifest/index writer can reduce
  repeated checkpoint/env setup without giving up checkpoint x seed fan-out.

Worker C cleanup, 2026-05-10:

- `scripts/lightzero_live_eval_queue.py` now defaults to 16 sampled eval seeds,
  matching the operator doc's serious fresh-seed eval wave. This increases the
  default checkpoint x seed panel while still grouping seeds inside each Modal
  eval command.
- Direct Pong and CurvyTron Modal eval entrypoints now default
  `quiet_framework_logs=True`. The queue already passed quiet logs by default;
  this makes direct CLI calls less noisy unless the operator explicitly passes
  `--no-quiet-framework-logs`.
- No deeper Volume read/write restructuring was attempted. The observed safe
  boundary is still one checkpoint/seed worker reading its checkpoint, writing
  one artifact, and committing the Volume, plus one root manifest writer per
  grouped Modal call.
