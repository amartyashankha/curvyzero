# Optimizer Working Memory

Date: 2026-05-10

Status: active optimizer lane front door.

Current launcher truth, 2026-05-11 late: Coach canonical CurvyZero training
uses
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode two-seat-selfplay`.
Fixed/frozen-opponent stock `train_muzero` runs are controls/profiling lanes
only. Stock LightZero in-loop eval is separate from CurvyZero checkpoint
eval/inspection/GIF jobs.

Current render truth, 2026-05-12: canonical two-seat self-play defaults to
`two_seat_trail_render_mode=browser_lines`. That route renders source-state RGB
browser-style lines at 704x704, converts/downsamples to gray64, and stacks the
64x64 policy tensor. `body_circles_fast` is an explicit speed comparison mode.
The same two-seat runner exposes `two_seat_death_mode=profile_no_death` for
optimizer long-survival profiles only.

Fresh render bottleneck read, 2026-05-12: long no-death profiles show render
redraw dominating policy/search once trails grow. Current research lives in
[CurvyTron render optimization research](render_optimization_research_2026-05-12.md).
The local render microbench is `scripts/benchmark_render_lane_microbench.py`;
use it for renderer iteration before launching full training profiles.
The active surface now renders 704x704 RGB and downsamples to 64x64, so older
direct-64 numbers are stale shape evidence only. Reprofile this active surface
before choosing render optimizations.

Fresh render optimization landing, 2026-05-12: the two-seat stack now reuses
one shared trail render for safe `P=2` player perspectives and falls back to
independent renders for unsafe palettes. Focused tests pass. The new granular
microbench says stack copy and downsample are small compared with long trail
redraw; next renderer work should target incremental/static trail rendering or
direct-luma drawing.

Profiling artifact hygiene, 2026-05-12: default Coach runs still create the GIF
browser marker and background GIF artifacts. Optimizer profiling commands should
use `--no-background-gif-enabled`; that now also suppresses the
`show_in_gif_browser.flag` marker so profile runs do not clutter the website.

This lane owns speed/training-loop setup synthesis: how CurvyTron visual
LightZero-style stacked-frame rollouts, sidecar scalar diagnostics,
policy/search, replay, reset/autoreset, Modal jobs, and learner boundaries
should be measured together.

This lane does not own environment fidelity, vector parity, or training-quality
claims. Environment claims stay in [environment active lanes](../environment/active_lanes.md).
Training and policy-quality claims stay in the [training state index](../training_state_index_2026-05-09.md).

## Start Here

- [Lane contract](lane_contract_2026-05-10.md) - separation of responsibility
  between Optimizer, Coach, and Environment/RAM reconstruction.
- [Current status](current_status_2026-05-09.md) - short optimizer read.
- [CurvyTron native LightZero profile](curvytron_native_lightzero_profile_2026-05-11.md) -
  archived/current-control source-state visual `train_muzero` timing, renderer
  fix, telemetry stride, MCTS/search read, and next bottlenecks. It is not the
  Coach canonical launcher.
- [CurvyTron render optimization research](render_optimization_research_2026-05-12.md) -
  current two-seat render bottleneck evidence, cost model, and optimization
  menu while Environment Reconstruction stabilizes richer visuals.
- [Runtime verdict](runtime_verdict_2026-05-10.md) - compact CurvyTron source
  path, CPU/GPU boundary, current profile, Modal Mctx evidence, and near-term
  architecture stance.
- [World model](world_model_2026-05-09.md) - what this project is and where
  optimizer fits.
- [Setup synthesis](setup_synthesis_2026-05-09.md) - LightZero/custom env
  setup read.
- [Framework working hypotheses](framework_decision_2026-05-09.md) - current
  base-runner hypotheses, LightZero control stance, and evidence gates.
- [Actor-loop architecture](actor_loop_architecture_2026-05-09.md) -
  framework-agnostic training-loop pieces.
- [Measurement plan](measurement_plan_2026-05-09.md) - Amdahl-style buckets,
  reports, and optimization gates.
- [MuZero loop bottleneck map](muzero_loop_bottleneck_map_2026-05-09.md) -
  actor-loop bucket map for self-play, search, replay, learner, checkpoint/eval,
  and policy freshness.
- [LightZero Modal loop](lightzero_modal_loop_2026-05-09.md) - current
  stock-ish Pong control loop, CPU/GPU split, Amdahl buckets, and disaggregation
  gates.
- [Profile report contract](profile_report_contract_2026-05-09.md) - shared
  timing and metadata report shape for repo-native and LightZero lanes.
- [Profile next steps](profile_next_steps_2026-05-09.md) - immediate report
  sequence and microbench matrix.
- [Blockers](blockers_2026-05-09.md) - what prevents production-speed or setup
  claims.
- [Profiling log](profiling_log_2026-05-09.md) - compact timing evidence and
  next measurement placeholders.
- [Questions](questions.md) - decisions this lane should answer.
- [Backlog](backlog.md) - small synthesis tasks.

## Key Inputs

- [Documentation map](../../README.md)
- [Training state index](../training_state_index_2026-05-09.md)
- [Environment active lanes](../environment/active_lanes.md)
- [Training-loop bottlenecks and Amdahl's law](../../research/training_loop_bottlenecks_amdhals_law_2026-05-09.md)
- [Self-play speed lane](../environment/selfplay_speed_lane_2026-05-09.md)

## Guardrails

- Primary CurvyTron training target is visual LightZero-style stacked frames.
  Do not treat scalar-ray `[B,2,106]` rows as the main coach-facing optimizer
  target unless new evidence explicitly justifies that switch.
- CurvyTron visual profiling is non-ALE. The canonical Coach launcher is
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode two-seat-selfplay`.
  The `env_variant=source_state_fixed_opponent` stock `train_muzero` surface is
  now controls/profiling only. Its source-state visual stack
  `curvyzero_source_state_gray64_stack4_player_perspective/v1` is a
  source-state geometry tensor, not an ALE path and not a browser/canvas pixel
  claim. The old `debug_visual_tensor` /
  `curvyzero_debug_occupancy_gray64/v0` surface is historical smoke plumbing.
- Current scalar-ray profiling is diagnostic sidecar evidence:
  `CurvyTronSourceEnv -> [B,2,106]` rows, not Atari/ALE and not a real
  LightZero visual CurvyTron env. The rows are trainer-wrapper diagnostics, not
  native CurvyTron source objects and not the primary observation contract.
- Scalar-ray sidecar observation shape: `24` ray directions * `4` channels plus
  `10` scalars equals `106` `float32` values per ego.
- Treat speed numbers as setup/runtime evidence unless they include the actor
  loop and comparison-valid payloads.
- Every speed number must say whether it includes env step, render,
  stack/normalize, policy/search, replay, and reset.
- Treat return curves, checkpoint quality, and reproduction status as coach-lane
  inputs, not optimizer outputs.
- Treat source-fidelity claims as Environment/RAM-lane inputs, not optimizer
  outputs.
- Treat Modal/JAX/Mctx runs as boundary evidence until real CurvyTron rollouts,
  replay, final observations, and trainer contracts are wired.
- Full GPU env/obs/model/search has no known fundamental blocker, but the
  current source env is a CPU object graph; a GPU env/obs rewrite needs a new
  tensor runtime plus parity tests.
- Be precise about GPU claims. The current native LightZero GPU entrypoint puts
  model inference and learner tensors on CUDA. MCTS is LightZero-native, but it
  is a CPU/C++ tree with CUDA model calls, not a pure CUDA tree-search kernel.
  CurvyTron env reset/step/render/stack and replay storage are CPU/NumPy today.
- Do not promote GPU env work, native rewrites, distributed actors, or larger
  batches until wall-clock shares and p95/p99 action latency justify them.
- Link to source evidence; do not duplicate experiment logs here.
- For optimizer speed profiles, prefer `--mode profile` with
  `--disable-death-for-profile` when the question is steady-state long-game
  cost. Bad early policies otherwise hide env/search/replay work.
