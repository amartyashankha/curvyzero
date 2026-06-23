# JAX/MCTX Toy Spike Critique - 2026-05-21

Scope: working memo only. No production code, trainer defaults, tournament
defaults, Modal training, or live-run state should change for this spike.

## Short Read

A tiny JAX/MCTX CurvyTron-like toy is worth doing now, but only as a hard
falsification probe for the next architecture phase. It should answer whether a
fixed-shape simultaneous-action line-game loop can keep state, observation,
model inference, and MCTX search batched/JIT/device-resident without collapsing
back into scalar Python/NumPy rows.

It should not be described as CurvyTron learning evidence, LightZero
replacement evidence, or visual-fidelity evidence.

## 1. What The Toy Would Test

The toy should test the system shape that stock LightZero currently cannot
preserve:

```text
state[B,2,...]
-> simultaneous step(actions[B,2])
-> obs[B,2,4,64,64]
-> roots[R=B*2]
-> tiny JAX model
-> mctx.gumbel_muzero_policy(...)
-> actions[R]
-> scatter back to actions[B,2]
```

Concrete questions:

- Can `jax.jit` compile a fixed-profile CurvyTron-like step/render/search loop
  with stable shapes and no surprise recompiles?
- Can a root batch of both seats, flattened to `R=B*2`, keep MCTX search fast
  enough to be interesting versus the current LightZero search bucket?
- Does MCTX tree memory remain boring with a vector latent such as `H=64` and
  modest simulations such as `8`, `16`, or `32`?
- Can row/player gather-scatter survive simultaneous actions, terminal masks,
  dead/padded roots, and legal-action masks without axis mistakes?
- Does the basic Pgx/Brax-style premise hold locally: large batch first,
  static shapes, compiled loop, coarse host readback only?

The toy's main win is not raw FPS. The win is learning whether the "full batch
stays alive through search" idea is technically plausible before spending more
effort on LightZero-compatible resident plumbing.

## 2. Minimal Setup

Use a deliberately false but CurvyTron-shaped game.

State:

- `pos[B,2,2]`, `heading[B,2]`, `alive[B,2]`, `trail[B,H,W]` or
  `trail[B,2,H,W]`.
- Fixed grid, fixed trail capacity or dense occupancy, no dynamic Python
  lists.
- Optional `age`/`step_count[B]` and `done[B]`.
- PRNG key as an explicit JAX array if any stochastic reset/spawn remains.

Actions:

- `A=3`: left, straight, right.
- Environment step consumes simultaneous `actions[B,2]`.
- Search consumes independent ego roots `action[R]`, then host/JAX scatter
  converts selected ego actions back to `actions[B,2]`.
- Padded/dead roots must carry a harmless valid mask and be ignored outside
  search. Do not feed all-invalid active rows to MCTX.

Observation:

- Start with direct learned observations, not browser-pixel parity.
- Shape: `obs[B,2,4,64,64]`, `float32` or `uint8` converted once inside the
  JAX profile.
- Include just enough visual semantics to stress axes: player-specific head
  colors or channels, trails, collision boundaries, and four-frame stack.
- Add asymmetric player fixtures so wrong player view and wrong row order are
  obvious.

Model/search:

- `representation(obs_roots[R,4,64,64]) -> hidden[R,H]`, with a tiny CNN or
  even a first-pass projection if the search API is the question.
- `prediction(hidden) -> prior_logits[R,3], value[R]`.
- `recurrent_fn(params, rng_key, action[R], hidden[R,H])` is pure JAX and
  returns reward, discount, prior logits, value, and next hidden.
- Use `mctx.gumbel_muzero_policy` first with fixed `num_simulations`,
  `max_depth`, `A=3`, and `invalid_actions[R,3]`.
- Time compile-plus-first-run separately from steady state with
  `.block_until_ready()`.

Minimum report:

- compile time, steady loop/search time, decisions/sec, simulations/sec;
- root count, active/padded roots, action histogram, finite action weights;
- memory snapshot or peak allocation if easy;
- evidence that outputs remain on device until the final small action readback;
- axis checks for `[B,2] -> [B*2] -> [B,2]` round trip.

## 3. What It Would Not Prove

- It would not prove CurvyTron can learn.
- It would not prove stock LightZero can use the result.
- It would not prove replay targets, value perspective, reward sign, or
  checkpoint semantics are correct.
- It would not prove browser-line visual fidelity, source-state parity, bonus
  rendering, RND latest-frame extraction, GIF/eval behavior, or tournament
  compatibility.
- It would not prove that independent per-seat search is the right multiplayer
  algorithm. It only proves that the approximation can be executed.
- It would not prove full-loop throughput unless replay, learner, checkpoint,
  eval, and artifact writes are later included.

The success label should be "JAX/MCTX resident batch/search viability," not
"MuZero training works."

## 4. Main Risks

Framework mismatch with PyTorch LightZero:

- MCTX requires pure JAX `representation`, `prediction`, and `dynamics`.
- A PyTorch LightZero model cannot sit inside a jitted MCTX `recurrent_fn`.
- A serious path means either a JAX learner/checkpoint/replay lane or a brittle
  Torch-to-JAX shadow-model conversion loop.
- Bouncing JAX env/search back into stock LightZero every step would recreate
  the host-boundary problem the spike is trying to escape.

Simultaneous actions:

- MCTX expands one action per root. CurvyTron physically steps both players at
  once.
- Independent `A=3` per-seat roots are the smallest useful approximation, but
  opponent behavior is folded into learned dynamics.
- A centralized `A=9` joint-action search is a useful control, but it changes
  semantics and raises branching cost.
- The spike must make row, player, value perspective, reward sign, terminal
  discount, and scatter-back conventions explicit.

Visual observations:

- Direct `64x64` learned observations are the right toy input; browser-pixel
  parity is the wrong first gate.
- Still, the toy must include enough visual structure to catch wrong
  perspective, wrong stack order, missing trails, and terminal-final-frame
  bugs.
- Spatial latents should not be first. MCTX stores embeddings in the tree, so a
  vector latent is the safe initial memory probe.

Replay/learner gap:

- MCTX supplies search outputs, not a training system.
- Replay chunks need observations, actions, rewards, final observations,
  action weights, raw/search values, masks, policy version, and config hashes.
- The learner needs target construction, optimizer state, checkpointing,
  evaluation, and actor refresh semantics.
- None of that should enter this toy. Adding it now would turn a viability
  probe into an unbounded second RL stack.

## 5. Recommendation

Do it now, but keep it tiny, scratch-only, and timeboxed. The current local
evidence says the stock LightZero path scalarizes env rows and converts model
roots back to CPU NumPy for MCTS, while external GPU RL systems win by keeping
the env/search batch resident. A JAX/MCTX toy is the cheapest direct test of
that alternate premise: if the fixed-shape toy cannot compile cleanly, handle
`[B,2]` axes, fit tree memory, and produce steady batched search timings, the
next phase should stay with LightZero-compatible vector facade and host-boundary
profiling. If it does work, it justifies a later, explicit alternate-lane design
for repo-owned JAX search/learner pieces. Either outcome is useful, and neither
requires touching production training.

## Local Context Read

- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/README.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_stock_boundary_batch_death_20260521.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_current_data_movement_audit_20260521.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_external_gpu_rl_patterns_20260521.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_gpu_rl_architecture_examples_20260521.md`
- `docs/working/optimizer/architecture_reexploration_2026-05-12/mctx_jax_search.md`
- `docs/research/mctx_integration.md`
