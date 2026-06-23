# LightZero Boundary Test Critique - 2026-05-21

Scope: review of the profile-only LightZero boundary probes in
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`,
the grid builder in `scripts/build_curvytron_hybrid_observation_profile_grid.py`,
and related local tests. I did not touch live training runs or production code.

## Bottom Line

The new profiles support one useful optimizer read:

- `policy._model.initial_inference(...)` over active pre-scalar
  `[B,2,4,64,64]` roots is fast in this canary.
- `MuZeroPolicy.collect_mode.forward(...)` over the same profile stack is much
  slower, so the next wall is probably LightZero collect/search/output handling,
  not the renderer.

But the current tests mostly prove that the probes run, not that their semantics
match stock training. The dangerous failure mode is silent wrongness: clean
throughput numbers with a subtly different `to_play`, root set, surface contract,
scalarization path, output decode, or denominator.

## Safe Semantics

The harness is correctly marked profile-only. The hybrid path returns
`profile_only=True`, `calls_train_muzero=False`, `stock_lightzero_integrated=False`,
and `touches_live_runs=False`. The grid builder also emits rows with
`calls_train_muzero=False` and `touches_live_runs=False`.

The probes are also correctly placed before scalar LightZero timestep
materialization. `HybridObservationProfileManager.step(...)` calls
`batched_stack_probe.run(observation_for_timestep, compact["action_mask"])` before
the optional `materialize_lightzero_scalar_timestep(...)` edge
(`source_state_hybrid_observation_profile.py:428`, `:454`). That makes the rows
good for a pre-scalar consumer canary.

The collect-forward probe is a real public LightZero API call:
`policy.collect_mode.forward(...)` with a real scratch `MuZeroPolicy`
(`source_state_batched_observation_boundary_profile.py:3903`). The
initial-inference probe intentionally bypasses MCTS and calls
`policy._model.initial_inference(...)` (`:4150`).

## Silent-Wrongness Risks

### 1. `to_play` Is Pinned To Fixed-Opponent Semantics

The collect probe flattens both player views but sends
`to_play=[-1] * active_root_count` (`source_state_batched_observation_boundary_profile.py:3806`).
The unit test asserts this exact value
(`tests/test_source_state_batched_observation_boundary_profile.py:500`).

That is only safe for the fixed-opponent CurvyTron profile policy. It is not a
generic two-seat self-play boundary. Existing two-seat collect code uses the
player ids as `to_play` (`curvytron_two_seat_lightzero_train_smoke.py` has the
batched collect path with `to_play = [int(item) for item in players]`).

Risk: a future reader may see `[B,2,...]` roots and assume this proves a two-seat
policy/search boundary. It does not. The current test locks in the fixed-opponent
simplification instead of failing if the surface metadata claims two-seat
self-play.

Required test guard: if `policy_metadata["surface"]` says two-seat/current-policy
collection, assert `to_play == players.tolist()`. If it says fixed opponent,
assert `to_play == [-1] * N` and report `lightzero_to_play_contract:
fixed_opponent_only`.

### 2. Zero-Mask Filtering Changes The Root Set

Both probes drop roots whose action mask is all zero
(`source_state_batched_observation_boundary_profile.py:3799`, `:4072`). The tests
prove dropped roots are counted and that `ready_env_id` is compacted
(`tests/test_source_state_batched_observation_boundary_profile.py:589`).

That is useful, but it hides two semantic traps:

- `ready_env_id=np.arange(active_root_count)` is not the original scalar env id
  (`row * player_count + player`). If a later probe needs stock env-id mapping,
  output decoding by row offset will silently become wrong.
- Profile summaries often carry both total stack roots and active LightZero
  roots. Amdahl charts must use active legal roots for collect/search throughput,
  not `B * 2`, when any zero masks were filtered.

Required test guard: one test should keep noncompact ready ids, for example
`[0, 3]`, and prove output decoding follows `ready_env_id` rather than row offset.
Another should assert that any reported collect roots/sec uses
`lightzero_root_count`, not `batched_stack_probe_total_roots`, when
`lightzero_filtered_zero_mask_root_count > 0`.

### 3. Scalar-On Rows Are Additive, Not Stock-Order LightZero

The grid builder sets synthetic probe simulations to zero for LightZero probes
but still emits `--hybrid-materialize-scalar-timestep` or
`--no-hybrid-materialize-scalar-timestep`
(`scripts/build_curvytron_hybrid_observation_profile_grid.py:84`, `:125`,
`:146`). In the runtime, scalar materialization happens after the pre-scalar
LightZero probe.

So a `scalaron` LightZero row means:

```text
pre-scalar stack -> LightZero probe -> scalar timestep materialization tax
```

It does not mean:

```text
scalar timestep -> stock LightZero collector -> policy/search
```

Risk: labeling these rows as "real scalar LightZero" will overstate production
fidelity. They are still valuable for measuring the extra materialization cost,
but only as an additive tax in this scaffold.

Required reporting guard: label rows as `pre_scalar_collect_forward + scalar_tax`
or `pre_scalar_initial_inference + scalar_tax`, never as stock collector speed.

### 4. Surface Labels Are Too Thin For A Contract Claim

`_build_profile_lightzero_policy(...)` stores only a small surface subset:
env variant, observation shape, policy observation backend, trail render mode,
and bonus render mode (`source_state_batched_observation_boundary_profile.py:3738`).

The full config surface has stronger labels: `policy_observation_contract_id`,
`observation_contract`, perspective schema, perspective owner, seat mapping,
visual surface, source fidelity, two-seat flags, and learner seat mode
(`lightzero_config_builder.py:1343`, `tests/test_lightzero_config_builder.py:521`).

Risk: the probe input is the profile renderer's `direct_gray64` persistent
policy-space surface, but telemetry can look like a normal source-state
LightZero surface. A wrong surface label can make a fast profile row look like a
training-equivalent policy boundary.

Required test guard: fake-policy tests should pass a complete surface contract
and assert collect-forward and initial-inference telemetry preserve it unchanged.
Profile reports should also include renderer backend, `render_surface`,
`hybrid_stack_storage_dtype`, and whether this is `direct_gray64` profile space.

### 5. Policy Output Decoding Is Under-Tested

Current tests cover dict-of-arrays and simple fake dict outputs
(`tests/test_source_state_batched_observation_boundary_profile.py:407`, `:469`).
The decode helpers support more shapes, but the tests do not pin the tricky
ones:

- string-keyed outputs keyed by ready env id;
- list outputs;
- nested `{0: {...}}` root outputs;
- `selected_action` / `selected_actions`;
- missing action failure;
- noncompact ready ids;
- illegal decoded action fail-closed.

Risk: collect-forward could return a valid LightZero structure that we decode
against row offset instead of environment id, or we could accept an action from
the wrong nested root.

Required test guard: expand `_policy_output_row_from_plain(...)` and
`_extract_eval_action_from_plain(...)` cases, then add one fake collect-mode test
where only action `1` is legal and the fake output chooses `2`; the probe should
raise before emitting success telemetry.

### 6. `model_eval_count` Is Not A Literal Model-Eval Count

Initial inference reports `model_eval_count = active_root_count`, which is a
literal count for that probe (`source_state_batched_observation_boundary_profile.py:4178`).
Collect-forward reports `model_eval_count = active_root_count * num_simulations`
(`:3969`).

That collect-forward value is at best a simulation-equivalent pressure count. It
does not prove how many LightZero initial or recurrent network calls actually
ran inside MCTS. Depending on LightZero internals, there is an initial inference
per root plus recurrent evaluations during search, with batching and terminal
cases in the mix.

Risk: Amdahl tables can compare "model evals" across initial-inference and
collect-forward rows as if they were the same unit.

Required reporting guard: rename or supplement the collect field with
`requested_search_simulation_count`, `simulation_equivalent_eval_count`, and
`lightzero_consumer_num_simulations`. Reserve `model_eval_count` for measured
network calls only.

### 7. Initial-Inference Labels Still Carry A Simulation Flag

The grid builder appends `-s{probe_simulations}` to every row label, including
initial-inference rows (`scripts/build_curvytron_hybrid_observation_profile_grid.py:186`).
It also passes `--hybrid-lightzero-consumer-num-simulations` for initial inference
(`:161`), even though `_LightZeroInitialInferenceStackProbe` reports
`simulations=0.0` (`source_state_batched_observation_boundary_profile.py:4173`).

Risk: a row named `s8-lzii` can be misread as "initial inference at 8 sims."
There are no sims in the model-only probe.

Required reporting guard: initial-inference labels should use `-modelonly` or
`-sNA`, and tables should show `sims=n/a`.

## Amdahl Read, With Correct Caveats

The current Amdahl statement should be:

> In the profile-only fixed-opponent, no-death, pre-scalar, scratch-policy canary,
> model initial inference over active legal roots is fast, while the public
> `collect_mode.forward` path is slow. The likely remaining wall is CPU/tree
> search, policy wrapper work, synchronization, or output decode.

It should not be:

> Stock CurvyTron LightZero training is now search-bound, or the production
> two-seat policy/search path has been proven slow.

The collect-forward minus initial-inference delta is a strong directional clue,
not a clean subtraction. The two probes differ in wrapper code, mask handling,
`to_play`, output handling, and search internals. Use it to choose the next
instrumentation target, not as a final production Amdahl proof.

## Test Matrix I Would Require Before Promotion

P0 tests:

- fixed-opponent `to_play=-1` and two-seat `to_play=players` are separate
  contracts;
- zero-mask filtered roots report active denominators and preserve env-id mapping;
- collect output decoding covers keyed, list, nested, selected-action, missing,
  and illegal-action cases;
- complete surface contract survives into telemetry for both probes;
- initial-inference rows use `sims=n/a` labels and collect rows use requested sims;
- scalar-on rows report scalar materialization as post-probe additive cost.

P1 tests:

- compact output preserves all LightZero telemetry keys;
- fake policy with noncompact `ready_env_id` returns keyed outputs and decode
  remains correct;
- model-only output summary proves shapes without full latent readback;
- root throughput helpers prefer `lightzero_root_count` over total stack roots.

## Recommendation

Keep using these rows as optimizer evidence, but label them sharply:

```text
profile-only fixed-opponent pre-scalar LightZero consumer canary
```

The next code-facing work should split `collect_mode.forward` internally into
tensor prep, initial inference, CPU tree/search, output decode, and action
legality checks. The next docs-facing work should scrub any table or Amdahl label
that says or implies full-loop, stock-training, two-seat, or production
LightZero equivalence.
