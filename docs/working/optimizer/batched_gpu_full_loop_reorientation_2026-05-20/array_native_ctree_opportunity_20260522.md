# Array-Native CTree Opportunity

Date: 2026-05-22

Status: doc-only sidecar audit. Do not touch live runs, Coach/trainer defaults,
or `train_muzero` launch defaults from this note.

## Plain Read

`direct_ctree_gpu_latent` is the practical LightZero-shaped baseline right now:
it keeps MuZero latent tensors on GPU during search, but still uses LightZero's
CPU CTree for selection and backup. Dense eager Torch search is fast at sim8 but
fails the sim16 scaling gate. The next credible speed lane is not more renderer
work and not another public `collect_mode.forward` wrapper cleanup; it is making
the CTree boundary array-native for CurvyTron's fixed `A=3` action space.

2026-05-22 caveat:

```text
The latest train_muzero repeat did not prove a direct_ctree_gpu_latent
full-loop win: stock 445.19 steps/sec, direct 438.56. Array-native CTree is
still plausible, but it must be justified against the full stock train_muzero
denominator, not just profile-only roots/sec.
```

The opportunity is precise:

```text
batched roots in arrays -> CTree consumes dense [N,3] arrays
per-sim recurrent outputs return compact arrays
CTree outputs dense visits/actions/values arrays
```

Today that path is still shaped by Python lists, `.tolist()`, CPU NumPy copies,
and Python per-root output assembly.

## Current Call Flow

Profile harness entry:

```text
scripts/build_curvytron_hybrid_observation_profile_grid.py
-> modal run -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile
-> _LightZeroCollectForwardStackProbe.run(observation, action_mask)
```

Exact repo files/symbols:

- `scripts/build_curvytron_hybrid_observation_profile_grid.py`
  - `MCTS_ARRAYS_BOUNDARY_IMPL_CHOICES`: `stock_facade`, `direct_ctree_arrays`, `direct_ctree_gpu_latent`.
  - `apply_next_direct_ctree_comparison_preset(...)`: H100/L4 fixed denominator, B512/A16/sim8, profile-only.
  - `_command(...)`: emits `--hybrid-lightzero-mcts-arrays-boundary-probe`, impl, input mode, and sim count.
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
  - constants at lines 161-174 name the current boundary semantics and impls.
  - `_build_profile_lightzero_policy(...)` builds a real `lzero.policy.muzero.MuZeroPolicy`.
  - `_LightZeroCollectForwardStackProbe.run(...)` validates `[B,P,4,64,64]`, flattens to roots, filters zero masks, prepares `to_play=[-1]*N`, and dispatches.
  - `_LightZeroCollectForwardStackProbe._prepare_observation_tensor(...)` handles `host_uint8`, pinned, float32, and stale resident profile input modes.
  - `_LightZeroCollectForwardStackProbe._run_direct_mcts_arrays(...)` is the current direct CTree compact boundary.
  - `_LightZeroCollectForwardStackProbe._run_direct_ctree_gpu_latent_search(...)` is the current GPU-latent loop.
  - `_plain_lightzero_value(...)` and `_compact_mcts_arrays_from_lightzero_plain(...)` decode public LightZero output for stock facade comparisons.
  - `_LightZeroModelCallTimer` and `_LightZeroCollectForwardInternalTimer` split initial/recurrent/search/CTree timing.
- `scripts/compare_curvytron_direct_ctree_stock.py`
  - `_run_impl(...)` compares stock facade, direct CTree, and GPU-latent direct CTree with debug compact arrays.

Installed LightZero public collect path:

```text
lzero/policy/muzero.py:MuZeroPolicy._forward_collect
-> _collect_model.initial_inference(data)
-> mz_network_output_unpack(network_output)
-> pred_values.detach().cpu().numpy()
-> latent_state_roots.detach().cpu().numpy()
-> policy_logits.detach().cpu().numpy().tolist()
-> legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] ...]
-> noises = [np.random.dirichlet(...).astype(np.float32).tolist() ...]
-> MCTSCtree.roots(active_collect_env_num, legal_actions)
-> roots.prepare(root_noise_weight, noises, reward_roots, policy_logits, to_play)
-> self._mcts_collect.search(roots, model, latent_state_roots, to_play)
-> roots.get_distributions()
-> roots.get_values()
-> per-env Python action/output dict assembly
```

Installed LightZero CTree search:

```text
lzero/mcts/tree_search/mcts_ctree.py:MuZeroMCTSCtree.search
for simulation_index in range(num_simulations):
  tree_muzero.ResultsWrapper(num=batch_size)
  tree_muzero.batch_traverse(...)
  for ix, iy in zip(...): latent_states.append(latent_state_batch_in_search_path[ix][iy])
  torch.from_numpy(np.asarray(latent_states)).to(device)
  torch.from_numpy(np.asarray(last_actions)).to(device).long()
  model.recurrent_inference(latent_states, last_actions)
  to_detach_cpu_numpy(latent_state/policy_logits/value/reward)
  reward.reshape(-1).tolist()
  value.reshape(-1).tolist()
  policy_logits.tolist()
  tree_muzero.batch_backpropagate(...)
```

Installed LightZero Cython/C++ boundary:

- `.venv/lib/python3.11/site-packages/lzero/mcts/ctree/ctree_muzero/mz_tree.cpython-311-darwin.so` is the installed compiled extension.
- Source mirror: `/private/tmp/lightzero-src-optimizer-20260521/lzero/mcts/ctree/ctree_muzero/mz_tree.pyx`
  - `Roots.__cinit__(int root_num, vector[vector[int]] legal_actions_list)` requires Python nested legal-action lists.
  - `Roots.prepare(float root_noise_weight, list noises, list value_prefix_pool, list policy_logits_pool, vector[int]& to_play_batch)` requires Python lists for noises/rewards/policies.
  - `batch_traverse(...)` returns Python lists: latent search-path indices, batch indices, last actions, virtual to-play.
  - `batch_backpropagate(..., list value_prefixs, list values, list policies, ...)` converts Python lists into C++ vectors.
- Source mirror: `/private/tmp/lightzero-src-optimizer-20260521/lzero/mcts/ctree/ctree_muzero/lib/cnode.cpp`
  - `CRoots::prepare(...)` expands each root and adds noise.
  - `CRoots::get_distributions()` returns `std::vector<std::vector<int>>`.
  - `CRoots::get_values()` returns `std::vector<float>`.
  - `cbatch_traverse(...)` fills result vectors for selected leaves.
  - `cbatch_backpropagate(...)` expands selected leaves and backs up values.

Current repo direct path:

```text
_run_direct_mcts_arrays(...)
-> model.initial_inference(obs_tensor)
-> pred_values: inverse_scalar_transform(...).detach().cpu().numpy()
-> latent_state_roots: either detach().cpu().numpy() or kept on GPU
-> policy_logits.detach().cpu().numpy()
-> policy_logits_np.tolist()
-> legal_actions = [[... enumerate(flat_mask[row]) ...]]
-> noises = [np.random.dirichlet(...).astype(np.float32).tolist() ...]
-> type(mcts).roots(active_root_count, legal_actions)
-> roots.prepare(..., reward_roots, policy_logits_list, to_play)
-> direct_ctree_arrays: mcts.search(...)
-> direct_ctree_gpu_latent: _run_direct_ctree_gpu_latent_search(...)
-> roots.get_distributions()
-> roots.get_values()
-> assemble actions[N], visits[N,3], searched_values[N], predicted_values[N], policy_logits[N,3]
```

`direct_ctree_gpu_latent` removes the repeated latent CPU copy, but it still does
these CPU/list crossings per simulation:

```text
tree_muzero.batch_traverse(...) -> Python list indices/actions
last_actions -> torch.as_tensor(..., device)
model.recurrent_inference(...)
reward/value/policy_logits -> detach().cpu().numpy()
reward/value/policy_logits -> .tolist()
tree_muzero.batch_backpropagate(...)
```

## Exact Hot Spots

Python lists and `.tolist()`:

- `lzero/policy/muzero.py:741-745`: initial value/latent/logit CPU copies and policy logits `.tolist()`.
- `lzero/policy/muzero.py:745`: legal actions built as nested Python lists.
- `lzero/policy/muzero.py:748-750`: root Dirichlet noises built as nested Python lists.
- `lzero/mcts/tree_search/mcts_ctree.py:288-295`: Python list append for latent gathering and NumPy-to-Torch conversion.
- `lzero/mcts/tree_search/mcts_ctree.py:306-316`: recurrent output CPU copies plus reward/value/policy `.tolist()`.
- `source_state_batched_observation_boundary_profile.py:5095-5106`: direct path builds `policy_logits_list`, `legal_actions`, `noises`, `Roots`, and `roots.prepare`.
- `source_state_batched_observation_boundary_profile.py:5665-5674`: GPU-latent path converts recurrent reward/value/policy to lists before `batch_backpropagate`.
- `source_state_batched_observation_boundary_profile.py:5153-5258`: direct output extraction converts CTree distributions/values back into dense arrays and samples actions.

CPU arrays / GPU copies:

- `lzero/policy/muzero.py:741-743`: public collect copies initial outputs to CPU.
- `lzero/mcts/tree_search/mcts_ctree.py:135,292`: selected latents go CPU NumPy -> Torch device each simulation.
- `lzero/mcts/tree_search/mcts_ctree.py:306-309`: recurrent outputs go Torch device -> CPU NumPy each simulation.
- `source_state_batched_observation_boundary_profile.py:5074-5088`: direct path copies predicted values and policy logits to CPU; copies latents unless GPU-latent mode is enabled.
- `source_state_batched_observation_boundary_profile.py:5653-5657`: GPU-latent path still copies recurrent reward/value/policy logits to CPU.

Root prep:

- `lzero/policy/muzero.py:754-759`: public collect roots and noisy prepare.
- `lzero/mcts/tree_search/mcts_ctree.py:222-231`: `MuZeroMCTSCtree.roots(...)`.
- `mz_tree.pyx:30-36`: `Roots` constructor and `prepare(...)` typed around nested vectors/lists.
- `cnode.cpp:301-339`: `CRoots` constructor and `CRoots::prepare(...)`.
- `source_state_batched_observation_boundary_profile.py:5095-5107`: direct root build/prep timing.

Traverse/backprop:

- `lzero/mcts/tree_search/mcts_ctree.py:277-286`: stock `tree_muzero.batch_traverse(...)`.
- `lzero/mcts/tree_search/mcts_ctree.py:324-327`: stock `tree_muzero.batch_backpropagate(...)`.
- `source_state_batched_observation_boundary_profile.py:5600-5614`: GPU-latent `batch_traverse(...)`.
- `source_state_batched_observation_boundary_profile.py:5669-5678`: GPU-latent `batch_backpropagate(...)`.
- `mz_tree.pyx:74-82`: Cython list-to-vector conversion for `batch_backpropagate`.
- `mz_tree.pyx:95-100`: Cython `batch_traverse` wrapper returns Python lists.
- `cnode.cpp:480-499`: C++ `cbatch_backpropagate`.
- `cnode.cpp:755-823`: C++ `cbatch_traverse`.

Output extraction:

- `lzero/policy/muzero.py:762-793`: public `roots.get_distributions`, `roots.get_values`, action selection, dict output.
- `mz_tree.pyx:44-48`: `Roots.get_distributions()` and `get_values()` wrappers.
- `cnode.cpp:387-416`: C++ distribution/value getters.
- `source_state_batched_observation_boundary_profile.py:4730-4748`: stock facade output decoded to compact arrays.
- `source_state_batched_observation_boundary_profile.py:5153-5275`: direct path output arrays assembled from CTree getters.

## Array-Native API Shape For Fixed A=3

The useful API should exploit CurvyTron's fixed action count instead of preserving
LightZero's variable legal-action list contract.

Minimal Cython/C++ API sketch:

```python
roots = mz_tree.RootsFixedA3(root_count: int, legal_mask: np.ndarray[np.uint8, ndim=2])

roots.prepare_arrays(
    root_noise_weight: float,
    noises: np.ndarray[np.float32, ndim=2],        # [N,3], zero for illegal
    rewards: np.ndarray[np.float32, ndim=1],       # [N]
    policy_logits: np.ndarray[np.float32, ndim=2], # [N,3]
    to_play: np.ndarray[np.int32, ndim=1],         # [N]
)

leaf = roots.batch_traverse_arrays(
    pb_c_base: int,
    pb_c_init: float,
    discount_factor: float,
    min_max_stats: MinMaxStatsList,
)
# leaf.path_index: int32[N]
# leaf.batch_index: int32[N]
# leaf.last_actions: int64[N] or int32[N]
# leaf.virtual_to_play: int32[N]

roots.batch_backpropagate_arrays(
    current_latent_state_index: int,
    discount_factor: float,
    rewards: np.ndarray[np.float32, ndim=1],       # [N]
    values: np.ndarray[np.float32, ndim=1],        # [N]
    policy_logits: np.ndarray[np.float32, ndim=2], # [N,3]
    min_max_stats: MinMaxStatsList,
    leaf_results: ResultsWrapper,
    virtual_to_play: np.ndarray[np.int32, ndim=1],
)

visits, values = roots.get_arrays()
# visits: int32 or float32 [N,3], dense action order 0/1/2
# values: float32 [N]
```

Python-side direct path with this API:

```text
initial_inference on GPU
pred_values/policy_logits/reward -> CPU contiguous arrays once for root prep
legal_mask [N,3] already exists from flat_mask
RootFixedA3.prepare_arrays(...)
for sim in S:
  leaf arrays = batch_traverse_arrays(...)
  latent_states = latent_pool[path_index, batch_index] on GPU
  last_actions = torch.as_tensor(leaf.last_actions, device)
  recurrent_inference on GPU
  reward/value/policy_logits -> CPU contiguous arrays
  batch_backpropagate_arrays(...)
visits[N,3], values[N] = roots.get_arrays()
sample actions over dense visits without per-root legal-list mapping
```

More ambitious API, if vendoring is accepted:

```text
batch_traverse_arrays returns NumPy memoryviews without Python list allocation.
batch_backpropagate_arrays accepts typed memoryviews, not Python lists.
get_arrays fills caller-provided output arrays to avoid fresh allocations.
```

The API does not need to solve arbitrary action counts. For this lane, fixed
`A=3` is a feature: dense arrays avoid variable-length child distributions,
legal-action indexing, per-row list construction, and final sparse-to-dense
scatter.

## Local Versus Vendored Changes

Local/repo-only changes:

- Add a new profile impl constant, e.g. `direct_ctree_gpu_latent_array_native`.
- Add a compatibility wrapper around the vendored Cython extension in
  `source_state_batched_observation_boundary_profile.py`.
- Reuse existing `_prepare_observation_tensor(...)`, timers, compact debug arrays,
  and `scripts/compare_curvytron_direct_ctree_stock.py`.
- Extend focused tests in `tests/test_source_state_batched_observation_boundary_profile.py`
  with a fake `RootsFixedA3` to prove array shapes, legal masks, output assembly,
  and stock/direct parity handling without requiring the compiled extension.
- Add a manifest row builder option only if/when the profile impl exists; keep
  `profile_only=true`, `calls_train_muzero=false`, `touches_live_runs=false`.

Vendored LightZero/Cython changes:

- Add new Cython entry points in `ctree_muzero/mz_tree.pyx` using typed
  memoryviews or NumPy arrays instead of Python lists.
- Add a fixed-action C++ root representation or a fixed-action facade over
  existing `CNode`/`CRoots`.
- Change root preparation to consume dense `[N,3]` legal masks, noises, rewards,
  and policy logits.
- Change traverse result export to fill dense integer arrays rather than returning
  Python lists.
- Change backprop to consume dense arrays directly.
- Change distribution/value extraction to fill dense `[N,3]` and `[N]` arrays.

Do not modify installed `.venv` directly as the durable solution. Prototype in a
source checkout or repo-owned vendored patch, then build the extension in an
isolated profile environment.

## Expected Speed Ceiling

Current attested profile memory:

- H100 public MCTS collect sim8: about `2572` roots/sec.
- H100 public pure-policy collect: about `6287` roots/sec.
- H100 recurrent toy sim8: about `8681` roots/sec.
- H100 policy arrays: about `9958` roots/sec.
- `direct_ctree_gpu_latent` fresh branch: sim8 about `7547` roots/sec, sim16
  about `6145` roots/sec.
- Dense eager Torch fixed-shape after semantic fixes: sim8 about `8288`
  roots/sec, sim16 about `4294` roots/sec.

Array-native CTree should not be expected to beat the recurrent toy ceiling. Its
realistic goal is:

```text
sim8: 8k-9k roots/sec if Python list/output overhead is the remaining wall
sim16: 6.5k-8k roots/sec if CTree's compiled selection/backprop scales better
       than eager dense Torch while list crossings are removed
```

The speed ceiling is bounded by recurrent model calls, root/input preparation,
remaining CPU CTree work, and host observation packaging. If only root prep and
final extraction are array-native while per-simulation backprop still calls
`.tolist()`, expect a small improvement, not a branch-changing result.

## Risks

- CTree uses variable legal-action children internally. A fixed `A=3` dense API
  must preserve exact legal-mask semantics and not assign visits to illegal moves.
- Root noise currently has variable length by legal action set. Dense `[N,3]`
  noise must match stock behavior over legal actions only.
- `batch_traverse` currently mutates/returns `virtual_to_play_batch` as Python
  lists. Array-native mutation and ownership need clear semantics.
- Tie-breaking uses C++ randomness in selection and NumPy randomness in action
  sampling. Parity gates must use enough sims/root noise to avoid false failures
  from tie-heavy tiny rows.
- Cython memoryview signatures can silently copy if arrays are not contiguous or
  dtype-exact.
- A fixed `A=3` patch is intentionally not generic LightZero. Keep it profile-only
  until the boundary proves enough speed and exact behavior.
- Installing/rebuilding the Cython extension can pollute local/runtime
  environments. Use an isolated source checkout and no live trainer imports.
- The full-loop denominator may move: even a faster CTree boundary can expose
  collector, replay, RND, or observation packaging as the next wall.

## First No-Live-Run Experiment

Do not run Modal live jobs and do not touch trainer defaults. The first
experiment should be local and synthetic:

1. Build a tiny source-checkout Cython spike named something like
   `RootsFixedA3` against `/private/tmp/lightzero-src-optimizer-20260521`, not
   the active `.venv`.
2. Implement only these methods first:
   `prepare_arrays(...)`, `get_arrays(...)`, and a no-search smoke around existing
   C++ root expansion.
3. Add a repo-side adapter in the profile harness behind a new impl name, but
   guard it so missing compiled support skips/fails clearly.
4. Run only local focused tests and the existing local comparison helper at tiny
   roots:

```text
uv run pytest tests/test_source_state_batched_observation_boundary_profile.py \
  -k "direct_ctree or mcts_arrays_boundary"

uv run python scripts/compare_curvytron_direct_ctree_stock.py \
  --seeds 8 \
  --batch-rows 4 \
  --num-simulations 8 \
  --impls direct_ctree_gpu_latent
```

5. Only after root prep and output extraction prove shape/parity, add
   `batch_backpropagate_arrays(...)`. The branch-changing test is whether
   removing per-sim `.tolist()` improves sim16 against `direct_ctree_gpu_latent`;
   root/output arrays alone are not enough.

Success criteria before any remote profile row:

- exact action/visit/value parity versus current direct CTree for fixed seeds;
- zero illegal actions under mixed legal masks;
- debug arrays included for roots <= 16;
- no changes to `train_muzero`, Coach config, live-run manifests, or trainer
  defaults.
