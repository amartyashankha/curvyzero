# CurvyZero: Algorithm and Systems Strategy

## Executive conclusion

Curvytron is not a normal board-game search problem and not quite an Atari problem. It is a **real-time, simultaneous-action, sparse-reward stochastic game** with:

- 60 Hz continuous kinematics;
- only three steering controls;
- permanent, topology-changing trails;
- discontinuous collision outcomes;
- long-term territory and enclosure strategy;
- short-term collision avoidance;
- simultaneous opponent actions;
- optional random gaps, spawns, and bonuses.

The recommended path is:

1. **Make the simulator correct, deterministic, fixed-step, and extremely fast.**
2. **Establish a recurrent model-free self-play baseline**, preferably with PufferLib or an equally lean vectorized PPO stack.
3. **Represent long-term strategy in the policy and value function** using multiscale spatial observations, recurrence, league self-play, and strategic auxiliary targets.
4. **Add planning at a meaningful temporal scale**, not at every 60 Hz physics tick.
5. Prefer a **GPU-dense trajectory planner**—beam search, CEM, sequential halving, or batched MPC over action chunks—before investing further in fully general MCTS.
6. Treat **joint-action Gumbel MuZero** as a research branch, not the default architecture. Use small search budgets, explicit opponent/chance modeling, reanalysis, and aggressive empirical gates.
7. Keep PPO and search-based training semantics clean. Do not collect planner actions and train them as if they were sampled from PPO's old policy.

The central diagnosis is:

> The expensive synchronization may be partly an implementation bug, but ordinary MCTS is also structurally mismatched to a small-batch, 60 Hz, simultaneous-control workload. Moving the same algorithm to the GPU does not automatically make it GPU-shaped.

---

## 1. Scope and audit status

The connected GitHub application could not access `amartyashankha/curvyzero`, so this document does **not** claim a line-by-line audit of that repository. It is based on:

- the public legacy Curvytron implementation;
- MuZero, Gumbel MuZero, Stochastic MuZero, and simultaneous-move MCTS literature;
- DeepMind's JAX `mctx` implementation;
- current PufferLib 4.0 documentation and self-play code;
- the symptoms described: GPU MCTS, synchronization cost, and concern about preserving long-term strategy.

The code-specific sections are therefore framed as **high-probability failure modes and tests**, not as confirmed defects in CurvyZero.

---

## 2. What kind of problem is Curvytron?

### 2.1 Physics and control

The legacy implementation runs at 60 Hz. A player travels at 16 world units per second, turns at approximately 2.8 radians per second, has radius 0.6, and receives a steering factor that effectively selects left, straight, or right.

A simplified fixed-step model is:

\[
x_{t+1} = x_t + v\Delta t\cos\theta_t
\]

\[
y_{t+1} = y_t + v\Delta t\sin\theta_t
\]

\[
\theta_{t+1} = \theta_t + \omega\Delta t\,a_t,
\qquad a_t\in\{-1,0,+1\}
\]

with:

\[
\Delta t=\frac{1}{60},\qquad v\approx 16,\qquad \omega\approx 2.8.
\]

Derived quantities:

- travel per physics tick: about \(0.267\) units;
- turn per tick: about \(0.0467\) radians, or \(2.67^\circ\);
- constant-turn radius: \(v/\omega\approx 5.7\) units;
- four-tick action repeat: about \(66.7\) ms, \(1.07\) units of travel, and \(10.7^\circ\) of turning.

This explains why raw-tick MCTS is wasteful. Most adjacent tree levels differ by a tiny arc segment while consuming an entire search depth.

### 2.2 Topology, not merely reflexes

Long-term strategy absolutely matters. Good play includes:

- preserving future turning room;
- creating and defending escape corridors;
- controlling large reachable regions;
- threatening cuts several seconds ahead;
- deciding when to approach or avoid an opponent;
- exploiting gaps;
- avoiding self-enclosure;
- trading immediate safety against future territory.

The correct conclusion is not “use only a short-horizon reflex controller.” It is:

> Use a long-horizon learned policy/value system and perform explicit search over temporally meaningful commitments.

### 2.3 Simultaneous actions

For two players, the correct transition is:

\[
s_{t+1}\sim P(\cdot\mid s_t,a_t^1,a_t^2,c_t),
\]

where \(c_t\) represents chance events such as a gap toggle or bonus event.

A single-agent MuZero model that conditions only on our action,

\[
z_{t+1}=g(z_t,a_t^1),
\]

is omitting an essential causal input. It implicitly learns:

\[
P_{\pi^-}(s'\mid s,a^1)
=
\sum_{a^2}
\pi^-(a^2\mid s)
P(s'\mid s,a^1,a^2).
\]

This transition changes whenever the opponent policy \(\pi^-\) changes. In self-play, it is therefore non-stationary and often multimodal. The model may average mutually incompatible futures and search may exploit the resulting fantasy dynamics.

For a proper learned model, use at least:

\[
z_{t+1}=g(z_t,a_t^1,a_t^2,c_t).
\]

For two players with three actions each, the joint action space has only \(3\times3=9\) entries. That is manageable. For \(N\) free-for-all players it grows as \(3^N\), so a different factorization or sampled-opponent approach becomes necessary.

### 2.4 Stochasticity

The public game uses random:

- gap lengths;
- trail-print lengths;
- spawn positions and directions;
- bonus times, types, and positions.

Standard MuZero's latent dynamics is deterministic. If randomness is hidden from the observation, deterministic MuZero must blur over possible outcomes. Stochastic MuZero exists precisely because this can damage planning.

The first training environment should therefore be simpler:

1. fixed-seed deterministic spawns;
2. no bonuses;
3. either no random gaps or an explicit, reproducible gap process;
4. two players;
5. fixed physics timestep;
6. reproducible simultaneous collision resolution.

Stochastic features should be added only after the core game is learned and benchmarked.

---

## 3. A critical simulator issue: update order

The legacy server iterates players, moves one player, may add its new trail body immediately, and then checks collision before processing the next player. This means a nominally simultaneous frame can have iteration-order-dependent outcomes.

For RL, choose one rule deliberately.

### Option A: exact legacy compatibility

Preserve the original update order. This is useful for reproducing browser matches, but the policy can potentially exploit player index or update order.

### Option B: clean simultaneous semantics

Use a two-phase frame:

1. read every player's action;
2. compute every new head pose;
3. compute every new trail segment;
4. detect all border, trail, and head interactions against a common frame state;
5. resolve deaths and ties simultaneously;
6. commit trail updates.

This is fairer and better aligned with the mathematical game. It will not be perfectly identical to the old server.

### Recommendation

Use Option B for the research environment and keep a compatibility mode for golden tests. Document the difference as a rules change.

Also replace wall-clock ages and timers with integer tick counters, and replace global randomness with a seeded, explicit PRNG state.

---

## 4. The correct temporal abstraction

### 4.1 Separate three clocks

A useful starting architecture has three rates:

| Layer | Suggested rate | Role |
|---|---:|---|
| Physics | 60 Hz | exact movement, trail creation, collision |
| Control policy | 10–30 Hz | left/straight/right reaction |
| Strategic planner | 1–5 Hz, plus critical triggers | territory, route commitment, opponent interaction |

The exact values should be swept. A good first baseline is physics at 60 Hz and policy decisions every four ticks, or 15 Hz, matching the useful Atari lesson of action repeat without copying Atari blindly.

### 4.2 Macro-actions

Define a macro-action:

\[
u=(a,d),
\]

where \(a\in\{-1,0,+1\}\) and \(d\) is a duration in physics ticks.

For example:

\[
d\in\{2,4,8\}.
\]

The macro return and discount are:

\[
R_t^{(u)}
=
\sum_{i=0}^{d-1}\gamma^i r_{t+i},
\qquad
\Gamma_u=\gamma^d.
\]

A short explicit planning horizon in macro-actions can cover a useful physical horizon while the leaf value estimates everything beyond it:

\[
Q_H(s,u_0)
=
\mathbb{E}\left[
\sum_{k=0}^{H-1}
\left(\prod_{j<k}\Gamma_{u_j}\right)R^{(u_k)}
+
\left(\prod_{j<H}\Gamma_{u_j}\right)V(s_H)
\right].
\]

Long-term reasoning comes from two places:

- coarse, strategic actions that cover meaningful time;
- a value function trained on complete games.

It does **not** require expanding a tree all the way to terminal states.

### 4.3 Event-triggered planning

Run the expensive planner when one of these is true:

- policy entropy is high;
- value ensemble disagreement is high;
- predicted time-to-collision is low;
- the opponent enters interaction range;
- a corridor becomes narrow;
- a bonus or gap changes the topology;
- a periodic strategic timer fires.

Use the fast policy on ordinary frames.

---

## 5. Why GPU MCTS is probably synchronizing or underperforming

There are three distinct problems that are often confused.

## 5.1 Actual host-device synchronization

Common hard synchronization points in a PyTorch search loop include:

- `.item()`;
- `.cpu()` or `.numpy()`;
- `bool(tensor)`, `int(tensor)`, or a Python `if` based on a CUDA tensor;
- CPU tree traversal followed by GPU leaf evaluation;
- copying selected node IDs or actions to the host every simulation;
- printing CUDA values;
- timing every inner operation with `torch.cuda.synchronize()`;
- dynamic Python containers holding tree nodes;
- a CPU environment step inside each simulated expansion;
- per-simulation tensor allocation;
- framework crossings such as Torch to JAX through host memory.

A classic anti-pattern is:

```python
for simulation in range(num_simulations):
    leaf_id = select_on_cpu(tree)
    action = gpu_actions[leaf_id].item()   # forced synchronization
    output = recurrent_model(..., batch_size=1)
    backup_on_cpu(tree, output.cpu())
```

The desired hot path performs one host call for an entire search batch and returns only final root statistics.

## 5.2 Sequential dependency that remains on the GPU

JAX `mctx` avoids Python-host traversal by using JIT-compatible loops and static tree tensors, but the MCTS recurrence is still sequential.

With \(S\) simulations and \(B\) independent roots, cost is approximately:

\[
T_{\text{search}}
\approx
S\left[
T_{\text{select}}(B)
+
T_{\text{recurrent}}(B)
+
T_{\text{backup}}(B)
\right].
\]

It is **not** a single neural call of size \(B\times S\).

Each simulation expands approximately one leaf per root. Therefore:

- parallelism is primarily across independent roots;
- simulations inside one tree depend on previous visit counts;
- a batch of one or eight games gives tiny recurrent-model batches;
- fifty simulations still imply fifty sequential model evaluations;
- JIT removes host round trips but cannot remove the dependency.

If GPU utilization rises sharply as the number of independent roots rises, the primary problem is under-batching rather than a hidden synchronization.

## 5.3 Irregular GPU workload

Tree search contains:

- pointer-like indexed traversal;
- variable search depths;
- divergent control flow;
- random gathers;
- scatter updates;
- repeated backups;
- possible atomic conflicts;
- small kernels;
- large, sparsely accessed tree buffers.

These are not ideal GPU operations. A GPU is happiest with large, dense, regular tensor programs.

### Mctx-specific trap

In `mctx`, omitting `max_depth` makes it default to `num_simulations`. This may create unnecessary traversal depth and increasingly expensive backups. Set a deliberate small maximum depth and let the value network handle the tail.

### Tree memory

If the tree stores one latent embedding per node, embedding memory alone is:

\[
M_{\text{embed}}
=
B(S+1)Dq,
\]

where \(D\) is latent size and \(q\) is bytes per element.

For \(B=4096\), \(S=32\), \(D=256\), and bfloat16, this is about 69 MB before visits, child arrays, rewards, discounts, parents, and temporary buffers.

A spatial latent is far more expensive. A \(16\times16\times32\) latent contains 8192 values. At only \(B=1024\), \(S=32\), and bfloat16, node embeddings alone require about 554 MB.

Use a compact latent for search. Keep high-resolution observation processing mainly in the root representation network.

---

## 6. A profiling plan that isolates the real fault

Measure these variants with fixed shapes and warmed-up kernels.

### Benchmark A: policy only

No search. Measure environment, root network, and action transfer.

### Benchmark B: recurrent model only

Call the recurrent model \(S\) times with the same leaf batch size, but no tree operations.

### Benchmark C: tree only

Use a dummy recurrent function returning constants. Measure selection, expansion, and backup.

### Benchmark D: full search

Measure the real implementation.

For every benchmark record:

- decisions per second;
- environment steps per second;
- root batch \(B\);
- effective leaf batch per simulation;
- number of simulations \(S\);
- mean and maximum traversal depth;
- kernel count per decision;
- H2D and D2H copy count;
- GPU active percentage;
- memory bandwidth;
- allocator calls;
- compile/recompile count;
- planner latency distribution, not just mean.

Useful experiments:

1. Sweep \(B\): 1, 8, 32, 128, 512, 2048.
2. Sweep \(S\): 1, 4, 8, 16, 32.
3. Hold \(S\) fixed and cap depth at 4, 8, 12.
4. Replace the network with a tiny MLP.
5. Replace tree logic with dense random trajectory rollout.
6. Keep data on one device for the entire call.
7. Search the source for scalar extraction and synchronization:
   ```bash
   rg -n '\.item\(|\.cpu\(|\.numpy\(|cuda\.synchronize|bool\(|int\(' .
   ```
8. Use Nsight Systems or a CUDA-aware framework profiler and inspect whether each simulation contains a D2H copy or CPU wait.

Interpretation:

- **Many D2H copies:** real host synchronization bug.
- **No copies, low utilization at small \(B\):** insufficient independent roots.
- **High utilization but poor speed:** network or memory bandwidth dominates.
- **Tree-only becomes superlinear in \(S\):** traversal depth or backup cost is growing.
- **Frequent compile or allocation events:** dynamic shapes or non-static tree storage.
- **Environment thread dominates:** moving only the search did not address the actual bottleneck.

---

## 7. Why dense trajectory planning may be better than MCTS

For Curvytron, the most promising planner may be a batched trajectory optimizer.

Sample or enumerate \(K\) macro-action sequences:

\[
U^{(k)}=(u_0^{(k)},\ldots,u_{H-1}^{(k)}).
\]

Roll them through an exact or learned model in a dense tensor:

\[
z_{h+1}^{(k)}
=
g(z_h^{(k)},u_h^{(k)},u_{h,\text{opp}}^{(k)}).
\]

Score each:

\[
J^{(k)}
=
\sum_{h=0}^{H-1}\Gamma^h \hat r_h^{(k)}
+
\Gamma^H V(z_H^{(k)}).
\]

Then use:

- beam search;
- cross-entropy method;
- Gumbel top-\(k\);
- sequential halving;
- particle/trajectory sampling;
- model-predictive control.

Advantages:

- dense \(B\times K\times H\) tensor operations;
- no adaptive pointer-chasing tree;
- no per-leaf host decisions;
- one or a few large launches;
- straightforward opponent trajectory sampling;
- easy fixed-shape compilation;
- natural receding-horizon reuse.

MCTS's adaptive allocation is valuable when some branches deserve far more computation than others. But it must earn that complexity empirically.

A good comparison is not “MCTS versus no planning.” It is:

1. policy only;
2. full small-depth enumeration;
3. beam search;
4. sampled MPC;
5. Gumbel MuZero;
6. conventional PUCT.

Compare win rate at equal GPU-seconds and equal action latency.

---

## 8. Long-term strategy architecture

### 8.1 Observation design

Do not begin with browser pixels. Use structured state rendered into neural-friendly tensors.

Recommended inputs:

#### Global low-resolution map

Channels for:

- borders;
- own mature trail;
- opponent mature trail;
- fresh non-collidable tail;
- player heads;
- current gaps;
- bonuses;
- optional trail owner/age.

#### Local high-resolution egocentric crop

Centered on the controlled head and rotated so forward is up. This gives precise collision geometry without making the global map enormous.

#### Vector features

- own position, heading, angular action, speed, radius;
- opponent relative position, heading, speed, and alive state;
- trail-printing state;
- gap process state if observable;
- active bonus effects and timers;
- score and round phase;
- borderless mode;
- strategic option or target, if using hierarchy.

Use symmetry augmentation: rotation, translation where valid, and reflection with left/right action swapping.

### 8.2 Network

A practical first network:

- small CNN for the global map;
- small CNN for the local crop;
- MLP or entity encoder for players and bonuses;
- recurrent core such as MinGRU, GRU, or LSTM;
- policy head;
- value or outcome-distribution head;
- auxiliary heads.

Do not use an Atari-scale residual tower by default. The action space and geometry are simple enough that throughput matters more than raw parameter count.

### 8.3 Auxiliary targets for strategy

Long-term strategy can be learned more reliably with auxiliary predictions:

- time-to-collision under current action;
- signed distance to nearest obstacle;
- curvature-constrained reachable free-space;
- number and width of escape corridors;
- future occupancy at several horizons;
- opponent turn/action distribution;
- probability of enclosure;
- probability of surviving the next \(n\) macro-steps;
- terminal win/rank distribution.

A particularly meaningful feature is reachable area in \((x,y,\theta)\), not ordinary flood-fill area. Curvytron is a Dubins-like vehicle: open space behind a tight corner may not be reachable at the current turning radius.

Prefer these as representation-learning targets. Direct dense reward shaping can create pathological strategies.

### 8.4 Optional hierarchy

Only after the recurrent baseline works, add a strategic option layer that selects, for example:

- target heading;
- target waypoint or sector;
- “expand,” “cut,” “escape,” or “shadow” mode;
- desired curvature over the next 0.5–2 seconds.

The low-level controller remains responsible for exact steering and safety. The strategic planner reasons in the option space.

---

## 9. Model-free baseline: why PufferLib is useful

PufferLib is highly relevant for the **baseline and simulator pipeline**, not as an MCTS implementation.

Its current native backend emphasizes:

- static contiguous allocations;
- CUDA Graph capture;
- asynchronous pinned CPU-GPU transfers;
- separate rollout buffers and CUDA streams;
- C environments with large contiguous observation/action/reward arrays;
- a fast PPO variant;
- a recurrent MinGRU-based default network.

That engineering is well aligned with a headless Curvytron simulator.

### Recommended use

1. Implement the environment as a Puffer Ocean-style C environment, or expose an equally compact C/C++ environment.
2. Keep thousands of independent matches in contiguous buffers.
3. Train a recurrent shared policy with PPO self-play.
4. Use historical checkpoints and explicit evaluation opponents.
5. Benchmark actual Curvytron throughput; do not assume generic PufferLib headline numbers transfer.

### Important limitations

PufferLib does not make MCTS GPU-native. Its built-in self-play pool is currently designed around two equal teams and its native CUDA backend. This fits 1v1 Curvytron, but not arbitrary free-for-all play without customization.

PufferLib and JAX `mctx` also belong to different hot-loop ecosystems. Crossing Puffer CUDA/Torch and JAX every decision can reintroduce synchronization. Prefer separate clean branches:

- **Production baseline:** PufferLib + C environment + recurrent PPO.
- **Search research:** JAX/Flax + JAX environment/model + `mctx`.
- **All-Torch search:** Torch model plus a custom static CUDA/Triton search implementation.

Do not combine all three in one per-frame loop without a measured zero-copy design.

---

## 10. Self-play design

Latest-policy self-play alone is not enough. It can cycle, forget counters, and overfit to transient opponent behavior.

Use a league containing:

- current mirror self-play;
- recent checkpoints;
- older checkpoints sampled across training history;
- scripted specialists;
- exploiters trained against the current main policy;
- behaviorally diverse policies.

Useful scripted opponents:

- conservative wall avoider;
- tight looper;
- aggressive cutter;
- opponent chaser;
- open-space maximizer;
- random action persistence;
- gap exploiter.

Track a full matchup matrix, not only one Elo number.

For two-player training, use terminal rewards:

\[
r_T\in\{-1,0,+1\}.
\]

Avoid a large positive reward for every survival tick; it can favor passive endless looping. Avoid naive territory rewards that can be farmed. If shaping is necessary, use small potential differences:

\[
r'_t=r_t+\gamma\Phi(s_{t+1})-\Phi(s_t),
\]

and still audit for multi-agent exploitation. Strategic quantities are often safer as auxiliary targets than as rewards.

For multiplayer, use a normalized rank or constant-sum transformation and evaluate kingmaking and coalition-like behavior separately.

---

## 11. Atari-style approaches: what transfers and what does not

### What transfers

- fixed action repeat;
- frame-independent fixed-step simulation;
- replay for world-model or value learning;
- recurrent agents for long horizons;
- target networks or reanalysis;
- distributional outcome modeling;
- large-scale vectorized actors;
- strong evaluation protocols;
- policy distillation from a stronger planner.

MuZero's Atari setup searched every fourth environment frame and repeated the chosen action four times. That is a highly relevant temporal-design lesson.

### What does not transfer directly

- raw RGB pixels when full structured state is available;
- treating the problem as a single-agent MDP;
- assuming deterministic latent dynamics in the presence of opponent actions and hidden randomness;
- reward clipping designed for heterogeneous Atari scores;
- huge generic convolutional towers;
- stale replay without opponent-policy/version information.

### Strong alternative baselines

#### Recurrent PPO

Best first choice when the simulator is cheap and opponent non-stationarity is severe.

#### R2D2/Rainbow-style recurrent Q-learning

Potentially strong because the action set is tiny and replay can improve sample efficiency. However, self-play makes old replay behavior and recurrent states stale. Store policy version and opponent checkpoint, limit replay age, and evaluate carefully.

#### Dreamer-style world model

Potentially more GPU-friendly than MuZero because actor/value learning occurs through dense imagined rollouts rather than online tree search. It still needs a joint-action or opponent-conditioned model. It is a serious model-based alternative if online MCTS remains expensive.

---

## 12. A clean MuZero branch

MuZero is justified only if one or more of these are true:

- an exact simulator is difficult to use inside planning;
- a compact latent model makes planning dramatically cheaper;
- search produces a measurable policy improvement;
- reanalysis improves sample efficiency;
- the learned model captures useful strategic abstractions.

It is **not** automatically justified merely because Curvytron is a game. The environment rules are known, and collision errors are catastrophic.

### 12.1 Model

Use:

\[
z_t=h(o_{\le t}),
\]

\[
(z_{t+1},\hat r_t)=g(z_t,a_t^1,a_t^2,c_t),
\]

\[
(\hat p_t^1,\hat p_t^2,\hat v_t)=f(z_t).
\]

If chance is hidden, either:

- introduce a stochastic/chance latent variable;
- use an afterstate/chance-node model;
- sample deterministic realizations;
- or defer random features.

### 12.2 Losses

A useful loss is:

\[
\mathcal{L}
=
\mathcal{L}_{\text{policy}}
+
c_v\mathcal{L}_{\text{value}}
+
c_r\mathcal{L}_{\text{reward}}
+
c_c\mathcal{L}_{\text{consistency}}
+
c_x\mathcal{L}_{\text{geometry}}.
\]

Geometry auxiliaries should include collision, occupancy, and perhaps future head position. MuZero's value-equivalent latent is not required to reconstruct the world; for this game, some grounding is valuable because search can exploit tiny geometric model errors.

### 12.3 Search

Use Gumbel MuZero before classic PUCT when the budget is small.

Starting research sweep:

- simulations: \(4,8,16,32\);
- max depth: \(4,8,12\) macro-actions;
- action repeat: \(2,4,6\);
- planner frequency: every control decision versus periodic/triggered;
- root batch: as large as the accelerator permits;
- model latent: compact vector or small spatial bottleneck.

Do not assume more simulations are better. Model exploitation can make additional search worse.

### 12.4 Simultaneous-node policy

Ordinary PUCT chooses one action as though the agent controls the transition. A simultaneous two-player node is a matrix game with estimates:

\[
Q(a^1,a^2).
\]

Possible practical solvers:

- opponent-policy expectation;
- maximin action;
- regret matching;
- Exp3/no-regret selection;
- approximate Nash mixture;
- sampled opponent trajectories with a pessimistic quantile.

Theoretical work on simultaneous-move MCTS shows that naive selection is not automatically equivalent to equilibrium play. Start with opponent-policy expectation for engineering simplicity, then evaluate exploitability against best-response-style opponents.

### 12.5 Reanalysis and distillation

Do not require online search for every actor state forever.

- Store trajectories.
- Re-run the latest planner on a selected subset of states.
- Train the policy toward the improved action distribution.
- Train the value toward updated search/outcome targets.
- Let actors increasingly use the distilled policy.

This is often a better use of search compute than blocking every real-time action.

---

## 13. PPO, imitation, and search must not be mixed incorrectly

There are three clean training contracts.

### Contract A: PPO behavior

Actions are sampled from \(\pi_{\text{old}}\). Store their exact log probabilities and use PPO ratios.

### Contract B: MuZero/AlphaZero behavior

Actions come from search. Train the policy by cross-entropy to the search distribution and train value/reward/model heads. Do not pretend this is PPO.

### Contract C: explicit behavior mixture

Behavior is:

\[
\mu(a\mid s)
=
(1-\alpha)\pi(a\mid s)+\alpha\pi_{\text{plan}}(a\mid s).
\]

Store or reconstruct \(\log\mu(a\mid s)\), and use an off-policy correction such as V-trace or an explicitly derived objective.

A planner action with no valid behavior probability cannot be placed into vanilla PPO as though it came from the old policy.

A safe hybrid is:

- PPO rollouts remain policy-generated;
- planner states become supervised imitation examples;
- planner value becomes an auxiliary target;
- online planner is used for evaluation or a controlled fraction of off-policy data.

---

## 14. Recommended implementation phases

## Phase 0: rules and simulator

Deliverables:

- fixed \(1/60\) timestep;
- seeded PRNG;
- explicit simultaneous collision semantics;
- deterministic reset;
- headless vectorized environment;
- legacy compatibility tests;
- throughput benchmark;
- scripted opponents.

Golden tests:

- constant straight motion;
- constant left/right circles;
- border collision;
- self-trail latency;
- opponent-trail collision;
- same-frame head/trail collision;
- simultaneous death/tie;
- gap start/end;
- deterministic replay from seed;
- player-index symmetry.

## Phase 1: Puffer PPO baseline

- 2-player core game;
- action repeat sweep;
- global map + local crop + vectors;
- recurrent shared policy;
- sparse terminal outcome;
- latest self-play plus historical pool;
- scripted-opponent evaluation.

Do not proceed until the agent reliably:

- avoids simple walls;
- escapes open corridors;
- defeats random and scripted policies;
- generalizes across spawn seeds;
- remains symmetric across player slots.

## Phase 2: strategic learning

Add:

- reachability and collision auxiliary heads;
- opponent intent prediction;
- multi-horizon value/outcome targets;
- stronger league;
- strategic metrics;
- curriculum for random gaps and bonuses.

## Phase 3: dense planner

Implement batched macro-action beam/CEM/MPC using:

- exact simulator if practical;
- otherwise a jointly conditioned learned model;
- terminal learned value;
- sampled opponent sequences;
- fixed-shape GPU batches.

## Phase 4: Gumbel MuZero experiment

Only after Phase 3 provides a fair planning baseline:

- JAX-native end-to-end branch;
- static batches;
- explicit `max_depth`;
- joint-action model;
- small simulation count;
- reanalysis;
- planner distillation;
- stochastic extension only when deterministic core is solved.

## Phase 5: multiplayer

- factorized opponent model;
- sampled joint actions;
- custom league and matchmaking;
- rank/constant-sum reward;
- coalition and kingmaking diagnostics;
- no assumption that PufferLib's two-team self-play helper applies directly.

---

## 15. Decision gates

A planner should remain in the system only if it passes all of these.

### Strength gate

At the same network checkpoint, planned play must beat policy-only play reliably across seeds and opponent classes.

### Compute gate

The gain must survive comparison at equal:

- GPU-seconds;
- training wall-clock;
- action latency;
- energy or cost budget.

### Model gate

For horizons \(1,4,8,16\) macro-steps, measure:

- head-position error;
- heading error;
- collision precision and recall;
- death-time error;
- occupancy error;
- reward/outcome calibration;
- opponent-action prediction;
- reachable-area error.

Collision false negatives are particularly dangerous because search will seek them out.

### Robustness gate

Search should not collapse against:

- policies outside the training league;
- adversarial action sequences;
- unusual gap schedules;
- mirrored/rotated maps;
- player-slot swaps;
- long loops and near-ties.

### Systems gate

The search hot path should have:

- no per-simulation D2H copies;
- static shapes;
- bounded depth;
- no per-search allocator churn;
- sufficiently large root batches;
- stable latency;
- a clear utilization profile.

If online search fails the compute gate but improves targets, retain it as an offline reanalysis teacher.

---

## 16. Recommended default architecture

```text
                    ┌──────────────────────────────┐
60 Hz fixed physics │ Exact deterministic C env    │
                    │ seeded RNG, two-phase update │
                    └──────────────┬───────────────┘
                                   │ every 2–6 ticks
                    ┌──────────────▼───────────────┐
                    │ Multiscale recurrent policy  │
                    │ global map + local crop      │
                    │ vectors + MinGRU/GRU         │
                    └───────┬──────────────┬───────┘
                            │              │
                    fast action      value/uncertainty
                            │              │
                            │     periodic or critical
                            │              ▼
                            │    ┌──────────────────────┐
                            │    │ Dense macro planner  │
                            │    │ beam/CEM/MPC first   │
                            │    │ joint opponent model │
                            │    └──────────┬───────────┘
                            └───────────────▼
                                    chosen control

Training:
Puffer PPO self-play → checkpoint league → strategic auxiliaries
                         │
                         └→ selected-state reanalysis/planner distillation
```

The MuZero branch can replace the dense planner later, but it should be compared against this architecture rather than assumed to be the destination.

---

## 17. Bottom line

The likely correct answer for Curvytron is **not pure PPO, pure MCTS, or a direct copy of Atari MuZero**.

It is:

- exact fixed-step simulation;
- recurrent population self-play;
- multiscale spatial representation;
- long-horizon learned value;
- temporal abstraction;
- short explicit planning over meaningful action chunks;
- explicit opponent and chance modeling;
- dense accelerator-friendly planning where possible;
- Gumbel MuZero only after the simpler planner and model-free baseline are strong;
- reanalysis and imitation to amortize planning;
- hard profiling and ablation gates.

The most likely current mistake is trying to make an inherently sequential, irregular, raw-timestep tree look like a dense GPU workload. First remove true synchronization bugs. Then test whether the remaining sequential cost is fundamental. If it is, change the planner—not merely the kernel.

---

## References

- [Curvytron public source](https://github.com/Curvytron/curvytron)
- [MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)
- [Mctx: MCTS in JAX](https://github.com/google-deepmind/mctx)
- [Policy Improvement by Planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO)
- [Planning in Stochastic Environments with a Learned Model](https://openreview.net/forum?id=X6D9bAHhBQ1)
- [Convergence of MCTS in Simultaneous Move Games](https://arxiv.org/abs/1310.8613)
- [PufferLib documentation](https://puffer.ai/docs.html)
- [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
