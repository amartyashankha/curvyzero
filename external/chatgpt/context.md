# CurvyZero: Algorithm and Systems Strategy

_Living design document · Updated 2026-06-23_

### Revision 2: long-horizon planning and exploration

This revision adds a concrete multi-timescale planning design, a critical analysis of Random Network Distillation, an NGU-inspired episodic novelty design, model-disagreement exploration, simulator-state archives, PufferLib integration notes, and explicit ablation gates. The recommendation is deliberately conservative: **do not add intrinsic reward until measurements show that state-space exploration—not credit assignment, self-play cycling, or model error—is the bottleneck.**

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
8. Implement long-term planning as a **hierarchy**: a terminal-outcome value, multi-horizon strategic predictions, a slow goal/option policy, and a fast exact safety controller. Search over seconds-long commitments, then replan.
9. Treat RND as an optional ablation, not the default exploration mechanism. For Curvytron, **league diversity, episodic strategic novelty, model-ensemble disagreement, and resettable hard-state archives** are usually better aligned with the real failure modes.

The central diagnosis is:

> The expensive synchronization may be partly an implementation bug, but ordinary MCTS is also structurally mismatched to a small-batch, 60 Hz, simultaneous-control workload. Moving the same algorithm to the GPU does not automatically make it GPU-shaped.

> Long-term strategy does not require a raw-tick tree reaching the end of the round. It requires a value function that understands terminal consequences, strategic state abstractions, temporally extended actions, and enough explicit lookahead to choose among commitments whose consequences unfold over seconds.

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

### 4.4 Discount in physical time, not decision count

A fixed numerical discount has a different meaning at 60 Hz, 15 Hz, and a variable-duration option level. Define discount from a physical half-life:

\[
\gamma(\Delta t)
=
2^{-\Delta t/T_{1/2}}.
\]

At a 15 Hz control rate, examples are:

| Return half-life | Per-control-step \(\gamma\) |
|---:|---:|
| 5 seconds | 0.9908 |
| 10 seconds | 0.9954 |
| 20 seconds | 0.9977 |
| 30 seconds | 0.9985 |

These are calibration examples, not recommended constants. A very high discount increases variance and critic difficulty. Curvytron can avoid much of this trade-off by separating:

\[
V_{\infty}(s)=\mathbb{E}[z\mid s],
\qquad z\in\{-1,0,+1\},
\]

an undiscounted final-outcome prediction, from shorter-horizon survival, collision, and territory heads.

For an option lasting \(d\) physics ticks, always use:

\[
\Gamma(d)=\gamma_{\text{physics}}^d.
\]

Do not accidentally discount a two-tick and a sixteen-tick option equally.

### 4.5 Planning horizon should be measured in seconds

Report all planning experiments in both abstract steps and physical time. A tree of depth 12 means almost nothing without the action duration:

- depth 12 at 60 Hz covers 0.2 seconds;
- depth 12 with four-tick actions covers 0.8 seconds;
- depth 8 with 0.5-second options covers 4 seconds.

For Curvytron, a useful planner should normally combine:

- exact tactical prediction over roughly 0.2–1.0 seconds;
- strategic option prediction over roughly 2–10 seconds;
- a terminal-outcome value for everything beyond the explicit horizon.

### 4.6 Receding-horizon commitment

Planning should choose a *commitment*, not permanently surrender control. If the planner selects option \(u_t\) for nominal duration \(d\), the controller may replan early when:

\[
\text{risk}(s_t) > \rho,
\quad
\text{uncertainty}(s_t) > \upsilon,
\quad
\text{or}
\quad
\text{option-feasibility}(s_t,u_t)=0.
\]

This preserves strategic intent while allowing millisecond-scale reactions to a new trail, gap, or opponent turn.

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

### 8.5 What long-term planning should mean here

A useful Curvytron planner has four layers of foresight:

1. **Immediate safety:** exact collision checks for candidate actions over the next few tenths of a second.
2. **Route commitment:** select a curvature pattern, target heading, waypoint, or corridor over the next 0.5–2 seconds.
3. **Strategic topology:** estimate how the choice changes reachable territory, escape routes, and interception opportunities over several seconds.
4. **Terminal consequence:** estimate final win/loss probability after the explicit planning horizon.

The planner does not need to render every future frame until the round ends. The decomposition is:

\[
Q(s,u)
\approx
R_{0:H}(s,u,u^-)
+
\Gamma_H V_{\infty}(s_H),
\]

where \(u^-\) denotes an opponent option or sampled opponent trajectory. The hard part is ensuring that \(V_{\infty}\) and the strategic state at \(s_H\) preserve the consequences of enclosure, route loss, and future turning room.

### 8.6 Multi-horizon value and hazard heads

One scalar critic is a weak training signal for a game in which the final outcome can be tens of seconds away. Use a shared representation with several heads:

\[
V_{\infty}(s)=\mathbb{E}[z\mid s]
\]

\[
S_h(s)=P(\text{alive at }t+h\mid s),
\qquad
h\in\{0.5,1,2,4,8\}\text{ seconds}
\]

\[
C_h(s)=P(\text{collision before }t+h\mid s)
\]

\[
W_h(s)=P(\text{eventual win}\mid s,\text{survive to }t+h).
\]

An equivalent formulation predicts a discrete time-to-death or outcome distribution. These heads improve representation learning, provide calibrated planning cutoffs, and expose whether the agent lacks short-term safety or long-term strategy.

Do not sum all heads into the environment reward. They are supervised predictions and planner inputs.

### 8.7 Start with interpretable goals, not unconstrained latent options

A hierarchical policy can be written as:

\[
g_k\sim\pi_H(g\mid z_{kK}),
\]

\[
a_t\sim\pi_L(a\mid o_t,g_k),
\qquad kK\le t<(k+1)K.
\]

The first goal space should be easy to inspect and simulate:

- target heading and curvature;
- target point or map sector;
- target corridor mouth;
- desired clearance from obstacles;
- an opponent-relative intercept point;
- a route represented by two or three waypoints.

A fully learned latent goal space is attractive, but it creates three problems at once: semantic drift, option collapse, and a search space whose geometry is unknown. Director-style latent goals are a valuable later experiment, not the first implementation.

Use fixed option durations initially. Learned termination is useful only after options are stable; otherwise the high-level action space changes while the planner and critic are learning it.

A second non-stationarity appears when the low-level controller improves: the same goal no longer induces the same state transition. Prefer geometrically defined goals, train the high level mostly on-policy, update the low level more slowly, or relabel old high-level transitions using the behavior low-level policy. Off-policy high-level replay without such correction can learn from an action space that no longer exists.

### 8.8 Multi-timescale dynamics models

A one-step model repeatedly unrolled for hundreds of physics ticks accumulates error and is expensive. Train skip models at several temporal scales:

\[
G_K(z_t,u_t,u_t^-)
\rightarrow
(\hat z_{t+K},\hat R_{t:t+K},\hat d_{t:t+K}),
\]

for, for example, \(K\in\{4,16,64\}\) physics ticks. The targets should include:

- next strategic latent state;
- cumulative reward and discount;
- collision/death event;
- head pose and heading;
- coarse future trail occupancy;
- change in reachable-space topology.

Use consistency or latent-overshooting losses so the long skip model agrees with encoded future observations:

\[
\mathcal{L}_{\text{skip}}
=
\left\|G_K(z_t,u_t,u_t^-)-\operatorname{sg}(h(o_{t+K}))\right\|^2.
\]

The exact simulator should remain the source of truth for immediate collision geometry. The learned coarse model is for ranking strategic routes, not certifying safety.

### 8.9 Strategic maps and topological state

A global occupancy image is necessary but not sufficient. Add derived representations that make future topology easier to learn:

#### Curvature-constrained reachability

Estimate the free states reachable by a Dubins-like vehicle in \((x,y,\theta)\), not by an unconstrained flood fill.

#### Earliest-arrival fields

For player \(i\), estimate:

\[
T_i(x,y)=\text{earliest feasible arrival time}.
\]

Then an influence field can be approximated by:

\[
I(x,y)=T_{\text{opp}}(x,y)-T_{\text{self}}(x,y).
\]

Positive margin suggests controllable space; a sign change identifies contested boundaries.

#### Corridor graph

Compress free space into regions connected by narrow passages. Candidate options can target corridor entrances, and the planner can reason about cutting an edge rather than individual pixels.

#### Future occupancy

Predict discounted trail occupancy maps:

\[
M(s)=\mathbb{E}\left[\sum_{k\ge0}\gamma^k\phi(s_k)\mid s\right].
\]

Separate own and opponent occupancy. This is close to a spatial successor representation and directly exposes enclosure and interception consequences.

These maps are best introduced as observations and auxiliary targets. If used as dense rewards, they can be gamed.

### 8.10 Robust opponent-aware planning

A plan that succeeds only when the opponent follows its modal action is brittle. For every candidate self plan \(U\), evaluate several plausible opponent plans \(U^-_m\):

\[
J_m(U)=J(U,U^-_m).
\]

Possible robust objectives are:

\[
J_{\text{robust}}(U)
=
\frac{1}{M}\sum_m J_m(U)-\lambda\operatorname{Std}_m[J_m(U)],
\]

or a lower-tail conditional value at risk:

\[
J_{\text{robust}}(U)=\operatorname{CVaR}_{\alpha}\{J_m(U)\}.
\]

For 1v1 with three primitive actions, the first joint-action step has only nine possibilities and can often be evaluated exactly. Farther ahead, sample opponent options from:

- the current opponent policy;
- historical opponent checkpoints;
- an opponent-intent ensemble;
- adversarial scripted responses.

Pure maximin planning can become excessively conservative; pure expectation can be exploitable. Compare both against a risk-adjusted middle ground.

### 8.11 A practical two-level planner

A concrete implementation is:

#### Strategic planner, 1–3 Hz

- state: coarse global map, reachability fields, opponent intent, recurrent memory;
- action: waypoint, target heading, corridor, or 0.5–2 second curvature option;
- horizon: roughly 3–10 seconds;
- search: beam search, categorical CEM, Gumbel top-\(k\), or small option-level MCTS;
- transition: coarse exact rollout where possible, otherwise a skip model;
- leaf: terminal-outcome value plus calibrated risk.

#### Tactical controller, 15–30 Hz

- follows the selected option;
- uses the high-resolution local crop;
- checks exact short-horizon collisions;
- can veto an unsafe option;
- triggers early replanning.

A safety veto should choose the safest feasible primitive action, not silently change the training target into the planner action. Log veto frequency; frequent vetoes mean the high-level model is untrustworthy.

Use territory, corridor width, and reachability primarily as constraints, auxiliary predictions, or tie-breakers. Adding all of them to the planner score can double-count information already represented in the terminal value and create a hand-shaped policy that wins the heuristic rather than the game. Every additive planning heuristic needs an ablation against outcome-only leaf values.

Variable-duration options also require semi-Markov accounting. Otherwise the planner may prefer short options merely because they create more branch points, or long options merely because they incur fewer decision penalties. Compare plans at equal physical horizon and use duration-aware discounting.

### 8.12 Reuse plans and search only where leverage is high

Warm-start the next planning call by shifting the previous sequence forward. Preserve beams or tree statistics when the root transition is consistent with the predicted next state.

Define strategic leverage as sensitivity of outcome value to available commitments:

\[
L(s)=\max_u Q(s,u)-\min_u Q(s,u).
\]

Spend more planning compute when leverage, uncertainty, or collision risk is high. Low-leverage open-space cruising should normally use the policy directly.

### 8.13 Long-horizon benchmark scenarios

Average self-play Elo can hide a policy that never learned delayed consequences. Build deterministic scenario tests such as:

- a wide corridor that becomes a dead end only after 4–8 seconds;
- an early escape turn needed to avoid later enclosure;
- a short-term loss of free area that enables an opponent cut;
- a choice between a safe loop and a risky intercept;
- a corridor whose owner is determined by earliest arrival, not nearest Euclidean distance;
- a gap that enables a route only if approached several seconds early;
- an adversary that deliberately baits a greedy space-maximizing policy.

For each scenario, report success as a function of explicit planning horizon, value-network checkpoint, opponent model, and compute budget. This directly measures long-term planning rather than hoping Elo reveals it.

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

### Current PufferLib integration caveat

The current PufferLib 4.0 Torch backend stores a single reward stream and a single value stream in its rollout buffers, and its GPU environment path calls `torch.cuda.synchronize()` after each environment step. A dual-stream intrinsic/extrinsic critic therefore requires a fork of the rollout buffers, model outputs, advantage calculation, and loss. The Torch backend is useful for prototyping, but it should not be the production hot path for GPU planning.

The native backend is architecturally better suited to static buffers and CUDA Graphs, but adding RND or episodic neural novelty inside the captured path still requires deliberate static allocation. Avoid Python callbacks or per-step tensor creation; they discard the systems advantage PufferLib was chosen for.

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

## 11. Exploration: RND and better alternatives

### 11.1 Diagnose the failure before adding novelty

Curvytron always supplies a terminal outcome, so it is not necessarily a hard-exploration problem in the Montezuma's-Revenge sense. A policy can fail despite visiting a broad set of states because the true bottleneck is delayed credit, self-play cycling, weak opponent diversity, or model error.

| Observed symptom | Likely bottleneck | First intervention |
|---|---|---|
| Agents repeat a tiny set of loops and never enter large safe regions | State/behavior exploration | Episodic strategic novelty or simple counts |
| Agents visit broadly but choose locally safe, globally losing routes | Credit and hierarchy | Multi-horizon values and option planning |
| Training alternates among rock-paper-scissors-like tactics | Opponent-policy exploration | Historical league, exploiters, PSRO-like mixtures |
| Planner fails mainly in rare collision configurations | Model coverage | Joint-action model ensemble and targeted data |
| Strong self-play policy loses to simple held-out scripts | Overfitting | Opponent population and best-response evaluation |
| Intrinsic score rises while external win rate stagnates | Objective misalignment | Reduce/remove intrinsic reward |

Collect these diagnostics before deciding that RND is needed.

### 11.2 RND, carefully stated

RND uses a fixed random target network \(f_{\xi}\) and a trainable predictor \(\hat f_{\psi}\). For an input representation \(x_t\):

\[
r_t^{\text{RND}}
=
\left\|\hat f_{\psi}(x_t)-f_{\xi}(x_t)\right\|_2^2,
\]

and the predictor minimizes:

\[
\mathcal{L}_{\text{RND}}(\psi)
=
\mathbb{E}_{x\sim\mathcal{D}}
\left[\left\|\hat f_{\psi}(x)-f_{\xi}(x)\right\|_2^2\right].
\]

States unlike the predictor's training data tend to retain larger error. A minimal correct implementation needs:

- frozen target parameters;
- normalized inputs to target and predictor;
- normalized or clipped intrinsic returns;
- a predictor update rate controlled independently of actor count;
- a small replay reservoir or old-state audit to detect predictor forgetting;
- separate intrinsic and extrinsic value estimates;
- capped or suppressed terminal-frame novelty so rare deaths do not become attractive;
- no intrinsic reward during evaluation or checkpoint selection.

For PPO, use:

\[
\delta_t^{E}
=r_t^{E}+\gamma_E V_E(s_{t+1})-V_E(s_t),
\]

\[
\delta_t^{I}
=r_t^{I}+\gamma_I V_I(s_{t+1})-V_I(s_t),
\]

and combine advantages only after computing them separately:

\[
A_t=A_t^E+\beta A_t^I.
\]

A single critic for the sum is possible but less stable because the extrinsic target is stationary while novelty decays as the predictor learns.

### 11.3 What should enter the novelty encoder?

Do not feed arbitrary raw state directly. Construct a canonical strategic input \(x_t=\phi(s_t)\) containing:

- a coarse occupancy map in an egocentric or symmetry-canonical frame;
- own pose and heading;
- opponent-relative geometry;
- curvature-constrained reachable-space summary;
- corridor and enclosure features;
- gap/bonus state only when it is observable and strategically controllable.

Exclude or canonicalize:

- player IDs and colors;
- absolute rotations or reflections that are strategically equivalent;
- wall-clock time;
- random seeds;
- unobservable hidden timers;
- high-frequency animation or rendering noise.

Compute novelty at the control or strategic rate, not every 60 Hz physics tick. Otherwise a smooth turn produces many almost-identical bonuses and the intrinsic return becomes dominated by duration.

### 11.4 Why pure lifelong RND is a questionable default

### The arena is finite

Once the predictor covers common occupancy patterns, lifelong novelty can disappear even though the agent still needs to practice difficult counterstrategies. Conversely, changing opponents can continually manufacture rare joint configurations that are not under the agent's control.

### Novel geometry is not necessarily useful strategy

RND can reward:

- unusual spirals;
- narrow but losing corridors;
- novel death locations;
- rapidly changing trail patterns;
- encounters generated by eccentric opponents.

The random target knows nothing about win probability, reachability, or strategic leverage.

### Intrinsic rewards break the zero-sum game

If both players receive positive state novelty, the training objective becomes general-sum. Agents can implicitly cooperate to visit unusual states or synchronize into elaborate but strategically weak behavior. Keep the main game value purely extrinsic and evaluate only the \(\beta=0\) policy.

### Scale changes the bonus lifetime

More actors generate more predictor training data. If predictor batch size or update count scales automatically with environment throughput, novelty may collapse much faster. The original RND work explicitly controlled predictor training rate across different actor counts.

### Randomness can still be a trap

RND avoids predicting a stochastic next state because its target is deterministic. It does not prevent rare random observations, random bonuses, or opponent-generated patterns from looking novel. A random target over nuisance features is still a nuisance detector.

### Predictor forgetting can manufacture novelty

If the predictor is trained only on the newest rollout distribution, it can forget old regions and assign them high error again. That may look like healthy recurrent exploration while actually producing cycles. Track prediction error on a fixed probe set and use a modest reservoir of historical strategic embeddings when forgetting is substantial.

### 11.5 Recommended exploration stack

The recommended order is:

### 1. Opponent-policy exploration

Use a historical league, exploiters, scripted specialists, and approximate best responses to mixtures. This explores *strategy space*, which is the most important source of novelty in a competitive game.

A PSRO-like loop is conceptually:

1. maintain a population \(\Pi=\{\pi_1,\ldots,\pi_n\}\);
2. estimate the empirical payoff matrix;
3. compute an opponent mixture \(\sigma\);
4. train an approximate best response to \(\sigma\);
5. add it to the population.

Full PSRO may be more machinery than needed, but the principle—train against mixtures rather than only the latest policy—is directly relevant. Empirical payoff estimates can be noisy, so retain held-out opponents and do not let the same matchup matrix both choose the meta-strategy and certify progress.

### 2. Episodic strategic novelty

Reset novelty memory every round. Let \(e_t=\phi(s_t)\) be a controllable strategic embedding and \(\mathcal{M}_t\) the current round's memory. A simple kernel count is:

\[
n_t
=
\sum_{m\in\mathcal{M}_t}
K\!\left(\frac{\|e_t-m\|^2}{\sigma^2}\right),
\]

\[
r_t^{\text{epi}}
=
\frac{1}{\sqrt{n_t+\epsilon}}.
\]

This encourages new routes *within the current arena* without permanently declaring useful strategic motifs boring. An inverse-dynamics or action-prediction objective can bias the embedding toward controllable changes, as in NGU-style exploration.

For Curvytron, inverse dynamics should account for simultaneous play. Predict our action from own pose change and local geometry, and either condition on the opponent action or prevent opponent-only motion from dominating the embedding.

### 3. Model-disagreement exploration

Train an ensemble of joint-action dynamics or outcome models \(g_j\). A disagreement score can be:

\[
u_t
=
\operatorname{Var}_{j}
\left[g_j(z_t,a_t,a_t^-)\right].
\]

Prefer disagreement over controllable quantities:

- head pose;
- collision probability;
- trail occupancy;
- reachable-area change;
- reward/outcome logits.

Use this uncertainty primarily for:

- selecting replay samples;
- choosing simulator start states;
- deciding when to invoke planning;
- collecting targeted exploratory rollouts;
- rejecting untrustworthy long-horizon plans.

Directly rewarding raw disagreement can recreate a noisy-TV problem when the opponent or chance process is unpredictable. Joint-action conditioning and an explicit stochastic model help separate epistemic uncertainty from aleatoric randomness.

For explorer-conditioned actors, the strategic planner can seek *expected future* epistemic uncertainty rather than paying novelty only after arrival:

\[
J_{\text{explore}}(U)
=
J_E(U)
+
\beta
\sum_{h=0}^{H-1}\Gamma^h u(z_h).
\]

This is a Plan2Explore-like use of the world model. Keep it restricted to explorer policies; the production policy should plan with \(\beta=0\).

### 4. A resettable hard-state archive

The simulator can save exact reachable snapshots. Archive states that are:

- strategically novel;
- high leverage;
- high value/model disagreement;
- near an enclosure transition;
- frequently mishandled;
- rare under the current league.

Sample archived states as curriculum starts, preserving:

- all trails and trail ages;
- player poses and active effects;
- gap/bonus timers;
- RNG state;
- score and round phase;
- recurrent burn-in context or enough state to reconstruct it.

This is a Curvytron-friendly version of “first return, then explore.” It avoids repeatedly replaying a long easy prefix merely to practice one rare strategic decision.

The archive must contain only actually reachable states. Always validate improvements from full-game starts because an agent can overfit to an artificial reset distribution.

### 5. Optional lifelong RND as a gate

If episodic novelty repeatedly rewards the same globally familiar states, combine it with a bounded lifelong term:

\[
r_t^I
=
r_t^{\text{epi}}
\left(1+\eta\,\operatorname{clip}(\tilde r_t^{\text{RND}},0,L)\right).
\]

This is preferable to simply adding two unbounded novelty bonuses. RND says whether a state is globally unfamiliar; episodic novelty says whether the current route is new within this round.

### 11.6 A cheaper baseline than RND

Before adding two neural networks, try a count over a coarse strategic descriptor:

\[
d(s)=
(\text{position bin},
\text{heading bin},
\text{reachable-area bin},
\text{corridor-count bin},
\text{opponent-relative bin},
\text{enclosure-risk bin}).
\]

Then:

\[
r_t^{\text{count}}
=
\frac{1}{\sqrt{N_{\text{episode}}(d(s_t))+1}}.
\]

Use a hash or random projection if the descriptor is large. This is:

- cheap enough to implement in the C environment;
- easy to inspect;
- naturally episodic;
- less likely to reward irrelevant pixel novelty;
- a strong test of whether exploration is actually the bottleneck.

If this does not improve downstream extrinsic strength, a more complex RND module probably will not fix the underlying problem.

### 11.7 Exploration-conditioned policies

Do not force one policy to use one fixed intrinsic weight. Condition the policy and critics on an exploration setting:

\[
\pi(a\mid s,\beta),
\qquad
V_E(s,\beta),
\qquad
V_I(s,\beta).
\]

Run a family ranging from \(\beta=0\) to strongly exploratory. The exact weights must be selected after reward normalization; raw constants are not portable.

Practical rules:

- keep a large fraction of actors at \(\beta=0\);
- sample exploratory settings independently for the two players;
- do not use intrinsic reward in Elo or match outcomes;
- distill discoveries into the \(\beta=0\) policy through shared representation, imitation, replay, or curriculum;
- adapt actor allocation based on external learning progress, not intrinsic return.

This is closer to the NGU/Agent57 idea of learning a family of exploration policies than to globally annealing one coefficient.

### 11.8 Techniques that are less suitable

### Raw forward-prediction curiosity

Opponent actions, gap randomness, and bonuses make prediction error high for reasons unrelated to useful novelty. Without careful stochastic modeling, the policy may seek unpredictability.

### RIDE-style raw impact reward

Every Curvytron step extends or changes a trail. Rewarding representational change can therefore pay the agent simply for drawing more unusual geometry, even when it reduces winning chances.

### Pure state-entropy maximization

RE3-style entropy is simple and stable, but maximum state entropy is not the game objective. It is a useful ablation or pretraining signal, especially on a canonical strategic embedding, not a replacement for competitive self-play.

### Symmetric intrinsic self-play

Giving both copies of the same policy the same novelty objective can create correlated loops and implicit collusion. Randomize exploration settings and use frozen or historical opponents for explorer actors.

### 11.9 PufferLib implementation sketch

A clean PPO integration needs the policy to output:

```text
policy logits
extrinsic value V_E
intrinsic value V_I
optional strategic embedding e(s)
```

Rollout storage needs:

```text
extrinsic reward
intrinsic reward
old V_E and V_I
beta/exploration-policy id
per-environment episodic memory state
```

Training computes separate advantages and then combines them for the policy loss. Predictor loss is optimized separately:

\[
\mathcal{L}
=
\mathcal{L}_{\text{PPO}}
+c_E\mathcal{L}_{V_E}
+c_I\mathcal{L}_{V_I}
+c_R\mathcal{L}_{\text{RND}}
-c_H\mathcal{H}.
\]

For the native PufferLib backend, preserve its static-memory design:

- allocate all novelty buffers at initialization;
- use fixed embedding and memory sizes;
- avoid dynamic k-nearest-neighbor structures in Python;
- use a fixed-size episodic reservoir or hashed count table;
- include the exploration-policy ID in the observation;
- capture predictor/value work in stable CUDA graphs only after shapes are fixed.

The lowest-risk first implementation is a C-side strategic count bonus. The next is a GPU RND predictor computed in batches after rollout collection. Online neural episodic kNN across thousands of environments is the most systems-complex option and should come last.

### 11.10 Exploration ablation matrix

Run at least:

| Variant | League | Episodic novelty | Lifelong novelty | Model uncertainty | Archive |
|---|---:|---:|---:|---:|---:|
| A | latest only | no | no | no | no |
| B | historical | no | no | no | no |
| C | historical | strategic count | no | no | no |
| D | historical | kNN | no | no | no |
| E | historical | no | RND | no | no |
| F | historical | kNN | bounded RND | no | no |
| G | historical | kNN | optional | ensemble | no |
| H | historical | kNN | optional | ensemble | yes |

Measure:

- held-out win rate and payoff matrix;
- exploitability proxy against trained best responses;
- strategic descriptor coverage;
- diversity of trajectories and options;
- terminal outcome calibration;
- model error in rare states;
- early-suicide and passive-loop rates;
- fraction of policy advantage coming from intrinsic reward;
- strength of the \(\beta=0\) policy after intrinsic rewards are disabled.

The go/no-go criterion is **external competitive strength**, not novelty score.

---

## 12. Atari-style approaches: what transfers and what does not

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

## 13. A clean MuZero branch

MuZero is justified only if one or more of these are true:

- an exact simulator is difficult to use inside planning;
- a compact latent model makes planning dramatically cheaper;
- search produces a measurable policy improvement;
- reanalysis improves sample efficiency;
- the learned model captures useful strategic abstractions.

It is **not** automatically justified merely because Curvytron is a game. The environment rules are known, and collision errors are catastrophic.

### 13.1 Model

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

### 13.2 Losses

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

### 13.3 Search

Use Gumbel MuZero before classic PUCT when the budget is small.

Starting research sweep:

- simulations: \(4,8,16,32\);
- max depth: \(4,8,12\) macro-actions;
- action repeat: \(2,4,6\);
- planner frequency: every control decision versus periodic/triggered;
- root batch: as large as the accelerator permits;
- model latent: compact vector or small spatial bottleneck.

Do not assume more simulations are better. Model exploitation can make additional search worse.

### 13.4 Simultaneous-node policy

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

### 13.5 Reanalysis and distillation

Do not require online search for every actor state forever.

- Store trajectories.
- Re-run the latest planner on a selected subset of states.
- Train the policy toward the improved action distribution.
- Train the value toward updated search/outcome targets.
- Let actors increasingly use the distilled policy.

This is often a better use of search compute than blocking every real-time action.

---

## 14. PPO, imitation, and search must not be mixed incorrectly

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

## 15. Recommended implementation phases

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

## Phase 2: strategic learning and diagnosis

Add:

- reachability and collision auxiliary heads;
- opponent intent prediction;
- multi-horizon survival and terminal-outcome heads;
- stronger historical league and exploiters;
- long-horizon scenario benchmarks;
- strategic coverage and self-play cycling diagnostics.

Do not add intrinsic reward until these measurements distinguish under-exploration from delayed credit or opponent overfitting.

## Phase 2B: exploration experiments

In this order:

1. strategic episodic counts;
2. resettable archive of reachable hard states;
3. model-ensemble disagreement for data selection and planning triggers;
4. NGU-like episodic kNN;
5. bounded RND as an optional lifelong gate.

Keep a \(\beta=0\) policy and evaluate only external outcomes.

## Phase 3: hierarchical dense planner

Implement batched macro-action beam/CEM/MPC using:

- interpretable options or waypoint goals;
- an exact short-horizon simulator;
- multi-timescale skip models for longer horizons;
- terminal learned value;
- sampled historical and current opponent sequences;
- risk-adjusted scoring;
- fixed-shape GPU batches;
- warm starts and event-triggered replanning.

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

## 16. Decision gates

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

### Exploration gate

An intrinsic mechanism remains only if it improves the externally evaluated \(\beta=0\) policy across held-out opponents. Reject it when it primarily increases:

- novelty return;
- trajectory entropy;
- unusual death states;
- collusive behavior;
- model disagreement caused by opponent or chance noise.

Require ablations at matched environment steps and wall-clock time. Also measure performance after intrinsic reward is turned off; otherwise the policy may depend on a training-only objective.

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

## 17. Recommended default architecture

```text
                           TRAINING POPULATION
      ┌──────────────────────────────────────────────────────────┐
      │ β=0 main policy │ historical league │ explorer policies │
      │ exploiters      │ scripted bots     │ archived states   │
      └───────────────────────────┬──────────────────────────────┘
                                  │
                    ┌─────────────▼────────────────┐
60 Hz fixed physics │ Exact deterministic C env    │
                    │ seeded RNG, two-phase update │
                    └─────────────┬────────────────┘
                                  │ every 2–6 ticks
                    ┌─────────────▼────────────────┐
                    │ Multiscale recurrent policy  │
                    │ global + local + strategic   │
                    │ policy, V∞, Vh, uncertainty  │
                    └───────┬──────────────┬───────┘
                            │              │
                    fast primitive   1–3 Hz or event trigger
                            │              │
                 ┌──────────▼───┐   ┌──────▼──────────────────┐
                 │ Exact safety │   │ Hierarchical planner     │
                 │ short rollout│   │ options/waypoints        │
                 └──────────┬───┘   │ beam/CEM/Gumbel/MCTS    │
                            │       │ joint opponent model     │
                            │       └──────────┬──────────────┘
                            └──────────────────▼
                                      chosen control

Learning loops:
external PPO/self-play ───────────────→ β=0 competitive policy
strategic auxiliaries ────────────────→ long-horizon representation
ensemble disagreement/archive ────────→ targeted data and scenarios
offline planner reanalysis ───────────→ imitation/value targets
optional episodic novelty/RND ────────→ explorer-conditioned policies
```

The MuZero branch can replace the strategic planner later, but it should be compared against this architecture rather than assumed to be the destination. The default exploitative policy remains defined by external game outcomes; exploration modules are training infrastructure, not game rules.

---

## 18. Bottom line

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
- opponent-population exploration before raw state novelty;
- episodic strategic novelty and hard-state archives when coverage is deficient;
- model disagreement for targeted data and planning triggers;
- RND only as a bounded, measured auxiliary;
- hard profiling and ablation gates.

The most likely current planning mistake is trying to make an inherently sequential, irregular, raw-timestep tree look like a dense GPU workload. First remove true synchronization bugs. Then test whether the remaining sequential cost is fundamental. If it is, change the planner—not merely the kernel.

The most likely exploration mistake would be assuming that sparse terminal reward automatically implies a need for RND. In Curvytron, failure is at least as likely to come from delayed credit, weak strategic abstractions, or self-play overfitting. Add novelty only after diagnosing those alternatives, and keep the final objective anchored to competitive outcomes.

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
- [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
- [Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/abs/2002.06038)
- [Agent57: Outperforming the Atari Human Benchmark](https://arxiv.org/abs/2003.13350)
- [Episodic Curiosity through Reachability](https://arxiv.org/abs/1810.02274)
- [Planning to Explore via Self-Supervised World Models](https://arxiv.org/abs/2005.05960)
- [State Entropy Maximization with Random Encoders for Efficient Exploration](https://arxiv.org/abs/2102.09430)
- [RIDE: Rewarding Impact-Driven Exploration](https://arxiv.org/abs/2002.12292)
- [First Return, Then Explore](https://arxiv.org/abs/2004.12919)
- [Deep Hierarchical Planning from Pixels](https://arxiv.org/abs/2206.04114)
- [The Option-Critic Architecture](https://arxiv.org/abs/1609.05140)
- [Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296)
- [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832)
- [PufferLib 4.0 Torch backend](https://github.com/PufferAI/PufferLib/blob/4.0/pufferlib/torch_pufferl.py)
