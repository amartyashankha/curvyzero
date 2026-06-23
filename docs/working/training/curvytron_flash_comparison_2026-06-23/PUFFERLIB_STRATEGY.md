# PufferLib Strategy Note - 2026-06-23

Status: critical integration read, not a speed claim.

Inspected sources:

- CurvyZero workspace at `/Users/shankha/curvy`.
- User-supplied synthesis in `external/chatgpt/context.md`.
- Flash comparison packet in this directory.
- PufferAI/PufferLib default branch `4.0` at commit
  `9a4eb87e6b58c0aa5f22affefb65c7006d384972`.
- PufferAI/PufferLib branch `5.0` at commit
  `c3e28db186bed5a341fda705faa2a33fd5dc97ab`.

`git ls-remote --symref` showed upstream `HEAD` still points at `refs/heads/4.0`.
The `5.0` branch exists and adds/changes some environment and curriculum surface,
but the main strategic reading does not change: PufferLib is a fast PPO/native
runtime stack, not a MuZero or MCTS stack.

## Short Verdict

The current best systems strategy is a multi-lane program:

1. Keep the stock-ish LightZero reward/RND campaign as the canonical near-term
   learning-quality lane until a replacement baseline has earned parity.
2. Keep CurvyZero's compact-owner MuZero/search speed lane, with its same-work
   H100 proof discipline.
3. Add a bounded policy-first baseline lane using PufferLib or a Puffer-shaped
   runtime: fixed buffers, compiled/vectorized env API, recurrent PPO, action
   masks, frozen-bank self-play, and explicit policy-quality gates.
4. Treat Puffer as a feasibility spike and baseline pressure test, not as the
   center of the current campaign.

Pure raw-tick MCTS should not be the center of gravity. Planning may still be
valuable, but it should be evaluated as macro-action search, dense GPU trajectory
planning, reanalysis/distillation, or a gated Gumbel MuZero branch. Treating
every 60 Hz tick as a tree level is probably the wrong shape for both CurvyTron
and GPUs.

## What PufferLib Already Has

PufferLib 4.0/5.0 includes these reusable pieces:

- A native static vector environment ABI (`src/vecenv.h`).
- Flat fixed buffers for observations, actions, rewards, terminals, and optional
  action masks.
- Pinned CPU host buffers plus GPU buffers for C/Ocean environments.
- Per-buffer CUDA streams and OpenMP CPU stepping for environment chunks.
- A native CUDA PPO learner (`src/pufferlib.cu`) with fixed rollout buffers and
  transposed train buffers.
- CUDA Graph capture/replay paths for rollout and train kernels.
- MinGRU as the default recurrent policy backbone.
- Masked discrete action support.
- GAE/V-trace-like "puff advantage" kernels.
- PPO loss forward/backward fused into CUDA-side machinery.
- Muon optimizer, including NCCL all-reduce support for multi-GPU.
- Frozen policy banks and a self-play pool intended for two equal teams.
- A sweep/tuning layer and many Ocean C environments.
- A Torch fallback path, but it is not the serious max-throughput path.

What it does not have:

- MuZero dynamics/reward/policy/value model training.
- MCTS/PUCT/Gumbel MuZero search.
- Replay/reanalysis semantics for search-generated targets.
- A CurvyTron-specific env.
- General arbitrary free-for-all self-play semantics. The stock self-play helper
  assumes even agents and two equal teams.

Critical note: parts of the repo are active/mixed. Some examples/templates still
look stale relative to the 4.0 native ABI. Any integration must start with a
build/train smoke in the target Modal/Docker image, not trust docs alone.

## What CurvyZero Has Instead

CurvyZero has stronger proof discipline and CurvyTron-specific semantics, but it
does not currently have a production PPO runtime comparable to PufferLib.

The repo-native PPO files are deliberately small scaffolds:

- `scripts/repo_native_ppo_actor_loop_dry_run.py` is a contract/measurement
  dry run. It uses Python scalar `CurvyTronEnv` instances, loops per row, samples
  a masked uniform policy, and writes NPZ rollout artifacts.
- `scripts/repo_native_ppo_learner_smoke.py` is a tiny optional-Torch smoke. It
  uses CPU Torch, a small MLP, loads NPZ rollout data, computes GAE, and performs
  one clipped PPO update.

That is useful plumbing evidence, not a fast PPO implementation. Compared with
PufferLib, local PPO is missing:

- compiled/vectorized environment stepping;
- GPU-resident rollout buffers;
- native action sampling / policy inference in the rollout loop;
- recurrent policy state at scale;
- CUDA Graph capture;
- fused PPO update kernels;
- native optimizer/multi-GPU support;
- frozen-bank self-play;
- production train/eval/export loop.

So the gap is not a handful of optimization patches. It is an ownership-model
gap.

## Flash Versus Puffer Versus CurvyZero

Flash is the CurvyTron-specific GPU-resident control. It has:

- PyTorch/Triton accelerated CurvyTron env;
- raycast/compact observations;
- PPO training/export surfaces;
- playable bot loop;
- very high raw H100 environment controls;
- one diagnostic PPO profile row around `438k agent_steps/s`.

But the Flash packet is right to keep denominators separate. Flash raw mechanics,
Flash PPO, and CurvyZero MuZero/search do different work.

PufferLib is the reusable RL systems pattern:

- static env ABI;
- fixed rollout/train buffers;
- recurrent PPO;
- frozen-bank self-play;
- CUDA graph/native learner machinery.

CurvyZero is the stricter research ledger and existing MuZero/search workspace:

- same-work H100 speed gate;
- compact-owner/search/replay experiments;
- LightZero/MuZero integration;
- careful artifact/proof culture.

The right move is not "port Flash wholesale" or "replace CurvyZero with
PufferLib." The right move is to borrow the ownership pattern and create a clean
PPO/self-play baseline lane whose results are kept out of the MuZero speed
ledger unless the denominator is truly equivalent.

The planning/RND integration doctrine is tracked in
`../reward_axis_h100_plan_2026-06-23/LONG_TERM_PLANNING_RND_STRATEGY.md`.
That note is the bridge between this PPO/Puffer baseline idea, the current
stock-ish LightZero RND lane, and any future macro-action planner work.

## Could PufferLib Be Faster?

Against CurvyZero's repo-native PPO smoke: yes, it should be much faster if the
native build path works. The local PPO smoke is CPU/Python/NPZ scaffolding.
PufferLib is a compiled fixed-buffer PPO runtime. This statement is scoped only
to the smoke PPO scripts, not to Flash PPO or CurvyZero MuZero/search.

Against Flash PPO: uncertain. PufferLib's native PPO learner and recurrent
policy machinery may be faster or more scalable than Flash's PyTorch PPO update.
But if CurvyTron is implemented as a CPU Ocean C environment, PufferLib still
copies actions down and observations/rewards/dones back each tick. Flash's
Triton env is already GPU-resident and may dominate on raw CurvyTron mechanics
and raycast generation.

Against CurvyZero MuZero/search: not apples-to-apples. PPO can produce more
agent steps per second while doing less algorithmic work. Compare by policy
quality, eval behavior, and wall-clock learning milestones, not by raw speed row.

Working speed hypothesis:

- PufferLib is a credible route to a serious recurrent model-free baseline if we
  can express CurvyTron cheaply through its ABI.
- Flash remains the env-mechanics ceiling/control and the best source of
  CurvyTron-specific GPU kernels.
- CurvyZero's MuZero speed problem is still about whole-loop owner boundaries,
  not merely env stepping.

## Recommended Combined Strategy

1. Preserve the existing CurvyZero speed ledger.
   Same-work H100 full-loop rows remain the only promotion-grade speed proof.
   Flash/Puffer PPO rows get their own denominators.

2. Run a Puffer feasibility spike before any deep port.
   Implement a minimal 2-agent CurvyTron Ocean env: action size 3, fixed compact
   or ray observation, terminal sparse reward, optional action mask, no bonuses
   at first, deterministic seeds, parity tests against a trusted reference.

3. Benchmark three layers separately.
   Measure raw env throughput, PPO rollout/update throughput, and policy-quality
   learning/eval. Do not merge them into one "faster" claim.

4. Use recurrence and self-play early.
   CurvyTron is partially observed and strategic over time. A MinGRU PPO baseline
   with historical opponents is more relevant than a feed-forward one-update
   PPO profile.

5. Keep planning off the PPO contract.
   PPO rollouts must store actions sampled from the old policy with valid old
   logprobs. Search-generated actions belong to imitation/search-target training,
   not vanilla PPO.

6. Prefer macro/dense planning before pure raw-tick MCTS.
   Test action chunks, beam/CEM/MPC, and policy-guided rollout evaluation before
   spending more effort on fully general per-tick tree search.

7. Reuse Flash selectively.
   Use Flash as a parity/control source, a playable export-loop reference, and a
   GPU kernel/ABI study target. Do not transplant its whole Modal app or interpret
   its raw env rows as CurvyZero MuZero speed proof.

## First Concrete Experiment

Create `puffer_curvytron_minimal` as an external spike or isolated workspace,
not a repo rewrite:

- 2 players, 3 actions each.
- Fixed world size and tick cap.
- Fixed compact/raycast observation shape.
- Sparse terminal reward first; shaped rewards only after baseline parity.
- Deterministic seed fixture and parity traces.
- Puffer native build smoke on GPU image.
- Tiny train run with MinGRU PPO.
- Evaluation against random/scripted/Flash-exported reference opponents.

Promotion criteria:

- Native build works in the same intended Modal/Docker environment.
- Environment parity passes representative collision/trail/wall/head-on cases.
- PPO profile reports meaningful rollout/update split.
- Learning curve beats random/scripted baselines under a fixed wall-clock budget.
- All artifacts are labeled as `puffer_ppo_baseline`, not CurvyZero MuZero speed.

Kill criteria:

- CurvyTron observation/raycast copies dominate enough that Flash GPU env is
  clearly a better substrate.
- Puffer native ABI fights dynamic CurvyTron features we actually need.
- Self-play helper assumptions break the required game mode.
- Build/dependency surface becomes larger than a lean local PPO/Triton lane.

## Bottom Line

PufferLib is worth a serious feasibility spike because it already implements
many of the systems ideas CurvyZero is trying to grow toward: static ownership,
fixed buffers, recurrent PPO, native learner kernels, action masks, and frozen
self-play banks.

But PufferLib is not a search solution. It should be used to establish a fast,
credible model-free baseline and to pressure-test the whole "maybe we do not
need raw-tick MCTS" hypothesis. If that baseline learns strong CurvyTron play
quickly, then planning should be added only where it earns its complexity:
macro-actions, dense GPU planners, reanalysis, or a carefully gated Gumbel
MuZero branch.
