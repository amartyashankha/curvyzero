# LightZero Repro Library Audit - 2026-05-09

Scope: high-level audit of the LightZero reproduction/library lane. Read inputs
were the current coach handoff, state index, exact reproduction decision,
framework reliability deep dive, coach optimizer reorientation, and local
`src/curvyzero/infra/modal/lightzero*.py` wrappers.

No code was edited. No pytest was run. No Modal job was launched. No web lookup
was needed for this pass.

## Short Answer

We are close enough to upstream for one thing: an installed-package
`LightZero==0.2.0` Atari Pong reproduction wrapper now exists and is narrow
enough to call a real upstream-style control. In exact mode it imports
`zoo.atari.config.atari_muzero_config`, patches only `exp_name` for Modal
Volume output, runs from `/runs`, and calls stock `lzero.entry.train_muzero`
with the installed package's `max_env_step=200000`.

That is not the same as having reproduced LightZero Pong. The full exact train
has not run. The faithful-short run is useful plumbing evidence, but it is not
exact reproduction because it changes `train_muzero.max_env_step` from
`200000` to `8192`.

Do not switch libraries now. LightZero remains the MuZero replication/control
lane until either a credible stock reproduction appears or a clear blocker is
documented. The repo-native `[B,P]` runner remains a parallel CurvyTron
architecture probe, not a replacement decision.

## Wrapper Read

`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py` is the only
wrapper that should carry the "close to upstream" label.

What it preserves:

- installed `LightZero==0.2.0`;
- stock module `zoo.atari.config.atari_muzero_config`;
- stock env id `PongNoFrameskip-v4`;
- stock trainer `lzero.entry.train_muzero`;
- stock surface: 50 MCTS sims, 8 collectors, 3 evaluators, batch 256,
  `update_per_collect=None`, replay ratio 0.25, segment length 400, CUDA true,
  no episode caps;
- exact mode `max_env_step=200000`;
- no `max_train_iter` argument.

What it changes in exact mode:

- `main_config.exp_name`, so artifacts land under the Modal Volume;
- process working directory in train mode, so DI-engine relative paths stay
  inside `/runs`.

That is an artifact patch, not a learning-semantics patch.

What faithful-short changes:

- all of the above, plus `train_muzero.max_env_step=8192`.

That is a rehearsal. It should never be summarized as exact reproduction.

The older official visual Pong smoke wrapper,
`lightzero_pong_tiny_train_smoke.py`, is intentionally not upstream-close. It
caps training, caps episodes, reduces env counts, reduces simulations, changes
batch size, can force `update_per_collect`, sets frequent checkpointing, and
can change segment length. It is a mechanical train/checkpoint/eval smoke.

The eval wrapper, `lightzero_pong_eval_smoke.py`, is useful and stricter than
earlier eval attempts because it can do strict checkpoint load, no model
fallback, manual eval telemetry, and stock `MuZeroEvaluator` comparison. It is
an evaluation/control tool, not a reproduction train wrapper.

The custom dummy Pong wrappers are a separate bridge lane. They replace the env
with `DummyPongLightZero-v0`, use an MLP model and 3 actions, add target replay
telemetry, expose scorecards, support frozen-checkpoint opponents, and preserve
episode/survival telemetry. They are valuable for CurvyTron-like debugging, but
they are not evidence that stock visual Atari Pong was reproduced.

## What Has Not Been Reproduced

Still missing:

- no full exact installed-package LightZero Atari Pong train at 200k env steps;
- no current-GitHub LightZero Pong recipe reproduction, where local docs say
  the current surface is about 500k env steps rather than installed 0.2.0's
  200k;
- no credible learned Atari Pong policy from our runs;
- no clean improvement from faithful-short 8192-step runs;
- no reliable custom dummy Pong held-out improvement;
- no successful strict eval of the public OpenDILab pretrained Pong checkpoint
  with a matching 96x96/downsample config/checkpoint pair;
- no resolved explanation for the reset-looking `ckpt_best` artifact observed
  in the 8192/sim25 run;
- no CurvyTron MuZero training run.

The latest near-upstream and faithful-short evidence is therefore:

- setup fidelity improved;
- checkpoint/load/eval plumbing improved;
- strict no-fallback eval exists;
- learning signal is still absent.

That is not an indictment of LightZero yet. It is a sign that previous runs
were too short, partly off-recipe, or still carrying setup/artifact questions.

## Library Decision

Do not switch libraries now.

The framework deep dive still points to a layered setup:

- keep simulator state, reset/final-observation semantics, scorecards, and
  Modal artifacts repo-owned;
- use PettingZoo-style `ParallelEnv` as a public simultaneous-action adapter;
- keep repo-native PPO/CleanRL-style `[T,B,P]` as the first CurvyTron
  learnability baseline;
- keep LightZero as the contained MuZero control lane;
- keep Mctx as a later owned-search substrate only if we decide to own replay,
  targets, learner, checkpointing, and eval ourselves.

Switching away from LightZero now would mostly avoid the open question instead
of answering it. We need either a clean stock reproduction or a precise blocker.

## Next Reproduction/Control

Next LightZero action should be a managed exact installed-package reproduction,
not another tiny same-shape smoke.

Recommended sequence:

1. Repeat exact dry validation if the environment or wrapper changed.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode dry \
  --compute cpu \
  --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id dry-exact-installed-0.2.0-stock-surface
```

2. With explicit human approval, run the exact installed-package GPU train:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-exact-installed-0.2.0-stock-surface \
  --progress-interval-sec 300
```

3. Evaluate periodic checkpoints with `lightzero_pong_eval_smoke.py` using:

- strict policy model load required;
- no model fallback;
- manual 512-step telemetry at minimum;
- stock `MuZeroEvaluator` comparison when the cap/path matches;
- return, positive rewards, nonzero rewards, survival steps, action histogram,
  dominant-action share, entropy, and manual-vs-stock match.

4. Record checkpoint accounting:

- checkpoint count;
- checkpoint bytes;
- whether files stayed under the intended Volume root;
- whether `ckpt_best` is a real trained checkpoint or reset-looking;
- train elapsed time and env-step overshoot.

The exact run can be expensive and can write many checkpoints. It should be
watched, not fire-and-forget.

## Clear Blocker Criteria

A clear LightZero blocker would be one of these:

- exact installed-package 200k train cannot complete on the managed Modal GPU
  path after artifact-root fixes, with a concrete reproducible error;
- exact train completes but produces unusable checkpoints under stock config:
  strict load fails, stock evaluator cannot run, or checkpoint metadata is
  corrupt/reset-looking across periodic checkpoints;
- exact train completes and strict eval works, but periodic/final checkpoints
  show no improvement over initialization by the same stock evaluator and
  manual telemetry, and the setup matches upstream closely enough that budget
  and wrapper differences are no longer plausible explanations;
- a matching upstream/pretrained Pong checkpoint/config pair still cannot be
  loaded strictly after the 96x96/downsample surface is correctly reproduced;
- custom-env LightZero cannot store or expose correct root visit targets,
  support-scale targets, checkpoint lineage, or eval metadata needed for
  CurvyTron-style debugging.

Anything weaker is not a blocker. Tiny capped runs that fail to learn are
mostly budget/setup evidence. Custom dummy Pong collapse is a bridge-lane bug
or target-quality problem, not a reason by itself to abandon LightZero.

## Bottom Line

LightZero is now set up well enough to run the real installed-package control.
It has not yet earned a learning claim. The next useful move is the exact 200k
installed-package run plus strict periodic eval and checkpoint accounting.
Keep the repo-native `[B,P]` lane moving in parallel, but do not call it a
library switch.
