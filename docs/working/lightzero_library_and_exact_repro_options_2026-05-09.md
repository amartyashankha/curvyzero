# LightZero Library And Exact Repro Options - 2026-05-09

Scope: side-explorer review for the training coach lane. This note audits the
current exact installed-package LightZero Atari Pong wrapper, names every known
mutation from stock, compares installed `LightZero==0.2.0`, current GitHub
upstream, and other MuZero libraries, and sets evidence gates for the next step.

No pytest was run. No training was launched. No large downloads were performed.
Network/web was available, so primary sources were checked in addition to local
docs and source.

## Recommendation

Do not switch libraries yet.

The next exact reproduction step should stay on the installed
`LightZero==0.2.0` lane and finish one of these, in order:

1. wait for the already-launched faithful-short relpath GPU run
   `train-faithful-short-installed-0.2.0-s0-8192-relpath` to produce a summary,
   then audit its patch list, checkpoint location, and progress snapshots;
2. if that summary is clean, run a slightly larger faithful-short installed
   package rehearsal that preserves all stock settings and changes only
   `train_muzero.max_env_step`;
3. only after the artifact/checkpoint accounting is clean, ask for explicit
   approval before the full installed-package exact train:
   `LightZero==0.2.0`, `PongNoFrameskip-v4`, stock `200000` env steps,
   `50` simulations, `8` collectors, `3` evaluators, `batch_size=256`,
   `game_segment_length=400`, `update_per_collect=None`, no episode caps.

Switching to current GitHub `main` is not a library switch; it is a different
source target. It is justified only after we choose to reproduce the current
upstream recipe instead of the installed package recipe. That target has
`max_env_step=int(5e5)`, so it needs a pinned GitHub commit/source identity and
a separate dry-exact gate.

Switching to MiniZero, muzero-general, EfficientZero, or a smaller MuZero repo
is not justified yet. They do not answer the present failure better than
finishing the LightZero exact-source gate, and they would add a fresh
integration/debug surface before we know whether the existing failure is setup,
budget, checkpoint accounting, evaluator parity, or framework behavior.

## Is The Current Wrapper Close To Installed Stock?

Yes, for the `exact` path, with one important wording constraint: it is close to
the installed PyPI package surface in our Modal image, not exact current GitHub
upstream.

The wrapper at
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py` imports:

```text
zoo.atari.config.atari_muzero_config
lzero.entry.train_muzero
```

from installed `LightZero==0.2.0`, deep-copies `main_config` and
`create_config`, validates the stock surface, and in train mode calls:

```python
train_muzero([main_config, create_config], seed=seed, max_env_step=actual_max_env_step)
```

In exact mode, `actual_max_env_step` is the installed package value:
`200000`.

The stock installed surface captured by the dry exact run is:

| Setting | Installed `LightZero==0.2.0` exact value |
| --- | --- |
| env id | `PongNoFrameskip-v4` |
| env type | `atari_lightzero` |
| env manager | `subprocess` |
| trainer | `lzero.entry.train_muzero` |
| policy | `muzero` |
| model | conv |
| observation | `[4, 64, 64]` |
| action space | `6` |
| collectors / collect episodes | `8` / `8` |
| evaluators / eval episodes | `3` / `3` |
| simulations | `50` |
| batch | `256` |
| update | `update_per_collect=None`, `replay_ratio=0.25` |
| segment length | `400` |
| CUDA | `True` |
| eval frequency | `2000` |
| replay buffer | `1000000` |
| exact env steps | `200000` |
| episode caps | unset |
| forced checkpoint cadence | unset |

## Known Mutations From Stock

These are the known mutations in the current wrapper, separated by whether
they change training semantics.

### Exact Mode, Training Semantics

Only one stock config field is changed:

| Mutation | Why it exists | Training semantics? |
| --- | --- | --- |
| `main_config.exp_name` is changed from the stock `data_muzero/...` path to a Modal Volume attempt path | Keep logs/checkpoints inside `curvyzero-runs` and recover artifacts from Modal | Output placement only |

The current code uses a relative Volume ref for `exp_name` and changes the
process working directory to `/runs` before training. This is an artifact-path
guard. It was added after an earlier faithful-short attempt wrote under a
`.//runs/...`-style path while the progress scanner looked under `/runs/...`.

### Exact Mode, Wrapper/Infrastructure

These are real differences from running the upstream script directly, but they
do not intentionally change the LightZero config or learner/evaluator
semantics:

- Modal image installs pinned `LightZero==0.2.0`, `opencv-python-headless`, and
  `AutoROM[accept-rom-license]`, then runs `AutoROM --accept-license`.
- Local `src/` is added to the Modal image and `PYTHONPATH` for the wrapper and
  run-management helpers.
- The wrapper records JSON summaries, package versions, ROM-license notes,
  config surfaces, patches, artifact scans, attempt manifests, and latest-run
  pointers.
- A progress watcher periodically scans the output directory and writes
  sidecar progress JSON files.
- CPU train mode is blocked before `train_muzero`; exact train is expected to
  run on GPU (`gpu-l4-t4`).
- Modal resources, memory, timeout, and GPU choice are declared by the wrapper.
- Dry mode imports/configures/validates and exits without env creation or
  training.

### Faithful-Short Mode

Faithful-short is deliberately not exact reproduction. It preserves the stock
installed config and the same output-path patch, but also changes the trainer
call:

| Mutation | Example | Claim allowed |
| --- | --- | --- |
| `train_muzero.max_env_step` is reduced below installed stock | `200000 -> 8192` | Installed-package path rehearsal only |

This is useful for proving artifact paths, checkpoint accounting, and
longer-than-smoke mechanics. It is not a quality claim and not an exact
installed-package reproduction.

### Older Tiny/Control Wrappers, Not The Exact Wrapper

The earlier `lightzero_pong_tiny_train_smoke.py` path made many additional
mutations: reduced env steps, reduced simulations, reduced collector/evaluator
counts, small batches, episode caps, shorter segments, changed
`update_per_collect`, forced checkpoint cadence, and manual eval harnesses.
Those are valid smokes/control runs, but they are not the current exact wrapper
and should not be counted as exact reproduction evidence.

## Source Target Comparison

| Target | What it means | Pros | Risks / blockers | Current verdict |
| --- | --- | --- | --- | --- |
| Installed `LightZero==0.2.0` | Reproduce the PyPI package already used in our Modal image. Exact env budget is `200000`. | Pinned, already installed in the image, exact dry passed, ROM path works, Modal wrappers and artifact layout are ready. | Full train not completed; faithful-short path still needs a clean completed summary; stock `update_per_collect=None` can create a large checkpoint/update burst; `ckpt_best` behavior from near-upstream runs is suspicious. | Best next target. |
| Current GitHub `main` | Reproduce the latest upstream source from GitHub. Exact env budget is `500000`. | Matches current upstream docs/source, highest "today's upstream" fidelity. | Requires pinned GitHub commit/source identity, possibly a new Modal install/mount path, a separate dry-exact gate, and more budget. It is not the same target as PyPI 0.2.0. | Keep as a later exact-upstream target, not the immediate run. |
| MiniZero | Separate AlphaZero/MuZero/Gumbel framework with Atari support. | Serious system and supports Atari 57 games. | New C++/Python build/runtime stack, no local Modal proof, custom env bridge still needed, checkpoint/eval UX would need new tooling. | Do not switch now; possible later second external benchmark. |
| muzero-general | Educational PyTorch MuZero implementation with Ray, checkpoints, and examples. | Readable reference for MuZero structure. | Atari example is Breakout, not exact Pong; batch MCTS and >2-player support are listed as future ideas; older educational posture. | Reference only. |
| EfficientZero repo | Atari sample-efficiency research implementation. | Relevant ideas for reanalyze/sample efficiency. | Not plain MuZero, GPL-3.0, C++/Cython build, Ray/distributed assumptions, quick start is Breakout. | Do not switch. |
| Mctx-owned MuZero | Use DeepMind Mctx for search, own replay/learner/eval/checkpoints. | Best long-term owned-search substrate; exposes MuZero and Gumbel MuZero policies and action-weight targets. | Mctx is search only. We would need to write trainer, replay, target builder, model, Modal orchestration, and eval. | Later, after repo-native PPO proves environment learnability. |

## Criteria-Based Read

| Criterion | Installed 0.2.0 | GitHub `main` | Other MuZero repo/library |
| --- | --- | --- | --- |
| Maintained enough | Yes: PyPI 0.2.0 exists and GitHub still has active docs/source signals. | Yes enough for a source target, but must pin commit. | Mixed: MiniZero serious but no release package; muzero-general/EfficientZero are less attractive as active backbones. |
| Exact Pong/Atari recipe | Yes: installed `atari_muzero_config` with Pong and 200k steps. | Yes: same recipe family with 500k steps. | Weak: MiniZero has Atari, but not our exact LightZero Pong recipe; muzero-general/EfficientZero emphasize other Atari examples. |
| Ease of Modal | Best: already proven with ROM image, dry, env, train smokes. | Medium: needs source install/mount and pinning, but image pattern should transfer. | Worse: new build/dependency/runtime shape. |
| Checkpoint/eval clarity | Best current tooling, but `ckpt_best` must be audited. | Unknown until wrapper exists; source identity clearer if pinned. | Unknown/new; would require rebuilding strict eval and artifact conventions. |
| Custom env bridge | Medium: LightZero custom env path works for dummy Pong but is ego-wrapper-shaped and needs sidecars. | Same as installed, plus source drift risk. | No clear win; Mctx would give control but requires owning everything. |
| Risk | Known risks, visible artifacts. | Higher cost and target drift. | Highest integration risk before the current blocker is diagnosed. |

## Evidence Gates

### Gate 1: Installed-Package Faithful-Short Clean Summary

Pass only if:

- summary exists for the relpath faithful-short attempt;
- `run_kind=faithful-short`;
- `is_exact_reproduction=false`;
- the only semantic extra patch is `train_muzero.max_env_step`;
- stock config values remain `50` sims, `8` collectors, `3` evaluators,
  `batch_size=256`, `game_segment_length=400`, `update_per_collect=None`;
- checkpoints and progress scans agree on the same Volume-backed output root;
- no fallback/non-strict eval is used for any promoted checkpoint result.

### Gate 2: Installed-Package Exact Dry Before Full Train

Pass only if:

- installed package reports `LightZero==0.2.0`;
- imported `zoo.atari.config.atari_muzero_config` reports
  `max_env_step=200000`;
- the patched config differs only in `exp_name`;
- dry mode does not instantiate envs and does not call `train_muzero`;
- summary records source/package versions and artifact refs.

### Gate 3: Full Installed Exact Train Approval

Before launching, require explicit human approval because the run may be long
and stock `update_per_collect=None` can produce many checkpoints/updates. The
train command must leave `--max-env-step-override` unset.

Pass after completion only if:

- summary says `run_kind=exact` and `is_exact_reproduction=true`;
- `actual_max_env_step=200000`;
- only `exp_name` changed;
- periodic checkpoints and `ckpt_best` strict-load under the same config;
- stock evaluator and independent/manual eval agree on action prefixes and
  directional returns;
- `ckpt_best` is not reset-looking when compared to a late periodic checkpoint.

### Gate 4: Current GitHub Upstream

Open this only if the installed-package lane is clean or if the user explicitly
wants "latest upstream today." Pass dry only if a pinned GitHub source reports
`max_env_step=500000` for `atari_muzero_config.py` and all other stock Pong
settings match the current source.

### Gate 5: Library Switch

Switch away from LightZero only if at least one of these is true:

- exact or faithful installed-source runs pass artifact/eval gates but still
  expose a LightZero-specific trainer/checkpoint/evaluator blocker;
- custom env bridging requires invasive LightZero/DI-engine forks;
- independent scorecards remain contradictory while source-faithful stock Pong
  reproduction succeeds elsewhere;
- repo-native PPO proves the environment and a small owned Mctx MuZero is
  cheaper than continuing to bend LightZero.

Do not switch just because short capped Pong runs did not learn.

## Sources

Local sources:

- `src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py`
- `src/curvyzero/infra/modal/lightzero_atari_rom_image.py`
- `docs/working/training_coach_handoff_2026-05-09.md`
- `docs/working/training_state_index_2026-05-09.md`
- `docs/working/rl_framework_reliability_deep_dive_2026-05-09.md`
- `docs/working/lightzero_exact_reproduction_decision_2026-05-09.md`
- `docs/working/lightzero_official_visual_pong_pattern_2026-05-09.md`
- `docs/working/lightzero_exact_upstream_atari_command_2026-05-09.md`
- `docs/working/muzero_library_alternatives_2026-05-09.md`
- `docs/working/lightzero_setup_fidelity_audit_2026-05-09.md`

Primary sources checked:

- LightZero quick start:
  https://opendilab.github.io/LightZero/tutorials/installation/installation_and_quickstart.html
- Current LightZero GitHub Atari MuZero config:
  https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_config.py
- LightZero config docs:
  https://opendilab.github.io/LightZero/tutorials/config/config.html
- PyPI LightZero package:
  https://pypi.org/project/LightZero/
- LightZero GitHub README:
  https://github.com/opendilab/LightZero
- MiniZero GitHub README:
  https://github.com/rlglab/minizero
- muzero-general GitHub README:
  https://github.com/werner-duvaud/muzero-general
- EfficientZero GitHub README:
  https://github.com/YeWR/EfficientZero
- Mctx GitHub README:
  https://github.com/google-deepmind/mctx

