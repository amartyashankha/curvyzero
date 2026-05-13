# Coach Speed Recommendations

Date: 2026-05-12

Scope: speed recommendations from Optimizer profile runs. These are not
learning claims.

## Trusted Path

Use the stock LightZero path for the current trusted lane:

```text
mode=train or profile through stock train_muzero
env_variant=source_state_fixed_opponent
opponent_policy_kind=frozen_lightzero_checkpoint
opponent_use_cuda=false
```

Do not use the old custom `two-seat-selfplay` path as speed or learning
evidence.

## Default Speed Shape

Recommended starting point:

```text
compute=gpu-l4-t4-cpu40
env_manager_type=subprocess
collector_env_num=96
n_episode=96
num_simulations=8 or 16
batch_size=16
source_state_trail_render_mode=browser_lines
```

Reason:

- C96 reached about 41 steps/s in normal browser/sim8 profiles.
- C128 and C160 did not improve enough to be the default.
- Sim16 at C96 was about as fast as sim8 at C96, so sim16 is reasonable if
  Coach wants a stronger search setting.
- L4+CPU40 beat CPU64 at the same C32/sim8 shape.
- H100 and multi-GPU are not justified yet because search/GPU work is not the
  main bottleneck.

## Render Choice

Use `browser_lines` for trusted fidelity runs.

Use `body_circles_fast` only as a deliberate speed/fidelity tradeoff:

- At C32 no-death, fast render reached about 160 steps/s versus 91 steps/s for
  browser render.
- At short normal C32, fast render only moved about 21 steps/s to 24 steps/s.
- This means fast render matters most once policies survive longer.

Plain recommendation: keep browser render for proof runs; consider fast render
for exploration runs only if Coach and Environment accept the visual
approximation.

## Frozen Opponent Cost

Frozen checkpoint opponent inference is a real cost in this fixed-opponent
lane.

Timing lens:

- C32 frozen checkpoint: about 21 steps/s.
- C32 fixed-straight: about 29 steps/s.
- C96 frozen checkpoint: about 41 steps/s.
- C96 fixed-straight: about 59 steps/s.

Do not switch learning runs to fixed-straight because of this. It is only a
profiling lens. For real fixed-opponent control runs, keep the frozen checkpoint
opponent on CPU unless the subprocess CUDA issue is fixed.

## Reward And Artifacts

Dense survival reward bookkeeping was not a major bottleneck in profiles.

For profile runs:

```text
lightzero_eval_freq=0
skip_lightzero_eval_in_profile=true
save_ckpt_after_iter=9999
background_eval_enabled=false
background_gif_enabled=false
profile_volume_commit=false
```

For real training runs, checkpoint/eval/GIF cadence is a Coach decision. Keep it
sparse enough that artifact work is amortized, but do not use Optimizer profile
settings as learning defaults.

## Current Bottleneck Read

The main cost is still collection-side work: subprocess env workers, visual
observation/render, and frozen-opponent inference. MCTS/search is visible but
not dominant at sim8-sim16 in these profiles. Learner time is small.

This points to two next optimizer lanes:

1. Better actor/fanout architecture so searched collection can scale beyond one
   stock trainer container.
2. Render/frozen-opponent optimization if Coach stays on the fixed-opponent
   stock lane long enough for those costs to matter.
