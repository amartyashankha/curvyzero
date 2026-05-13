# CurvyTron Stock Long v1c Monitor Critique

Timestamp: 2026-05-12 17:16 EDT

Scope: `stock-long-v1c` rows 01-08 from
`artifacts/local/curvytron_stock_train_manifests/stock-long-v1c.commands.sh`.
No launches or stops were performed.

## Checked

- Modal app list showed 8 active detached
  `curvyzero-lightzero-curvytron-visual-survival-train` apps created at
  17:12 EDT.
- Manifest/commands use stock LightZero `--mode train`, not custom two-seat.
- All rows use `source_state_fixed_opponent`, `gpu-l4-t4-cpu40`,
  `env_manager_type=subprocess`, `max_train_iter=20000`,
  `save_ckpt_after_iter=250`, `lightzero_eval_freq=0`, and background
  CurvyZero eval/GIF polling.
- Main rows 01-06 use C32 plus `body_circles_fast`; sentinel rows 07-08 use
  C16 plus `browser_lines`.
- Sampled row 01 fixed-fast, row 03 frozen-fast, and row 07 frozen-browser
  heartbeats all reported `status=running`, `stage=auto_resume_checked`, and
  `opponent_use_cuda=false`.
- Row 03 poller was running and had scheduled checkpoint eval/GIF for
  iterations `0`, `250`, `500`, `750`, and `1000`.

## Pivot Alignment

- Aligned: stock `train_muzero`, source-state fixed-opponent env, subprocess
  scaling, L4+CPU40, stock LightZero eval off, frozen checkpoint opponent kept
  on CPU, and explicit `body_circles_fast` speed path with `browser_lines`
  sentinels.
- Slightly outside the strict optimizer pivot: rows 01, 02, and 08 are
  fixed-straight controls, while the trusted learning path is frozen checkpoint
  opponent. That is fine as a control/baseline, not as the main learning claim.
- The manifest guard still says `source_state_trail_render_mode=body_circles_fast`
  even though rows 07-08 intentionally use `browser_lines`; treat that guard as
  stale wording, not a runtime mismatch.

## Risks

- Checkpoint every 250 over 20k iterations means about 80 checkpoints per row.
  With background eval and GIF on every checkpoint, the batch can create a lot
  of Modal tasks and volume artifacts. Expect volume/rate-limit friction during
  monitoring.
- `opponent_use_cuda=false` is present in observed heartbeats but not explicit
  in the command text, so the run relies on the current default. That default is
  correct today, but explicit CLI flags would be clearer in future matrices.
- Existing optimizer notes record a long-survival LightZero
  `ValueError: 'a' and 'p' must have same size` in a no-death profile. v1c uses
  normal death, so this is not an immediate launch failure, but it remains a
  possible latent issue if policies start surviving much longer.
- `body_circles_fast` is a deliberate speed/fidelity tradeoff. The browser
  sentinels are important for checking whether any signal depends on the faster
  approximate renderer.
