# LightZero Live GPU Eval Loop - 2026-05-09

Scope: live eval of official LightZero Atari Pong checkpoints while the
`32768` faithful-short train is still running. No pytest was run for this note.

## Answer

Yes, `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py` can now run on a
cheap GPU for stock-ish eval. The code change is intentionally small:

- Add `--compute gpu-l4-t4`.
- Route eval workers to a separate Modal function with `gpu=["L4", "T4"]`.
- Patch the eval LightZero policy config with `policy.cuda=true`.
- Keep strict checkpoint load and no model fallback visible in the result.
- Record `config.runtime_compute` and compiled policy `cuda/device`.

CPU remains the default with `--compute cpu`.

## Active Run

```text
train_app: ap-xiGLACKHPZLvL1eYgygqvm
gpu_eval_proof_app: ap-3icJTrptdJEw38GZAoK5wx
run_id: lz-visual-pong-exact-installed-0.2.0-s0
attempt_id: train-faithful-short-installed-0.2.0-s0-32768-relpath
checkpoint_dir: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/train/lightzero_exp/ckpt
eval_id: faithful-short-32768-live-gpu-stockeval-s0
```

GPU proof result for `iteration_0`: compute `gpu-l4-t4`, strict load true,
fallback false, CUDA true on NVIDIA L4, manual return `-13`, stock return
`-13`, `512` steps, action `2` for all steps, verdict `collapsed_action`.
This is only the starting baseline for the active `32768` run. It does not
contradict the earlier `8192` final-vs-initial weak improvement, because that
comparison used a later final checkpoint against its own initial checkpoint.

## Poll Checkpoints

List checkpoint files in the Modal Volume:

```sh
uv run --extra modal modal volume ls curvyzero-runs \
  training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/train/lightzero_exp/ckpt
```

Only consider complete periodic files named `iteration_*.pth.tar`. Do not use
`ckpt_best.pth.tar` for the live quality loop.

## Avoid Duplicate Eval

Use the eval artifact tree as the durable ledger. Before launching another live
eval, list the eval root:

```sh
uv run --extra modal modal volume ls curvyzero-runs \
  training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/eval/faithful-short-32768-live-gpu-stockeval-s0
```

Skip any checkpoint whose directory already exists with this shape:

```text
iteration_<N>_low_steps512_seed0
```

If the eval root does not exist yet, that is fine. Start with `iteration_0` or
the newest visible periodic checkpoint.

## Eval One New Checkpoint

Use `--parallel` even for one checkpoint so the wrapper writes a manifest table.
Replace only `<N>` with the iteration visible in the checkpoint listing.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --compute gpu-l4-t4 \
  --parallel \
  --eval-pass low \
  --eval-id faithful-short-32768-live-gpu-stockeval-s0 \
  --checkpoint-refs 'training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/train/lightzero_exp/ckpt/iteration_<N>.pth.tar' \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-relpath \
  --low-detail-max-eval-steps 512 \
  --max-episode-steps 512 \
  --low-detail-step-detail-limit 8 \
  --max-env-step 200000 \
  --max-train-iter 1 \
  --collector-env-num 8 \
  --evaluator-env-num 3 \
  --num-simulations 50 \
  --batch-size 256 \
  --update-per-collect 1 \
  --game-segment-length 400 \
  --no-allow-model-fallback \
  --run-stock-evaluator
```

## Eval Several New Checkpoints

If multiple new periodic checkpoints appear, pass only the not-yet-evaluated
refs in one comma-separated `--checkpoint-refs` value:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --compute gpu-l4-t4 \
  --parallel \
  --eval-pass low \
  --eval-id faithful-short-32768-live-gpu-stockeval-s0 \
  --checkpoint-refs 'training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/train/lightzero_exp/ckpt/iteration_<A>.pth.tar,training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/train/lightzero_exp/ckpt/iteration_<B>.pth.tar' \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-relpath \
  --low-detail-max-eval-steps 512 \
  --max-episode-steps 512 \
  --low-detail-step-detail-limit 8 \
  --max-env-step 200000 \
  --max-train-iter 1 \
  --collector-env-num 8 \
  --evaluator-env-num 3 \
  --num-simulations 50 \
  --batch-size 256 \
  --update-per-collect 1 \
  --game-segment-length 400 \
  --no-allow-model-fallback \
  --run-stock-evaluator
```

## Readout

Use the printed manifest first. The key fields are `checkpoint_ref`,
`strict_load`, `fallback_used`, `eval_cap_steps`, `return`, `stock_return`,
`positive_reward_count`, `dominant_action_share`, `stock_manual_match`,
`elapsed_sec`, and `verdict`.

Treat GPU as a placement choice, not a semantic change. The live loop must keep
strict load, no fallback, `max_episode_steps=512`, `low_detail_max_eval_steps=512`,
`num_simulations=50`, `collector_env_num=8`, `evaluator_env_num=3`,
`batch_size=256`, and `game_segment_length=400`.

## Caveats

- A checkpoint can appear while training is still writing. If a live eval fails
  with a load error, wait for the next poll and rerun that same checkpoint once.
- Modal Volume listing is eventually consistent enough for this workflow, but
  the eval artifact ledger is the source of truth for duplicates.
- Stock/manual action mismatch is an eval-harness diagnostic, not a load
  failure by itself.
- Do not evaluate every tiny progress artifact. Keep the live set to
  `iteration_0`, any meaningful new periodic checkpoint, and the final
  `iteration_<N>`.
