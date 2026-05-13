# Eval/GIF Status Note - 2026-05-13

Scope: `curvy-mix2-clean-20260513a` status artifacts, especially
`/private/tmp/curvy_mix2clean_status_3.json` and
`/private/tmp/curvy_mix2_clean_cadence_joined.json`.

Short answer: eval and GIF artifacts are being created. This looks like a
status-reader/summary semantics issue, not a real artifact outage.

Evidence from the stripped JSON status at 2026-05-13T12:25:32Z:

- rows scanned: 156
- rows with eval manifests: 38
- eval manifests total: 39
- rows with GIF artifacts: 95
- GIF artifacts total: 105
- poller seen/scheduled totals: 55 / 55
- poller completed/eval-completed/GIF-completed totals: 0 / 0 / 0
- `latest_eval_checkpoint` key present in raw JSON rows: false

Why this is confusing:

- Raw JSON status stores eval detail under `eval_checkpoints`. For rows with
  evals, the latest checkpoint is available as `eval_checkpoints[-1].checkpoint`.
  In this snapshot those latest eval checkpoints were `iteration_0`.
- The TSV/table renderers derive a convenience field named
  `latest_eval_checkpoint`, but `_eval_manifest_rollup()` does not add that key
  to raw JSON rows.
- Poller `completed_count`, `eval_completed_count`, and `gif_completed_count`
  count jobs only after `_run_checkpoint_eval_poller()` joins outstanding Modal
  function calls near poller exit. While the poller is still `running`, artifacts
  can already exist on the Volume while these completed counters remain zero.
- `gif_scheduled_count` is nonzero in poller status, and `gif_artifact_count`
  independently proves summaries exist under
  `attempts/*/eval/*/selfplay/summary.json`.

Website visibility:

- The GIF browser scans
  `training/lightzero-curvytron-visual-survival/<run>/attempts/*/eval/*/selfplay/summary.json`.
- It validates and serves `raw.gif` and `collect_t1.gif` under the same
  selfplay directory.
- The status refs match that shape, for example
  `.../eval/live_checkpoint_iteration_10000/selfplay/raw.gif`.
- Default browser listing only includes runs with `show_in_gif_browser.flag`.
  These runs were launched with background GIF enabled, which writes that marker
  for the run picker. For an exact run page/filter, the current artifact refs are
  in the browser's accepted path format.

Practical read:

- `eval_manifest_count > 0` plus nonempty `eval_checkpoints` means numeric evals
  exist.
- `gif_artifact_count > 0` plus `latest_gif_ref` means GIF summaries exist and
  should be browser-readable.
- Do not treat zero `background_poller_*_completed_count` during `running` as
  "no eval/GIF artifacts"; it means "the poller has not joined/recorded completed
  spawned calls yet."
- For JSON consumers, use `eval_checkpoints[-1].checkpoint` or add a raw JSON
  alias later if needed.
