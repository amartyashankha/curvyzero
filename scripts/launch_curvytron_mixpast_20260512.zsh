#!/usr/bin/env zsh
set -euo pipefail

MODULE="curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
if [[ "${ALLOW_HISTORICAL_CUSTOM_TWO_SEAT_RERUN:-0}" != "1" ]]; then
  print -r -- "This historical launcher reruns the custom two-seat adapter that failed the May 12 learning audit." >&2
  print -r -- "Set ALLOW_HISTORICAL_CUSTOM_TWO_SEAT_RERUN=1 only for postmortem reproduction." >&2
  exit 2
fi
BATCH_TAG="${BATCH_TAG:-mixpast-v1}"
DATE_TAG="${DATE_TAG:-20260512}"
MAX_TRAIN_ITER="${MAX_TRAIN_ITER:-3000}"
CKPT_EVERY="${CKPT_EVERY:-50}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/curvytron_${BATCH_TAG}_launch_${DATE_TAG}.log}"
START_AT="${1:-${START_AT:-1}}"

R27_I250="training/lightzero-curvytron-visual-survival/curvy2seat-overnight40a-27-no-obs-noise-fast-b64-sim8-obs-noise-0/checkpoints/lightzero/iteration_250.pth.tar"
R27_I100="training/lightzero-curvytron-visual-survival/curvy2seat-overnight40a-27-no-obs-noise-fast-b64-sim8-obs-noise-0/checkpoints/lightzero/iteration_100.pth.tar"
R27_I050="training/lightzero-curvytron-visual-survival/curvy2seat-overnight40a-27-no-obs-noise-fast-b64-sim8-obs-noise-0/checkpoints/lightzero/iteration_50.pth.tar"
R18_I150="training/lightzero-curvytron-visual-survival/curvy2seat-selfplay-overnight40a-18-lr-1e-4-fast-gray64-direct-b64-sim8-lr1e-4/checkpoints/lightzero/iteration_150.pth.tar"
R04_I150="training/lightzero-curvytron-visual-survival/curvy2seat-selfplay-overnight40a-04-main-seed-fast-gray64-direct-b64-sim8/checkpoints/lightzero/iteration_150.pth.tar"
R04_I100="training/lightzero-curvytron-visual-survival/curvy2seat-selfplay-overnight40a-04-main-seed-fast-gray64-direct-b64-sim8/checkpoints/lightzero/iteration_100.pth.tar"
R05_I150="training/lightzero-curvytron-visual-survival/curvy2seat-selfplay-overnight40a-05-main-seed-fast-gray64-direct-b64-sim8/checkpoints/lightzero/iteration_150.pth.tar"

mkdir -p "${LOG_DIR}"
: > "${LOG_PATH}"

launch_row() {
  local id="$1"
  local lane="$2"
  local base="$3"
  local seed="$4"
  local frozen_probability="$5"
  local frozen_checkpoint_ref="$6"
  local frozen_player_id="$7"

  if (( 10#${id} < ${START_AT} )); then
    return 0
  fi

  local safe_lane="${lane//_/-}"
  local run_id="${BATCH_TAG}-${id}-${safe_lane}-${DATE_TAG}"
  local attempt_id="${BATCH_TAG}-${id}-${safe_lane}-${DATE_TAG}"

  local cmd=(
    uv run --extra modal modal run --quiet --detach -m "${MODULE}"
    --mode two-seat-selfplay
    --compute gpu-l4-t4
    --seed "${seed}"
    --run-id "${run_id}"
    --attempt-id "${attempt_id}"
    --max-train-iter "${MAX_TRAIN_ITER}"
    --lightzero-eval-freq 0
    --save-ckpt-after-iter "${CKPT_EVERY}"
    --batch-size 64
    --num-simulations 8
    --two-seat-collect-steps-per-iteration 64
    --two-seat-updates-per-iteration 4
    --two-seat-replay-scope accumulated
    --two-seat-learner-sample-size 256
    --two-seat-max-replay-rows 65536
    --two-seat-death-mode normal
    --two-seat-max-ticks 65536
    --two-seat-trail-render-mode fast_gray64_direct
    --two-seat-action-noop-probability 0
    --two-seat-policy-action-repeat-min 1
    --two-seat-policy-action-repeat-max 1
    --two-seat-policy-action-repeat-extra-probability 0
    --background-eval-launch-kind poller
    --output-detail compact
  )

  case "${base}" in
    default)
      cmd+=(--two-seat-observation-noise-std 0.10)
      ;;
    noobs)
      cmd+=(--two-seat-observation-noise-std 0)
      ;;
    lr1e4)
      cmd+=(--two-seat-observation-noise-std 0.10 --two-seat-learning-rate 0.0001)
      ;;
    *)
      print -r -- "Unknown base config: ${base}" >&2
      return 2
      ;;
  esac

  if [[ "${frozen_probability}" != "0" && "${frozen_probability}" != "0.0" ]]; then
    cmd+=(
      --two-seat-frozen-opponent-probability "${frozen_probability}"
      --two-seat-frozen-opponent-checkpoint-ref "${frozen_checkpoint_ref}"
      --two-seat-frozen-opponent-player-id "${frozen_player_id}"
      --two-seat-frozen-opponent-num-simulations 8
    )
  fi

  {
    print -r -- ""
    print -r -- "===== launch ${id} ${run_id} base=${base} p=${frozen_probability} fseat=${frozen_player_id} seed=${seed} $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    if [[ "${frozen_probability}" != "0" && "${frozen_probability}" != "0.0" ]]; then
      print -r -- "frozen_checkpoint_ref=${frozen_checkpoint_ref}"
    fi
    print -r -- "${cmd[@]}"
  } | tee -a "${LOG_PATH}"
  "${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
}

launch_row 01 baseline-r27-noobs-p0 noobs 2401 0 "" 1
launch_row 02 baseline-default-p0 default 2402 0 "" 1
launch_row 03 r27near250-p10-f1 noobs 2403 0.10 "${R27_I250}" 1
launch_row 04 r27near250-p25-f1 noobs 2404 0.25 "${R27_I250}" 1
launch_row 05 r27near250-p25-f0 noobs 2405 0.25 "${R27_I250}" 0
launch_row 06 r27near250-p50-f1 noobs 2406 0.50 "${R27_I250}" 1
launch_row 07 r27mid100-p25-f1 noobs 2407 0.25 "${R27_I100}" 1
launch_row 08 r27mid100-p25-f0 noobs 2408 0.25 "${R27_I100}" 0
launch_row 09 r27old50-p25-f1 noobs 2409 0.25 "${R27_I050}" 1
launch_row 10 r27old50-p25-f0 noobs 2410 0.25 "${R27_I050}" 0
launch_row 11 default-r04near150-p25-f1 default 2411 0.25 "${R04_I150}" 1
launch_row 12 default-r04near150-p25-f0 default 2412 0.25 "${R04_I150}" 0
launch_row 13 default-r05near150-p25-f1 default 2413 0.25 "${R05_I150}" 1
launch_row 14 default-r05near150-p25-f0 default 2414 0.25 "${R05_I150}" 0
launch_row 15 default-r04mid100-p25-f1 default 2415 0.25 "${R04_I100}" 1
launch_row 16 baseline-r18-lr1e4-p0 lr1e4 2416 0 "" 1
launch_row 17 r18near150-p25-f1 lr1e4 2417 0.25 "${R18_I150}" 1
launch_row 18 r18near150-p25-f0 lr1e4 2418 0.25 "${R18_I150}" 0
launch_row 19 r18near150-p10-f1 lr1e4 2419 0.10 "${R18_I150}" 1
launch_row 20 r18near150-p50-f1 lr1e4 2420 0.50 "${R18_I150}" 1

print -r -- ""
print -r -- "All launch commands completed. Log: ${LOG_PATH}"
