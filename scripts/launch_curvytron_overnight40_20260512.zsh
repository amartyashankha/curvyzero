#!/usr/bin/env zsh
set -euo pipefail

MODULE="curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
BATCH_TAG="${BATCH_TAG:-overnight40a}"
DATE_TAG="${DATE_TAG:-20260512}"
MAX_TRAIN_ITER="${MAX_TRAIN_ITER:-3000}"
CKPT_EVERY="${CKPT_EVERY:-50}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/curvytron_${BATCH_TAG}_launch_${DATE_TAG}.log}"
START_AT="${1:-${START_AT:-1}}"

mkdir -p "${LOG_DIR}"
: > "${LOG_PATH}"

launch_row() {
  local id="$1"
  local lane="$2"
  local compute="$3"
  local render="$4"
  local batch="$5"
  local sim="$6"
  local collect="$7"
  local updates="$8"
  local sample="$9"
  local lr="${10}"
  local reward="${11}"
  local stochastic="${12}"
  local seed="${13}"

  if (( 10#${id} < ${START_AT} )); then
    return 0
  fi

  local safe_lane="${lane//_/-}"
  local safe_reward="${reward//_/-}"
  local safe_stochastic="${stochastic//_/-}"
  local safe_render="${render//_/-}"
  local render_tag="${safe_render}"
  if [[ "${render}" == "fast_gray64_direct" ]]; then
    render_tag="fast"
  elif [[ "${render}" == "browser_lines" ]]; then
    render_tag="browser"
  fi
  local run_id="curvy2seat-${BATCH_TAG}-${id}-${safe_lane}-${render_tag}-b${batch}-sim${sim}"
  if [[ "${reward}" != "default" ]]; then
    run_id="${run_id}-${safe_reward}"
  fi
  if [[ "${stochastic}" != "default" ]]; then
    run_id="${run_id}-${safe_stochastic}"
  fi
  if [[ "${lr}" != "unset" ]]; then
    run_id="${run_id}-lr${lr//./p}"
  fi
  local attempt_id="${BATCH_TAG}-${id}-${safe_lane}-${DATE_TAG}"

  local cmd=(
    uv run --extra modal modal run --quiet --detach -m "${MODULE}"
    --mode two-seat-selfplay
    --compute "${compute}"
    --seed "${seed}"
    --run-id "${run_id}"
    --attempt-id "${attempt_id}"
    --max-train-iter "${MAX_TRAIN_ITER}"
    --lightzero-eval-freq 0
    --save-ckpt-after-iter "${CKPT_EVERY}"
    --batch-size "${batch}"
    --num-simulations "${sim}"
    --two-seat-collect-steps-per-iteration "${collect}"
    --two-seat-updates-per-iteration "${updates}"
    --two-seat-replay-scope accumulated
    --two-seat-learner-sample-size "${sample}"
    --two-seat-max-replay-rows 65536
    --two-seat-death-mode normal
    --two-seat-trail-render-mode "${render}"
    --background-eval-launch-kind poller
    --output-detail compact
  )

  case "${lr}" in
    unset) ;;
    *) cmd+=(--two-seat-learning-rate "${lr}") ;;
  esac

  case "${reward}" in
    default) ;;
    no_bonus)
      cmd+=(--two-seat-bonus-pickup-reward-per-catch 0)
      ;;
    terminal_only)
      cmd+=(--two-seat-alive-reward 0 --two-seat-bonus-pickup-reward-per-catch 0)
      ;;
    terminal_x2)
      cmd+=(--two-seat-terminal-outcome-reward-per-step 0.02)
      ;;
    survival_only)
      cmd+=(--two-seat-terminal-outcome-reward-per-step 0 --two-seat-bonus-pickup-reward-per-catch 0)
      ;;
    bonus_x2)
      cmd+=(--two-seat-bonus-pickup-reward-per-catch 0.10)
      ;;
    *)
      print -r -- "Unknown reward variant: ${reward}" >&2
      return 2
      ;;
  esac

  case "${stochastic}" in
    default)
      cmd+=(--two-seat-observation-noise-std 0.10 --two-seat-action-noop-probability 0 --two-seat-policy-action-repeat-min 1 --two-seat-policy-action-repeat-max 1 --two-seat-policy-action-repeat-extra-probability 0)
      ;;
    obs_noise_0)
      cmd+=(--two-seat-observation-noise-std 0 --two-seat-action-noop-probability 0 --two-seat-policy-action-repeat-min 1 --two-seat-policy-action-repeat-max 1 --two-seat-policy-action-repeat-extra-probability 0)
      ;;
    obs_noise_05)
      cmd+=(--two-seat-observation-noise-std 0.05 --two-seat-action-noop-probability 0 --two-seat-policy-action-repeat-min 1 --two-seat-policy-action-repeat-max 1 --two-seat-policy-action-repeat-extra-probability 0)
      ;;
    obs_noise_20)
      cmd+=(--two-seat-observation-noise-std 0.20 --two-seat-action-noop-probability 0 --two-seat-policy-action-repeat-min 1 --two-seat-policy-action-repeat-max 1 --two-seat-policy-action-repeat-extra-probability 0)
      ;;
    repeat_20pct)
      cmd+=(--two-seat-observation-noise-std 0.10 --two-seat-action-noop-probability 0 --two-seat-policy-action-repeat-min 1 --two-seat-policy-action-repeat-max 3 --two-seat-policy-action-repeat-extra-probability 0.20)
      ;;
    action_noop_5pct)
      cmd+=(--two-seat-observation-noise-std 0.10 --two-seat-action-noop-probability 0.05 --two-seat-policy-action-repeat-min 1 --two-seat-policy-action-repeat-max 1 --two-seat-policy-action-repeat-extra-probability 0)
      ;;
    none)
      cmd+=(--two-seat-observation-noise-std 0 --two-seat-action-noop-probability 0 --two-seat-policy-action-repeat-min 1 --two-seat-policy-action-repeat-max 1 --two-seat-policy-action-repeat-extra-probability 0)
      ;;
    *)
      print -r -- "Unknown stochasticity variant: ${stochastic}" >&2
      return 2
      ;;
  esac

  {
    print -r -- ""
    print -r -- "===== launch ${id} ${run_id} seed=${seed} $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    print -r -- "${cmd[@]}"
  } | tee -a "${LOG_PATH}"
  "${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
}

launch_row 01 main gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset default default 1201
launch_row 02 main_seed gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset default default 1202
launch_row 03 main_seed gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset default default 1203
launch_row 04 main_seed gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset default default 1204
launch_row 05 main_seed gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset default default 1205
launch_row 06 search16 gpu-l4-t4 fast_gray64_direct 64 16 64 4 256 unset default default 1206
launch_row 07 search32 gpu-l4-t4 fast_gray64_direct 64 32 64 4 256 unset default default 1207
launch_row 08 small_batch gpu-l4-t4 fast_gray64_direct 32 8 64 4 128 unset default default 1208
launch_row 09 large_batch_l4 gpu-l4-t4 fast_gray64_direct 128 8 64 4 512 unset default default 1209
launch_row 10 large_search_l4 gpu-l4-t4 fast_gray64_direct 128 16 64 4 512 unset default default 1210
launch_row 11 collect128 gpu-l4-t4 fast_gray64_direct 64 8 128 4 256 unset default default 1211
launch_row 12 collect256 gpu-l4-t4 fast_gray64_direct 64 8 256 4 256 unset default default 1212
launch_row 13 updates8 gpu-l4-t4 fast_gray64_direct 64 8 64 8 256 unset default default 1213
launch_row 14 updates16 gpu-l4-t4 fast_gray64_direct 64 8 64 16 256 unset default default 1214
launch_row 15 learner512 gpu-l4-t4 fast_gray64_direct 64 8 64 4 512 unset default default 1215
launch_row 16 learner1024 gpu-l4-t4 fast_gray64_direct 128 8 64 4 1024 unset default default 1216
launch_row 17 lr_3e-5 gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 3e-5 default default 1217
launch_row 18 lr_1e-4 gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 1e-4 default default 1218
launch_row 19 lr_3e-4 gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 3e-4 default default 1219
launch_row 20 lr_1e-3 gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 1e-3 default default 1220
launch_row 21 lr_3e-3 gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 3e-3 default default 1221
launch_row 22 no_bonus gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset no_bonus default 1222
launch_row 23 terminal_only gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset terminal_only default 1223
launch_row 24 stronger_terminal gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset terminal_x2 default 1224
launch_row 25 survival_only_ctrl gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset survival_only default 1225
launch_row 26 bonus_heavy gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset bonus_x2 default 1226
launch_row 27 no_obs_noise gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset default obs_noise_0 1227
launch_row 28 obs_noise_05 gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset default obs_noise_05 1228
launch_row 29 obs_noise_20 gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset default obs_noise_20 1229
launch_row 30 action_repeat gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset default repeat_20pct 1230
launch_row 31 action_noop_05 gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset default action_noop_5pct 1231
launch_row 32 no_stochasticity gpu-l4-t4 fast_gray64_direct 64 8 64 4 256 unset default none 1232
launch_row 33 large_batch_h100 gpu-h100-cpu40 fast_gray64_direct 128 8 64 4 512 unset default default 1233
launch_row 34 large_search_h100 gpu-h100-cpu40 fast_gray64_direct 128 16 64 4 512 unset default default 1234
launch_row 35 h100_search32 gpu-h100-cpu40 fast_gray64_direct 128 32 64 4 512 unset default default 1235
launch_row 36 h100_collect128 gpu-h100-cpu40 fast_gray64_direct 128 16 128 4 512 unset default default 1236
launch_row 37 h100_b256 gpu-h100-cpu40 fast_gray64_direct 256 8 64 4 1024 unset default default 1237
launch_row 38 h100_lr_3e-4 gpu-h100-cpu40 fast_gray64_direct 128 8 64 4 512 3e-4 default default 1238
launch_row 39 browser_sentinel gpu-l4-t4 browser_lines 16 8 64 4 128 unset default default 1239
launch_row 40 browser_sentinel_lr gpu-l4-t4 browser_lines 16 8 64 4 128 3e-4 default default 1240

print -r -- ""
print -r -- "All launch commands completed. Log: ${LOG_PATH}"
