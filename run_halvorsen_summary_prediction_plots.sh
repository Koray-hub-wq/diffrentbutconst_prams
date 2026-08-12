#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/export/home/klenkeit/DynaMix-python-b-tipping"
RESULTS_ROOT="../DynaMix-python-b-tipping/results/single_halvorsen_constparams"
DATA_DIR="../training_data/single_halvorsen_constparams/14to209_9params"
PLOTS_ROOT="../Dynamixclub_präsi/plots"
GPU_ID="${GPU_ID:-5}"

models=(
  "14to209_noisy_5negativeparams"
  "14to209_noisy"
  "14to209_noisy_5positiveparams"
)

trained_params=(
  "-1 -0.75 -0.5 -0.25 0"
  "-1 -0.5 0 0.5 1"
  "0 0.25 0.5 0.75 1"
)

unseen_params=(
  "0.25 0.5 0.75 1"
  "-0.75 -0.25 0.25 0.75"
  "-1 -0.75 -0.5 -0.25"
)

for i in "${!models[@]}"; do
  model="${models[$i]}"
  read -r -a trained <<< "${trained_params[$i]}"
  read -r -a unseen <<< "${unseen_params[$i]}"

  cache_dir="${PLOTS_ROOT}/summary_plots_all_params_${model}"
  with_phi_plot_dir="${PLOTS_ROOT}/${model}"
  no_phi_plot_dir="${PLOTS_ROOT}/${model}_nophi"

  mkdir -p "$cache_dir" "$with_phi_plot_dir" "$no_phi_plot_dir"

  echo "=== ${model}: with-phi predictions ==="
  python make_with_phi_prediction_cache_and_plot.py \
    --repo-root "$REPO_ROOT" \
    --model-run "${RESULTS_ROOT}/${model}-with-phi" \
    --model-label "with phi" \
    --data-dir "$DATA_DIR" \
    --output-dir "$cache_dir" \
    --device cuda \
    --gpu-id "$GPU_ID" \
    --parameter-space phi \
    --label-space raw \
    --raw-name a \
    --title "All parameter values"

  echo "=== ${model}: with-phi training subset ==="
  python plot_with_phi_prediction_grid_from_cache.py \
    --cache "${cache_dir}/with_phi_median_prediction_cache.npz" \
    --output "${with_phi_plot_dir}/training_subset.png" \
    --params "${trained[@]}" \
    --parameter-space phi \
    --single-row \
    --label-space none

  echo "=== ${model}: with-phi unseen subset ==="
  python plot_with_phi_prediction_grid_from_cache.py \
    --cache "${cache_dir}/with_phi_median_prediction_cache.npz" \
    --output "${with_phi_plot_dir}/unseen_subset.png" \
    --params "${unseen[@]}" \
    --parameter-space phi \
    --single-row \
    --label-space none

  echo "=== ${model}: no-phi predictions ==="
  python make_with_phi_prediction_cache_and_plot.py \
    --repo-root "$REPO_ROOT" \
    --model-run "${RESULTS_ROOT}/${model}-no-phi" \
    --model-label "no phi" \
    --data-dir "$DATA_DIR" \
    --output-dir "$cache_dir" \
    --device cuda \
    --gpu-id "$GPU_ID" \
    --parameter-space phi \
    --label-space raw \
    --raw-name a \
    --title "All parameter values"

  echo "=== ${model}: no-phi training subset ==="
  python plot_with_phi_prediction_grid_from_cache.py \
    --cache "${cache_dir}/no_phi_median_prediction_cache.npz" \
    --output "${no_phi_plot_dir}/training_subset.png" \
    --params "${trained[@]}" \
    --parameter-space phi \
    --single-row \
    --label-space none

  echo "=== ${model}: no-phi unseen subset ==="
  python plot_with_phi_prediction_grid_from_cache.py \
    --cache "${cache_dir}/no_phi_median_prediction_cache.npz" \
    --output "${no_phi_plot_dir}/unseen_subset.png" \
    --params "${unseen[@]}" \
    --parameter-space phi \
    --single-row \
    --label-space none
done

