#!/bin/bash

set -e  # Stoppt das Skript bei Fehlern

# vor erstem run:
# sed -i 's/\r$//' birfurcation_diagrams_dadras_e.sh
# chmod +x birfurcation_diagrams_dadras_e.sh
# dann starten mit:
# ./birfurcation_diagrams_dadras_e.sh

cd ..
cd DynaMix-python-b-tipping
source venv/bin/activate

cd ..
cd diffrentbutconst_prams

GPU_ID=${1:-${GPU_ID:-0}}
REGIMES=("dadras_e_49_to_64" "tsucs1_e_20_to_245")

for REGIME in "${REGIMES[@]}"; do
  echo "========================================"
  echo "Bifurcation diagrams for ${REGIME}"
  echo "========================================"

  # Cphi modell
  python compare_model_bifurcation_diagrams.py \
    --repo-root ../DynaMix-python-b-tipping \
    --regime "$REGIME" \
    --model-runs \
      "../DynaMix-python-b-tipping/results/single_new_regimes_const_values8/${REGIME}_random_full_b_tipping_phi" \
      "../DynaMix-python-b-tipping/results/single_new_regimes_const_values8/${REGIME}_random_phi_lt_0_b_tipping_phi" \
      "../DynaMix-python-b-tipping/results/single_new_regimes_const_values8/${REGIME}_random_phi_gt_0_b_tipping_phi" \
      "../DynaMix-python-b-tipping/results/single_new_regimes_const_values8/${REGIME}_values_8_b_tipping_phi" \
    --model-labels chaotic cyclic both 8values \
    --components z \
    --output "../DynaMix-python-b-tipping/results/single_new_regimes_const_values8/birfurcation_diagrams/${REGIME}_bifurcation_compare.png" \
    --cache "../DynaMix-python-b-tipping/results/single_new_regimes_const_values8/birfurcation_diagrams/${REGIME}_bifurcation_cache.npz" \
    --device cuda \
    --num-parameters 320 \
    --gpu-id "$GPU_ID"

  # context modell
  python compare_model_bifurcation_diagrams.py \
    --repo-root ../DynaMix_context_phi \
    --regime "$REGIME" \
    --model-runs \
      "../DynaMix_context_phi/results/single_new_regimes_const_values8/${REGIME}_random_full_context_phi" \
      "../DynaMix_context_phi/results/single_new_regimes_const_values8/${REGIME}_random_phi_lt_0_context_phi" \
      "../DynaMix_context_phi/results/single_new_regimes_const_values8/${REGIME}_random_phi_gt_0_context_phi" \
      "../DynaMix_context_phi/results/single_new_regimes_const_values8/${REGIME}_values_8_context_phi" \
    --model-labels chaotic cyclic both 8values \
    --components z \
    --output "../DynaMix_context_phi/results/single_new_regimes_const_values8/birfurcation_diagrams/${REGIME}_bifurcation_compare.png" \
    --cache "../DynaMix_context_phi/results/single_new_regimes_const_values8/birfurcation_diagrams/${REGIME}_bifurcation_cache.npz" \
    --device cuda \
    --num-parameters 320 \
    --gpu-id "$GPU_ID"
done


