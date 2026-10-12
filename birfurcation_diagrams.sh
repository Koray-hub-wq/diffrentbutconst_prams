#!/bin/bash

set -e  # Stoppt das Skript bei Fehlern

# vor erstem run:
# sed -i 's/\r$//' birfurcation_diagrams.sh
# chmod +x birfurcation_diagrams.sh
# dann starten mit:
# ./birfurcation_diagrams.sh

cd ..
cd DynaMix-python-b-tipping
source venv/bin/activate

cd ..
cd diffrentbutconst_prams

GPU_ID=${1:-${GPU_ID:-0}}

python compare_model_bifurcation_diagrams.py \
  --repo-root ../DynaMix_context_phi \
  --regime halvorsen \
  --model-runs ../DynaMix_context_phi/results/single_halvorsen_constparams/3neg_params_with_phi ../DynaMix_context_phi/results/single_halvorsen_constparams/5pos_params_with_phi ../DynaMix_context_phi/results/single_halvorsen_constparams/5params_with_phi \
  --model-labels chaotic cyclic both \
  --components z \
  --output ../DynaMix_context_phi/results/single_halvorsen_constparams/birfurcation_diagrams/halvorsen_bifurcation_compare.png \
  --cache ../DynaMix_context_phi/results/single_halvorsen_constparams/birfurcation_diagrams/halvorsen_bifurcation_cache.npz \
  --device cuda \
  --num-parameters 320 \
  --gpu-id $GPU_ID

python compare_model_bifurcation_diagrams.py \
  --repo-root ../DynaMix-python-b-tipping \
  --regime halvorsen \
  --model-runs ../DynaMix-python-b-tipping/results/single_halvorsen_constparams/14to158_3params_with_phi ../DynaMix-python-b-tipping/results/single_halvorsen_constparams/14to209_noisy_5positiveparams-with-phi ../DynaMix-python-b-tipping/results/single_halvorsen_constparams/14to209_noisy-with-phi \
  --model-labels chaotic cyclic both \
  --components z \
  --output ../DynaMix-python-b-tipping/results/single_halvorsen_constparams/birfurcation_diagrams/320params_15ksteps/halvorsen_bifurcation_compare.png \
  --cache ../DynaMix-python-b-tipping/results/single_halvorsen_constparams/birfurcation_diagrams/320params_15ksteps/halvorsen_bifurcation_cache.npz \
  --device cuda \
  --num-parameters 320 \
  --gpu-id $GPU_ID
