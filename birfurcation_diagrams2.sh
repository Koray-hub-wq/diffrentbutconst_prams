#!/bin/bash

set -e  # Stoppt das Skript bei Fehlern

# vor erstem run:
# sed -i 's/\r$//' birfurcation_diagrams2.sh
# chmod +x birfurcation_diagrams2.sh
# dann starten mit:
# ./birfurcation_diagrams2.sh

cd ..
cd DynaMix-python-b-tipping
source venv/bin/activate

cd ..
cd diffrentbutconst_prams

GPU_ID=${1:-${GPU_ID:-0}}

#pretrained
python compare_model_bifurcation_diagrams.py \
  --model-type pretrained \
  --pretrained-repo-root ../Dynamix_original \
  --pretrained-model-names dynamix-3d-alrnn-v1.0 \
  --model-labels pretrained \
  --regime halvorsen \
  --components z \
  --output ../more_results/birfurcation_diagrams/pretrained/halvorsen_pretrained_bifurcation_compare.png \
  --cache ../more_results/birfurcation_diagrams/pretrained/halvorsen_pretrained_bifurcation_cache.npz \
  --device cuda \
  --gpu-id $GPU_ID \
  --num-parameters 320

# context+cphi
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
