#!/bin/bash
# Script to run mhh_inference.py for all points in the true_base directory
# Usage: bash inference_all_points.sh

TRUE_BASE="/eos/user/e/emartinv/event_level_reweighting_HH/ggHH_basispoint_13_13p6"
INFERENCE_SCRIPT="$(dirname "$0")/mhh_inference.py"

for folder in "$TRUE_BASE"/testrun_params_klambda_*; do
    if [ -d "$folder" ]; then
        # Extract parameters from folder name using regex
        if [[ $folder =~ testrun_params_klambda_([-0-9.eE]+)_ct_([-0-9.eE]+)_ctt_([-0-9.eE]+)_cggh_([-0-9.eE]+)_cgghh_([-0-9.eE]+)_energy_([0-9]+)_TeV ]]; then
            kl="${BASH_REMATCH[1]}"
            kt="${BASH_REMATCH[2]}"
            c2="${BASH_REMATCH[3]}"
            cg="${BASH_REMATCH[4]}"
            c2g="${BASH_REMATCH[5]}"
            energy="${BASH_REMATCH[6]}"
            echo "Running inference for kl=$kl, kt=$kt, c2=$c2, cg=$cg, c2g=$c2g, energy=$energy"
            python "$INFERENCE_SCRIPT" --kl "$kl" --kt "$kt" --c2 "$c2" --cg "$cg" --c2g "$c2g" --energy "$energy" --compare
        else
            echo "Could not parse parameters from $folder"
        fi
    fi
done
