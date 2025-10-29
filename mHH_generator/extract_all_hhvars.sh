#!/bin/bash
# Script to execute mhh_extractor on all sample folders and save outputs in HH_variables

MHH_EXTRACTOR="/afs/cern.ch/user/e/emartinv/public/HyperEvolution-mhh/mHH_generator/mhh_extractor.py"
# BASE_DIR="/eos/user/e/emartinv/event_level_reweighting_HH/ggHH_basispoint_13_13p6"
BASE_DIR="/eos/user/e/emartinv/event_level_reweighting_HH/HEFT_13p6"

for sample_dir in $(ls "$BASE_DIR"); do
  full_sample_dir="$BASE_DIR/$sample_dir"
  if [ -d "$full_sample_dir" ]; then
    outdir="$full_sample_dir/HH_variables"
    outroot="$outdir/output.root"
    # Check if output already exists and is valid
    if [[ -s "$outroot" ]]; then
      echo "Skipping $full_sample_dir (output exists and is non-empty)"
      continue
    fi
    mkdir -p "$outdir"
    echo "Processing $full_sample_dir -> $outroot"
    python3 "$MHH_EXTRACTOR" "$full_sample_dir" "$outroot"
    status=$?
    if [ $status -ne 0 ]; then
      echo " Error processing $full_sample_dir"
    fi
  fi
done

echo "--- Processing complete ---"
