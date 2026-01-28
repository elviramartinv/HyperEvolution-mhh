#!/bin/bash
# Script to convert all .lhe files in the source directory to .root files
# and save them in the destination directory, preserving the folder structure.
# Usage: ./convert_all_lhe.sh

# SRC_BASE="/eos/cms/store/group/phys_b2g/event_level_reweighting_HH/ggHH_basispoint_13_13p6"
SRC_BASE="/eos/cms/store/group/phys_b2g/event_level_reweighting_HH/HEFT_13p6"
# DST_BASE="/eos/user/e/emartinv/event_level_reweighting_HH/ggHH_basispoint_13_13p6"
DST_BASE="/eos/user/e/emartinv/event_level_reweighting_HH/HEFT_13p6"
LHE_CONVERTER="/afs/cern.ch/user/e/emartinv/public/HyperEvolution-mhh/LHEReader/LHEConverter.py"

echo "SRC_BASE: $SRC_BASE"
echo "DST_BASE: $DST_BASE"
echo "LHE_CONVERTER: $LHE_CONVERTER"

echo "--- INITIALIZING ---"

count=0

for dir in $(ls "$SRC_BASE"); do
  src_dir="$SRC_BASE/$dir"
  dst_dir="$DST_BASE/$dir"
  echo ""
  echo "Processing folder: $src_dir"
  mkdir -p "$dst_dir"
  lhe_files=$(ls "$src_dir"/*.lhe 2>/dev/null)
  if [ -z "$lhe_files" ]; then
    echo "  ⚠️  No .lhe files found in $src_dir"
  fi
  for lhe in $lhe_files; do
    fname=$(basename "$lhe" .lhe)
    outroot="$dst_dir/$fname.root"
    # Check if output exists and is non-empty
    if [[ -s "$outroot" ]]; then
      echo "  Skipping $lhe (output exists and is non-empty)"
      continue
    fi
    echo "  Converting $lhe -> $outroot"
    python3 "$LHE_CONVERTER" -i "$lhe" -o "$outroot"
    status=$?
    ((count++))

    if [ $status -ne 0 ]; then
      echo "  ❌ Error converting $lhe"
    fi

    if (( count % 100 == 0 )); then
      echo ""
      echo "  --- Converted $count files so far ---"
      read -p " Continue? (y/n): " answer
      case $answer in
        [Yy]* ) ;;
        [Nn]* ) echo " Aborted by user."; exit ;;
        * ) echo "  Invalid input. Continuing...";;
      esac
      echo ""
    fi
  done
done

echo "--- COMPLETE ---"
echo "Total files converted: $count"