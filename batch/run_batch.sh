#!/bin/bash

# Batch system execution script for HyperEvolution-mhh
# Usage: ./run_batch.sh [config_file]

set -e

# Detect batch system and set default configuration
if command -v sbatch &> /dev/null; then
    BATCH_SYSTEM="SLURM"
    CONFIG_FILE=${1:-"hyperevol/examples/config/pso_cfg_slurm_b23.json"}
elif command -v condor_submit &> /dev/null; then
    BATCH_SYSTEM="HTCondor"
    CONFIG_FILE=${1:-"hyperevol/examples/config/pso_cfg_condor_b23.json"}
else
    echo "❌ Error: No batch system found (sbatch or condor_submit)"
    echo "For local execution, use directly: python mhh_scoring.py -c config/pso_cfg_test.json -p None"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$PROJECT_ROOT"

echo "Starting optimization with $BATCH_SYSTEM..."
echo "Project directory: $SCRIPT_DIR"
echo "Configuration file: $CONFIG_FILE"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Load environment
echo "Setting up environment..."
cd "$PROJECT_ROOT"
if [ -f "setup_environment.sh" ]; then
    source setup_environment.sh
else
    source Hopt/bin/activate
    source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.36.04/x86_64-almalinux9.6-gcc115-opt/bin/thisroot.sh
fi

# Change to examples directory and execute
cd hyperevol/examples
echo "Running optimization..."
python3 mhh_scoring.py -c "../../$CONFIG_FILE" -p None

echo "Results will be saved according to the configuration in $CONFIG_FILE"