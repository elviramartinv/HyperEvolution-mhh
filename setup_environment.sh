#!/bin/bash

# Script to configure HyperEvolution-mhh environment on CERN/lxplus

echo "Setting up HyperEvolution-mhh environment..."

# Activate virtual environment
source Hopt/bin/activate

# Configure ROOT
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.36.04/x86_64-almalinux9.6-gcc115-opt/bin/thisroot.sh

# Install additional dependencies
pip install sympy

# Verify ROOT works
echo "✅ Verifying ROOT..."
python -c "import ROOT; print('✅ ROOT version:', ROOT.gROOT.GetVersion())" || {
    echo "❌ Error: ROOT is not working properly"
    exit 1
}

# Verify sympy works
echo "✅ Verifying sympy..."
python -c "import sympy; print('✅ sympy version:', sympy.__version__)" || {
    echo "❌ Error: sympy is not working properly"
    exit 1
}

# Verify other dependencies
echo "✅ Verifying other dependencies..."
python -c "
try:
    import docopt
    import numpy as np
    import hyperevol.tools.particle_swarm as pso
    print('✅ All dependencies are available')
except ImportError as e:
    print(f'❌ Error importing dependency: {e}')
    exit(1)
"

echo ""
echo "Environment configured successfully!"
echo ""