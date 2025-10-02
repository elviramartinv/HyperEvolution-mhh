# HyperEvolution-mhh

Application of **hyperparameter optimization** algorithms for **HEFT basis optimization** (Higgs Effective Field Theory). Based on [HyperEvolution](https://github.com/Laurits7/HyperEvolution).

## Setup

**Create virtual environment:**
   ```bash
   python -m venv Hopt
   ```

**Setup environment automatically:**
   ```bash
   source setup_environment.sh
   ```

**Run quick test:**
   ```bash
   cd hyperevol/examples
   python mhh_scoring.py -c config/pso_cfg_test.json -p None
   ```