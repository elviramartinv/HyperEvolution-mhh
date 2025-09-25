# HyperEvolution-mhh

Application of **hyperparameter optimization** algorithms for **HEFT basis optimization** (Higgs Effective Field Theory). Based on [HyperEvolution](https://github.com/Laurits7/HyperEvolution).

## Setup

**Create virtual environment:**
   ```bash
   python -m venv Hopt
   ```

**Setup environment automatically:**
   ```bash
   ./setup_environment.sh
   ```

**Run quick test:**
   ```bash
   cd hyperevol/examples
   python mhh_scoring.py -c config/pso_cfg_test.json -p None
   ```

**Run on cluster:**
   ```bash
   ./batch/run_batch.sh hyperevol/examples/config/pso_cfg_b23_1.json
   ```