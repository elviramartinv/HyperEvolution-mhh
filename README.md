## Setup

**Create virtual environment:**
   ```bash
   python -m venv Hopt
   ```

**Setup environment automatically:**
   ```bash
   source setup_environment.sh
   ```
## hyperevol

Application of **hyperparameter optimization** algorithms for **HEFT basis optimization** (Higgs Effective Field Theory). Based on [HyperEvolution](https://github.com/Laurits7/HyperEvolution).


**Run quick test:**
   ```bash
   cd hyperevol/examples
   python mhh_scoring.py -c config/pso_cfg_test.json -p None
   ```

---

## LHEReader

Converts LHE files into ROOT NTuples for analysis.

### Usage
```bash
python LHEReader/LHEConverter.py -i input.lhe -o output.root
```
## mHH_generator

Extracts di-Higgs kinematic variables from LHE-converted ROOT files.

### Key Features
Computes physics observables for di-Higgs production:
- **mHH**: Invariant mass of HH system
- **pT(H1), pT(H2)**: Transverse momentum of each Higgs
- **pT(HH)**: Transverse momentum of HH system
- **cosTheta**: Opening angle between Higgs bosons

### Usage

**Single file:**
```bash
python mHH_generator/mhh_extractor.py input.root output.root
```

**Entire directory:**
```bash
python mHH_generator/mhh_extractor.py /path/to/samples/ output.root
```


---

## Studies

### 1. `reweight_LHE_comparison.py` - Convention Validation

Compares mHH distributions from two sources:
- **reweighting**: Using `calcDist()`
- **LHE events**: Direct Monte Carlo simulations


#### Usage

**Example:**
```bash
python reweight_LHE_comparison.py <index>
```
- `<index>`: Basis point index (0-22)


### 2. `mhh_morphing.py` - Distribution Morphing & Validation

Performs mHH distribution morphing using 23 basis points and validates against real LHE data with matrix inversion method

#### Key Features
- Matrix inversion
- Automatic interpolation/extrapolation detection using convex hull
- Phase space visualization

#### Usage

**Example:**
```bash
python mhh_morphing.py [options]
```

