''' This script runs the Particle swarm optimization (PSO) with a given batch
size for 1000 repeats each optimization consists out of 10k total evaluations.
Call with 'python'

Usage:
    mhh_scoring.py [--parameter_file=PTH] --pso_file=PTH
    mhh_scoring.py --resume=DIR [--pso_file=PTH]
    mhh_scoring.py --monitor=DIR

Options:
    -p --parameter_file=PTH         Path to parameters to be run
    -c --pso_file=PTH               PSOconfig
    --resume=DIR                    Resume interrupted session from directory
    --monitor=DIR                   Monitor running jobs in directory
'''

import functools
from multiprocessing import Pool
import ROOT
import sympy
import math
import random
import os
import json
import docopt
import numpy as np
from hyperevol.examples.helper import read_cfg, save_results
from hyperevol.tools import particle_swarm as pso
from array import array
import glob
from pathlib import Path
from textwrap import dedent
import subprocess
import time
import shutil
import json
import signal
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# new reweighting weights  (differential XS) from matheus
# new theocoffs
masslist=set()
theocoeffs = []

import json

def load_coefficients(energy='13p6'):
    global theocoeffs, masslist

    # Select file and JSON key based on energy
    if energy == '13':
        coeffs_file = 'mhhcoffs_13.json'
        json_key = 'acop_0.0-1.0_theta_0.0-1.0'
    elif energy == '13p6':
        coeffs_file = 'mhhcoffs_13p6.json'
        json_key = 'pthh_0.0-10000_theta_0.0-1.0'
    else:
        raise ValueError(f"Unknown energy: {energy}. Use '13' or '13p6'")

    coeffs_path = os.path.join(os.path.dirname(__file__), coeffs_file)

    with open(coeffs_path) as mhhfile:
        mhhjson = json.load(mhhfile)[json_key]

    masslist = set()
    theocoeffs = []

    for key in mhhjson.keys():
        masslist.add(int(key.split('-')[0]))
        masslist.add(int(key.split('-')[1]))
        mass = float(int(key.split('-')[0])+(int(key.split('-')[1])-int(key.split('-')[0]))/2)
        theocoeffs.append([mass]+mhhjson[key]['fitted_parameters'])

    theocoeffs = sorted(theocoeffs, key = lambda x: x[0])
    masslist = sorted(list(masslist))

    return theocoeffs, masslist

theocoeffs, masslist = load_coefficients('13p6')
#mHH = []
#for t in theocoeffs:
#    mHH.append(t[0])

# total XS also from https://arxiv.org/abs/2304.01968
xscoffs = [0.06149445582502288,
 0.3136672029876887,
 0.009301808057440153,
 0.009795448621341175,
 0.35145334073835405,
 -0.2559133980849674,
 -0.04296432612700188,
 0.09085341492517679,
 0.0782758166292356,
 0.4489365969942281,
 -0.03517971157073585,
 -0.16202459507304207,
 0.018071792664938892,
 0.07830485152133954,
 0.08840058043879137,
 -0.0002544699820427092,
 0.0006506059098975965,
 0.0005760121632388893,
 0.0009511381868594431,
 -0.0008664442180013471,
 0.002169652647647945,
 0.000422901508452418,
 0.003354598203554643]

# NLO formula for gghH XS
def calcXS(coeffs, kl, kt, c2, cg, c2g):
    # comment this and change in config to use Nicholas convention
    cg = cg/1.5
    c2g = -c2g/3
    xs = coeffs[0]*kt*kt*kt*kt
    xs += coeffs[1]*c2*c2
    xs += (coeffs[2]*kt*kt+coeffs[3]*cg*cg)*kl*kl
    xs += coeffs[4]*c2g*c2g
    xs += (coeffs[5]*c2+coeffs[6]*kt*kl)*kt*kt
    xs += (coeffs[7]*kt*kl+coeffs[8]*cg*kl)*c2
    xs += coeffs[9]*c2*c2g
    xs += (coeffs[10]*cg*kl+coeffs[11]*c2g)*kt*kt
    xs += (coeffs[12]*kl*cg+coeffs[13]*c2g)*kt*kl
    xs += coeffs[14]*cg*c2g*kl
    xs += coeffs[15]*kt*kt*kt*cg
    xs += coeffs[16]*kt*c2*cg
    xs += coeffs[17]*kt*cg*cg*kl
    xs += coeffs[18]*kt*cg*c2g
    xs += coeffs[19]*kt*kt*cg*cg
    xs += coeffs[20]*c2*cg*cg
    xs += coeffs[21]*cg*cg*cg*kl
    xs += coeffs[22]*cg*cg*c2g
    return xs

# differential XS distribution
def calcTotXS(kl, kt, c2, cg, c2g):
    return calcXS(xscoffs, kl, kt, c2, cg, c2g)

def calcXSDist(kl, kt, c2, cg, c2g, kSM=1.115):
    xslist = []
    #mhhlist = []
    theo = theocoeffs
    for t in theo:
        if len(t) != 24: continue
        #mhhlist.append(t[0])
        xslist.append(kSM*calcXS(t[1:], kl, kt, c2, cg, c2g))
    xslist = np.array(xslist) * (1/sum(xslist))
    mhhlist = masslist
    #return xslist, (mhhlist+[13000])
    return xslist, mhhlist

# dont always rethrow toys, save instead using key
def makeKey(kl, kt, c2, cg, c2g, samplesize):
    return "kl_{0}_kt_{1}_c2_{2}_cg_{3}_c2g_{4}_{5}events".format(kl, kt, c2, cg, c2g, samplesize)

# throw toys according to differential XS, normalize to 138 fb and 13 TeV XS
@functools.lru_cache(None)
def calcDist(kl, kt, c2, cg, c2g, kSM=1.115, samplesize=5000, maxmhh=1050):
    key = makeKey(kl, kt, c2, cg, c2g, samplesize)
    xs, mhhs = calcXSDist(kl, kt, c2, cg, c2g, kSM)
    mhhsplot = mhhs
    if maxmhh:
        idx = -1
        idx = len(mhhs)-mhhs.index(maxmhh)
        mhhsplot = mhhs[:-idx]
        # renormalize...
        ### xs = xs[:-idx]
        #xs *= 1/sum(xs)
    random.seed(12345)
    data = random.choices(mhhs[:-1], weights=xs, k=samplesize)
    h = ROOT.TH1D(key, key, len(mhhsplot)-1, array('d', mhhsplot))
    h.Sumw2()
    weight = 1000*calcTotXS(kl, kt, c2, cg, c2g)*138/samplesize
    for d in data: h.Fill(d, weight)
    return h

def createScanset(case):
    toys = []
    kl, kt, c2, cg, c2g = [1, 1, 0, 0, 0]
    ranges= [-15,15,0.5,2.5]
    granularity = [30,20]
    if case in ['c2cg', 'c2cg_corr']:
        ranges= [-3,3,-1,1]
        granularity = [30,20]
    elif case=='c2kt':
        ranges= [-4,4,0.5,2.5]
        granularity = [20,20]
    elif case in ['klcg', 'klcg_corr']:
        ranges= [-15,15,-1,1]
        granularity = [30,20]
    elif case=='c2c2g':
        ranges= [-3,3,-1,1]
        granularity = [30,20]
    elif case=='cgc2g':
        ranges= [-3,3,-3,3]
        granularity = [30,30]
    poi1_vals = np.arange(ranges[0], ranges[1], (ranges[1]-ranges[0])/granularity[0])
    poi2_vals = np.arange(ranges[2], ranges[3], (ranges[3]-ranges[2])/granularity[1])
    for poi1 in poi1_vals:
        for poi2 in poi2_vals:
            poi1, poi2 = [round(float(poi1),4), round(float(poi2),4)]
            if case=='klkt':
                kl,kt = [poi1,poi2]
            elif case in ['c2cg', 'c2cg_corr']:
                c2,cg = [poi1,poi2]
                if case=='c2cg_corr': c2g=-cg
            elif case=='c2kt':
                c2,kt = [poi1,poi2]
            elif case in ['klcg','klcg_corr']:
                kl,cg = [poi1,poi2]
                if case=='klcg_corr': c2g=-cg
            elif case=='c2c2g':
                c2,c2g = [poi1,poi2]
            elif case=='cgc2g':
                cg,c2g = [poi1,poi2]
            toys.append([kl,kt,c2,cg,c2g])
    return toys

def createScansets():
    toyset = []
    for case in ['klkt', 'c2cg', 'c2kt', 'klcg', 'c2c2g', 'cgc2g', 'klcg_corr', 'c2cg_corr']:
        toyset+=createScanset(case)
    return toyset

@functools.lru_cache(None)
def makeTestSet(seed=12345678, size=5000, samplesize=5000, c2glimited=False):
    random.seed(seed)
    toys = []
    for i in range(size):
        kl = random.randrange(-150,150,1)/10.
        kt = random.randrange(-50,50,1)/10.
        c2 = random.randrange(-50,50,1)/10.
        cg = random.randrange(-50,50,1)/10.
        c2g = random.randrange(-30,30,1)/10.
        if c2glimited:
            c2g = random.randrange(-10,10,1)/10.
        calcDist(kl, kt, c2, cg, c2g, samplesize=samplesize)
        toys.append([kl, kt, c2, cg, c2g])
    toys += createScansets()
    return toys

# Matrix inversion formula (NLO)
def func5D_LO(sample):
    """LO terms (15 terms)"""
    kl, kt, c2, cg, c2g = sample
    return [
        kl**2 * kt**2,
        2*kl**2 * kt * cg,
        kl**2 * cg**2,
        2*kl * kt**3,
        2*kl * kt**2 * cg,
        2*kl * kt * c2,
        2*kl * kt * c2g,
        2*kl * c2 * cg,
        2*kl * cg * c2g,
        kt**4,
        2*kt**2 * c2,
        2*kt**2 * c2g,
        c2**2,
        2*c2 * c2g,
        c2g**2,
    ]

def func5D(sample):
    """NLO terms (23 terms)"""
    kl, kt, c2, cg, c2g = sample
    return [
        kl**2 * kt**2,
        2*kl**2 * kt * cg,
        kl**2 * cg**2,
        2*kl * kt**3,
        2*kl * kt**2 * cg,
        2*kl * kt * c2,
        2*kl * kt * c2g,
        2*kl * c2 * cg,
        2*kl * cg * c2g,
        kt**4,
        2*kt**2 * c2,
        2*kt**2 * c2g,
        c2**2,
        2*c2 * c2g,
        c2g**2,
        kt**3 * cg,
        kt * c2 * cg,
        kt * cg**2 * kl,
        kt * cg * c2g,
        kt**2 * cg**2,
        c2 * cg**2,
        cg**3 * kl,
        cg**2 * c2g,
    ]

def model_5D(inputs, kl, kt, c2, cg, c2g, use_LO=False):
    func = func5D_LO if use_LO else func5D
    M = sympy.Matrix([
         func(sample)  for i, sample in enumerate(inputs)
        ])
    c = sympy.Matrix(func([kl, kt, c2, cg, c2g]))
    M_inv = M.pinv()
    coeffs = c.transpose() * M_inv
    return [float(co) for co in coeffs]

def makebase(ipt, start=[]):
    base = [ ]
    for j in range(int(len(ipt)/5)):
        s = str(j+1)
        base.append([ipt[c+s] for c in ["kl_","kt_", "c2_", "cg_", "c2g_"]])
    if start:  # Only add start offset if it's provided
        for j in range(len(base)):
            for i in range(len(base[j])):
                base[j][i] = base[j][i] + start[j][i]
    return base

# calc dist using matrix inverison model instead of toys by linear combination of input scenarios
def calcDistModel(kl, kt, c2, cg, c2g, inputs, samplesize=5000, start=[], use_LO=False):
    base = makebase(inputs, start=start)
    coeffs = model_5D(base, kl, kt, c2, cg, c2g, use_LO=use_LO)
    inputdists = []
    for i in base:
        kl_i, kt_i, c2_i, cg_i, c2g_i = i
        h_i = calcDist(kl_i, kt_i, c2_i, cg_i, c2g_i, samplesize)
        inputdists.append(h_i)
    key = makeKey(kl, kt, c2, cg, c2g, samplesize)
    h = inputdists[0].Clone(key)
    h.Reset()
    for ii in range(len(base)):
        h.Add(inputdists[ii], coeffs[ii])
    return h

def scoretoy(basis, toy, start=[], samplesize=5000):
    kl, kt, c2, cg, c2g = toy
    h1 = calcDist(kl, kt, c2, cg, c2g, samplesize=samplesize)
    h2 = calcDistModel(kl, kt, c2, cg, c2g, inputs=basis, start=start, samplesize=samplesize)
    n_bins = h1.GetNbinsX()

    CHI2 = sum([math.pow((h1.GetBinContent(i+1)-h2.GetBinContent(i+1)),2)/(math.pow(h1.GetBinError(i+1),2)+math.pow(h2.GetBinError(i+1),2)) for i in range(n_bins)])
    CHI2 = CHI2 / n_bins

    # Poisson NLL: -2lnL = 2[m - d + d*ln(d/m)]
    nll = 0.0
    bad_bins = 0
    for i in range(1, n_bins + 1):
        d = h1.GetBinContent(i)
        m = h2.GetBinContent(i)
        if m <= 1e-10:
            bad_bins += 1
            nll += 1e2
            continue
        if d > 0:
            nll += 2 * (m - d + d * math.log(d / m))
        else:
            nll += 2 * m
    NLL = (nll / n_bins) + 10 * bad_bins

    stat = sum([abs(h2.GetBinError(i+1)/(h2.GetBinContent(i+1)+0.0000001)) for i in range(n_bins)])
    statref = sum([abs(h1.GetBinError(i+1)/(h1.GetBinContent(i+1)+0.0000001)) for i in range(n_bins)])
    STAT = stat / statref

    return CHI2, NLL, STAT

#def scorefunc(ks, stat, ksstregth=0.5, statstrength=0.5):
#    return ksstregth*(math.log(ks)+1) + statstrength*(-np.power(2,min(stat-1,50)) + 1/(stat+0.0001))

def scorefunc(primary_metric, stat, primary_strength=0.5, statstrength=0.5):
    """Primary_metric is either chi2 or nll."""
    return 1 / (primary_strength * primary_metric + statstrength * math.pow(stat, 1))

def scorebasis(basis, toys, start=[], samplesize=5000, extra=False,
               chi2strength=None, nllstrength=None, statstrength=0.5):
    """
    Metric is selected from the config:
      - nllstrength set  → use Poisson NLL + STAT
      - chi2strength set  → use chi2 + STAT  (default if neither/both set)
    """
    if nllstrength is not None and chi2strength is None:
        use_nll = True
        primary_strength = nllstrength
    else:
        use_nll = False
        primary_strength = chi2strength if chi2strength is not None else 0.5

    avg_primary = 0.0
    avgstat = 0.0
    for it, t in enumerate(toys):
        if it % 100 == 0:
            print("evaluating basis:", hash(str(basis)), ":", it, "/", len(toys))
        chi2, nll, stat = scoretoy(basis, t, start, samplesize=samplesize)
        avg_primary += nll if use_nll else chi2
        avgstat += stat

    avg_primary /= len(toys)
    avgstat /= len(toys)
    score = scorefunc(avg_primary, avgstat, primary_strength, statstrength)
    if extra:
        return -1 * score, avg_primary, avgstat
    return -1 * score

def ensemble_score(
        parameter_dicts,
        settings=None,
        toys=[],
        start=[],
):
    # Detect metric from config: nllstrength → NLL+STAT, chi2strength → CHI2+STAT
    nllstrength = settings.get('nllstrength', None)
    chi2strength = settings.get('chi2strength', None)
    statstrength = settings.get('statstrength', 0.5)
    sb = functools.partial(
        scorebasis,
        toys=toys,
        start=start,
        samplesize=settings['samplesize'],
        chi2strength=chi2strength,
        nllstrength=nllstrength,
        statstrength=statstrength,
    )
    pool = Pool(processes=25)
    print(len(parameter_dicts))
    out = pool.map(sb, parameter_dicts)
    return out

def prepare_condor_submit(output_dir, settings):
    submit_file = os.path.join(output_dir, 'submit_iter.sub')
    wrapper_script = os.path.join(output_dir, 'run_job.sh')
    logs_dir = os.path.join(output_dir, 'logs')

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    run_script = "/afs/cern.ch/user/e/emartinv/public/HyperEvolution-mhh/hyperevol/examples/mhh_scoring.py"
    pso_file = settings['pso_file']
    setup_env = "/afs/cern.ch/user/e/emartinv/public/HyperEvolution-mhh/setup_environment.sh"

    with open(wrapper_script, 'wt') as fh:
        fh.write(f"""#!/bin/bash
# Kerberos token renewal if possible
if command -v kinit &> /dev/null && [ -n "${{KRB5CCNAME:-}}" ]; then
    kinit -R 2>/dev/null || true
fi

source {setup_env}
python3 {run_script} --parameter_file "$1" --pso_file {pso_file}
""")
    os.chmod(wrapper_script, 0o755)

    with open(submit_file, 'wt') as fh:
        fh.write(
f"""universe   = vanilla
executable = {wrapper_script}
arguments  = $(PARAMFILE)
output     = {logs_dir}/job_$(SAMPLE).log
error      = {logs_dir}/job_$(SAMPLE).log
log        = {logs_dir}/condor_$(SAMPLE).log
request_cpus = 2
request_memory = 2GB
+JobFlavour = "tomorrow"
"""
        )

        samples_dir = os.path.join(output_dir, 'samples')
        sample_list = []
        for sample_dir in sorted(glob.glob(os.path.join(samples_dir, '*')), key=lambda x: int(os.path.basename(x))):
            sample_nr = os.path.basename(sample_dir)
            paramfile = os.path.join(sample_dir, 'parameters.json')
            sample_list.append((sample_nr, paramfile))

        fh.write("queue SAMPLE, PARAMFILE from (\n")
        for sample_nr, paramfile in sample_list:
            fh.write(f"  {sample_nr}, {paramfile}\n")
        fh.write(")\n")

    return submit_file


def read_json_cfg(path):
    """ Reads the json info from a given path

    Parameters:
    ----------
    path : str
        Path to the .json file

    Returns:
    --------
    info : dict
        The json dict that was loaded
    """
    with open(path, 'rt') as jsonFile:
        info = json.load(jsonFile)
    return info

def read_fitness(output_dir, fitness_key="fitness"):
    """Creates the list of score dictionaries of each sample. List is ordered
    according to the number of the sample

    Parameters:
    ----------
    output_dir : str
        Path to the directory of output

    Returns:
    -------
    scores : list of floats
        List of fitnesses
    """
    samples = os.path.join(output_dir, 'samples')
    wild_card_path = os.path.join(samples, '*', 'score.json')
    number_samples = len(glob.glob(wild_card_path))
    score_dicts = []
    for number in range(number_samples):
        path = os.path.join(samples, str(number), 'score.json')
        score_dict = read_json_cfg(path)
        score_dicts.append(score_dict)
    scores = [score_dict[fitness_key] for score_dict in score_dicts]
    return scores

def check_error(output_dir):
    """In case of warnings or errors during job execution,
    raises SystemExit(0) to stop the optimization

    Parameters:
    ----------
    output_dir : str
        Path to the directory of the output, where the error file is located

    Returns:
    -------
    Nothing
    """

    number_errors = 0
    error_list = [
        'Aborted',
        'aborted',
        'Hold reason',
        'hold reason',
        'Shadow exception',
        'shadow exception',
        'failed',
        'Failed',
        'exit code',
        'Exit code',
        'terminated',
        'Terminated',
        'Cannot',
        'cannot',
        'No such file',
        'No such process'
    ]

    log_files = os.path.join(output_dir, 'logs', 'job_*.log')

    for log_file in glob.glob(log_files):
        if os.path.exists(log_file):
            with open(log_file, 'rt') as file:
                for line in file:
                    for error in error_list:
                        if error in line:
                            number_errors += 1
                            print(f"[ERROR] {log_file}: {line.strip()}")

    if number_errors > 0:
        print(f"Found {number_errors} errors. Stopping optimization.")
        raise SystemExit(0)

def renew_kerberos_token():
    """Attempt to renew Kerberos and AFS tokens

    Returns:
    --------
    success : bool
        True if renewal was successful or not needed, False otherwise
    """
    try:
        # Try to renew Kerberos ticket
        result = subprocess.run(
            ['kinit', '-R'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            subprocess.run(['aklog'], capture_output=True, timeout=5)
            print("✓ Kerberos/AFS tokens renewed successfully")
            return True
        else:
            print("⚠ Token renewal failed. You may need to run 'kinit' manually in another terminal.")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        return False

def wait_iteration(output_dir, sample_size, cluster_id=None):
    """Waits until all batch jobs are finised and in case of and warning
    or error that appears in the error file, stops running the optimization

    Parameters:
    ----------
    output_dir : str
        Path to the directory of output
    sample_size : int
        Number of particles (parameter-sets)
    cluster_id : str, optional
        HTCondor cluster ID for monitoring

    Returns:
    -------
    Nothing
    """
    wild_card_path = os.path.join(output_dir, 'samples', '*', 'score.json')
    status_file = os.path.join(output_dir, 'status.json')
    start_time = time.time()
    last_update = 0
    last_token_renewal = time.time()
    token_renewal_interval = 36000  # Renew tokens every 10 hours

    print(f"\nWaiting for {sample_size} jobs...")
    print(f"Use: python3 mhh_scoring.py --monitor={output_dir}")
    print(f"{'='*70}\n")

    while True:
        completed = len(glob.glob(wild_card_path))

        if completed == sample_size:
            elapsed = time.time() - start_time
            status = {
                'completed': completed,
                'total': sample_size,
                'elapsed': elapsed,
                'status': 'completed',
                'cluster_id': cluster_id,
                'iteration': find_iter_number(os.path.join(output_dir, 'previous_files'))
            }
            try:
                with open(status_file, 'w') as f:
                    json.dump(status, f, indent=2)
            except PermissionError:
                print("⚠ Warning: Unable to write status file (permission denied). Attempting token renewal...")
                if renew_kerberos_token():
                    try:
                        with open(status_file, 'w') as f:
                            json.dump(status, f, indent=2)
                    except PermissionError:
                        print("❌ Still unable to write final status file after token renewal.")
            print(f"✓ All jobs completed ({completed}/{sample_size}) in {elapsed:.1f}s")
            break

        current_time = time.time()

        # Renew tokens periodically
        if current_time - last_token_renewal >= token_renewal_interval:
            print(f"\nRenewing Kerberos/AFS tokens (after {token_renewal_interval/3600:.1f} hours)...")
            renew_kerberos_token()
            last_token_renewal = current_time

        if current_time - last_update >= 300:
            elapsed = current_time - start_time

            status = {
                'completed': completed,
                'total': sample_size,
                'elapsed': elapsed,
                'status': 'running',
                'cluster_id': cluster_id,
                'iteration': find_iter_number(os.path.join(output_dir, 'previous_files')),
                'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            if cluster_id:
                try:
                    result = subprocess.run(
                        ['condor_q', cluster_id, '-af', 'JobStatus'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        statuses = result.stdout.strip().split('\n')
                        status['running'] = sum(1 for s in statuses if s == '2')
                        status['idle'] = sum(1 for s in statuses if s == '1')
                        status['held'] = sum(1 for s in statuses if s == '5')
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

            try:
                with open(status_file, 'w') as f:
                    json.dump(status, f, indent=2)
            except PermissionError:
                print("⚠ Warning: Unable to write status file (permission denied). Attempting token renewal...")
                if renew_kerberos_token():
                    # Try writing again after renewal
                    try:
                        with open(status_file, 'w') as f:
                            json.dump(status, f, indent=2)
                    except PermissionError:
                        print("❌ Still unable to write status file after token renewal.")

            last_update = current_time

        check_error(output_dir)
        time.sleep(5)

def move_previous_files(output_dir, previous_files_dir):
    """Deletes the files from previous iteration

    Parameters:
    -------
    output_dir : str
        Path to the directory of the output

    Returns:
    -------
    Nothing
    """
    iter_nr = find_iter_number(previous_files_dir)
    samples_dir = os.path.join(output_dir, 'samples')
    iter_dir = os.path.join(previous_files_dir, 'iteration_' + str(iter_nr))
    shutil.copytree(samples_dir, iter_dir)
    shutil.rmtree(samples_dir)
    wild_card_path = os.path.join(output_dir, 'parameter_*.sh')
    for path in glob.glob(wild_card_path):
        os.remove(path)

def find_iter_number(previous_files_dir):
    """Finds the number iterations done

    Parameters:
    ----------
    previous_files_dir : str
        Path to the directory where old iterations are saved

    Returns:
    -------
    iter_number : int
        Number of the current iteration
    """
    wild_card_path = os.path.join(previous_files_dir, 'iteration_*')
    iter_number = len(glob.glob(wild_card_path))
    return iter_number

def cleanup_old_iterations(previous_files_dir, keep_last=50):
    """
    Example with keep_last=50:
      - At iteration 50  → clears contents of iterations 0-49
      - At iteration 100 → clears contents of iterations 50-99
      - etc.
    """
    current_iter = find_iter_number(previous_files_dir)
    if current_iter == 0 or current_iter % keep_last != 0:
        return
    batch_start = current_iter - keep_last
    batch_end = current_iter - 1
    cleaned = 0
    for i in range(batch_start, batch_end + 1):
        iter_dir = os.path.join(previous_files_dir, f'iteration_{i}')
        if os.path.exists(iter_dir) and os.listdir(iter_dir):
            shutil.rmtree(iter_dir)
            os.makedirs(iter_dir)  # keep empty dir to preserve counter
            cleaned += 1
    if cleaned > 0:
        print(f"Auto-cleanup: cleared contents of iterations {batch_start}-{batch_end} ({cleaned} dirs freed)")

def parameters_to_file(output_dir, hyperparameter_sets):
    """Saves the parameters to the subdirectory (name=sample number) of the
    output_dir into a parameters.json file

    Parameters:
    ----------
    output_dir : str
        Path to the output directory
    hyperparameter_sets : list dicts
        Parameter-sets of all particles

    Returns:
    -------
    Nothing
    """
    samples = os.path.join(output_dir, 'samples')
    if not os.path.exists(samples):
        os.makedirs(samples)
    for number, parameter_dict in enumerate(hyperparameter_sets):
        nr_sample = os.path.join(samples, str(number))
        if not os.path.exists(nr_sample):
            os.makedirs(nr_sample)
        parameter_file = os.path.join(nr_sample, 'parameters.json')
        with open(parameter_file, 'w') as file:
            json.dump(parameter_dict, file)

def ensemble_score_condor(parameter_dicts, settings):
    output_dir = os.path.expandvars(settings['output_dir'])
    previous_files_dir = os.path.join(output_dir, 'previous_files')

    if not os.path.exists(previous_files_dir):
        os.makedirs(previous_files_dir)

    # Clean old score.json files before starting new iteration from checkpoint
    samples_dir = os.path.join(output_dir, 'samples')
    if os.path.exists(samples_dir):
        for score_file in glob.glob(os.path.join(samples_dir, '*', 'score.json')):
            try:
                os.remove(score_file)
            except OSError:
                pass

    parameters_to_file(output_dir, parameter_dicts)
    submit_file = prepare_condor_submit(output_dir, settings)
    print(f"\nSending {len(parameter_dicts)} jobs to HTCondor...")
    result = subprocess.run(
        ['condor_submit', submit_file],
        capture_output=True,
        text=True
    )

    # Show output of condor_submit
    if result.stdout:
        print(result.stdout)

    # Check for errors
    if result.returncode != 0:
        print(f"\n❌ ERROR sending jobs:")
        if result.stderr:
            print(result.stderr)
        print(f"\nSubmit file: {submit_file}")
        print("Check the .sub file for possible errors.")
        raise RuntimeError("Failed to submit jobs to HTCondor")

    # Extract cluster ID from output
    cluster_id = None
    jobs_submitted = 0
    for line in result.stdout.split('\n'):
        if 'submitted to cluster' in line.lower():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    jobs_submitted = int(parts[0])
                except ValueError:
                    pass
            cluster_id = parts[-1].rstrip('.')
            break

    if cluster_id:
        print(f"✓ {jobs_submitted} jobs successfully submitted (Cluster ID: {cluster_id})")
    else:
        print(f"⚠ Jobs submitted but could not obtain Cluster ID")
        print(f"Check with: condor_q {os.getenv('USER', 'emartinv')}")

    # 4. Wait for all score.json files to appear with monitoring
    wait_iteration(output_dir, len(parameter_dicts), cluster_id)

    time.sleep(10)

    scores = read_fitness(output_dir)
    move_previous_files(output_dir, previous_files_dir)
    cleanup_old_iterations(previous_files_dir)

    return scores

def save_checkpoint(swarm, output_dir, iteration):
    """Save the state of the swarm to resume later"""
    checkpoint_file = os.path.join(output_dir, 'checkpoint.json')

    # Extract information from the particles
    particles_state = []
    for particle in swarm.swarm:
        particle_state = {
            'hyperparameters': particle.hyperparameters,
            'speed': particle.speed,
            'personal_best': particle.personal_best,
            'personal_best_fitness': particle.personal_best_fitness,
            'global_best': particle.global_best,
            'global_best_fitness': particle.global_best_fitness,
            'fitness': particle.fitness,
            'w': particle.w
        }
        particles_state.append(particle_state)

    checkpoint = {
        'iteration': iteration,
        'global_best': swarm.global_best,
        'global_bests': swarm.global_bests,
        'particles': particles_state,
        'settings': swarm.settings
    }

    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"Checkpoint saved at iteration {iteration}")

def load_checkpoint(output_dir):
    """Load the saved state of the swarm"""
    checkpoint_file = os.path.join(output_dir, 'checkpoint.json')

    if not os.path.exists(checkpoint_file):
        return None

    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)

    return checkpoint

def resume_optimization(output_dir, pso_cfg):
    """Resume an interrupted optimization"""
    print(f"\nAttempting to resume optimization from: {output_dir}")

    checkpoint = load_checkpoint(output_dir)

    if checkpoint is None:
        print("No checkpoint found. Starting optimization from scratch.")
        return main(output_dir, pso_cfg)

    print(f"Checkpoint found at iteration {checkpoint['iteration']}")
    print(f"Global best so far: {checkpoint['global_best']}")

    # Clean up iterations after checkpoint (in case of previous incomplete run)
    previous_files_dir = os.path.join(output_dir, 'previous_files')
    checkpoint_iter = checkpoint['iteration']
    for old_iter_dir in glob.glob(os.path.join(previous_files_dir, 'iteration_*')):
        iter_num = int(os.path.basename(old_iter_dir).split('_')[1])
        if iter_num > checkpoint_iter:
            print(f"Cleaning up incomplete iteration {iter_num}")
            shutil.rmtree(old_iter_dir)

    hyperparameters = read_cfg(pso_cfg["hpconfig"])
    toys = makeTestSet(size=2500, samplesize=pso_cfg['samplesize'], c2glimited=pso_cfg['c2glimited'])
    start = pso_cfg['basis']

    use_condor = pso_cfg.get('use_condor', True)

    if use_condor:
        print("Using HTCondor to execute jobs...")
        swarm = pso.ParticleSwarm(ensemble_score_condor, hyperparameters, pso_cfg)
    else:
        print("Using local execution with multiprocessing...")
        ensemble = functools.partial(ensemble_score, toys=toys, start=start, settings=pso_cfg)
        swarm = pso.ParticleSwarm(ensemble, hyperparameters, pso_cfg)

    # Restore state from checkpoint
    swarm.global_best = checkpoint['global_best']
    swarm.global_bests = checkpoint['global_bests']

    for i, particle in enumerate(swarm.swarm):
        particle_state = checkpoint['particles'][i]
        particle.hyperparameters = particle_state['hyperparameters']
        particle.speed = particle_state['speed']
        particle.personal_best = particle_state['personal_best']
        particle.personal_best_fitness = particle_state['personal_best_fitness']
        particle.global_best = particle_state['global_best']
        particle.global_best_fitness = particle_state['global_best_fitness']
        particle.fitness = particle_state['fitness']
        particle.w = particle_state['w']

    start_iteration = checkpoint['iteration'] + 1
    print(f"Resuming from iteration {start_iteration}/{swarm.iterations}\n")

    # Continue optimization with checkpoints
    iteration = start_iteration
    while iteration < swarm.iterations:
        print(f'{iteration}/{swarm.iterations}')
        all_locations = [particle.hyperparameters for particle in swarm.swarm]
        fitnesses = swarm.objective_function(all_locations, swarm.settings)
        swarm.set_particle_fitnesses(fitnesses)
        swarm.check_global_best()

        for particle in swarm.swarm:
            particle.next_iteration(swarm.swarm)

        # Save checkpoint after each iteration
        save_checkpoint(swarm, output_dir, iteration)

        iteration += 1

    # Save final checkpoint
    save_checkpoint(swarm, output_dir, swarm.iterations)

    best_fitness, best_location = swarm.find_best_hyperparameters()
    pso_best_fitness = -1 * best_fitness
    bestbasis = makebase(best_location, start=start)

    print(f"Found optimal parameters: {bestbasis}")
    print(f"Found optimal value with optimal parameters: {pso_best_fitness}")
    print("--------------------------------------------------------")
    print("Saving results:")
    save_results(bestbasis, pso_best_fitness, output_dir)

def main(output_dir: str, pso_cfg: dict) -> None:
    ''' Runs the particle swarm optimization to optimize the Rosenbrock function
    and saves the result to a file in the specified folder.
        Since no additional parameters need to be given to the Rosenbrock fn,
    then no additional 'settings=xyz' will be specified for PSO here.
        After optimization, the other logging info (e.g. score evolution) can
    be accessed easily by e.g "swarm.global_bests"

    Args:
        output_dir : str
            The directory where the output will be written

    Returns:
        None
    '''
    os.makedirs(output_dir, exist_ok=True)
    hyperparameters = read_cfg(pso_cfg["hpconfig"])
    toys=makeTestSet(size=2500, samplesize=pso_cfg['samplesize'], c2glimited=pso_cfg['c2glimited'])
    start=pso_cfg['basis']

    use_condor = pso_cfg.get('use_condor', False)

    if use_condor:
        print("Using HTCondor to execute jobs...")
        swarm = pso.ParticleSwarm(ensemble_score_condor, hyperparameters, pso_cfg)
    else:
        print("Using local execution...")
        ensemble = functools.partial(ensemble_score, toys=toys, start=start, settings=pso_cfg)
        swarm = pso.ParticleSwarm(ensemble, hyperparameters, pso_cfg)

    # Optimize with automatic checkpoints
    iteration = 0
    np.random.seed(swarm.seed)
    all_locations = [particle.hyperparameters for particle in swarm.swarm]
    fitnesses = swarm.objective_function(all_locations, swarm.settings)
    swarm.set_particle_fitnesses(fitnesses, initial=True)
    swarm.check_global_best()

    for particle in swarm.swarm:
        particle.next_iteration(swarm.swarm)

    save_checkpoint(swarm, output_dir, 0)
    iteration = 1

    while iteration < swarm.iterations:
        print(f'{iteration}/{swarm.iterations}')
        all_locations = [particle.hyperparameters for particle in swarm.swarm]
        fitnesses = swarm.objective_function(all_locations, swarm.settings)
        swarm.set_particle_fitnesses(fitnesses)
        swarm.check_global_best()

        for particle in swarm.swarm:
            particle.next_iteration(swarm.swarm)

        # Save checkpoint after each iteration
        save_checkpoint(swarm, output_dir, iteration)

        iteration += 1

    # Save final checkpoint
    save_checkpoint(swarm, output_dir, swarm.iterations)

    best_fitness, best_location = swarm.find_best_hyperparameters()
    pso_best_fitness = -1 * best_fitness
    bestbasis = makebase(best_location, start=start)

    print(f"Found optimal parameters: {bestbasis}")
    print(f"Found optimal value with optimal parameters: {pso_best_fitness}")
    print("--------------------------------------------------------")
    print("Saving results:")
    save_results(bestbasis, pso_best_fitness, output_dir)


def monitor_status(output_dir):
    """Monitor the status of running jobs"""
    status_file = os.path.join(output_dir, 'status.json')

    if not os.path.exists(status_file):
        print(f"No status file found in: {output_dir}")
        return

    with open(status_file, 'r') as f:
        status = json.load(f)

    print(f"\n{'='*70}")
    print(f"Optimization status in: {output_dir}")
    print(f"{'='*70}\n")

    print(f"Current iteration:  {status.get('iteration', 0)}")
    print(f"Completed jobs:    {status['completed']}/{status['total']}")

    if status['status'] == 'completed':
        print(f"Status:            ✓ COMPLETED")
        print(f"Total time:        {status['elapsed']:.1f}s")
    else:
        print(f"Status:            In progress")
        print(f"Elapsed time:      {status['elapsed']:.1f}s")
        print(f"Last update:       {status.get('last_update', 'N/A')}")

        if 'running' in status:
            print(f"\nRunning jobs:      {status['running']}")
            print(f"Idle jobs:         {status['idle']}")
            if status.get('held', 0) > 0:
                print(f"Held jobs:         {status['held']} ⚠️")

        if status.get('cluster_id'):
            print(f"\nCluster ID:        {status['cluster_id']}")

    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)

        # Monitor status
        if arguments['--monitor']:
            monitor_status(arguments['--monitor'])
            sys.exit(0)

        # Resume optimization
        if arguments['--resume']:
            resume_dir = arguments['--resume']

            if arguments['--pso_file']:
                pso_cfg = read_cfg(arguments['--pso_file'])
            else:
                # Try to read the config from the checkpoint
                checkpoint = load_checkpoint(resume_dir)
                if checkpoint and 'settings' in checkpoint and 'pso_file' in checkpoint['settings']:
                    pso_cfg = read_cfg(checkpoint['settings']['pso_file'])
                else:
                    # Last option: search in config directory
                    pso_file_guess = os.path.join(resume_dir, '..', 'config', 'pso_cfg_*.json')
                    pso_files = glob.glob(pso_file_guess)

                    if not pso_files:
                        print("Error: Could not find PSO configuration file.")
                        print("Use: --resume=DIR --pso_file=PATH")
                        sys.exit(1)

                    pso_cfg = read_cfg(pso_files[0])
                    print(f"Using config: {pso_files[0]}")

            pso_cfg['output_dir'] = resume_dir
            resume_optimization(resume_dir, pso_cfg)
            sys.exit(0)

        parameter_file = arguments['--parameter_file']
        pso_cfg = read_cfg(arguments['--pso_file'])
        output_dir=pso_cfg['output_dir']
        if "parameters" not in parameter_file:
            main(output_dir, pso_cfg)
        else:
            hyperparameters = read_json_cfg(parameter_file)
            toys=makeTestSet(size=pso_cfg['toysize'], samplesize=pso_cfg['samplesize'], c2glimited=pso_cfg['c2glimited'])
            start=pso_cfg['basis']
            nllstrength = pso_cfg.get('nllstrength', None)
            chi2strength = pso_cfg.get('chi2strength', None)
            statstrength = pso_cfg.get('statstrength', 0.5)
            score, avg_primary, avgstat = scorebasis(
                hyperparameters, toys, start,
                samplesize=pso_cfg['samplesize'],
                extra=True,
                chi2strength=chi2strength,
                nllstrength=nllstrength,
                statstrength=statstrength,
            )
            metric_label = 'avgnll' if (nllstrength is not None and chi2strength is None) else 'avgchi2'
            path = Path(parameter_file)
            save_dir = str(path.parent)
            basis = makebase(hyperparameters, start=start)
            score_path = os.path.join(save_dir, 'score.json')
            score_dict = {
                'fitness': score,
                'score': -1*(score),
                'basis': basis,
                metric_label: avg_primary,
                'avgstat': avgstat
            }
            with open(score_path, 'w') as score_file:
                json.dump(score_dict, score_file)
    except docopt.DocoptExit as e:
        print(e)