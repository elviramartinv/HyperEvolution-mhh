#!/usr/bin/env python3
"""
Script to perform mHH distribution morphing from basis points to any target point.
"""

import ROOT
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
plt.style.use(hep.style.CMS)
import os
import re
from array import array
import sympy
from scipy.optimize import linprog
from itertools import combinations
from scipy.spatial import Delaunay

# Basis points from ggHH_basispoint_13_13p6
basis_points_LHE = [
    [2.2132, -2.6934, 2.9657, 0.7573, 0.716],
    [9.7606, -4.0, 3.0808, 3.3198, -0.5983],
    [-8.504, 2.4571, 2.0457, 3.3289, -0.106],
    [12.4217, 4.8885, 1.1736, -1.0999, -0.9986],
    [-8.0281, 1.4186, 4.6373, 0.7337, 0.7138],
    [-0.6418, -3.4585, 1.5364, 0.9665, 1.0],
    [4.5168, 3.2463, 0.0318, 3.0105, 1.0],
    [6.777, 2.7536, 1.1292, -3.3324, 1.1768],
    [-7.2955, -3.5677, 4.9271, 3.6465, -0.182],
    [10.1679, 3.1681, 3.989, -2.6637, -0.8349],
    [-9.2377, -0.883, 1.5075, -1.1046, -0.3458],
    [15.8717, 2.9542, -1.4832, -3.0989, 0.1064],
    [-6.0921, -1.253, -0.6615, 2.0905, -0.8246],
    [-13.7652, -1.4549, 2.202, 4.0, -0.6921],
    [-13.9569, -3.8672, 2.1213, 2.0444, -0.6579],
    [8.288, -4.0, 9.6111, 2.3036, -0.3201],
    [-3.3685, -1.2352, -1.3215, 3.6573, 0.5436],
    [-1.1493, 0.0946, 3.0765, -2.425, -0.4578],
    [17.4, 1.8726, 0.2734, -1.712, 0.9525],
    [9.0192, 0.9359, -4.3684, 2.6212, -0.909],
    [10.0926, -0.256, 0.5458, 0.7075, -1.9215],
    [-7.5947, -3.5505, 0.968, -0.3777, -0.9394],
    [-2.1679, -0.4639, 3.0918, 1.3489, 0.4264],
]

BASIS_DIR = "/eos/user/e/emartinv/event_level_reweighting_HH/ggHH_basispoint_13_13p6"
TARGET_DIR = "/eos/user/e/emartinv/event_level_reweighting_HH/HEFT_13p6"
PLOT_DIR = "mhh_morphing_output"

REF_NBINS = 50
REF_XMIN = 200.0
REF_XMAX = 1500.0
REF_BIN_WIDTH = (REF_XMAX - REF_XMIN) / REF_NBINS
REF_BIN_EDGES = array('d', [REF_XMIN + i * REF_BIN_WIDTH for i in range(REF_NBINS + 1)])

# Matrix inversion formulas from mhh_scoring.py
def func5D(sample):
    """
    15 components (LO terms)
    """
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
        2*c2*c2g,
        c2g**2,
    ]

def func5D_nlo(sample):
    """
    23 components (LO + NLO terms)
    """
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
        # additional terms from NLO parametrization (calcXS in mhh_scoring.py)
        kt**3 * cg,
        kt * c2 * cg,
        kt * cg**2 * kl,
        kt * cg * c2g,
        kt**2 * cg**2,
        c2 * cg**2,
        cg**3 * kl,
        cg**2 * c2g,
    ]

def model_5D(inputs, target_point, use_nlo=False):
    kl, kt, c2, cg, c2g = target_point
    func = func5D_nlo if use_nlo else func5D
    M = sympy.Matrix([
         func(sample)  for i, sample in enumerate(inputs)
    ])
    c = sympy.Matrix(func([kl, kt, c2, cg, c2g]))
    M_inv = M.pinv()
    coeffs = c.transpose() * M_inv

    return [float(co) for co in coeffs]

def get_basis_histogram(basis_idx, energy=6800):
    kl, kt, c2, cg, c2g = basis_points_LHE[basis_idx]

    dirname = f"testrun_params_klambda_{kl}_ct_{kt}_ctt_{c2}_cggh_{cg}_cgghh_{c2g}_energy_{energy}_TeV"
    root_path = os.path.join(BASIS_DIR, dirname, "HH_variables", "output.root")

    if not os.path.exists(root_path):
        print(f"Warning: Basis file not found: {root_path}")
        return None

    f = ROOT.TFile.Open(root_path, "READ")
    if not f or f.IsZombie():
        print(f"Error: Could not open file {root_path}")
        return None

    tree = f.Get("vars")
    if not tree:
        print(f"Error: Could not find tree 'vars' in {root_path}")
        f.Close()
        return None

    h_temp = ROOT.TH1D("h_temp", "mHH", REF_NBINS, REF_BIN_EDGES)
    h_temp.Sumw2()
    tree.Draw("mHH>>h_temp", "", "goff")

    h = h_temp.Clone(f"h_basis_{basis_idx}")
    h.SetDirectory(0)

    f.Close()

    return h

def parse_target_filename(dirname):
    # Try pattern with gauss_params first (HEFT_13p6)
    pattern = r"testrun_gauss_params_klambda_([\-\d.eE]+)_ct_([\-\d.eE]+)_ctt_([\-\d.eE]+)_cggh_([\-\d.eE]+)_cgghh_([\-\d.eE]+)_energy_(\d+)_TeV"
    match = re.match(pattern, dirname)

    if not match:
        # Try pattern without gauss (ggHH_basispoint_13_13p6)
        pattern = r"testrun_params_klambda_([\-\d.eE]+)_ct_([\-\d.eE]+)_ctt_([\-\d.eE]+)_cggh_([\-\d.eE]+)_cgghh_([\-\d.eE]+)_energy_(\d+)_TeV"
        match = re.match(pattern, dirname)

    if match:
        return {
            'kl': float(match.group(1)),
            'kt': float(match.group(2)),
            'c2': float(match.group(3)),
            'cg': float(match.group(4)),
            'c2g': float(match.group(5)),
            'energy': int(match.group(6))
        }
    return None

def scan_target_points(basis_point_mode=False):
    search_dir = BASIS_DIR if basis_point_mode else TARGET_DIR

    if not os.path.exists(search_dir):
        print(f"Error: Directory not found: {search_dir}")
        return []

    targets = []
    for dirname in os.listdir(search_dir):
        full_path = os.path.join(search_dir, dirname)
        if not os.path.isdir(full_path):
            continue

        root_path = os.path.join(full_path, "HH_variables", "output.root")
        if not os.path.exists(root_path):
            continue

        params = parse_target_filename(dirname)
        if params:
            params['dirname'] = dirname
            params['root_path'] = root_path
            params['is_basis_point'] = basis_point_mode
            targets.append(params)

    return targets

def get_target_histogram(root_path, reference_hist=None):
    f = ROOT.TFile.Open(root_path, "READ")
    if not f or f.IsZombie():
        print(f"Error: Could not open file {root_path}")
        return None

    tree = f.Get("vars")
    if not tree:
        print(f"Error: Could not find tree 'vars' in {root_path}")
        f.Close()
        return None

    h = ROOT.TH1D("h_target", "mHH from target", REF_NBINS, REF_BIN_EDGES)
    h.Sumw2()
    tree.Draw("mHH>>h_target", "", "goff")

    h_clone = h.Clone("h_target_clone")
    h_clone.SetDirectory(0)

    f.Close()
    return h_clone

# def is_interpolation(target_point, basis_points=basis_points_LHE):
#     """
#     A point is interpolation if all its parameters are within the convex hull
#     of the basis points (simplified: within min/max range for each parameter).
#     """
#     for i in range(5):
#         param_values = [bp[i] for bp in basis_points_LHE]
#         min_val = min(param_values)
#         max_val = max(param_values)
#         if target_point[i] < min_val or target_point[i] > max_val:
#             return False
#     return True

def is_interpolation(target_point, basis_points=basis_points_LHE):
    """
    A point is interpolation if all its parameters are within the convex hull
    of the basis points (simplified: within min/max range for each parameter).
    """
    target_point = np.array(target_point, dtype=float)
    B = np.array(basis_points, dtype=float)
    N = B.shape[0]

    A_eq = []
    b_eq = []

    A_eq.append(np.ones(N))
    b_eq.append(1.0)

    for dim in range(5):
        A_eq.append(B[:, dim])
        b_eq.append(target_point[dim])

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    bounds = [(0, None)] * N
    c = np.zeros(N)
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not result.success:
        return False
    return True

def morph_distribution(target_point, energy=6800, debug=False, use_nlo=False):
    model_name = "NLO" if use_nlo else "LO"
    print(f"\nTarget: kl={target_point[0]:.4f}, kt={target_point[1]:.4f}, "
          f"c2={target_point[2]:.4f}, cg={target_point[3]:.4f}, c2g={target_point[4]:.4f}")
    print(f"Using {model_name} parametrization")

    weights = model_5D(basis_points_LHE, target_point, use_nlo=use_nlo)

    if debug:
        print(f"Weights (sum={sum(weights):.4f}):")
        for i, w in enumerate(weights):
            bp = basis_points_LHE[i]
            print(f"  Basis {i} (kl={bp[0]:.2f}, kt={bp[1]:.2f}, c2={bp[2]:.2f}, cg={bp[3]:.2f}, c2g={bp[4]:.2f}): {w:.4f}")

    # Load basis histograms and combine
    h_morphed = None

    for i, w in enumerate(weights):
        h_basis = get_basis_histogram(i, energy=energy)
        if h_basis is None:
            print(f"Warning: Could not load basis histogram {i}, skipping...")
            continue

        if h_morphed is None:
            h_morphed = h_basis.Clone("h_morphed")
            h_morphed.Reset()
            h_morphed.SetDirectory(0)

        h_morphed.Add(h_basis, w)

    if h_morphed and h_morphed.Integral() > 0:
        for i in range(1, h_morphed.GetNbinsX() + 1):
            if h_morphed.GetBinContent(i) < 0:
                print(f"Warning: Negative bin content at bin {i}, setting to zero")
                h_morphed.SetBinContent(i, 0)
                h_morphed.SetBinError(i, 0)

    return h_morphed

def compare_morphed_to_target(target_info, save_plots=True, debug=False, use_nlo=False, replace=False, plot_phase_space=False):
    kl = target_info['kl']
    kt = target_info['kt']
    c2 = target_info['c2']
    cg = target_info['cg']
    c2g = target_info['c2g']
    energy = target_info['energy']
    is_basis_point = target_info.get('is_basis_point', False)

    # Check if output already exists
    if save_plots and not replace:
        # Determine output directory structure
        if is_basis_point:
            category_subdir = "basis_points"
        else:
            is_int = is_interpolation((kl, kt, c2, cg, c2g))
            category_subdir = "interpolation" if is_int else "extrapolation"

        model_subdir = "nlo" if use_nlo else "lo"
        param_folder = f"kl_{kl:.3f}_kt_{kt:.3f}_c2_{c2:.3f}_cg_{cg:.3f}_c2g_{c2g:.3f}"
        output_dir = os.path.join(PLOT_DIR, model_subdir, category_subdir, param_folder)
        plotname = os.path.join(output_dir, "output.png")

        if os.path.exists(plotname):
            print(f"\n{'='*80}")
            print(f"Output already exists (skipping): {plotname}")
            print(f"Use 'replace' or 'r' flag to overwrite existing outputs")
            print(f"{'='*80}")
            return None

    print(f"\n{'='*80}")
    if is_basis_point:
        print(f"Basis point:")
    else:
        print(f"Target point:")
    print(f"  kl={kl:.4f}, kt={kt:.4f}, c2={c2:.4f}, cg={cg:.4f}, c2g={c2g:.4f}")
    print(f"  Energy: {energy} GeV")
    if not is_basis_point:
        print(f"  Is interpolation: {'Yes' if is_interpolation((kl, kt, c2, cg, c2g)) else 'No'}")
    print(f"{'='*80}")

    h_morphed = morph_distribution((kl, kt, c2, cg, c2g), energy=energy, debug=debug, use_nlo=use_nlo)
    if h_morphed is None or h_morphed.Integral() == 0:
        print("Error: Could not create morphed distribution")
        return None

    # LHE target distribution
    print(f"\nLHE target distribution from: {target_info['root_path']}")
    h_target = get_target_histogram(target_info['root_path'], reference_hist=h_morphed)
    if h_target is None or h_target.Integral() == 0:
        print("Error: Could not load target distribution")
        return None

    # Normalize both to same integral
    target_integral = h_target.Integral()
    morphed_integral = h_morphed.Integral()

    if morphed_integral > 0:
        scale_factor = target_integral / morphed_integral
        h_morphed.Scale(scale_factor)
    else:
        print(f"Warning: morphed integral <= 0 ({morphed_integral:.3f}), skipping normalization")
        # print(f"\nScaled morphed histogram by {scale_factor:.4f} to match target normalization")

    ks = h_morphed.KolmogorovTest(h_target, "")

    chi2 = 0
    ndof = 0
    for i in range(h_morphed.GetNbinsX()):
        val_morphed = h_morphed.GetBinContent(i+1)
        val_target = h_target.GetBinContent(i+1)
        err_morphed = h_morphed.GetBinError(i+1)
        err_target = h_target.GetBinError(i+1)

        if err_morphed**2 + err_target**2 > 0:
            chi2 += (val_morphed - val_target)**2 / (err_morphed**2 + err_target**2)
            ndof += 1
    print(f"\n{'='*80}")
    print(f"\nStatistics:")
    print(f"  Morphed - Mean: {h_morphed.GetMean():.2f}, RMS: {h_morphed.GetRMS():.2f}, Integral: {h_morphed.Integral():.2f}")
    print(f"  Target  - Mean: {h_target.GetMean():.2f}, RMS: {h_target.GetRMS():.2f}, Integral: {h_target.Integral():.2f}")
    print(f"  KS test p-value: {ks:.4f}")
    print(f"  Chi2/ndof: {chi2:.2f}/{ndof} = {chi2/ndof if ndof > 0 else 0:.2f}")
    print(f"\n{'='*80}")

    if save_plots:
        os.makedirs(PLOT_DIR, exist_ok=True)

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[4, 1])

        hep.histplot(h_morphed, histtype='errorbar', label='From mHH basis',
                     linewidth=2, color='blue', ax=ax)
        hep.histplot(h_target, histtype='errorbar', label='From LHE',
                     linewidth=2, color='red', ax=ax)

        ax.legend(loc='upper right', fontsize=12)
        ax.set_ylabel("Events", fontsize=16)
        ax.set_xlim(200, 1500)

        text_y = 1.02 * ax.get_ylim()[1]
        coupling_text = (r"$\kappa_\lambda$={:.3f}, $\kappa_t$={:.3f}, $c_2$={:.3f}, "
                        r"$c_g$={:.3f}, $c_{{2g}}$={:.3f}, KS={:.3f}".format(kl, kt, c2, cg, c2g, ks))
        ax.text(200, text_y, coupling_text, fontsize=12, verticalalignment='bottom')

        hep.cms.lumitext(r'138 fb$^{-1}$ (13 TeV)', ax=ax)

        ratio = h_target.Clone("ratio")
        ratio.Divide(h_morphed)
        hep.histplot(ratio, histtype='errorbar', linewidth=2, color='red', ax=ax2)
        ax2.axhline(1, color='blue', linewidth=2, linestyle='--')
        ax2.set_xlabel(r"$m_{HH}$ (GeV)", fontsize=12)
        ax2.set_ylabel("Ratio", fontsize=16)
        ax2.set_ylim(0.5, 1.5)
        ax2.set_xlim(200, 1500)

        ax.set_xticklabels([])
        plt.subplots_adjust(wspace=0, hspace=0.05)

        if is_basis_point:
            category_subdir = "basis_points"
        else:
            is_int = is_interpolation((kl, kt, c2, cg, c2g))
            category_subdir = "interpolation" if is_int else "extrapolation"

        model_subdir = "nlo" if use_nlo else "lo"

        param_folder = f"kl_{kl:.3f}_kt_{kt:.3f}_c2_{c2:.3f}_cg_{cg:.3f}_c2g_{c2g:.3f}"

        output_dir = os.path.join(PLOT_DIR, model_subdir, category_subdir, param_folder)
        os.makedirs(output_dir, exist_ok=True)

        plotname = os.path.join(output_dir, "output.png")
        rootname = os.path.join(output_dir, "output.root")

        # Save outputs (check was already done at function start)
        plt.savefig(plotname, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {plotname}")
        plt.close()

        # Save ROOT file with both histograms inside mHH directory
        f_out = ROOT.TFile.Open(rootname, "RECREATE")
        f_out.mkdir("mHH")
        f_out.cd("mHH")
        h_morphed_save = h_morphed.Clone("h_morphed")
        h_target_save = h_target.Clone("h_LHE")
        h_morphed_save.Write()
        h_target_save.Write()
        f_out.Close()
        print(f"ROOT file saved as: {rootname}")

        # Generate phase space plots if requested
        if plot_phase_space:
            plot_phase_space_2D(basis_points_LHE, (kl, kt, c2, cg, c2g), output_dir)

    results = {
        'kl': kl, 'kt': kt, 'c2': c2, 'cg': cg, 'c2g': c2g,
        'energy': energy,
        'ks': ks,
        'chi2': chi2,
        'ndof': ndof,
        'chi2_ndof': chi2/ndof if ndof > 0 else 0
    }

    return results

def compare_all_targets(save_plots=True, debug=False, max_targets=None, filter_mode=None, use_nlo=False, replace=False, param_ranges=None, plot_phase_space=False, basis_point_mode=False):
    targets = scan_target_points(basis_point_mode=basis_point_mode)
    search_dir = BASIS_DIR if basis_point_mode else TARGET_DIR
    point_type = "basis points" if basis_point_mode else "target points"
    print(f"\nFound {len(targets)} {point_type} in {search_dir}")

    # Filter by parameter ranges
    if param_ranges:
        filtered_targets = []
        for t in targets:
            passes = True
            for param, (pmin, pmax) in param_ranges.items():
                if not (pmin <= t[param] <= pmax):
                    passes = False
                    break
            if passes:
                filtered_targets.append(t)
        targets = filtered_targets
        print(f"Filtered by parameter ranges to {len(targets)} points")

    # Filter by interpolation/extrapolation (not applicable for basis points)
    if filter_mode and not basis_point_mode:
        filtered_targets = []
        for t in targets:
            is_int = is_interpolation((t['kl'], t['kt'], t['c2'], t['cg'], t['c2g']))
            if (filter_mode == 'int' and is_int) or (filter_mode == 'ext' and not is_int):
                filtered_targets.append(t)
        targets = filtered_targets
        mode_name = "interpolation" if filter_mode == 'int' else "extrapolation"
        print(f"Filtered to {len(targets)} {mode_name} points")
    elif filter_mode and basis_point_mode:
        print("Warning: int/ext filter ignored in basis point mode")

    if max_targets:
        targets = targets[:max_targets]
        print(f"Limiting to first {max_targets} targets")
    else:
        targets = targets[:25]
        print(f"Limiting to first 25 targets (default)")

    results = []

    for i, target_info in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] Processing target...")
        result = compare_morphed_to_target(target_info, save_plots=save_plots, debug=debug, use_nlo=use_nlo, replace=replace, plot_phase_space=plot_phase_space)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nSuccessfully processed {len(results)} target points")

    if len(results) > 0:
        avg_ks = np.mean([r['ks'] for r in results])
        avg_chi2_ndof = np.mean([r['chi2_ndof'] for r in results])
        print(f"\nAverage KS test p-value: {avg_ks:.4f}")
        print(f"Average Chi2/ndof: {avg_chi2_ndof:.2f}")

        # Show best and worst matches
        results_sorted = sorted(results, key=lambda x: x['ks'], reverse=True)
        print(f"\nBest match (highest KS p-value):")
        best = results_sorted[0]
        print(f"  kl={best['kl']:.2f}, kt={best['kt']:.2f}, c2={best['c2']:.2f}, "
              f"cg={best['cg']:.2f}, c2g={best['c2g']:.2f}")
        print(f"  KS p-value: {best['ks']:.4f}, Chi2/ndof: {best['chi2_ndof']:.2f}")

        print(f"\nWorst match (lowest KS p-value):")
        worst = results_sorted[-1]
        print(f"  kl={worst['kl']:.2f}, kt={worst['kt']:.2f}, c2={worst['c2']:.2f}, "
              f"cg={worst['cg']:.2f}, c2g={worst['c2g']:.2f}")
        print(f"  KS p-value: {worst['ks']:.4f}, Chi2/ndof: {worst['chi2_ndof']:.2f}")

    return results

def plot_phase_space_2D(basis_points, target_point, output_dir):
    """
    Create 2D phase space plots showing basis points and target point.
    """
    param_names = [
        r"$\kappa_\lambda$",
        r"$\kappa_t$",
        r"$c_2$",
        r"$c_g$",
        r"$c_{2g}$"
    ]
    param_plain = ["kl", "kt", "c2", "cg", "c2g"]

    basis = np.array(basis_points)
    target = np.array(target_point)

    phase_dir = os.path.join(output_dir, "phase_space")
    os.makedirs(phase_dir, exist_ok=True)

    for i, j in combinations(range(5), 2):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot basis points
        ax.scatter(basis[:, i], basis[:, j],
                  c='royalblue', s=80, edgecolor='black',
                  linewidth=1.2, alpha=0.85, label='Basis points', zorder=2)

        # Plot target point
        ax.scatter(target[i], target[j],
                  c='crimson', s=150, edgecolor='black',
                  linewidth=1.5, marker='*', label='Target point', zorder=3)

        # Determine if inside convex hull for this 2D projection
        try:
            hull_2d = Delaunay(basis[:, [i, j]])
            inside_2d = hull_2d.find_simplex(target[[i, j]]) >= 0
            geom_text = "Interpolation" if inside_2d else "Extrapolation"
        except Exception:
            geom_text = ""

        ax.set_xlabel(param_names[i], fontsize=14)
        ax.set_ylabel(param_names[j], fontsize=14)
        ax.set_title(f"{param_names[i]} vs {param_names[j]}", fontsize=15, pad=15)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')

        if geom_text:
            ax.text(0.02, 0.98, geom_text, transform=ax.transAxes,
                   fontsize=10, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        fname = os.path.join(phase_dir, f"phase_space_{param_plain[i]}_{param_plain[j]}.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Phase space plots saved in: {phase_dir}")

if __name__ == "__main__":
    import sys

    debug = "debug" in sys.argv or "d" in sys.argv
    sys.argv = [arg for arg in sys.argv if arg not in ["debug", "d"]]

    replace = "replace" in sys.argv or "r" in sys.argv
    sys.argv = [arg for arg in sys.argv if arg not in ["replace", "r"]]

    test_mode = "test" in sys.argv or "t" in sys.argv
    sys.argv = [arg for arg in sys.argv if arg not in ["test", "t"]]

    use_nlo = "NLO" in sys.argv or "nlo" in sys.argv
    sys.argv = [arg for arg in sys.argv if arg not in ["NLO", "nlo"]]

    plot_phase_space = "phase-space" in sys.argv or "ps" in sys.argv
    sys.argv = [arg for arg in sys.argv if arg not in ["phase-space", "ps"]]

    basis_point_mode = "basis-point" in sys.argv or "bp" in sys.argv
    sys.argv = [arg for arg in sys.argv if arg not in ["basis-point", "bp"]]

    filter_mode = None
    if "int" in sys.argv:
        filter_mode = 'int'
        sys.argv = [arg for arg in sys.argv if arg != "int"]
    elif "ext" in sys.argv:
        filter_mode = 'ext'
        sys.argv = [arg for arg in sys.argv if arg != "ext"]

    param_ranges = {}
    param_names = ['kl', 'kt', 'c2', 'cg', 'c2g']
    for param in param_names:
        range_arg = f"{param}-rg"
        matching_args = [arg for arg in sys.argv if arg.startswith(range_arg + "=")]
        if matching_args:
            arg = matching_args[0]
            try:
                range_str = arg.split('=')[1]
                pmin, pmax = map(float, range_str.split(','))
                param_ranges[param] = (pmin, pmax)
                print(f"Parameter range for {param}: [{pmin}, {pmax}]")
                sys.argv.remove(arg)
            except (ValueError, IndexError):
                print(f"Warning: Invalid format for {range_arg}, expected: {range_arg}=min,max")
                sys.argv.remove(arg)

    if len(sys.argv) > 1:
        # Process specific target by index
        target_idx = int(sys.argv[1])
        targets = scan_target_points(basis_point_mode=basis_point_mode)
        if target_idx >= 0 and target_idx < len(targets):
            compare_morphed_to_target(targets[target_idx], save_plots=True, debug=debug, use_nlo=use_nlo, replace=replace, plot_phase_space=plot_phase_space)
        else:
            print(f"Error: Target index {target_idx} out of range (0-{len(targets)-1})")
    else:
        # Process all targets
        max_targets = 5 if test_mode else None
        compare_all_targets(save_plots=True, debug=debug, max_targets=max_targets, filter_mode=filter_mode, use_nlo=use_nlo, replace=replace, param_ranges=param_ranges if param_ranges else None, plot_phase_space=plot_phase_space, basis_point_mode=basis_point_mode)
