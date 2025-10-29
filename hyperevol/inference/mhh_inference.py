"""
Script to infer the mHH distribution for any (kl, kt, c2, cg, c2g) point
using the mHH distributions of the basis points and matrix inversion.

- Loads mHH histograms from output.root files for all basis points
- Uses the model_5D and matrix inversion from mhh_scoring.py
- Combines basis histograms to infer the target distribution

Usage:
    python mhh_inference.py --kl 1.0 --kt 1.0 --c2 0.0 --cg 0.0 --c2g 0.0

"""
import numpy as np
import os
import argparse
import re
from pathlib import Path
from hyperevol.examples.mhh_scoring import model_5D
import json
import uproot
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from numpy.linalg import norm
import matplotlib as mpl
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

def plot_phase_space(basis, target, output_folder):
    param_names = [
        r"\kappa_\lambda",  # kl
        r"\kappa_t",      # kt
        r"c_2",            # c2
        r"c_g",            # cg
        r"c_{2g}"           # c2g
    ]

    mpl.rcParams['axes.facecolor'] = '#f7f7f7'
    mpl.rcParams['grid.color'] = '#cccccc'
    mpl.rcParams['grid.linestyle'] = '--'
    mpl.rcParams['axes.edgecolor'] = '#333333'
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    phase_dir = output_folder / "phase_space_points"
    phase_dir.mkdir(parents=True, exist_ok=True)
    # 2D plots
    for i, j in combinations(range(5), 2):
        plt.figure(figsize=(9,6))
        plt.scatter(
            basis[:,i], basis[:,j],
            c='royalblue', s=80, edgecolor='black', linewidth=1.2, alpha=0.85, label='Basis'
        )
        legend_label = None
        if target is not None:
            # Show parameter values in legend
            param_str = (
                r"$\kappa_\lambda = {:.2f}, \kappa_t = {:.2f}, c_2 = {:.2f}, c_g = {:.2f}, c_{{2g}} = {:.2f}$"
                .format(*target)
            )
            plt.scatter(
                target[i], target[j],
                c='crimson', s=120, edgecolor='black', linewidth=1.5, marker='*', label="Target"
            )
            plt.legend(fontsize=12)
        try:
            hull_2d = Delaunay(basis[:,[i,j]])
            inside_2d = hull_2d.find_simplex(target[[i,j]]) >= 0
            geom_text_2d = "Interpolation" if inside_2d else "Extrapolation"
        except Exception as e:
            geom_text_2d = "Hull error"
        plt.xlabel(f"${param_names[i]}$", fontsize=15)
        plt.ylabel(f"${param_names[j]}$", fontsize=15)
        plt.title(f"${param_names[i]}$ vs ${param_names[j]}$", fontsize=17, pad=15)
        plt.grid(True, which='major', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.gcf().text(0.02,0.98,param_str, fontsize=10, va='top', ha='left')
        plt.gcf().text(0.98,0.98,geom_text_2d, fontsize=10, va='top', ha='right',bbox=dict(facecolor='white', alpha=0.7))
        plt.gca().set_axisbelow(True)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.2)
        plt.gca().spines['bottom'].set_linewidth(1.2)
        param_plain = ["kl", "kt", "c2", "cg", "c2g"]
        fname_png = phase_dir / f"phase_space_{param_plain[i]}_{param_plain[j]}.png"
        fname_pdf = phase_dir / f"phase_space_{param_plain[i]}_{param_plain[j]}.pdf"
        plt.savefig(str(fname_png), dpi=150)
        plt.savefig(str(fname_pdf))
        plt.close()
    # 3D plots
    for i, j, k in combinations(range(5), 3):
        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            basis[:,i], basis[:,j], basis[:,k],
            c='royalblue', s=60, edgecolor='black', linewidth=1.0, alpha=0.85, label='Basis'
        )
        if target is not None:
            param_str = (
                r"$\kappa_\lambda = {:.2f}, \kappa_t = {:.2f}, c_2 = {:.2f}, c_g = {:.2f}, c_{{2g}} = {:.2f}$"
                .format(*target)
            )
            ax.scatter(
                target[i], target[j], target[k],
                c='crimson', s=120, edgecolor='black', linewidth=1.5, marker='*', label="Target"
            )
            ax.legend(fontsize=12)
        try:
            hull_3d = Delaunay(basis[:,[i,j,k]])
            inside_3d = hull_3d.find_simplex(target[[i,j,k]]) >= 0
            geom_text_3d = "Interpolation" if inside_3d else "Extrapolation"
        except Exception as e:
            geom_text_3d = "Hull error"
        ax.set_xlabel(f"${param_names[i]}$", fontsize=13)
        ax.set_ylabel(f"${param_names[j]}$", fontsize=13)
        ax.set_zlabel(f"${param_names[k]}$", fontsize=13)
        ax.set_title(f"${param_names[i]}$ vs ${param_names[j]}$ vs ${param_names[k]}$", fontsize=15, pad=15)
        fname_png = phase_dir / f"phase_space_{param_plain[i]}_{param_plain[j]}_{param_plain[k]}.png"
        fname_pdf = phase_dir / f"phase_space_{param_plain[i]}_{param_plain[j]}_{param_plain[k]}.pdf"
        plt.gcf().text(0.02,0.98,param_str, fontsize=10, va='top', ha='left')
        plt.gcf().text(0.98,0.98,geom_text_3d, fontsize=10, va='top', ha='right',bbox=dict(facecolor='white', alpha=0.7))
        plt.savefig(str(fname_png), dpi=150)
        plt.savefig(str(fname_pdf))
        plt.close()

def diagnose_target_integration(basis_points, target_point, coeffs, tol=1e-8):

    print("\n--- Target integration diagnostics ---")
    # Asegura que coeffs es un array de numpy
    coeffs = np.array(coeffs)
    # Geometric (Delaunay)
    try:
        hull = Delaunay(basis_points)
        inside = hull.find_simplex(target_point) >= 0
        if inside:
            print("Convex hull check: Target is INSIDE the convex hull of the basis points (interpolation).")
        else:
            print("Convex hull check: Target is OUTSIDE the convex hull of the basis points (extrapolation).")
    except Exception as e:
        print(f"[Geométrico] No se pudo calcular el convex hull: {e}")
    # Algebraic (coefficients)
    min_coeff = np.min(coeffs)
    max_coeff = np.max(coeffs)
    sum_coeff = np.sum(coeffs)
    all_positive = np.all(coeffs >= -tol)
    print(f"Coeffs: Min: {min_coeff:.3g}, Max: {max_coeff:.3g}, Sum: {sum_coeff:.5f}")
    if all_positive and abs(sum_coeff - 1) < 1e-1:
        print("Coeffs: All coefficients are positive and sum to 1: INTERPOLATION.")
    else:
        print("Coeffs: There are negative coefficients or the sum differs from 1: EXTRAPOLATION.")
    print("---\n")

# Path to all basis points' HH_variables folders
BASIS_DIR = Path("/eos/user/e/emartinv/event_level_reweighting_HH/ggHH_basispoint_13_13p6/")
pattern = re.compile(r"klambda_([-\d\.]+)_ct_([-\d\.]+)_ctt_([-\d\.]+)_cggh_([-\d\.]+)_cgghh_([-\d\.]+)")

# Find all basis points
sample_dirs = [p for p in BASIS_DIR.glob("*params_klambda_*") if p.is_dir()]

basis_params = []
basis_names = []

for sample_dir in sample_dirs:
    hh_dir = sample_dir / "HH_variables"
    root_path = hh_dir / "output.root"
    if not root_path.exists():
        continue
    try:
        with uproot.open(root_path) as f:
            pass  # Just to check if file can be opened
    except Exception as e:
        print(f"Error reading {root_path}: {e}")
        continue
    
    param_dir = sample_dir / "Parameters"
    param_dir.mkdir(exist_ok=True)
    param_path = param_dir / "parameters.json"

    if not param_path.exists():
        match = pattern.search(sample_dir.name)
        if not match:
            print(f"No match for {sample_dir.name}")
            continue
        kl, kt, c2, cg, c2g = map(float, match.groups())
        params = {
            "kl": kl,
            "kt": kt,
            "c2": c2,
            "cg": cg,
            "c2g": c2g
        }
        with open(param_path, "w") as f:
            json.dump(params, f, indent=2)
    # print(f"parameters.json created in {param_path}")
    else:
        with open(param_path) as f:
            params = json.load(f)
            # print(f"Loaded parameters from {param_path}")
    basis_params.append([
        params["kl"],
        params["kt"],
        params["c2"],
        params["cg"],
        params["c2g"]
    ])
    # No need to append histograms here
    basis_names.append(str(sample_dir))

basis_params = np.array(basis_params)

parser = argparse.ArgumentParser()
parser.add_argument("--kl", type=float, required=True)
parser.add_argument("--kt", type=float, required=True)
parser.add_argument("--c2", type=float, required=True)
parser.add_argument("--cg", type=float, required=True)
parser.add_argument("--c2g", type=float, required=True)
parser.add_argument("--energy", type=int, choices=[6500, 6800], default=6800, help="Energy [GeV] (6500 or 6800)")
parser.add_argument("--compare", action="store_true", help="Compare with true distribution if available")
# parser.add_argument("--variable", type=str, default="mHH", choices=["mHH", "pT_H1", "pT_H2", "pT_HH", "cosTheta"], help="Variable to compare/plot")
args = parser.parse_args()

# Get interpolation coefficients for target point

# Filter folders by energy
energy_str = f"energy_{args.energy}_TeV"
basis = np.array([
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
])

filtered_params = []
filtered_names = []
for folder, params in zip(basis_names, basis_params):
    for ref in basis:
        if np.allclose(params, ref, atol=1e-4):
            if energy_str in folder:
                filtered_params.append(params)
                filtered_names.append(folder)
                break

filtered_params = np.array(filtered_params)

if len(filtered_params) < 15:
    raise RuntimeError("Not enough basis points for interpolation")
# print(f"Selected basis for energy {args.energy} GeV:")
# for i, (name, params) in enumerate(zip(filtered_names, filtered_params)):
#     # print(f"Base number: {i} -> params: {params}")
# print("-----")
# print(f"Total selected basis: {len(filtered_params)}")

# Get interpolation coefficients for target point
coeffs = model_5D(filtered_params, args.kl, args.kt, args.c2, args.cg, args.c2g)
# print(coeffs)
# print(f"Coefficients sum: {np.sum(coeffs)}")



def get_output_folder(kl, kt, c2, cg, c2g, energy):
    return Path(f"output/kl_{kl}_kt_{kt}_c2_{c2}_cg_{cg}_c2g_{c2g}_energy_{energy}")

VARIABLES = {
    "mHH": {
        "outname": "mhh_inferred",
        "histname": "mHH_hist",
        "x_range": (250, 3000),
        "nbins": 100,
        "xlabel": "mHH [GeV]"
    },
    "pT_H1": {
        "outname": "pT_H1_inferred",
        "histname": "pT_H1_hist",
        "x_range": (0, 2000),
        "nbins": 100,
        "xlabel": "pT(H1) [GeV]"
    },
    "pT_H2": {
        "outname": "pT_H2_inferred",
        "histname": "pT_H2_hist",
        "x_range": (0, 2000),
        "nbins": 100,
        "xlabel": "pT(H2) [GeV]"
    },
    "pT_HH": {
        "outname": "pT_HH_inferred",
        "histname": "pT_HH_hist",
        "x_range": (0, 1000),
        "nbins": 100,
        "xlabel": "pT(HH) [GeV]"
    },
    "cosTheta": {
        "outname": "cosTheta_inferred",
        "histname": "cosTheta_hist",
        "x_range": (-1, 1),
        "nbins": 50,
        "xlabel": "cos(Theta)"
    },
}




output_folder = get_output_folder(args.kl, args.kt, args.c2, args.cg, args.c2g, args.energy)
output_folder.mkdir(parents=True, exist_ok=True)

# Plot phase space before inference
target_point = np.array([args.kl, args.kt, args.c2, args.cg, args.c2g])
plot_phase_space(basis, target_point, output_folder)

diagnose_target_integration(basis, target_point, coeffs)

for var, vinfo in VARIABLES.items():
    outname = vinfo["outname"]
    histname = vinfo["histname"]
    x_range = vinfo["x_range"]
    nbins = vinfo["nbins"]
    basis_var_histos = []
    bin_edges_var = None
    for folder in filtered_names:
        sample_dir = Path(folder)
        hh_dir = sample_dir / "HH_variables"
        root_path = hh_dir / "output.root"
        if not root_path.exists():
            continue
        try:
            with uproot.open(root_path) as f:
                if histname in f:
                    h = f[histname]
                    hist = h.values()
                    bin_edges_var = h.edges
                else:
                    if "vars" in f and var in f["vars"].keys():
                        arr = f["vars"][var].array(library="np")
                        bin_edges_var = np.linspace(x_range[0], x_range[1], nbins+1)
                        hist, _ = np.histogram(arr, bins=bin_edges_var)
                    else:
                        continue
            basis_var_histos.append(hist)
        except Exception as e:
            continue
    basis_var_histos = np.array(basis_var_histos)
    inferred = np.sum(basis_var_histos.T * coeffs, axis=1)
    root_out_var = output_folder / f"{var}_inferred.root"
    with uproot.recreate(str(root_out_var)) as fout:
        fout[outname] = (inferred, bin_edges_var)
    # print(f"Saved inferred {var} distribution to {root_out_var}")


if args.compare:

    # folder_name = f"testrun_gauss_params_klambda_{args.kl}_ct_{args.kt}_ctt_{args.c2}_cggh_{args.cg}_cgghh_{args.c2g}_energy_{args.energy}_TeV"
    # true_base = Path(f"/eos/user/e/emartinv/event_level_reweighting_HH/HEFT_{13 if args.energy == 6500 else '13p6'}") / folder_name / "HH_variables" / "output.root"

    folder_name = f"testrun_params_klambda_{args.kl}_ct_{args.kt}_ctt_{args.c2}_cggh_{args.cg}_cgghh_{args.c2g}_energy_{args.energy}_TeV"
    true_base = Path(f"/eos/user/e/emartinv/event_level_reweighting_HH/ggHH_basispoint_13_13p6/") / folder_name / "HH_variables" / "output.root"

    for var, vinfo in VARIABLES.items():
        var_name = var
        hist_name = vinfo["histname"]
        x_label = vinfo["xlabel"]
        if "pT(" in x_label:
            var_inside = x_label[x_label.find('(')+1:x_label.find(')')]
            x_label = f"$p_{{\\mathrm{{T}}}}({var_inside})$ [GeV]"
        x_range = vinfo["x_range"]
        nbins = vinfo["nbins"]

        if true_base.exists():
            try:
                with uproot.open(true_base) as f:
                    tree = f["vars"]
                    arr_true = tree[var_name].array(library="np")
                    bin_edges = np.linspace(x_range[0], x_range[1], nbins+1)
                    hist_true, _ = np.histogram(arr_true, bins=bin_edges)
            except Exception as e:
                print(f"Error reading true distribution for {var}: {e}")
                hist_true = None
        else:
            print(f"True distribution not found at {true_base}")
            hist_true = None

        inferred_file = output_folder / f"{var}_inferred.root"
        inferred_hist = None
        if inferred_file.exists():
            try:
                with uproot.open(str(inferred_file)) as f:
                    if f.keys():
                        hname = f.keys()[0]
                        h = f[hname]
                        inferred_hist = h.values()
            except Exception as e:
                print(f"Error reading inferred distribution for {var}: {e}")
                inferred_hist = None

        if hist_true is not None and inferred_hist is not None:
            sum_true = np.sum(hist_true)
            hist_true_norm = hist_true / sum_true if sum_true > 0 else hist_true
            sum_inferred = np.sum(inferred_hist)
            hist_inferred_norm = inferred_hist / sum_inferred if sum_inferred > 0 else inferred_hist
            err_true = np.sqrt(hist_true) / sum_true if sum_true > 0 else np.zeros_like(hist_true)
            err_inferred = np.sqrt(inferred_hist) / sum_inferred if sum_inferred > 0 else np.zeros_like(inferred_hist)
            diff = hist_inferred_norm - hist_true_norm

            if np.any(inferred_hist < -0.1):
                print(f"WARNING: Bins with values < -0.1 in {var} : {inferred_hist[inferred_hist < -0.1]}")
            inferred_hist = np.where(inferred_hist < 0, 0, inferred_hist)

            sample_true = np.repeat(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist_true.astype(int))
            sample_inferred = np.repeat(0.5 * (bin_edges[:-1] + bin_edges[1:]), inferred_hist.astype(int))
            ks_stat, ks_pval = ks_2samp(sample_inferred, sample_true)

            threshold = 0.01 * np.max(hist_true_norm)
            nonzero_bins = np.where(hist_true_norm > threshold)[0]
            xmin = bin_edges[nonzero_bins[0]]
            xmax = bin_edges[nonzero_bins[-1]+1]
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            plt.figure(figsize=(8,5))
            plt.step(centers, hist_true_norm, where='mid', color='royalblue', lw=2, alpha=0.7, label=None, zorder=2)
            plt.fill_between(centers, hist_true_norm-err_true, hist_true_norm+err_true, step='mid', color='royalblue', alpha=0.18, label=None, zorder=1)
            plt.errorbar(centers, hist_true_norm, yerr=err_true, fmt='o', color='royalblue', label='From model', capsize=2, zorder=5, markersize=3, linewidth=1)
            plt.step(centers, hist_inferred_norm, where='mid', color='black', lw=2, alpha=0.7, label=None, zorder=4)
            plt.fill_between(centers, hist_inferred_norm-err_inferred, hist_inferred_norm+err_inferred, step='mid', color='black', alpha=0.18, label=None, zorder=3)
            plt.errorbar(centers, hist_inferred_norm, yerr=err_inferred, fmt='s', color='black', label='From reweighting', capsize=2, zorder=6, markersize=3, linewidth=1)

            plt.xlabel(x_label, fontsize=14)
            plt.ylabel("Events (normalized to unity)", fontsize=14)
            plt.xlim(xmin, xmax)
            param_text = (
                f"kl = {args.kl:.2f}, kt = {args.kt:.2f}, c2 = {args.c2:.2f}, "
                f"cg = {args.cg:.2f}, c2g = {args.c2g:.2f}, KS = {ks_stat:.3f}"
            )
            plt.gcf().text(0.13, 0.97, param_text, fontsize=12, ha='left', va='top')
            plt.legend(fontsize=12, loc='upper right')
            plt.tight_layout(rect=[0,0,1,0.94])
            plot_png = output_folder / f"{var}_comparison.png"
            plot_pdf = output_folder / f"{var}_comparison.pdf"
            plt.savefig(str(plot_png), dpi=150)
            plt.savefig(str(plot_pdf))
            # print(f"Saved comparison plot as {plot_png} and {plot_pdf}")
            plt.close()