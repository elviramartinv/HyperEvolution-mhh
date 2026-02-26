#!/usr/bin/env python3
"""
Explore the 5D phase space (kl, kt, c2, cg, c2g) by generating random target
points and comparing the mHH distributions predicted by two bases (pso_0 and
psotemp) against the reweighting.

Usage:
    python explore_phase_space.py --output-dir <folder> --n-points <N>
    python explore_phase_space.py --output-dir results/exploration --n-points 20
    python explore_phase_space.py --help
"""

import os
import sys
import json
import argparse
import random
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

# Optional: plotly for interactive 3D plots
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import math
import ROOT
ROOT.TH1.AddDirectory(False)
import mplhep as hep

try:
    from mhh_scoring import calcDist, calcDistModel
except ImportError:
    from hyperevol.examples.mhh_scoring import calcDist, calcDistModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Find project root (directory containing 'hyperevol' folder)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = _script_dir
while not os.path.exists(os.path.join(_project_root, "hyperevol")) and _project_root != "/":
    _project_root = os.path.dirname(_project_root)
_DEFAULT_CONFIG = os.path.join(_project_root, "hyperevol", "examples", "config", "mhh_23.json")

pso_0 = [
    [2.213221919449151, -2.6934464500856405, 2.965671189693319, 1.1360050891505196, -2.1481034846500293],
    [9.76060116521356, -4.0, 3.0808116991517873, 4.979656543379024, 1.7948537607896533],
    [-8.504037399703757, 2.4570674701218005, 2.0456701185442197, 4.993273200619232, 0.31806830918129725],
    [12.42168994697454, 4.8885158267788515, 1.1736452461594093, -1.649885452005208, 2.9957802625445793],
    [-8.028050651896733, 1.4185892815035221, 4.637324985625828, 1.100571508815925, -2.141440691112374],
    [-0.6417775386928877, -3.4585451693407876, 1.5364031301770482, 1.449765057791862, -3.0],
    [4.516788454318963, 3.246315195973261, 0.03179409373963871, 4.515743016527684, -3.0],
    [6.776972542859125, 2.7536031422527922, 1.1291947583909658, -4.998564441693077, -3.530381379838789],
    [-7.295476241252043, -3.567728937382591, 4.927055574269499, 5.4698157741693745, 0.5460308050762466],
    [10.16791838833636, 3.168115455700297, 3.988981859905667, -3.9955456956734263, 2.5047215502921873],
    [-9.237668351371102, -0.8830087027316762, 1.507497463102394, -1.656941023505912, 1.0373600385652952],
    [15.871698091121043, 2.954226053612609, -1.4832395231493958, -4.648315411280079, -0.31921721853099594],
    [-6.0920951495343445, -1.2529968110851288, -0.661498001605271, 3.1356841635278725, 2.473949503061431],
    [-13.765201883693987, -1.4548873762027377, 2.2019807467420858, 6.0, 2.0762352620706594],
    [-13.956921393217277, -3.867174172277096, 2.121296077606367, 3.066609957858563, 1.9738262253820915],
    [8.2879617206356, -4.0, 9.611068631424631, 3.4554065505161273, 0.9602330195006838],
    [-3.368487622557269, -1.235213414828737, -1.3214826590555613, 5.485928293425935, -1.6308816657564937],
    [-1.1492785231960068, 0.09464020330641909, 3.076473122787405, -3.6375285759285543, 1.3735185288149214],
    [17.4, 1.8726121309906936, 0.2733568393951952, -2.568047729842719, -2.85757278314612],
    [9.019237551902098, 0.9358945607455986, -4.368407508947331, 3.9318179425747295, 2.72703446921505],
    [10.092619602076475, -0.25598532434815535, 0.5458135306798126, 1.061229295632444, 5.764408396915333],
    [-7.594748729686678, -3.5504975021060012, 0.9679708236284144, -0.5666309250467718, 2.8180805513778475],
    [-2.167934532063678, -0.46393247419740047, 3.091840758463215, 2.0233283474829102, -1.2791407684980047],
]

psotemp = [
    [
        5.76306775897426,
        0.60957255930352,
        0.2757471461722233,
        0.3539034679345107,
        1.6470864190895298
    ],
    [
        -3.264926168334715,
        3.951560064234396,
        5.0,
        0.07060844214613118,
        4.0
    ],
    [
        -3.66094197241397,
        1.0779919430503124,
        1.1444876325325517,
        -4.543414129193672,
        -2.086324376343318
    ],
    [
        13.776142713343612,
        6.0,
        4.8,
        -2.2471194229079066,
        2.914575463372937
    ],
    [
        -0.5102446669232454,
        -4.0,
        5.7,
        -1.8918011813608135,
        2.523631269874509
    ],
    [
        -17.5,
        -2.2192457731974353,
        -1.561925426134284,
        3.587673607734981,
        -1.9097200749524326
    ],
    [
        -17.248692794237684,
        0.25541740841360383,
        -1.0545614554215015,
        5.0,
        -2.360300550565078
    ],
    [
        -6.167280087615459,
        2.898361032670763,
        3.1047347943825936,
        2.6888845448700023,
        -0.7993986852185984
    ],
    [
        -8.42531627433404,
        -2.7668781210641837,
        -4.981786788480192,
        1.534339186975958,
        -0.9216953993789947
    ],
    [
        -8.858010458211641,
        -3.798082301577028,
        1.8756163268862356,
        1.512411203148254,
        2.2487785553632356
    ],
    [
        11.02650511878629,
        2.9632586454393097,
        1.348357475088319,
        -3.2235380258824056,
        3.9176005249108194
    ],
    [
        11.229088936147944,
        -1.5731677725730204,
        1.5331118641968002,
        4.285894664167079,
        0.8707429805808593
    ],
    [
        7.50077391701546,
        0.6423086612184954,
        4.2702578284904495,
        1.765297528403499,
        -0.6817536396734726
    ],
    [
        4.815076407029188,
        3.854636736030917,
        2.94968225346022,
        6.0,
        -0.9731513842322967
    ],
    [
        -5.390428177680153,
        0.08809131336044707,
        -0.6500408810826548,
        9.34997253744265,
        1.6513702705760684
    ],
    [
        15.428842238949272,
        5.250828933568307,
        14.623892303287555,
        -3.600536082070807,
        1.9429364946062562
    ],
    [
        0.7200237879529017,
        -0.1848591996030451,
        4.340128637534834,
        6.0,
        -1.5721225585844174
    ],
    [
        0.7742911431843156,
        4.294395624534779,
        3.5870500704526624,
        -1.4993240151206564,
        0.24039833217398765
    ],
    [
        0.8282652927738547,
        -3.682996998942519,
        5.0,
        1.294025367125575,
        0.44492459625784425
    ],
    [
        6.156700174967869,
        4.78800924152869,
        4.441178744579316,
        2.0033997687370015,
        1.3977761733162892
    ],
    [
        35.0,
        3.465547133658958,
        -1.9566635858237562,
        0.1285393896793815,
        5.920278547684675
    ],
    [
        9.270557685309987,
        0.43323227168253386,
        1.0112324652261968,
        -2.106083384790988,
        -2.3123399950800057
    ],
    [
        3.5041663192328274,
        3.593645251058606,
        2.952695773181962,
        5.0,
        2.6889000092102133
    ]
]

def convert_to_dict(basis_list):
    """Convert basis list to dictionary format."""
    result = {}
    for i, point in enumerate(basis_list):
        idx = str(i + 1)
        result[f"kl_{idx}"]  = point[0]
        result[f"kt_{idx}"]  = point[1]
        result[f"c2_{idx}"]  = point[2]
        result[f"cg_{idx}"]  = point[3]
        result[f"c2g_{idx}"] = point[4]
    return result

pso_0_dict   = convert_to_dict(pso_0)
psotemp_dict = convert_to_dict(psotemp)
samplesize   = 50000

PARAM_NAMES = ["kl", "kt", "c2", "cg", "c2g"]
PARAM_LABELS = [
    r"$\kappa_\lambda$",
    r"$\kappa_t$",
    r"$c_2$",
    r"$c_g$",
    r"$c_{2g}$",
]
PARAM_LABELS_HTML = ["kl", "kt", "c2", "cg", "c2g"]

def load_limits(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    limits = {
        "kl":  (cfg["kl_1"]["min"],  cfg["kl_1"]["max"]),
        "kt":  (cfg["kt_1"]["min"],  cfg["kt_1"]["max"]),
        "c2":  (cfg["c2_1"]["min"],  cfg["c2_1"]["max"]),
        "cg":  (cfg["cg_1"]["min"],  cfg["cg_1"]["max"]),
        "c2g": (cfg["c2g_1"]["min"], cfg["c2g_1"]["max"]),
    }
    return limits

def generate_random_points(limits, n, seed=None):
    """Generate n uniformly random points inside *limits*."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n):
        pts.append([
            float(rng.uniform(*limits["kl"])),
            float(rng.uniform(*limits["kt"])),
            float(rng.uniform(*limits["c2"])),
            float(rng.uniform(*limits["cg"])),
            float(rng.uniform(*limits["c2g"])),
        ])
    return pts


def compute_chi2_nll(model, ref):
    ndf = ref.GetNbinsX()
    chi2_val = nll_val = 0.0

    for b in range(model.GetNbinsX()):
        mc = model.GetBinContent(b + 1)
        rc = ref.GetBinContent(b + 1)
        re = ref.GetBinError(b + 1)
        diff = mc - rc
        chi2_val += diff ** 2 / re ** 2 if re > 0 else 0

        eps = 1e-10
        if rc > 0 and mc > eps:
            nll_val += 2 * (mc - rc + rc * math.log(rc / mc))
        elif rc == 0 and mc > 0:
            nll_val += 2 * mc

    ks_val = ref.KolmogorovTest(model, "")

    return chi2_val / ndf, nll_val, ks_val


def plot_mhh_for_point(kl, kt, c2, cg, c2g, out_dir):
    reweight      = calcDist(kl, kt, c2, cg, c2g, samplesize=samplesize)
    model_pso0    = calcDistModel(kl, kt, c2, cg, c2g, pso_0_dict,   samplesize=samplesize, use_LO=True)
    model_psotemp = calcDistModel(kl, kt, c2, cg, c2g, psotemp_dict, samplesize=samplesize, use_LO=False)

    models     = [model_pso0,    model_psotemp]
    names      = ["pso_0",       "psotemp"]
    colors_mod = ["tab:orange",  "tab:green"]

    metrics = []
    for m in models:
        chi2_ndf, nll, ks_val = compute_chi2_nll(m, reweight)
        metrics.append((ks_val, chi2_ndf, nll))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[4, 1])
    hep.histplot(reweight, histtype="errorbar", label="from reweight",
                 linewidth=3, density=False, color="black", ax=ax)

    for m, name, col, (ks, chi2r, nll) in zip(models, names, colors_mod, metrics):
        lbl = (r"{} (KS={:.3f}, $\chi^2$/ndf={:.2f}, NLL={:.1f})"
               .format(name, ks, chi2r, nll))
        hep.histplot(m, histtype="band",    linewidth=3, density=False,
                     alpha=0.5, color=col, ax=ax)
        hep.histplot(m, histtype="errorbar", linewidth=3, density=False,
                     label=lbl, color=col, ax=ax)

        ratio = m.Clone(m.GetName() + "_ratio")
        ratio.Divide(reweight)
        hep.histplot(ratio, histtype="band",    linewidth=3, density=False,
                     alpha=0.5, color=col, ax=ax2)
        hep.histplot(ratio, histtype="errorbar", linewidth=3, density=False,
                     color=col, ax=ax2)

    hep.cms.lumitext(r'138 fb$^{-1}$ (13.6 TeV)', ax=ax)
    title = (r"$\kappa_\lambda$={:.2f}, $\kappa_t$={:.2f}, $c_2$={:.2f},"
             r" $c_g$={:.2f}, $c_{{2g}}$={:.2f}").format(kl, kt, c2, cg, c2g)
    ax.text(0.02, 0.97, title, transform=ax.transAxes, fontsize=11,
            va="top", ha="left")
    ax.legend(fontsize=8)
    ax.set_xticklabels([])
    ax.set_ylabel("Events\n")
    ax.set_xlim(200, 1000)
    ymin = min(model_pso0.GetMinimum(), model_psotemp.GetMinimum(),
               reweight.GetMinimum(), 0)
    ax.set_ylim(ymin, ax.get_ylim()[1])

    ax2.set_xlabel(r"$m_{HH}$ (GeV)")
    ax2.set_ylabel("ratio\n")
    ax2.set_xlim(200, 1000)
    ax2.set_ylim(max(-2, ax2.get_ylim()[0]), min(5, ax2.get_ylim()[1]))
    ax2.axhline(1, linewidth=2, color="black")
    plt.subplots_adjust(wspace=0, hspace=0.1)

    fname = os.path.join(
        out_dir,
        f"mhh_kl{kl:.2f}_kt{kt:.2f}_c2{c2:.2f}_cg{cg:.2f}_c2g{c2g:.2f}.png",
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def compute_metrics_for_points(target_points):
    """
    Compute chi2/ndf and NLL for every target point against both bases.
    """
    metrics = []
    n = len(target_points)
    for idx, (kl, kt, c2, cg, c2g) in enumerate(target_points):
        print(f"  Metrics [{idx+1}/{n}] "
              f"kl={kl:.3f} kt={kt:.3f} c2={c2:.3f} cg={cg:.3f} c2g={c2g:.3f}")
        try:
            reweight       = calcDist(kl, kt, c2, cg, c2g, samplesize=samplesize)
            model_pso0     = calcDistModel(kl, kt, c2, cg, c2g, pso_0_dict,
                                           samplesize=samplesize, use_LO=True)
            model_psotemp  = calcDistModel(kl, kt, c2, cg, c2g, psotemp_dict,
                                           samplesize=samplesize, use_LO=False)
            chi2_p0,  nll_p0, _  = compute_chi2_nll(model_pso0,    reweight)
            chi2_pt,  nll_pt, _  = compute_chi2_nll(model_psotemp, reweight)
        except Exception as e:
            print(f"    WARNING: metric computation failed — {e}")
            chi2_p0 = nll_p0 = chi2_pt = nll_pt = float("nan")
        metrics.append({
            "pso0_chi2":    chi2_p0,
            "pso0_nll":     nll_p0,
            "psotemp_chi2": chi2_pt,
            "psotemp_nll":  nll_pt,
        })
    return metrics


# ── Phase-space visualisation ─────────────────────────────────────────────────

def _basis_array(basis_list):
    return np.array(basis_list)

_COL_PSO0    = "#7B2D8B"   # purple
_COL_PSOTEMP = "#1F9ECF"   # steel blue
_COL_TARGETS = "#4472C4"   # blue (target points, no metric)


def plot_phase_space(target_points, out_dir, limits, metrics=None):
    os.makedirs(out_dir, exist_ok=True)

    pso0_arr    = _basis_array(pso_0)      # (23, 5)
    psotemp_arr = _basis_array(psotemp)    # (23, 5)
    targets_arr = np.array(target_points)  # (N, 5)

    lims_list = list(limits.values())
    ext_lims  = [(lo - 2, hi + 2) for (lo, hi) in lims_list]

    kw_targets = dict(color=_COL_TARGETS, alpha=0.40, s=35, zorder=1)
    kw_pso0    = dict(color=_COL_PSO0,    alpha=0.90, s=60, marker="o",
                      edgecolors="black", linewidths=0.5, zorder=3)
    kw_psotemp = dict(color=_COL_PSOTEMP, alpha=0.90, s=60, marker="s",
                      edgecolors="black", linewidths=0.5, zorder=3)

    from matplotlib.lines import Line2D
    legend_elements_2d = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_COL_TARGETS,
               markersize=9, alpha=0.6, label="Target points (random)"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_COL_PSO0,
               markeredgecolor='black', markersize=9, label="pso_0 basis"),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=_COL_PSOTEMP,
               markeredgecolor='black', markersize=9, label="psotemp basis"),
    ]

    # ── 2D diagonal: one 1D histogram per parameter ───────────────────────────
    for p in range(len(PARAM_NAMES)):
        xl = ext_lims[p]
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = 25
        ax.hist(targets_arr[:, p], bins=bins, range=xl,
                color=_COL_TARGETS, alpha=0.4, density=True, label="Targets (random)")
        ax.hist(pso0_arr[:, p],    bins=bins, range=xl,
                color=_COL_PSO0,    alpha=0.9, density=True, histtype="step",
                linewidth=2, label="pso_0")
        ax.hist(psotemp_arr[:, p], bins=bins, range=xl,
                color=_COL_PSOTEMP, alpha=0.9, density=True, histtype="step",
                linewidth=2, label="psotemp")
        ax.set_xlim(*xl)
        ax.set_xlabel(PARAM_LABELS[p], fontsize=13)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Marginal distribution — {PARAM_NAMES[p]}", fontsize=12)
        ax.legend(fontsize=10)
        plt.tight_layout()
        fname_1d = os.path.join(out_dir, f"phase_space_1d_{PARAM_NAMES[p]}.png")
        plt.savefig(fname_1d, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved 1D plot: {fname_1d}")

    # ── 2D off-diagonal: one scatter per pair (row > col to avoid duplicates) ─
    for row in range(len(PARAM_NAMES)):
        for col in range(row):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(targets_arr[:, col], targets_arr[:, row], **kw_targets)
            ax.scatter(pso0_arr[:,    col], pso0_arr[:,    row],  **kw_pso0)
            ax.scatter(psotemp_arr[:, col], psotemp_arr[:, row],  **kw_psotemp)
            ax.set_xlim(*ext_lims[col])
            ax.set_ylim(*ext_lims[row])
            ax.set_xlabel(PARAM_LABELS[col], fontsize=13)
            ax.set_ylabel(PARAM_LABELS[row], fontsize=13)
            ax.set_title(
                f"Phase-space — {PARAM_NAMES[col]} vs {PARAM_NAMES[row]}",
                fontsize=12)
            ax.legend(handles=legend_elements_2d, fontsize=9)
            plt.tight_layout()
            fname_2d = os.path.join(
                out_dir,
                f"phase_space_2d_{PARAM_NAMES[col]}_{PARAM_NAMES[row]}.png",
            )
            plt.savefig(fname_2d, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved 2D scatter: {fname_2d}")

    # ── 2D scatter plots with metric colouring ────────────────────────────────
    if metrics is not None:
        print("\n  Creating 2D plots with metric colouring...")
        cmap = matplotlib.colormaps["RdYlGn_r"]

        basis_configs = [
            ("pso_0",   pso0_arr,    _COL_PSO0,    "o",
             "chi2", [m["pso0_chi2"]    for m in metrics],
             r"$\chi^2$/ndf"),
            ("pso_0",   pso0_arr,    _COL_PSO0,    "o",
             "nll",  [m["pso0_nll"]     for m in metrics],
             "NLL"),
            ("psotemp", psotemp_arr, _COL_PSOTEMP, "s",
             "chi2", [m["psotemp_chi2"] for m in metrics],
             r"$\chi^2$/ndf"),
            ("psotemp", psotemp_arr, _COL_PSOTEMP, "s",
             "nll",  [m["psotemp_nll"]  for m in metrics],
             "NLL"),
        ]

        for row in range(len(PARAM_NAMES)):
            for col in range(row):
                for (bname, barr, bcol, bmk, mkey, mvals, mlabel) in basis_configs:
                    mvals_arr = np.array(mvals, dtype=float)
                    finite    = mvals_arr[np.isfinite(mvals_arr)]
                    vmin      = finite.min() if len(finite) else 0
                    vmax      = finite.max() if len(finite) else 1
                    norm      = plt.Normalize(vmin=vmin, vmax=vmax)

                    fig, ax = plt.subplots(figsize=(7, 6))

                    # Target points coloured by metric
                    sc = ax.scatter(
                        targets_arr[:, col], targets_arr[:, row],
                        c=mvals_arr, cmap=cmap, norm=norm,
                        alpha=0.80, s=50, zorder=1,
                        label="Targets (random)",
                    )
                    cbar = fig.colorbar(sc, ax=ax, label=mlabel)
                    cbar.ax.tick_params(labelsize=9)

                    # Basis points — neutral fixed colour, distinct from colormap
                    ax.scatter(
                        barr[:, col], barr[:, row],
                        color=bcol, s=90, marker=bmk,
                        edgecolors="black", linewidths=0.7,
                        label=f"{bname} basis", zorder=3,
                    )

                    ax.set_xlim(*ext_lims[col])
                    ax.set_ylim(*ext_lims[row])
                    ax.set_xlabel(PARAM_LABELS[col], fontsize=13)
                    ax.set_ylabel(PARAM_LABELS[row], fontsize=13)
                    title = (f"Phase-space 2D — {PARAM_NAMES[col]} vs {PARAM_NAMES[row]}\n"
                             f"{bname} — {mlabel}")
                    ax.set_title(title, fontsize=11)
                    ax.legend(fontsize=9, loc="best")
                    plt.tight_layout()

                    fname_2d_metric = os.path.join(
                        out_dir,
                        f"phase_space_2d_{PARAM_NAMES[col]}_{PARAM_NAMES[row]}_{bname}_{mkey}.png",
                    )
                    plt.savefig(fname_2d_metric, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                print(f"  Saved 2D metric plots: {PARAM_NAMES[col]} vs {PARAM_NAMES[row]}")

    # ── Interactive 2D plots with plotly (HTML) ───────────────────────────────
    if PLOTLY_AVAILABLE:
        print("\n  Creating interactive 2D plots (HTML)...")
        _create_interactive_2d_plots(targets_arr, pso0_arr, psotemp_arr,
                                     ext_lims, out_dir, metrics)
    else:
        print("\n  Plotly not available - skipping interactive 2D HTML plots.")

    # ── 3D scatter plots for all possible triplets ────────────────────────────
    triplets = list(combinations(range(len(PARAM_NAMES)), 3))

    def _save_3d(fig, i, j, k, suffix=""):
        fname = os.path.join(
            out_dir,
            f"phase_space_3d_{PARAM_NAMES[i]}_{PARAM_NAMES[j]}_{PARAM_NAMES[k]}{suffix}.png",
        )
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved 3D plot: {fname}")

    def _setup_3d_axes(ax3, i, j, k, title):
        ax3.set_xlabel(PARAM_LABELS[i], fontsize=14, labelpad=14)
        ax3.set_ylabel(PARAM_LABELS[j], fontsize=14, labelpad=14)
        ax3.set_zlabel(PARAM_LABELS[k], fontsize=14, labelpad=14)
        ax3.set_xlim(*ext_lims[i])
        ax3.set_ylim(*ext_lims[j])
        ax3.set_zlim(*ext_lims[k])
        ax3.set_title(title, fontsize=13)
        ax3.tick_params(axis='both', which='major', labelsize=10)

    for (i, j, k) in triplets:
        base_title = "Phase-space 3D — {}, {}, {}".format(
            PARAM_NAMES[i], PARAM_NAMES[j], PARAM_NAMES[k])

        # ── plain plot: both bases + semi-transparent targets (always produced) ─
        fig = plt.figure(figsize=(13, 9))
        ax3 = fig.add_subplot(111, projection="3d")
        ax3.scatter(targets_arr[:, i], targets_arr[:, j], targets_arr[:, k],
                    color=_COL_TARGETS, alpha=0.45, s=40,
                    label="Targets (random)", zorder=1)
        ax3.scatter(pso0_arr[:, i],    pso0_arr[:, j],    pso0_arr[:, k],
                    color=_COL_PSO0, s=90, marker="o",
                    edgecolors="black", linewidths=0.6,
                    label="pso_0 basis", zorder=3)
        ax3.scatter(psotemp_arr[:, i], psotemp_arr[:, j], psotemp_arr[:, k],
                    color=_COL_PSOTEMP, s=90, marker="s",
                    edgecolors="black", linewidths=0.6,
                    label="psotemp basis", zorder=3)
        _setup_3d_axes(ax3, i, j, k, base_title)
        ax3.legend(fontsize=12, loc="upper left")
        plt.tight_layout()
        _save_3d(fig, i, j, k)

        if metrics is not None:
            # ── 4 metric-coloured plots per triplet ───────────────────────────
            # RdYlGn_r: low value (good fit) → green, high value (bad fit) → red
            cmap = matplotlib.colormaps["RdYlGn_r"]

            basis_configs = [
                ("pso_0",   pso0_arr,    _COL_PSO0,    "o",
                 "chi2", [m["pso0_chi2"]    for m in metrics],
                 r"$\chi^2$/ndf"),
                ("pso_0",   pso0_arr,    _COL_PSO0,    "o",
                 "nll",  [m["pso0_nll"]     for m in metrics],
                 "NLL"),
                ("psotemp", psotemp_arr, _COL_PSOTEMP, "s",
                 "chi2", [m["psotemp_chi2"] for m in metrics],
                 r"$\chi^2$/ndf"),
                ("psotemp", psotemp_arr, _COL_PSOTEMP, "s",
                 "nll",  [m["psotemp_nll"]  for m in metrics],
                 "NLL"),
            ]

            for (bname, barr, bcol, bmk, mkey, mvals, mlabel) in basis_configs:
                mvals_arr = np.array(mvals, dtype=float)
                finite    = mvals_arr[np.isfinite(mvals_arr)]
                vmin      = finite.min() if len(finite) else 0
                vmax      = finite.max() if len(finite) else 1
                norm      = plt.Normalize(vmin=vmin, vmax=vmax)

                fig = plt.figure(figsize=(13, 9))
                ax3 = fig.add_subplot(111, projection="3d")

                # Target points coloured by metric
                sc = ax3.scatter(
                    targets_arr[:, i], targets_arr[:, j], targets_arr[:, k],
                    c=mvals_arr, cmap=cmap, norm=norm,
                    alpha=0.80, s=55, zorder=1,
                    label="Targets (random)",
                )
                cbar = fig.colorbar(sc, ax=ax3, shrink=0.55, pad=0.12,
                                    label=mlabel)
                cbar.ax.tick_params(labelsize=10)

                # Basis points — neutral fixed colour, distinct from colormap
                ax3.scatter(
                    barr[:, i], barr[:, j], barr[:, k],
                    color=bcol, s=110, marker=bmk,
                    edgecolors="black", linewidths=0.7,
                    label=f"{bname} basis", zorder=3,
                )

                title = f"{base_title}\n{bname} — {mlabel}"
                _setup_3d_axes(ax3, i, j, k, title)
                ax3.legend(fontsize=12, loc="upper left")
                plt.tight_layout()
                _save_3d(fig, i, j, k, suffix=f"_{bname}_{mkey}")

    # ── Interactive 3D plots with plotly (HTML) ───────────────────────────────
    if PLOTLY_AVAILABLE:
        print("\n  Creating interactive 3D plots (HTML)...")
        _create_interactive_3d_plots(triplets, targets_arr, pso0_arr, psotemp_arr,
                                     ext_lims, out_dir, metrics)
    else:
        print("\n  Plotly not available - skipping interactive 3D HTML plots.")
        print("  Install with: pip install plotly")


def _create_interactive_2d_plots(targets_arr, pso0_arr, psotemp_arr,
                                  ext_lims, out_dir, metrics=None):
    """Create interactive 2D scatter plots using plotly (saved as HTML)."""
    for row in range(len(PARAM_NAMES)):
        for col in range(row):  # strictly lower triangle
            # ── Simple plot: both bases + targets ──────────────────────────
            fig = go.Figure()

            # Target points
            fig.add_trace(go.Scatter(
                x=targets_arr[:, col], y=targets_arr[:, row],
                mode='markers',
                name='Target points (random)',
                marker=dict(size=6, color=_COL_TARGETS, opacity=0.5),
                text=[f'Point {i+1}' for i in range(len(targets_arr))],
                hovertemplate='%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>',
            ))

            # pso_0 basis
            fig.add_trace(go.Scatter(
                x=pso0_arr[:, col], y=pso0_arr[:, row],
                mode='markers',
                name='pso_0 basis',
                marker=dict(size=10, color=_COL_PSO0, symbol='circle',
                           line=dict(color='black', width=1)),
                text=[f'pso_0 #{i+1}' for i in range(len(pso0_arr))],
                hovertemplate='%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>',
            ))

            # psotemp basis
            fig.add_trace(go.Scatter(
                x=psotemp_arr[:, col], y=psotemp_arr[:, row],
                mode='markers',
                name='psotemp basis',
                marker=dict(size=10, color=_COL_PSOTEMP, symbol='square',
                           line=dict(color='black', width=1)),
                text=[f'psotemp #{i+1}' for i in range(len(psotemp_arr))],
                hovertemplate='%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>',
            ))

            fig.update_layout(
                title=dict(
                    text=f"Phase-space 2D — {PARAM_NAMES[col]} vs {PARAM_NAMES[row]} (interactive)",
                    x=0.5, xanchor='center'
                ),
                xaxis=dict(title=PARAM_LABELS_HTML[col], range=ext_lims[col]),
                yaxis=dict(title=PARAM_LABELS_HTML[row], range=ext_lims[row]),
                width=800, height=600,
                font=dict(size=13),
                hovermode='closest',
                legend=dict(
                    x=0.02, y=0.98,
                    xanchor='left', yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black', borderwidth=1
                ),
            )

            fname_html = os.path.join(
                out_dir,
                f"phase_space_2d_{PARAM_NAMES[col]}_{PARAM_NAMES[row]}_interactive.html",
            )
            fig.write_html(fname_html)
            print(f"  Saved interactive 2D: {PARAM_NAMES[col]} vs {PARAM_NAMES[row]}")

    # ── Metric-coloured 2D plots ───────────────────────────────────────────
    if metrics is not None:
        basis_configs = [
            ("pso_0",   pso0_arr,    "chi2", [m["pso0_chi2"]    for m in metrics],
             "chi2/ndf"),
            ("pso_0",   pso0_arr,    "nll",  [m["pso0_nll"]     for m in metrics],
             "NLL"),
            ("psotemp", psotemp_arr, "chi2", [m["psotemp_chi2"] for m in metrics],
             "chi2/ndf"),
            ("psotemp", psotemp_arr, "nll",  [m["psotemp_nll"]  for m in metrics],
             "NLL"),
        ]

        for row in range(len(PARAM_NAMES)):
            for col in range(row):  # strictly lower triangle
                for (bname, barr, mkey, mvals, mlabel) in basis_configs:
                    mvals_arr = np.array(mvals, dtype=float)
                    finite = mvals_arr[np.isfinite(mvals_arr)]
                    vmin = finite.min() if len(finite) else 0
                    vmax = finite.max() if len(finite) else 1

                    fig = go.Figure()

                    # Target points coloured by metric
                    fig.add_trace(go.Scatter(
                        x=targets_arr[:, col], y=targets_arr[:, row],
                        mode='markers',
                        name='Target points',
                        marker=dict(
                            size=8,
                            color=mvals_arr,
                            colorscale='RdYlGn_r',
                            cmin=vmin, cmax=vmax,
                            opacity=0.8,
                            colorbar=dict(
                                title=mlabel,
                                thickness=20,
                                len=0.7,
                            ),
                        ),
                        text=[f"{mlabel}: {v:.2f}" for v in mvals_arr],
                        hovertemplate='%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>',
                    ))

                    # Basis points - fixed color
                    bcol = _COL_PSO0 if bname == "pso_0" else _COL_PSOTEMP
                    bsymbol = 'circle' if bname == "pso_0" else 'square'
                    fig.add_trace(go.Scatter(
                        x=barr[:, col], y=barr[:, row],
                        mode='markers',
                        name=f'{bname} basis',
                        marker=dict(size=11, color=bcol, symbol=bsymbol,
                                   line=dict(color='black', width=1.5)),
                        text=[f'{bname} #{i+1}' for i in range(len(barr))],
                        hovertemplate='%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>',
                    ))

                    fig.update_layout(
                        title=dict(
                            text=f"Phase-space 2D — {PARAM_NAMES[col]} vs {PARAM_NAMES[row]}<br>{bname} — {mlabel} (interactive)",
                            x=0.5, xanchor='center'
                        ),
                        xaxis=dict(title=PARAM_LABELS_HTML[col], range=ext_lims[col]),
                        yaxis=dict(title=PARAM_LABELS_HTML[row], range=ext_lims[row]),
                        width=850, height=600,
                        font=dict(size=13),
                        hovermode='closest',
                        legend=dict(
                            x=0.02, y=0.98,
                            xanchor='left', yanchor='top',
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='black', borderwidth=1
                        ),
                    )

                    fname_html_metric = os.path.join(
                        out_dir,
                        f"phase_space_2d_{PARAM_NAMES[col]}_{PARAM_NAMES[row]}_{bname}_{mkey}_interactive.html",
                    )
                    fig.write_html(fname_html_metric)
                print(f"  Saved interactive 2D metric plots: {PARAM_NAMES[col]} vs {PARAM_NAMES[row]}")


def _create_interactive_3d_plots(triplets, targets_arr, pso0_arr, psotemp_arr,
                                  ext_lims, out_dir, metrics=None):
    """Create interactive 3D scatter plots using plotly (saved as HTML)."""
    for (i, j, k) in triplets:
        # ── Simple plot: both bases + targets ──────────────────────────────────
        fig = go.Figure()

        # Target points
        fig.add_trace(go.Scatter3d(
            x=targets_arr[:, i], y=targets_arr[:, j], z=targets_arr[:, k],
            mode='markers',
            name='Target points (random)',
            marker=dict(size=4, color=_COL_TARGETS, opacity=0.5),
        ))

        # pso_0 basis
        fig.add_trace(go.Scatter3d(
            x=pso0_arr[:, i], y=pso0_arr[:, j], z=pso0_arr[:, k],
            mode='markers',
            name='pso_0 basis',
            marker=dict(size=8, color=_COL_PSO0, symbol='circle',
                       line=dict(color='black', width=1)),
        ))

        # psotemp basis
        fig.add_trace(go.Scatter3d(
            x=psotemp_arr[:, i], y=psotemp_arr[:, j], z=psotemp_arr[:, k],
            mode='markers',
            name='psotemp basis',
            marker=dict(size=8, color=_COL_PSOTEMP, symbol='square',
                       line=dict(color='black', width=1)),
        ))

        fig.update_layout(
            title=dict(
                text=f"Phase-space 3D — {PARAM_NAMES[i]}, {PARAM_NAMES[j]}, {PARAM_NAMES[k]} (interactive)",
                x=0.5, xanchor='center'
            ),
            scene=dict(
                xaxis=dict(title=PARAM_LABELS_HTML[i], range=ext_lims[i]),
                yaxis=dict(title=PARAM_LABELS_HTML[j], range=ext_lims[j]),
                zaxis=dict(title=PARAM_LABELS_HTML[k], range=ext_lims[k]),
                domain=dict(x=[0, 0.85], y=[0, 1]),  # Make scene wider
            ),
            width=1000, height=750,
            font=dict(size=13),
            margin=dict(l=0, r=0, t=80, b=0),
            legend=dict(
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black', borderwidth=1
            ),
        )

        fname_html = os.path.join(
            out_dir,
            f"phase_space_3d_{PARAM_NAMES[i]}_{PARAM_NAMES[j]}_{PARAM_NAMES[k]}_interactive.html",
        )
        fig.write_html(fname_html)
        print(f"  Saved interactive 3D: {fname_html}")

        # ── Metric-coloured plots ──────────────────────────────────────────────
        if metrics is not None:
            basis_configs = [
                ("pso_0",   pso0_arr,    "chi2", [m["pso0_chi2"]    for m in metrics],
                 r"χ²/ndf"),
                ("pso_0",   pso0_arr,    "nll",  [m["pso0_nll"]     for m in metrics],
                 "NLL"),
                ("psotemp", psotemp_arr, "chi2", [m["psotemp_chi2"] for m in metrics],
                 r"χ²/ndf"),
                ("psotemp", psotemp_arr, "nll",  [m["psotemp_nll"]  for m in metrics],
                 "NLL"),
            ]

            for (bname, barr, mkey, mvals, mlabel) in basis_configs:
                mvals_arr = np.array(mvals, dtype=float)
                finite = mvals_arr[np.isfinite(mvals_arr)]
                vmin = finite.min() if len(finite) else 0
                vmax = finite.max() if len(finite) else 1

                fig = go.Figure()

                # Target points coloured by metric
                fig.add_trace(go.Scatter3d(
                    x=targets_arr[:, i], y=targets_arr[:, j], z=targets_arr[:, k],
                    mode='markers',
                    name='Target points',
                    marker=dict(
                        size=5,
                        color=mvals_arr,
                        colorscale='RdYlGn_r',
                        cmin=vmin, cmax=vmax,
                        opacity=0.8,
                        colorbar=dict(
                            title=mlabel,
                            thickness=20,
                            len=0.7,
                            x=1.0,
                            xanchor='left',
                        ),
                    ),
                    text=[f"{mlabel}: {v:.2f}" for v in mvals_arr],
                    hovertemplate='%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>',
                ))

                # Basis points - fixed color
                bcol = _COL_PSO0 if bname == "pso_0" else _COL_PSOTEMP
                bsymbol = 'circle' if bname == "pso_0" else 'square'
                fig.add_trace(go.Scatter3d(
                    x=barr[:, i], y=barr[:, j], z=barr[:, k],
                    mode='markers',
                    name=f'{bname} basis',
                    marker=dict(size=9, color=bcol, symbol=bsymbol,
                               line=dict(color='black', width=1.5)),
                ))

                fig.update_layout(
                    title=dict(
                        text=f"Phase-space 3D — {PARAM_NAMES[i]}, {PARAM_NAMES[j]}, {PARAM_NAMES[k]}<br>{bname} — {mlabel} (interactive)",
                        x=0.5, xanchor='center'
                    ),
                    scene=dict(
                        xaxis=dict(title=PARAM_LABELS_HTML[i], range=ext_lims[i]),
                        yaxis=dict(title=PARAM_LABELS_HTML[j], range=ext_lims[j]),
                        zaxis=dict(title=PARAM_LABELS_HTML[k], range=ext_lims[k]),
                        domain=dict(x=[0, 0.75], y=[0, 1]),  # Leave space for colorbar
                    ),
                    width=1000, height=750,
                    font=dict(size=13),
                    margin=dict(l=0, r=100, t=80, b=0),  # Right margin for colorbar
                    legend=dict(
                        x=0.02, y=0.98,
                        xanchor='left', yanchor='top',
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='black', borderwidth=1
                    ),
                )

                fname_html_metric = os.path.join(
                    out_dir,
                    f"phase_space_3d_{PARAM_NAMES[i]}_{PARAM_NAMES[j]}_{PARAM_NAMES[k]}_{bname}_{mkey}_interactive.html",
                )
                fig.write_html(fname_html_metric)
            print(f"  Saved interactive 3D metric plots: {PARAM_NAMES[i]}, {PARAM_NAMES[j]}, {PARAM_NAMES[k]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explore the 5D (kl,kt,c2,cg,c2g) phase space by "
                    "comparing mHH distributions at random target points."
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Top-level output folder (will be created if needed)."
    )
    parser.add_argument(
        "--n-points", "-n", type=int, default=20,
        help="Number of random target points to generate (default: 20)."
    )
    parser.add_argument(
        "--config", default=_DEFAULT_CONFIG,
        help="Path to parameter-range JSON (default: hyperevol/examples/config/mhh_23.json)."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--skip-mhh", action="store_true",
        help="Skip mHH distribution plots (only make phase-space plots)."
    )
    parser.add_argument(
        "--skip-metrics", action="store_true",
        help="Skip metric computation — 3D scatter plots will have no colour coding."
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**32)
    print(f"Random seed: {seed}")

    # ── Load limits ───────────────────────────────────────────────────────────
    limits = load_limits(args.config)
    print("Parameter limits loaded:")
    for k, (lo, hi) in limits.items():
        print(f"  {k}: [{lo}, {hi}]")

    # ── Generate random target points ─────────────────────────────────────────
    target_points = generate_random_points(limits, args.n_points, seed=seed)
    print(f"\nGenerated {len(target_points)} random target points.")

    # Save the target points to a JSON for reference
    targets_json = os.path.join(args.output_dir, "target_points.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(targets_json, "w") as f:
        json.dump(
            [{"kl": p[0], "kt": p[1], "c2": p[2], "cg": p[3], "c2g": p[4]}
             for p in target_points],
            f, indent=2,
        )
    print(f"\nTarget points saved to {targets_json}")

    # ── Metric computation ────────────────────────────────────────────
    point_metrics = None
    if not args.skip_metrics:
        print(f"\n--- Computing metrics for {len(target_points)} target points ---")
        point_metrics = compute_metrics_for_points(target_points)
        metrics_json = os.path.join(args.output_dir, "target_metrics.json")
        with open(metrics_json, "w") as f:
            json.dump(point_metrics, f, indent=2)
        print(f"Metrics saved to {metrics_json}")
    else:
        print("\n--skip-metrics flag set, skipping metric computation.")

    # ── Phase-space visualisation ─────────────────────────────────────────────
    ps_dir = os.path.join(args.output_dir, "phase_space")
    os.makedirs(ps_dir, exist_ok=True)
    print("\n--- Phase-space plots ---")
    plot_phase_space(target_points, ps_dir, limits, metrics=point_metrics)

    if args.skip_mhh:
        print("\n--skip-mhh flag set, skipping mHH distribution plots.")
        print("\nDone!")
        print(f"  Phase-space plots : {ps_dir}")
        return

    # ── mHH distribution plots ────────────────────────────────────────────────
    mhh_dir = os.path.join(args.output_dir, "mhh_distributions")
    os.makedirs(mhh_dir, exist_ok=True)
    print(f"\n--- mHH distribution plots ({len(target_points)} points) ---")

    failed = []
    for idx, (kl, kt, c2, cg, c2g) in enumerate(target_points):
        print(f"\n[{idx+1}/{len(target_points)}] "
              f"kl={kl:.3f} kt={kt:.3f} c2={c2:.3f} cg={cg:.3f} c2g={c2g:.3f}")
        try:
            plot_mhh_for_point(kl, kt, c2, cg, c2g, mhh_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append((kl, kt, c2, cg, c2g, str(e)))

    if failed:
        print(f"\nFailed points ({len(failed)}):")
        for item in failed:
            print(f"  kl={item[0]:.3f} kt={item[1]:.3f} c2={item[2]:.3f} "
                  f"cg={item[3]:.3f} c2g={item[4]:.3f} — {item[5]}")

    print("\nDone!")
    print(f"  Phase-space plots : {ps_dir}")
    print(f"  mHH distributions : {mhh_dir}")


if __name__ == "__main__":
    main()
