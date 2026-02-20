#!/usr/bin/env python3
"""
Script to compare mHH distributions:
  1. "from LHE"          - direct LHE events read from ROOT files
  2. "from reweight"     - analytical reweighting via calcDist
  3. "from model (toys)" - morphing prediction via calcDistModel (from toys)
  4. "from model (LHE)"  - morphing prediction via calcDistModel (from real LHE basis histograms)

Basis points:
  /eos/user/e/emartinv/event_level_reweighting_HH/ggHH_basispoint_13_13p6

Coupling convention:
    cg_LHE  = cg_reweight  / 1.5
    c2g_LHE = -c2g_reweight / 3
"""

import ROOT
ROOT.TH1.AddDirectory(False)
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True})
import numpy as np
import mplhep as hep
plt.style.use(hep.style.CMS)
import seaborn as sns
import os
import sys
import math
import sympy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hyperevol.examples.mhh_scoring import calcDist, calcDistModel, makeKey

# ---------------------------------------------------------------------------
# LHE convention differs in cg and c2g:
#   cg_LHE  =  cg_rw / 1.5
#   c2g_LHE = -c2g_rw / 3
# The conversion is applied on-the-fly in compare_triple().
# ---------------------------------------------------------------------------
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

def rw_to_lhe(kl, kt, c2, cg, c2g):
    """Convert reweight/model couplings to LHE file-naming convention.
    Values are rounded to 4 decimal places to match the directory names on EOS."""
    return (round(kl, 4), round(kt, 4), round(c2, 4),
            round(cg / 1.5, 4), round(-c2g / 3, 4))

def func5D_LO(sample):
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


def model_5D_LO(basis_points_lhe, target_point_lhe):
    M = sympy.Matrix([func5D_LO(pt) for pt in basis_points_lhe])
    c = sympy.Matrix(func5D_LO(target_point_lhe))
    coeffs = c.transpose() * M.pinv()
    return [float(x) for x in coeffs]


def get_lhe_morphed_distribution(target_point_lhe, reference_hist, energy_gev):
    """
    target_point_lhe : [kl, kt, c2, cg, c2g] in LHE convention
    reference_hist   : TH1 used only to define the binning
    energy_gev       : 6500 or 6800
    """
    # All 23 pso_0 basis points converted to LHE convention
    basis_lhe = [list(rw_to_lhe(*pt)) for pt in pso_0]

    coeffs = model_5D_LO(basis_lhe, target_point_lhe)

    h_morphed = None
    for i, (bp, w) in enumerate(zip(basis_lhe, coeffs)):
        kl, kt, c2, cg, c2g = bp
        dirname = (
            f"testrun_params_klambda_{kl}_ct_{kt}_ctt_{c2}"
            f"_cggh_{cg}_cgghh_{c2g}_energy_{energy_gev}_TeV"
        )
        root_path = os.path.join(EOS_BASE_PATH, dirname, "HH_variables", "output.root")
        if not os.path.exists(root_path):
            raise FileNotFoundError(
                f"[lhe_morph] basis {i} not found: {root_path}"
            )

        h_basis = get_mhh_from_root(root_path, reference_hist)
        if h_basis is None:
            raise RuntimeError(
                f"[lhe_morph] could not read mHH from basis {i}: {root_path}"
            )

        if h_morphed is None:
            h_morphed = h_basis.Clone("h_model_lhe")
            h_morphed.Reset()
            h_morphed.SetDirectory(0)

        h_morphed.Add(h_basis, w)

    # Zero out negative bins (can appear due to negative coefficients)
    if h_morphed is not None:
        for i in range(1, h_morphed.GetNbinsX() + 1):
            if h_morphed.GetBinContent(i) < 0:
                h_morphed.SetBinContent(i, 0)
                h_morphed.SetBinError(i, 0)

    return h_morphed


EOS_BASE_PATH = "/eos/user/e/emartinv/event_level_reweighting_HH/ggHH_basispoint_13_13p6"
PLOT_DIR = "triple_comparison"


def convert_to_dict(basis_list):
    result = {}
    for i, point in enumerate(basis_list):
        idx = str(i + 1)
        result[f"kl_{idx}"] = point[0]
        result[f"kt_{idx}"] = point[1]
        result[f"c2_{idx}"] = point[2]
        result[f"cg_{idx}"] = point[3]
        result[f"c2g_{idx}"] = point[4]
    return result

def scan_available_indices(energy='13p6'):
    energy_gev = 6800 if energy == '13p6' else 6500
    available = []
    for idx, pt in enumerate(pso_0):
        kl, kt, c2, cg, c2g = rw_to_lhe(*pt)
        dirname = (
            f"testrun_params_klambda_{kl}_ct_{kt}_ctt_{c2}"
            f"_cggh_{cg}_cgghh_{c2g}_energy_{energy_gev}_TeV"
        )
        root_path = os.path.join(EOS_BASE_PATH, dirname, "HH_variables", "output.root")
        if os.path.exists(root_path):
            available.append(idx)
        else:
            print(f"  [scan] idx={idx}: not found -> {root_path}")
    return available

def get_mhh_from_root(root_file_path, reference_hist):
    f = ROOT.TFile.Open(root_file_path, "READ")
    if not f or f.IsZombie():
        print(f"Error: Could not open file {root_file_path}")
        return None

    tree = f.Get("vars")
    if not tree:
        print(f"Error: Could not find tree 'vars' in {root_file_path}")
        f.Close()
        return None

    nbins = reference_hist.GetNbinsX()
    xaxis = reference_hist.GetXaxis()
    from array import array as c_array
    bin_edges = c_array('d', [xaxis.GetBinLowEdge(i) for i in range(1, nbins + 2)])

    h = ROOT.TH1D("h_mhh_lhe_tmp", "mHH from LHE", nbins, bin_edges)
    h.Sumw2()
    h.SetDirectory(ROOT.gDirectory)
    tree.Draw("mHH>>h_mhh_lhe_tmp", "", "goff")
    h.SetDirectory(0)

    h_clone = h.Clone(f"h_mhh_lhe_{os.path.basename(root_file_path)}_{id(h)}")
    h_clone.SetDirectory(0)
    f.Close()
    return h_clone


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------
def compare_triple(idx, save_plots=True, debug=False, energy='13p6'):
    """
      - from LHE              (red)
      - from reweight         (black)
      - from model (toys)     (blue)
      - from model (LHE)      (green)
    """
    if idx < 0 or idx >= len(pso_0):
        print(f"Error: Index {idx} out of range (0-{len(pso_0)-1})")
        return None

    from hyperevol.examples.mhh_scoring import load_coefficients
    load_coefficients(energy)

    energy_gev = 6800 if energy == '13p6' else 6500

    kl_rw, kt_rw, c2_rw, cg_rw, c2g_rw = pso_0[idx]
    kl_lhe, kt_lhe, c2_lhe, cg_lhe, c2g_lhe = rw_to_lhe(kl_rw, kt_rw, c2_rw, cg_rw, c2g_rw)

    print(f"\n{'='*80}")
    print(f"Comparing basis point idx={idx}  |  energy={energy} TeV")
    print(f"  Reweight : kl={kl_rw:.4f}, kt={kt_rw:.4f}, c2={c2_rw:.4f}, cg={cg_rw:.4f}, c2g={c2g_rw:.4f}")
    print(f"  LHE      : kl={kl_lhe:.4f}, kt={kt_lhe:.4f}, c2={c2_lhe:.4f}, cg={cg_lhe:.4f}, c2g={c2g_lhe:.4f}")
    print(f"{'='*80}")

    samplesize = 50000

    print("  Computing reweight distribution...")
    h_reweight = calcDist(kl_rw, kt_rw, c2_rw, cg_rw, c2g_rw, samplesize=samplesize)

    print("  Computing model distribution (pso_0)...")
    pso_0_dict = convert_to_dict(pso_0)
    h_model = calcDistModel(kl_rw, kt_rw, c2_rw, cg_rw, c2g_rw, pso_0_dict,
                            samplesize=samplesize, use_LO=True)

    dirname = (
        f"testrun_params_klambda_{kl_lhe}_ct_{kt_lhe}_ctt_{c2_lhe}"
        f"_cggh_{cg_lhe}_cgghh_{c2g_lhe}_energy_{energy_gev}_TeV"
    )
    lhe_path = os.path.join(EOS_BASE_PATH, dirname, "HH_variables", "output.root")

    if not os.path.exists(lhe_path):
        print(f"  Warning: LHE file not found: {lhe_path}")
        return None

    print(f"  Reading LHE file: {lhe_path}")
    h_lhe = get_mhh_from_root(lhe_path, h_reweight)
    if h_lhe is None:
        print("  Error: Could not read mHH from LHE file")
        return None

    print("  Computing model distribution from real LHE basis histograms...")
    h_model_lhe = get_lhe_morphed_distribution(
        list(rw_to_lhe(kl_rw, kt_rw, c2_rw, cg_rw, c2g_rw)),
        h_reweight, energy_gev
    )
    if h_model_lhe is None:
        print("  Warning: Could not build LHE-morphed distribution")

    # Normalise to same integral as LHE
    lhe_integral = h_lhe.Integral()
    if lhe_integral > 0:
        for h in [h_reweight, h_model] + ([h_model_lhe] if h_model_lhe else []):
            integral = h.Integral()
            if integral > 0:
                h.Scale(lhe_integral / integral)

    # ---------------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------------
    def ks(h1, h2):
        return h1.KolmogorovTest(h2, "")

    def chi2_reduced(h_pred, h_ref):
        chi2_val, ndof = 0.0, 0
        for i in range(h_ref.GetNbinsX()):
            pred = h_pred.GetBinContent(i + 1)
            ref  = h_ref.GetBinContent(i + 1)
            err_pred = h_pred.GetBinError(i + 1)
            err_ref  = h_ref.GetBinError(i + 1)
            denom = err_pred**2 + err_ref**2
            if denom > 0:
                chi2_val += (pred - ref)**2 / denom
                ndof += 1
        return chi2_val / ndof if ndof > 0 else 0.0

    def nll(h_pred, h_ref):
        nll_val = 0.0
        for i in range(h_ref.GetNbinsX()):
            m = h_pred.GetBinContent(i + 1)
            d = h_ref.GetBinContent(i + 1)
            eps = 1e-10
            if d > 0 and m > eps:
                nll_val += 2 * (m - d + d * math.log(d / m))
            elif d == 0 and m > 0:
                nll_val += 2 * m
        return nll_val

    ks_rw        = ks(h_lhe, h_reweight)
    ks_model     = ks(h_lhe, h_model)
    chi2_rw      = chi2_reduced(h_reweight, h_lhe)
    chi2_model   = chi2_reduced(h_model,    h_lhe)
    nll_rw       = nll(h_reweight, h_lhe)
    nll_model    = nll(h_model,    h_lhe)

    ks_model_lhe   = ks(h_lhe, h_model_lhe)   if h_model_lhe else float('nan')
    chi2_model_lhe = chi2_reduced(h_model_lhe, h_lhe) if h_model_lhe else float('nan')
    nll_model_lhe  = nll(h_model_lhe, h_lhe)  if h_model_lhe else float('nan')

    print(f"\n  Reweight:           KS={ks_rw:.4f},  chi2/ndf={chi2_rw:.2f},  NLL={nll_rw:.1f}")
    print(f"  Model pso_0 (toys): KS={ks_model:.4f}, chi2/ndf={chi2_model:.2f}, NLL={nll_model:.1f}")
    print(f"  Model pso_0 (LHE):  KS={ks_model_lhe:.4f}, chi2/ndf={chi2_model_lhe:.2f}, NLL={nll_model_lhe:.1f}")

    if debug:
        print(f"\n  Histogram stats (after normalisation to LHE):")
        print(f"    LHE            - mean={h_lhe.GetMean():.1f},  integral={h_lhe.Integral():.1f}")
        print(f"    Reweight       - mean={h_reweight.GetMean():.1f},  integral={h_reweight.Integral():.1f}")
        print(f"    Model toys     - mean={h_model.GetMean():.1f},  integral={h_model.Integral():.1f}")
        if h_model_lhe:
            print(f"    Model LHE      - mean={h_model_lhe.GetMean():.1f},  integral={h_model_lhe.Integral():.1f}")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    if save_plots:
        output_dir = os.path.join(PLOT_DIR, energy)
        os.makedirs(output_dir, exist_ok=True)

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[4, 1])

        # ---- main pad ----
        hep.histplot(h_lhe, histtype='band',
                     linewidth=3, alpha=0.3, color='tab:red', ax=ax)
        hep.histplot(h_lhe, histtype='errorbar',
                     label=r"from LHE",
                     linewidth=3, color='tab:red', ax=ax)

        hep.histplot(h_reweight, histtype='errorbar',
                     label=r"from reweight (KS={:.3f}, $\chi^2$/ndf={:.2f}, NLL={:.1f})".format(
                         ks_rw, chi2_rw, nll_rw),
                     linewidth=3, color='black', ax=ax)

        hep.histplot(h_model, histtype='band',
                     linewidth=3, alpha=0.5, color='tab:blue', ax=ax)
        hep.histplot(h_model, histtype='errorbar',
                     label=r"from model pso$\_$0 toys (KS={:.3f}, $\chi^2$/ndf={:.2f}, NLL={:.1f})".format(
                         ks_model, chi2_model, nll_model),
                     linewidth=3, color='tab:blue', ax=ax)

        if h_model_lhe is not None:
            hep.histplot(h_model_lhe, histtype='band',
                         linewidth=3, alpha=0.4, color='tab:green', ax=ax)
            hep.histplot(h_model_lhe, histtype='errorbar',
                         label=r"from model pso$\_$0 LHE (KS={:.3f}, $\chi^2$/ndf={:.2f}, NLL={:.1f})".format(
                             ks_model_lhe, chi2_model_lhe, nll_model_lhe),
                         linewidth=3, color='tab:green', ax=ax)

        # Title text (coupling values)
        title_text = (
            r"$\kappa_\lambda$={0:.2f}, $\kappa_t$={1:.2f}, $c_2$={2:.2f}, "
            r"$c_g$={3:.2f}, $c_{{2g}}$={4:.2f}"
        ).format(kl_rw, kt_rw, c2_rw, cg_rw, c2g_rw)
        ax.text(230, 1.025 * ax.get_ylim()[1], title_text, fontsize=14)

        lumi_label = r'138 fb$^{-1}$ (13.6 TeV)' if energy == '13p6' else r'138 fb$^{-1}$ (13 TeV)'
        hep.cms.lumitext(lumi_label, ax=ax)

        ax.legend(fontsize=9, loc='upper right')
        ax.set_ylabel("Events\n")
        ax.set_xlim(200, 1000)
        all_hists = [h_reweight, h_model, h_lhe] + ([h_model_lhe] if h_model_lhe else [])
        ymin = min(min(h.GetMinimum() for h in all_hists), 0)
        ax.set_ylim(ymin, ax.get_ylim()[1])
        ax.set_xticklabels([])

        # ---- ratio pad (X / LHE) ----
        ratio_rw = h_reweight.Clone("ratio_rw")
        ratio_rw.Divide(h_lhe)
        ratio_model = h_model.Clone("ratio_model")
        ratio_model.Divide(h_lhe)

        hep.histplot(ratio_rw, histtype='errorbar',
                     label=r"reweight / LHE",
                     linewidth=3, color='black', ax=ax2)

        hep.histplot(ratio_model, histtype='band',
                     linewidth=3, alpha=0.5, color='tab:blue', ax=ax2)
        hep.histplot(ratio_model, histtype='errorbar',
                     label=r"model toys / LHE",
                     linewidth=3, color='tab:blue', ax=ax2)

        if h_model_lhe is not None:
            ratio_model_lhe = h_model_lhe.Clone("ratio_model_lhe")
            ratio_model_lhe.Divide(h_lhe)
            hep.histplot(ratio_model_lhe, histtype='band',
                         linewidth=3, alpha=0.4, color='tab:green', ax=ax2)
            hep.histplot(ratio_model_lhe, histtype='errorbar',
                         label=r"model LHE / LHE",
                         linewidth=3, color='tab:green', ax=ax2)

        ax2.axhline(1, color='black', linewidth=2, linestyle='--')
        ax2.set_xlabel(r"$m_{HH}$ (GeV)")
        ax2.set_ylabel("Ratio\n")
        ax2.set_xlim(200, 1000)
        ax2.set_ylim(max(-2, ax2.get_ylim()[0]), min(5, ax2.get_ylim()[1]))

        plt.subplots_adjust(wspace=0, hspace=0.05)

        plotname = os.path.join(
            output_dir,
            f"triple_basis{idx}_kl{kl_rw:.2f}_kt{kt_rw:.2f}_c2{c2_rw:.2f}.png"
        )
        plt.savefig(plotname, dpi=300, bbox_inches='tight')
        print(f"  Plot saved: {plotname}")
        plt.close()

    return {
        'idx': idx,
        'kl': kl_rw, 'kt': kt_rw, 'c2': c2_rw, 'cg': cg_rw, 'c2g': c2g_rw,
        'ks_rw':         ks_rw,         'chi2_rw':         chi2_rw,         'nll_rw':         nll_rw,
        'ks_model':      ks_model,      'chi2_model':      chi2_model,      'nll_model':      nll_model,
        'ks_model_lhe':  ks_model_lhe,  'chi2_model_lhe':  chi2_model_lhe,  'nll_model_lhe':  nll_model_lhe,
    }


def compare_all(save_plots=True, debug=False, energy='13p6'):
    print(f"\nScanning for available LHE files at {energy} TeV ...")
    available = scan_available_indices(energy)
    print(f"Found {len(available)}/{len(pso_0)} basis points with LHE files.")

    results = []
    for idx in available:
        result = compare_triple(idx, save_plots=save_plots, debug=debug, energy=energy)
        if result:
            results.append(result)

    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        header = f"{'idx':>4}  {'kl':>7}  {'kt':>6}  {'c2':>6}  {'KS_rw':>8}  {'chi2_rw':>8}  {'KS_mToys':>9}  {'chi2_mT':>8}  {'KS_mLHE':>9}  {'chi2_mL':>8}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r['idx']:>4}  {r['kl']:>7.3f}  {r['kt']:>6.3f}  {r['c2']:>6.3f}"
                f"  {r['ks_rw']:>8.4f}  {r['chi2_rw']:>8.2f}"
                f"  {r['ks_model']:>9.4f}  {r['chi2_model']:>8.2f}"
                f"  {r['ks_model_lhe']:>9.4f}  {r['chi2_model_lhe']:>8.2f}"
            )

        avg_ks_rw        = np.mean([r['ks_rw']        for r in results])
        avg_ks_model     = np.mean([r['ks_model']     for r in results])
        avg_ks_model_lhe = np.mean([r['ks_model_lhe'] for r in results if not math.isnan(r['ks_model_lhe'])])
        print(f"\nAverage KS (reweight / LHE):         {avg_ks_rw:.4f}")
        print(f"Average KS (model toys / LHE):       {avg_ks_model:.4f}")
        print(f"Average KS (model LHE basis / LHE):  {avg_ks_model_lhe:.4f}")

    return results

if __name__ == "__main__":
    debug  = "--debug" in sys.argv or "-d" in sys.argv
    sys.argv = [a for a in sys.argv if a not in ("--debug", "-d")]

    energy = '13p6'
    if "--energy" in sys.argv:
        i = sys.argv.index("--energy")
        if i + 1 < len(sys.argv):
            energy = sys.argv[i + 1]
            sys.argv = sys.argv[:i] + sys.argv[i + 2:]
    elif "--energy=13p6" in sys.argv or "-e13p6" in sys.argv:
        energy = '13p6'
        sys.argv = [a for a in sys.argv if a not in ("--energy=13p6", "-e13p6")]
    elif "--energy=13" in sys.argv or "-e13" in sys.argv:
        energy = '13'
        sys.argv = [a for a in sys.argv if a not in ("--energy=13", "-e13")]

    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        compare_triple(idx, save_plots=True, debug=debug, energy=energy)
    else:
        compare_all(save_plots=True, debug=debug, energy=energy)
