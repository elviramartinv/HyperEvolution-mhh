#!/usr/bin/env python3
"""
Script to compare mHH distributions from two sources:
1. "from reweight" using calcDist (analytical reweighting)
2. LHE events from ROOT files

The goal is to verify the coupling convention conversion:
    cg_LHE = cg_reweight / 1.5
    c2g_LHE = -c2g_reweight / 3
"""

import ROOT
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
plt.style.use(hep.style.CMS)
import os
import sys
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hyperevol.examples.mhh_scoring import calcDist

# Basis points in reweight convention (from plotCompare.py)
psotemp_reweight = [
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

# Basis points in LHE convention (from LHE files)
psotemp_LHE = [
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

EOS_BASE_PATH = "/eos/user/e/emartinv/event_level_reweighting_HH/ggHH_basispoint_13_13p6"
PLOT_DIR = "reweight_LHE_comparison_plots"

def find_lhe_file(kl, kt, c2, cg, c2g, energy=6800):
    dirname = f"testrun_params_klambda_{kl}_ct_{kt}_ctt_{c2}_cggh_{cg}_cgghh_{c2g}_energy_{energy}_TeV"
    full_path = os.path.join(EOS_BASE_PATH, dirname, "HH_variables", "output.root")

    if os.path.exists(full_path):
        return full_path
    else:
        print(f"Warning: File not found: {full_path}")
        return None

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

    from array import array
    bin_edges = array('d', [xaxis.GetBinLowEdge(i) for i in range(1, nbins + 2)])

    h = ROOT.TH1D("h_mhh_lhe", "mHH from LHE", nbins, bin_edges)
    h.Sumw2()

    tree.Draw("mHH>>h_mhh_lhe", "", "goff")

    h_clone = h.Clone(f"h_mhh_lhe_{os.path.basename(root_file_path)}")
    h_clone.SetDirectory(0)

    f.Close()
    return h_clone

def compare_distributions(idx, save_plots=True, debug=False, energy='13p6'):
    if idx < 0 or idx >= len(psotemp_reweight):
        print(f"Error: Index {idx} out of range (0-{len(psotemp_reweight)-1})")
        return None

    # Load coefficients for specified energy
    from hyperevol.examples.mhh_scoring import load_coefficients
    load_coefficients(energy)

    # Set energy in GeV for LHE files
    energy_gev = 6800 if energy == '13p6' else 6500

    # Reweight convention
    kl_rw, kt_rw, c2_rw, cg_rw, c2g_rw = psotemp_reweight[idx]

    # LHE couplings
    kl_exp, kt_exp, c2_exp, cg_exp, c2g_exp = psotemp_LHE[idx]

    print(f"\n{'='*80}")
    print(f"Comparing basis point {idx} at {energy} TeV")
    print(f"{'='*80}")
    print(f"Reweight convention: kl={kl_rw:.4f}, kt={kt_rw:.4f}, c2={c2_rw:.4f}, cg={cg_rw:.4f}, c2g={c2g_rw:.4f}")
    print(f"LHE convention:      kl={kl_exp:.4f}, kt={kt_exp:.4f}, c2={c2_exp:.4f}, cg={cg_exp:.4f}, c2g={c2g_exp:.4f}")

    print("\nDistribution from reweight...")
    h_reweight = calcDist(kl_rw, kt_rw, c2_rw, cg_rw, c2g_rw, samplesize=50000)

    lhe_file = find_lhe_file(kl_exp, kt_exp, c2_exp, cg_exp, c2g_exp, energy=energy_gev)
    if not lhe_file:
        print("Error: Could not find LHE file")
        return None

    print(f"Reading LHE file: {lhe_file}")
    h_lhe = get_mhh_from_root(lhe_file, h_reweight)
    if not h_lhe:
        print("Error: Could not read mHH from LHE file")
        return None

    # Normalize LHE histogram to match reweight scale (events)
    # reweight is already scaled to 138 fb^-1
    lhe_integral = h_lhe.Integral()
    rw_integral = h_reweight.Integral()

    if lhe_integral > 0:
        scale_factor = rw_integral / lhe_integral
        h_lhe.Scale(scale_factor)
        print(f"\nScaled LHE histogram by {scale_factor:.4f} to match reweight normalization")

    if debug:
        print("\n" + "="*80)
        print("KS TEST DETAILS")
        print("="*80)

        print(f"\nHistogram statistics:")
        print(f"  Reweight - Mean: {h_reweight.GetMean():.2f}, RMS: {h_reweight.GetRMS():.2f}, Integral: {h_reweight.Integral():.2f}")
        print(f"  LHE      - Mean: {h_lhe.GetMean():.2f}, RMS: {h_lhe.GetRMS():.2f}, Integral: {h_lhe.Integral():.2f}")

        print(f"\nCumulative Distribution Functions (CDF) at selected points:")
        print(f"{'Bin':<5} {'mHH':<10} {'CDF_rw':<12} {'CDF_lhe':<12} {'|Diff|':<10}")
        print("-" * 55)

        max_diff = 0
        max_diff_bin = 0

        for i in range(0, h_reweight.GetNbinsX(), max(1, h_reweight.GetNbinsX()//10)):
            bin_center = h_reweight.GetBinCenter(i+1)

            cdf_rw = sum([h_reweight.GetBinContent(j+1) for j in range(i+1)]) / h_reweight.Integral()
            cdf_lhe = sum([h_lhe.GetBinContent(j+1) for j in range(i+1)]) / h_lhe.Integral()
            diff = abs(cdf_rw - cdf_lhe)

            print(f"{i+1:<5} {bin_center:<10.1f} {cdf_rw:<12.6f} {cdf_lhe:<12.6f} {diff:<10.6f}")

            if diff > max_diff:
                max_diff = diff
                max_diff_bin = i+1

        print(f"\nMaximum CDF difference: {max_diff:.6f} at bin {max_diff_bin} (mHH ≈ {h_reweight.GetBinCenter(max_diff_bin):.1f} GeV)")
        print(f"\nThis maximum difference D = {max_diff:.6f} is converted to a p-value by ROOT's KS test.")
        print(f"The p-value represents the probability that two samples from the same distribution")
        print(f"would have a KS statistic at least as large as the observed one.\n")

    ks = h_reweight.KolmogorovTest(h_lhe, "")

    # chi2
    chi2 = 0
    ndof = 0
    for i in range(h_reweight.GetNbinsX()):
        val_rw = h_reweight.GetBinContent(i+1)
        val_lhe = h_lhe.GetBinContent(i+1)
        err_rw = h_reweight.GetBinError(i+1)
        err_lhe = h_lhe.GetBinError(i+1)

        if err_rw**2 + err_lhe**2 > 0:
            chi2 += (val_rw - val_lhe)**2 / (err_rw**2 + err_lhe**2)
            ndof += 1

    print(f"\nComparison statistics:")
    print(f"  KS test p-value: {ks:.4f}")
    print(f"  Chi2/ndof: {chi2:.2f}/{ndof} = {chi2/ndof if ndof > 0 else 0:.2f}")

    # Comparison plot
    if save_plots:
        # Create energy-specific output directory
        output_dir = os.path.join(PLOT_DIR, energy)
        os.makedirs(output_dir, exist_ok=True)

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[4, 1])

        hep.histplot(h_reweight, histtype='errorbar', label='From reweight',
                     linewidth=2, color='black', ax=ax)
        hep.histplot(h_lhe, histtype='errorbar', label='From LHE events',
                     linewidth=2, color='red', ax=ax)

        ax.legend(loc='upper right')
        ax.set_ylabel("Events")
        ax.set_xlim(200, 1000)

        # Couplings text with both conventions for cg and c2g
        # First part: kl, kt, c2 (only reweight values)
        text_base = (r"$\kappa_\lambda$={0:.2f}, $\kappa_t$={1:.2f}, $c_2$={2:.2f}, ".format(
            kl_rw, kt_rw, c2_rw))

        text_y = 1.025*ax.get_ylim()[1]

        # Display base text (kl, kt, c2)
        text_obj = ax.text(200, text_y, text_base, fontsize=14, verticalalignment='bottom')
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        base_width = bbox.transformed(ax.transData.inverted()).width

        # Add cg with both values (black for reweight, red for LHE)
        x_pos = 200 + base_width
        cg_label = r"$c_g$="
        text_obj = ax.text(x_pos, text_y, cg_label, fontsize=14,
                          verticalalignment='bottom', color='black')
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        cg_label_width = bbox.transformed(ax.transData.inverted()).width
        x_pos += cg_label_width

        # Black value (from reweight)
        cg_rw_text = f"{cg_rw:.2f}"
        text_obj = ax.text(x_pos, text_y, cg_rw_text, fontsize=14,
                          verticalalignment='bottom', color='black')
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        cg_rw_width = bbox.transformed(ax.transData.inverted()).width
        x_pos += cg_rw_width

        # Slash
        text_obj = ax.text(x_pos, text_y, "/", fontsize=14,
                          verticalalignment='bottom', color='black')
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        slash_width = bbox.transformed(ax.transData.inverted()).width
        x_pos += slash_width

        # Red value (from LHE)
        cg_lhe_text = f"{cg_exp:.2f}"
        text_obj = ax.text(x_pos, text_y, cg_lhe_text, fontsize=14,
                          verticalalignment='bottom', color='red')
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        cg_lhe_width = bbox.transformed(ax.transData.inverted()).width
        x_pos += cg_lhe_width

        # Comma and space
        text_obj = ax.text(x_pos, text_y, ", ", fontsize=14,
                          verticalalignment='bottom', color='black')
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        comma_width = bbox.transformed(ax.transData.inverted()).width
        x_pos += comma_width

        # Add c2g with both values (black for reweight, red for LHE)
        c2g_label = r"$c_{2g}$="
        text_obj = ax.text(x_pos, text_y, c2g_label, fontsize=14,
                          verticalalignment='bottom', color='black')
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        c2g_label_width = bbox.transformed(ax.transData.inverted()).width
        x_pos += c2g_label_width

        # Black value (from reweight)
        c2g_rw_text = f"{c2g_rw:.2f}"
        text_obj = ax.text(x_pos, text_y, c2g_rw_text, fontsize=14,
                          verticalalignment='bottom', color='black')
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        c2g_rw_width = bbox.transformed(ax.transData.inverted()).width
        x_pos += c2g_rw_width

        # Slash
        text_obj = ax.text(x_pos, text_y, "/", fontsize=14,
                          verticalalignment='bottom', color='black')
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        slash2_width = bbox.transformed(ax.transData.inverted()).width
        x_pos += slash2_width

        # Red value (from LHE)
        c2g_lhe_text = f"{c2g_exp:.2f}"
        text_obj = ax.text(x_pos, text_y, c2g_lhe_text, fontsize=14,
                          verticalalignment='bottom', color='red')

        # Set lumitext based on energy
        if energy == '13':
            hep.cms.lumitext(r'138 fb$^{-1}$ (13 TeV)', ax=ax)
        else:  # 13p6
            hep.cms.lumitext(r'138 fb$^{-1}$ (13.6 TeV)', ax=ax)

        # Ratio plot
        ratio = h_lhe.Clone("ratio")
        ratio.Divide(h_reweight)
        hep.histplot(ratio, histtype='errorbar', linewidth=2, color='red', ax=ax2)
        ax2.axhline(1, color='black', linewidth=2, linestyle='--')
        ax2.set_xlabel(r"$m_{HH}$ (GeV)")
        ax2.set_ylabel("Ratio")
        ax2.set_ylim(0.5, 1.5)
        ax2.set_xlim(200, 1000)

        ax.set_xticklabels([])
        plt.subplots_adjust(wspace=0, hspace=0.05)

        plotname = os.path.join(output_dir, f"comparison_basis_{idx}_kl{kl_rw:.2f}_kt{kt_rw:.2f}.png")
        plt.savefig(plotname, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {plotname}")
        plt.close()

    results = {
        'idx': idx,
        'kl': kl_rw,
        'kt': kt_rw,
        'c2': c2_rw,
        'cg_rw': cg_rw,
        'c2g_rw': c2g_rw,
        'cg_lhe': cg_exp,
        'c2g_lhe': c2g_exp,
        'ks': ks,
        'chi2': chi2,
        'ndof': ndof,
        'chi2_ndof': chi2/ndof if ndof > 0 else 0
    }

    return results

def compare_all(save_plots=True, debug=False, energy='13p6'):
    results = []

    print("\n" + "="*80)
    print(f"COMPARING ALL BASIS POINTS at {energy} TeV")
    print("="*80)

    for idx in range(len(psotemp_reweight)):
        result = compare_distributions(idx, save_plots=save_plots, debug=debug, energy=energy)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nDistribution comparison statistics:")
    avg_ks = np.mean([r['ks'] for r in results])
    avg_chi2_ndof = np.mean([r['chi2_ndof'] for r in results])
    print(f"  Average KS test p-value: {avg_ks:.4f}")
    print(f"  Average Chi2/ndof: {avg_chi2_ndof:.2f}")

    return results

if __name__ == "__main__":
    import sys

    debug = "--debug" in sys.argv or "-d" in sys.argv
    sys.argv = [arg for arg in sys.argv if arg not in ["--debug", "-d"]]

    # Parse energy argument (default: 13p6)
    energy = '13p6'
    if "--energy=13" in sys.argv or "-e13" in sys.argv:
        energy = '13'
        sys.argv = [arg for arg in sys.argv if arg not in ["--energy=13", "-e13"]]
    elif "--energy=13p6" in sys.argv or "-e13p6" in sys.argv:
        energy = '13p6'
        sys.argv = [arg for arg in sys.argv if arg not in ["--energy=13p6", "-e13p6"]]

    if len(sys.argv) > 1:
        # Compare specific basis point
        idx = int(sys.argv[1])
        compare_distributions(idx, save_plots=True, debug=debug, energy=energy)
    else:
        # Compare all basis points
        compare_all(save_plots=True, debug=debug, energy=energy)
