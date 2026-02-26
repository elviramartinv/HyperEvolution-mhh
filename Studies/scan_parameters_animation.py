#!/usr/bin/env python3
"""
Scan each parameter (kl, kt, c2, cg, c2g) across its full range while keeping
others fixed at SM values, and create animations showing how mHH distributions
change.

Usage:
    python scan_parameters_animation.py --output-dir results/parameter_scan --step 0.1
    python scan_parameters_animation.py --help

"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

import ROOT
ROOT.TH1.AddDirectory(False)

try:
    from mhh_scoring import calcDist, calcDistModel
except ImportError:
    from hyperevol.examples.mhh_scoring import calcDist, calcDistModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
    [5.76306775897426, 0.60957255930352, 0.2757471461722233, 0.3539034679345107, 1.6470864190895298],
    [-3.264926168334715, 3.951560064234396, 5.0, 0.07060844214613118, 4.0],
    [-3.66094197241397, 1.0779919430503124, 1.1444876325325517, -4.543414129193672, -2.086324376343318],
    [13.776142713343612, 6.0, 4.8, -2.2471194229079066, 2.914575463372937],
    [-0.5102446669232454, -4.0, 5.7, -1.8918011813608135, 2.523631269874509],
    [-17.5, -2.2192457731974353, -1.561925426134284, 3.587673607734981, -1.9097200749524326],
    [-17.248692794237684, 0.25541740841360383, -1.0545614554215015, 5.0, -2.360300550565078],
    [-6.167280087615459, 2.898361032670763, 3.1047347943825936, 2.6888845448700023, -0.7993986852185984],
    [-8.42531627433404, -2.7668781210641837, -4.981786788480192, 1.534339186975958, -0.9216953993789947],
    [-8.858010458211641, -3.798082301577028, 1.8756163268862356, 1.512411203148254, 2.2487785553632356],
    [11.02650511878629, 2.9632586454393097, 1.348357475088319, -3.2235380258824056, 3.9176005249108194],
    [11.229088936147944, -1.5731677725730204, 1.5331118641968002, 4.285894664167079, 0.8707429805808593],
    [7.50077391701546, 0.6423086612184954, 4.2702578284904495, 1.765297528403499, -0.6817536396734726],
    [4.815076407029188, 3.854636736030917, 2.94968225346022, 6.0, -0.9731513842322967],
    [-5.390428177680153, 0.08809131336044707, -0.6500408810826548, 9.34997253744265, 1.6513702705760684],
    [15.428842238949272, 5.250828933568307, 14.623892303287555, -3.600536082070807, 1.9429364946062562],
    [0.7200237879529017, -0.1848591996030451, 4.340128637534834, 6.0, -1.5721225585844174],
    [0.7742911431843156, 4.294395624534779, 3.5870500704526624, -1.4993240151206564, 0.24039833217398765],
    [0.8282652927738547, -3.682996998942519, 5.0, 1.294025367125575, 0.44492459625784425],
    [6.156700174967869, 4.78800924152869, 4.441178744579316, 2.0033997687370015, 1.3977761733162892],
    [35.0, 3.465547133658958, -1.9566635858237562, 0.1285393896793815, 5.920278547684675],
    [9.270557685309987, 0.43323227168253386, 1.0112324652261968, -2.106083384790988, -2.3123399950800057],
    [3.5041663192328274, 3.593645251058606, 2.952695773181962, 5.0, 2.6889000092102133],
]

PARAM_NAMES = ["kl", "kt", "c2", "cg", "c2g"]
PARAM_LABELS_LATEX = [
    r"$\kappa_\lambda$",
    r"$\kappa_t$",
    r"$c_2$",
    r"$c_g$",
    r"$c_{2g}$",
]

SM_VALUES = {"kl": 1.0, "kt": 1.0, "c2": 0.0, "cg": 0.0, "c2g": 0.0}

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


def convert_to_dict(basis_list):
    result = {}
    for i, point in enumerate(basis_list):
        idx = str(i + 1)
        result[f"kl_{idx}"]  = point[0]
        result[f"kt_{idx}"]  = point[1]
        result[f"c2_{idx}"]  = point[2]
        result[f"cg_{idx}"]  = point[3]
        result[f"c2g_{idx}"] = point[4]
    return result


def compute_distributions(kl, kt, c2, cg, c2g, pso0_dict, psotemp_dict, samplesize=50000):
    reweight      = calcDist(kl, kt, c2, cg, c2g, samplesize=samplesize)
    model_pso0    = calcDistModel(kl, kt, c2, cg, c2g, pso0_dict,   samplesize=samplesize, use_LO=True)
    model_psotemp = calcDistModel(kl, kt, c2, cg, c2g, psotemp_dict, samplesize=samplesize, use_LO=False)

    return reweight, model_pso0, model_psotemp


def scan_parameter(param_name, param_idx, limits, step, sm_values, output_dir, pso0_dict, psotemp_dict):
    """
    Scan one parameter while keeping others at SM values.
    """
    print(f"\n=== Scanning {param_name} ===")

    param_min, param_max = limits[param_name]
    values = np.arange(param_min, param_max + step, step)

    results = []
    frames_dir = os.path.join(output_dir, f"frames_{param_name}")
    os.makedirs(frames_dir, exist_ok=True)

    for i, value in enumerate(values):
        # Set parameters: scan parameter varies, others at SM
        params = sm_values.copy()
        params[param_name] = value

        kl  = params["kl"]
        kt  = params["kt"]
        c2  = params["c2"]
        cg  = params["cg"]
        c2g = params["c2g"]

        print(f"  [{i+1}/{len(values)}] {param_name}={value:.2f} "
              f"(kl={kl:.2f}, kt={kt:.2f}, c2={c2:.2f}, cg={cg:.2f}, c2g={c2g:.2f})")

        try:
            reweight, pso0, psotemp = compute_distributions(
                kl, kt, c2, cg, c2g, pso0_dict, psotemp_dict
            )
            results.append((value, reweight, pso0, psotemp))

            # Save individual frame
            save_frame(param_name, param_idx, value, reweight, pso0, psotemp,
                      frames_dir, i, params)

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    return results, values


def save_frame(param_name, param_idx, param_value, reweight, pso0, psotemp,
               output_dir, frame_num, all_params):
    """Save a single frame as PNG."""
    import math

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[4, 1])

    # Plot distributions
    hep.histplot(reweight, histtype="errorbar", label="from reweight",
                 linewidth=3, density=False, color="black", ax=ax)

    models = [pso0, psotemp]
    names = ["pso_0", "psotemp"]
    colors = ["tab:orange", "tab:green"]

    ndf = reweight.GetNbinsX()

    for model, name, col in zip(models, names, colors):
        # Compute metrics
        chi2_val = nll_val = 0.0
        for b in range(model.GetNbinsX()):
            mc = model.GetBinContent(b + 1)
            rc = reweight.GetBinContent(b + 1)
            re = reweight.GetBinError(b + 1)
            diff = mc - rc
            chi2_val += diff**2 / re**2 if re > 0 else 0
            eps = 1e-10
            if rc > 0 and mc > eps:
                nll_val += 2 * (mc - rc + rc * math.log(rc / mc))
            elif rc == 0 and mc > 0:
                nll_val += 2 * mc

        ks_val = reweight.KolmogorovTest(model, "")
        chi2_ndf = chi2_val / ndf

        lbl = f"{name} (KS={ks_val:.3f}, χ²/ndf={chi2_ndf:.2f}, NLL={nll_val:.1f})"
        hep.histplot(model, histtype="band", linewidth=3, density=False,
                     alpha=0.5, color=col, ax=ax)
        hep.histplot(model, histtype="errorbar", linewidth=3, density=False,
                     label=lbl, color=col, ax=ax)

        ratio = model.Clone(model.GetName() + "_ratio")
        ratio.Divide(reweight)
        hep.histplot(ratio, histtype="band", linewidth=3, density=False,
                     alpha=0.5, color=col, ax=ax2)
        hep.histplot(ratio, histtype="errorbar", linewidth=3, density=False,
                     color=col, ax=ax2)

    hep.cms.lumitext(r'138 fb$^{-1}$ (13.6 TeV)', ax=ax)

    # Title showing current parameter value and all parameters
    title = (f"Scanning {PARAM_LABELS_LATEX[param_idx]} = {param_value:.2f}\n"
             f"kl={all_params['kl']:.2f}, kt={all_params['kt']:.2f}, "
             f"c2={all_params['c2']:.2f}, cg={all_params['cg']:.2f}, c2g={all_params['c2g']:.2f}")

    ax.legend(fontsize=8, loc='upper right')

    ax.text(0.98, 0.68, title, transform=ax.transAxes, fontsize=10,
            va="top", ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xticklabels([])
    ax.set_ylabel("Events")
    ax.set_xlim(200, 1000)
    ymin = min(pso0.GetMinimum(), psotemp.GetMinimum(), reweight.GetMinimum(), 0)
    ax.set_ylim(ymin, ax.get_ylim()[1])

    ax2.set_xlabel(r"$m_{HH}$ (GeV)")
    ax2.set_ylabel("ratio")
    ax2.set_xlim(200, 1000)
    ax2.set_ylim(max(-2, ax2.get_ylim()[0]), min(5, ax2.get_ylim()[1]))
    ax2.axhline(1, linewidth=2, color="black")
    plt.subplots_adjust(wspace=0, hspace=0.1)

    fname = os.path.join(output_dir, f"frame_{frame_num:04d}.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(fig)


def create_animation(param_name, frames_dir, output_dir):
    """Create animated GIF from saved frames."""
    import glob
    from PIL import Image

    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))

    if not frame_files:
        print(f"  No frames found for {param_name}")
        return

    print(f"  Creating animation from {len(frame_files)} frames...")

    # Load images
    images = [Image.open(f) for f in frame_files]

    # Save as GIF
    output_file = os.path.join(output_dir, f"scan_{param_name}.gif")
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=500,  # milliseconds per frame (0.5 seconds)
        loop=0
    )

    print(f"  Saved animation: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan each parameter while keeping others at SM values and create animations."
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Output directory for frames and animations."
    )
    parser.add_argument(
        "--step", type=float, default=0.1,
        help="Step size for parameter scan (default: 0.1)."
    )
    parser.add_argument(
        "--config", default=_DEFAULT_CONFIG,
        help="Path to parameter-range JSON (default: hyperevol/examples/config/mhh_23.json)."
    )
    parser.add_argument(
        "--parameters", nargs="+", default=None,
        help="Specific parameters to scan (default: all). E.g., --parameters kl kt"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    limits = load_limits(args.config)
    print("Parameter limits:")
    for k, (lo, hi) in limits.items():
        print(f"  {k}: [{lo}, {hi}]")

    pso0_dict = convert_to_dict(pso_0)
    psotemp_dict = convert_to_dict(psotemp)

    params_to_scan = args.parameters if args.parameters else PARAM_NAMES

    print(f"\nScanning parameters: {', '.join(params_to_scan)}")
    print(f"Step size: {args.step}")
    print(f"SM values: {SM_VALUES}\n")

    for param_name in params_to_scan:
        if param_name not in PARAM_NAMES:
            print(f"WARNING: Unknown parameter '{param_name}', skipping.")
            continue

        param_idx = PARAM_NAMES.index(param_name)

        results, values = scan_parameter(
            param_name, param_idx, limits, args.step, SM_VALUES,
            args.output_dir, pso0_dict, psotemp_dict
        )

        if results:
            # Create animation
            frames_dir = os.path.join(args.output_dir, f"frames_{param_name}")
            create_animation(param_name, frames_dir, args.output_dir)
        else:
            print(f"  No successful scans for {param_name}")

    print("\n=== Done ===")
    print(f"Animations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
