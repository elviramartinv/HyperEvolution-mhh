
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
})
# import cmsstyle  # Not needed, will use seaborn colors instead
import sympy
import math
import ROOT
ROOT.TH1.AddDirectory(False)
from plothist import make_hist, plot_error_hist
import mplhep as hep
plt.style.use(hep.style.CMS)
import seaborn as sns
vlag = sns.color_palette("Spectral_r", as_cmap=True)
petroff_10 = sns.color_palette("tab10", 10)  # Alternative color palette
from mhh_scoring import calcDist, calcDistModel, makeKey


# compare toy based distribution with linear combination
def plotCompare(kl, kt, c2, cg, c2g, inputs, makeplot=True, drawmodel=True, lim=True, inputnames=["1"], debug=False, highstat=False, use_LO=None):
    samplesize = 50000
    if highstat: samplesize*=10

    if use_LO is None:
        # Auto-detect: use LO if "atlas" or "pso_0" in name
        use_LO_list = [("atlas" in name.lower() or "pso_0" in name.lower()) for name in inputnames]
    elif isinstance(use_LO, bool):
        use_LO_list = [use_LO] * len(inputs)
    else:
        use_LO_list = use_LO

    colors=['tab:blue', 'tab:orange', 'tab:green']
    if len(colors) == 0:
        colors = [petroff_10[i] for i in range(len(inputs))]
    elif len(colors) < len(inputs):
        colors = list(colors) + [petroff_10[i] for i in range(len(colors), len(inputs))]

    model = calcDistModel(kl, kt, c2, cg, c2g, inputs[0], samplesize=samplesize, use_LO=use_LO_list[0])
    reweight = calcDist(kl, kt, c2, cg, c2g, samplesize=samplesize)
    if lim == False:
        model = calcDistModel(kl, kt, c2, cg, c2g, inputs[0], samplesize=samplesize+1, use_LO=use_LO_list[0])
        reweight = calcDist(kl, kt, c2, cg, c2g, samplesize=samplesize+1)
    othermodels = [model]
    for im,m in enumerate(inputs):
        if im!=0:
            othermodels.append(calcDistModel(kl, kt, c2, cg, c2g, m, samplesize=samplesize, use_LO=use_LO_list[im]))
    ks =  reweight.KolmogorovTest(model, "")
    stat = sum([abs(model.GetBinError(i+1)/(model.GetBinContent(i+1)+0.0000001)) for i in range(model.GetNbinsX())])
    statref = sum([abs(reweight.GetBinError(i+1)/(reweight.GetBinContent(i+1)+0.0000001)) for i in range(reweight.GetNbinsX())])
    if debug:
        stat2 = [abs(model.GetBinError(i+1)/(model.GetBinContent(i+1)+0.0000001)) for i in range(model.GetNbinsX())]
        statref2 = [abs(reweight.GetBinError(i+1)/(reweight.GetBinContent(i+1)+0.0000001)) for i in range(reweight.GetNbinsX())]
        print( list(s1/(statref2[is1]+0.0000001) for is1, s1 in enumerate(stat2)))
    chi2 = 0
    for i in range(model.GetNbinsX()):
        chi2 += math.pow((model.GetBinContent(i+1)-reweight.GetBinContent(i+1)),2)/(math.pow(model.GetBinError(i+1),2)+math.pow(reweight.GetBinError(i+1),2))
    if makeplot:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[4, 1])
        d=False
        hep.histplot(reweight, histtype='errorbar', label=r"from reweight", linewidth=3, density=d, color="black",ax=ax)
        #ax.set_ylim( ax.get_ylim()[0], 1.2* ax.get_ylim()[1])
        if drawmodel:
            for im,m in enumerate(othermodels):
                hep.histplot(m, histtype='band', linewidth=3, density=d, alpha=0.5, color=colors[im], ax=ax)
                hep.histplot(m, histtype='errorbar', linewidth=3, density=d, label=r"from model {}".format(inputnames[im]),color=colors[im],ax=ax)

        #hep.cms.label(rlabel="")
        hep.cms.lumitext(r'138 fb$^{-1}$ (13 TeV)', ax=ax)

        if drawmodel:
            title_text = r"$\kappa_\lambda$={0}, $\kappa_t$={1}, $c_2$={2}, $c_g$={3}, $c_{{2g}}$={4}, KS={5}".format(kl, kt, c2, cg, c2g, round(ks,3))
            ax.text(230, 1.025*ax.get_ylim()[1], title_text, fontsize=15)
        ax.legend()

        if drawmodel:
            for im,m in enumerate(othermodels):
                rationame = m.GetName()+"_ratio"
                ratio = m.Clone("rationame")
                ratio.Divide(reweight)
                hep.histplot(ratio, histtype='band', linewidth=3, density=d, alpha=0.5, color=colors[im], ax=ax2)
                hep.histplot(ratio, histtype='errorbar', linewidth=3, density=d, label=r"from model {}".format(inputnames[im]),color=colors[im],ax=ax2)

        ax.set_xticklabels([])
        plt.subplots_adjust(wspace=0, hspace=0.1)
        ax2.set_xlabel(r"$m_{HH}$ (GeV)")
        ax.set_ylabel("Events\n")
        ax2.set_ylabel("ratio\n")
        if lim:
            ax.set_xlim(200, 1000)
            ax2.set_xlim(200, 1000)
        ymin = min(min(model.GetMinimum(), reweight.GetMinimum()),0)
        ax.set_ylim(ymin, ax.get_ylim()[1])
        ax2.set_ylim(max(-2,ax2.get_ylim()[0]),min(5,ax2.get_ylim()[1]))
        ax2.axline((ax2.get_xlim()[0], 1), (ax2.get_xlim()[1], 1), linewidth=3, color='black')
        if lim ==False: ax.set_xscale('log')

        # Save the plot
        import os
        output_dir = "plot_compare/pso_23_1_condor_13p6coffs"
        os.makedirs(output_dir, exist_ok=True)
        plotname = os.path.join(output_dir, f"compare_kl{kl}_kt{kt}_c2{c2}_cg{cg}_c2g{c2g}.png")
        plt.savefig(plotname, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plotname}")

    return ks, stat, statref, chi2

# Basis
pso_0 = [[2.213221919449151, -2.6934464500856405, 2.965671189693319, 1.1360050891505196, -2.1481034846500293], [9.76060116521356, -4.0, 3.0808116991517873, 4.979656543379024, 1.7948537607896533], [-8.504037399703757, 2.4570674701218005, 2.0456701185442197, 4.993273200619232, 0.31806830918129725], [12.42168994697454, 4.8885158267788515, 1.1736452461594093, -1.649885452005208, 2.9957802625445793], [-8.028050651896733, 1.4185892815035221, 4.637324985625828, 1.100571508815925, -2.141440691112374], [-0.6417775386928877, -3.4585451693407876, 1.5364031301770482, 1.449765057791862, -3.0], [4.516788454318963, 3.246315195973261, 0.03179409373963871, 4.515743016527684, -3.0], [6.776972542859125, 2.7536031422527922, 1.1291947583909658, -4.998564441693077, -3.530381379838789], [-7.295476241252043, -3.567728937382591, 4.927055574269499, 5.4698157741693745, 0.5460308050762466], [10.16791838833636, 3.168115455700297, 3.988981859905667, -3.9955456956734263, 2.5047215502921873], [-9.237668351371102, -0.8830087027316762, 1.507497463102394, -1.656941023505912, 1.0373600385652952], [15.871698091121043, 2.954226053612609, -1.4832395231493958, -4.648315411280079, -0.31921721853099594], [-6.0920951495343445, -1.2529968110851288, -0.661498001605271, 3.1356841635278725, 2.473949503061431], [-13.765201883693987, -1.4548873762027377, 2.2019807467420858, 6.0, 2.0762352620706594], [-13.956921393217277, -3.867174172277096, 2.121296077606367, 3.066609957858563, 1.9738262253820915], [8.2879617206356, -4.0, 9.611068631424631, 3.4554065505161273, 0.9602330195006838], [-3.368487622557269, -1.235213414828737, -1.3214826590555613, 5.485928293425935, -1.6308816657564937], [-1.1492785231960068, 0.09464020330641909, 3.076473122787405, -3.6375285759285543, 1.3735185288149214], [17.4, 1.8726121309906936, 0.2733568393951952, -2.568047729842719, -2.85757278314612], [9.019237551902098, 0.9358945607455986, -4.368407508947331, 3.9318179425747295, 2.72703446921505], [10.092619602076475, -0.25598532434815535, 0.5458135306798126, 1.061229295632444, 5.764408396915333], [-7.594748729686678, -3.5504975021060012, 0.9679708236284144, -0.5666309250467718, 2.8180805513778475], [-2.167934532063678, -0.46393247419740047, 3.091840758463215, 2.0233283474829102, -1.2791407684980047]]
psotemp = [
    [3.4172559352759126, -5.0, -1.44668802839265, -0.781558122576403, -0.7133515290520358],
    [13.18210896486622, -4.347950455100658, 2.49351576542362, 1.039032749789954, -1.609129718460307],
    [5.827899874452427, -3.2137652170105655, 2.840339092461808, 1.1934690778040968, -1.1538081916911085],
    [-15.0, -2.547662754634844, 5.0, 2.0577295513962732, 0.9068664277305499],
    [-11.258856337614343, 0.45111382030777314, 2.525899901478942, -3.3746894961475826, 2.435336881684718],
    [-15.0, 3.2703884367347964, 2.166330879716739, 2.570970985717951, 0.47186633178790216],
    [3.0594344551767474, 2.5734373249436153, 5.0, 0.5775248272351745, -0.7268839472372766],
    [2.510098508570067, 1.0140261486136761, -0.5607111292859983, -1.8675776329487066, 2.0502831156323604],
    [2.0973468503670976, 1.0352248632738257, -1.035217267081034, 4.996616651782531, -1.9664613530516961],
    [11.344833400049993, 2.9985629661632203, -5.0, -3.0420360410153684, 1.0197042974760435],
    [2.523722448928199, 3.69289807861643, 4.553473956273982, -4.638515738396874, -1.0226818096064623],
    [-1.227246207145753, -2.5667570308222047, 2.059268138083004, -0.8408176140170136, -3.0],
    [14.288363824580978, -2.1492678104419176, 3.264200485223529, -1.2259581673011057, 1.1979018647419841],
    [0.5365986908899849, -1.2518952710548144, 0.4048154964198152, -0.06620515021629902, 2.3664896432948015],
    [6.444510531042768, 0.6748562153825208, -0.9324840306706532, -2.488324819549029, 0.36886764409023504],
    [3.0613288689408753, 3.8844578301699246, -3.0199898735744233, 1.1474180951273385, 2.4909544967249104],
    [-8.885893592839025, -0.5083865397957323, -2.394476846345226, -1.1876379186152333, -1.1702972593212029],
    [-0.5244405928262119, 3.985592286873424, 2.4330580273185305, -1.1971432392600745, 2.290811392835656],
    [13.002833603012812, -0.054924571822372426, -3.1854931404844766, 4.888480751320017, 2.0814511043679653],
    [5.489907083942336, -2.3794279408486436, 2.8082464056436667, 1.724204509225884, -1.5008740397037936],
    [10.3411122925061, 1.211489358285835, 3.3618933909567508, -5.0, -1.8022125899903019],
    [5.548980853110949, 0.8088842085619108, 1.1277402958565577, 5.0, -3.0],
    [13.628517992661992, 0.2043409689924247, 0.7753741110021695, -1.6161233297903999, 2.763370395129628],
]
atlas_inspired = [
    [ 1, 1, 0, 0, 0],
    [ 1, 1, 0, 0, 1],
    [ 1, 1, 0, 0, -1.2],
    [ 1, 1, -0.2, 0, 0],
    [ 1, 1, 0.7, 0, 0],
    [ -2.5, 1, 0, 0, 0],
    [ 9, 1, 0, 0, 0],
    [ -2.5, 1, 0, 0, -1.2],
    [ -10, 1, 1, 0, 0],
    [ 1, 1, -0.2, 0, -1.2],
    [1.0, 1.0, 0., 1, -2],
    [1.0, 1.0, 0., 1, 1.1],
    [1.0, 1.0, 0., -3, 3],
    [1.0, 1.0, 0., -0.5, -0.7],
    [1.0, 1.0, 0., 1.5, -0.5],
   # [1, 1.0, 1., 0, 0],
  #  [-4, 1.0, 0., 0, 0],
 #   [-1, 1.0, 0., 0, -3],
    [1, 1.0, 1., 1, 0],
     [1, 1.0, 1., 6, 0],
     [1, 1.0, 10., 1, 0],
     [1, 1.0, 0., 1, 0],
  # [5., 1.0, 0, -0.2, 0.2],
   #  [1, 1.0, 1, -0.2, 0.2],
    [1.0, 1.0, 0.5, -0.8, 0.6],
    [2.4, 1.0, 0.0, 0.2, -0.2],
    [15.0, 1.0, 0.0, -1.0, 1.0],
     [1.0, 5.0, 0., 0, 0.],
    [20, 1, 0, 0, 5],
   # [15.0, 1.0, 0.0, -1.0, 1.0],
    [1.0, 1.0, 1.0, -0.6, 0.6],
    [0.0, 0.0, 0.0, 0., 1],
]


# Convert psotemp list to dictionary format expected by makebase
def convert_to_dict(basis_list):
    """Convert list of basis points to dictionary format expected by mhh_scoring"""
    result = {}
    for i, point in enumerate(basis_list):
        idx = str(i + 1)
        result[f"kl_{idx}"] = point[0]
        result[f"kt_{idx}"] = point[1]
        result[f"c2_{idx}"] = point[2]
        result[f"cg_{idx}"] = point[3]
        result[f"c2g_{idx}"] = point[4]
    return result

if __name__ == "__main__":
    # Convert bases to dictionary format
    psotemp_dict = convert_to_dict(psotemp)
    atlas_inspired_dict = convert_to_dict(atlas_inspired)
    pso_0_dict = convert_to_dict(pso_0)

    plotCompare(1.0, 2.0, 2.0, -0.2, 1,
                [atlas_inspired_dict, pso_0_dict, psotemp_dict],
                lim=True,
                inputnames=["atlas inspired", "pso_0", "psotemp"])

    # Alternative: Compare only two bases
    # plotCompare(1.0, 1.0, 0.5, -0.8, 0.6,
    #             [pso_0_dict, psotemp_dict],
    #             lim=True,
    #             inputnames=["pso_0", "psotemp"],
    #             colors=['tab:orange', 'tab:green'])