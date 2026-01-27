
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
def plotCompare(kl, kt, c2, cg, c2g, inputs, makeplot=True, drawmodel=True, lim=True, inputnames=["1"], debug=False, highstat=False):
    samplesize = 50000
    if highstat: samplesize*=10
    model = calcDistModel(kl, kt, c2, cg, c2g, inputs[0], samplesize=samplesize)
    reweight = calcDist(kl, kt, c2, cg, c2g, samplesize=samplesize)
    if lim == False:
        model = calcDistModel(kl, kt, c2, cg, c2g, inputs[0], samplesize=samplesize+1)
        reweight = calcDist(kl, kt, c2, cg, c2g, samplesize=samplesize+1)
    othermodels = [model]
    for im,m in enumerate(inputs):
        if im!=0:
            othermodels.append(calcDistModel(kl, kt, c2, cg, c2g, m, samplesize=samplesize))
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
                hep.histplot(m, histtype='band', linewidth=3, density=d, alpha=0.5, color=petroff_10[im], ax=ax)
                hep.histplot(m, histtype='errorbar', linewidth=3, density=d, label=r"from model {}".format(inputnames[im]),color=petroff_10[im],ax=ax)

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
                hep.histplot(ratio, histtype='band', linewidth=3, density=d, alpha=0.5, color=petroff_10[im], ax=ax2)
                hep.histplot(ratio, histtype='errorbar', linewidth=3, density=d, label=r"from model {}".format(inputnames[im]),color=petroff_10[im],ax=ax2)

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
        output_dir = "plot_compare"
        os.makedirs(output_dir, exist_ok=True)
        plotname = os.path.join(output_dir, f"compare_kl{kl}_kt{kt}_c2{c2}_cg{cg}_c2g{c2g}.png")
        plt.savefig(plotname, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plotname}")

    return ks, stat, statref, chi2

# Basis
psotemp = [[2.213221919449151, -2.6934464500856405, 2.965671189693319, 1.1360050891505196, -2.1481034846500293], [9.76060116521356, -4.0, 3.0808116991517873, 4.979656543379024, 1.7948537607896533], [-8.504037399703757, 2.4570674701218005, 2.0456701185442197, 4.993273200619232, 0.31806830918129725], [12.42168994697454, 4.8885158267788515, 1.1736452461594093, -1.649885452005208, 2.9957802625445793], [-8.028050651896733, 1.4185892815035221, 4.637324985625828, 1.100571508815925, -2.141440691112374], [-0.6417775386928877, -3.4585451693407876, 1.5364031301770482, 1.449765057791862, -3.0], [4.516788454318963, 3.246315195973261, 0.03179409373963871, 4.515743016527684, -3.0], [6.776972542859125, 2.7536031422527922, 1.1291947583909658, -4.998564441693077, -3.530381379838789], [-7.295476241252043, -3.567728937382591, 4.927055574269499, 5.4698157741693745, 0.5460308050762466], [10.16791838833636, 3.168115455700297, 3.988981859905667, -3.9955456956734263, 2.5047215502921873], [-9.237668351371102, -0.8830087027316762, 1.507497463102394, -1.656941023505912, 1.0373600385652952], [15.871698091121043, 2.954226053612609, -1.4832395231493958, -4.648315411280079, -0.31921721853099594], [-6.0920951495343445, -1.2529968110851288, -0.661498001605271, 3.1356841635278725, 2.473949503061431], [-13.765201883693987, -1.4548873762027377, 2.2019807467420858, 6.0, 2.0762352620706594], [-13.956921393217277, -3.867174172277096, 2.121296077606367, 3.066609957858563, 1.9738262253820915], [8.2879617206356, -4.0, 9.611068631424631, 3.4554065505161273, 0.9602330195006838], [-3.368487622557269, -1.235213414828737, -1.3214826590555613, 5.485928293425935, -1.6308816657564937], [-1.1492785231960068, 0.09464020330641909, 3.076473122787405, -3.6375285759285543, 1.3735185288149214], [17.4, 1.8726121309906936, 0.2733568393951952, -2.568047729842719, -2.85757278314612], [9.019237551902098, 0.9358945607455986, -4.368407508947331, 3.9318179425747295, 2.72703446921505], [10.092619602076475, -0.25598532434815535, 0.5458135306798126, 1.061229295632444, 5.764408396915333], [-7.594748729686678, -3.5504975021060012, 0.9679708236284144, -0.5666309250467718, 2.8180805513778475], [-2.167934532063678, -0.46393247419740047, 3.091840758463215, 2.0233283474829102, -1.2791407684980047]]
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
    # Convert psotemp to dictionary format
    psotemp_dict = convert_to_dict(psotemp)
    atlas_inspired_dict = convert_to_dict(atlas_inspired)

    # Example call with specific parameters
    plotCompare(1.0, 2.0, 2.0, -0.2, 1, [atlas_inspired_dict,psotemp_dict], lim=True, inputnames = ["atlas inspired", "psotemp"])
    # plotCompare(1.0, 1.0, 0.5, -0.8, 0.6, [psotemp_dict], lim=True, inputnames=["psotemp"], debug=True)