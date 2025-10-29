#!/usr/bin/env python3
# Script to extract mHH, pT(H), pT(HH), cosTheta from ROOT ntuple

import uproot
import numpy as np
import awkward as ak
import sys
import os
import glob
import ROOT
from array import array

def get_higgs_indices(pdgids):
    return np.where(pdgids == 25)[0]

def calc_mHH(H1, H2):
    E = H1['E'] + H2['E']
    px = H1['px'] + H2['px']
    py = H1['py'] + H2['py']
    pz = H1['pz'] + H2['pz']
    return np.sqrt(E**2 - px**2 - py**2 - pz**2)

def calc_pT(px, py):
    return np.sqrt(px**2 + py**2)

def calc_cosTheta(H1, H2):
    p1 = np.array([H1['px'], H1['py'], H1['pz']])
    p2 = np.array([H2['px'], H2['py'], H2['pz']])
    dot = np.dot(p1, p2)
    norm1 = np.linalg.norm(p1)
    norm2 = np.linalg.norm(p2)
    return dot / (norm1 * norm2)

def main():
    if len(sys.argv) < 3:
        print("Usage: python mhh_extractor.py input_dir_or_root output.root")
        sys.exit(1)

    input_path = sys.argv[1]
    outroot = sys.argv[2]

    root_files = []
    if os.path.isdir(input_path):
        root_files = sorted(glob.glob(os.path.join(input_path, '*.root')))
        if not root_files:
            print(f"No .root files found in {input_path}")
            sys.exit(1)
    else:
        root_files = [input_path]

    # Prepare output ROOT file and tree
    f = ROOT.TFile(outroot, "RECREATE")
    t = ROOT.TTree("vars", "mHH and kinematic variables")

    mHH = array('f', [0])
    pT_H1 = array('f', [0])
    pT_H2 = array('f', [0])
    pT_HH = array('f', [0])
    cosTheta = array('f', [0])

    t.Branch("mHH", mHH, "mHH/F")
    t.Branch("pT_H1", pT_H1, "pT_H1/F")
    t.Branch("pT_H2", pT_H2, "pT_H2/F")
    t.Branch("pT_HH", pT_HH, "pT_HH/F")
    t.Branch("cosTheta", cosTheta, "cosTheta/F")

    for infile in root_files:
        print(f"Processing {infile}")
        try:
            tree = uproot.open(infile)["mytree"]
        except Exception as e:
            print(f"Could not open tree in {infile}: {e}")
            continue

        pdgID = ak.to_numpy(tree["pdgID"].array())
        px = ak.to_numpy(tree["px"].array())
        py = ak.to_numpy(tree["py"].array())
        pz = ak.to_numpy(tree["pz"].array())
        E  = ak.to_numpy(tree["E"].array())

        for iev in range(len(pdgID)):
            higgs_idx = get_higgs_indices(pdgID[iev])
            if len(higgs_idx) < 2:
                continue

            H1 = {'px': px[iev][higgs_idx[0]], 'py': py[iev][higgs_idx[0]], 'pz': pz[iev][higgs_idx[0]], 'E': E[iev][higgs_idx[0]]}
            H2 = {'px': px[iev][higgs_idx[1]], 'py': py[iev][higgs_idx[1]], 'pz': pz[iev][higgs_idx[1]], 'E': E[iev][higgs_idx[1]]}

            mHH[0] = calc_mHH(H1, H2)
            pT_H1[0] = calc_pT(H1['px'], H1['py'])
            pT_H2[0] = calc_pT(H2['px'], H2['py'])
            pT_HH[0] = calc_pT(H1['px'] + H2['px'], H1['py'] + H2['py'])
            cosTheta[0] = calc_cosTheta(H1, H2)

            t.Fill()

    f.Write()
    f.Close()
    print(f"Saved to {outroot}")

if __name__ == "__main__":
    main()
