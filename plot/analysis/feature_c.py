import os
import sys
from turtle import color

sys.path.insert(0,"/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
import pandas as pd
import uproot3 as uproot
import numpy as np
import matplotlib.pyplot as plt

import ROOT as root
import atlasplots as aplt

def feature_c():
    # Set the ATLAS Style
    aplt.set_atlas_style()

    # Create a figure and axes
    fig, ax = aplt.subplots(1, 1, name="fig1", figsize=(800, 600))
    x = [20, 40, 60, 80, 100]
    y = [
1352.0527949455206,
1344.535544379441,
1283.1335853924536,
1281.0912387711671,
1293.9812781521514,
         ]
    err = [
42.610679521202535,
70.07527091597395,
79.24194299912502,
65.11957494096988,
75.4199337014194,
    ]

    graph = ax.graph(x, y, yerr=err, labelfmt="EP", options="P",markercolor=root.kBlue+2, label="Significance", textsize=20)
    graph.Fit("pol1", "0")



    #ax.add_margins(top=0.18, left=0.1, right=0.5, bottom=0.05)



    #Plot fit function and extend its range to fill plot area
    func = graph.GetFunction("pol1")
    func.SetRange(*ax.get_xlim())
    func.SetNpx(1000)
    # ax.plot(func, linecolor=root.kRed + 2, linestyle=root.kDashed, expand=False, label="Fit",
    #         size=22, labelfmt="L", textsize=40)
    #aplt.atlas_label(text="Internal", loc="upper left")
    ax.text(0.2, 0.81, "Test: LQ 1500 GeV", align=10, size = 35)
    ax.text(0.2, 0.74, "CatBoost", align=10, size = 35)
    ax.legend(loc=(0.53, 0.90, 0.95, 0.75),textsize=40)
    ax.set_xlabel("Used features [%] (89)", loc='center', labelsize=40, titlesize=40, titleoffset=1, )
    ax.set_ylabel("Significance [#sigma]", loc='center', titleoffset=1.7 , labelsize=40, titlesize=40,labeloffset = 0.01)
    #ax.add_margins(left=0.5)
    ax.cd()
    ax.set_ylim(400, 1800)
    ax.set_xlim(19, 101)
    fig.savefig("/home/lucas/Documents/KYR/bc_thesis/data_processed/imgs/feature_cut/cat_3_feat_cut.pdf")