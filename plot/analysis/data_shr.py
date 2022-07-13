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

def data_shr():
    # Set the ATLAS Style
    aplt.set_atlas_style()

    # Create a figure and axes
    fig, ax = aplt.subplots(1, 1, name="fig1", figsize=(800, 600))
    x = np.linspace(20, 100, 5)
    y = [
1308.964690367761,
1407.2768822967848,
1342.171846812253,
1404.6583505712667,
1322.8335941772857
         ]
    err = [
162.32164247669365,
159.61495960626863,
143.41986602662044,
127.52918288672582,
113.3064369722806
    ]

    graph = ax.graph(x, y, yerr=err, labelfmt="EP", options="P",markercolor=root.kBlue+2, label="Significance", textsize=20)
    graph.Fit("pol1", "0")



    #ax.add_margins(top=0.18, left=0.1, right=0.5, bottom=0.05)



    #Plot fit function and extend its range to fill plot area
    func = graph.GetFunction("pol1")
    func.SetRange(*ax.get_xlim())
    func.SetNpx(1000)
    # ax.plot(func, linecolor=root.kRed + 2, linestyle=root.kDashed, expand=False, label="Fit",
    #         size=22, labelfmt="L", textsize=10)
    #aplt.atlas_label(text="Internal", loc="upper left")
    ax.text(0.2, 0.81, "Test: LQ 1500 GeV", align=10, size=35)
    ax.text(0.2, 0.74, "CatBoost", align=10, size=35)
    ax.legend(loc=(0.53, 0.90, 0.95, 0.80),textsize=40)
    ax.set_xlabel("Used simulated data [%]", loc='center', labelsize=40, titlesize=40, titleoffset=1)
    ax.set_ylabel("Significance [#sigma]", loc='center', titleoffset=1.7 , labelsize=40, titlesize=40,labeloffset = 0.01)
    #ax.add_margins(left=0.5)
    ax.cd()
    ax.set_ylim(500, 1800)
    ax.set_xlim(19, 101)
    fig.savefig("/home/lucas/Documents/KYR/bc_thesis/data_processed/imgs/data_shrink/cat_2_dat_shrink.pdf")