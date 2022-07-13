import os
import sys
from turtle import color

sys.path.insert(0,"/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
from root_load.root_pandas_converter import RPConverter
import pandas as pd
import uproot3 as uproot
import numpy as np
import matplotlib.pyplot as plt
import statistics

import ROOT as root
import atlasplots as aplt

def mass_guess():
    # Set the ATLAS Style
    aplt.set_atlas_style()

    mass = [
        [
            602,
            599,
            584,
            569,
            593
        ],
        [
            652,
            653,
            652,
            628,
            665
        ],
        [
            733,
            749,
            754,
            733,
            727
        ],
        [
            825,
            830,
            825,
            828,
            780
        ],
        [
            912,
            901,
            912,
            880,
            915
        ],
        [
            976,
            985,
            988,
            987,
            999
        ],
        [
            1084,
            1072,
            1062,
            1083,
            1081
        ],
        [
            1141,
            1162,
            1136,
            1139,
            1163
        ],
        [
            1212,
            1207,
            1186,
            1186,
            1217
        ],
        [
            1272,
            1252,
            1241,
            1238,
            1304
        ],
        [
            1325,
            1325,
            1297,
            1320,
            1264
        ],
        [
            1342,
            1329,
            1386,
            1380,
            1407
        ]
    ]

    mass_avg = []
    mass_dev = []

    for val in mass:
        mass_avg.append(statistics.mean(val))
        mass_dev.append(statistics.stdev(val))


    print("Mass mean: "+str(mass_avg))
    print("Mass variance: "+str(mass_dev))
    # Create a figure and axes
    fig, ax = aplt.subplots(1, 1, name="fig1", figsize=(800, 600))
    x = np.linspace(500, 1600, 12)
    y_2 = np.linspace(500, 1600, 12)
    y = mass_avg#[592, 677, 726, 816, 907, 994, 1067, 1123, 1181, 1242, 1259, 1367]
    yerr = mass_dev
    graph = ax.graph(x, y, yerr=yerr, labelfmt="EP", options="P")
    graph.Fit("pol1", "0")
    graph2 = ax.graph(x, y_2, labelfmt="EP", options=" ")
    graph2.Fit("pol1", "0")

    #ax.add_margins(top=0.18, left=0.1, right=0.5, bottom=0.05)



    # Plot fit function and extend its range to fill plot area
    func = graph.GetFunction("pol1")
    func.SetRange(*ax.get_xlim())
    func.SetNpx(1000)
    ax.plot(func, linecolor=root.kMagenta + 2, linestyle=root.kDashed, expand=False, label="Fit",
            size=22, labelfmt="L", textsize=10)
    func2 = graph2.GetFunction("pol1")
    func2.SetRange(*ax.get_xlim())
    func2.SetNpx(1000)
    ax.plot(func2, linecolor=1, linestyle=1, expand=False, label="Ideal prediction",
            size=22, labelfmt="L", textsize=10)
    #aplt.atlas_label(text="Internal", loc="upper left")
    ax.legend(loc=(0.18, 0.9, 0.95, 0.75),textsize=30)
    ax.set_xlabel("Expected mass [GeV]", loc='center')
    ax.set_ylabel("Predicted mass [GeV]", loc='center', titleoffset=1.8)
    #ax.add_margins(left=0.5)
    ax.cd()
    ax.set_xlim(495, 1605)
    ax.set_ylim(500, 1600)
    fig.savefig("/home/lucas/Documents/KYR/bc_thesis/data_processed/imgs/expected_mass.pdf")
