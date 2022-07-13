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

def sep_a():
    # Set the ATLAS Style
    aplt.set_atlas_style()

    tmp = [
        [
            1088.5612775608183,
            1181.049781065162,
            1388.6843560994714,
        ],
        [
            1227.958163960575,
            301.0975280759219,
            0.03702712540974036,

        ],
        [
            80.6183855789191,
        1157.9713662691593,
            0.01631886639814789,

        ],
        [
            82.2707188275208,
            60.91081018142879,
            145.9942929279843,
        ],
        [
            1295.475955048873,
            1151.1307906642407,
            1377.2787891662458,
        ],
        [
            1328.6339930992433,
            291.5599931853622,
            325.63956755935385,
        ],
        [
            281.12727331559967,
            1097.7673250182197,
            269.35498587367755,
        ],
        [
            83.2348422931978,
            63.67678222589491,
            1108.8643278658965,
        ]
    ]

    tmp_stdev = [
        [
           81.9181426028953,
            49.62255550622814,
            111.25806782019902,
        ],
        [
            41.87619154674135,
22.66319693543358,
0.003938917819055359,
        ],
        [
            4.9679808496365405,
            68.80825631704188,
            0.016447911906652023,

        ],
        [
            3.5381274136186236,
            5.190797332078529,
            11.48390988186201,
        ],
        [
            210.2344379616187,
            131.5657546454862,
            178.82932010157305,
        ],
        [
            39.427152947756895,
            17.68630062705821,
            43.433319479606276,
        ],
        [
            13.178040703912473,
            89.34403177877213,
            111.04607669842311,
        ],
        [
            6.652491232647046,
            4.281272695170009,
            134.7814256469776,
        ]
    ]

    # Create a figure and axes
    fig, ax = aplt.subplots(1, 1, name="fig1", figsize=(800, 600))
    x = [600, 1000, 1500]
#     y = [
# 83.2348422931978,
#         63.67678222589491,
#         1108.8643278658965,
#          ]
#     err = [
# 6.652491232647046,
#         4.281272695170009,
#         134.7814256469776,
#     ]

    y = tmp[7]
    err = tmp_stdev[7]
    graph = ax.graph(x, y, yerr=err, labelfmt="EP", options="P",markercolor=root.kBlue+2, label="Significance")
    graph.Fit("pol2", "0")



    #ax.add_margins(top=0.18, left=0.1, right=0.5, bottom=0.05)



    #Plot fit function and extend its range to fill plot area
    func = graph.GetFunction("pol2")
    func.SetRange(*ax.get_xlim())
    func.SetNpx(1000)
    # ax.plot(func, linecolor=root.kRed + 2, linestyle=root.kDashed, expand=False, label="Fit",
    #         size=22, labelfmt="L", textsize=10)
    #aplt.atlas_label(text="Internal", loc="upper left")
    ax.text(0.2, 0.84, "Train: LQ 1500 GeV", align=10, size=35)
    ax.text(0.2, 0.77, "TabNet", align=10, size = 35)
    ax.legend(loc=(0.53, 0.90, 0.95, 0.75),textsize=40)
    ax.set_xlabel("Mass [GeV]", loc='center', labelsize=32, titlesize=40, titleoffset=1, labeloffset = 0.01)
    ax.set_ylabel("Significance [#sigma]", loc='center', titleoffset=1.7 , labelsize=32, titlesize=40,labeloffset = 0.02)
    #ax.add_margins(left=0.5)
    ax.cd()
    ax.set_ylim(0, 1900)
    ax.set_xlim(550, 1550)
    fig.savefig("/home/lucas/Documents/KYR/bc_thesis/data_processed/imgs/sep_all/tab_4_sep_all.pdf")