import os
import sys
from turtle import color

sys.path.insert(0,"/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
from root_load.root_pandas_converter import RPConverter
import pandas as pd
import uproot3 as uproot
import numpy as np
import matplotlib.pyplot as plt

import ROOT as root
import atlasplots as aplt

def xs_m(xs, xs_new_1,xs_new_2, xs_new_3, xs_new_4,xs_new_5, m):
  #   x = m
  #   y1 = xs
  #   y2 = xs_new_1
  #   y3 = xs_new_2
  #   y4 = xs_new_3
  #   y5 = xs_new_4
  #   y6 = xs_new_5

  #   z = np.polyfit(x, y2, 8)
  #   p = np.poly1d(z)
  #   xp = np.linspace(500, 1600, 100)

  #   z2 = np.polyfit(x, y3, 8)
  #   p2 = np.poly1d(z2)
  #   xp2 = np.linspace(500, 1600, 100)

  #   z3 = np.polyfit(x, y4, 8)
  #   p3 = np.poly1d(z3)
  #   xp3 = np.linspace(500, 1600, 100)

  #   z4 = np.polyfit(x, y5, 8)
  #   p4 = np.poly1d(z4)
  #   xp4 = np.linspace(500, 1600, 100)

  #   z5 = np.polyfit(x, y6, 8)
  #   p5 = np.poly1d(z5)
  #   xp5 = np.linspace(500, 1600, 100)

  #   plt.xlabel("Mass [GeV]")
  #   plt.ylabel("xs [pb]")
  #   plt.title("Cross section as function of mass")
    
  #   plt.plot(x,y1, label = "Theoretical cross section", color = 'red')
    
  #   # plt.plot(xp,p(xp), label = "Polynomial fit ADA", color = 'green',
  #   #  linestyle = '--')
    
  #   plt.plot(xp2,p2(xp2), label = "Polynomial fit RFC", color = 'purple',
  #    linestyle = '--')

  #   plt.plot(xp3,p3(xp3), label = "Polynomial fit NN", color = 'orange',
  #    linestyle = '--')

  #   plt.plot(xp4,p4(xp4), label = "Polynomial fit RFC new signif.", color = 'blue',
  #    linestyle = '--')

  #   plt.plot(xp5,p5(xp5), label = "Polynomial fit CAT new signif.", color = 'yellow',
  #    linestyle = '--')
    
  #   # plt.plot(x,y2, label = "Computed cross section ADA", color='green',
  #   #  linestyle=' ', marker = 'x', markerfacecolor = 'green',
  #   #   markersize = 8)
    
  #   plt.plot(x,y3, label = "Computed cross section RFC", color='purple',
  #    linestyle=' ', marker = 'x', markerfacecolor = 'purple',
  #     markersize = 8)
    
  #   plt.plot(x,y4, label = "Computed cross section NN", color='orange',
  #    linestyle=' ', marker = 'x', markerfacecolor = 'orange',
  #     markersize = 8)
      
  #   plt.plot(x,y5, label = "Computed cross section RFC new signif.", color='blue',
  #    linestyle=' ', marker = 'x', markerfacecolor = 'blue',
  #     markersize = 8)

  #   plt.plot(x,y6, label = "Computed cross section CAT new signif.", color='yellow',
  #    linestyle=' ', marker = 'x', markerfacecolor = 'yellow',
  #     markersize = 8)
    
  #   plt.yscale("log")
  #   plt.legend()
  #  # plt.xlim(800,1200)
  #   plt.show()


        # Set the ATLAS Style
    aplt.set_atlas_style()

    # Create a figure and axes
    fig, ax = aplt.subplots(1, 1, name="fig1", figsize=(800, 600))

    # Plot a line with Gaussian noise
    #x = np.arange(20)
    x = np.linspace(500, 1600, 12)
    #y = 2 * x + np.random.normal(size=20)
    #y = xs_new_2
    y = [0.003656, 0.002486, 0.002826, 0.00162,
         0.001588, 0.001604, 0.001586, 0.001608, 0.001742, 0.001742, 0.001742, 0.001742]

    x_2 = np.linspace(800, 1200, 21)
    y_2 = [0.00335714285714286,0.00321428571428571,0.00307142857142857,0.00299641362821279
,0.00283333333333333,0.00277777777777778,0.00266666666666667,0.00261111111111111
,0.00255555555555556,0.00244444444444444,0.00238888888888889,0.00238888888888889
,0.00233333333333333,0.00233333333333333,0.00227777777777778,0.00227777777777778
,0.00222222222222222,0.00216666666666667,0.00211111111111111,0.00205555555555556
,0.002]

    x_3 = [800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000,
    1020, 1040, 1060, 1080, 1100, 1120, 1140, 1160, 1180, 1200,
    1250, 1300, 1350, 1400, 1450, 1500]
    y_3 = [0.0321428571428571,0.0277777777777778,0.0233333333333333,0.0196875,
0.0175,0.015,0.013125,0.010625,0.009,0.008,0.007,0.005875,0.005,0.0044,
0.00378571428571429,0.00335714285714286,0.00294444444444444,0.00255555555555556,
0.00222222222222222,0.0019375,0.00175, 0.00126785714285714, 0.000866666666666667,
0.000641666666666667, 0.000473684210526316, 0.000347826086956522, 0.000258823529411765
]

    #yerr = np.random.normal(loc=np.linspace(1, 2, num=12), scale=0.1, size=20)


  

    graph = ax.graph(x, y, labelfmt="EP", options=" ")
    graph_2 = ax.graph(x_2, y_2, labelfmt="EP", options=" ")
    graph_3 = ax.graph(x_3, y_3, labelfmt="EP", options=" ")
    

    # Fit the graph; store graphics func but do not draw
    graph.Fit("pol5", "0")
    graph_2.Fit("pol2", "0")
    graph_3.Fit("pol6", "0")

    # Add extra space at top of plot to make room for labels
    ax.add_margins(top=0.18, left=0.2, right=0.05, bottom=0.05)


    # Plot fit function and extend its range to fill plot area
    func = graph.GetFunction("pol5")
    func.SetRange(*ax.get_xlim())
    #func.SetNpx(1000)
    ax.plot(func, linecolor=root.kMagenta + 2,linestyle=root.kDashed, expand=False, label="2lSS + #tau (new result)", size=22, labelfmt="L", textsize=10)

    func_2 = graph_2.GetFunction("pol2")
    func_2.SetRange(*ax.get_xlim())
    func_2.SetNpx(1000)
    ax.plot(func_2, linecolor=root.kBlack,linestyle=root.kDashed, expand=False, label="2lSS/3l + #tau (old result)", labelfmt="L")

    func_3 = graph_3.GetFunction("pol6")
    func_3.SetRange(*ax.get_xlim())
    func_3.SetNpx(1000)
    ax.plot(func_3, linecolor=root.kRed,linestyle=root.kSolid, expand=False, label="Theory (NNLO+NNLL)", labelfmt="L")

    # Set axis titles
    ax.set_xlabel("m_{LQ^{d}_{3}} [GeV]")
    ax.set_ylabel("\sigma(pp\\rightarrowLQ^{d}_{3}LQ^{d}_{3}) [pb]")

    ax.set_yscale("log")
    ax.set_xlim(800,1200)
    ax.set_ylim(0.0001,0.5)

    # Add the ATLAS Label
    #aplt.atlas_label(text="Internal", loc="upper left")
    ax.text(0.2, 0.85,"#sqrt{s} = 13 TeV, 139 fb^{-1}", align=10)
    ax.text(0.2, 0.78,"LQ^{d}_{3}LQ^{d}_{3}\\rightarrow t\\taut\\tau", align=10)
    ax.text(0.2, 0.73,"95% CL", align=10)
    # Add legend
    ax.legend(loc=(0.58, 0.9, 0.95, 0.70),textsize=20)
    # Save the plot as a PDF
    fig.savefig("/home/lucas/Documents/KYR/bc_thesis/data_processed/imgs/2lss_1tau_graph_zoom.pdf")
    ax.set_xlim(800,1300)
    fig.savefig("/home/lucas/Documents/KYR/bc_thesis/data_processed/imgs/2lss_1tau_graph.pdf")