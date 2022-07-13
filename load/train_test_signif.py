import os
import sys
from turtle import color

sys.path.insert(0,"/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
import numpy as np

import ROOT as root
import atlasplots as aplt

def train_test_signif():

#-------------------------------train vs test comparison DATA--------------------------
    mass_arr = ['LQ 500','LQ 600','LQ 700','LQ 800','LQ 900','LQ 1000','LQ 1100','LQ 1200',
    'LQ 1300','LQ 1400','LQ 1500','LQ 1600',]
    lq = [[1.634662544820075,1.497641776771227, 1.197692367480464,1.218745003216545,
    0.9502884066826501, 0.992046736492839, 0.8900310753822318, 0.7414889277467411,
    0.4275478148448807, 0.30335963001089944, 0.29313815343285893, 0.4214179530574859 ],
    
  [2.457687572872928,2.174752501458932,2.0533025613441542,2.368393230077858,
1.9420454676677346,1.733316587948239,1.5705437241750704,1.4704205180255383,0.8847257398365204,
0.6724951690885331,0.7063863055007172,0.7250889240923236],
    
  [2.5073492525020584,2.0965929578776774,2.2706507137283127,2.528467047925054,
    2.3310516695555448,2.1241683967780562,1.9546052949557904,1.9137662250906822,1.3998382139861274,
 1.1740864860798617,1.109149994324672,1.327350569664545
],
  [2.2843739611786744,1.9271998707963707,2.1820100269152185,2.3527289531520004,
2.13931523002056,2.2214799253186435,2.0610813627840106,2.097451099712387,
 1.4936742329223585,1.5163738179871098,1.42166058548727,1.6205231702330998
],
 [2.112900895735674,1.8380774672145748,2.086166244399338,2.384398244763309,
2.2715634204114505,2.22451903905854,2.329854102693669,2.2339279598868367,
 1.7985282369283329,1.7307758840998153,1.5811947511010647,1.7348263536582935],
    
    [2.3609415528943734,2.0895447253705304,2.4099245480242066,
2.6701200807516967,2.546768448026086,2.5881220535238065,2.7209486417889024,
 2.5178831040208034,2.166554367563956,2.231484984501038,2.0771889190883877,2.161625479412732
],
   [2.045241672237781,1.8718992472758247,2.073739624821091,2.252206953628311,
2.290014321298263,2.284622204950984,2.284728025230671,2.2680162404950037,1.923943457738991,
 2.0035111005934643,1.910284337799687,2.1018400494465777
],
    [2.5197264478891794,2.2372048561264224,2.542729046407169,2.729189633888998,
    2.6931329453233057,2.7341698386353985,2.739242890504144,2.7475904259998702,
    2.3755557269352408,2.554524521846126,2.39039492938371,2.636626865105986
],
    [2.499411125701156,2.246776141813049,2.5107874739667637,2.8425712498379774,
2.7810130985021613,2.823685486736845,2.812324925506369,2.739236209491152,
 2.5131976316751437,2.5913113651884396,2.594398691714135,2.639627063943426
],
    [1.8268512990749544,1.416109024563747,1.6897023762134495,1.8823465887929436,
1.8196329135776907,1.8132442442747205,1.9105996299909043,1.8457844361041056,
 1.694285318137562,1.7921184851976362,1.6575911503996148,1.782130473105419
],
    [1.8378268353706348,1.4904406867372135,1.7956590150248608,2.1125661395358706,
1.9854625737716134,2.006437068176511,2.0905100864356885,2.0386484723967437,
 1.8483721437263165,1.9712251375279393,1.8788500692380712,1.9613363494945835
],
    [2.0899831272843947,1.78074177103969,2.1006322039353718,2.2818775158975972,
2.2228099625120423,2.310793655139004,2.353466884086617,2.3235092107431714,
2.1381324224496687,2.3089499882800584,2.262956350775604,2.2598456802480893
]]

    lq_all = [2.8742003667861518,2.3529506002643172,
2.3000358308349007,2.2577438804858305,
1.925235420325598,1.914491726607149,
2.0137709461138633,1.0953129654601144,
2.1952173976294467,1.88187965470548,
2.059264612274501,2.4011863773394473
]

    lq_max = [
        3.0506611852885492,
2.9997258895255463,
2.232749460668739,
2.3861474623860692,
2.009670144661442,
 1.7869885398826195,
 2.141788202402958,
 1.8982049443900337,
 1.8829316693644003,
 0.23508856207728032,
 1.8164975621992439,
 0.19542580123071635
    ]
#     lq_all = [1.8883713623736096,2.690265715516196,3.1730098133883784,
# 3.068729699913984,3.0513930516350594,3.081533449675624,2.71107484433694,
# 3.3568910960448384,3.3250266718128842,2.3324232925666286,
# 2.4145282982373426,2.9400735232314315
# ]




#----------------------------------test vs train END




     # Set the ATLAS Style
    aplt.set_atlas_style()

   

    figs = [None] * 12
    axes = [None] * 12
    x = np.linspace(500, 1600, 12)
    graphs = []
    # lq_max = []

    # for i in range(12):
    #     lq_max.append(max(lq[i]))
    # print(lq_max)



    img_all, ax_all =  aplt.subplots(1, 1, name='lq_all', figsize=(800, 600))
    graph_all = ax_all.graph(x, lq_all, labelfmt="EP", options=" ")
    graph_max = ax_all.graph(x, lq_max, labelfmt="EP", options=" ")

    # Create a figure and axes
    for i in range(12):
        figs[i], axes[i] = aplt.subplots(1, 1, name=mass_arr[i], figsize=(800, 600))
        graphs.append(axes[i].graph(x, lq[i], labelfmt="EP", options="P"))
    
   
    # graphs.append(ax1.graph(x, lq_500, labelfmt="EP", options="P"))
    # graphs.append(ax2.graph(x, lq_600, labelfmt="EP", options="P"))
    # graphs.append(ax3.graph(x, lq_700, labelfmt="EP", options="P"))
    # graphs.append(ax4.graph(x, lq_800, labelfmt="EP", options="P"))
    # graphs.append(ax5.graph(x, lq_900, labelfmt="EP", options="P"))
    # graphs.append(ax6.graph(x, lq_1000, labelfmt="EP", options="P"))
    # graphs.append(ax7.graph(x, lq_1100, labelfmt="EP", options="P"))
    # graphs.append(ax8.graph(x, lq_1200, labelfmt="EP", options="P"))
    # graphs.append(ax9.graph(x, lq_1300, labelfmt="EP", options="P"))
    # graphs.append(ax10.graph(x, lq_1400, labelfmt="EP", options="P"))
    # graphs.append(ax11.graph(x, lq_1500, labelfmt="EP", options="P"))
    # graphs.append(ax12.graph(x, lq_1600, labelfmt="EP", options="P"))
    for i, graph in enumerate(graphs):
        graph.Fit("pol5","0")
        func = graph.GetFunction("pol5")
        func.SetRange(*axes[i].get_xlim())
        #func.SetNpx(1000)
        axes[i].plot(func, linecolor=root.kRed,linestyle=root.kDashed,
        expand=False, label="Signif. as function of train. mass",
         size=22,labelfmt="L", textsize=10)

        axes[i].set_xlabel("Training mass [GeV]")
        axes[i].set_ylabel("Significance [-]")

        # Add the ATLAS Label
        aplt.atlas_label(text="Internal", loc="upper left")
        axes[i].text(0.2, 0.81,"Tested on: "+mass_arr[i], align=10)

        axes[i].legend(loc=(0.45, 1.1, 0.95, 0.70),textsize=20)
        figs[i].savefig("/home/lucas/Documents/KYR/bc_thesis/data_processed/imgs/train_test_tabnet/"+mass_arr[i]+".pdf")

    graph_all.Fit("pol5", "0")
    graph_max.Fit("pol5", "0")

    func_all = graph_all.GetFunction("pol5")
    func_all.SetRange(*ax_all.get_xlim())
    func_all.SetNpx(1000)
    ax_all.plot(func_all, linecolor=root.kRed + 2,linestyle=root.kDashed,
    expand=False, label="LQ all mass. train", size=22, labelfmt="L", textsize=10)

    func_max = graph_max.GetFunction("pol5")
    func_max.SetRange(*ax_all.get_xlim())
    func_max.SetNpx(1000)
    ax_all.plot(func_max, linecolor=root.kBlue,linestyle=root.kDashed, expand=False,
     label="LQ best sep. mass. train ", labelfmt="L")

    ax_all.set_xlabel("Testing mass [GeV]")
    ax_all.set_ylabel("Significance [-]")

    aplt.atlas_label(text="Internal", loc="upper left")
    ax_all.text(0.2, 0.81,"All masses VS One best", align=10)

    ax_all.legend(loc=(0.62, 0.95, 0.85, 0.85),textsize=20)
    img_all.savefig("/home/lucas/Documents/KYR/bc_thesis/data_processed/imgs/train_test_tabnet/all_max.pdf")