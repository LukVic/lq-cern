import os
import sys
from turtle import color

sys.path.insert(0,"/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
import numpy as np

import ROOT as root
import atlasplots as aplt

def data_cut():

    mass_arr = ['LQ 500','LQ 600','LQ 700','LQ 800','LQ 900','LQ 1000','LQ 1100','LQ 1200',
    'LQ 1300','LQ 1400','LQ 1500','LQ 1600',]
    set_10_99 = [[ 0.3332544523614893,
 0.32911526352753684,
 0.42637415462470546,
 0.376950045348905,
 0.3733378744423685,
 0.2034821966269893,
 0.3410322089311678,
 0.3225608852024233,
 0.2602099109357473,
 0.1906036834230128,
 0.249683083749834,
 0.325430055393965]
    ,[
        0.4776134227412641,
0.6513308084036182,
0.7120558392300022,
0.6094845379218035,
0.7659439554502399,
0.5663229771433144,
0.6533913390815523,
0.6387568055706098,
0.5571149711130388,
0.41129679090052607,
0.4577748020186936,
0.5630839306273585
]
    ,[
        0.5859183605770149,
0.8421908249169064,
0.9072290537959791,
0.9454824610268544,
1.1784214024050892,
0.8773323567600293,
0.9317187790469652,
0.9506072328092006,
0.7405600245689042,
0.6182958586322397,
0.7872099566744151,
0.7782417880436538
    ]
    ,[
        0.9310954172511567,
1.2828253202007498,
1.242054988989415,
1.198738128607665,
1.5864389524485023,
1.2073430169538033,
1.2628760012025722,
1.2625185453441499,
1.0766027639263391,
0.7832480843408879,
1.1008796906813563,
1.0590126338607255
],[
        1.2599082074892118,
1.5860028973182925,
1.7629249359104315,
1.5052344355966394,
1.8814030945825235,
1.7008833956186085,
1.4909879782700084,
1.6265775826778295,
1.4855928183414948,
0.9771446313664721,
1.3444658353103471,
1.2423922915685268
    ]
    ,[
        1.5892191711633037,
1.8934754893923202,
1.993003229469477,
1.7512688275803925,
1.9728914728326836,
2.0364661690663968,
1.718959717762886,
1.960504463323158,
1.7175252018530234,
1.2640257139453646,
1.4402847421948906,
1.5133828165524823
    ]
    ,[
        1.7582736398119683,
2.2311815358799207,
2.3547890821324984,
1.9605523245861953,
2.4909485808100817,
2.483109692919498,
2.098169358102392,
2.2679309258317613,
2.1288794471235373,
1.6118646986756235,
1.7573041585387432,
1.7207190491851658
    ]
    ,[
        2.0551747941027676,
2.480715504870704,
2.5643001470447895,
2.061795014840198,
2.7524493876118394,
2.757427824775784,
2.4092733680557354,
2.608357050834093,
2.3513949986119145,
1.7558641666471952,
1.9640601660320112,
1.9455064114970655
    ]
    ,[
        2.2700910929725135,
2.785640616413188,
2.7017167480891384,
2.4324905610144154,
3.0267296688205376,
3.0355204973828305,
2.5892189879149936,
2.804518327757047,
2.7134552739939446,
1.9836287555626815,
2.1324217855331162,
2.3115872074105024
    ]
    ,[
        2.3701428887681373,
2.83027200639868,
3.1589800780355017,
2.754529542967744,
3.1622925521153142,
3.1718885055725115,
2.9340816701344945,
3.3033023621648754,
3.076090662227486,
2.316436021558171,
2.387361404923281,
2.528090530038918
    ]]
    # Set the ATLAS Style
    aplt.set_atlas_style()
    figs = [None] * 12
    axes = [None] * 12
    x = np.linspace(0, 100, 10)
    x_2 = np.linspace(800, 1200, 21)
    y_2 = [0.00335714285714286,0.00321428571428571,0.00307142857142857,0.00299641362821279
,0.00283333333333333,0.00277777777777778,0.00266666666666667,0.00261111111111111
,0.00255555555555556,0.00244444444444444,0.00238888888888889,0.00238888888888889
,0.00233333333333333,0.00233333333333333,0.00227777777777778,0.00227777777777778
,0.00222222222222222,0.00216666666666667,0.00211111111111111,0.00205555555555556
,0.002]
    all = []
    graphs = []
    graphs_2 = []
   


    for i in range(12):
        tmp = []
        for j in range(10): 
            tmp.append(set_10_99[j][i]) 
        all.append(tmp)
    
    print(all)
    for k in range(12):
        print(len(all[k]))
    # Create a figure and axes
    for i in range(12):
        figs[i], axes[i] = aplt.subplots(1, 1, name=mass_arr[i], figsize=(800, 600))
        graphs.append(axes[i].graph(x, all[i], labelfmt="EP", options="P"))
        graphs_2.append(axes[i].graph(x_2, y_2, labelfmt="EP", options=" "))

    for i, (graph, graph_2) in enumerate(zip(graphs, graphs_2)):
        graph.Fit("pol1","0")
        func = graph.GetFunction("pol1")
        func.SetRange(*axes[i].get_xlim())
        func.SetNpx(1000)
        axes[i].plot(func, linecolor=root.kRed,linestyle=root.kDashed,
        expand=False, label="Signif. as func. of statistics",
         size=22,labelfmt="L", textsize=10)

        graph_2.Fit("pol5","0")
        func_2 = graph_2.GetFunction("pol5")
        func_2.SetRange(*axes[0].get_xlim())
        func_2.SetNpx(1000)
        axes[i].plot(func_2, linecolor=root.kRed,linestyle=root.kDashed,
        size=22,labelfmt="L", textsize=10)

        axes[i].set_xlabel("Used simulated data [%]")
        axes[i].set_ylabel("Significance [\sigma]")

        # Add the ATLAS Label
        aplt.atlas_label(text="Internal", loc="upper left")
        axes[i].text(0.2, 0.81,"Tested on: "+mass_arr[i], align=10)
        axes[i].text(0.2, 0.74,"Classifier: Tabnet", align=10)
        axes[i].set_xlim(0,150)
        axes[i].set_ylim(0,5)
        axes[i].legend(loc=(0.45, 1.1, 0.95, 0.70),textsize=20)
        figs[i].savefig("/home/lucas/Documents/KYR/bc_thesis/data_processed/imgs/data_cut_tabnet/"+mass_arr[i]+".pdf")
