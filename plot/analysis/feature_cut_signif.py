import os
import sys
from turtle import color

sys.path.insert(0,"/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
import numpy as np

import ROOT as root
import atlasplots as aplt

def feature_cut_signif():
    mass_arr = ['LQ 500','LQ 600','LQ 700','LQ 800','LQ 900','LQ 1000','LQ 1100','LQ 1200',
    'LQ 1300','LQ 1400','LQ 1500','LQ 1600',]

    cat = [1.9277983114939785,
3.1391337597905253,
2.938910109971109,
2.7101392651497256,
1.9076370609171835,
0.657269345771408,
2.324602871392195,
0.922912850170182,
0.5329280864767474,
0.34184087396430257,
0.350484897527557,
0.5192135427315062]

    cat = [
        2.1474183344617326,
2.1218803422814227,
2.3550651872506183,
2.2005144210123335,
1.9632435349657047,
1.9862442719545697,
2.088584833507871,
1.960395160090699,
2.1485445479881937,
1.8379298832837985,
2.14292124680047,
2.4437397415184026
    ]


    tabnet = [
        1.9103667758041771,
2.551591817747289,
2.528722999730817,
2.1741808182768856,
2.3690848893555176,
2.478061830240305,
2.2374544128064544,
2.786257743147252,
2.9374367269968187,
1.9149701599289177,
2.1555917523003947,
2.368713194486336]

    tabnet = [
        2.8742003667861518,
2.3529506002643172,
2.3000358308349007,
2.2577438804858305,
1.925235420325598,
1.914491726607149,
2.0137709461138633,
1.9388960458548634,
2.1952173976294467,
1.88187965470548,
2.059264612274501,
2.4011863773394473
    ]

    xgb =[
        0.9257338218876651,
1.7091701253935256,
1.8523680654419081,
1.8657767787755357,
1.5855001932287243,
1.2419073439284796,
1.185625777745038,
0.9300758538624369,
0.9154205455864934,
0.5501360936619015,
0.4847988880916038,
0.390578178546852]

    xgb = [
        1.3846213562728882,
1.4112034784733933,
1.5652533637254418,
1.7342000892925724,
1.6027062520863387,
1.5317987127990793,
1.6558579847074035,
1.6732769712214683,
1.8591782506610546,
1.6481788334386878,
1.7992728752775253,
2.078833800881345
    ]

    lgb=[
        3.1347790960844986,
3.8476428321885097,
3.6836127075052274,
3.093666697543711,
2.9131237911912784,
3.3461632934297594,
2.914894583977325,
3.4734875022219276,
3.2451488568504385,
2.288445238026571,
2.5202655905526203,
2.6506897200712536]

    lgb = [
        2.802342232905456,
2.473522721652862,
2.3621179680511784,
2.326213567021541,
1.9290864676293857,
1.9288212626776637,
2.057841442905036,
1.9635685919593393,
2.133639060172994,
1.9046974944702952,
2.0476087864693886,
2.4272245119219384
    ]

    mlp = [1.107, 1.282, 1.190, 1.080, 1.297, 1.092, 1.153, 1.135, 0.652, 0.473, 0.415, 0.352]

    mlp = [
        0.7760691947015536,
0.6562447201594996,
0.7905478605519649,
1.005046072561659,
1.0561940241825505,
1.541669924952308,
1.4196046000506037,
1.2420297514631138,
1.7994012503946493,
1.54375144685524,
1.8920539818037727,
2.1707814903871485
    ]

    for i, val in enumerate(mlp):
        mlp[i] = mlp[i]*1/(1-0.84)/5

    # Set the ATLAS Style
    aplt.set_atlas_style()

    # Create a figure and axes
    fig, ax = aplt.subplots(1, 1, name="fig1", figsize=(800, 600))

    # Plot a line with Gaussian noise
    #x = np.arange(20)
    x = np.linspace(500, 1600, 12)


    graph = ax.graph(x, cat, labelfmt="EP", options="",linecolor=root.kRed)
    graph_2 = ax.graph(x, tabnet, labelfmt="EP", options="",linecolor=root.kBlue)
    graph_3 = ax.graph(x, xgb, labelfmt="EP", options="",linecolor=root.kGreen)
    graph_4 = ax.graph(x, lgb, labelfmt="EP", options="",linecolor=root.kOrange)
    graph_5 = ax.graph(x, mlp, labelfmt="EP", options="", linecolor=root.kMagenta)
    

    # Fit the graph; store graphics func but do not draw
    graph.Fit("pol9", "0")
    graph_2.Fit("pol9", "0")
    graph_3.Fit("pol9", "0")
    graph_4.Fit("pol9", "0")
    graph_5.Fit("pol9", "0")

    # Add extra space at top of plot to make room for labels
    ax.add_margins(top=0.18, left=0.2, right=0.05, bottom=0.05)

    
    # Plot fit function and extend its range to fill plot area
    func = graph.GetFunction("pol9")
    # func.SetRange(*ax.get_xlim())
    # func.SetNpx(1000)
    ax.plot(func, linecolor=root.kRed,linestyle=root.kDashed, expand=False, label="CATB", size=22, labelfmt="L", textsize=10)

    func_2 = graph_2.GetFunction("pol9")
    # func_2.SetRange(*ax.get_xlim())
    # func_2.SetNpx(1000)
    ax.plot(func_2, linecolor=root.kBlue,linestyle=root.kDashed, expand=False, label="TABNET", labelfmt="L")

    func_3 = graph_3.GetFunction("pol9")
    # func_3.SetRange(*ax.get_xlim())
    # func_3.SetNpx(1000)
    ax.plot(func_3, linecolor=root.kGreen,linestyle=root.kDashed, expand=False, label="XGB", labelfmt="L")

    func_4 = graph_4.GetFunction("pol9")
    # func_4.SetRange(*ax.get_xlim())
    # func_4.SetNpx(1000)
    ax.plot(func_4, linecolor=root.kOrange,linestyle=root.kDashed, expand=False, label="LGBM", labelfmt="L")

    func_5 = graph_5.GetFunction("pol9")
    # func_4.SetRange(*ax.get_xlim())
    # func_4.SetNpx(1000)
    ax.plot(func_5, linecolor=root.kMagenta,linestyle=root.kDashed, expand=False, label="MLP", labelfmt="L")
    
    # Set axis titles
    ax.set_xlabel("m_{LQ^{d}_{3}} [GeV]")
    ax.set_ylabel("Significance [#sigma]")


    # Add the ATLAS Label
    ax.set_xlim(500,1600)
    aplt.atlas_label(text="Internal", loc="upper left")
    ax.text(0.2, 0.81,"all the masses vs. one mass", align=10)
    # Add legend
    ax.legend(loc=(0.78, 0.9, 0.95, 0.70),textsize=20)
    # Save the plot as a PDF
    fig.savefig("/home/lucas/Documents/KYR/bc_thesis/data_processed/imgs/all_one_t_c_x_l/all_max.pdf")
