import os
import sys
import statistics

sys.path.insert(0,"/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
import numpy as np
import matplotlib.pyplot as plt

dvacet = [
       [
           1111.5981895225484,
1582.9838619266318,
1127.5269437297102,
1254.3934931697245,
795.249650833488,
1380.9062189756135,
1707.881575852268,
1761.5514931871303,
1496.8013067791444,
1600.125611823871
       ],
        [
816.666923335601,
1718.6999621354985,
2251.0550155811325,
1585.2897208566055,
1432.4187102146482,
1617.0518823626255,
1624.9363589399704,
2125.1508620948493,
2231.312547697614,
1619.7112836508982
        ],
        [
              751.1389825252339,
1667.2188564575804,
981.4849525491869,
1450.058918392972,
1752.6885502904902,
1601.8904214705674,
1079.9799856083132,
932.3452641276128,
1268.9991268784727,
1591.9738108302488
        ],
        [
               1278.3218641501173,
615.8908991122329,
451.3212794229611,
1214.6884316449816,
1034.9160789743964,
488.66132814020557,
850.629924259095,
751.9203670638716,
840.8492421999539,
1182.4214198726047

        ],
        [
             880.0780019837961,
647.9011122391,
1264.7763879357099,
800.1309653022917,
885.7883979779095,
1138.7130749892908,
552.3124027540094,
887.7904431037153,
758.0403211649062,
1055.8543728022348
        ],
        [
                1960.9260936838523,
1170.8483087437533,
1573.8347187965483,
1090.3157684403357,
1501.8645243807807,
1352.9914752152215,
1175.0175303054875,
1748.4713619830416,
1060.300044019513,
1117.4682333533192
        ],
        [
              1555.5248997280246,
1298.6487632893954,
1433.7368488727307,
1112.3085410039155,
1288.3292282633242,
1480.3825642562056,
1233.0067396503614,
1165.040995696757,
1351.266046290956,
1391.770815039293
        ],
        [
              1402.814089063656,
1472.5469723544218,
1104.5715812912858,
1301.1488455092506,
1030.6258478599905,
1174.4480922137404,
1490.2405443160674,
1333.795041088693,
1483.6433933975743,
1295.812496582931

        ]
]
ctyricet = [
        [
             1362.9041346302583,
1035.9859291398827,
874.4152057079614,
1247.3222733182706,
1535.1570783495567,
1518.0921839026362,
1195.467013066934,
1382.0916304520028,
1455.7206784063633,
788.3621334258879
        ],
        [
           1660.3571285238897,
1735.7877654383362,
1805.3259478856876,
1986.4278008189408,
1202.2454502327528,
1639.5412155062104,
1585.4592146011803,
1484.6356358123326,
1569.0967560924405,
1972.5508352192624
        ],
        [
           1361.4814904607974,
1546.158238944126,
1473.307762722669,
1124.7401822318075,
1460.0917466177664,
1639.257872257806,
1088.7414122456319,
967.0343292631403,
918.6829758739078,
1448.1453566678629
        ],
        [
            1117.287656554813,
1174.8833317276155,
918.9279181325913,
1255.527855395667,
1138.018360774946,
1563.819431582085,
1101.8205688602814,
1052.590366811743,
805.9060758643806,
1161.604948377172
        ],
        [
            915.4115328811878,
1149.1870698130122,
1296.56609507615,
1122.9281486905516,
876.6539771238744,
886.1603609079303,
1063.353946393261,
1057.0155073315148,
1108.983748772654,
1095.4424116020293
        ],
        [
            1435.0394062158027,
1581.72295405767,
1373.586322747949,
1662.976359001062,
878.0796670005625,
1682.2925810847003,
1448.6698563432599,
1503.3077900604933,
1285.9165520684076,
1350.7766696731
        ],
        [
            1461.0934419981936,
1237.1245278477063,
1436.313796845207,
1760.0924011985737,
1218.632622523604,
1603.855518850217,
1325.7443028099174,
1488.1037867989376,
1382.2855038739876,
1327.7285315155918
        ],
        [
             1379.8905212218638,
1585.710514229633,
1371.7176548796203,
1608.8049888215482,
1533.8642534383912,
1224.6848831233042,
1142.9783346167249,
1553.859850751808,
1379.5291078634486,
1291.7287140215062
        ]
]

sedesat = [
[
      1006.7319600034438,
1479.028965017374,
1239.2497318452952,
1381.1522998741805,
1338.424749634692,
1444.2120703278247,
1464.938386675261,
1316.5154530066952,
1209.9202151980162,
1467.7894616053463,

],
        [
              1522.4419471885524,
1295.3921745432128,
1676.7875715039818,
1574.491996434911,
1537.254852405441,
1601.5876676967448,
1844.8708113849536,
1850.013284220551,
1669.1451654303319,
1612.1919032921137,
        ],
        [
            1516.973162420016,
1189.904163495756,
1603.1057588597273,
1639.2803133039515,
1579.571524278592,
1356.5420306672936,
1560.0996003454586,
1505.2343765030844,
1661.4447532394827,
1498.5289034388318
        ],
        [
             1282.8366307763104,
1275.3902675489435,
1269.3990385705652,
1058.1395266672123,
977.9274538896902,
1043.892269362334,
1064.7415615955053,
1143.4590736557234,
1191.5417834042844,
958.8719458544272,
        ],
        [
              1167.1099153246043,
897.0227572260669,
931.1027794973112,
1012.0923068010877,
852.2744941125093,
1200.191228196577,
1056.9186674510304,
1071.8831279187812,
1089.2178867555417,
894.8418363603162,
        ],
        [
               1460.4318545585863,
1620.9434672477905,
1560.1829413267167,
1705.0846170662721,
1645.5096200875298,
1402.5463694249406,
1668.0090304221665,
1395.2917770733748,
1212.8418531380717,
1509.0623381356388
        ],
        [
             1555.6079594525993,
1452.3808416990146,
1405.0852495943357,
1470.8816356212176,
1428.7881844774874,
1557.515102732363,
1377.1062918829234,
1475.2570442345914,
1418.554912022418,
1555.1628312642047
        ],
        [
            1182.9995664403518,
1324.790243372358,
1176.24011583348,
1400.938623706216,
1526.3690860355034,
1167.8431566541017,
1337.8602275372687,
1315.3454152540235,
1402.6545851688802,
1586.6774481203481
        ]
]


osmdesat = [
[
        1523.9310430876865,
1475.8985725340308,
1465.0861377265908,
1394.1794412641032,
1055.2133044315747,
1465.1445809114457,
1491.1875112654434,
1433.3196019384172,
1329.3359082307186,
1450.5445064309474,
],
        [
             1635.4950950436157,
1580.636037743148,
1478.7860206227724,
1730.1805310945695,
1883.9826241405256,
1608.0571179923727,
1781.8104817541018,
1779.574323172543,
1817.9824624379096,
1743.1813897958748,
        ],
        [
             1508.887819432907,
1516.8654387369688,
1325.9520396968915,
1170.4616657536305,
1508.2818191383558,
1647.8097929107867,
1506.7563275055224,
1546.5841403322613,
1664.4485625984858,
1601.8813231258966,
        ],
        [
             1112.5326276605479,
1449.6244032759653,
1135.228409429922,
1139.8579595766803,
1068.5327321020386,
1096.1079382453347,
1158.5362838004617,
1020.9976692740358,
1556.6784242975555,
1399.0903049816689,
        ],
        [
              1164.7438226805712,
1215.8574476783074,
1149.3224106759405,
1170.4171380396178,
1124.4458981159828,
1031.63352015151,
918.2779824661259,
1178.353200683819,
1040.3744388041969,
1122.255987434489,
        ],
        [
              1448.7986682400535,
1643.9147185471666,
1916.0332266100847,
1769.0299219699,
1510.683387313601,
1659.072347555077,
1711.7547256257924,
1634.660385463254,
1762.8237782886222,
1673.312008108813,
        ],
        [
1433.4509237466452,
1406.5620348958787,
1553.3051351035108,
1367.8435600681353,
1307.9126008869227,
1627.3244735278658,
1476.62403875448,
1515.9151923549568,
1418.0522541995242,
1576.4248362792848,

        ],
        [
            1332.9301384566447,
1304.3689142710223,
1248.4801782214192,
1255.6778811775878,
1591.1134450643501,
1500.594641561203,
1432.0137857610962,
1335.3445319508685,
1463.7095451366456,
1582.3504441118293,
        ]
]

sto = [
        [
               1347.7641772361847,
1385.1486912544178,
1225.968707084698,
1103.5517802178629,
969.5742535369227,
1438.0180794712521,
1430.123765057713,
1316.6357205938605,
1361.7411219912608,
1557.0261247889969
        ],
        [
            1676.8823281778743,
1544.4139807622118,
1794.0473785997046,
1756.1290568848713,
1748.623802538359,
1727.3057596015349,
1644.4234400125822,
1856.2299238499086,
1666.7653041112978,
1766.803528342967
        ],
        [
            1546.1501412215039,
1607.5789407628079,
1274.3296421222988,
1200.5612135716724,
1350.9391345219697,
1478.915377488051,
1532.798855566001,
1239.4035192604063,
1420.644035288309,
1206.6030418412977
        ],
        [
              1309.5930187968272,
1277.3507091544946,
1330.7873336477724,
1187.059904986446,
1400.688402661083,
1340.565193097551,
1200.1959832175396,
1340.6347439408678,
1420.949179822425,
1158.3598196681507
        ],
        [
             1032.4464388632991,
1048.6804204694376,
1253.291522736009,
1103.792742502574,
1137.804609862152,
1065.6620370923547,
1125.6053728825414,
1064.4983922685021,
1182.367073824711,
1040.6460850889493
        ],
        [
            1654.1669234558265,
1521.982898043649,
1507.118129608067,
1687.097542708898,
1661.1807449259184,
1535.596023191829,
1615.5132733495957,
1483.1811546700324,
1675.6580756050337,
1588.6670958275486
        ],
        [
             1448.8292601380415,
1419.0925343482086,
1408.5697667712104,
1454.8638728038934,
1234.3964774770589,
1442.8017812043254,
1141.2319600160047,
1575.1413159583487,
881.0537665692572,
1768.899187282067
        ],
        [
               1423.0948007694647,
1269.060114203945,
1381.5758406271989,
1249.385560490848,
1089.248337223325,
1360.45487216365,
1327.981382818353,
1262.03201562815,
1503.0768798720817,
1362.426137975841
        ]
]

for i in range(8):
        dva_data = [
        dvacet[i][0],
        dvacet[i][1],
        dvacet[i][2],
        dvacet[i][3],
        dvacet[i][4],
                dvacet[i][5],
                dvacet[i][6],
                dvacet[i][7],
                dvacet[i][8],
                dvacet[i][9],

        ]
        ctyri_data = [
        ctyricet[i][0],
        ctyricet[i][1],
        ctyricet[i][2],
        ctyricet[i][3],
        ctyricet[i][4],
                ctyricet[i][5],
                ctyricet[i][6],
                ctyricet[i][7],
                ctyricet[i][8],
                ctyricet[i][9],

        ]

        sest_data = [
        sedesat[i][0],
        sedesat[i][1],
        sedesat[i][2],
        sedesat[i][3],
        sedesat[i][4],
        sedesat[i][5],
        sedesat[i][6],
        sedesat[i][7],
        sedesat[i][8],
        sedesat[i][9],

        ]
        osm_data = [
        osmdesat[i][0],
        osmdesat[i][1],
        osmdesat[i][2],
        osmdesat[i][3],
        osmdesat[i][4],
        osmdesat[i][5],
        osmdesat[i][6],
        osmdesat[i][7],
        osmdesat[i][8],
        osmdesat[i][9],

        ]
        deset_data = [
        sto[i][0],
        sto[i][1],
        sto[i][2],
        sto[i][3],
        sto[i][4],
        sto[i][5],
        sto[i][6],
        sto[i][7],
        sto[i][8],
        sto[i][9],
        ]

        x = ["XGBoost","LightGBM", "CatBoost", "TabNet", "MLP"]
        dva_mean = statistics.mean(dva_data)
        dva_stdev = statistics.stdev(dva_data)

        ctyri_mean = statistics.mean(ctyri_data)
        ctyri_stdev = statistics.stdev(ctyri_data)

        sest_mean = statistics.mean(sest_data)
        sest_stdev = statistics.stdev(sest_data)

        osm_mean = statistics.mean(osm_data)
        osm_stdev = statistics.stdev(osm_data)

        deset_mean = statistics.mean(deset_data)
        deset_stdev = statistics.stdev(deset_data)

        print(str(dva_mean)+',')
        print(str(ctyri_mean)+',')
        print(str(sest_mean)+',')
        print(str(osm_mean)+',')
        print(str(deset_mean)+',')
        print(" ")
        print(str(dva_stdev) + ',')
        print(str(ctyri_stdev) + ',')
        print(str(sest_stdev) + ',')
        print(str(osm_stdev) + ',')
        print(str(deset_stdev) + ',')
        print("-------------------------------")
        y = []
        y.append(dva_mean)
        y.append(ctyri_mean)
        y.append(sest_mean)
        y.append(osm_mean)
        y.append(deset_mean)
        e = []
        e.append(dva_stdev)
        e.append(ctyri_stdev)
        e.append(sest_stdev)
        e.append(osm_stdev)
        e.append(deset_stdev)

        plt.figure()
        plt.scatter(x, y)
        plt.errorbar(x, y,yerr=e, ls='None',fmt='o',capsize=10,color='black',
                     ecolor='red',label='mean significance with std. dev.')
        plt.ylabel("Significance [$\sigma$]")
        plt.xlabel("Classifier")
        plt.legend()
        plt.savefig("/home/lucas/Documents/KYR/bc_thesis/data_processed/imgs/performance_all/data_shrink_new"+str(i)+".png")
