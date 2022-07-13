import os
import sys
import statistics

sys.path.insert(0,"/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
import numpy as np
import matplotlib.pyplot as plt

xgb = [
        [1154.403446390471,
1319.4256626417323,
 1512.942574849907,
 1205.9961674622866

],
        [
           1186.257475630833,
1504.3933943118568,
 1167.5474098111542,
 1239.0636588134266

        ],
[
1351.3221338386725,
1375.8790601381825,
 1321.1693279404888,
 1168.9502954175437


],
        [1155.950220641904,
1577.4445030517181,
 1411.158238763401,
 1107.4293584166603
],
        [1102.0775396867825,
1789.0290621428842,
 1310.8982035954887,
 1274.9512327580924

],
        [
             1289.6680938437873,
1610.9663623537251,
 1384.8663662167028,
 1185.9593693225129

        ],
        [
              1162.4176123761545,
1528.7359071387716,
 1205.8148375908575,
 1299.9951197571354
        ],
        [
            1082.4104893415738,
1559.8905837493849,
 1223.4152701580615,
 1152.3599368306263
        ],
        [
                1230.0848385204201,
1605.9008913308148,
 1271.1050701250701,
 1227.755220772829
        ],
        [
            1366.5477310869119,
1742.136695465186,
 1383.4630475518384,
 997.4765719196043

        ]
]

mlp = [
[
        1493.3089091531679,
1335.735981721375,

1535.234779927964,

1518.6877309469735
],
        [1299.0645172095296,
1634.471894612976,

1409.491813081735,

1315.711294850349
],
[
     1277.6544628051738,
1556.329201415682,

1504.5698066334573,

1165.4732950566229
],
[
    1426.7391398534005,
1569.6909236875658,

1443.1791280073276,

1332.9590171779703
],
[
1471.7277500643982,
1745.830535737944,

1423.7977218771541,

1236.1781416800204
],
[
        1493.3089091531679,
1335.735981721375,

1535.234779927964,

1518.6877309469735
],
        [1299.0645172095296,
1634.471894612976,

1409.491813081735,

1315.711294850349
],
[
     1277.6544628051738,
1556.329201415682,

1504.5698066334573,

1165.4732950566229
],
[
    1426.7391398534005,
1569.6909236875658,

1443.1791280073276,

1332.9590171779703
],
[
1471.7277500643982,
1745.830535737944,

1423.7977218771541,

1236.1781416800204
]
]

lgb = [
[
        1348.9580541535463,
1770.2228076156848,
 1327.59809924072,
 1324.4750441612534
],
        [
           1272.6916416699241,
1561.6486224232435,
 1699.2091256756007,
 1554.6917566477423
        ],
        [
               1242.2263835219974,
1682.977450181012,
 1653.0950872905212,
 1407.5632743807678
        ],
        [
              1225.3189119928993,
1794.121000381687,
 1564.6131260363345,
 1310.9404435664921
        ],
        [
             1193.1525124598497,
1681.5621507844198,
 1519.4360106853612,
 1305.7598537529232
        ],
        [
             1236.2181522067572,
1496.3930408558037,
 1463.8872490699848,
 1377.9117185691143
        ],
        [
              1354.0678626953907,
1634.2626402047204,
 1393.9882418542718,
 1321.7181402923227
        ],
        [
              1305.9015590981853,
1741.8534877179984,
 1438.4602906518471,
 1359.562774720452
        ],
        [
             1104.7990320503097,
1635.548707105484,
 1573.2774319887335,
 1340.9635303256914
        ],
        [
             1303.4852885433813,
1757.6224691871155,
 1481.9114589340486,
 1276.603011495471
        ]
]
cat = [
[
        1013.3502279794591,
1586.9714974142744,
 1547.6671155754543,
 1429.711246827818
],
        [
               1062.145059947159,
1561.869620164917,
 1524.1477586884148,
 1307.9004958319338
        ],
        [
              1224.9720925815986,
1524.7835628000141,
 1434.312931295131,
 1385.685125885722
        ],
        [
            1230.3487343432062,
1668.7272356759895,
 1418.8132484036435,
 1349.5186052601916
        ],
        [
             1116.9662007192865,
1677.8393542051529,
 1473.270906485425,
 1288.6053721350718
        ],
        [
              1071.9284920868006,
1785.7050927823236,
 1384.8015567527843,
 1297.9849569925227
        ],
        [
              1086.2997963220837,
1678.4606919453445,
 1512.8893525992619,
 1099.546920452673
        ],
        [
           1205.5741548441756,
1655.919132781563,
 1534.3340623696101,
 1320.6273610687856
        ],
        [
           1048.451456181658,
1387.5117587294706,
 1519.5439792969307,
 1347.7920977917447
        ],
        [
            1153.721763405335,
1659.2922513811466,
 1470.0814258769608,
 1155.7139394126427
        ]
]

tabnet = [
[1433.2218589679726,
1495.6792731895764,
 1520.0252380324473,
 1573.9137488098481
],
        [
             1138.752075832084,
1735.543352612734,
 1419.7620643047233,
 1422.04543293922
        ],
        [
             1170.3544809483112,
1862.2839834672288,
 1583.1480562053646,
 1536.4896617173642
        ],
        [
           1356.672894382449,
1785.2909152479767,
 1575.7262844433383,
 1159.929914551086
        ],
        [
             1159.7884191597857,
1613.2021278993798,
 1207.4107594261543,
 1256.3249889837336
        ],
        [
              1297.60794039899,
1810.6963714165968,
 1464.8172787003718,
 1569.4503108902027
        ],
        [
             1185.42639355385,
1724.2581986075447,
 1298.159328311559,
 1181.6467298295297
        ],
        [
              1529.578709635474,
1761.5494828037938,
 1399.501253656868,
 1347.927095127747
        ],
        [
             1437.5271678405409,
1704.9622072227326,
 1523.9733348773411,
 1294.1529159826462
        ],
        [
              1407.992706236032,
1892.3540502428648,
 1610.3037000535312,
 1305.5632234240663
        ]
]
for i in range(4):
        tabnet_data = [
        tabnet[0][i],
        tabnet[1][i],
        tabnet[2][i],
        tabnet[3][i],
        tabnet[4][i],
        tabnet[5][i],
        tabnet[6][i],
        tabnet[7][i],
        tabnet[8][i],
        tabnet[9][i]
        ]
        mlp_data = [
        mlp[0][i],
        mlp[1][i],
        mlp[2][i],
        mlp[3][i],
        mlp[4][i],
                mlp[5][i],
                mlp[6][i],
                mlp[7][i],
                mlp[8][i],
                mlp[9][i],
        ]

        xgb_data = [
        xgb[0][i],
        xgb[1][i],
        xgb[2][i],
        xgb[3][i],
        xgb[4][i],
                xgb[5][i],
                xgb[6][i],
                xgb[7][i],
                xgb[8][i],
                xgb[9][i]
        ]
        lgb_data = [
        lgb[0][i],
        lgb[1][i],
        lgb[2][i],
        lgb[3][i],
        lgb[4][i],
                lgb[5][i],
                lgb[6][i],
                lgb[7][i],
                lgb[8][i],
                lgb[9][i]
        ]
        cat_data = [
        cat[0][i],
        cat[1][i],
        cat[2][i],
        cat[3][i],
        cat[4][i],
                cat[5][i],
                cat[6][i],
                cat[7][i],
                cat[8][i],
                cat[9][i],
        ]

        x = ["XGBoost","LightGBM", "CatBoost", "TabNet", "MLP"]
        mass = ['600', '900', '1200', '1500']
        tabnet_mean = statistics.mean(tabnet_data)
        tabnet_stdev = statistics.stdev(tabnet_data)

        mlp_mean = statistics.mean(mlp_data)
        mlp_stdev = statistics.stdev(mlp_data)

        xgb_mean = statistics.mean(xgb_data)
        xgb_stdev = statistics.stdev(xgb_data)

        lgb_mean = statistics.mean(lgb_data)
        lgb_stdev = statistics.stdev(lgb_data)

        cat_mean = statistics.mean(cat_data)
        cat_stdev = statistics.stdev(cat_data)

        y = []
        y.append(xgb_mean)
        y.append(lgb_mean)
        y.append(cat_mean)
        y.append(tabnet_mean)
        y.append(mlp_mean)
        e = []
        e.append(xgb_stdev)
        e.append(lgb_stdev)
        e.append(cat_stdev)
        e.append(tabnet_stdev)
        e.append(mlp_stdev)

        plt.figure()
        plt.scatter(x, y)
        plt.errorbar(x, y,yerr=e, ls='None',fmt='o',capsize=10,color='black',
                     ecolor='red',label='mean significance with std. dev. \n '+str(mass[i]) + ' GeV')
        plt.ylabel("Significance [$\sigma$]", fontsize=16)
        plt.xlabel("Classifier", fontsize=16)
        plt.legend(fontsize=16)
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
        plt.savefig("/home/lucas/Documents/KYR/bc_thesis/data_processed/imgs/performance_all/performance_all_new"+str(i)+".png")
