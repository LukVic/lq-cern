import os
import sys
import statistics

sys.path.insert(0,"/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
import numpy as np
import matplotlib.pyplot as plt

all = [
[
    1179.4373326493098,
 1173.4037013688608,
 1358.9148827861115,
],
    [
        1041.7613422269285,
 1124.383862191384,
 1463.6229711541603
    ],
    [
        1132.5220917667584,
 1245.2189695722657,
 1487.7816033339745
    ],
    [
        1000.5243436002761,
 1181.1925911281378,
 1244.4179671236393
    ],
    [
        1086.8355696776705,
 1064.5478676896494,
 1468.7424844722493
    ],
    [
        1447.8069864929455,
 1319.6857084879102,
 1061.663176792473
    ],
    [
       993.213069145022,
 972.9765666147116,
 1504.1413056002
    ],
    [
        1287.527156011131,
 1082.9598692355091,
 1454.1896764271276
    ],
    [
        1216.777221752435,
 1163.3970706577115,
 1428.575736199293
    ],
    [
        1532.0553418428308,
 1216.634738325361,
 1437.824050812136
    ]
]

_600 = [
[
    1249.4463375781359,
 311.6998430790763,
 0.04113324319172504
],
    [
        1203.8367213965269,
 275.82355089663974,
 0.03962612337182902
    ],
    [
        1183.3738528775623,
 289.96650176197306,
 0.03414940358237259
    ],
    [
        1275.1757439900741,
 326.90021656599845,
 0.03319973149303479
    ],
    [
        1162.4624228251164,
 303.89563306648284,
 0.029638470227266192
    ],
    [
        1269.3572101893187,
 313.3636729497777,
 371.89261882373813
    ],
    [
        1308.6871463429627,
 277.68030958908435,
 255.0922326545847
    ],
    [
        1366.9884192866227,
 301.351864215291,
 343.5869645815316
    ],
    [
        1348.7686232279702,
 295.55445504251196,
 335.41949275728825
    ],
    [
        1349.3685664493419,
 269.8496641301459,
 322.20652897962657
    ]
]

_1000 = [
[
    85.11486394728703,
 1122.2089641147925,
 0.007721954227151447
],
    [
        78.80121958320132,
 1105.9727351859647,
 0.00780532145066134
    ],
    [
        84.12293901846462,
 1258.2873411227067,
 0.0409804632695902
    ],
    [
        74.43451976672345,
 1145.4164246531736,
 0.008767726645188571
    ],
    [
        78.29317371331219,
 1114.864550023316,
 0.00768083365986066
    ],
    [
        273.0484788068283,
 1117.952769935431,
 299.8014983345579
    ],
    [
        302.06716049217687,
 1236.178910637751,
 77.81334670053691
    ],
    [
        271.96997274405294,
 1019.5276274559948,
 348.1305794104741
    ],
    [
        286.3528782314176,
 1096.40942669511,
 277.98870110179604
    ],
    [
        272.1978763035227,
 1018.7678903668119,
 343.0408038210227
    ]
]
_1500 = [
[
    82.19166275591353,
 62.620499725716535,
 145.69415877639787
],
    [
        77.3878233122394,
 57.820884846497115,
 146.40672113686767
    ],
    [
        83.916339446781,
 67.4065257417614,
 159.99826816071925
    ],
    [
        85.58704979514924,
 55.79533041174012,
 131.87802363795237
    ],
    [
        84.19945006219993,
 67.84405616206625,
 132.72456144254073
    ],
    [
        83.63762633631984,
 65.0089644949361,
 1217.9338313118938
    ],
    [
        82.43446290336169,
 56.446473333716646,
 1159.974439048025
    ],
    [
        94.18165734353273,
 65.15049087061395,
 1211.5843156521416
    ],
    [
        77.35033969473253,
 67.80021502381646,
 1058.180343639342
    ],
    [
        78.57012518804221,
 63.9777674063914,
 896.6487096780801
    ]
]

for i in range(3):
        all_data = [
        all[0][i],
        all[1][i],
        all[2][i],
        all[3][i],
        all[4][i],
        all[5][i],
        all[6][i],
        all[7][i],
        all[8][i],
        all[9][i]
        ]

        _600_data = [
        _600[0][i],
        _600[1][i],
        _600[2][i],
        _600[3][i],
        _600[4][i],
        _600[5][i],
        _600[6][i],
        _600[7][i],
        _600[8][i],
        _600[9][i]
        ]

        _1000_data = [
            _1000[0][i],
            _1000[1][i],
            _1000[2][i],
            _1000[3][i],
            _1000[4][i],
            _1000[5][i],
            _1000[6][i],
            _1000[7][i],
            _1000[8][i],
            _1000[9][i]
        ]

        _1500_data = [
            _1500[0][i],
            _1500[1][i],
            _1500[2][i],
            _1500[3][i],
            _1500[4][i],
            _1500[5][i],
            _1500[6][i],
            _1500[7][i],
            _1500[8][i],
            _1500[9][i]
        ]

        x = ["XGBoost","LightGBM", "CatBoost", "TabNet", "MLP"]
        tabnet_mean = statistics.mean(all_data[0:4])
        tabnet_stdev = statistics.stdev(all_data[0:4])

        mlp_mean = statistics.mean(_600_data[0:4])
        mlp_stdev = statistics.stdev(_600_data[0:4])

        xgb_mean = statistics.mean(_1000_data[0:4])
        xgb_stdev = statistics.stdev(_1000_data[0:4])

        lgb_mean = statistics.mean(_1500_data[0:4])
        lgb_stdev = statistics.stdev(_1500_data[0:4])


        tabnet_mean2 = statistics.mean(all_data[5:])
        tabnet_stdev2 = statistics.stdev(all_data[5:])

        mlp_mean2 = statistics.mean(_600_data[5:])
        mlp_stdev2 = statistics.stdev(_600_data[5:])

        xgb_mean2 = statistics.mean(_1000_data[5:])
        xgb_stdev2 = statistics.stdev(_1000_data[5:])

        lgb_mean2 = statistics.mean(_1500_data[5:])
        lgb_stdev2 = statistics.stdev(_1500_data[5:])


        print(str(tabnet_mean) + ',')
        print(str(mlp_mean) + ',')
        print(str(xgb_mean) + ',')
        print(str(lgb_mean) + ',')
        print(str(tabnet_mean2) + ',')
        print(str(mlp_mean2) + ',')
        print(str(xgb_mean2) + ',')
        print(str(lgb_mean2) + ',')
        print(" ")
        print(str(tabnet_stdev) + ',')
        print(str(mlp_stdev) + ',')
        print(str(xgb_stdev) + ',')
        print(str(lgb_stdev) + ',')
        print(str(tabnet_stdev2) + ',')
        print(str(mlp_stdev2) + ',')
        print(str(xgb_stdev2) + ',')
        print(str(lgb_stdev2) + ',')
        print("-------------------------------")

print("STOP")