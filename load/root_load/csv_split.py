import logging
import math
import os
import sys
import csv
from agents.classify import CLASSIFIER, train


sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/utils/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/")

import numpy as np
from config.conf import setup_logging
import config.constants_new as C
# import config.constants as C
import pandas as pd
import ROOT
import ml_utils
from ml_utils import *
from signif_ref import signif_ref

PATH = "/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/"
FEATURES_FILE = "features_with_taus_vectorized_withoutOR.csv.csv"
#{'lq':0,'tth':1,'ttw':2,'ttz':3,'ttbar':4,'vv': 5,'other': 6}
def csv_split():
    print(PATH)
    X = ml_utils.load(PATH + 'lq_all/X.pkl')
    y = ml_utils.load(PATH + 'lq_all/y.pkl')
    y_binary = ml_utils.load(PATH + 'lq_all/lq_800/y_test_pred_TABNET.pkl')
    y_proba = ml_utils.load(PATH + 'lq_all/lq_800/y_test_pred_proba_TABNET.pkl')
    w = ml_utils.load(PATH + 'lq_all/w.pkl')
    f = ml_utils.load(PATH + 'lq_all/f.pkl')

    type(X)
    type(y)
    # for idx in range(len(y_binary)):
    #     if y_binary[idx] == 0:
    #         y_binary[idx] = 1
    #     else:
    #         y_binary[idx] = 0

    df_new = pd.DataFrame(data=X, columns=f)
    df_new['y'] = y
    # df_new['y_binary'] = y_binary
    # df_new['y_proba'] = y_proba[:,0]
    df_new['weight'] = w
    df_new['ratio'] = 1/(1-0.8)
    #df_new = df_new.astype({"y": int, "y_binary": int, "ratio": int})
    df_new = df_new.astype({"y": int, "ratio": int})
    # df_lq = df_new[df_new.y == 0]
    # df_tth = df_new[df_new.y == 1]
    # df_ttw = df_new[df_new.y == 2]
    # df_ttz = df_new[df_new.y == 3]
    # df_ttbar = df_new[df_new.y == 4]
    # df_vv = df_new[df_new.y == 5]
    # df_other = df_new[df_new.y == 6]

    # df_lq = df_lq.drop(columns=["y"])
    # df_tth = df_tth.drop(columns=["y"])
    # df_ttw = df_ttw.drop(columns=["y"])
    # df_ttz = df_ttz.drop(columns=["y"])
    # df_ttbar = df_ttbar.drop(columns=["y"])
    # df_vv = df_vv.drop(columns=["y"])
    # df_other = df_other.drop(columns=["y"])

    PATH_NEW = PATH + '/lq_all/lq_800//root/'

    # df_lq.to_csv(PATH_NEW + 'lq.csv')
    # df_tth.to_csv(PATH_NEW + 'tth.csv')
    # df_ttw.to_csv(PATH_NEW + 'ttw.csv')
    # df_ttz.to_csv(PATH_NEW + 'ttz.csv')
    # df_ttbar.to_csv(PATH_NEW + 'ttbar.csv')
    # df_vv.to_csv(PATH_NEW + 'vv.csv')
    # df_other.to_csv(PATH_NEW + 'other.csv')
    df_new.to_csv(PATH_NEW +''+ 'limit.csv')




    # df = pd.read_csv(PATH + "ds.csv")
    # features = list(pd.read_csv(C.FEATURES_DIRECTORY + FEATURES_FILE)["features_names"])
    # columns_in_order = features + ["y", "weight"]
    # df_new = df[columns_in_order]
    # print(df_new)
    # df_new['tth'] = df_new.y == 0
    # df_new['ttw'] = df_new.y == 1
    # df_new['ttz'] = df_new.y == 2
    # df_new['ttbar'] = df_new.y == 3
    # df_new['vv'] = df_new.y == 4
    # df_new['other'] = df_new.y =
    # = 5
    #
    # df_tth = df_new[df_new.tth]
    # df_ttw = df_new[df_new.ttw]
    # df_ttz = df_new[df_new.ttz]
    # df_ttbar = df_new[df_new.ttbar]
    # df_vv = df_new[df_new.vv]
    # df_other = df_new[df_new.other]
    #
    # df_tth = df_tth.drop(columns=["y","tth","ttw","ttz","ttbar","vv","other"])
    # df_ttw = df_ttw.drop(columns=["y","tth","ttw","ttz","ttbar","vv","other"])
    # df_ttz = df_ttz.drop(columns=["y","tth","ttw","ttz","ttbar","vv","other"])
    # df_ttbar = df_ttbar.drop(columns=["y","tth","ttw","ttz","ttbar","vv","other"])
    # df_vv = df_vv.drop(columns=["y","tth","ttw","ttz","ttbar","vv","other"])
    # df_other = df_other.drop(columns=["y","tth","ttw","ttz","ttbar","vv","other"])
    #
    # df_tth.to_csv(PATH + 'tth.csv')
    # df_ttw.to_csv(PATH + 'ttw.csv')
    # df_ttz.to_csv(PATH + 'ttz.csv')
    # df_ttbar.to_csv(PATH + 'ttbar.csv')
    # df_vv.to_csv(PATH + 'vv.csv')
    # df_other.to_csv(PATH + 'other.csv')









def main():
    csv_split()

if __name__ == '__main__':
    main()



