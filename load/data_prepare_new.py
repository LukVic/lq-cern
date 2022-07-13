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
from ml_utils import save
from signif_ref import signif_ref
NORMAL = 0
EXTERN = 1
AUGMENT = 0
FEATURE_CUT = 0


def prepare_stage_3(INPUT, OUTPUT, FEATURES_FILE, lq_mass, lq_mass_f_cut, lq_scale, phi_shift_bound_1,
phi_shift_bound_2, signifs ,compensation,EXTERN_PATH, stamp):
    if stamp == "TRAIN":
       DICT = C.all_classes_to_num_dict_train(lq_mass)
        # DICT = C.ALL_CLASSES_TO_NUM_DICT_TRAIN
    else:
       DICT = C.all_classes_to_num_dict_test(lq_mass)
        # DICT = C.ALL_CLASSES_TO_NUM_DICT_TEST

    features = list(pd.read_csv(C.FEATURES_DIRECTORY + FEATURES_FILE)["features_names"])
    file_names = os.listdir(INPUT)
    
    file_names = filter(lambda x: os.path.splitext(x)[1] == ".csv", file_names)
    setup_logging("PreparationLogger", file_handler=False)
    logger = logging.getLogger("PreparationLogger")

    #feature cut to experiment with
    feature_cut = []
    if FEATURE_CUT == 1:
        if stamp == "TRAIN":
            path = "/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/" +lq_mass+ "/"+ lq_mass_f_cut+ "/"+ "imgs" + "/" + CLASSIFIER+"_features.csv"
        else:
            path = "/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/" +lq_mass_f_cut+ "/"+ lq_mass+ "/"+ "imgs" + "/" + CLASSIFIER+"_features.csv"
        file = open(path, "r")
        csv_reader = csv.reader(file)
        for row in csv_reader:
            feature_cut.append(row[0])
        #big_df = pd.DataFrame(columns=feature_cut)
        features = feature_cut

    #ADDING SPECIAL COLUMN OF LQ ALL
    if lq_mass == 'lq_all':
        columns_in_order = features + ["y_lq_all","y", "weight"]
    else:
        columns_in_order = features + ["y", "weight"]
    
    big_df = pd.DataFrame(columns = columns_in_order)   
    
    # DATA AUGMENTATION
    
    for file in file_names:
        #WEIGHTS
        shift_index = 10
        for cl in DICT:
            if cl in str(file):
                shift_index = DICT[cl]
            # if cl in file:
            #     global_index =
        #if "lq" in file:
        features_to_change = ["taus_phi_0", "lep_Phi_0", "lep_Phi_1", "met_phi", "jet_tauOR_phi_VECTORIZED_0",
                                "jet_tauOR_phi_VECTORIZED_1", "jet_tauOR_phi_VECTORIZED_2", "jet_tauOR_phi_VECTORIZED_3",
                                "jet_tauOR_phi_VECTORIZED_4", "jet_tauOR_phi_VECTORIZED_5"]
#           features_to_change = ["jet_tauOR_phi_VECTORIZED_0"]
        df = pd.read_csv(INPUT + "/" +file)
        df = df[columns_in_order]
        df_prev = df
        phi_shift = phi_shift_bound_1
        counter = 0
        if AUGMENT == 1:
            while(phi_shift <= phi_shift_bound_2):
                counter = counter + 1
                for jdx, fea in enumerate(features_to_change):
                    for idx, sample in enumerate(df[fea].to_numpy()):
                        phi_cur = df[fea].to_numpy()[idx]
            
                        if phi_cur < math.pi - phi_shift:
                            df[fea].to_numpy()[idx] = phi_cur + phi_shift
                        else:
                            df[fea].to_numpy()[idx] = - (math.pi - phi_cur + phi_shift)
                big_df = pd.concat([big_df, df])
                #   phi_shift = phi_shift + phi_shifts[shift_index]
            big_df = pd.concat([big_df, df_prev])

        df = pd.read_csv(INPUT + "/" +file)
        logger.info("Processing file {} with shape {}".format(file, df.shape))
        df = df[columns_in_order]

        if list(df.columns) != list(big_df.columns):
            raise ValueError("Bad columns")
        big_df = pd.concat([big_df, df])

    logger.info("Big dataframe has shape shape {}".format(big_df.shape))

        
    logger.warning("IMPLEMENTING ALL CLASS CONFIGURATION")

    if stamp == "TRAIN":
        lq_scale = signif_ref(lq_scale, signifs[-1])
        if lq_mass != 'lq_all':
            for idx, sample in enumerate(big_df.y.to_numpy()):
                if sample == lq_mass:
                     big_df.weight.to_numpy()[idx] = big_df.weight.to_numpy()[idx] * lq_scale # * compensation * 1000
        else:
            for idx, sample in enumerate(big_df.y_lq_all.to_numpy()):
                if sample in C.MASS_TO_NUM_DICT.keys():
                    big_df.weight.to_numpy()[idx] = big_df.weight.to_numpy()[idx] * lq_scale[C.MASS_TO_NUM_DICT[sample]]# * compensation[C.MASS_TO_NUM_DICT[sample]] * 1000
    if stamp == "TEST":
        lq_scale = signif_ref(lq_scale, signifs[-1])
        for idx, sample in enumerate(big_df.y.to_numpy()):
            if sample == lq_mass:
                big_df.weight.to_numpy()[idx] = big_df.weight.to_numpy()[idx] *lq_scale #* 0.12# * compensation * 1000
    # for idx, sample in enumerate(big_df.y.to_numpy()):
        # if sample == lq_mass[i]:
        #     print("signal: {0} vaha: {1}".format(sample, big_df.weight.to_numpy()[idx]))

    big_df["ynew"] = big_df["y"].apply(lambda x: DICT[x])
    big_df["y"] = big_df["ynew"]
    big_df = big_df.drop(columns=["ynew"])
    big_df = big_df.astype(np.float64)
    logger.info(big_df.y.value_counts())
    # for row in big_df:
    #     if row.y == 1.0:
    #         print(row.y)

    # EXTERNAL SOURCE OF THE DATA
    df_tmp = pd.read_csv(EXTERN_PATH + "/" +"ds.csv")
    #if FEATURE_CUT:
    df_tmp = df_tmp[features + ["y", "weight"]]
    for idx, sample in enumerate(df_tmp.y.to_numpy()):
        df_tmp.y.to_numpy()[idx] = df_tmp.y.to_numpy()[idx] + 1
    big_df = pd.concat([big_df, df_tmp])

    # print("This is the output folder: ")
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    big_df.to_csv(OUTPUT + "big_dataframe.csv",index = False)
    y = big_df.y.to_numpy()
    w = big_df.weight.to_numpy()
    f = list(big_df.columns)
    f.remove("y")
    f.remove("weight")
    big_df = big_df.drop(columns = ["weight","y"])
    if lq_mass == 'lq_all':
        y_lq_all = big_df.y_lq_all.to_numpy()

        f.remove("y_lq_all")
        big_df = big_df.drop(columns = ["y_lq_all"])
        save(y_lq_all, OUTPUT, 'y_lq_all')

    X = big_df.to_numpy()
    logger.info('X dimensions: {}'.format(X.shape))
    logger.info('y dimensions: {}'.format(y.shape))
    logger.info('w dimensions: {}'.format(w.shape))
    logger.info('f dimensions: {}'.format(len(f)))
    logger.info("Final Memory size of X is {}".format(X.size * X.itemsize))
    logger.info("Saving dataset..")
    save(X, OUTPUT, 'X')
    save(y, OUTPUT, 'y')
    save(f, OUTPUT, 'f')
    save(w, OUTPUT, 'w')
    return lq_scale