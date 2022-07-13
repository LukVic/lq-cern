import os
import sys
sys.path.insert(0,"/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
from root_load.root_pandas_converter import RPConverter
import config.constants as C
import pandas as pd
import uproot3 as uproot
import numpy as np

def convert_stage_1(FEATURES_FILE, CUT_VARIABLES_FILE, INPUT, OUTPUT, lq_mass):
    UNIFIED = 1
    # takes features and creates list out of feature csv
    features = list(pd.read_csv(C.FEATURES_DIRECTORY + FEATURES_FILE)["features_names"])
    # takes cut_vars and creates list aout of cut_vars csv
    cut_vars = list(pd.read_csv(C.CUTVARS_DIRECTORY + CUT_VARIABLES_FILE)["features_names"])
    rp_cv = RPConverter()
    # for every root dir
    idx = 0
    for idx, dir in enumerate(INPUT):
        file_names = os.listdir(dir)
        file_names = filter(lambda x: os.path.splitext(x)[1] == ".root", file_names)
        if "tth" in dir and not ("ttz" in dir or "ttw" in dir or "ttbar" in dir or "lq" in dir):
            CLASS = "tth"
        elif "ttz" in dir and not ("tth" in dir or "ttw" in dir or "ttbar" in dir or "lq" in dir):
            CLASS = "ttz"
        elif "ttw" in dir and not ("ttz" in dir or "tth" in dir or "ttbar" in dir or "lq" in dir):
            CLASS = "ttw"
        elif "ttbar" in dir and not ("ttz" in dir or "ttw" in dir or "tth" in dir or "lq" in dir):
            CLASS = "ttbar"
        # elif lq_mass[idx] in dir and not ("ttz" in dir or "ttw" in dir or "tth" in dir or "ttbar" in dir):
        #     CLASS = lq_mass[idx]
        elif lq_mass[idx] in dir and not ("ttz" in dir or "ttw" in dir or "tth" in dir or "ttbar" in dir):
            CLASS = "lq_all"
        else:
            raise ValueError("Unknown class")
        #FOR LQ_ALL INDEX IS 0
        #FOR LQ_X INDEX IS i
        if not os.path.exists(OUTPUT[0]):
            os.makedirs(OUTPUT[0])
        for file in file_names:
            #cut out lq mass
            sep_1 = file.split('.',1)[0]
            sep_2 = sep_1.split('_',3)[3]
            df = rp_cv.convert_root_to_df(dir + "/" + file, cut_vars, features, class_name=CLASS,lq_all=sep_2 )
            # if CLASS == "lq":
            #     output_file_name = OUTPUT_FOLDER_SIGNAL + CLASS + "_" + file + ".csv"
            # else:
            #     output_file_name = OUTPUT_FOLDER_BACKGROUND + CLASS + "_" + file + ".csv"
            output_file_name = OUTPUT[0]+ "/" + CLASS + "_" + file + ".csv"
            df.to_csv(output_file_name, index=False)
            print("Created csv: ", output_file_name)