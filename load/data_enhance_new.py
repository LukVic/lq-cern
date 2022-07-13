import logging
import os
import sys
import numpy as np
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
from config.conf import setup_logging
from root_load.root_pandas_converter import RPConverter
import config.constants as C
import pandas as pd
CUT_VARIABLES_FILE = "ttbar_long.csv"
FEATURES_FILE = "50features.csv"
# FOLDER_WITH_CSV_FILES_BACKGROUND = "/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data/background/"
# FOLDER_WITH_CSV_FILES_SIGNAL = "/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data/signal/"
# OUTPUT_FOLDER_BACKGROUND = "/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_weights/background/"
# OUTPUT_FOLDER_SIGNAL = "/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_weights/signal/"

FOLDER_WITH_CSV_FILES = "/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data/"
OUTPUT_FOLDER = "/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_weights/"
phi_shift = 0.1

def lq_enhance(file_names, FOLDER_WITH_CSV_FILES, OUTPUT_FOLDER):
     for file in file_names:
        print(str(file))
        df = pd.read_csv(FOLDER_WITH_CSV_FILES + file)
        df_l2SS1tau = df[df.l2SS1tau]
        print(df_l2SS1tau["taus_phi_0"])
        print(df_l2SS1tau.shape)



def main():
    features = list(pd.read_csv(C.FEATURES_DIRECTORY + FEATURES_FILE)["features_names"])
    # file_names_back = os.listdir(FOLDER_WITH_CSV_FILES_BACKGROUND)
    # file_names_back = filter(lambda x: os.path.splitext(x)[1] == ".csv", file_names_back)
    # file_names_sig = os.listdir(FOLDER_WITH_CSV_FILES_SIGNAL)
    # file_names_sig = filter(lambda x: os.path.splitext(x)[1] == ".csv", file_names_sig)
    file_names = os.listdir(FOLDER_WITH_CSV_FILES)
    file_names = filter(lambda x: os.path.splitext(x)[1] == ".csv", file_names)
    lq_files = []
    for file in file_names:
        if "lq" in file:
            lq_files.append(file)
    lq_enhance(lq_files, FOLDER_WITH_CSV_FILES, OUTPUT_FOLDER)
    print("End")

if __name__ == '__main__':
    main()