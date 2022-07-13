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
FEATURES_FILE = "features_with_taus_vectorized.csv"
FOLDER_WITH_PREPARED_CSV_FILES = "/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_weights/"

def main():
    features = list(pd.read_csv(C.FEATURES_DIRECTORY + FEATURES_FILE)["features_names"])
    file_names = os.listdir(FOLDER_WITH_PREPARED_CSV_FILES)
    file_names = filter(lambda x: os.path.splitext(x)[1] == ".csv", file_names)
    setup_logging("PreparationLogger", file_handler=False)
    logger = logging.getLogger("PreparationLogger")
    columns_in_order = features + ["y", "weight"]
    big_df = pd.DataFrame(columns = columns_in_order)
    for file in file_names:
        # WEIGHTS
        df = pd.read_csv(FOLDER_WITH_PREPARED_CSV_FILES + file)
        logger.info("Processing file {} with shape {}".format(file, df.shape))
        df = df[columns_in_order]
        if list(df.columns) != list(big_df.columns):
            raise ValueError("Bad columns")
        big_df = pd.concat([big_df, df])
        logger.info("Big dataframe has shape shape {}".format(big_df.shape))
    print("jsem v lq")
    y = big_df.y.to_numpy()
    w = big_df.weight.to_numpy()
    for idx, sample in enumerate(big_df.y.to_numpy()):
        if sample == 'lq':
            big_df.weight.to_numpy()[idx] = big_df.weight.to_numpy()[idx]*1000
    print("jsem v lq")
    for idx, sample in enumerate(big_df.y.to_numpy()):
        if sample == 'lq':
            print("signal: {0} vaha: {1}".format(sample, big_df.weight.to_numpy()[idx]))
if __name__ == '__main__':
    main()
