import logging
import os
import sys
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/utils/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/")

import numpy as np
from config.conf import setup_logging
from root_load.root_pandas_converter import RPConverter
import config.constants as C
import pandas as pd
from ml_utils import save

def show(mass):
    classes=pd.DataFrame(columns = ["classnumber","classname"],data=[[0, "lq"],[1,"tth"],[2,"ttw"],[3,"ttz"],[4,"tt"],[5,"vv"],[6,"other"]])
    df = pd.read_csv("/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/"+ mass +"big_dataframe.csv")
    df = df[["y","weight"]]
    df = df.groupby("y", as_index = False)
    df = df.agg({"weight":["mean","count","sum"]})
    df = df.merge(classes, how="left",left_on="y",right_on="classnumber")
    df = df.drop(columns = ["classnumber"])
    df.columns = ["y","w_mean","events","w_sum","classname"]
    print(df)
