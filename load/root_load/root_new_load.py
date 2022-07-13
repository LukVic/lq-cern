import os
import sys
sys.path.insert(0,"/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
import logging
from typing import List
import pandas as pd
import copy
import uproot3 as uproot
from config.conf import setup_logging
import config.constants as C

FILE = "/home/lucas/Documents/KYR/bc_thesis/final_data/lq/LQ.root"
FILE2 = "/home/lucas/Documents/KYR/bc_thesis/final_data/ttbar/tt.root"

def main():
    #df_cols: List[str] = list(set(cut_variables + features))
    file = uproot.open(FILE)
    file2 = uproot.open(FILE2)
    #print(file.array("sumW"))
    print(file["sumW"].values[0])
    print(file2["sumW"].values[1:10])

if __name__ == '__main__':
    main()