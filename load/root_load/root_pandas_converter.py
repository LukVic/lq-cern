import logging
from typing import List
import pandas as pd
import copy
import uproot3 as uproot
from config.conf import setup_logging
import config.constants as C

class RPConverter():

    def __init__(self):
        setup_logging("RPConverterLogger", file_handler=False)
        self.logger = logging.getLogger()

    def convert_root_to_df(self, root_file_path: str, cut_variables: List[str], features: List[str], class_name: str, lq_all: str) \
            -> pd.DataFrame:
        file = uproot.open(root_file_path)
        self.logger.info("Converting file {}".format(root_file_path))

        df_cols: List[str] = list(set(cut_variables + features))
        df = pd.DataFrame(columns = df_cols)
        for i, col in enumerate(df_cols):
            self.logger.info("Converting column {} : {}".format(i, col))
            if C.VECTORIZED in col:
                col_to_take, index_to_take = col.split(C.VECTORIZED)
                self.logger.info("col_to_take, index_to_take: {}, {}".format(col_to_take, index_to_take))
                self.logger.info(file["nominal"].array(col_to_take))
                self.logger.info("Len is {} ".format(len(file["nominal"].array(col_to_take))))
                self.logger.info("Len of index 0 is {} ".format(len(file["nominal"].array(col_to_take)[0])))
                test = [len(x) for x in file["nominal"].array(col_to_take)]
                self.logger.info("min is {} ".format(min(test)))
                data = [x[int(index_to_take)] if len(x) > int(index_to_take) else 0 for x in file["nominal"].array(col_to_take)]
            else:
                data = file["nominal"].array(col)
            df[str(col)] = data
        df["y"] = class_name
        df["y_lq_all"] = lq_all
        return df
