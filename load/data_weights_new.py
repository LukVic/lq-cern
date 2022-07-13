import logging
import os
import sys
import numpy as np
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
from config.conf import setup_logging
from root_load.root_pandas_converter import RPConverter
import config.constants as C
import pandas as pd

def weights_stage_2(file_names, FOLDER_WITH_CSV_FILES, OUTPUT_FOLDER):
    setup_logging("FilteringLogger", file_handler=False)
    logger = logging.getLogger("FilteringLogger")
    for file in file_names:
        #WEIGHTS
        df = pd.read_csv(FOLDER_WITH_CSV_FILES+"/"+file)
        logger.info("Processing file {} with shape {}".format(file, df.shape))
        
        df["temp_cond1"] = (((df.RunYear == 2015) | (df.RunYear == 2016)).astype(int))
        df["temp_cond2"] = (df.RunYear == 2017).astype(int)
        df["temp_cond3"] = (df.RunYear == 2018).astype(int)
        # df["weight"] = ((36207.66 * df.temp_cond1 + 44307.4 * (df.temp_cond2) + 58450.1 * df.temp_cond3) *
        #                 df.custTrigSF_LooseID_FCLooseIso_DLT * df.weight_pileup * df.jvtSF_customOR * df.bTagSF_weight_DL1r_70 *
        #                 df.weight_mc * df["xs"] / df.totalEventsWeighted / 138965.2) * (df.lep_SF_CombinedTight_0 * df.lep_SF_CombinedTight_1) * 138965.2
        df["weight"] = ((36207.66 * df.temp_cond1 + 44307.4 * (df.temp_cond2) + 58450.1 * df.temp_cond3) *
                     df.custTrigSF_LooseID_FCLooseIso_DLT * df.weight_pileup * df.jvtSF_customOR * df.bTagSF_weight_DL1r_70 *
                     df.weight_mc * 1 / df.totalEventsWeighted / 138965.2) * (df.lep_SF_CombinedTight_0 * df.lep_SF_CombinedTight_1) * 138965.2
        # # pre-selection
        #df["is_selected"] = (df["l2SS1tau"]) & (df.nJets_OR > 3) & (df.nJets_OR_DL1r_70 > 0 )
        

        #arthur
        df["is_selected"] = \
            (((abs(df.lep_ID_0)==13) & (df.lep_isMedium_0>0) & (df.lep_isolationFCLoose_0>0)&
            (df.passPLIVVeryTight_0>0)) | ((abs(df.lep_ID_0)==11) & (df.lep_isolationFCLoose_0>0)&
            (df.lep_isTightLH_0>0) & (df.lep_chargeIDBDTResult_recalc_rel207_tight_0>0.7) &
            (df.passPLIVVeryTight_0>0)))&(((abs(df.lep_ID_1)==13) & (df.lep_isMedium_1>0) &
            (df.lep_isolationFCLoose_1>0) & (df.passPLIVVeryTight_1>0)) | ((abs(df.lep_ID_1)==11) &
            (df.lep_isolationFCLoose_1>0) & (df.lep_isTightLH_1>0) &
            ((df.lep_chargeIDBDTResult_recalc_rel207_tight_1>0.7)) &
            (df.passPLIVVeryTight_1>0)))&((((abs(df.lep_ID_0) == 13)) | ( (abs( df.lep_ID_0 ) == 11)&
            (df.lep_ambiguityType_0 == 0) & (~(((df.lep_Mtrktrk_atPV_CO_0<0.1) &
            (df.lep_Mtrktrk_atPV_CO_0>0))& ~((df.lep_RadiusCO_0>20)  &((df.lep_Mtrktrk_atConvV_CO_0<0.1)&
            (df.lep_Mtrktrk_atConvV_CO_0>0))))&~((df.lep_RadiusCO_0>20)&((df.lep_Mtrktrk_atConvV_CO_0<0.1)&
            (df.lep_Mtrktrk_atConvV_CO_0>0)))))) & (((abs( df.lep_ID_1 ) == 11) & (df.lep_ambiguityType_1 == 0)&~
            (((df.lep_Mtrktrk_atPV_CO_1<0.1)&(df.lep_Mtrktrk_atPV_CO_1>0)) & ~((df.lep_RadiusCO_1>20)&
            ((df.lep_Mtrktrk_atConvV_CO_1<0.1)&(df.lep_Mtrktrk_atConvV_CO_1>0))))&~((df.lep_RadiusCO_1>20)&
            ((df.lep_Mtrktrk_atConvV_CO_1<0.1)&(df.lep_Mtrktrk_atConvV_CO_1>0))))|((abs(df.lep_ID_1) == 13))))&\
            (df.nTaus_OR==1)&((df.nJets_OR_TauOR>2) &
            (df.nJets_OR_DL1r_70>0))&((df.dilep_type>0) & ((df.lep_ID_0*df.lep_ID_1)>0))
         
        df = df[df.is_selected]
        logger.info("Outputting file {} with shape {}".format(file, df.shape))
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        df.to_csv(OUTPUT_FOLDER+"/" + file , index = False)
