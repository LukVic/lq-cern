from fileinput import filename
import os
import sys
import logging

sys.path.insert(0,"/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
sys.path.insert(0,"/home/lucas/Documents/KYR/bc_thesis/thesis/project/plot/analysis")
# from root_load.root_pandas_converter import RPConverter
import config.constants as C
import numpy as np
import time
import optuna
from optuna import Trial
from datetime import date

from data_convert_new import convert_stage_1
from data_weights_new import weights_stage_2
from data_prepare_new import prepare_stage_3
from data_split_new import split_stage_4
from agents.classify import train, test, histo
from agents.finalize import finalize
from agents.recognize import recognize
from utils.classifiers import objective
from agents.deep_learning.tabnet_agent import tabAgent
from agents.deep_learning.mlp_run import mlp_run
from xs_m_plot import xs_m
from data_show_new import show
from train_test_signif import train_test_signif
from feature_cut_signif import feature_cut_signif
from data_cut import data_cut
from mass_guess import mass_guess
from data_shr import data_shr
from compute_imp_sum import compute_imp_sum
from feature_c import feature_c
from sep_a import sep_a

ML = 0
OWN = 0
dataset = 1


if dataset == 1:
    dataset_folder_1 = "final_data"
    dataset_folder_2 = "data_processed"
elif dataset == 2:
    dataset_folder_1 = "final_data_2"
    dataset_folder_2 = "data_processed_2" 

#---------------------------- DATASET 1----------------------------------------#
# LQ SCALING FACOR
#ada
lq_scale_ada = [0.010, 0.024, 0.075, 0.12, 0.25, 0.75, 1.15, 2.5, 4.9, 7, 14, 30]
#nn
lq_scale_nn = [0.0152, 0.030, 0.062, 0.10, 0.21, 0.55, 0.95, 1.85, 3.2, 6.1, 12.5, 23]
# rfc
lq_scale_rfc = [0.018, 0.024, 0.063, 0.118, 0.20, 0.56, 0.83, 1.9, 3.8, 5.8, 13.2, 26]
lq_scale_rfc_new = [0.027, 0.046, 0.12, 0.24, 0.42, 1.1, 2.0, 3.3, 6.6, 14, 26, 49]
#tabnet
lq_scale_tabnet = [0.010, 0.024, 0.075, 0.12, 0.25, 0.75, 1.15, 2.5, 4.9, 7, 14, 30]
lq_scale = lq_scale_tabnet
#cat
lq_scale_cat = [0.010, 0.024, 0.075, 0.12, 0.25, 0.75, 1.15, 2.5, 4.9, 7, 14, 30]
lq_scale_cat_new = [0.0237, 0.0382, 0.119, 0.322, 0.459, 1.221, 2.26, 3.968, 7.781,
 16.859, 29.966, 60.595]
lq_scale_xgb =[0.013, 0.022, 0.061, 0.088, 0.16, 0.49, 0.85, 1.5, 3.4, 6.8, 13.2, 26]
#[0.0013, 0.0020, 0.0065, 0.0080, 0.014, 0.040, 0.068, 0.14, 0.32, 0.65, 1.32, 2.7]
#---------------------------- DATASET 2---------------------------------------#
#TODO


lq_scale_train =[[1,1,1,1,1,1,1,1,1,1,1,1]]#[[0.016, 0.025, 0.067, 0.095, 0.19, 0.55, 0.9, 1.7, 3.7, 7.1, 13.7, 28]]#lq_scale_tabnet#lq_scale_tabnet#[0.024] * 12
lq_scale_test =[1]#[1,1,1,1]*10#[0.19]#lq_scale_xgb#lq_scale_tabnet#lq_scale_tabnet#[0.024] * 12

lq_mass_train =['lq_all']#['lq_all','lq_all','lq_all','lq_all']*10#['lq_500','lq_600','lq_700','lq_800','lq_900','lq_1000','lq_1100','lq_1200','lq_1300','lq_1400','lq_1500','lq_1600']#['lq_all']*12#['lq_500','lq_600','lq_700','lq_800','lq_900','lq_1000','lq_1100','lq_1200','lq_1300','lq_1400','lq_1500','lq_1600']#['lq_all']*12#['lq_500','lq_600','lq_700','lq_800','lq_900','lq_1000','lq_1100','lq_1200','lq_1300','lq_1400','lq_1500','lq_1600']#['lq_all']*12#['lq_500','lq_500','lq_800','lq_800','lq_800','lq_1100','lq_900','lq_1200','lq_800','lq_1100','lq_800','lq_1100'] #lq_mass_1
lq_mass_test =['lq_800']#['lq_600','lq_900','lq_1200','lq_1500']*10#['lq_500','lq_600','lq_700','lq_800','lq_900','lq_1000','lq_1100','lq_1200','lq_1300','lq_1400','lq_1500','lq_1600']#['lq_500']*12#['lq_500','lq_600','lq_700','lq_800','lq_900','lq_1000','lq_1100','lq_1200','lq_1300','lq_1400','lq_1500','lq_1600'] #lq_mass_1


compensation_train = [[0.0120375,0.01321875,0.0104961538461538,
0.0140666666666667,0.0137666666666667,0.0134777777777778,
0.0127958333333333,0.0122291666666667,0.0118583333333333,
0.0116666666666667,0.0108,0.0108692307692308]]*500

compensation_test = [0.0120375,0.01321875,0.0104961538461538,
0.0140666666666667,0.0137666666666667,0.0134777777777778,
0.0127958333333333,0.0122291666666667,0.0118583333333333,
0.0116666666666667,0.0108,0.0108692307692308]*100

lq_xs = [0.2715, 0.09198, 0.03512, 0.01455, 0.006408,
 0.003028, 0.001474, 0.000749, 0.0003864, 0.0002061, 0.0001115, 0.00006099]


lq_new_xs_ada = np.multiply(lq_scale_ada, lq_xs)
lq_new_xs_rfc = np.multiply(lq_scale_rfc, lq_xs)
lq_new_xs_rfc_new = np.multiply(lq_scale_rfc_new, lq_xs)
lq_new_xs_nn = np.multiply(lq_scale_nn, lq_xs)
lq_new_xs_tabnet = np.multiply(lq_scale_nn, lq_xs)
lq_new_xs_cat_new = np.multiply(lq_scale_cat_new, lq_xs)

# DATA AUGMENTATION
phi_shift_bound_1 = 0.1
phi_shift_bound_2 = 3.0
phi_shifts = [5, 5, 5, 5, 5]





#calculate size of a matrix required for ratios placement
h = len(lq_mass_train)
ratios = [ 0.8 for x in range(h)]
signifs = [ [0] for x in range(h)]
 

CUT_VARIABLES_FILE = "ttbar_long.csv"
FEATURES_FILE = "features_with_taus_vectorized_withoutOR.csv"


#------------------------------------------- FIRST DATASET -----------------------------

INPUT_CONVERT_STAGE_1 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_1+"/"
        + i for i in lq_mass_train] 


OUTPUT_CONVERT_STAGE_1 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data/" 
+ i for i in lq_mass_train]

INPUT_WEIGHTS_STAGE_2 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data/"
+ i for i in lq_mass_train]
OUTPUT_WEIGHTS_STAGE_2 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_weights/"
+ i for i in lq_mass_train]

INPUT_TRAIN_PREPARE_STAGE_3 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_weights/"
+ i for i in lq_mass_train]
INPUT_TEST_PREPARE_STAGE_3 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_weights/"
+ i for i in lq_mass_test]

OUTPUT_TRAIN_PREPARE_STAGE_3 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]
OUTPUT_TEST_PREPARE_STAGE_3 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_test]
EXTERN_3 = "/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_weights_2/"

SPLIT_TRAIN_STAGE_4 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]
SPLIT_TEST_STAGE_4 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_test]


INPUT_HISTO_STAGE_5_0 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]

INPUT_TRAIN_STAGE_5_1 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]
MODEL_TRAIN_STAGE_5_1 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]
OUTPUT_STAGE_5_1 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_test]

INPUT_TRAIN_STAGE_5_2 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]
INPUT_TEST_STAGE_5_2 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_test]
MODEL_STAGE_5_2 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]
OUTPUT_STAGE_5_2 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]

INPUT_TRAIN_STAGE_6 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]
INPUT_TEST_STAGE_6 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_test]
MODEL_STAGE_6 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]
OUTPUT_STAGE_6 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]

INPUT_TRAIN_STAGE_7 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]
INPUT_TEST_STAGE_7 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_test]
MODEL_STAGE_7 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]
OUTPUT_STAGE_7 = ["/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/"
+ i for i in lq_mass_train]

EXPERIMENTS = "/home/lucas/Documents/KYR/bc_thesis/"+dataset_folder_2+"/final_data_analysis_weights/experiments/"
# ----------------------------------------SECOND DATASET---------------------------------------------



def main():
    SPLIT_RATIO = 0.8
    start = time.time()
    # for i in range(len(lq_xs)):
    #     print(lq_xs[i]*lq_scale_train[0][i])
    #
    # convert_stage_1(FEATURES_FILE, CUT_VARIABLES_FILE,
    #  INPUT_CONVERT_STAGE_1, OUTPUT_CONVERT_STAGE_1, lq_mass)
    for out_f, in_f in zip(OUTPUT_WEIGHTS_STAGE_2, INPUT_WEIGHTS_STAGE_2):
        file_names = os.listdir(in_f)
        file_names = filter(lambda x: os.path.splitext(x)[1] == ".csv", file_names)

        weights_stage_2(file_names, in_f, out_f)


    logger = logging.getLogger('signifs_logger')

    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(EXPERIMENTS +'/feacure_cut.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)


    for k in range(h):

        while signifs[k][-1] < 1.95 or signifs[k][-1] > 2.01:
           # PART 3

            lq_scale_train[k] = prepare_stage_3(INPUT_TRAIN_PREPARE_STAGE_3[k], OUTPUT_TRAIN_PREPARE_STAGE_3[k], FEATURES_FILE,
                lq_mass_train[k], lq_mass_test[k], lq_scale_train[k],
                 phi_shift_bound_1, phi_shift_bound_2,signifs[k],compensation_train[k], EXTERN_3, "TRAIN")
            if lq_mass_test != lq_mass_train and lq_mass_train != 'lq_all':
                lq_scale_test[k] = prepare_stage_3(INPUT_TEST_PREPARE_STAGE_3[k], OUTPUT_TEST_PREPARE_STAGE_3[k], FEATURES_FILE,
                    lq_mass_test[k],lq_mass_train[k], lq_scale_test[k],
                    phi_shift_bound_1, phi_shift_bound_2,signifs[k],compensation_test[k], EXTERN_3, "TEST")
    #
            # #PART 4
            ratios[k]= split_stage_4(SPLIT_TRAIN_STAGE_4[k], SPLIT_TEST_STAGE_4[k], lq_mass_train[k], lq_mass_test[k])

            # for mass in lq_mass_train:
            #     show(mass)
            #
            # histo(INPUT_HISTO_STAGE_5_0[k], lq_mass_train[k])

            if ML == 0:
                if OWN == 0:
                    tabAgent(INPUT_TRAIN_STAGE_5_1[k], MODEL_TRAIN_STAGE_5_1[k], lq_mass_test[k])
                else:
                    mlp_run(INPUT_TRAIN_STAGE_5_1[k], MODEL_TRAIN_STAGE_5_1[k],lq_mass_train[k], lq_mass_test[k])

            else:
                #PART 5_1
                train(INPUT_TRAIN_STAGE_5_1[k], MODEL_TRAIN_STAGE_5_1[k], lq_mass_train[k], lq_mass_test[k])

            #PART 5_2
            test(INPUT_TRAIN_STAGE_5_2[k], INPUT_TEST_STAGE_5_2[k], MODEL_STAGE_5_2[k], OUTPUT_STAGE_5_2[k],
            lq_mass_train[k], lq_mass_test[k])

            #PART 6
            finalize(INPUT_TRAIN_STAGE_6[k], INPUT_TEST_STAGE_6[k], OUTPUT_STAGE_6[k], MODEL_STAGE_6[k], ratios[k],lq_mass_train[k], lq_mass_test[k], signifs[k])
            print("These are the new significances "+ str(signifs[k]))
            print("These are the new scaling factors "+ str(lq_scale_test[k]))

           # PART 7
            #recognize(INPUT_TRAIN_STAGE_7[k],INPUT_TEST_STAGE_7[k], OUTPUT_STAGE_7[k], MODEL_STAGE_7[k], lq_mass_test[k])

    #         # logger.info("These are the new significances ")
    #         # logger.debug("These are the new significances "+ str(signifs[k]))
    #         # logger.debug("These are the new scaling factors "+ str(lq_scale_test[k]))
            logger.debug("Significance from: " + str(lq_mass_train[k])+ " | "  + str(lq_mass_test[k])+ " | "  + str(signifs[k][1]) + ",")
    #         # logger.debug("These are the new scaling factors "+ str(lq_scale_test[k]))
            end = time.time()
            min = int(end-start)/60
            sec = end-start - 60*int(min)
            print("Time of the training and testing procedure: " + str(int(min)) + " min, " + str(int(sec)) + " sec.")
            break
    #
    logger.debug("Time of the overall procedure: " + str(int(min)) + " min, " + str(int(sec)) + " sec.")
    logger.debug("-------------------------------------------------------")



         # For Optuna ----------------------------------------------------
         #    print("Following masses under study.")
         #    print("Mass: Train: |"+lq_mass_train[k]+"|")
         #    print("Mass: Test: |"+lq_mass_test[k]+"|")
         #    print(str(date.today()))
         #    study = optuna.create_study(study_name='LGB: '+ '2022-05-03'+"all_chosen_2",
         #        direction='maximize', storage='sqlite:///example.db',load_if_exists=True,
         #        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
         #    study.optimize(lambda trial: objective(trial, k, k,ratios[k]), n_trials=2)
         #    print("Following masses study is finished.")
         #    print("Mass: Train: |"+lq_mass_train[k]+"|")
         #    print("Mass: Test: |"+lq_mass_test[k]+"|")
         #
         #    print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
         #    trials_df = study.trials_dataframe()
         #    #print(trials_df)
         #    fig = optuna.visualization.plot_optimization_history(study)
         #    fig.show()
         #    fig=optuna.visualization.plot_intermediate_values(study)
         #    fig.show()
         #    fig = optuna.visualization.plot_param_importances(study)
         #    fig.show()
         #    fig =optuna.visualization.plot_slice(study)
         #    fig.show()
         #    fig = optuna.visualization.plot_parallel_coordinate(study)
         #    fig.show()
         #    break
    #----------------------------------------------------------------------
        #plot new cross section

       # m = [800, 900, 1000, 1100, 1200]
       # lq_xs_sub = lq_xs[4:9]
       # lq_new_xs_sub = lq_new_xs[4:9]
    # m = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
    # lq_xs_sub = lq_xs
    # lq_new_xs_sub_ada = lq_new_xs_ada
    # lq_new_xs_sub_rfc = lq_new_xs_rfc
    # lq_new_xs_sub_nn = lq_new_xs_nn
    # lq_new_xs_sub_rfc_new = lq_new_xs_rfc_new
    # lq_new_xs_sub_cat_new = lq_new_xs_cat_new
    # print(lq_new_xs_sub_rfc_new)
    # print(lq_new_xs_sub_rfc)
    # xs_m(lq_xs_sub, lq_new_xs_sub_ada, lq_new_xs_sub_rfc,
    # lq_new_xs_sub_nn, lq_new_xs_sub_rfc_new,lq_new_xs_sub_cat_new,m)

    #train_test_signif()
    #feature_cut_signif()
    #data_cut()
    #mass_guess()
    #data_shr()
    #compute_imp_sum()
    sep_a()
    #feature_c()

if __name__ == '__main__':
    main()