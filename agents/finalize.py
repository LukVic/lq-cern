"""
Last modified May 19 2020
@author: Jakub Maly
"""
import logging
import os
import sys
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/utils/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/utils/significance/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/agents/")
from signif_tools import *

import ml_utils as ml_utils
import matplotlib.pyplot as plt
import numpy as np

import confmatrix_prettyprint as cm


def finalize(INPUT_TRAIN, INPUT_TEST, OUTPUT, MODEL, SPLIT_RATIO,lq_mass_train, lq_mass_test, signifs):
    TEST_TRAIN_SPLIT = 1 - SPLIT_RATIO
    other_bgr = 0

    CLASSIFIER = 'TABNET'
    CUT = 18#int(89*(100/100))

    X_test = ml_utils.load(INPUT_TRAIN + '/' + str(lq_mass_test) + '/X_test.pkl')
    y_test = ml_utils.load(INPUT_TRAIN + '/' + str(lq_mass_test) + '/y_test.pkl')
    weights_test = ml_utils.load(INPUT_TRAIN + '/' + str(lq_mass_test) + '/w_test.pkl')
    print('Number of features:' + str(X_test.shape))
    f = ml_utils.load(INPUT_TRAIN + '/' + str(lq_mass_test) + '/f_new.pkl')
    model = ml_utils.load(MODEL +'/'+str(lq_mass_test)+ '/models/'+CLASSIFIER+'_weights.pkl')

    y_pred_test = ml_utils.load(INPUT_TRAIN+ '/' + str(lq_mass_test)+'/y_test_pred_'+CLASSIFIER+'.pkl')
    y_probas_test = ml_utils.load(INPUT_TRAIN+ '/' +str(lq_mass_test)+ '/y_test_pred_proba_'+CLASSIFIER+'.pkl')

    predicted_probs_test = y_probas_test
    true_y_test = y_test

    test_set_scale_factor = 1 / (1 - TEST_TRAIN_SPLIT)
    train_set_scale_factor = 1 / TEST_TRAIN_SPLIT


    #for test
    x_values, y_S, y_B, y_signif, y_signif_simp, y_signif_imp, best_significance_threshold = \
        calculate_significance_all_thresholds_new_method(
            predicted_probs_test, true_y_test, weights_test, other_bgr, test_set_scale_factor)
    maxsignificance = max(y_signif)
    maxsignificance_simple = max(y_signif_simp)



    # Signal to threshold
    plot_threshold(x_values, [y_S, y_B], ['max', 'min'], 'S & B to Threshold characteristics',
                    'Expected events',
                    ['green', 'sienna'], ['S - LQ classifed as LQ', 'B - background classified as LQ'], savepath=OUTPUT+"/" + str(lq_mass_test) + "/imgs/"+CLASSIFIER+"_sb_to_thresh.png",
                    force_threshold_value=best_significance_threshold)
    # Significance
    best_signif_simp = plot_threshold(x_values, [y_signif, y_signif_simp, y_signif_imp], ['max', 'max', 'max'], 
                    'Significance Approximations to Threshold characteristics',
                    'Significance Approximation', ['darkred', 'r', 'purple'],
                    ['S/sqrt(S+B)', 'S/sqrt(B)', 'S/(3/2+sqrt(B))'],
                    savepath=OUTPUT+"/"+ str(lq_mass_test) + "/imgs/"+CLASSIFIER+"_significance_test_"+
                    lq_mass_train+"_"+lq_mass_test+".png")
    signifs.append(best_signif_simp)

    cm.plot_confusion_matrix_from_data(true_y_test
    , calculate_class_predictions_basedon_decision_threshold(predicted_probs_test, best_significance_threshold),
        weights_test, OUTPUT+ "/" + str(lq_mass_test)+ "/imgs/"+CLASSIFIER+"_cm_threshold.png" ,pred_val_axis='col')

    # Feature importance
    if CLASSIFIER != 'MLP':
        plot_feat_imp("Importances: "+CLASSIFIER, model, f, CUT, OUTPUT+"/" +str(lq_mass_test) + "/imgs/",CLASSIFIER)
        print("Finalization for "+str(lq_mass_train)+" <- "+str(lq_mass_test)+": "
        +CLASSIFIER+" was used.")
    
    #_____________________TRAIN SUPPORT________________________
    # x_values, y_S, y_B, y_signif, y_signif_simp,y_signif_imp, best_significance_threshold = \
    # calculate_significance_all_thresholds_new_method(
    # predicted_probs_train, true_y_train, weights_train, other_bgr, train_set_scale_factor)
    
    # plot_threshold(x_values, [y_signif, y_signif_simp, y_signif_imp], ['max', 'max', 'max'], 
    # 'Significance Approximations to Threshold characteristics',
    # 'Significance Approximation [-]', ['darkred', 'r', 'purple'], ['S/sqrt(B)', 'S/sqrt(B)', 'S/(3/2+sqrt(B))'],
    # savepath=OUTPUT+"/"+ str(lq_mass_test) + "/imgs/"+CLASSIFIER+"_significance_train_"+
    # lq_mass_train+"_"+lq_mass_test+".png")

    return signifs