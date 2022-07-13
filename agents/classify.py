import logging
import os
import sys
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/utils/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/agents/")
import numpy as np
import pickle
import pandas as pd
from ml_utils import save
import ml_utils as ml_utils
import confmatrix_prettyprint as cm
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import config.constants_new as C

#for knc, number of cores available
N_JOBS = 12

CLASSIFIER = 'TABNET'

def histo(INPUT_FOLDER_HISTO, lq_mass):
    f = pickle.load(open(INPUT_FOLDER_HISTO + "/f.pkl", "rb"))
    X = ml_utils.load(INPUT_FOLDER_HISTO + '/X.pkl')
    y = ml_utils.load(INPUT_FOLDER_HISTO + '/y.pkl')

    X = np.array(X)
    y = np.array(y)
    f = np.array(f)


    for idx, feature in enumerate(f):
        ml_utils.plot_histograms_multiclass(X=X[:,idx],
        y=y, classes=C.all_classes_dict_train(lq_mass),
        feature=feature,saving=True, save_folder=INPUT_FOLDER_HISTO + '/histograms/histo_')
    print("Histograms for "+str(lq_mass)+" generated.")


def train(INPUT_FOLDER_TRAIN, MODEL_TRAIN, lq_mass, lq_mass_model):
    X_train = pickle.load(open(INPUT_FOLDER_TRAIN + '/' + lq_mass_model + "/X_train.pkl", "rb"))
    y_train = pickle.load(open(INPUT_FOLDER_TRAIN + '/' + lq_mass_model + "/y_train.pkl", "rb"))

    w_train = pickle.load(open(INPUT_FOLDER_TRAIN + '/' + lq_mass_model + "/w_train.pkl", "rb"))


    # ADA train
    if CLASSIFIER == "ADA":
        parameters_ada = {'base_estimator': DecisionTreeClassifier(max_depth=4),'n_estimators': 50}

        model = ml_utils.train_model(X_train, y_train, w_train, 'ADA', parameters_ada, 80)
        ml_utils.save(model, MODEL_TRAIN[i] + '/models', 'ADA_weights')

    #RFC train
    if CLASSIFIER == "RFC":
        parameters_rfc = {'n_estimators': 220,                    #150
                        'max_depth': 18,                       #8
                        'min_samples_split': 90,               #117,
                        'max_features': 0.60,
                        'max_leaf_nodes':700,                #500
                        'min_impurity_decrease': 0.0,
                        'ccp_alpha': 0.0,
                        'max_samples': 0.58786,
                        'oob_score': False,
                        'criterion': 'entropy',
                        'n_jobs': N_JOBS}
                        # Best trial: score 1.0854704222680494, params {'classifier': 'rfc', 'criterion': 'entropy', 'max_depth': 10, 'n_estimators': 556, 'min_samples_split': 181, 'max_features': 0.7634356924942423, 'max_leaf_nodes': 792, 'min_impurity_decrease': 0.0007574884014088842,
                        #  'ccp_alpha': 0.0006194858263901468, 'max_samples': 0.7027733135469232, 'oob_score': True}
       # {'classifier': 'rfc', 'criterion': 'entropy', 'max_depth': 24, 'n_estimators': 80,
       #  'min_samples_split': 150, 'max_features': 0.9, 'max_leaf_nodes': 400, 'min_impurity_decrease': 0.0, 'ccp_alpha': 0.0, 'max_samples': 0.7, 'oob_score': True}
        model = ml_utils.train_model(X_train, y_train, w_train, 'RFC', parameters_rfc)
        ml_utils.save(model, MODEL_TRAIN+ '/' + str(lq_mass_model) + '/models', 'RFC_weights')

    #XGB
    if CLASSIFIER == "XGB":
        model = ml_utils.train_model(X_train, y_train, w_train, 'XGB')
        ml_utils.save(model, MODEL_TRAIN+'/' + str(lq_mass_model) + '/models', 'XGB_weights')
    #3{'colsample_bytree': 0.8, 'gamma': 0.0008049250829527084, 'learning_rate': 0.09960916370354586, 'max_depth': 20,
    # 'min_child_weight': 89, 'n_estimators': 2384, 'reg_alpha': 8.778376722666817, 'reg_lambda': 0.0010671313085034643,
    # 'subsample': 0.6}
    #LGB
    if CLASSIFIER == "LGB":
        model = ml_utils.train_model(X_train, y_train, w_train, 'LGB')
        ml_utils.save(model, MODEL_TRAIN+'/' + str(lq_mass_model) + '/models', 'LGB_weights')

     #CAT
    if CLASSIFIER == "CAT":
        model = ml_utils.train_model(X_train, y_train, w_train, 'CAT')
        ml_utils.save(model, MODEL_TRAIN+'/' + str(lq_mass_model) + '/models', 'CAT_weights')

    print("Classification for " + lq_mass + " finished.")


def test(INPUT_FOLDER_TRAIN, INPUT_FOLDER_TEST, MODEL, OUTPUT_FOLDER, lq_mass_train, lq_mass_test):

    print("Training for: "+str(lq_mass_train)+" mass tested on "+str(lq_mass_test))
    X_train = ml_utils.load(INPUT_FOLDER_TRAIN + '/' + lq_mass_test +'/X_train.pkl')
    y_train = ml_utils.load(INPUT_FOLDER_TRAIN + '/' + lq_mass_test + '/y_train.pkl')
    w_train = pickle.load(open(INPUT_FOLDER_TRAIN + '/' + lq_mass_test + "/w_train.pkl", "rb"))

    X_test = ml_utils.load(INPUT_FOLDER_TRAIN+ '/' + lq_mass_test + '/X_test.pkl')
    y_test = ml_utils.load(INPUT_FOLDER_TRAIN+ '/' + lq_mass_test + '/y_test.pkl')
    w_test = pickle.load(open(INPUT_FOLDER_TRAIN+ '/' + lq_mass_test + "/w_test.pkl", "rb"))

    classes = {0: 'lq',
                1: 'ttH',
                2: 'ttW',
                3: 'ttZ',
                4: 'ttBar',
                5: 'vv',
                6: 'other',
                }

    print(CLASSIFIER+" classifier was used.")
    model = ml_utils.load(MODEL + '/' +str(lq_mass_test) +'/models/'+CLASSIFIER+'_weights.pkl')
    #config\constants.py

    if CLASSIFIER != 'MLP':
        y_train_pred = ml_utils.predict(model, X_train)
        y_train_pred_proba = ml_utils.predict_proba(model, X_train)
        y_test_pred = ml_utils.predict(model, X_test)
        y_test_pred_proba = ml_utils.predict_proba(model, X_test)
    else:
        y_test_pred = ml_utils.load(MODEL + '/' +str(lq_mass_test) +'/y_test_pred_'+CLASSIFIER+'.pkl')
        y_test_pred_proba = ml_utils.load(MODEL + '/' + str(lq_mass_test) + '/y_test_pred_proba_' + CLASSIFIER + '.pkl')

    save(y_test_pred, OUTPUT_FOLDER+"/"+str(lq_mass_test), 'y_test_pred_'+CLASSIFIER)
    save(y_test_pred_proba, OUTPUT_FOLDER+"/"+str(lq_mass_test), 'y_test_pred_proba_'+CLASSIFIER)


    # save(y_train_pred, OUTPUT_FOLDER+"/"+str(lq_mass_test), 'y_train_pred_'+CLASSIFIER)
    # save(y_train_pred_proba, OUTPUT_FOLDER+"/"+str(lq_mass_test), 'y_train_pred_proba_'+CLASSIFIER)

    # acc_train = ml_utils.multiclass_accuracy(y_train, y_train_pred)
    # f1_train = ml_utils.multiclass_f1(y_train, y_train_pred, average='weighted')
    # auc_train = ml_utils.multiclass_roc_auc_score(y_train, y_train_pred_proba, average='weighted', multi_class='ovr')
    acc_test = ml_utils.multiclass_accuracy(y_test, y_test_pred)
    f1_test = ml_utils.multiclass_f1(y_test, y_test_pred, average='weighted')
    auc_test = ml_utils.multiclass_roc_auc_score(y_test, y_test_pred_proba, average='weighted', multi_class='ovr')

    # ml_utils.plot_roc_multiclass(CLASSIFIER+'_%s train (80%% of %s)' % ('', 'LQ '+str(lq_mass_train)+'_'+str(lq_mass_test)+' GeV'), y_train, y_train_pred_proba, classes,
    #                                 {'Accuracy': acc_train, 'F1 Weighted': f1_train,
    #                                 'ROC AUC Weighted': auc_train}, OUTPUT_FOLDER + "/"+ str(lq_mass_test) +"/imgs/"+CLASSIFIER+"_train_")
    ml_utils.plot_roc_multiclass(CLASSIFIER+'_%s test (20%% of %s)' % ('', 'LQ '+str(lq_mass_train)+'_'+str(lq_mass_test)+' GeV'), y_test, y_test_pred_proba, classes,
                                    {'Accuracy': acc_test, 'F1 Weighted': f1_test, 'ROC AUC Weighted': auc_test}, OUTPUT_FOLDER +"/"+str(lq_mass_test)+ '/imgs/'+CLASSIFIER+'_test_')


    #cm.plot_confusion_matrix_from_data(y_train, y_train_pred, w_train, OUTPUT_FOLDER+"/"+str(lq_mass_test) + '/imgs/'+CLASSIFIER+'_train_', pred_val_axis='col')
    cm.plot_confusion_matrix_from_data(y_test, y_test_pred, w_test,OUTPUT_FOLDER+"/"+str(lq_mass_test) + '/imgs/'+CLASSIFIER+'_test_', pred_val_axis='col')
    print("Classification process for "+str(lq_mass_train)+" <- "+str(lq_mass_test)+" was completed.")