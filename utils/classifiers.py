# from gc import callbacks
import sys

from agents.classify import N_JOBS
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/load/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/utils/")
from typing import List, Dict, Tuple
from easydict import EasyDict
from signif_tools import *
import config.constants as C
import root_load.root_reader as reader
import seaborn as sns
import pickle
import compress_pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import ml_utils as ml_utils
import torch
import optuna

from optuna import Trial

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc

lq_mass_train = ['lq_all']
lq_mass_test = ['lq_500']


INPUT_TRAIN_STAGE_5_1 = ["/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/"
+ i for i in lq_mass_train]
MODEL_TRAIN_STAGE_5_1 = ["/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/"
+ i for i in lq_mass_train]
OUTPUT_STAGE_5_1 = ["/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/"
+ i for i in lq_mass_test]

INPUT_TRAIN_STAGE_5_2 = ["/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/"
+ i for i in lq_mass_train]
INPUT_TEST_STAGE_5_2 = ["/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/"
+ i for i in lq_mass_test]
MODEL_STAGE_5_2 = ["/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/"
+ i for i in lq_mass_train]
OUTPUT_STAGE_5_2 = ["/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/"
+ i for i in lq_mass_train]



def objective(trial, i, j, SPLIT_RATIO):
            test_set_scale_factor = 1 / (1 - 0.8)
            other_bgr = 0
            ML = 1
            score = 0

            X_train = ml_utils.load(INPUT_TRAIN_STAGE_5_2[i] + '/' + lq_mass_test[j] +'/X_train.pkl')
            y_train = ml_utils.load(INPUT_TRAIN_STAGE_5_2[i] + '/' + lq_mass_test[j] + '/y_train.pkl')
            w_train = pickle.load(open(INPUT_TRAIN_STAGE_5_2[i] + '/' + lq_mass_test[j] + "/w_train.pkl", "rb"))

            X_test = ml_utils.load(INPUT_TRAIN_STAGE_5_2[i]+ '/' + lq_mass_test[j] + '/X_test.pkl')
            y_test = ml_utils.load(INPUT_TRAIN_STAGE_5_2[i]+ '/' + lq_mass_test[j] + '/y_test.pkl')
            w_test = pickle.load(open(INPUT_TRAIN_STAGE_5_2[i]+ '/' + lq_mass_test[j] + "/w_test.pkl", "rb"))
            f = ml_utils.load(INPUT_TRAIN_STAGE_5_2[i] + '/' + str(lq_mass_test[j]) + '/f_new.pkl')
            # classifier_name = trial.suggest_categorical("classifier", ['rfc','ada'])
            #classifier_name = trial.suggest_categorical("classifier", ['lgb'])
            classifier_name = 'lgb'
            #classifier_name = 'cat'
            sc = StandardScaler()
        
            if classifier_name == 'rfc':
                N_JOBS = 12

                criterions = trial.suggest_categorical('criterion', ['entropy'])
                max_depths = trial.suggest_int('max_depth', 8, 24, step=2)
                n_estimators = trial.suggest_int('n_estimators', 20, 500, step=10)
                min_samples_split = trial.suggest_int('min_samples_split', 30, 500, step=10)
                max_features = trial.suggest_float('max_features', 0.4, 1.0)
                max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 400, 1200, step = 100)
                min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.001)
                ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.001)
                max_samples = trial.suggest_float("max_samples", 0.4, 1.0)
                oob_score = trial.suggest_categorical('oob_score', [True, False])

            

                parameters_rfc = {'n_estimators': n_estimators,                    #150
                                'max_depth': max_depths,                       #8
                                'min_samples_split': min_samples_split,               #117,
                                'max_features': max_features,
                                'max_leaf_nodes': max_leaf_nodes,                #500
                                'min_impurity_decrease': min_impurity_decrease,
                                'ccp_alpha': ccp_alpha,
                                'max_samples': max_samples,
                                'oob_score': oob_score,
                                'criterion': criterions,
                                'n_jobs': N_JOBS}

                clf = RandomForestClassifier(**parameters_rfc)
                
           
                
            # elif classifier_name == 'ada':
                
            #     criterions = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            #     max_depths = trial.suggest_int('max_depth', 4, 16, step=4)
            #     n_estimators = trial.suggest_int('n_estimators', 20, 120, 20)

            #     parameters_ada = {
            #         'base_estimator': DecisionTreeClassifier(max_depth=max_depths),
            #         'n_estimators': n_estimators
            #         }

            #     clf = AdaBoostClassifier(**parameters_ada)


            elif classifier_name == 'xgb':
                
                # train = XGBClassifier.DMatrix(X_train, label = y_train)
                # test = XGBClassifier.DMatrix(X_test, label = y_test)

                # max_depths = trial.suggest_int('max_depth', 8, 32, step=6)
                # gammas = trial.suggest_int('gamma', 0.1, 0.6, step=0.2)
                # learning_rates = trial.suggest_categorical('learning_rate', [0.01, 0.001])
                # n_estimators = trial.suggest_int('n_estimator', 20, 400, step=20)
                # min_child_weights = trial.suggest_categorical('min_child_weight', [0,0.01,0.1,1])
                # subsample = trial.suggest_float('subsample', 0.1, 0.9, step=0.2)
                # colsample_bytrees = trial.suggest_float('colsample_bytree', 0.5, 0.9, step=0.2)

                # parameters_xgb = {
                #     'learning_rate': learning_rates,
                #     'n_estimator': n_estimators,
                #     'max_depth': max_depths,
                #     'min_child_weights': min_child_weights,
                #     'gamma': gammas,
                #     'subsample': subsample,
                #     'colsample_bytree': colsample_bytrees,
                #     # 'njobs': njobs,
                #     # 'scale_pos_weight': scale_pos_weights,
                #     # 'seed': seeds
                #     }

                d_train = xgb.DMatrix(X_train, label=y_train)
                d_test = xgb.DMatrix(X_test, label=y_test)
                watchlist = [d_train, d_test]
                # parameters_xgb = {
                #             'max_depth': trial.suggest_int('max_depth', 3, 12),
                #             'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.02, 0.05, 0.08, 0.1]),
                #             'n_estimators': trial.suggest_int('n_estimators', 50, 4000),
                #             #'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
                #             'gamma': trial.suggest_float('gamma', 0.0001, 1.0, log = True),
                #             'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 10.0, log = True),
                #             'reg_lambda': trial.suggest_float('lambda', 0.0001, 10.0, log = True),
                #             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.8),
                #             'subsample': trial.suggest_float('subsample', 0.1, 0.8),
                #             'tree_method': 'gpu_hist',
                #             'booster': 'gbtree',
                #             'random_state': 42,
                #             # 'use_label_encoder': False,
                #             'num_class': 7
                #             }


                parameters_xgb = {
                    # "objective": "multi:softprob",
                    # "eval_metric": "auc",
                    'min_child_weight': trial.suggest_float('min_child_weight', 0.005, 0.05, step=0.001),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0005, 0.005, step = 0.0001),
                   # 'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 10.0, log = True),
                    'gamma': trial.suggest_float('gamma', 0.00005, 0.0005, step=0.00001),
                    'num_class': 7,
                    'tree_method': 'gpu_hist',
                    'learning_rate': trial.suggest_float('learning_rate',0.3, 0.7, step = 0.1),
                    'n_estimators': trial.suggest_int('n_estimators', 2000, 4000, step = 100),
                    'colsample_bytree': trial.suggest_categorical('colsample_bytree',[ 0.8, 0.9, 1.0]),
                    'subsample': trial.suggest_categorical('subsample', [0.8,1.0]),
                    'max_depth': trial.suggest_categorical('max_depth', [6,7,8,9,10,11,12]),
                    'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [6, 7, 8, 9, 10, 11, 12]),
                    'random_state': 42,
                    # 'booster': 'gbtree'
                }

                #clf = XGBClassifier(**parameters_xgb)
                #pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
                clf = XGBClassifier(**parameters_xgb)
                #model = xgb.train(parameters_xgb, d_train, evals=[(d_test, "validation")], callbacks=[pruning_callback])

            elif classifier_name == 'lgb':
                
                # colsample_bytrees = trial.suggest_float('colsample_bytree', 0.5, 0.9, step=0.2)
                # learning_rates = trial.suggest_categorical('learning_rate', [0.01,0.1,1])
                # max_depths = trial.suggest_int('max_depth', 4, 16, step=4)
                # #min_child_weights = trial.suggest_categorical('min_child_weight', [0,1])
                # n_estimators = trial.suggest_int('n_estimator', 20, 120, step=20)
                # njobs = trial.suggest_categorical('njobs', [12])
                # subsample = trial.suggest_float('subsample', 0.5, 0.9, step=0.2)
                # #scale_pos_weights= trial.suggest_categorical('scale_pos_weight',[7])
                # seeds = trial.suggest_categorical('seed', [23])

                # parameters_lgb = {
                #     'learning_rate': learning_rates,
                #     'n_estimators': n_estimators,
                #     'max_depth': max_depths,
                #     'subsample': subsample,
                #     'colsample_bytree': colsample_bytrees,
                #     'njobs': njobs,
                #     'seed': seeds
                #     }
             
                early_stop = 20

                d_train = lgb.Dataset(X_train, label=y_train)
                d_test = lgb.Dataset(X_test, label=y_test)
                watchlist = [d_train, d_test]

                parameters_lgb = {
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1, step=0.01),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, step=0.1),
                    #'num_leaves': trial.suggest_int('num_leaves', 10, 800),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'max_depth': trial.suggest_int('max_depth', 1, 10),
                    #'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05,0.1, 0.5, 0.99]),
                   'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5, step=0.1),
                    'n_estimators': trial.suggest_int('n_estimators', 500, 4000, step=100),
                    'cat_smooth' : trial.suggest_int('cat_smooth', 10, 100, step=1),
                    'cat_l2': trial.suggest_int('cat_l2', 1, 20, step=1),
                    'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200, step=1),
                   'cat_feature' : trial.suggest_int('cat_feature', 11, 67, step=1),
                    'random_state': 42,
                    'device': 'cuda',
                    'num_class': 7,
                }
            #     pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc_mu", valid_name="valid_1")
                clf = LGBMClassifier(**parameters_lgb)

            #     model = lgb.train(parameters_lgb,
            #           train_set=d_train,
            #           num_boost_round=1000,
            #           valid_sets=watchlist,
            #           early_stopping_rounds=early_stop,
            #           callbacks=[pruning_callback])


            elif classifier_name == 'cat':
                
                #colsample_bytrees = trial.suggest_float('colsample_bytrees', 0.5, 0.9, step=0.2)
              #  learning_rates = trial.suggest_categorical('learning_rate', [0.01,0.1,1])
             #   max_depths = trial.suggest_int('max_depth', 4, 10, step=4)
               # min_child_weights = trial.suggest_categorical('min_child_weight', [0,1])
             #   n_estimators = trial.suggest_int('n_estimator', 20, 120, step=20)
                #njobs = trial.suggest_categorical('njobs', [12])
                #subsample = trial.suggest_float('subsample', 0.5, 0.9, step=0.2)
                #scale_pos_weights= trial.suggest_categorical('scale_pos_weight',[7])
                #seeds = trial.suggest_categorical('seed', [23])


                # parameters_cat = {
                #     'learning_rate': learning_rates,
                #     'n_estimators': n_estimators,
                #     'max_depth': max_depths,
                #     }

                parameters_cat = {
                    'max_depth': trial.suggest_int('max_depth', 3, 12, step=1),
                   'learning_rate': trial.suggest_float('learning_rate', 0.1, 1, step=0.1),
                    'iterations': trial.suggest_int('n_estimators', 500, 2000, step = 50),
             #      'max_bin': trial.suggest_int('max_bin', 100, 1000, step=100),
             #      'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
                   'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.05, 0.5, step = 0.01),
                   'subsample': trial.suggest_float('subsample', 0.1, 0.8),
                #    'scale_pos_weight': trial.suggest_float('scale_pos_weight',0.01, 1.0, step=0.01),
                    'border_count': trial.suggest_int('border_count',1, 255,step=1),
                    'random_strength': trial.suggest_float('random_strength',1e-9, 10, log=True),
                    'bagging_temperature': trial.suggest_float('bagging_temperature',0.0, 1.0, step=0.1),
                   'random_seed': 42,
                    'task_type': 'GPU',
                    'bootstrap_type': 'Poisson',
                    }  


                clf = CatBoostClassifier(**parameters_cat)

            elif classifier_name == 'tabnet':
                MAX_EPOCHS = 40
                BATCH_SIZE = 2048
                VIRTUAL_BATCH_SIZE = 512
               
                DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

                mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
                #n_da = trial.suggest_int("n_da", 56, 64, step=4)
                n_steps = trial.suggest_int("n_steps", 1, 5, step=1)
                gamma = trial.suggest_float("gamma", 1., 1.4, step=0.1)
                n_shared = trial.suggest_int("n_shared", 1, 3, step=1)
                momentum = trial.suggest_float("momentum", 0.3, 0.8, step=0.1)
                lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
                epsilon = 1e-16

                parameters_tab =  dict( n_steps=n_steps, gamma=gamma, epsilon=epsilon,
                    lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=2e-2, weight_decay=3e-6),
                    mask_type=mask_type, n_shared=n_shared, momentum=momentum,
                    scheduler_params=dict(mode="min",
                                        patience=trial.suggest_int("patienceScheduler",low=3,high=10), # changing sheduler patience to be lower than early stopping patience 
                                        min_lr=1e-5,
                                        factor=0.5,),
                    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    verbose=0,
                    device_name=DEVICE
                    )
            
                clf = TabNetClassifier(**parameters_tab)

                clf.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    max_epochs=MAX_EPOCHS , patience=50,
                    batch_size=BATCH_SIZE, virtual_batch_size=VIRTUAL_BATCH_SIZE,
                    num_workers=0,
                    drop_last=False
                )
                y_test_pred = ml_utils.predict(clf, X_test)
                y_test_pred_proba = ml_utils.predict_proba(clf, X_test) 

                x_values, y_S, y_B, y_signif, y_signif_simp, y_signif_imp, best_significance_threshold = \
                calculate_significance_all_thresholds_new_method(
                    y_test_pred_proba, y_test, w_test, other_bgr, test_set_scale_factor)

                score = threshold_value(x_values, [y_signif_simp], ['max']) 
               

            if classifier_name == 'rfc' or classifier_name == 'cat' or classifier_name == 'lgb' or classifier_name == 'xgb':
                pipe = Pipeline(steps=[('sc', sc),
                            ('clf', clf)])

                #model = pipe.fit(X_train, y_train,**{'clf__sample_weight': w_train})
                model = pipe.fit(X_train, y_train)

                y_test_pred = ml_utils.predict(model, X_test)
                y_test_pred_proba = ml_utils.predict_proba(model, X_test)

                x_values, y_S, y_B, y_signif, y_signif_simp, y_signif_imp, best_significance_threshold = \
                    calculate_significance_all_thresholds_new_method(
                        y_test_pred_proba, y_test, w_test, other_bgr, test_set_scale_factor)

                score = threshold_value(x_values, [y_signif_simp], ['max'])
            
            # if classifier_name == 'xgb': #or classifier_name == 'lgb':
            #     y_test_pred_proba = model.predict(d_test)
            #     #y_test_pred_proba = model.predict_proba(X_test, num_iteration=model.best_iteration)
            #
            #     #y_pred = model.predict(X_test)
            #     #accuracy = accuracy_score(y_test, y_pred)
            #     x_values, y_S, y_B, y_signif, y_signif_simp, y_signif_imp, best_significance_threshold = \
            #         calculate_significance_all_thresholds_new_method(
            #             y_test_pred_proba, y_test, w_test, other_bgr, test_set_scale_factor)
            #
            #     score = threshold_value(x_values, [y_signif_simp], ['max'])

            return score
