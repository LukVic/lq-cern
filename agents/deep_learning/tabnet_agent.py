import sys
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/utils/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
import os




import torch
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import ml_utils as ml_utils
import wget
from pathlib import Path
import shutil
import gzip
import random

import optuna
from optuna import Trial, visualization

np.random.seed(0)

def tabAgent(INPUT_FOLDER_TRAIN, MODEL, lq_mass):
    CUDA_LAUNCH_BLOCKING=1
    random_state = 0
    train_split = 80
    RECOG = 1

    if RECOG == 0:
        X_train = pickle.load(open(INPUT_FOLDER_TRAIN+ '/' + str(lq_mass) + "/X_train.pkl", "rb"))
        y_train = pickle.load(open(INPUT_FOLDER_TRAIN+ '/' + str(lq_mass) + "/y_train.pkl", "rb"))
    else:
        X_train = pickle.load(open(INPUT_FOLDER_TRAIN + '/' + str(lq_mass) + "/X_tr_new.pkl", "rb"))
        y_train = pickle.load(open(INPUT_FOLDER_TRAIN + '/' + str(lq_mass) + "/y_tr_new.pkl", "rb"))

    MAX_EPOCHS = 40
    BATCH_SIZE = 2048
    VIRTUAL_BATCH_SIZE = 512
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using {}".format(DEVICE))
    # {'gamma': 1.4, 'lambda_sparse': 0.00014440245699539302, 'mask_type': 'entmax', 'momentum': 0.5, 'n_shared': 3,
    #  'n_steps': 2, 'patienceScheduler': 10}
    clf = TabNetClassifier(
                      #  n_d=64, n_a=64, n_steps=5,
                      #  gamma=1.5, n_independent=2, n_shared=2,
                      #  cat_emb_dim=1, lambda_sparse=1e-4, 
                      #  momentum=0.3, clip_value=2., optimizer_fn=torch.optim.Adam,
                      #  optimizer_params=dict(lr=1e-2), scheduler_params = {"gamma": 0.95,
                      #    "step_size": 20},
                      #  scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15,
                       
                    #   n_d=60, n_a=60, n_steps=1,
                    #   gamma=1.0, n_shared=3, mask_type='entmax',
                    #   lambda_sparse= 0.0001330247277926223,
                    #   momentum=0.5,
                    #   optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                    #   scheduler_params=dict(mode="min",
                    #                     patience=6, # changing sheduler patience to be lower than early stopping patience 
                    #                     min_lr=1e-5,
                    #                     factor=0.5,),
                    #   scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    #   verbose=0,

                      # n_d=56, n_a=56, n_steps=2,
                      # gamma=1.2, n_shared=3, mask_type='entmax',
                      # lambda_sparse= 0.000042388926657543235 ,
                      # optimizer_fn=torch.optim.Adam,
                      # momentum=0.5,
                      # optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                      # scheduler_params=dict(mode="min",
                      #                   patience=9, # changing sheduler patience to be lower than early stopping patience
                      #                   min_lr=1e-5,
                      #                   factor=0.5,),
                      # scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,

                    #n_d=64, n_a=64,
        # {'gamma': 1.4, 'lambda_sparse': 0.00014440245699539302, 'mask_type': 'entmax', 'momentum': 0.5, 'n_shared': 3,
        #  'n_steps': 2, 'patienceScheduler': 10}
                    n_steps=3,gamma=1.3,n_shared=1,
                    mask_type='entmax',
                    verbose=1,
                    device_name=DEVICE,
                    optimizer_fn = torch.optim.Adam,
                    optimizer_params=dict(lr=2e-2, weight_decay=3e-6),
                    momentum=0.7,
                    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    epsilon=1e-16,
                    scheduler_params = dict(mode="min",
                    patience=9, # changing sheduler patience to be lower than early stopping patience
                    min_lr=1e-5,
                    factor=0.5)
    )
    clf.fit(
    X_train, y_train,
    #eval_set=[(X_test, y_test)],
    max_epochs=MAX_EPOCHS , patience=9,
    batch_size=BATCH_SIZE, virtual_batch_size=VIRTUAL_BATCH_SIZE,
    num_workers=0,
    drop_last=False
    )
    if RECOG:
        ml_utils.save(clf, MODEL+'/' + str(lq_mass) + '/models', 'TABNET_weights_recog')
    else:
        ml_utils.save(clf, MODEL + '/' + str(lq_mass) + '/models', 'TABNET_weights')
    # preds = clf.predict(X_test)
    # print(preds)
    print("TABNET classification finished for "+ str(lq_mass))
    return clf


    
    





