import logging
import os
import sys
sys.path.append("//")
sys.path.append("/utils/")
sys.path.append("/agents/")
import numpy as np
import pickle
import pandas as pd
import torch
import ml_utils as ml_utils
import confmatrix_prettyprint as cm
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import config.constants_new as C
from sklearn.model_selection import train_test_split
from config.conf import setup_logging
from pytorch_tabnet.tab_model import TabNetClassifier
import random
import math
from agents.deep_learning.tabnet_agent import tabAgent
CLASSIFIER = 'TABNET'

def recognize(FILE_PATH_TRAIN, FILE_PATH_TEST, FILE_OUTPUT, lq_mass_train, lq_mass_test):
    setup_logging("SplittingLogger", file_handler=False)
    logger = logging.getLogger("SplittingLogger")

    dicti = {
        500 : 0,
        600 : 1,
        700 : 2,
        800 : 3,
        900 : 4,
        1000: 5,
        1100 : 6,
        1200 : 7,
        1300 : 8,
        1400 : 9,
        1500 : 10,
        1600 : 11
    }

    classes = {0: 'lq_500',
                1: 'lq_600',
                2: 'lq_700',
                3: 'lq_800',
                4: 'lq_900',
                5: 'lq_1000',
                6: 'lq_1100',
               7: 'lq_1200',
               8: 'lq_1300',
               9: 'lq_1400',
               10: 'lq_1500',
               11: 'lq_1600',
                }

    #Load data for train
    logger.info("loading ")
    X_train = ml_utils.load(FILE_PATH_TRAIN+ '/' + lq_mass_test + '/X_train.pkl')
    y_train = ml_utils.load(FILE_PATH_TRAIN+ '/' + lq_mass_test + '/y_train.pkl')
    f_new = ml_utils.load(FILE_PATH_TRAIN+ '/' + lq_mass_test + '/f_new.pkl')
    w_train = ml_utils.load(FILE_PATH_TRAIN+ '/' + lq_mass_test + '/w_train.pkl')
    y_lq_all_train = ml_utils.load(FILE_PATH_TRAIN+ '/' + lq_mass_test + '/y_lq_all_train.pkl')

    df_train = pd.DataFrame(data=X_train)
    df_train['y_lq_all_train'] = y_lq_all_train
    df_train['y'] = y_train
    df_train['weight'] = w_train
    print(type(y_lq_all_train[0]))
    df_train['is_lq'] = ((df_train.y_lq_all_train >= 499.0) & (df_train.y_lq_all_train <= 1601.0))
    print(len(df_train))
    df_train = df_train[df_train.is_lq]
    print(len(df_train))

    y_tr_new = df_train.y.to_numpy()
    w_tr_new = df_train.weight.to_numpy()
    y_lq_tr_new = df_train.y_lq_all_train.to_numpy()
    df_train = df_train.drop(columns=["weight", "y", 'y_lq_all_train','is_lq'])
    X_tr_new = df_train.to_numpy()

    CLASSIFIER = 'TABNET'

    for idx, (event_1, event_2) in enumerate(zip(y_tr_new, y_lq_tr_new)):
            y_tr_new[idx] = dicti[y_lq_tr_new[idx]]

    ml_utils.save(X_tr_new, FILE_PATH_TRAIN +'/' +lq_mass_test, 'X_tr_new')
    ml_utils.save(y_tr_new, FILE_PATH_TRAIN +'/'+lq_mass_test, 'y_tr_new')

    #LETS TRAIN


    model = tabAgent(FILE_PATH_TRAIN, FILE_PATH_TRAIN, lq_mass_test)
    # model = ml_utils.train_model(X_tr_new, y_tr_new, w_train, 'CAT')
    # ml_utils.save(model, FILE_PATH_TRAIN + '/' + str(lq_mass_test) + '/models', 'CAT_weights_recog')
    # model = ml_utils.load(FILE_PATH_TRAIN + '/' +str(lq_mass_test) +'/models/'+CLASSIFIER+'_weights_recog.pkl')

    MAX_EPOCHS = 60
    BATCH_SIZE = 2048
    VIRTUAL_BATCH_SIZE = 512
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using {}".format(DEVICE))
    clf = TabNetClassifier(
        n_steps=3, gamma=1.3, n_shared=1,
        mask_type='entmax',
        verbose=1,
        device_name=DEVICE,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=3e-6),
        momentum=0.7,
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        epsilon=1e-16,
        scheduler_params=dict(mode="min",
                              patience=9,  # changing sheduler patience to be lower than early stopping patience
                              min_lr=1e-5,
                              factor=0.5)
    )
    clf.fit(
        X_tr_new, y_tr_new,
        # eval_set=[(X_test, y_test)],
        max_epochs=MAX_EPOCHS, patience=9,
        batch_size=BATCH_SIZE, virtual_batch_size=VIRTUAL_BATCH_SIZE,
        num_workers=0,
        drop_last=False
    )
    model = clf
    #Load data for test
    X_test = ml_utils.load(FILE_PATH_TRAIN + '/' + lq_mass_test + '/X_test.pkl')
    y_test = ml_utils.load(FILE_PATH_TRAIN + '/' + lq_mass_test + '/y_test.pkl')
    f_new = ml_utils.load(FILE_PATH_TRAIN + '/' + lq_mass_test + '/f_new.pkl')
    w_test = ml_utils.load(FILE_PATH_TRAIN + '/' + lq_mass_test + '/w_test.pkl')
    y_lq_all_test = ml_utils.load(FILE_PATH_TRAIN + '/' + lq_mass_test + '/y_lq_all_test.pkl')
    df_test = pd.DataFrame(data=X_test)
    df_test['y_lq_all_test'] = y_lq_all_test
    df_test['y'] = y_test
    df_test['weight'] = w_test
    df_test['is_lq'] = ((df_test.y_lq_all_test >= 499.0) & (df_test.y_lq_all_test <= 1601.0))
    print(len(df_test))
    df_test = df_test[df_test.is_lq]
    print(len(df_test))

    y_te_new = df_test.y.to_numpy()
    w_te_new = df_test.weight.to_numpy()
    y_lq_te_new = df_test.y_lq_all_test.to_numpy()
    df_test = df_test.drop(columns=["weight", "y", 'y_lq_all_test','is_lq'])
    X_te_new = df_test.to_numpy()

    for idx, (event_1, event_2) in enumerate(zip(y_te_new, y_lq_te_new)):
            y_te_new[idx] = dicti[y_lq_te_new[idx]]

    y_test_pred = ml_utils.predict(model, X_te_new)
  #  y_test_pred_proba = ml_utils.predict_proba(model, X_te_new)

    bins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for val in y_test_pred:
        bins[int(val)] += 1

    names = ['500', '600', '700', '800', '900', '1000', '1100', '1200', '1300', '1400', '1500', '1600']
    mss = [500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600]

    mean = 0
    for idx in range(12):
        mean += bins[idx]*mss[idx]
    mean_final = mean/len(X_te_new)
    print('-----------------------------------------')
    print(mean_final)

    stdev = 0
    for idx in range(12):
        stdev += (mean -  bins[idx]*mss[idx])**2

    stdev_final = math.sqrt(stdev/len(X_te_new))
    print(stdev_final)
    print('-----------------------------------------')
    plt.figure(figsize=(9, 3))
    plt.bar(names, bins)
    plt.show()