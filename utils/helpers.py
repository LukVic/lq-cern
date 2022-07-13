import ast

import pandas as pd
import os
import logging, sys
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/utils/")
from config.conf import get_config_from_json


def get_params_of_best_experiment(experiment_results_path: str) -> tuple:
    df = pd.read_csv(experiment_results_path)
    df = df.sort_values(by="significance_simple", ascending=False)
    layers = ast.literal_eval(df.loc[0, "layers"])
    epochs = int(df.loc[0, "epochs"])
    opt = str(df.loc[0, "optimizer"])
    lr = float(df.loc[0, "lr"])
    batch = int(df.loc[0, "batch"])
    we = bool(df.loc[0, "use_weights"])
    return layers, epochs, batch, opt, we, lr

def get_nn_test_scores(path: str):
    df = pd.read_csv(path)
    f1 = df.iloc[0,0]
    acc = df.iloc[0,1]
    auc = df.iloc[0,2]
    return acc, f1, auc
def get_nn_sign_scores(path: str):
    df = pd.read_csv(path)
    s = df.iloc[0,0]
    ss = df.iloc[0,1]
    sens = df.iloc[0,3]
    return s, ss, sens
def get_nn_config_data(path: str):
    c = get_config_from_json(path)
    layers = c.layers
    batch = c.batch_size
    opt = c.optimizer
    lr = c.learning_rate
    prods = c.productions_included
    epoch = c.max_epoch
    weights_used = c.use_weights_in_loss
    return layers,batch,lr,opt,epoch, prods,weights_used

if __name__ == '__main__':
    input = sys.argv[1]
    print(input)