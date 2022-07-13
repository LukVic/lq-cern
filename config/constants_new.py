import sys, os

BASE_DIRECTORY = "/home/lucas/Documents/KYR/bc_thesis/thesis/project/"
EXP_RESULT_FILE = "experiment_results.csv"
JSON_CONFIG_DIRECTORY = BASE_DIRECTORY + "config/json/"
FEATURES_DIRECTORY = BASE_DIRECTORY + "config/feature_variations/"
EXPERIMENTS_DIRECTORY = BASE_DIRECTORY + "experiments/"
PKL_ROOT_FOLDER = BASE_DIRECTORY + "data/converted/"
CUTVARS_DIRECTORY = BASE_DIRECTORY + "config/cutvariable_variations/"
ROOT_FILE_OVERVIEW = BASE_DIRECTORY + "config/documents/event_scale_factors_all.xlsx"

MASS_TO_NUM_DICT = {500:0, 600: 1, 700: 2, 800:3, 900:4, 1000:5, 1100:6, 1200:7,
1300:8, 1400:9, 1500:10, 1600:11}

def all_classes_dict_train(mass:str):
    return {0:mass,1: 'ttH',2: 'ttW',3: 'ttZ',4: 'ttbar',5: 'vv',6: 'other'}
    
def all_classes_dict_test(mass:str):
    return {0:mass,1: 'ttH',2: 'ttW',3: 'ttZ',4: 'ttbar',5: 'vv',6: 'other'}
    
def all_classes_to_num_dict_train(mass:str):
    return  {mass:0,'tth':1,'ttw':2,'ttz':3,'ttbar':4,'vv': 5,'other': 6}

def all_classes_to_num_dict_test(mass:str):
    return {mass:0,'tth':1,'ttw':2,'ttz':3,'ttbar':4,'vv': 5,'other': 6}


