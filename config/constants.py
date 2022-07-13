import sys, os
#"l2SS1tau"
BASE_DIRECTORY = "/home/lucas/Documents/KYR/bc_thesis/thesis/project/"
EXP_RESULT_FILE = "experiment_results.csv"
JSON_CONFIG_DIRECTORY = BASE_DIRECTORY + "config/json/"
FEATURES_DIRECTORY = BASE_DIRECTORY + "config/feature_variations/"
EXPERIMENTS_DIRECTORY = BASE_DIRECTORY + "load/experiments/"
PKL_ROOT_FOLDER = BASE_DIRECTORY + "data/converted/"
CUTVARS_DIRECTORY = BASE_DIRECTORY + "config/cutvariable_variations/"
ROOT_FILE_OVERVIEW = BASE_DIRECTORY + "config/documents/event_scale_factors_all.xlsx"
# ROOT_FILE_OVERVIEW = BASE_DIRECTORY + "config/documents/event_scale_factors_ttanalysis.xlsx"
ADAM_OPTIMIZER = "adam"
SGD_OPTIMIZER = "sgd"
LQ_CLASSES_DICT = {0:"lq",
                        1: 'ttH',
                      2: 'ttW',
                      3: 'ttZ',
                      4: 'ttbar'}

ALL_CLASSES_DICT_TRAIN = [
{0:"lq",1: 'ttH',2: 'ttW',3: 'ttZ',4: 'ttbar',5: 'vv',6: 'other'}
]
ALL_CLASSES_DICT_TEST = [
{0:"lq",1: 'ttH',2: 'ttW',3: 'ttZ',4: 'ttbar',5: 'vv',6: 'other'}
]

ALL_CLASSES_TO_NUM_DICT_TRAIN = [
{'lq':0,'tth':1,'ttw':2,'ttz':3,'ttbar':4,'vv': 5,'other': 6}
]
ALL_CLASSES_TO_NUM_DICT_TEST = [
{'lq':0,'tth':1,'ttw':2,'ttz':3,'ttbar':4,'vv': 5,'other': 6}
]




LQ_CLASSES_SYMBOLIC_DICT = {
                    0: '$t\overline{t}$',
                    1: '$t\overline{t}H$',
                   2: '$t\overline{t}W$',
                   3: '$t\overline{t}Z$',
                    4: '$t\overline{t}T$',
                             }

ALL_CLASSES_SYMBOLIC_DICT = {
                    0: '$t\overline{t}$',
                    1: '$t\overline{t}H$',
                   2: '$t\overline{t}W$',
                   3: '$t\overline{t}Z$',
                    4: '$t\overline{t}T$',
                    5: '$t\overline{t}V$',
                    6: '$t\overline{t}O$',
                             }
ALL_CLASSES_SYMBOLIC_DICT_2 = {
                    '$t\overline{t}$' : 0,
                    '$t\overline{t}H$' : 1,
                    '$t\overline{t}W$' : 2,
                   '$t\overline{t}Z$' : 3,
                    '$t\overline{t}T$' : 4,
                    '$t\overline{t}V$' : 5,
                    '$t\overline{t}O$': 6,
                             }


EXPERIMENT_TYPE_LQ_ANALYSIS = "lq"
LQ_CLASS_NAMES = ['t$\overline{t}$','t$\overline{t}$H', 't$\overline{t}$W', 't$\overline{t}$Z', 't$\overline{t}$T']
LQ_CLASS_NAMES_LIST = [v.lower() for k,v in LQ_CLASSES_DICT.items()]

EXPERIMENT_TYPE_ALL_ANALYSIS = "all"
ALL_CLASS_NAMES = ['t$\overline{t}$','t$\overline{t}$H', 't$\overline{t}$W', 't$\overline{t}$Z', 't$\overline{t}$T', 't$\overline{t}$V' , 't$\overline{t}$O']
ALL_CLASS_NAMES_LIST_TRAIN = []
ALL_CLASS_NAMES_LIST_TEST = []
for idx, dict in enumerate(ALL_CLASSES_DICT_TRAIN):
    ALL_CLASS_NAMES_LIST_TRAIN.append([v.lower() for k,v in dict.items()])
for idx, dict in enumerate(ALL_CLASSES_DICT_TEST):
    ALL_CLASS_NAMES_LIST_TEST.append([v.lower() for k,v in dict.items()])

VECTORIZED = "_VECTORIZED_"