import sys
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/agents/deep_learning/")

import numpy as np
from config.conf import *
import config.constants as C
from mlp_agent import *
#from utils.helpers import generate_excel_file
from utils.helpers import get_params_of_best_experiment

INPUT_FEATURES: int = 89 
OUTPUT_CLASSES: int = 7 
EPOCH_ARRAY: List[int] = [60]
BATCH_ARRAY: List[int] = [256]
OPT_ARRAY: List[str] = ["adam"]
USE_WEIGHTS_IN_LOSS: List[bool] = [True]
#LAYERS_ARRAY: List[List[List[int]]] = [
#[[INPUT_FEATURES, 50], [50, 128], [128, 256], [256, 512], [512, 256], [256, 128], [128, 50], [50, OUTPUT_CLASSES]]]
#LAYERS_ARRAY: List[List[List[int]]] = [[[INPUT_FEATURES, 100], [100, 200], [200,100], [100, OUTPUT_CLASSES]]]
#LAYERS_ARRAY: List[List[List[int]]] = [[[INPUT_FEATURES, 20], [20, 200], [200, 100], [100, 20], [20, OUTPUT_CLASSES]]]
LAYERS_ARRAY: List[List[List[int]]] = [[[INPUT_FEATURES, 20], [20, 50], [50, 20], [20, OUTPUT_CLASSES]]]
LR_ARRAY: List[float] = [0.0002]

def mlp_run(INPUT, MODEL, lq_mass_train, lq_mass_test):
    configfiles = ["test_vm.json"]
    experiment_series = "test"
    experiment_series = experiment_series
    lrarray = LR_ARRAY[0]
    epocharray = EPOCH_ARRAY[0]
    batcharray = BATCH_ARRAY[0]
    layersarray = LAYERS_ARRAY[0]
    optarray = OPT_ARRAY[0]
    use_weights = USE_WEIGHTS_IN_LOSS[0]
    f = configfiles[0]
    print(experiment_series)

    config_dict_template: EasyDict = get_config_from_json(C.JSON_CONFIG_DIRECTORY + f)
    config_dict_template["exp_series"] = experiment_series
    config_dict_template["exp_name"] = "EXP_F_"+"_OPT_"
    config_dict_template["layers"] = layersarray
    config_dict_template["max_epoch"] = int(epocharray)
    config_dict_template["learning_rate"] = float(lrarray)
    config_dict_template["n_features"] = INPUT_FEATURES
    config_dict_template["output_classes"] = OUTPUT_CLASSES
    config_dict_template["batch_size"] = batcharray
    config_dict_template["optimizer"] = optarray
    config_dict_template["use_weights_in_loss"] = use_weights
    config_dict_template["dataset_path"] = INPUT + "/"
    config = process_config(config_dict_template)
    print("---------------------------------------------")
    print(config.agent)
    #agent_class = globals()[config.agent]
    agent = MLPAgent(config, INPUT, lq_mass_train, lq_mass_test)
    agent.run()
    visualize_network_training_progress(agent.train_losses, agent.val_losses, path=agent.path)
    visualize_network_accuracy_progress(agent.train_acc, agent.val_acc, path=agent.path)
    agent.save_predictions(agent.path, agent.all_targets, agent.all_p_classes, agent.all_p_probs, agent.all_weights)
    #agent.finalize(INPUT)








