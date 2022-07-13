"""
Main - overview
-get the config file
-process the config file and load the configuration
-create an agent instance
-run agent training
-generate Excel result
"""
import sys
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/agents/deep_learning/")

import numpy as np
from config.conf import *
import config.constants as C
from mlp_agent import *
from utils.ml_utils import get_random_experiment_name
#from utils.helpers import generate_excel_file
from utils.helpers import  get_params_of_best_experiment
INPUT_FEATURES: int = 89 
OUTPUT_CLASSES: int = 7 
EPOCH_ARRAY: List[int] = [20]
BATCH_ARRAY: List[int] = [16]
OPT_ARRAY: List[str] = ["adam"] 
USE_WEIGHTS_IN_LOSS: List[bool] = [True ]
LAYERS_ARRAY: List[List[List[int]]] = [
[[INPUT_FEATURES, 50], [50, 100], [100, 100], [100, 50], [50, OUTPUT_CLASSES]]]
LR_ARRAY: List[float] = [0.001]


def main():
    # condition to choose the correct config file based on where the experiment is ran (either locally, Azure VM, Russian Dubna server...)
    configfiles = ["test_vm.json"]
    expindex = 0
    experiment_series = get_random_experiment_name()
    experiment_series = experiment_series[0]
    for file_index, f in enumerate(configfiles):
        lrarray = LR_ARRAY
        epocharray = EPOCH_ARRAY
        batcharray = BATCH_ARRAY
        layersarray = LAYERS_ARRAY
        optarray = OPT_ARRAY
        use_weights = USE_WEIGHTS_IN_LOSS
        # Iterate over all parameter combinations and run the training
        for lr in lrarray:
            for opt in optarray:
                for ep in epocharray:
                    for layers_set in layersarray:
                        for batch in batcharray:
                            for we in use_weights:
                                print(experiment_series)
                                config_dict_template: EasyDict = get_config_from_json(C.JSON_CONFIG_DIRECTORY + f)
                                config_dict_template["exp_series"] = experiment_series
                                config_dict_template["exp_name"] = "EXP_F_"+str(file_index)+"_OPT_"+str(expindex)
                                config_dict_template["layers"] = layers_set
                                config_dict_template["max_epoch"] = int(ep)
                                config_dict_template["learning_rate"] = float(lr)
                                config_dict_template["n_features"] = INPUT_FEATURES
                                config_dict_template["output_classes"] = OUTPUT_CLASSES
                                config_dict_template["batch_size"] = batch
                                config_dict_template["optimizer"] = opt
                                config_dict_template["use_weights_in_loss"] = we
                                config = process_config(config_dict_template)
                                print("---------------------------------------------")
                                print(config.agent)
                                agent_class = globals()[config.agent]
                                agent = agent_class(config)
                                agent.run()
                                agent.finalize()
                                expindex += 1
        #generate_excel_file(C.EXPERIMENTS_DIRECTORY + experiment_series)
        output_path = C.EXPERIMENTS_DIRECTORY + experiment_series
        output_csv = output_path + "/"+ C.EXP_RESULT_FILE
        layers_set, ep, batch, opt, we, lr = get_params_of_best_experiment(output_csv)
        config_dict_template: EasyDict = get_config_from_json(C.JSON_CONFIG_DIRECTORY + f)
        config_dict_template["exp_series"] = experiment_series
        config_dict_template["exp_name"] = "BEST_RUN"
        config_dict_template["layers"] = layers_set
        config_dict_template["max_epoch"] = int(ep)
        config_dict_template["learning_rate"] = float(lr)
        config_dict_template["n_features"] = INPUT_FEATURES
        config_dict_template["output_classes"] = OUTPUT_CLASSES
        config_dict_template["batch_size"] = batch
        config_dict_template["optimizer"] = opt
        config_dict_template["use_weights_in_loss"] = we
        # if config_dict_template["calculate_feature_importances_end"] == True:
        #     config_dict_template["calculate_feature_importances"] = True
        # else:
        #     config_dict_template["calculate_feature_importances"] = False
        config_dict_template["calculate_feature_importances"] = True
        config = process_config(config_dict_template)
        agent_class = globals()[config.agent]
        agent = agent_class(config)
        agent.run()
        agent.finalize()
        print("finish")
        print(experiment_series)
if __name__ == '__main__':
    main()
