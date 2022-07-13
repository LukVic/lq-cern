import os
import datetime
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
import shutil
import json
from typing import List

from easydict import EasyDict
from pprint import pprint

def setup_logging(log_dir : str, file_handler = True):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)
    if len(main_logger.handlers) > 0:
        for i in range(len(main_logger.handlers)):
            main_logger.handlers.pop()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))
    main_logger.addHandler(console_handler)

    if file_handler:
        exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
        exp_file_handler.setLevel(logging.DEBUG)
        exp_file_handler.setFormatter(Formatter(log_file_format))

        exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
        exp_errors_file_handler.setLevel(logging.WARNING)
        exp_errors_file_handler.setFormatter(Formatter(log_file_format))
        main_logger.addHandler(exp_file_handler)
        main_logger.addHandler(exp_errors_file_handler)




def get_config_from_json(json_file : str) -> EasyDict:
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            config = EasyDict(config_dict)
            return config
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def process_config(input_config : EasyDict) -> EasyDict:
    """
    -get the json file
    -process it with EasyDict to be accessible as attributes
    -edit the path of the experiments folder
    -create some important directories in the experiment folder
    -setup the logging in the whole program
    Then return the config
    :param input_config: the path of the config file
    :return: config object(namespace)
    """
    config = input_config
    try:
        print(" --------------------------------------- ")
        print("The experiment name is {}".format(config.exp_name))
        print(" --------------------------------------- ")
    except AttributeError:
        print("Experiment name not provided..")
        exit(-1)
        
    today = datetime.datetime.today().date().isoformat()
    experiment = datetime.datetime.now().time().isoformat()
    config.exp_id = config.exp_series+"/"+today + "/" + config.exp_name 

    config.summary_dir = os.path.join("experiments", config.exp_id, "summaries/")
    config.checkpoint_dir = os.path.join("experiments", config.exp_id, "checkpoints/")
    config.out_dir = os.path.join("experiments", config.exp_id, "out/")
    config.log_dir = os.path.join("experiments", config.exp_id, "logs/")
    config.figures_dir = os.path.join("experiments", config.exp_id, "figures/")
    config.predictions_dir = os.path.join("experiments", config.exp_id, "predictions/")

    create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir, config.figures_dir, config.predictions_dir])
    with open(config.log_dir + 'config.json', 'w') as fp:
        json.dump(config, fp)
    setup_logging(config.log_dir)
    logging.getLogger().info(" --------------------------------------- ")
    logging.getLogger().info("The experiment name is {}".format(config.exp_name))
    logging.getLogger().info(" --------------------------------------- ")
    logging.getLogger().info("The experiment has the following configuration..")
    logging.getLogger().info(config)
    logging.getLogger().info("The pipeline of the project will begin now.")
    return config


def create_dirs(dirs : List[str]):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger().info("Creating directories error: {0}".format(err))
        exit(-1)
