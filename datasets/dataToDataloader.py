from typing import List
import numpy as np
import torch
import gc
import torchvision.utils as v_utils
from easydict import EasyDict
from torch.utils.data import DataLoader, TensorDataset, Dataset
from utils import  ml_utils
import config.constants as C
from sklearn.model_selection import train_test_split
from datasets.datasets import lq_DataSet
from utils.ml_utils import print_class_frequencies_in_dataset
import logging
import sys
class lq_DataLoader:
    """
    This class serves as a loader of the data for training and testing, it takes config of the project as input and
    based on it loads data from the specified directories - which is saved as .pkl files. This class is later used during
    training and testing to iterate over the dataset because it contains the DataLoaders self.train_loader, self.test_loader,
    and self.valid_loader
    """
    def __init__(self, config : EasyDict, path: str, mass_train: str, mass_test: str):
        """
        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("DataLoader")
        self.dataset_path = path
        self.logger.info("Dataset path: {}".format(self.dataset_path))

        x_trainval = ml_utils.load(self.dataset_path + "X_train.pkl")
        y_trainval = ml_utils.load(self.dataset_path + "y_train.pkl")
        w_trainval = ml_utils.load(self.dataset_path + "w_train.pkl")

        x_test = ml_utils.load(self.dataset_path + "X_test.pkl")
        y_test = ml_utils.load(self.dataset_path + "y_test.pkl")
        w_test = ml_utils.load(self.dataset_path + "w_test.pkl")

        f_new = ml_utils.load(self.dataset_path + "f_new.pkl")


        self.logger.info("The whole Train dataset is made of: {} rows with {} features ".format(x_trainval.shape[0], x_trainval.shape[1]))
        self.logger.info("The whole Test dataset is made of: {} rows with {} features ".format(x_test.shape[0], x_test.shape[1]))

        self.logger.info("Memory size of X_train_val = " + str(sys.getsizeof(x_trainval)) + " bytes")
        self.logger.info("Memory size of X_train_val = " + str(sys.getsizeof(x_trainval)) + " bytes")

        x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(x_trainval, y_trainval, w_trainval,
                                                                  test_size= 1 - config.train_val_split,
                                                                  random_state=0,
                                                                  stratify=y_trainval)
        del x_trainval
        gc.collect()
        # w_train = x_train[:,0]
        # x_train = x_train[:,1:]
        # w_val = x_val[:,0]
        # x_val = x_val[:,1:]
        # w_test = x_test[:, 0]
        # x_test = x_test[:, 1:]
        print_class_frequencies_in_dataset(y_train, "y_train", self.logger)
        print_class_frequencies_in_dataset(y_val, "y_val", self.logger)
        print_class_frequencies_in_dataset(y_test, "y_test", self.logger)

        trainset = lq_DataSet(x_train, y_train, w_train)
        valset = lq_DataSet(x_val, y_val, w_val)
        testset = lq_DataSet(x_test, y_test, w_test)

        self.train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valset, batch_size=config.batch_size, shuffle=False)
        self.test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False)

    def load_data_from_datasets(self, path_full: str,config ,method = "pkl"):
        """
        The method enables loading data from multiple datasets together (mc16e, mc16a ...) by using the string separator
        :param dataset_joined_path: path to the datasets separated by string "|"
        :param config: config file
        :param method: it can either load .pkl files or .npy files
        :return:
        """
        X_train = ml_utils.load_numpyarray(path_full + "X_train.pkl")
        y_train = ml_utils.load_numpyarray(path_full + "y_train.pkl")
        w_train = ml_utils.load_numpyarray(path_full + "w_train.pkl")

        X_test = ml_utils.load_numpyarray(path_full + "X_test.pkl")
        y_test = ml_utils.load_numpyarray(path_full + "y_test.pkl")
        w_test = ml_utils.load_numpyarray(path_full + "w_test.pkl")

        f_new = ml_utils.load_numpyarray(path_full + "f_new.npy")
        return

