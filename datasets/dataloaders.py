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
    def __init__(self, config : EasyDict):
        """
        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("DataLoader")
        self.dataset_path = C.BASE_DIRECTORY + config.dataset_path
        self.logger.info("Dataset path: {}".format(self.dataset_path))
        X, y, f, w = self.load_data_from_datasets(self.dataset_path, config,method = "pkl")


        self.logger.info("The whole dataset is made of: {} rows with {} features ".format(X.shape[0], X.shape[1]))
        self.logger.info("Memory size of X = " + str(sys.getsizeof(X)) + " bytes")
        X = np.concatenate((w.reshape(w.shape[0],1), X), axis = 1)
        x_trainval, x_test, y_trainval, y_test = train_test_split(X, y,
                                                            test_size= 1 - config.train_test_split,
                                                            random_state=0,
                                                            stratify=y)
        del X
        gc.collect()
        positive_weight_mask = np.array(np.where(x_trainval[:,0] > 0)).squeeze()
        self.logger.info("Removing {} % events from the training set due to negative weights"
                         .format((x_trainval.shape[0]-len(positive_weight_mask))/x_trainval.shape[0]*100))

        x_trainval = np.take(x_trainval,positive_weight_mask, axis = 0)
        y_trainval = np.take(y_trainval,positive_weight_mask, axis = 0)

        self.logger.info("Memory size of x_trainval = " + str(sys.getsizeof(x_trainval)) + " bytes")

        x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval,
                                                                  test_size= 1 - config.train_val_split,
                                                                  random_state=0,
                                                                  stratify=y_trainval)
        del x_trainval
        gc.collect()
        w_train = x_train[:,0]
        x_train = x_train[:,1:]
        w_val = x_val[:,0]
        x_val = x_val[:,1:]
        w_test = x_test[:, 0]
        x_test = x_test[:, 1:]
        print_class_frequencies_in_dataset(y_train, "y_train", self.logger)
        print_class_frequencies_in_dataset(y_val, "y_val", self.logger)
        print_class_frequencies_in_dataset(y_test, "y_test", self.logger)

        trainset = lq_DataSet(x_train, y_train, w_train)
        valset = lq_DataSet(x_val, y_val, w_val)
        testset = lq_DataSet(x_test, y_test, w_test)

        self.train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valset, batch_size=config.batch_size, shuffle=False)
        self.test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False)

    def load_data_from_datasets(self, dataset_joined_path: str,config ,method = "pkl"):
        """
        The method enables loading data from multiple datasets together (mc16e, mc16a ...) by using the string separator
        :param dataset_joined_path: path to the datasets separated by string "|"
        :param config: config file
        :param method: it can either load .pkl files or .npy files
        :return:
        """
        #datasets: List =  dataset_joined_path.split("|")
        datasets: List = [config.dataset_path]
        X_full = None
        y_full = None
        f_full = None
        weights_full = None
        for ds in datasets:
            if method == "numpy":
                X = ml_utils.load_numpyarray(ds + "X.npy")
                y = ml_utils.load_numpyarray(ds + "y.npy")
                f = ml_utils.load(ds + "f.pkl")
                w = ml_utils.load(ds + "w.pkl")
            elif method == "pkl":
                if sys.platform == "darwin":
                    # MacOS local
                    try:
                        X = ml_utils.load_compress(ds + "X.pkl.gzip", "gzip")
                    except:
                        X = ml_utils.load(ds + "X.pkl")
                else:
                    # Server VM Linux
                    X = ml_utils.load(ds + "X.pkl")
                    y = ml_utils.load(ds + "y.pkl")
                    f = ml_utils.load(ds + "f.pkl")
                    w = ml_utils.load(ds + "w.pkl")
            else:
                raise ValueError("Unknown loading method")
            if X_full is None:
                X_full = X
                y_full = y
                f_full = f
                weights_full = w
            else:
                X_full = np.concatenate([X_full, X])
                y_full = np.concatenate([y_full, y])
                f_full = f_full + f
                weights_full = np.concatenate([weights_full, w])
        return X_full, y_full, f_full, weights_full

