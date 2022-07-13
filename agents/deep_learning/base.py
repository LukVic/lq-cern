import logging
import os
from typing import List, Optional

import numpy as np
from easydict import EasyDict

from utils import ml_utils


class BaseAgent:
    """
    This abstract class contains the functions that an agent should implement (does not have to implement all of them).
    """

    def __init__(self, config : EasyDict, path: str, mass_train, mass_test):
        self.config = config
        self.logger = logging.getLogger("Agent")
        self.model = None
        self.features: Optional[List[str]] = None
        self.path = path + '/' + mass_test + '/'
        self.mass_train = mass_train
        self.mass_test = mass_test

    def load_checkpoint(self, file_name : str):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def read_features(self, config: EasyDict):
        if self.features is None:
            self.features = ml_utils.load(os.path.join(config.dataset_path,"f.pkl"))
        return self.features
    def save_checkpoint(self, file_name : str, is_best : int = 0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError

    def test(self):
        """
        Tests the model on test data and produces test results
        :return:
        """
        raise NotImplementedError

    def printmodel(self):
        raise NotImplementedError

    def save_predictions(self, true_y: np.array, predicted_y: np.array, predicted_probs: np.array, weights: np.array):
        raise NotImplementedError
