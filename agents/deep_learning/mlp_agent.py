import sys
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/utils/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/agents/")
import gc
import traceback
import torch
from torch import nn
from torch.backends import cudnn
import torch.optim as optim
from torchinfo import summary
import config.constants as C
from base import BaseAgent
from models.mlp import MLP
from datasets.dataToDataloader import lq_DataLoader
from utils.ml_utils import save
from utils.ml_utils import *


from captum.attr import IntegratedGradients
import numpy as np

class MLPAgent(BaseAgent):
    """
    This Agent is an implementation of BaseAgent implementing the functionality of a fully connected feed-forward network
    It takes configuration from the passed configuration file and runs the training and testing phase, then also implements the
    evaluation of the optimal working point for the classification based either on significance or special tt selection
    """

    def __init__(self, config, path, mass_train, mass_test):
        super().__init__(config, path, mass_train, mass_test)
        self.model = MLP(config)
        print("toto bude print model")
        self.printmodel()
        print("toto byl print model")
        self.data_loader = lq_DataLoader(config=config, path=self.path, mass_train=mass_train, mass_test=mass_test)
        if self.config.use_weights_in_loss:
            # if this parameter is true, than weights are applied during the computatiton of the loss function
            self.criterion = nn.NLLLoss(reduction='none')
        else:
            self.criterion = nn.NLLLoss()

        if config.optimizer == C.ADAM_OPTIMIZER:
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate,
                                        betas=(config.beta1, config.beta2),
                                        weight_decay=config.weight_decay)
        elif config.optimizer == C.SGD_OPTIMIZER:
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=self.config.momentum)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.current_epoch = 1
        self.current_iteration = 0
        self.best_metric = 0
        self.device = config.device
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []

    def run(self):
        try:
            self.logger.info("Using {} optimizer".format(self.optimizer.__str__()))
            self.logger.info("Training on {} examples".format(len(self.data_loader.train_loader.dataset)))
            self.logger.info("Validating on {} examples".format(len(self.data_loader.valid_loader.dataset)))
            self.logger.info("Testset size on {} examples".format(len(self.data_loader.test_loader.dataset)))
            self.train()
            if self.config.run_test_phase:
                self.test()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        """
        for epoch in range(1, self.config.max_epoch + 1):
            self.train_one_epoch()
            self.validate()
            if self.should_early_stop():
                break
            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        """

        self.model.train()
        training_loss = 0
        correct = 0
        for batch_idx, (data, target, weights) in enumerate(self.data_loader.train_loader):
            data, target, weights = data.to(self.device), target.to(self.device), weights.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            if self.config.use_weights_in_loss:
                loss = (loss * weights / weights.sum()).sum().mean()
            loss.backward()
            self.optimizer.step()
            training_loss += loss.item()
            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    self.current_epoch, batch_idx * len(data), len(self.data_loader.train_loader.dataset),
                                        100. * batch_idx / len(self.data_loader.train_loader)))
            self.current_iteration += 1
            # Accuracy
            predictions_all_classes = torch.exp(output)
            top_prediction, top_class = predictions_all_classes.topk(1, dim=1)
            equals = top_class == target.view(*top_class.shape)
            correct += equals.sum().item()
        else:
            self.train_acc.append(correct / len(self.data_loader.train_loader.dataset))
            self.train_losses.append(training_loss / len(self.data_loader.train_loader))
            self.logger.info('END of Training Epoch: {}\t{}\tTRAINING LOSS: {:.4f}\n'.format(
                self.current_epoch, "-" * 5 + ">", training_loss / len(self.data_loader.train_loader)))

    def validate(self):
        """
        One cycle of model validation
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target, weights in self.data_loader.valid_loader:
                data, target, weights = data.to(self.device), target.to(self.device), weights.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                if self.config.use_weights_in_loss:
                    val_loss += (loss * weights / weights.sum()).sum().mean().item()
                else:
                    val_loss += loss.item()
                predictions_all_classes = torch.exp(output)
                top_prediction, top_class = predictions_all_classes.topk(1, dim=1)
                equals = top_class == target.view(*top_class.shape)
                correct += equals.sum().item()
        self.val_acc.append(correct / len(self.data_loader.valid_loader.dataset))
        self.val_losses.append(val_loss / len(self.data_loader.valid_loader))
        self.logger.info('Validation Epoch {}: Validation Loss: {:.4f}, ACCURACY: {}/{} ({:.0f}%)\n'.format(
            self.current_epoch, val_loss / len(self.data_loader.valid_loader), correct,
            len(self.data_loader.valid_loader.dataset),
                                100. * correct / len(self.data_loader.valid_loader.dataset)))

    def test(self):
        """
        Test the model on test set
        :return:
        """
        self.model.eval()
        test_loss = 0
        self.all_p_classes = np.array([]).reshape((0, 1))
        self.all_weights = np.array([]).reshape(0)
        self.all_targets = np.array([]).reshape((0))
        self.all_p_probs = np.array([]).reshape((0, self.config.output_classes))
        with torch.no_grad():
            for batch_idx, (data, target, weights) in enumerate(self.data_loader.test_loader):
                data, target, weights = data.to(self.device), target.to(self.device), weights.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                if self.config.use_weights_in_loss:
                    test_loss += (loss * weights / weights.sum()).sum().mean().item()
                else:
                    test_loss += loss.item()
                predictions_all_classes = torch.exp(output)
                _, top_class = predictions_all_classes.topk(1, dim=1)
                weights = weights.cpu()
                predictions_all_classes = predictions_all_classes.cpu()
                top_class = top_class.cpu()
                target = target.cpu()

                self.all_weights = np.concatenate((self.all_weights, weights))
                self.all_p_probs = np.concatenate((self.all_p_probs, predictions_all_classes))
                self.all_p_classes = np.concatenate((self.all_p_classes, top_class))
                self.all_targets = np.concatenate((self.all_targets, target))
                if batch_idx % self.config.log_interval == 0:
                    self.logger.info('Testing: [{}/{} ({:.0f}%)]'.format(
                        batch_idx * len(data), len(self.data_loader.test_loader.dataset),
                        100. * batch_idx / len(self.data_loader.test_loader)))
                # if sys.platform == "darwin":
                #     if batch_idx > 5:
                #         self.logger.warning("Testing only 5 batch!!!!!! NOT FOR PRODUCTION")
                #         break
        self.all_p_classes = self.all_p_classes.flatten()
        self.all_targets = self.all_targets.flatten()

    def get_feature_importances(self, wr_class: int):
        """

        :param wr_class: with respect to which class should the feature attribution be computed
        :return:
        """
        test_set_full=torch.empty(size=(0,self.config.n_features))
        test_set_targets_full = np.array([]).reshape((0))
        for batch_idx, (data, target, weights) in enumerate(self.data_loader.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = data.cpu()
            target = target.cpu()
            test_set_full = torch.cat([test_set_full, data], dim=0)
            # test_set_targets_full = np.concatenate((test_set_targets_full, target))
        ig = IntegratedGradients(self.model)
        test_set_full.requires_grad_()
        test_set_full = test_set_full.cpu()
        # del self.model
        # gc.collect()
        # torch.cuda.empty_cache()
        # self.model = self.model.cpu()
        # attr, delta = ig.attribute(test_set_full, target=wr_class, return_convergence_delta=True)
        # attr = attr.detach().numpy()
        # #attr = attr.cpu()
        # visualize_importances(self.read_features(self.config), np.mean(attr, axis=0), config=self.config)


    def printmodel(self):
        self.logger.info("Printing whole model")
        self.logger.info(self.model)
        model_stats = str(summary(self.model, (1, 1, self.config.n_features), verbose=0))
        self.logger.info("Printing model parts used in forward pass")
        self.logger.info(model_stats)

    def save_predictions(self, path: str, true_y: np.array, predicted_y: np.array, predicted_probs: np.array, weights: np.array):
        """
        Saves predictions as numpy files
        :param true_y: true classes
        :param predicted_y:  predicted classes
        :param predicted_probs:  predicted probabilities of the classes
        :param weights: sample weights
        :return:
        """
        with open(path + 'MLP_true_y.npy', 'wb') as f:
            np.save(f, true_y)
        with open(path+'MLP_test_pred_y.npy', 'wb') as f:
            np.save(f, predicted_y)
        with open(path + 'MLP_test_pred_probsa.npy', 'wb') as f:
            np.save(f, predicted_probs)
        with open(path + 'MLP_weights.npy', 'wb') as f:
            np.save(f, weights)
        save(weights, path + '/models', 'MLP_weights')
        save(predicted_y, path, 'y_test_pred_MLP')
        save(predicted_probs, path, 'y_test_pred_proba_MLP')

    def should_early_stop(self) -> bool:
        if self.config.earlystop_enabled:
            if len(self.val_losses) > self.config.earlystop_epochs:
                reference_num: float = self.val_losses[-(self.config.earlystop_epochs + 1)]
                for num in self.val_losses[(len(self.val_losses) - self.config.earlystop_epochs):]:
                    if num < reference_num:
                        return False
                self.logger.info("Stopping training after {} epochs".format(self.current_epoch))
                return True
        return False
