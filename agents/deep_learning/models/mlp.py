import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict


class MLP(nn.Module):
    """
    This class serves as a fully-connected feedforward neural network builder and the model itself, it receives a project config with
    the architecture of the network and creates a pytorch ModuleList with the corresponding number of layers etc.

    It also implements the forward method used during training and predicting.
    """
    def __init__(self, config: EasyDict):
        super().__init__()
        array = []
        for i, l in enumerate(config.layers):
            array.append(nn.Linear(l[0], l[1]))
            array.append(nn.Dropout(p=0.2))
            if i < len(config.layers) - 1:
                array.append(nn.ReLU())
            else:
                array.append(nn.LogSoftmax(dim=1))
        self.module_list = nn.ModuleList(array)

    def forward(self, x):
        # make sure input tensor is flattened
        # x = x.view(x.shape[0], -1)
        for f in self.module_list:
            x = f(x)
        return x

