import torch
import torch.nn as nn
from pyheaven.torch_utils import *

class WORLDParamClassifier(nn.Module):
    def __init__(self, input_dim=2048, output_dim=60):
        super(WORLDParamClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lin1 = FC(self.input_dim, self.input_dim // 2, dropout=None, activation=nn.LeakyReLU(inplace=True))
        self.lin2 = FC(self.input_dim // 2, self.input_dim // 4, dropout=None, activation=nn.LeakyReLU(inplace=True))
        self.lin3 = FC(self.input_dim // 4, self.output_dim, dropout=None, activation=nn.Identity())

    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(x)))

class Net(nn.Module):
    def __init__(self, backbone, classifier):
        super(Net, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        x = nn.Embedding(62, 62, 0)(x)
        x = self.backbone(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x