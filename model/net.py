import torch
import torch.nn as nn
from pyheaven.torch_utils import *

class WORLDParamClassifier(nn.Module):
    def __init__(self, input_dim=2048, output_dim=60):
        super(WORLDParamClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lin1 = FC(self.input_dim, self.input_dim // 2, flatten=False, dropout=None, activation=nn.LeakyReLU(inplace=True))
        self.lin2 = FC(self.input_dim // 2, self.input_dim // 4, flatten=False, dropout=None, activation=nn.LeakyReLU(inplace=True))
        self.lin3 = FC(self.input_dim // 4, self.output_dim, flatten=False, dropout=None, activation=nn.Identity())

    def forward(self, x):
        # print(x.shape)
        return self.lin3(self.lin2(self.lin1(x)))

class Net(nn.Module):
    def __init__(self, backbone, classifier):
        super(Net, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.embedding = nn.Embedding(62, 62, 0)
        
    def forward(self, x):
        # print("herere")
        # print(x.device)
        # print(x.shape)
        x = torch.squeeze(x, 2)
        # print(x.shape)
        # print("herere")
        # print(x.device)
        # tmp = nn.Embedding(62, 62, 0)
        # print(tmp.device)
        x = self.embedding(x)
        x = self.backbone(x)
        x = self.classifier(x)
        # print(x.shape)
        return x