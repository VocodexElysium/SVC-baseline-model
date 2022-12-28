import torch.nn as nn

class MainBackbone(nn.Module):
    def __init__(self, net):
        super().__init__
        self.net = net
    
    def forward(self, x):
        return self.net(x['main'].unsqueeze(1))