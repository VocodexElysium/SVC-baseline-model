import torch.nn as nn
from pyheaven.torch_utils import *

class LSTMBackbone(nn.Module):
    def __init__(self,
        input_dim = 144,
        hidden_dim = 512,
        num_layers = 3,
        output_dim = 513,
    ):
        super(LSTMBackbone, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.linear = FC(2 * hidden_dim * num_layers, output_dim, activation=nn.LeakyReLU(inplace=True))

    def forward(self, x):
        tmpx = x['main']
        tmp = self.model(tmpx)[1][0].transpose(0, 1)
        x = tmp.reshape(tmpx.shape[0], -1)
        return self.linear(x)