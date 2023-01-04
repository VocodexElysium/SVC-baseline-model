import torch.nn as nn
from pyheaven.torch_utils import *

class LSTMBackbone(nn.Module):
    def __init__(self,
        input_dim = 62,
        hidden_dim = 512,
        num_layers = 3,
        output_dim = 2048,
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
        self.linear = FC(2 * hidden_dim, output_dim, flatten=False, activation=nn.LeakyReLU(inplace=True))

    def forward(self, x):
        y = x
        # print(y.shape)
        t = self.model(y)[0]
        # print(t.shape)
        return self.linear(t)