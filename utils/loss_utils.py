import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class maeloss(nn.Module):
    def forward(self, pred, true):
        return nn.L1Loss(pred, true)
    
class mseloss(nn.Module):
    def forward(self, pred, true):
        return nn.MSELoss(pred, true)

LOSSES_MAPPING = {
    'maeloss': {
        'regl': maeloss(),
    },
    'mseloss': {
        'regl': mseloss(),
    }
}