from utils import *
import pytorch_lightning as pl
from pyheaven.torch_utils import HeavenDataset, HeavenDataLoader
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

import numpy as np

class SVCModel(pl.LightningModule):
    def __init__(self, net, args):
        super().__init__()
        self.net = net
        self.args = args
        self.learning_rate = args.learning_rate
        
    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.net.parameters(),
            lr = self.args.learning_rate,
            weight_decay = self.args.weight_decay   
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs = self.args.warmup_epochs,
            max_epochs = self.args.num_epochs,
            warmup_start_lr = self.args.warmup_start_lr_ratio * self.args.learning_rate,
            eta_min = self.args.eta_min
        )
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return HeavenDataLoader(DataLoader(self.args.datasets.train, batch_size=self.args.batch_size, num_workers=8, shuffle=True), self.args.datasets.train)

    def val_dataloader(self):
        return HeavenDataLoader(DataLoader(self.args.datasets.val,batch_size=self.args.batch_size,num_workers=8,shuffle=False), self.args.datasets.val)
    
    def test_dataloader(self):
        return HeavenDataLoader(DataLoader(self.args.datasets.test,batch_size=1,num_workers=8,shuffle=False), self.args.datasets.test)

    def run_batch(self, batch, split='train', batch_idx=-1):
        ppg, mgc = batch
        pred = self(ppg)
        loss = mseloss(pred, mgc)
        result = {
            'src': batch,
            'prd': pred,
            'mgc': mgc,
            'loss': loss
        }
        return result

    def training_step(self, train_batch, batch_idx):
        result = self.run_batch(train_batch, split='train', batch_idx=batch_idx)
        if torch.any(torch.isnan(result['loss'])):
            return None
        else:
            return result['loss']

    def validation_step(self, val_batch, batch_idx):
        result = self.run_batch(val_batch, split='valid', batch_idx=batch_idx)
        if torch.any(torch.isnan(result['loss'])):
            return None
        else:
            return result['loss']

    def test_step(self, test_batch, batch_idx):
        result = self.run_batch(test_batch, split='test', batch_idx=batch_idx)
        if torch.any(torch.isnan(result['loss'])):
            return None
        else:
            return result['loss']

def SplitDatasets(dataset, train=0.7, val=0.2, test=0.1, cut=-1):
    n = len(dataset)
    indices = [i for i in range(n)]
    np.random.shuffle(indices)
    datasets = {}
    num_train = int(train * n)
    num_val = int(val * n)
    datasets['train'] = Subset(dataset, indices[: num_train])
    datasets['val'] = Subset(dataset, indices[num_train: num_train + num_val])
    datasets['test'] = Subset(dataset, indices[num_train + num_val: ])
    for name, dataset in datasets.items():
        datasets[name] = HeavenDataset(dataset)
    return MemberDict(datasets)

def Identifier(args):
    if args.identifier is None:
        res = f"{args.backbone}_{args.model}_{FORMATTED_TIME('%Y-%m-%d_%H.%M')}"
    else:
        res = args.identifier
    return res
