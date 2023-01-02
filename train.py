from utils import *
from dataset import DATASET_MAPPING
from model import get_backbone, Net

import pytorch_lightning as pl
from pytorch_lightning.plugins import DeepSpeedPrecisionPlugin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint, EarlyStopping

from SVC import SVCModel, SplitDatasets, Identifier

if __name__=="__main__":
    args = HeavenArguments.from_parser([
        LiteralArgumentDescriptor("dataset_type", short="dt", choices=DATASET_MAPPING.keys(), default="param"),
        StrArgumentDescriptor("dataset", short="ds", default="WORLD"),
        StrArgumentDescriptor("project", short="pj", default="SVC"),
        ListArgumentDescriptor("train_splits", short="train", type=str),
        ListArgumentDescriptor("val_splits", short="val", type=str),
        ListArgumentDescriptor("test_splits", short="test", type=str),
        SwitchArgumentDescriptor("split_train_to_val", short="split2val"),
        SwitchArgumentDescriptor("split_train_to_test", short="split2test"),
    
        StrArgumentDescriptor("backbone", short="bb", default="multimodal"),
        IntArgumentDescriptor("feature_dim", short="f", default=2048),
        IntArgumentDescriptor("num_epochs", short="e", default=30),
        IntArgumentDescriptor("batch_size", short="b", default=2),
        IntArgumentDescriptor("grad_accum", short="ga", default=-64),
        IntArgumentDescriptor("examples", short="ex", default=128),

        FloatArgumentDescriptor("limit_train_batches", short="limtr", default=1.0),
        FloatArgumentDescriptor("limit_val_batches", short="limvl", default=1.0),
        FloatArgumentDescriptor("learning_rate", short="lr", default=2e-4),
        FloatArgumentDescriptor("warmup_start_lr_ratio", short="wlr", default=.01),
        FloatArgumentDescriptor("eta_min", short="em", default=1e-8),
        IntArgumentDescriptor("warmup_epochs", short="we", default=4),
        FloatArgumentDescriptor("weight_decay", short="wd", default=1e-4),
        FloatArgumentDescriptor("noise_augment", short="na", default=1e-4),
        FloatArgumentDescriptor("mode_dropout", short="dp", default=0.0),
    
        StrArgumentDescriptor("identifier", short="id", default=None),
        StrArgumentDescriptor("cuda", short="cd", default="0"),
        IntArgumentDescriptor("seed", short="sd", default=11451419),
        IntArgumentDescriptor("debug", default=-1),
        SwitchArgumentDescriptor("clean"),
        SwitchArgumentDescriptor("wandb",short='wandb'),
    ])

    if args.clean:
        CMD("rm -rf lightning_logs/*")
        CMD("rm -rf examples/*")
        CMD("rm -rf logs/*")
    if args.grad_accum < 0:
        args.grad_accum = -args.grad_accum // args.batch_size
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    