from utils import *
from dataset import DATASET_MAPPING
from model import get_backbone, get_classifier, Net

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint, EarlyStopping

from SVC import SVCModel, SplitDatasets, Identifier

if __name__=="__main__":
    args = HeavenArguments.from_parser([
        LiteralArgumentDescriptor("dataset_type", short="dt", choices=DATASET_MAPPING.keys(), default="param"),
        StrArgumentDescriptor("dataset", short="ds", default="WORLD"),
        StrArgumentDescriptor("project", short="pj", default="SVC"),
    
        StrArgumentDescriptor("backbone", short="bb", default="lstm"),
        StrArgumentDescriptor("classifier", short="cl", default="parameter"),
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
    args.n_gpu = len([d for d in args.cuda.split(', ') if d.strip() != ''])

    # print(args.n_gpu)

    seed_everything(args.seed, workers = True)

    net = Net(
        backbone = get_backbone(args.backbone, args),
        classifier = get_classifier(args.classifier, args),
    )

    args.identifier = Identifier(args)
    SaveJson(args, pjoin("tasks", f"{args.indentifier}.json"), indent=4)
    
    dataset = DATASET_MAPPING[args.dataset_type](args.dataset)

    # print(dataset.device) 

    args.datasets = SplitDatasets(dataset, train=0.7, val=0.2, test=0.1)

    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.project,
            entity='<WANDB_ENTITY>',
            log_model="all",
            id=args.identifier
        )

    model = SVCModel(net, args=args)

    # print(next(model.parameters()).device)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.n_gpu > 0) else "cpu")
    model = model.cuda()
    # print(next(model.parameters()).device)

    trainer = pl.Trainer(
        max_epochs = args.num_epochs,
        gradient_clip_val = 1.0,
        accumulate_grad_batches = args.grad_accum,
        callbacks = [
            StochasticWeightAveraging(swa_lrs=0.05),
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint()
            # ModelCheckpoint(monitor="valid_celoss", mode="min"),
            # EarlyStopping(monitor="valid_celoss", mode="min", patience=8, check_finite=True),
        ],

        gpus = args.n_gpu,
        auto_select_gpus = True,

        logger = wandb_logger if args.wandb else True,
        log_every_n_steps = 100,
        
        benchmark = True,
        strategy = DDPStrategy(find_unused_parameters = (args.model == 'group')),
        limit_train_batches = args.limit_train_batches,
        limit_val_batches = args.limit_val_batches,

        auto_lr_find = False
    )

    if args.wandb:
        wandb_logger.watch(model)
    
    trainer.fit(model)
    trainer.test(ckpt_path="best")