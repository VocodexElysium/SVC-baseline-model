from utils import *
from pyheaven.plot_utils import *

_HEADER = """export PYTHONPATH="./"
"""

_TRAIN = """python3 train.py --dataset SVC --clean \\
                 --dataset-type {0} --backbone {1} --classifier {2} --batch-size {3} \\
                 --limit-train-batches {4} --limit-val-batches {5} \\
                 --identifier {6} $@
"""

_TEST = """python3  test.py --dataset SVC \\
                 --dataset-type {0} --backbone {1} --classifier {2} --batch-size {3} \\
                 --checkpoint ./experiment/{4}/checkpoints/ \\
                 --identifier {5} $@
"""

TASK_TEST_TEMPLATE = _HEADER + _TEST
TASK_TEMPLATE = _HEADER + _TRAIN + _TEST

BACKBONE_DATASETS = {
    'param': ("spec","conv",4),
}

if __name__=="__main__":
    args = HeavenArguments.from_parser([
        LiteralArgumentDescriptor("backbone",choices=['lstm'],default='lstm'),
        LiteralArgumentDescriptor("classifier",choices=['parameter'],default='parameter'),
        StrArgumentDescriptor("exp-info",short='info',default=None),
        FloatArgumentDescriptor("limit_train_batches", short="limtr", default=1.0),
        FloatArgumentDescriptor("limit_val_batches", short="limvl", default=1.0),
        SwitchArgumentDescriptor("no-metric",short="nom"),
        SwitchArgumentDescriptor("add-to-ensemble",short="add"),
        SwitchArgumentDescriptor("clear",short="clr"),
        StrArgumentDescriptor("conda-env",short="conda",default="source /home/<USERNAME>/miniconda3/bin/activate; conda activate base"),
        # StrArgumentDescriptor("conda-env",short="conda",default="conda activate py38"),
        StrArgumentDescriptor("cuda", short="cd", default="0"),
    ])
    print("Initializing ...")
    