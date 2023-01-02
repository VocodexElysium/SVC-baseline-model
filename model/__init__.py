from .main_backbone import MainBackbone
from .lstm_backbone import LSTMBackbone
from .net import WORLDParamClassifier, Net

from pyheaven.torch_utils import TimmBackbone

def get_backbone(backbone_type, args):
    feature_dim = args.feature_dim
    if "lstm" in backbone_type:
        backbone = LSTMBackbone(output_dim=feature_dim)
    else:
        raise NotImplementedError
    return backbone

def get_classifier(classifier_type, args):
    feature_dim = args.feature_dim
    if classifier_type == "parameter":
        classifier = WORLDParamClassifier(
            input_dim=feature_dim
        )
    else:
        raise NotImplementedError
    return classifier