from torch.nn import (
    BCELoss, CosineEmbeddingLoss, CTCLoss, HingeEmbeddingLoss,
    KLDivLoss, MarginRankingLoss, MSELoss, MultiLabelMarginLoss, MultiLabelSoftMarginLoss, SmoothL1Loss,
    MultiMarginLoss, NLLLoss, PoissonNLLLoss, SoftMarginLoss, TripletMarginLoss, NLLLoss2d, L1Loss
)

from .crossentropy import (LabelSmoothingCrossEntropyLoss, SoftTargetCrossEntropyLoss,
                           OhemCrossEntropyLoss, WeightedBCEWithLogitsLoss, WeightedCrossEntropyLoss,
                           MultiScaleCrossEntropyLoss, BCEWithLogitsLoss, CrossEntropyLoss,
                           SelfCrossEntropyLoss, SelfBCELoss)
from .lovasz import LovaszLoss
from .dice import DiceLoss
from .jaccard import JaccardLoss
from .focal import FocalLoss
