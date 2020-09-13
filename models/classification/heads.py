import torch.nn.functional as F
from torch import nn, Tensor

from models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from registry import HEADS


@HEADS.register_class
class ClassificationHead(nn.Module):
    def __init__(self, num_features, num_classes, global_pool='avg', drop_rate=0):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)

        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)
        else:
            self.classifier = nn.Identity()

        self.init_weights()

    def forward(self, input: Tensor) -> Tensor:
        input = self.forward_pool(input)
        input = self.forward_classifier(input)
        return input

    def forward_pool(self, x: Tensor) -> Tensor:
        x = self.global_pool(x).flatten(1)
        return x

    def forward_classifier(self, x: Tensor) -> Tensor:
        if self.drop_rate > 0. and self.num_classes:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        if self.num_classes == 1:
            x = x[:, 0]
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
