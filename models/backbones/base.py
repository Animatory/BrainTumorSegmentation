import torch.nn as nn


class BackboneBase(nn.Module):
    def __init__(self):
        super(BackboneBase, self).__init__()

    def forward_features(self, x):
        raise NotImplementedError

    def forward_neck(self, x):
        raise NotImplementedError

    def forward(self, x):
        y = self.forward_features(x)
        y = self.forward_neck(y)
        return y

    def forward_backbone_features(self, x):
        last_features = self.forward_features(x)
        backbone_features = self.feature_hooks.get_output(x.device)
        backbone_features = [x] + backbone_features
        return last_features, backbone_features

    def forward_stage_features(self, x):
        x = self.forward_features(x)
        return self.feature_hooks.get_output(x.device)

    @staticmethod
    def init_weights(module):
        # #------- init weights --------
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
