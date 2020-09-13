import numpy as np
import torch
from torch import nn

from config_structure import CriterionParams
from .utils.helpers import load_checkpoint
from .utils.registry import is_model, is_model_in_modules, model_entrypoint, list_models
from . import criterions


def create_backbone(
        model_name,
        pretrained=False,
        num_classes=1000,
        in_chans=3,
        checkpoint_path='',
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        num_classes (int): number of classes for final fully connected layer (default: 1000)
        in_chans (int): number of input channels / colors (default: 3)
        checkpoint_path (str): path of checkpoint to load after model is initialized

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """

    margs = dict(pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    # Only EfficientNet and MobileNetV3 models have support for batchnorm params or drop_connect_rate passed as args
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])
    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    # Parameters that aren't supported by all models should default to None in command line args,
    # remove them if they are present and not set so that non-supporting models don't break.
    if kwargs.get('drop_block_rate', None) is None:
        kwargs.pop('drop_block_rate', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    if kwargs.get('drop_path_rate', None) is None:
        kwargs.pop('drop_path_rate', None)

    if is_model(model_name):
        create_fn = model_entrypoint(model_name)

        pretrained_models = list_models(pretrained=True)
        pretrain = pretrained and (model_name in pretrained_models)
        margs['pretrained'] = pretrain
        if pretrained and not pretrain:
            print("WARNING: the model doesn't have pretrained weights")
        elif pretrain:
            print(f'Pretrained weights for {model_name} will be downloaded')

        model = create_fn(**margs, **kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model


def create_criterion(params: CriterionParams):
    criterion_list = params.criterion_list
    criterion_weights = params.weights
    return JointLoss(criterion_list, criterion_weights)


class JointLoss(nn.Module):
    def __init__(self, criterion_list, criterion_weights=None):
        super(JointLoss, self).__init__()
        classes_dict = criterions.__dict__
        classes_dict = {name: content for name, content in classes_dict.items() if self.is_loss(content)}
        self.criterions = nn.ModuleList()

        for criterion_params in criterion_list:
            criterion_class = classes_dict[criterion_params.name]
            target_fields = criterion_params.params.pop('target_fields', None)
            criterion = criterion_class(**criterion_params.params)
            criterion.target_fields = target_fields
            self.criterions.append(criterion)

        if criterion_weights is None:
            self.weights = np.ones((len(self.criterions,))) / len(self.criterions)
        else:
            self.weights = np.array(criterion_weights)
            self.weights = self.weights / self.weights.sum()
            if len(self.criterions) != len(self.weights):
                raise ValueError('Length of weight must be equal to the number of losses or be None')

    def forward(self, *args,  **kwargs):
        total_loss = 0
        for i, criterion in enumerate(self.criterions):
            if criterion.target_fields is None:
                loss = criterion(*args, **kwargs)
            else:
                targeted_kwargs = self.map_arguments(criterion.target_fields, kwargs)
                if targeted_kwargs:
                    loss = criterion(*args, **targeted_kwargs)
                else:
                    loss = 0
            total_loss += loss * self.weights[i]
        return total_loss

    @staticmethod
    def is_loss(cls):
        return isinstance(cls, type) and cls.__name__.endswith('Loss') and issubclass(cls, nn.Module)

    @staticmethod
    def map_arguments(target_fields, kwargs):
        targeted_kwargs = {}
        for target_arg, source_arg in target_fields.items():
            if source_arg in kwargs:
                targeted_kwargs[target_arg] = kwargs[source_arg]
        return targeted_kwargs
