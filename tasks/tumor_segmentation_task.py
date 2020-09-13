from copy import deepcopy

import torch
import torch.nn as nn

from config_structure import TrainConfigParams
from metrics import MetricManager
from models import create_backbone, create_criterion
from registry import TASKS, HEADS, SEGMENTATION_HEADS
from tasks.task_factory import MetaTask
from models.optim import create_scheduler, create_optimizer


@TASKS.register_class
class TumorSegmentationTask(MetaTask):

    def __init__(self, hparams: TrainConfigParams):
        super().__init__()
        self.hparams = hparams
        task_params = hparams.task.params
        self.backbone_name = task_params['backbone_name']
        self.backbone_params = task_params.get('backbone_params', {})
        self.backbone = create_backbone(model_name=self.backbone_name, **self.backbone_params)
        self.use_ema = task_params.get('use_ema', False)

        heads = task_params['heads']
        self.heads = nn.ModuleDict()
        self.head_tasks = {}
        for head in heads:
            head_name = head['name']
            head_type = head['type']
            head_params = head['params']
            if head_type == 'ClassificationHead':
                head_params['num_features'] = self.backbone.num_features
                head_task = 'classification'
            elif head_type in SEGMENTATION_HEADS:
                head_params['encoder_channels'] = self.backbone.encoder_channels
                head_task = 'segmentation'
            else:
                head_task = 'classification'

            self.heads[head_name] = HEADS.get(head_type)(**head_params)
            self.head_tasks[head_name] = head_task

        if 'checkpoint' in task_params and task_params.get('load_first', False):
            state_dict = torch.load(task_params['checkpoint'])['state_dict']
            print(self.load_state_dict(state_dict, strict=False))

        # creating metric manager
        self.metric_manager = MetricManager(hparams.metrics)
        self.criterion = create_criterion(self.hparams.criterion)

    def forward(self, x):
        last_features, backbone_features = self.backbone.forward_backbone_features(x)
        last_features = self.backbone.forward_neck(last_features)
        mask_pred = self.heads['mask'](backbone_features)
        label_pred = self.heads['label'](last_features)
        return mask_pred, label_pred

    def forward_with_gt(self, batch):
        output = {}
        input_data = batch['input']
        last_features, backbone_features = self.backbone.forward_backbone_features(input_data)
        label_out = self.heads['label'](self.backbone.forward_neck(last_features))
        mask_out = self.heads['mask'](backbone_features)

        output[f'prediction_mask'] = mask_out
        output[f'target_mask'] = batch['target_mask']

        output[f'prediction_label'] = label_out
        output[f'target_label'] = batch['target_label']

        return output

    def training_step(self, batch, batch_idx):
        output = self.forward_with_gt(batch)
        loss = self.criterion(**output)
        self.metric_manager.update('train', **output)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        output = self.forward_with_gt(batch)
        val_loss = self.criterion(**output)
        self.metric_manager.update('valid', **output)
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        output = self.forward_with_gt(batch)
        val_loss = self.criterion(**output)
        return {'test_loss': val_loss}

    def configure_optimizers(self):
        if self.use_ema:
            decay = self.hparams.optimizer.params.pop('alpha', 0.995)
            optimizer = create_optimizer(self, self.hparams.optimizer)
            ema_optimizer = EMAWeightOptimizer(optimizer, self, decay=decay)

            if self.hparams.scheduler is not None:
                scheduler = create_scheduler(optimizer, self.hparams.scheduler)
                return [ema_optimizer], [scheduler]
            else:
                return [ema_optimizer]
        else:
            return super().configure_optimizers()


class EMAWeightOptimizer:
    """
    Exponential moving average weight optimizer
    """

    def __init__(self, optimizer, model, decay=0.99):
        self.optimizer = optimizer
        self.model = model
        self.ema = ExponentialMovingAverage(model.parameters(), decay=decay)
        self.alpha = decay

    def step(self):
        self.optimizer.step()
        self.ema.update(self.model.parameters())

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def zero_grad(self):
        self.optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        self.ema.shadow_params = state_dict['ema_parameters']
        self.optimizer.load_state_dict(state_dict['optimizer_parameters'])

    def state_dict(self):
        result = {
            'ema_parameters': self.ema.shadow_params,
            'optimizer_parameters': self.optimizer.state_dict()
        }
        return result


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """
    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copies current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)
