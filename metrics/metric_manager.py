"""Provides class MetricManager, which stores and manages all metrics for
the model."""
from typing import List

import torch

from config_structure import StructureParams
from registry import METRICS


class MetricManager:
    epochs = ['train', 'valid', 'test']
    """Manages all metrics for the model,
    stores their values at checkpoints"""

    def __init__(self, params: List[StructureParams]):
        self.metrics = {epoch: {} for epoch in self.epochs}
        for epoch in self.epochs:
            for metric in params:
                metric_obj = METRICS.get(metric.name)(**metric.params)
                self.metrics[epoch][metric_obj.name] = metric_obj

    def update(self, epoch, *args, **kwargs):
        """Update states of all metrics on training/validation loop"""
        if epoch not in self.epochs:
            raise ValueError(f'Incorrect epoch setting. '
                             f'Please choose one of {self.epochs}')
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = arg.cpu().detach().numpy()
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.cpu().detach().numpy()

        for name, metric in self.metrics[epoch].items():
            if metric.target_fields is None:
                metric.update(*args, **kwargs)
            else:
                targeted_kwargs = self.map_arguments(metric.target_fields, kwargs)
                if targeted_kwargs:
                    metric.update(*args, **targeted_kwargs)

    def on_epoch_end(self, epoch):
        """Summarize epoch values and return log"""
        if epoch not in self.epochs:
            raise ValueError(f'Incorrect epoch setting. '
                             f'Please choose one of {self.epochs}')
        log = {f'{epoch}/{name}': torch.tensor(metric.on_epoch_end())
               for name, metric in self.metrics[epoch].items()}
        return log

    @staticmethod
    def map_arguments(target_fields, kwargs):
        targeted_kwargs = {}
        for target_arg, source_arg in target_fields.items():
            if source_arg in kwargs:
                targeted_kwargs[target_arg] = kwargs[source_arg]
        return targeted_kwargs
