from torch.optim import lr_scheduler

from config_structure import StructureParams


def create_scheduler(optimizer, param: StructureParams):
    if param.name not in lr_scheduler.__dict__:
        raise ValueError(f'Scheduler with name `{param.name}` not found')

    scheduls = get_schedulers(lr_scheduler)

    return scheduls[param.name](optimizer, **param.params)


def get_schedulers(module):
    schedulers = {k: v for k, v in module.__dict__.items() if
                  isinstance(v, type) and issubclass(v, lr_scheduler._LRScheduler)}
    return schedulers
