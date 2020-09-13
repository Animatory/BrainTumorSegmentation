from typing import Union, List

from torch.nn import Module
from config_structure import StructureParams
from models.optim import optimizers


def create_optimizer(model: Union[Module, List[Module]], optim_param: StructureParams):
    if optim_param.name not in optimizers.optimizers:
        raise ValueError(f'Optimizer with name `{optim_param.name}` not found')

    optims = optimizers.optimizers

    # if weight_decay are not exist, set it to default 5e-5 (0.00005)
    if "weight_decay" not in optim_param.params:
        optim_param.params["weight_decay"] = 5e-5

    # if weight_decay  are equal to 0, we don't need to make two groups of parameters
    if optim_param.params["weight_decay"] != 0:
        if isinstance(model, list):
            parameters = []
            for module in model:
                parameters += add_weight_decay(module, optim_param.params["weight_decay"])
        else:
            parameters = add_weight_decay(model, optim_param.params["weight_decay"])
    else:
        parameters = model.parameters()

    return optims[optim_param.name](parameters, **optim_param.params)


def add_weight_decay(model: Module, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]
