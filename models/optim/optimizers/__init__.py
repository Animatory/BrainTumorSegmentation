from torch import optim

# If added new custom optimizer then add it here too
from models.optim.optimizers import novograd


# Dictionary with optimizers Example:{'Name_of_optim' : <class 'Name_of_optim'>}
def get_optimizers(module):
    optimizers = {k: v for k, v in module.__dict__.items() if
                  isinstance(v, type) and issubclass(v, optim.Optimizer)}
    return optimizers


local_optimizers = get_optimizers(novograd)
torch_optimizers = get_optimizers(optim)

# union of two dictionaries, with priority to local optimizers
optimizers = {**torch_optimizers, **local_optimizers}
