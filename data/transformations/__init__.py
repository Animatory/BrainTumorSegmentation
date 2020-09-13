import sys
from types import ModuleType
import albumentations
import albumentations.pytorch
from albumentations import BasicTransform

# after adding extra custom classes import them like this and update local_transformations creation
from . import pixelwise

def _get_transformations(module: ModuleType):
    module_contents = module.__dict__
    return {name: content for name, content in module_contents.items()
            if isinstance(content, type) and issubclass(content, BasicTransform)}


# get all transformations from submodules
# if you add more submodules, get transformations from them also.
local_transformations = _get_transformations(pixelwise)

# maybe albumentations hiding some transforms in other packages, or will add more,
# in that case copypaste the following lines respectively
alb_transformations = _get_transformations(albumentations)
torch_transformations = _get_transformations(albumentations.pytorch)

# local transformations are last because they have higher priority and should override library transformations in case
# of same name.
transformations = {**alb_transformations, **torch_transformations, **local_transformations}
