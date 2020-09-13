import albumentations.augmentations.functional as F
from albumentations import ImageOnlyTransform


class InstanceNormalize(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(InstanceNormalize, self).__init__(always_apply, p)

    def apply(self, image, **params):
        mean = image.mean(axis=(0, 1), keepdims=True)
        std = image.mean(axis=(0, 1), keepdims=True)
        std[std == 0] = 1
        return F.normalize(image, mean, std, 1)

    def get_transform_init_args_names(self):
        return ()

