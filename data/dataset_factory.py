import albumentations

from config_structure import DatasetParams
from data.transformations import transformations
from registry import DATASETS


def create_dataset(dataset_name: str, params: DatasetParams):
    transform = create_transforms(params.transform)
    if params.augment:
        augment = create_transforms(params.augment)
    else:
        augment = None

    dataset_class = DATASETS.get(dataset_name)
    return dataset_class(transform=transform, augment=augment, **params.params)


def _prepare_transforms_recursive(transforms):
    transforms_list = []

    for transform_info in transforms:
        if isinstance(transform_info, dict):
            transform_name = transform_info['name']
            transform_params = transform_info.get('params', {})
        else:
            transform_name = transform_info.name
            transform_params = transform_info.params

        if transform_name == 'Compose':
            transform = prepare_compose(**transform_params)
        elif transform_name == 'OneOf':
            transform = prepare_oneof(**transform_params)
        else:
            transform = transformations[transform_name](**transform_params)

        transforms_list.append(transform)

    return transforms_list


def prepare_compose(transforms, p=1.0):
    transforms_list = _prepare_transforms_recursive(transforms)
    transform = albumentations.Compose(transforms_list, p=p)
    return transform


def prepare_oneof(transforms, p=0.5):
    transforms_list = _prepare_transforms_recursive(transforms)
    transform = albumentations.OneOf(transforms_list, p=p)
    return transform


def create_transforms(transforms_params):
    if transforms_params is None:
        return None
    return prepare_compose(transforms_params)
