from .registry import Registry

DATASETS = Registry('datasets')
METRICS = Registry('metric')
TASKS = Registry('tasks')
HEADS = Registry('heads')
SEGMENTATION_HEADS = Registry('segmentation_heads')
SEGMENTATION_MODELS = Registry('segmentation_models')
DETECTION_HEADS = Registry('detection_heads')
