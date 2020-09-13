from __future__ import annotations

from typing import List, Dict, Optional, Union, Any
from argparse import Namespace

from pydantic import BaseModel


class StructureParams(BaseModel):
    name: str
    params: dict = {}


class DatasetParams(BaseModel):
    params: dict = {}
    transform: List[StructureParams]
    augment: List[StructureParams] = None


class DataLoaderParams(BaseModel):
    batch_size: int
    num_workers: int = 0
    shuffle: bool = True
    drop_last: bool = False
    use_custom_collate_fn: bool = False


class DataParams(BaseModel):
    dataset_name: str
    common_params: dict
    dataloader_params: DataLoaderParams
    train_params: DatasetParams
    valid_params: DatasetParams
    test_params: Optional[DatasetParams]


class TrainerParams(BaseModel):
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    process_position: int = 0
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Union[List[int], str, int, None] = None
    auto_select_gpus: bool = False
    tpu_cores: Union[List[int], int, None] = None
    log_gpu_memory: Optional[str] = None
    progress_bar_refresh_rate: int = 1
    overfit_pct: float = 0.0
    track_grad_norm: Union[int, float, str] = -1
    check_val_every_n_epoch: int = 1
    fast_dev_run: bool = False
    accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    train_percent_check: float = 1.0
    val_percent_check: float = 1.0
    test_percent_check: float = 1.0
    val_check_interval: float = 1.0
    log_save_interval: int = 1000
    row_log_interval: int = 500
    distributed_backend: Optional[str] = None
    precision: int = 32
    weights_summary: Optional[str] = 'top'
    weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    truncated_bptt_steps: Optional[int] = None
    benchmark: bool = False
    deterministic: bool = False
    reload_dataloaders_every_epoch: bool = False
    auto_lr_find: Union[bool, str] = False
    replace_sampler_ddp: bool = True
    terminate_on_nan: bool = False
    auto_scale_batch_size: Union[str, bool] = False


class LoggerParams(BaseModel):
    save_dir: str
    name: Optional[str] = "default"
    version: Union[int, str, None] = None


class EarlyStopParams(BaseModel):
    monitor: str
    mode: str
    min_delta: float
    patience: int
    verbose: bool = False
    strict: bool = True


class CheckpointParams(BaseModel):
    filepath: Optional[str]
    monitor: str = 'val_loss'
    verbose: bool = False
    save_last: bool = False
    save_top_k: int = 1
    save_weights_only: bool = False
    mode: str = 'auto'
    period: int = 1


class RestoreParams(BaseModel):
    version: str = None
    checkpoint: str = None
    weights_only: bool = False


class ProfilerParams(BaseModel):
    name: str = 'simple'
    save_profile: bool = False


class CriterionParams(BaseModel):
    criterion_list: List[StructureParams]
    weights: List[float] = None


class TrainConfigParams(BaseModel, Namespace):
    task: StructureParams
    criterion: CriterionParams
    optimizer: StructureParams
    scheduler: Optional[StructureParams]
    data: DataParams
    metrics: List[StructureParams]
    trainer: TrainerParams
    logger: LoggerParams
    checkpoint: CheckpointParams
    early_stop: Optional[EarlyStopParams]
    profiler: Optional[ProfilerParams]
    restore: Optional[RestoreParams]
