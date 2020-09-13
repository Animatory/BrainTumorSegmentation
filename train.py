import argparse
import datetime
import shutil
from pathlib import Path

import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

from config_structure import TrainConfigParams
from tasks import create_task


def create_logger(logger_params):
    save_dir = Path(logger_params.save_dir)
    version = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger_params.version = version
    logger = TensorBoardLogger(**logger_params.dict())

    checkpoint_path = save_dir / logger_params.name / version
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    return logger, checkpoint_path


def create_checkpoint_callback(checkpoint_params, checkpoint_path):
    checkpoint_params.filepath = checkpoint_path
    checkpoint = ModelCheckpoint(**checkpoint_params.dict())
    return checkpoint


def create_early_stop_callback(early_stop_params):
    if early_stop_params is None:
        return None
    else:
        return EarlyStopping(**early_stop_params.dict())


def create_profiler(profiler_params, checkpoint_path):
    if profiler_params is None:
        return None
    else:
        if profiler_params.save_profile:
            output_filename = checkpoint_path / 'profile.log'
        else:
            output_filename = None

        if profiler_params.name == 'simple':
            return SimpleProfiler(output_filename)
        elif profiler_params.name == 'advanced':
            return AdvancedProfiler(output_filename)
        else:
            raise ValueError('Given type of profiler is not supported. Use `simple` or `advanced`')


def restore_checkpoint(restore_params, checkpoint_path):
    if restore_params is None:
        return None
    else:
        checkpoint_restore_path = restore_params.checkpoint
        path = Path(checkpoint_restore_path)
        if path.is_file() and path.suffix == '.ckpt':
            for filename in path.parent.glob('**/events*'):
                shutil.copy(filename, checkpoint_path)
        else:
            raise ValueError(f'Invalid path to checkpoint: {path}')
        return checkpoint_restore_path


def create_trainer(train_config):
    logger, checkpoint_path = create_logger(train_config.logger)

    checkpoint_callback = create_checkpoint_callback(train_config.checkpoint, checkpoint_path)
    early_stopper = create_early_stop_callback(train_config.early_stop)
    profiler = create_profiler(train_config.profiler, checkpoint_path)
    checkpoint_restore_path = restore_checkpoint(train_config.restore, checkpoint_path)

    trainer_params = train_config.trainer.dict()

    trainer = Trainer(logger=logger, profiler=profiler,
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=early_stopper,
                      resume_from_checkpoint=checkpoint_restore_path,
                      **trainer_params)
    return trainer


def main(hparams):
    model = create_task(hparams)
    trainer = create_trainer(hparams)
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str,
                        help="Path to .yml file with configuration parameters .")
    config_path = parser.parse_args().hparams

    config_yaml = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    config = TrainConfigParams(**config_yaml)

    main(config)
