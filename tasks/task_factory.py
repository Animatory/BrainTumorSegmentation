from functools import partial

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from data import create_dataset
from models.optim import create_scheduler, create_optimizer
from registry import TASKS


def create_task(hparams):
    return TASKS.get(hparams.task.name)(hparams)


class MetaTask(LightningModule):
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        metric_logs = self.metric_manager.on_epoch_end('train')
        metric_logs['train/loss'] = avg_loss
        metric_logs['step'] = self.current_epoch
        return {'log': metric_logs}

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        metric_logs = self.metric_manager.on_epoch_end('valid')
        metric_logs['valid/loss'] = avg_loss
        metric_logs['step'] = self.current_epoch
        return {'log': metric_logs}

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test/loss'] for x in outputs]).mean()
        return {'test/loss': avg_loss}

    def configure_optimizers(self):
        optimizer = create_optimizer(self, self.hparams.optimizer)
        if self.hparams.scheduler is not None:
            scheduler = create_scheduler(optimizer, self.hparams.scheduler)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def prepare_data(self):
        # update train_params with common params

        data_params = self.hparams.data
        dataset_name = data_params.dataset_name
        common_params = data_params.common_params

        # update train_params with common params
        train_params = data_params.train_params
        train_params.params.update(common_params)

        # update valid_params with common params
        valid_params = data_params.valid_params
        valid_params.params.update(common_params)

        # create train and validation datasets
        self.dataset_train = create_dataset(dataset_name, train_params)
        self.dataset_valid = create_dataset(dataset_name, valid_params)

        # create test dataset
        if data_params.test_params is not None:
            # update test_params with common params
            test_params = data_params.test_params
            test_params.params.update(common_params)

            self.dataset_test = create_dataset(dataset_name, test_params)
        else:
            self.dataset_test = None

    def train_dataloader(self):
        dl_params = self.hparams.data.dataloader_params
        preloader = partial(DataLoader,
                            dataset=self.dataset_train,
                            batch_size=dl_params.batch_size,
                            drop_last=dl_params.drop_last,
                            shuffle=dl_params.shuffle,
                            num_workers=dl_params.num_workers)
        if dl_params.use_custom_collate_fn:
            loader = preloader(collate_fn=self.dataset_train.collate_fn)
        else:
            loader = preloader()
        return loader

    def val_dataloader(self):
        dl_params = self.hparams.data.dataloader_params
        preloader = partial(DataLoader,
                            dataset=self.dataset_valid,
                            batch_size=dl_params.batch_size,
                            drop_last=False,
                            shuffle=False,
                            num_workers=dl_params.num_workers)
        if dl_params.use_custom_collate_fn:
            loader = preloader(collate_fn=self.dataset_valid.collate_fn)
        else:
            loader = preloader()
        return loader

    def test_dataloader(self):
        if self.dataset_test:
            dl_params = self.hparams.data.dataloader_params
            preloader = partial(DataLoader,
                                dataset=self.dataset_test,
                                batch_size=dl_params.batch_size,
                                drop_last=False,
                                shuffle=False,
                                num_workers=dl_params.num_workers)
            if dl_params.use_custom_collate_fn:
                loader = preloader(collate_fn=self.dataset_test.collate_fn)
            else:
                loader = preloader()
            return loader
