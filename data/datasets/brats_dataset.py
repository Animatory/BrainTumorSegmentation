from pathlib import Path
import torch

from medpy.io import load
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
from registry import DATASETS


@DATASETS.register_class
class BRATSDataset(BaseDataset):

    def __init__(self, data_folder, path_to_datalist,
                 augment=None, transform=None,
                 input_dtype='float32', label_dtype='long'):
        self.data_folder = Path(data_folder)
        self.csv = pd.read_csv(self.data_folder / path_to_datalist)
        self.csv['path'] = self.csv['path'].apply(lambda x: self.data_folder / x)
        self.input_dtype = input_dtype
        self.label_dtype = label_dtype

        self.augment = augment
        self.transform = transform

    def get_raw(self, i):
        record = self.csv.iloc[i]
        opened, _ = load(str(record.path))
        mask, _ = load(str(record.label_path))
        index = int(record.ind)

        if int(record.v) == 0:
            image = opened[index, :, :]
            mask = mask[index, :, :]
        elif int(record.v) == 1:
            image = opened[:, index, :]
            mask = mask[:, index, :]
        else:
            image = opened[:, :, index]
            mask = mask[:, :, index]

        if image.ndim == 2:
            image = image[..., None]

        # apply augmentations
        if self.augment:
            sample = self.augment(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __getitem__(self, i):
        image, mask = self.get_raw(i)

        # apply preprocessing
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        image = image.type(torch.__dict__[self.input_dtype])
        mask = mask.type(torch.__dict__[self.label_dtype])

        return {"input": image, "target_mask": mask}

    def __len__(self):
        return len(self.csv)
