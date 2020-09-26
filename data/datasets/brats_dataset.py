from pathlib import Path
import torch

from medpy.io import load
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
from registry import DATASETS


@DATASETS.register_class
class BRATSDataset(BaseDataset):

    def __init__(self, data_folder, path_to_datalist,
                 augment=None, transform=None, input_dtype='float32'):
        self.data_folder = Path(data_folder)
        self.csv = pd.read_csv(self.data_folder / path_to_datalist)
        self.csv['path'] = self.csv['path'].apply(lambda x: self.data_folder / x)
        self.input_dtype = input_dtype

        self.augment = augment
        self.transform = transform

    def get_raw(self, i):
        opened, _ = load(self.csv.iloc[i].path)
        mask, _ = load(self.csv.iloc[i].label_path)
        
        if self.csv.iloc[i].view == 0:
            image = opened[self.csv.iloc[i].index,:,:]
            mask = mask[self.csv.iloc[i].index,:,:]
        elif self.csv.iloc[i].view == 1:
            image = opened[:,self.csv.iloc[i].index,:]
            mask = mask[:,self.csv.iloc[i].index,:]
        else:
            image = opened[:,:,self.csv.iloc[i].index]
            mask = mask[:,:,self.csv.iloc[i].index]
        
        labels = list(set(mask.flatten().tolist())) 

        if image.ndim == 2:
            image = image[..., None]
        # if image.dtype == 'int16':
        #     image = (image / (2 ** 8)).astype('uint8')
        if image.dtype == 'uint8':
            image = (image.astype(float) * (2 ** 8)).astype('int16')
        mask = mask

        # apply augmentations
        if self.augment:
            sample = self.augment(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, label

    def __getitem__(self, i):
        image, mask, label = self.get_raw(i)

        # apply preprocessing
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        image = image.type(torch.__dict__[self.input_dtype])
        mask = mask.type(torch.__dict__[self.input_dtype])

        return {"input": image, "target_mask": mask.float(), "target_label": label}

    def __len__(self):
        return len(self.csv)