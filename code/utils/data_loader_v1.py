from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
import torch
import numpy as np
import os
import torch
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator
import nibabel as nib


class Dataset_Union_ALL(Dataset):
    def __init__(self, paths, mode='train', data_type='Tr', image_size=128,
                 transform=None, threshold=500,
                 split_num=1, split_idx=0, pcc=False):
        self.paths = paths
        self.data_type = data_type
        self.split_num = split_num
        self.split_idx = split_idx

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        self.pcc = pcc

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        sitk_image = sitk.ReadImage(self.image_paths[index])
        subject = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
        )
        image_shape = subject.image.data.shape

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])
        if self.mode == "train" and self.data_type == 'Tr':
            return subject.image.data.clone().detach()
        else:
            return subject.image.data.clone().detach(), self.image_paths[index]

    def _set_file_paths(self, paths):
        print(f"Given paths: {paths}")
        self.image_paths = []
        for path in paths:
            d = os.path.join(path, f'images{self.data_type}')
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split('.nii.gz')[0]

                    self.image_paths.append(os.path.join(path, f'images{self.data_type}', f'{base}.nii.gz'))
                    print(f"Found {len(self.image_paths)} image(s) ")


class Dataset_Union_ALL_Val(Dataset_Union_ALL):
    def _set_file_paths(self, paths):
        self.image_paths = []

        for path in paths:
            for dt in ["Tr", "Val", "Ts"]:
                d = os.path.join(path, f'images{dt}')
                if os.path.exists(d):
                    for name in os.listdir(d):
                        base = os.path.basename(name).split('.nii.gz')[0]
                        self.image_paths.append(os.path.join(path, f'images{dt}', f'{base}.nii.gz'))
        self.image_paths = self.image_paths[self.split_idx::self.split_num]


