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
    def __init__(self, path, mode='train', data_type='Tr', image_size=128,
                 transform=None, threshold=500,
                 split_num=1, split_idx=0, pcc=False):
        self.path = path
        self.data_type = data_type
        self.split_num = split_num
        self.split_idx = split_idx

        self._set_file_paths(self.path)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        self.pcc = pcc

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # 加载npz文件
        data = np.load(self.image_paths[index])
        # 获取图像数据 (D, H, W)
        image = data['imgs']

        # 转换为torchio的Subject格式
        # 使用tensor=image[None]添加channel维度，将(D,H,W)转换为(1,D,H,W)
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image[None])
        )

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(f"Transform failed for {self.image_paths[index]}")

        if self.mode == "train" and self.data_type == 'Tr':
            return subject.image.data.clone().detach()
        else:
            return subject.image.data.clone().detach(), self.image_paths[index]

    def _set_file_paths(self, path):
        print(f"Given path: {path}")
        self.image_paths = []
        d = os.path.join(path, f'images{self.data_type}')
        if os.path.exists(d):
            for name in os.listdir(d):
                if name.endswith('.npz'):
                    self.image_paths.append(os.path.join(path, f'images{self.data_type}', name))
        print(f"Found {len(self.image_paths)} image(s)")


class Dataset_Union_ALL_Val(Dataset_Union_ALL):
    def _set_file_paths(self, path):
        self.image_paths = []

        for name in os.listdir(path):
            if name.endswith('.npz'):
                self.image_paths.append(os.path.join(path,  name))
        self.image_paths = self.image_paths[self.split_idx::self.split_num]


