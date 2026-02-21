import random
import h5py
import torch
from torch.utils import data
from detectron2.data import transforms as T
import torchvision.transforms as t_vision
import numpy as np
import cv2


class SegDataset(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        version: int,
        transform_input=None,
        transform_target=None,
        cfg=None,
        hflip=False,
        vflip=False,
        ColorJitter=False,
        RandomRotation=False,
        RandomCrop=False,
        bina=False,
        bbox=True,
        bina_read=False,
    ):
        self.input_paths = input_paths
        self.version = version
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.cfg = cfg
        self.hflip = hflip
        self.vflip = vflip
        self.ColorJitter = ColorJitter
        self.RandomRotation = RandomRotation
        self.RandomCrop = RandomCrop
        self.bina = bina
        self.bina_read = bina_read
        self.bbox = bbox

        # 数据增强方法
        self.tfm_gens = []  # 存储数据增强方法
        if self.hflip:
            self.tfm_gens.append(T.RandomFlip(horizontal=True, vertical=False))
        if self.vflip:
            self.tfm_gens.append(T.RandomFlip(horizontal=False, vertical=True))
        # 添加额外的强增强操作
        if self.ColorJitter:
            self.tfm_gens.append(T.RandomLighting(scale=0.1))  # 亮度变化
        if self.RandomRotation:
            self.tfm_gens.append(T.RandomRotation(angle=[-30, 30]))  # 随机旋转
        if self.RandomCrop:
            self.tfm_gens.append(T.RandomCrop( crop_type="relative", crop_size=(0.75, 0.75)))  # 随机裁剪

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        if self.version == 0:
            path = self.input_paths[index] + ".h5"
        else:
            path = self.input_paths[index] + "_" + str(self.version) + ".h5"
        
        # 读取 h5 数据
        f = h5py.File(path, "r")
        if self.bina_read:
            img_left = f['image_left'][:]
        else:
            img_left = f['image'][:]
        
        mask = f['mask'][:].astype('int64')
        # feat = f['feat'][:].transpose((2, 0, 1))
        name = f['name'][()].decode('utf-8')
        f.close()

        # 数据增强
        aug_input = T.AugInput(img_left, sem_seg=mask)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        img_left = aug_input.image
        mask = aug_input.sem_seg
        mask = mask.astype('int64')

        img_left = self.transform_input(img_left.copy())
        mask = self.transform_target(mask.copy())
        mask = mask.squeeze(0)

        if not self.bina:
            return img_left, mask,  name
        else:
            img_right = f['image_right'][:]
            try:
                flow_l2r = f['flow_l2r'][:]
            except:
                print(name)
                print(f.keys())
            img_right = self.transform_input(img_right)
            return (img_left, img_right, flow_l2r), mask,  name
