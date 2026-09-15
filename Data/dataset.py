import h5py
import numpy as np
import torch
from torch.utils import data
from detectron2.data import transforms as T
import cv2
import random

def pad_matrices(matrix, num_max=6):
    current_shape = matrix.shape
    first_dim = current_shape[0]
    padding = num_max - first_dim if first_dim < num_max else 0
    pad_width = ((0, padding),) + ((0, 0),) * (len(current_shape) - 1)
    padded_matrix = np.pad(matrix, pad_width=pad_width, mode='constant', constant_values=0)
    return padded_matrix

class Dataset_tedi(data.Dataset):
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
        reverse=False,
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
        self.reverse = reverse

        self.tfm_gens = []
        if self.hflip:
            self.tfm_gens.append(T.RandomFlip(horizontal=True, vertical=False))
        if self.vflip:
            self.tfm_gens.append(T.RandomFlip(horizontal=False, vertical=True))

        self.tfm_gens = []
        if self.hflip:
            self.tfm_gens.append(T.RandomFlip(horizontal=True, vertical=False))
        if self.vflip:
            self.tfm_gens.append(T.RandomFlip(horizontal=False, vertical=True))

        if self.ColorJitter:
            self.tfm_gens.append(T.RandomLighting(scale=0.1))
        if self.RandomRotation:
            self.tfm_gens.append(T.RandomRotation(angle=[-30, 30]))
        if self.RandomCrop:
            self.tfm_gens.append(T.RandomCrop( crop_type="relative", crop_size=(0.75, 0.75)))

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        img_left_list = []
        mask_list = []
        feat_list = []
        name_list = []

        path = self.input_paths[index] + (f"_{self.version}" if self.version else "") + ".h5"
        path_parts = path.split('.')[0].split('_')
        frame_idx = path_parts[-1]
        cur_idx = int(frame_idx[5:8])

        min_idx, max_idx = 0, 148

        num_read = self.cfg.MODEL.MEMORY_BANK.NUM_MEM_FRAMES+self.cfg.MODEL.MEMORY_BANK.NUM_DENOISING_FRAMES

        left = num_read // 2
        right = num_read - left - 1
        offsets = list(range(-left, right + 1))
        cnt = 0
        transforms = None

        for off in offsets:
            tgt_idx = cur_idx + off
            tgt_idx = min(max(tgt_idx, min_idx), max_idx)
            frame_idx_tgt = "frame" + str(tgt_idx).zfill(3)
            path_tgt = path.replace(frame_idx, frame_idx_tgt)
            with h5py.File(path_tgt, "r") as f:
                if self.bina_read:
                    img_left = f['image_left'][:]
                else:
                    img_left = f['image'][:]

                mask = f['mask'][:].astype('int64')

                name = f['name'][()].decode('utf-8')
            if cnt == 0:
                aug_input = T.AugInput(img_left, sem_seg=mask)
                aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
                img_left = aug_input.image
                mask = aug_input.sem_seg
            else:
                img_left = transforms.apply_image(img_left)
                mask = transforms.apply_segmentation(mask)

            img_left = self.transform_input(img_left.copy())
            mask = self.transform_target(mask.copy())
            mask = mask.squeeze(0)

            img_left_list.append(img_left)
            mask_list.append(mask)

            name_list.append(name)

            cnt += 1

            if self.reverse:
                prob=random.random()
                if prob>0.5:
                    img_left_list = img_left_list[::-1]
                    mask_list = mask_list[::-1]
                    name_list = name_list[::-1]
            img_left = np.stack(img_left_list, axis=0)
            mask = np.stack(mask_list, axis=0)

            name = name_list[-1]

            if not self.bina:

                return img_left, mask, name
