import h5py
import numpy as np
import torch
from torch.utils import data
from detectron2.data import transforms as T
import cv2
import random
# 用于填充矩阵的函数
def pad_matrices(matrix, num_max=6):
    current_shape = matrix.shape
    first_dim = current_shape[0]
    padding = num_max - first_dim if first_dim < num_max else 0
    pad_width = ((0, padding),) + ((0, 0),) * (len(current_shape) - 1)
    padded_matrix = np.pad(matrix, pad_width=pad_width, mode='constant', constant_values=0)
    return padded_matrix

class SegDataset_track(data.Dataset):
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

        # 初始化增强操作
        self.tfm_gens = []  # 存储数据增强方法
        if self.hflip:
            self.tfm_gens.append(T.RandomFlip(horizontal=True, vertical=False))
        if self.vflip:
            self.tfm_gens.append(T.RandomFlip(horizontal=False, vertical=True))
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
        if self.cfg.bbox:
            if self.version == 0:
                path = f"{self.input_paths[index]}.h5"
            else:
                path = f"{self.input_paths[index]}_{self.version}.h5"
            
            with h5py.File(path, "r") as f:  # 使用上下文管理器自动关闭文件
                # 读取关键数据
                # print(f.keys())
                img_key = 'image_left' if self.bina_read else 'image'
                img_left = f[img_key][:]
                img_right = f['image_right'][:]
                
                mask = f['mask'][:].astype(np.int64)  # 合并类型转换
                #     a=mask.transpose((1, 2, 0))
                # except:
                    
                #     print(mask.shape)
                #     print(a)
                # print(a.shape)
                feat = np.ascontiguousarray(f['feat'][:].transpose(2, 0, 1))  # 优化转置
                
                # 计算缩放系数
                H, W = img_left.shape[:2]
                w_scale, h_scale = 1/ W, 1 / H
                scale = np.array([[w_scale, h_scale, w_scale, h_scale]], dtype=np.float32)
                bbox=f['bbox'][:]
                cls = f['cls'][:]
                # 统一维度变换逻辑
                if self.bbox:
                    if mask.sum() == 0:
                        mask=np.zeros((1,1024,1280),dtype=np.int64)
                        bbox=np.zeros((1,4),dtype=np.float32)
                        cls=np.zeros((1),dtype=np.int64)
                        # cls=pad_matrices(cls)
                        # bbox=pad_matrices(bbox)
                    mask = self.transform_target(mask.transpose((1, 2, 0)))  # 提前转置
                # try:
                    # print(f['bbox'][:])
                    bbox = bbox * scale  # 直接应用缩放
                    # print(bbox)
                    mask=pad_matrices(mask)
                    bbox=pad_matrices(bbox)
                    cls=pad_matrices(cls)
                    return_data = (mask, bbox, cls)
                else:
                    mask = self.transform_target(mask.squeeze(0))
                    return_data = mask
                
                # 应用图像变换
                img_left = self.transform_input(img_left)
                img_right = self.transform_input(img_right)
                flow_l2r=f['flow_l2r'][:]
                img=(img_left,img_right,flow_l2r)
                name=f['name'][()].decode('utf-8')
            del f
            return (img, return_data, feat, name)

        else:
            img_left_list = []
            mask_list = []
            feat_list = []
            name_list = []

            # 获取文件路径
            path = self.input_paths[index] + (f"_{self.version}" if self.version else "") + ".h5"
            path_parts = path.split('.')[0].split('_')
            frame_idx = path_parts[-1]  # 例如 "frame012"
            cur_idx = int(frame_idx[5:8])

            min_idx, max_idx = 0, 148

            # 读 6 张：[-3, -2, -1, 0, +1, +2]
            # 如果你更想“前2后3”，改成 offsets = [-2, -1, 0, 1, 2, 3]
            num_read = self.cfg.MODEL.MEMORY_BANK.NUM_MEM_FRAMES+self.cfg.MODEL.MEMORY_BANK.NUM_TRACK_FRAMES
            # num_read = 3
            left = num_read // 2              # 3
            right = num_read - left - 1       # 2
            offsets = list(range(-left, right + 1))  # [-3, -2, -1, 0, 1, 2]

            cnt = 0
            transforms = None

            for off in offsets:
                tgt_idx = cur_idx + off
                # clamp：越界就用 0 或 148
                tgt_idx = min(max(tgt_idx, min_idx), max_idx)

                frame_idx_tgt = "frame" + str(tgt_idx).zfill(3)
                path_tgt = path.replace(frame_idx, frame_idx_tgt)

                with h5py.File(path_tgt, "r") as f:
                    # 读取图像
                    if self.bina_read:
                        img_left = f['image_left'][:]
                    else:
                        img_left = f['image'][:]

                    # 读取mask和feat数据
                    mask = f['mask'][:].astype('int64')
                    # feat = f['feat'][:].transpose((2, 0, 1))
                    name = f['name'][()].decode('utf-8')

                if cnt == 0:
                    # 第一张生成 transforms，后面全部复用，保证6张做同样增强
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
                # feat_list.append(feat)
                name_list.append(name)

                cnt += 1
                
            if self.reverse:
                prob=random.random()
                if prob>0.5:
                    img_left_list = img_left_list[::-1]
                    mask_list = mask_list[::-1]
                    # feat_list = feat_list[::-1]
                    name_list = name_list[::-1]
                
        
            # 将变换后的图像、mask 和 feat 堆叠成一个批次
            img_left = np.stack(img_left_list, axis=0)
            mask = np.stack(mask_list, axis=0)
            # feat = np.stack(feat_list, axis=0)
            name = name_list[-1]  # 使用最后一帧的名字（前后帧都使用相同名字）

            # 如果不需要二进制读取，则返回图像、mask 和特征
            if not self.bina:
                # return img_left, mask, feat, name
                return img_left, mask, name

        