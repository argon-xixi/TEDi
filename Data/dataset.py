import random
from skimage.io import imread
import os
# 一直误以为是用的cv2.imread
import h5py
import torch
from torch.utils import data
import torchvision.transforms.functional as TF
import h5py
import cv2
import numpy as np
from detectron2.data import transforms as T
def pad_matrices(matrix,num_max=6):
    
    
    current_shape = matrix.shape
    first_dim = current_shape[0]
    # 计算需要填充的数量
    padding = num_max - first_dim if first_dim < num_max else 0
    pad_width = ((0, padding),) + ((0, 0),) * (len(current_shape) - 1)
    padded_matrix = np.pad(matrix, pad_width=pad_width, mode='constant', constant_values=0)
    
    return padded_matrix
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
        affine=False,
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
        self.affine = affine
        self.bina=bina
        self.bina_read=bina_read
        self.bbox=bbox
        #数据增强
        self.tfm_gens = []  # 存储数据增强方法
        if self.hflip:
            self.tfm_gens.append(T.RandomFlip(horizontal=True, vertical=False))
        if self.vflip:
            self.tfm_gens.append(T.RandomFlip(horizontal=False, vertical=True))

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
            if self.version == 0:
                path=self.input_paths[index]+".h5"
            else:
                path=self.input_paths[index]+"_"+str(self.version)+".h5"
            f = h5py.File(path, "r")
            '''  # print(f.keys())
            # print(f.keys())
            if self.bina_read:
                img_left=f['image_left'][:]
            else:
                img_left=f['image'][:]
            
            # print(img.max())
            mask=f['mask'][:]
            
            feat=f['feat'][:]
            feat=feat.transpose((2,0,1))
            # bbox=f['bbox'][:]

            name=f['name'][()].decode('utf-8') # 读取标量要用[()]
            # 此处x应为unit8，y应为float'''
            if self.bina_read:
                img_left=f['image_left'][:]
            else:
                img_left=f['image'][:]
            
            # 读取mask和feat数据
            mask = f['mask'][:].astype('int64')
            feat = f['feat'][:].transpose((2, 0, 1))
            name = f['name'][()].decode('utf-8')
            f.close()
            h,w,c=img_left.shape
            
            # 创建AugInput实例，对当前帧图像和mask做一致的变换
            aug_input = T.AugInput(img_left, sem_seg=mask)  # 使用 sem_seg 作为目标
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            img_left = aug_input.image  # 变换后的图像
            mask = aug_input.sem_seg  # 变换后的语义分割标签
            mask = mask.astype('int64')
            
            img_left = self.transform_input(img_left.copy())
            mask = self.transform_target(mask.copy())
            mask= mask.squeeze(0)

            if not self.bina:
            
            # return img.float(), mask.float(),feat.float(),name
                return img_left, mask,feat,name

            else:
                img_right=f['image_right'][:]
                try:
                    flow_l2r=f['flow_l2r'][:]
                except:
                    print(name)
                    print(f.keys())
                img_right = self.transform_input(img_right)
                return (img_left,img_right,flow_l2r), mask,feat,name
