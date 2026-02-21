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
def pad_matrices(matrix,num_max=6):
    
    
    current_shape = matrix.shape
    first_dim = current_shape[0]
    # 计算需要填充的数量
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
            image_left_list=[]
            mask_list=[]
            feat_list=[]
            name_list=[]
            path = self.input_paths[index] + (f"_{self.version}" if self.version else "") + ".h5"
            path_parts = path.split('.')[0].split('_')
            frame_idx = path_parts[5]
           
            # 待修改
            for i in [1,0,-1]:
                if int(frame_idx[5:8]) not in [0,1,148]:
                    frame_idx_previous = "frame" + str(int(frame_idx[5:8]) - i).zfill(3)
                    path_previous = path.replace(frame_idx, frame_idx_previous)
                else:
                    path_previous = path
                with h5py.File(path_previous, "r") as f:
                # 判断是否是二进制读取
                    if self.bina_read:
                        img_left = f['image_left'][:]
                    else:
                        img_left = f['image'][:]
                    # 读取 mask 和 feat 数据
                    mask = f['mask'][:].astype('int64')
                    feat = f['feat'][:].transpose((2, 0, 1))
                    # 读取 name 数据
                    name = f['name'][()].decode('utf-8')
                f.close()
                # 数据转换操作
                img_left = self.transform_input(img_left)
                mask = self.transform_target(mask)
                # 压缩维度
                mask = mask.squeeze(0)
                image_left_list.append(img_left)
                mask_list.append(mask)
                feat_list.append(feat)
                name_list.append(name)
                
            
            img_left = np.stack(image_left_list, axis=0)
            mask = np.stack(mask_list, axis=0)
            feat = np.stack(feat_list, axis=0)
            name = name_list[1]
            
            if not self.bina:
                return img_left, mask, feat, name
        
                
            # frame_idx_previous = "frame" + str(int(path_parts[5][5:8]) - 1).zfill(3)
            # if int(path_parts[5][5:8]) != 0:
            #     path_previous = path.replace(frame_idx, frame_idx_previous)
            # else:
            #     path_previous = path

            # # 使用with语句确保文件操作完成后自动关闭文件
            # with h5py.File(path, "r") as f, h5py.File(path_previous, "r") as f_previous:
            #     # 判断是否是二进制读取
            #     if self.bina_read:
            #         img_left = f['image_left'][:]
            #         img_left_previous = f_previous['image_left'][:]
            #     else:
            #         img_left = f['image'][:]
            #         img_left_previous = f_previous['image'][:]

            #     # 读取 mask 和 feat 数据
            #     mask = f['mask'][:].astype('int64')
            #     mask_previous = f_previous['mask'][:].astype('int64')
            #     feat = f['feat'][:].transpose((2, 0, 1))
            #     feat_previous = f_previous['feat'][:].transpose((2, 0, 1))

            #     # 读取 name 数据
            #     name = f['name'][()].decode('utf-8')
            #     name_previous = f_previous['name'][()].decode('utf-8')

            # # 数据转换操作
            # img_left = self.transform_input(img_left)
            # img_left_previous = self.transform_input(img_left_previous)
            # mask = self.transform_target(mask)
            # mask_previous = self.transform_target(mask_previous)

            # # 压缩维度
            # mask = mask.squeeze(0)
            # mask_previous = mask_previous.squeeze(0)

            # if not self.bina:
            #     # 合并图像和特征数据
            #     img_left = np.stack((img_left, img_left_previous), axis=0)
            #     mask = np.stack((mask, mask_previous), axis=0)
            #     feat = np.stack((feat, feat_previous), axis=0)

            #     return img_left, mask, feat, name

            # else:
            #     img_right=f['image_right'][:]
            #     try:
            #         flow_l2r=f['flow_l2r'][:]
            #     except:
            #         print(name)
            #         print(f.keys())
            #     img_right = self.transform_input(img_right)
            #     return (img_left,img_right,flow_l2r), mask,feat,name
