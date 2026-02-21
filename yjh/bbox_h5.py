import os 
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math
import time
import argparse
import glob
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
train_path_ori= '/data/guojiayi_files/yjh_data/EndoVis2018/data_h5_bina/test/'
train_path_new='/data/guojiayi_files/yjh_data/EndoVis2018/data_h5_bbox/test/'
len_list=[]
def pad_matrices(matrix):
    
    
    current_shape = matrix.shape
    first_dim = current_shape[0]
    # 计算需要填充的数量
    padding = 8 - first_dim if first_dim < 8 else 0
    pad_width = ((0, padding),) + ((0, 0),) * (len(current_shape) - 1)
    padded_matrix = np.pad(matrix, pad_width=pad_width, mode='constant', constant_values=0)
    
    return padded_matrix

for i in tqdm(os.listdir(train_path_ori)):
    name=i.split('.')[0]
    # if 'seq_7_frame056' not in name:
    #     continue
    # if os.path.exists(os.path.join(train_path_new,name+'.h5')):
    #     continue
   
        

    if len(i.split('_'))==3 or int(i.split('.')[0].split('_')[-1])<=100:
        print(i)
        mask_list=[]
        cls_list=[]
        bbox_list=[]
        
        train_path=train_path_ori+i
        f = h5py.File(train_path, 'r')
        print(f.keys())
        # img_left=f['image_left'][:]
        img_right=f['image_right'][:]
        flow_l2r=['flow_l2r'][:]
        # H, W ,C= img_left.shape
        # feat=f['feat'][:]
        # name=i.split('.')[0] # 读取标量要用[()]
        # mask=f['mask'][:]
        #      # 生成所有k值的掩码（向量化操作）
        # ks = np.arange(1, 8)[:, np.newaxis, np.newaxis]  # 形状变为 (7, 1, 1)
        # mask = (mask == ks).astype(np.int8)  # 广播比较，生成形状 (7, H, W)
        # mask = list(mask)  # 转换为列表中的二维数组
        bbox=np.load(os.path.join('/data/guojiayi_files/yjh_data/EndoVis2018/bbox_old/test/',name+'.npy'))
        



        if bbox is not None:
            # for k in bbox:
            #     dict_bbox={}
            #     cls=int(k[-1])
            #     x1,y1,x2,y2=k[:4]
            #     x1=int(x1)
            #     y1=int(y1)
            #     x2=int(x2)
            #     y2=int(y2)
            #     mask_bina = np.zeros((H, W), dtype=img_left.dtype)
            #     mask_bina[y1:y2, x1:x2] = 1
            #     mask_new=mask_bina*mask[cls-1] #此处cls需要-1
            #     mask_list.append(mask_new)
            #     cls_list.append(cls)
            #     bbox_list.append([(x1 + x2) / 2, (y1 + y2) / 2,(x2 - x1), (y2 - y1)])
            try:
                with h5py.File(os.path.join(train_path_new,name+'.h5'), 'a') as f1:

                    f1.create_dataset('image_right', data=img_right)
                    f1.create_dataset('flow_l2r', data=flow_l2r)
            except:
                with h5py.File(os.path.join(train_path_new,name+'.h5'), 'a') as f1:

                    # f1.create_dataset('image_right', data=img_right)
                    f1.create_dataset('flow_l2r', data=flow_l2r)
              
                # f1.create_dataset('image_left', data=img_left)
                # f1.create_dataset('feat', data=feat)
                # f1.create_dataset('mask', data=mask_list)
                # f1.create_dataset('name', data=name)
                # f1.create_dataset('cls', data=cls_list)
                # f1.create_dataset('bbox', data=bbox_list)
        else:
            print('no bbox')
            print(name)
            exit()
                    
            
            