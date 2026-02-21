import os 
import h5py
import numpy as np
import cv2
import os 
import h5py
import numpy as np
import cv2
import sys
sys.path.append("/home/yjh/code_yjh_bishe/SurgicalSAM-main")
import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image
import random 
import torch 
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.nn import functional as F
import argparse 
import copy
from tqdm import tqdm
from augment import data_process,data_process_bina

# 检查掩膜是否正确
def checkmask(masks):
    mm={}
    for K in masks.flat:
        if K not in mm.keys():
            mm[K]=1
        else:
            mm[K]+=1
    print(mm)
    
def vis(pred_mask,image_left,name):
        palette = np.array([
        [0, 0, 0],        # class 0 - black
        [255, 0, 0],      # class 1 - red
        [0, 255, 0],      # class 2 - green
        [0, 0, 255],      # class 3 - blue
        [255, 255, 0],    # class 4 - yellow
        [255, 0, 255],    # class 5 - magenta
        [0, 255, 255],    # class 6 - cyan
        [255, 255, 255]   # class 7 - white
    ], dtype=np.uint8)
        #处理第一个样本
        pred_mask = cv2.resize(pred_mask, (128,128),interpolation=cv2.INTER_NEAREST) # (H, W)
        z_n1 = pred_mask.astype(np.uint8)
        
        # 检查数值范围 (重要!)
        z_n1 = np.clip(z_n1, 0, len(palette)-1)

        color_mask1 = palette[z_n1]  # 自动广播到 (H, W, 3)
        image_left=cv2.resize(image_left, (128,128))
        color_show = np.concatenate([color_mask1,image_left],axis=1)
        
        cv2.imwrite(
            f'/data1/yuanjiahong_files/bishe/data_recheck/{name}',
            cv2.cvtColor(color_show, cv2.COLOR_RGB2BGR)
        )

# 训练集（不增强)
def train_datah5_noargument(args):
    for i in tqdm(range(len(os.listdir(train_image_path)))):
        a=os.listdir(train_image_path)[i]
        train_image_left=cv2.imread(os.path.join(train_image_path,os.listdir(train_image_path)[i]))
        b,c,d=a.split('_')

        train_mask=cv2.imread(os.path.join(train_mask_path,os.listdir(train_mask_path)[i]))[:,:,0]
        print(train_mask.max())
        mm={}
        # for K in train_mask.flat:j
        #     if K not in mm.keys():
        #         mm[K]=1
        #     else:
        #         mm[K]+=1
        # print(mm)
        train_image_left=cv2.cvtColor(train_image_left,cv2.COLOR_BGR2RGB)
        vis(train_mask,train_image_left,os.listdir(train_image_path)[i])
        



# 测试集（不增强）
def test_datah5_noargument(args):
    for i in tqdm(range(len(os.listdir(test_image_path)))):
        
        # if 'seq_7_frame056' not in os.listdir(test_image_path)[i]:
        #     continue
        
        
        # train_mask_path_temp=i.replace('images','annotations')
        a=os.listdir(test_image_path)[i]
        test_image_left=cv2.imread(os.path.join(test_image_path,os.listdir(test_image_path)[i]))
        b,c,d=a.split('_')
        # 
        path=os.path.join(test_mask_path,os.listdir(test_image_path)[i])
        test_mask=cv2.imread(os.path.join(test_mask_path,os.listdir(test_image_path)[i]))[:,:,0]
        print(test_mask.max())
        mm={}
        # for K in test_mask.flat:j
        #     if K not in mm.keys():
        #         mm[K]=1
        #     else:
        #         mm[K]+=1
        # print(mm)
        test_image_left=cv2.cvtColor(test_image_left,cv2.COLOR_BGR2RGB)
        # if args.bina:
        #     test_image_right=cv2.imread(os.path.join(test_image_right_path,'test',b+'_'+c,'right_frames',d))
        #     test_image_right=cv2.cvtColor(test_image_right,cv2.COLOR_BGR2RGB)
        vis(test_mask,test_image_left,os.listdir(test_image_path)[i])
        
                                    
                
def get_args():
    parser = argparse.ArgumentParser(description="data argumentation")
    parser.add_argument("--n_version", default=4, type=int) #增强次数
    parser.add_argument("--start_version", default=13, type=int) #开始次数
    parser.add_argument("--task", default="train", type=str,choices=["train","test"]) 
    parser.add_argument("--sam", default=True, type=bool) #是否运行sam模块
    parser.add_argument("--bina", default=False, type=bool) #是否运行bina模块
    parser.add_argument("--bbox", default=False, type=bool) #是否运行bbox模块   
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # define augmentation factors 
    scale_factor = 0.2
    rotate_angle = 30
    colour_factor = 0.4
    train_image_right_path='/data1/yuanjiahong_files/bishe/EndoVis2018/miccai_challenge_2018/'
    train_image_path='/data1/yuanjiahong_files/bishe/EndoVis2018/data_raw/train/images/'
    train_mask_path='/data1/yuanjiahong_files/bishe/EndoVis2018/data_raw/train/annotations/'
    test_image_path='/data1/yuanjiahong_files/bishe/EndoVis2018/data_raw/test/images/'
    test_mask_path='/data1/yuanjiahong_files/bishe/EndoVis2018/data_raw/test/annotations/'
    
    train_h5_ori="/data1/yuanjiahong_files/bishe/EndoVis2018/data_h5_baseline/train/"
    test_h5_ori="/data1/yuanjiahong_files/bishe/EndoVis2018/data_h5_baseline/test/"
    if not os.path.exists(train_h5_ori):
        os.makedirs(train_h5_ori)
    if not os.path.exists(test_h5_ori):
        os.makedirs(test_h5_ori)
    # test_image_path='/data1/yuanjiahong_files/bishe/EndoVis2018/data_raw/test/images/'  
    # train_h5_ori="/data1/yuanjiahong_files/bishe/EndoVis2018/data_h5_bina/test/"   
    # train_mask_path='/data1/yuanjiahong_files/bishe/EndoVis2018/data_raw/test/annotations/'  
    height, width = 1024, 1280
    args=get_args()
    train_datah5_noargument(args)
    test_datah5_noargument(args)

'''
python /home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/yjh/data_recheck.py
'''
