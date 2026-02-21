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
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
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
        if args.bina:
            train_image_right=cv2.imread(os.path.join(train_image_right_path,'train',b+'_'+c,'right_frames',d))
            train_image_right=cv2.cvtColor(train_image_right,cv2.COLOR_BGR2RGB)
        # train_image=Image.fromarray(train_image)
        predictor.set_image(train_image_left)
        feat = predictor.features.squeeze().permute(1, 2, 0)
        feat = feat.cpu().numpy()
        
        name=os.listdir(train_image_path)[i].replace('.png','')
        train_h5_path=name+'.h5'
        with h5py.File(os.path.join(train_h5_ori,train_h5_path),'w') as f:
                f.create_dataset("name", data=name)
                f.create_dataset("mask", data=train_mask)
                f.create_dataset("image_left", data=train_image_left)
                if args.bina:
                    f.create_dataset("image_right", data=train_image_right)
                f.create_dataset("feat", data=feat)
                # f.create_dataset("bboxes", data=bboxes)
                f.close()


# 训练集（增强）
def train_datah5_argument(args):
    cnt=0
    cnt+=1
    print(cnt)
    for i in tqdm(range(len(os.listdir(train_image_path)))):
        # if 'seq_7_frame056' not in os.listdir(train_image_path)[i]:
        #     continue

        a=os.listdir(train_image_path)[i]
        train_image_left=cv2.imread(os.path.join(train_image_path,os.listdir(train_image_path)[i]))
        b,c,d=a.split('_')
        bboxes=None
        train_mask=cv2.imread(os.path.join(train_mask_path,os.listdir(train_mask_path)[i]))[:,:,0]
        train_image_left=cv2.cvtColor(train_image_left,cv2.COLOR_BGR2RGB)
        train_image_left=Image.fromarray(train_image_left)
        if args.bina:
            train_image_right=cv2.imread(os.path.join(train_image_right_path,'train',b+'_'+c,'right_frames',d))
            train_image_right=cv2.cvtColor(train_image_right,cv2.COLOR_BGR2RGB)
            train_image_right=Image.fromarray(train_image_right)
        name=os.listdir(train_image_path)[i].replace('.png','')
        for version in range(args.start_version,args.n_version+args.start_version):
            random.seed(version)
            torch.manual_seed(version)
            torch.cuda.manual_seed(version)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(version)
            if args.bina:
                frame_left,frame_right, masks,feat,rotate=data_process_bina(train_image_left,train_image_right,train_mask,version,args.n_version,scale_factor,rotate_angle,colour_factor,height,width)
            elif args.bbox:
                frame_left,masks,feat,bboxes,rotate=data_process(train_image_left,train_mask,bboxes,version,args.n_version,scale_factor,rotate_angle,colour_factor,height,width,nobbox=False)
            else:
                frame_left, masks,feat,rotate=data_process(train_image_left,train_mask,bboxes,version,args.n_version,scale_factor,rotate_angle,colour_factor,height,width,nobbox=True)
            
            if rotate:
                rotate_id=1
            else:
                rotate_id=0

            # print(bboxes)
            train_h5_path=name+'_'+str(version)+'.h5'
            print(f"perform augmentation for frame {train_h5_path}")
            with h5py.File(os.path.join(train_h5_ori,train_h5_path),'w') as f:
                f.create_dataset("name", data=name)
                f.create_dataset("mask", data=masks)
                f.create_dataset("image_left", data=frame_left)
                if args.sam:
                    f.create_dataset("feat", data=feat)
                if args.bina:
                    f.create_dataset("image_right", data=frame_right)
                # f.create_dataset("bboxes", data=bboxes)
                f.create_dataset("rotate", data=rotate_id)
                f.close()


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

        predictor.set_image(test_image_left)
        feat = predictor.features.squeeze().permute(1, 2, 0)
        feat = feat.cpu().numpy()
        name=os.listdir(test_image_path)[i].replace('.png','')
        test_h5_path=name+'.h5'
        with h5py.File(os.path.join(test_h5_ori,test_h5_path),'w') as f:
                f.create_dataset("name", data=name)
                f.create_dataset("mask", data=test_mask)
                f.create_dataset("image_left", data=test_image_left)
                if args.sam:
                    f.create_dataset("feat", data=feat)
                # if args.bina:
                #     f.create_dataset("image_right", data=test_image_right)
                # if args.bbox:
                #     f.create_dataset("bboxes", data=bboxes)
                f.close()
                
def get_args():
    parser = argparse.ArgumentParser(description="data argumentation")
    parser.add_argument("--n_version", default=8, type=int) #增强次数
    parser.add_argument("--start_version", default=33, type=int) #开始次数
    parser.add_argument("--task", default="train", type=str,choices=["train","test"]) 
    parser.add_argument("--sam", default=True, type=bool) #是否运行sam模块
    parser.add_argument("--bina", default=False, type=bool) #是否运行bina模块
    parser.add_argument("--bbox", default=False, type=bool) #是否运行bbox模块   
    args = parser.parse_args()
    return args\

if __name__ == "__main__":
    vit_mode = "h"
    sam_checkpoint = "/home/yjh/code_yjh_bishe/sam_vit_h_4b8939.pth"
    sam = sam_model_registry[f"vit_{vit_mode}"](checkpoint=sam_checkpoint)
    sam.cuda()
    predictor = SamPredictor(sam)
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
    if args.task=="train":
        # train_datah5_argument(args)
        train_datah5_noargument(args)
    elif args.task=="test":
        test_datah5_noargument(args)

'''
conda activate yjh
python /home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/yjh/endovis2018_h5.py
/home/yjh/anaconda3/envs/yjh/bin/python /home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/yjh/main_mask2former.py
'''
