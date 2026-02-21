import os 
import h5py
import numpy as np
import cv2
import sys
sys.path.append("/home/gjy/code_yjh/SurgicalSAM-main")
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
from augment import data_process_bina

    
    

# define augmentation factors 
scale_factor = 0.2
rotate_angle = 30
colour_factor = 0.4


image_test=False
height, width = 1024, 1280
h_start, w_start = 28, 320
train_path='/data/guojiayi_files/yjh_data/EndoVis2017/data_raw/cropped_train_left/'
val_image_path='/data/guojiayi_files/yjh_data/EndoVis2017/data_raw/Endovis2017/test/'
val_mask_path='/data/guojiayi_files/yjh_data/EndoVis2017/data_raw/test label/'
train_image_list=[]
train_mask_list=[]

val_image_list=[]
val_mask_list=[]
train_h5_ori="/data/guojiayi_files/yjh_data/EndoVis2017/data_h5_bina/train/"
val_h5_ori="/data/guojiayi_files/yjh_data/EndoVis2017/data_h5_bina/test_resized/"
image_path="/data/guojiayi_files/yjh_data/EndoVis2017/data_h5_bina/images"
cnt=0
n_version=16

for i in os.listdir(train_path):
    if os.path.isdir(os.path.join(train_path,i)):
        path1=os.path.join(train_path,i,'images')
        for i in os.listdir(path1):
            path2=os.path.join(path1,i)
            # path3=path2.replace('images','instruments_masks')
            train_image_list.append(path2)
            

# for i in os.listdir(val_mask_path):
#     if os.path.isdir(os.path.join(val_mask_path,i)):
#         path1=os.path.join(val_mask_path,i,'TypeSegmentation')
#         for i in os.listdir(path1):
#             path3=os.path.join(path1,i)
#             val_mask_list.append(path3)

# for i in tqdm(train_image_list):
#     train_image_left=cv2.imread(i)
#     train_image_left=cv2.cvtColor(train_image_left,cv2.COLOR_BGR2RGB)
    
#     path_right=i.replace('cropped_train_left','cropped_train_right')
#     train_image_right=cv2.imread(path_right)
#     train_image_right=cv2.cvtColor(train_image_right,cv2.COLOR_BGR2RGB)
#     train_image_left=Image.fromarray(train_image_left)
#     train_image_right=Image.fromarray(train_image_right)
#     maskPath=i.replace('images','instruments_masks')
#     train_mask=cv2.imread(maskPath.replace('.jpg','.png'))[:,:,0]
#     # print(type(train_mask))
    
#     train_mask=train_mask*(1/32)
#     train_mask=np.array(train_mask,dtype=np.uint8)
    
    
#     # print(type(train_mask))
    
#     name=i.split('/')[-3]+"_"+i.split('/')[-1].replace('.jpg','')
#     # train_bboxes=np.load("/data/guojiayi_files/yjh_data/bbox/train/"+name+'.npy')
    
#     for version in range(9,n_version+9):
       
#         # set seed for reproducibility
   
#         random.seed(version)
#         torch.manual_seed(version)
#         torch.cuda.manual_seed(version)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         np.random.seed(version)
        
#         frame_left,frame_right, masks,feat,rotate=data_process_bina(train_image_left,train_image_right,train_mask,version,n_version,scale_factor,rotate_angle,colour_factor,height,width)
#         if rotate:
#             rotate_id=1
#         else:
#             rotate_id=0
#         mm={}
#         # for K in masks.flat:
#         #     if K not in mm.keys():
#         #         mm[K]=1
#         #     else:
#         #         mm[K]+=1
#         # print(mm)
#         # print(bboxes)
#         train_h5_path=name+'_'+str(version)+'.h5'
#         print(f"perform augmentation for frame {train_h5_path}")

#         with h5py.File(os.path.join(train_h5_ori,train_h5_path),'w') as f:
#             f.create_dataset("name", data=name)
#             f.create_dataset("mask", data=masks)
#             f.create_dataset("image_left", data=frame_left)
#             f.create_dataset("image_right", data=frame_right)
#             f.create_dataset("feat", data=feat)
#             # f.create_dataset("bboxes", data=bboxes)
#             f.create_dataset("rotate", data=rotate_id)
#             f.close()

#             cnt+=1
#             print(cnt)
            
#         # if image_test:
        #     for k in bboxes:
        #         cls=int(k[-1])
        #         x1,y1,x2,y2,x3,y3=k[:-1]
        #         x1=int(x1)
        #         y1=int(y1)
        #         x2=int(x2)
        #         y2=int(y2)
        #         x3=int(x3)
        #         y3=int(y3)
               
        #         color=(cls*32,cls*32,cls*32)
        #         frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #         frame=cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        #     cv2.imwrite(image_path+name+'_'+str(version)+'.png',frame)
mmm={}
cnt=0
vit_mode = "h"
if vit_mode == "h":
    sam_checkpoint = "/home/gjy/code_yjh/sam_vit_h_4b8939.pth"
sam = sam_model_registry[f"vit_{vit_mode}"](checkpoint=sam_checkpoint)
sam.cuda()
predictor = SamPredictor(sam)
max_dict={}
# for i in tqdm(val_mask_list):#/data/gjy_files/yjh_data/data_raw/Endovis2017/test label/instrument_dataset_1/TypeSegmentation/

#     image_path_left=i.replace('TypeSegmentation','left_frames').replace("test label","test")
#     image_path_right=i.replace('TypeSegmentation','right_frames').replace("test label","test")
#     #/data/gjy_files/yjh_data/data_raw/Endovis2017/test/instrument_dataset_1/left_frames/
#     val_image_left=cv2.imread(image_path_left)[h_start: h_start + height, w_start: w_start + width,:]
#     val_image_right=cv2.imread(image_path_right)[h_start: h_start + height, w_start: w_start + width,:]
#     val_mask=cv2.imread(i)[h_start: h_start + height, w_start: w_start + width,0]
#     if val_mask.max() in max_dict.keys():
#         max_dict[val_mask.max()]+=1
#     else:
#         max_dict[val_mask.max()]=1
# print(max_dict)
    # val_mask[val_mask>7]=7
    # val_image_left=cv2.cvtColor(val_image_left,cv2.COLOR_BGR2RGB)
    # val_image_right=cv2.cvtColor(val_image_right,cv2.COLOR_BGR2RGB)
    # name=i.split('/')[-3]+"_"+i.split('/')[-1].replace('.png','')
    # val_h5_path=name+'.h5'
    # predictor.set_image(val_image_left)
    # feat = predictor.features.squeeze().permute(1, 2, 0)
    # feat = feat.cpu().numpy()
    # mm={}
    # for K in val_mask.flat:
        
    #     if K not in mm.keys():
    #         mm[K]=1
    #     else:
    #         mm[K]+=1
   

    # # bboxes=np.load("/data/guojiayi_files/yjh_data/bbox/test/"+name+'.npy')
    # with h5py.File(os.path.join(val_h5_ori,val_h5_path),'w') as f:
    #     f.create_dataset("name", data=name)
    #     f.create_dataset("mask", data=val_mask)
    #     f.create_dataset("image_left", data=val_image_left)
    #     f.create_dataset("image_right", data=val_image_right)
        
    #     f.create_dataset("feat", data=feat)
    #     # f.create_dataset("bboxes", data=bboxes)
    #     f.close()

    #     cnt+=1
    #     print(cnt)
 
   
for i in tqdm(train_image_list):
    path_right=i.replace('cropped_train_left','cropped_train_right')
    train_image_left=cv2.imread(i)
    train_image_right=cv2.imread(path_right)
    train_image_left=cv2.cvtColor(train_image_left,cv2.COLOR_BGR2RGB)
    train_image_right=cv2.cvtColor(train_image_right,cv2.COLOR_BGR2RGB)
    maskPath=i.replace('images','instruments_masks')
    train_mask=cv2.imread(maskPath.replace('.jpg','.png'))[:,:,0]
    # print(type(train_mask))
    
    train_mask=train_mask*(1/32)
    
    train_mask=np.array(train_mask,dtype=np.uint8)
    # print(type(train_mask))
    
    name=i.split('/')[-3]+"_"+i.split('/')[-1].replace('.jpg','')
    train_h5_path=name+'.h5'
    # train_bboxes=np.load("/data/guojiayi_files/yjh_data/bbox/train/"+name+'.npy')
    predictor.set_image(train_image_left)
    feat = predictor.features.squeeze().permute(1, 2, 0)
    feat = feat.cpu().numpy()
    mm={}
    for K in train_mask.flat:
        if K not in mm.keys():
            mm[K]=1
        else:
            mm[K]+=1
    # print(mm)

    # bboxes=np.load("/data/guojiayi_files/yjh_data/bbox/train/"+name+'.npy')
    with h5py.File(os.path.join(train_h5_ori,train_h5_path),'w') as f:
        f.create_dataset("name", data=name)
        f.create_dataset("mask", data=train_mask)
        f.create_dataset("image_left", data=train_image_left)
        f.create_dataset("image_right", data=train_image_right)
        f.create_dataset("feat", data=feat)
        # f.create_dataset("bboxes", data=bboxes)
        f.close()

        cnt+=1
        print(cnt)
# mm={} 
# mmm={}
# path='/data/guojiayi_files/yjh_data/data_h5/test/'
# for i in tqdm(os.listdir(path)):
#     if i.split('.')[-1]=='h5':
#         with h5py.File(os.path.join(path,i),'r') as f:
#             name=f['name'][()]
#             mask=f['mask'][:]
#             image=f['image'][:]
#             feat=f['feat'][:]
#             bboxes=f['bboxes'][:]
           
#     for K in mask.flat:
#         if K not in mm.keys():
#             mm[K]=1
#         else:
#             mm[K]+=1
#     # print(mm)
#     for j in mm.keys():
#         if j not in mmm.keys():
#             mmm[j]=1
#         else:
#             mmm[j]+=1
# print(mmm)
            
   

   
   

        