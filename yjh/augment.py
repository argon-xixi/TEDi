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
# define the SAM model 
vit_mode = "h"
if vit_mode == "h":
    sam_checkpoint = "/home/yjh/code_yjh/sam_vit_h_4b8939.pth"
sam = sam_model_registry[f"vit_{vit_mode}"](checkpoint=sam_checkpoint)
sam.cuda()
predictor = SamPredictor(sam)


def augmentation(image, masks, bboxes, scale_factor, rotate_angle, colour_factor, H, W,nobbox=True ,
                 scale=False, rotate=False, colour=False):
    """
    Args:
        bboxes: list of bboxes in format [[x1,y1,x2,y2], ...] (绝对坐标)
    Returns:
        bboxes: 变换后的bbox坐标列表
    """
    # 深拷贝原始坐标防止污染
    # image=Image.fromarray(image)
    # masks=[Image.fromarray(mask) for mask in masks]
    if not nobbox :
        cls=copy.deepcopy(bboxes)[:,6]
        current_bboxes = copy.deepcopy(bboxes)[:,:-1]
    
        
        # current_bboxes = copy.deepcopy(bboxes)[:,:-1]
        # current_bboxes = [np.array(box, dtype=np.float32) for box in bboxes]
        
        # ================== 缩放与裁剪 ==================
        if scale and random.random() > 0.5:
            # 随机缩放
            scale = random.random()*scale_factor + 1
            new_h, new_w = int(H*scale), int(W*scale)
            
            # 缩放图像和掩码
            resize = transforms.Resize(size=(new_h, new_w))
            image = resize(image)
            masks = [resize(mask) for mask in masks]
            
            # 缩放bbox坐标 (xmin, ymin, xmax, ymax) 乘以缩放因子
            current_bboxes = [box * scale for box in current_bboxes]
            
            # 随机裁剪回原始尺寸
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(H, W))
            image = TF.crop(image, i, j, h, w)
            masks = [TF.crop(mask, i, j, h, w) for mask in masks]
            
            # 裁剪bbox坐标：减去偏移量并限制在有效范围内
            for idx in range(len(current_bboxes)):
                box = current_bboxes[idx]
                # 计算裁剪后的坐标
                box[0] = max(box[0] - j, 0)    # xmin
                box[1] = max(box[1] - i, 0)    # ymin
                box[2] = min(box[2] - j, W)    # xmax
                box[3] = min(box[3] - i, H)    # ymax
                # 过滤无效bbox（宽或高<=0）
                if box[2] <= box[0] or box[3] <= box[1]:
                    current_bboxes[idx] = None
            # 清除无效bbox
            current_bboxes = [b for b in current_bboxes if b is not None]

        # ================== 水平翻转 ==================
        if random.random() > 0.5:
            # print("horizontal flip")
            image = TF.hflip(image)
            masks = [TF.hflip(mask) for mask in masks]
            
            for box in current_bboxes:
                x1, y1, x2, y2, x3, y3 = box
                # 水平翻转后的坐标
                new_x1 = W - x2
                new_x2 = W - x1
                new_x3 = new_x1  # 左下角 x 坐标镜像
                box[:] = [new_x1, y1, new_x2, y2, new_x3, y3]

        # ================== 垂直翻转 ==================
        if random.random() > 0.5:
            # print("vertical flip")
            image = TF.vflip(image)
            masks = [TF.vflip(mask) for mask in masks]
            
            # bbox垂直镜像：y坐标 = H - 原y坐标
            for box in current_bboxes:
               
                x1, y1, x2, y2, x3, y3 = box
                new_y1 = H - y2
                new_y2 = H - y1
                new_y3 = new_y2
                box[:] = [x1, new_y1, x2, new_y2, x3, new_y3]

        # ================== 旋转 ==================
        if rotate and random.random() > 0.5:
            # print("rotate")
            angle = rotate_angle * random.random() * (1 if random.random()>0.5 else -1)
            image = TF.rotate(image, angle)
            masks = [TF.rotate(mask, angle) for mask in masks]
            
            center_x, center_y = W/2, H/2
            rad = np.deg2rad(angle)
            
            for box in current_bboxes:
                x1, y1, x2, y2, x3, y3 = box
                # 定义四个角点（左上、右上、右下、左下）
                points = np.array([
                    [x1, y1],  # 左上
                    [x2, y1],  # 右上（新增）
                    [x2, y2],  # 右下
                    [x3, y3]   # 左下
                ])
                
                # 旋转变换每个点
                rotated_points = []
                for (x, y) in points:
                    x_centered = x - center_x
                    y_centered = y - center_y
                    x_rot = x_centered * np.cos(rad) + y_centered * np.sin(rad)
                    y_rot = -x_centered * np.sin(rad) + y_centered * np.cos(rad)
                    x_new = x_rot + center_x
                    y_new = y_rot + center_y
                    rotated_points.append([x_new, y_new])
                
                # 计算外接矩形
                rotated_points = np.array(rotated_points)
                new_xmin = max(0, np.min(rotated_points[:, 0]))
                new_ymin = max(0, np.min(rotated_points[:, 1]))
                new_xmax = min(W, np.max(rotated_points[:, 0]))
                new_ymax = min(H, np.max(rotated_points[:, 1]))
                
                # 确定左下角（x最小中的y最大）
                left_points = rotated_points[rotated_points[:, 0] == new_xmin]
                new_x3 = new_xmin
                new_y3 = np.max(left_points[:, 1]) if len(left_points) > 0 else new_ymax
                
                # 更新坐标
                box[:] = [new_xmin, new_ymin, new_xmax, new_ymax, new_x3, new_y3]

        # ================== 颜色抖动 ==================
        if colour:
            if random.random() > 0.5:
                colour_jitter = transforms.ColorJitter(brightness=colour_factor, contrast=colour_factor, saturation=colour_factor)
                image = colour_jitter(image)
       
        
               # 在所有变换处理后，根据current_bboxes的剩余数量过滤对应的类别
        valid_cls = [cls[i] for i in range(len(cls)) if i < len(current_bboxes)]  # 假设current_bboxes在过滤时按顺序保留
        # 注意：需根据实际过滤逻辑调整valid_cls的获取方式

        # 合并坐标与类别
        if len(current_bboxes) > 0:
            current_bboxes = np.concatenate([current_bboxes, np.array(valid_cls).reshape(-1,1)], axis=1)
        else:
            current_bboxes = []
        bboxes = current_bboxes  # 更新最终bboxes
        return image, bboxes, masks,rotate
    
    # 没有bbox
    else:
          # Scale and crop
        if scale:
            # Scale
            if random.random() > 0.5:
                scale = random.random()*scale_factor + 1
                resize = transforms.Resize(size=(int(H*scale), int(W*scale)))
                image = resize(image) 
                masks = [resize(mask) for mask in masks]

                # Crop
                i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(H, W))
                image = TF.crop(image, i, j, h, w) 
                masks = [TF.crop(mask, i, j, h, w) for mask in masks]
            
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            masks = [TF.hflip(mask) for mask in masks]

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            masks = [TF.vflip(mask) for mask in masks]
        
        # Rotate 
        if rotate:
            if random.random() > 0.5:
                angle = rotate_angle * random.random() * (random.random()>0.5)
                image = TF.rotate(image, angle)
                masks = [TF.rotate(mask, angle) for mask in masks]
        
        # Colour jitter
        if colour:
            if random.random() > 0.5:
                colour_jitter = transforms.ColorJitter(brightness=colour_factor, contrast=colour_factor, saturation=colour_factor)
                image = colour_jitter(image)
                
        return image, masks,rotate
    
def augmentation_bina(image_left,image_right, masks, scale_factor, rotate_angle, colour_factor, H, W,
                 scale=False, rotate=False, colour=False):
    
      # Scale and crop
        if scale:
            # Scale
            if random.random() > 0.5:
                scale = random.random()*scale_factor + 1
                resize = transforms.Resize(size=(int(H*scale), int(W*scale)))
                image_left = resize(image_left) 
                image_right = resize(image_right)
                masks = [resize(mask) for mask in masks]

                # Crop
                i, j, h, w = transforms.RandomCrop.get_params(
                image_left, output_size=(H, W))
                image_left = TF.crop(image_left, i, j, h, w) 
                image_right = TF.crop(image_right, i, j, h, w)
                masks = [TF.crop(mask, i, j, h, w) for mask in masks]
            
        # Random horizontal flipping
        if random.random() > 0.5:
            image_left = TF.hflip(image_left)
            image_right = TF.hflip(image_right)
            masks = [TF.hflip(mask) for mask in masks]

        # Random vertical flipping
        if random.random() > 0.5:
            image_left = TF.vflip(image_left)
            image_right = TF.vflip(image_right)
            masks = [TF.vflip(mask) for mask in masks]
        
        # Rotate 
        if rotate:
            if random.random() > 0.5:
                angle = rotate_angle * random.random() * (random.random()>0.5)
                image_left = TF.rotate(image_left, angle)
                image_right = TF.rotate(image_right, angle)
                masks = [TF.rotate(mask, angle) for mask in masks]
        
        # Colour jitter
        if colour:
            if random.random() > 0.5:
                colour_jitter = transforms.ColorJitter(brightness=colour_factor, contrast=colour_factor, saturation=colour_factor)
                image_left = colour_jitter(image_left)
                image_right = colour_jitter(image_right)
                
        return image_left, image_right, masks,rotate


def version_to_augmentation_toggles(version, n_version):
    """Generate the toggles for scale, rotate, and colour for different augmentation versions
       version - current version 
       n_version - total number of versions 
       
       The augmentation settings for different versions (if n_version == 40):
       Version | Flip | Scale | Rotate | Colour 
       1 - 10     √       x       x         x
       11 - 20    √       √       x         x
       21 - 30    √       x       √         x
       31 - 40    √       x       x         √

    """
    scale = False
    rotate = False
    colour = False 
    
    if (n_version//4) < version < ((2*n_version//4)+1):
        scale = True 
    elif (2*n_version//4) < version < ((3*n_version//4)+1):
        rotate = True 
    elif (3*n_version//4) < version < n_version+1:
        colour = True 
        
    return scale, rotate, colour

def set_mask(mask):   
    """ Transform the mask to the form expected by SAM, the transformed mask will be used to generate class embeddings
        Adapated from set_image in the official code of SAM https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py
    """
    input_mask = ResizeLongestSide(1024).apply_image(mask)
    input_mask_torch = torch.as_tensor(input_mask)
    input_mask_torch = input_mask_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    
    input_mask = set_torch_image(input_mask_torch)
    
    return input_mask


def set_torch_image(transformed_mask):
    input_mask = preprocess(transformed_mask)  # pad to 1024
    return input_mask


def preprocess(x):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def data_process(original_frame, original_masks, bboxes,version, n_version, scale_factor, rotate_angle, colour_factor, H, W,nobbox=True):

    
    # read all the original masks (without any augmentation) of the current frame and organise them into a list
   

    
    # for each frame, generate 40 different versions of augmentation
    
        
        # 生成所有k值的掩码（向量化操作）
    ks = np.arange(1, 8)[:, np.newaxis, np.newaxis]  # 形状变为 (7, 1, 1)
    original_masks = (original_masks == ks).astype(np.int8)  # 广播比较，生成形状 (7, H, W)
    original_masks = list(original_masks)  # 转换为列表中的二维数组
    original_masks_list=[]
    for k in original_masks:
        mask=Image.fromarray(k)
        original_masks_list.append(mask)
    
    scale, rotate, colour = version_to_augmentation_toggles(version,n_version)    
    frame, masks,rotate = augmentation(original_frame, original_masks_list,bboxes,scale_factor, rotate_angle, colour_factor, H, W, nobbox,scale = scale, rotate = rotate, colour = colour)

    frame = np.asarray(frame)
    fina_masks=np.zeros((1024,1280))
    masks=np.asarray(masks)
    for n,mask in enumerate(masks):
        fina_masks+=((n+1)*mask)
    fina_masks = fina_masks .astype(int)
    # obtain SAM feature of the augmented frame 
    predictor.set_image(frame)
    feat = predictor.features.squeeze().permute(1, 2, 0)
    feat = feat.cpu().numpy()
    # print(bboxes)
    if nobbox:
        return frame, fina_masks,feat, rotate
    else:
        return frame, fina_masks, feat, bboxes, rotate
    
def data_process_bina(original_frame_left,original_frame_right, original_masks,version, n_version, scale_factor, rotate_angle, colour_factor, H, W):

    
    # read all the original masks (without any augmentation) of the current frame and organise them into a list
   

    
    # for each frame, generate 40 different versions of augmentation
    
        
     # 生成所有k值的掩码（向量化操作）
    ks = np.arange(1, 8)[:, np.newaxis, np.newaxis]  # 形状变为 (7, 1, 1)
    original_masks = (original_masks == ks).astype(np.int8)  # 广播比较，生成形状 (7, H, W)
    original_masks = list(original_masks)  # 转换为列表中的二维数组
    original_masks_list=[]
    for k in original_masks:
        mask=Image.fromarray(k)
        original_masks_list.append(mask)
        
    scale, rotate, colour = version_to_augmentation_toggles(version,n_version)    
    
    frame_left,frame_right, masks,rotate = augmentation_bina(original_frame_left,original_frame_right, original_masks_list,scale_factor, rotate_angle, colour_factor, H, W,scale = scale, rotate = rotate, colour = colour)
   
    
    
    # perform augmentation to the frame and its masks based on the current version number
    
    frame_left = np.asarray(frame_left)
    frame_right = np.asarray(frame_right)
    fina_masks=np.zeros((1024,1280))
    masks=np.asarray(masks)
    for n,mask in enumerate(masks):
        fina_masks+=((n+1)*mask)
    fina_masks = fina_masks .astype(int)
        
    
    # obtain SAM feature of the augmented frame  只对左图提取特征
    predictor.set_image(frame_left)
    feat = predictor.features.squeeze().permute(1, 2, 0)
    feat = feat.cpu().numpy()
    
    return frame_left, frame_right, fina_masks,feat, rotate
 
    