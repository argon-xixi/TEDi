import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
   

    
    '''
    backward warp: use flow to warp feature
        feature : [B, C, H, W]
        flow : [B, 2, H, W]
        if feature 来自前一帧的image/feature:flow 使用backward_flow
            grid + flow: img1的每个像素坐标 + 光流flow = 即为img1中该像素点对应在img2的坐标
        if feature 来自后一帧的image/feature:flow 使用forward_flow
            grid + flow: img2的每个像素坐标 + 光流flow = 即为img2中该像素点对应在img1的坐标
    '''
    
    b, c, h, w = feature.size()  # feature/image size [B, C, H, W]
    assert flow.size(1) == 2  # x flow and y flow
    # 1. get coords grid
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]
    grid = torch.stack([x, y], dim=0).float()  # [2, H, W]
    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W]

    # 2. vgrid = grid + flow
    vgrid = grid.to(flow.device) + flow  # sample_coords: [B, 2, H, W] in image scale

    # 3. bilinear sampling
    if vgrid.size(1) != 2:  # [B, H, W, 2]
        vgrid = vgrid.permute(0, 3, 1, 2)  # [B, 2, H, W]
    # scale grid to [-1,1] : 2*coords/(coords_max_value-1) - 1 in [-1,1]
    x_grid = 2 * vgrid[:, 0] / (w - 1) - 1
    y_grid = 2 * vgrid[:, 1] / (h - 1) - 1
    vgrid = torch.stack([x_grid, y_grid], dim=-1)  # shape=[B, H, W, 2] for grid_sample
    img = F.grid_sample(img, vgrid, mode='bilinear', padding_mode=padding_mode, align_corners=True)
    if mask:  # mask过滤超出边界的点，并非occ_mask
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]
        return img, mask
    return img

def warp(x, flo): #反向变换，根据im1到im2的光流图，变换im2到im1
    
    """
    整个warp是从img2根据前向光流(前向光流是t -> t+1 帧的)warp到img1的过程,所以叫backwarp
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    # """
    # x=np.expand_dims(x,axis=0)
    # x=x.transpose(0,3,1,2)
    # x=torch.from_numpy(x)
    # flo=np.expand_dims(flo,axis=0)
    # flo=flo.transpose(0,3,1,2)
    # flo=torch.from_numpy(flo)
    # flo=flo.cuda()
    B, C, H, W = x.size()
   
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)  # (B,1,H,W)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)  # (B,1,H,W)
    grid = torch.cat((xx,yy),dim=1).float()  # (B,2,H,W) 目标坐标系（im1 的像素位置）
    x, grid = x.cuda(), grid.cuda()
    
    # img1的每个像素坐标 + 光流flo = 即为该像素点对应在img2的坐标，因此从img2中采样即可完成从img2到img1的变换
    vgrid = Variable(grid) + flo  # (B,2,H,W)
        # scale grid to [-1,1] 
        # 取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0 
        # 取出光流u这个维度，，原来范围是0~H-1，再除以H-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0  
    
    # reshape (B,2,H,W) -> (B,H,W,2) 为什么要这么变呢？是因为要配合grid_sample这个函数的使用
    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid,align_corners=True, mode='bilinear') #在x上按照vgrid的坐标采样
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid,align_corners=True)

        ##2019 author
    mask[mask<0.9999] = 0
    mask[mask>0] = 1

        #2019 code
        # mask = torch.floor(torch.clamp(mask, 0 ,1))

    return output*mask
    # return output


def flow_resize(flow_r2l, feature_size):
    H_flow, W_flow = flow_r2l.shape[2:]
    H_feat, W_feat = feature_size
    flow_r2l_resized = torch.nn.functional.interpolate(
    flow_r2l, 
    size=(H_feat, W_feat), 
    mode='bilinear', 
    align_corners=True
)
    flow_r2l_resized[:,0,:,:] *= (W_feat / W_flow)  # 缩放光流值
    flow_r2l_resized[:,1,:,:] *= (H_feat / H_flow)
    return flow_r2l_resized

def compute_similarity(feat_left, feat_right_warped):
    # 归一化特征
    feat_left_norm = torch.nn.functional.normalize(feat_left, p=2, dim=1)
    feat_right_norm = torch.nn.functional.normalize(feat_right_warped, p=2, dim=1)
    
    # 计算逐像素余弦相似度
    similarity = torch.sum(feat_left_norm * feat_right_norm, dim=1, keepdim=True)  # [B,1,H,W]
    return similarity
def fuse_features(feat_left, feat_right_warped, similarity):
    # 使用相似度作为注意力权重
    alpha = torch.sigmoid(similarity)  # 将相似度转换为权重 [B,1,H,W]
    
    # 加权融合
    # fused_feature = alpha * feat_left + (1 - alpha) * feat_right_warped
    fused_feature = feat_left + alpha * feat_right_warped
    return fused_feature
