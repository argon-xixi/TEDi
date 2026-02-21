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
        affine=False,
        bina=False,
        bbox=True,
        bina_read=False,
        istrain=True
    ):
        self.input_paths = input_paths
        self.version = version
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.cfg = cfg
        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine
        self.bina = bina
        self.bina_read = bina_read
        self.bbox = bbox
        self.istrain = istrain
        self.remove_list={'seq_2_frame004','seq_4_frame137','seq_6_frame104','seq_6_frame138','seq_10_frame072','seq_10_frame105','seq_15_frame139',
             'seq_16_frame030','seq_16_frame068'}

        # 初始化增强操作
        self.tfm_gens = []  # 存储数据增强方法
        if self.hflip:
            self.tfm_gens.append(T.RandomFlip(horizontal=True, vertical=False))
        if self.vflip:
            self.tfm_gens.append(T.RandomFlip(horizontal=False, vertical=True))
        # if self.affine:
        #     self.tfm_gens.append(T.RandomRotation(angle=[-30, 30], expand=True))  # 可根据需求调整旋转角度
           

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        if self.istrain:
            idx = random.choice([0, 1, 2, 3])
        else:
            idx = 0
        mask_features_list = []
        pred_embds_list = []
        pred_logits_list = []
        pred_masks_list = []
        target_list = []

        
        # 获取文件路径
        # path = self.input_paths[index] + (f"_{self.version}" if self.version else "") + ".h5"
        # /data1/yuanjiahong_files/bishe/EndoVis2018/mask2former_feature/baseline_augnew_1_bestiou/test/seq_9_frame143.h5
        # '/data1/yuanjiahong_files/bishe/EndoVis2018/data_h5_baseline/train/seq_10_frame095.h5'
        path = self.input_paths[index].replace('/data_h5_baseline/','/mask2former_feature/baseline_augnew_1_earlystop/') +  ".h5"
        path_parts = path.split('.')[0].split('_')
        frame_idx = path_parts[7]  # 获取当前帧的索引
        cnt=0
        # 数据增强应用到前一帧、当前帧、后一帧
        for i in [ 2, 1, 0]:
            if int(frame_idx[5:8]) not in [0,1]:
                frame_idx_previous = "frame" + str(int(frame_idx[5:8]) - i).zfill(3)
                path_previous = path.replace(frame_idx, frame_idx_previous)
                if path_previous.split('.')[0].split('/')[-1] in self.remove_list:
                    frame_idx_previous = "frame" + str(int(frame_idx[5:8]) - i-1).zfill(3)
                    path_previous = path.replace(frame_idx, frame_idx_previous)
            else:
                path_previous = path

            # 读取前一帧、当前帧或后一帧数据
            with h5py.File(path_previous, "r") as f:
                # print(f.keys())
                mask_features = f["mask_features"][:]
                pred_embds = f["pred_embds"][:]
                pred_logits = f["pred_logits"][:]
                pred_masks = f["pred_masks"][:]
                target = f["target"][:].astype('int64')
                name = f["name"][()].decode('utf-8')
                mask_features_list.append(mask_features[idx])
                pred_embds_list.append(pred_embds[idx])
                pred_logits_list.append(pred_logits[idx])
                pred_masks_list.append(pred_masks[idx])
                target_list.append(target[idx])
                
        mask_features = np.stack(mask_features_list, axis=0)
        pred_embds = np.stack(pred_embds_list, axis=0)
        pred_logits = np.stack(pred_logits_list, axis=0)
        pred_masks = np.stack(pred_masks_list, axis=0)
        target = np.stack(target_list, axis=0)

                
        return mask_features, pred_embds, pred_logits, pred_masks, target, name
        