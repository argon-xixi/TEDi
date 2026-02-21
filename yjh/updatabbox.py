import os 
import h5py
import numpy as np
import cv2

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
ori_path="/data/guojiayi_files/yjh_data/EndoVis2018/data_h5_bbox/test/"
mask_path='/data/guojiayi_files/yjh_data/EndoVis2018/data_h5_bina/test/'
# ori_path='/data/guojiayi_files/yjh_data/EndoVis2018/data_h5_bbox/xixi/'

# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return np.stack(b, dim=-1)
# cls_dict={1:0,2:0,3:0,4:0,5:0,6:0,7:0}   
for m in tqdm(os.listdir(ori_path)):
    dict_all={}
    dict_all_new={}
    bbox_new=[]
    mask_new=[]
    cls_new=[]
    with h5py.File(ori_path+m, 'r+') as f:
        # if not 'seq_14_frame089' in m:
        #     continue
       
        # cls=f['cls'][:]
        # if len(cls) != len(set(cls)):
        bbox=np.load(os.path.join('/data/guojiayi_files/yjh_data/EndoVis2018/bbox_old/test/',m.split('.')[0]+'.npy'))
        if bbox.shape[0]==0:
            continue
        cls = bbox[:, -1]
        if len(cls) != len(set(cls)):
        
        
            
  
            
            for i in range(0,len(cls)):
                if cls[i] not in dict_all:
                    dict_all[cls[i]]=[]
                dict_all[cls[i]].append(bbox[i])
            for k,v in dict_all.items():
                
                v=np.array(v)
                # v=box_cxcywh_to_xyxy(v)
                x1,y1=np.min(v,axis=0)[0],np.min(v,axis=0)[1]
                x2,y2=np.max(v,axis=0)[2],np.max(v,axis=0)[3]
                dict_all_new[k]=[(x1 + x2) / 2, (y1 + y2) / 2,(x2 - x1), (y2 - y1)]
            # print(dict_all)
            # print(dict_all_new)
            with h5py.File(mask_path+m,'r') as f1:
                mask_list=[]
                original_masks=f1['mask'][:]
                ks = np.arange(1, 8)[:, np.newaxis, np.newaxis]  # 形状变为 (7, 1, 1)
                original_masks = (original_masks == ks).astype(np.int8)  # 广播比较，生成形状 (7, H, W)
                for k,v in dict_all_new.items():
                    bbox_new.append(v)
                    mask_new.append(original_masks[k-1])
                    cls_new.append(k)
                  # 转换为列表中的二维数组
            del f['bbox']
            del f['cls']
            del f['mask']
            f.create_dataset('bbox', data=bbox_new)
            f.create_dataset('cls', data=cls_new)
            f.create_dataset('mask', data=mask_new)
                
            
            print('haha')

# cp /data/guojiayi_files/yjh_data/EndoVis2018/data_h5_bbox/train/seq_16_frame084_3.h5 /data/guojiayi_files/yjh_data/EndoVis2018/data_h5_bbox/xixi/ seq_16_frame084