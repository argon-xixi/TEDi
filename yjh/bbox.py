import os 
import h5py
import numpy as np
import cv2

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

def draw_merged_bboxes(name,cls,mask, image, morph_kernel_size=5, dbscan_eps=50):
    """
    处理不连续的语义分割掩膜，生成合并后的边界框
    
    参数:
        mask (numpy.ndarray): 二值化的语义分割掩膜
        image (numpy.ndarray): 要绘制边界框的原图（彩色图像）
        morph_kernel_size (int): 形态学操作的核大小
        dbscan_eps (float): DBSCAN聚类算法的eps参数
    
    返回:
        numpy.ndarray: 绘制了合并边界框的图像
        list: 合并后的边界框列表 [(x1,y1,x2,y2), ...]
    """
    # 1. 形态学预处理（闭运算）
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2. 连通区域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed_mask)
    boxes = []
    for i in range(1, num_labels):  # 跳过背景
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, x + w, y + h))

    if not boxes:
        return image, []  # 没有检测到区域

    # 3. 聚类合并（基于bbox中心点）
    centers = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for (x1, y1, x2, y2) in boxes])
    clustering = DBSCAN(eps=dbscan_eps, min_samples=1).fit(centers)
    cluster_labels = clustering.labels_

    # 4. 生成合并后的bbox
    merged_boxes = []
    for label in set(cluster_labels):
        if label == -1:
            continue  # 处理噪声点（可选）
        # 合并当前聚类的所有bbox
        cluster_indices = np.where(cluster_labels == label)[0]
        x1 = int(min(boxes[i][0] for i in cluster_indices))
        y1 = int(min(boxes[i][1] for i in cluster_indices))
        x2 = int(max(boxes[i][2] for i in cluster_indices))
        y2 = int(max(boxes[i][3] for i in cluster_indices))
        x3=x1
        y3=y2
        
        merged_boxes.append((x1, y1, x2, y2,x3,y3,int(cls)))
        img_height, img_width = mask.shape[:2]
        max_allow_width = img_width * 0.75
        max_allow_height= img_height * 0.75                  # 允许的最大宽度（40%图像宽度）
    
    
        if (x2 - x1) > max_allow_width or (y2 - y1 )> max_allow_height:
            progress= '处理图片：'+name+'  大区域分割：'+str((x1, y1, x2, y2))+'\n'
            with open('/home/gjy/code_yjh/Mask2Former-Simplify-master/yjh/大区域分割_old.txt', 'a') as f:
                f.write(progress)
            

 

    return  merged_boxes


# def draw_merged_bboxes(name,cls, mask, image, morph_kernel_size=5, dbscan_eps=50):
#     # 1. 自适应形态学处理（动态调整核尺寸）
#     avg_area = np.sum(mask) / (mask.shape[0] * mask.shape[1])
#     adaptive_kernel_size = max(3, int(np.sqrt(avg_area) * 0.1))
#     kernel = np.ones((adaptive_kernel_size, adaptive_kernel_size), np.uint8)
#     processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     # 2. 连通区域分析（保留轮廓信息）
#     contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     boxes = []
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         boxes.append((x, y, x + w, y + h))

#     if not boxes:
#         return image, []

#     # 3. 大区域检测与分割
#     refined_boxes = []
#     img_height, img_width = mask.shape[:2]
#     max_allow_width = img_width * 0.75
#     max_allow_height= img_height * 0.75                  # 允许的最大宽度（40%图像宽度）
    
#     for (x1, y1, x2, y2) in boxes:
#         # 判断是否需要分割
#         if (x2 - x1) > max_allow_width or (y2 - y1 )> max_allow_height:
#             progress= '处理图片：'+name+'  大区域分割：'+str((x1, y1, x2, y2))+'\n'
#             with open('/home/gjy/code_yjh/Mask2Former-Simplify-master/yjh/大区域分割.txt', 'a') as f:
#                 f.write(progress)
            
#             # 对大区域进行分水岭分割
#             roi = mask[y1:y2, x1:x2]
#             sub_boxes = split_large_region(roi, x1, y1)
#             refined_boxes.extend(sub_boxes)
#         else:
#             refined_boxes.append((x1, y1, x2, y2))

#     # 4. 几何特征聚类（结合位置+尺寸）
#     cluster_features = []
#     for (x1, y1, x2, y2) in refined_boxes:
#         center_x = (x1 + x2) / 2
#         center_y = (y1 + y2) / 2
#         width = x2 - x1
#         height = y2 - y1
#         cluster_features.append([center_x, center_y, width * 0.2, height * 0.2])  # 尺寸加权

#     clustering = DBSCAN(eps=dbscan_eps, min_samples=1).fit(cluster_features)
#     cluster_labels = clustering.labels_

#     # 5. 动态合并控制
#     merged_boxes = []
#     for label in set(cluster_labels):
#         cluster_indices = np.where(cluster_labels == label)[0]
#         if len(cluster_indices) == 1:
#             merged_boxes.append(refined_boxes[cluster_indices[0]] + (int(cls),))
#             continue

#         # 合并时检查宽高比异常
#         x1 = min(refined_boxes[i][0] for i in cluster_indices)
#         y1 = min(refined_boxes[i][1] for i in cluster_indices)
#         x2 = max(refined_boxes[i][2] for i in cluster_indices)
#         y2 = max(refined_boxes[i][3] for i in cluster_indices)
#         merged_width = x2 - x1
#         merged_height = y2 - y1
        
#         # 拒绝异常宽高比的合并
#         if merged_width / merged_height > 3.5 or merged_height / merged_width > 3.5:
#             for i in cluster_indices:
#                 merged_boxes.append(refined_boxes[i] + (int(cls),))
#         else:
#             merged_boxes.append((x1, y1, x2, y2, int(cls)))

#     return merged_boxes

# def split_large_region(roi_mask, offset_x, offset_y):
#     """使用分水岭算法分割大区域"""
#     # 计算距离变换
#     dist_transform = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
#     _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
#     sure_fg = np.uint8(sure_fg)
    
#     # 分水岭分割
#     unknown = cv2.subtract(roi_mask, sure_fg)
#     _, markers = cv2.connectedComponents(sure_fg)
#     markers += 1
#     markers[unknown == 255] = 0
#     cv2.watershed(cv2.cvtColor(roi_mask*255, cv2.COLOR_GRAY2BGR), markers)
    
#     # 提取分割后的区域
#     sub_boxes = []
#     for mark in np.unique(markers):
#         if mark <= 1:
#             continue
#         temp_mask = np.uint8(markers == mark)
#         cnts, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in cnts:
#             x, y, w, h = cv2.boundingRect(cnt)
#             sub_boxes.append((
#                 x + offset_x, 
#                 y + offset_y,
#                 x + w + offset_x,
#                 y + h + offset_y
#             ))
#     return sub_boxes

height, width = 1024, 1280
h_start, w_start = 28, 320
train_path='/data/guojiayi_files/yjh_data/EndoVis2017/data_h5_bina/train'
val_image_path='/data/guojiayi_files/yjh_data/data_raw/Endovis2017/test/'
val_mask_path='/data/guojiayi_files/yjh_data/data_raw/Endovis2017/test label/'
train_image_list=[]
train_mask_list=[]
val_image_list=[]
val_mask_list=[]
train_h5_ori="/data/guojiayi_files/yjh_data/EndoVis2017/bbox_old/train/"
if not os.path.exists(train_h5_ori):
    os.makedirs(train_h5_ori)
val_h5_ori="/data/guojiayi_files/yjh_data/bbox/test/"
cnt=0
# for i in os.listdir(train_path):
#     if os.path.isdir(os.path.join(train_path,i)):
#         path1=os.path.join(train_path,i,'images')
#         for i in os.listdir(path1):
#             path2=os.path.join(path1,i)
#             # path3=path2.replace('images','instruments_masks')
#             train_image_list.append(path2)
            

# for i in os.listdir(val_mask_path):
#     if os.path.isdir(os.path.join(val_mask_path,i)):
#         path1=os.path.join(val_mask_path,i,'TypeSegmentation')
#         for i in os.listdir(path1):
#             path3=os.path.join(path1,i)
#             val_mask_list.append(path3)
            
# train_path='/data/guojiayi_files/yjh_data/EndoVis2018/data_h5_bina/test'
# train_h5_ori="/data/guojiayi_files/yjh_data/EndoVis2018/bbox_old/test/"
for j in tqdm(os.listdir(train_path)): #训练集
    bboxes=[]
    # if 'seq_7_frame056' in j:
    #     print(j)
    name=j.replace('.h5','')
    train_h5_path=name+'.npy'
    if os.path.exists(os.path.join(train_h5_ori,train_h5_path)):
        continue
    with h5py.File(os.path.join(train_path,j),'a') as f: 
        train_image=f['image_left'][:]
        train_mask=f['mask'][:]
        # train_mask_new=train_mask[train_mask==1]=0
        # del f['mask']
        
        
    # train_image=cv2.imread(j)
    # train_image=cv2.cvtColor(train_image,cv2.COLOR_BGR2RGB)
    # maskPath=j.replace('images','instruments_masks')
    # train_mask=cv2.imread(maskPath.replace('.jpg','.png'))[:,:,0]
    # train_mask=train_mask*(1/32)
    # train_mask=np.array(train_mask,dtype=np.uint8)
    # name=j.split('/')[-3]+"_"+j.split('/')[-1].replace('.jpg','')

    # mm={}
    # for K in train_mask.flat:
    #     if K not in mm.keys():
    #         mm[K]=1
    #     else:
    #         mm[K]+=1
    # print(mm)
        a=np.unique(train_mask)
        for m in np.unique(train_mask):
            if m!=0:
                bina_mask=np.where(train_mask==m,1,0).astype(np.uint8) #cv2支持处理uint8
                # print(bina_mask.max())
                coords = draw_merged_bboxes(name,m,
                bina_mask, train_image,
                morph_kernel_size=15,  # 增大此值可连接更大间隔
                dbscan_eps=100         # 增大此值可合并更远区域
            )   
                for n in coords:
                    bboxes.append(n)
            print(bboxes)
                
        
        
        bboxes=np.array(bboxes)
        np.save(os.path.join(train_h5_ori,train_h5_path),bboxes)
        
        # with h5py.File(os.path.join(train_h5_ori,train_h5_path),'w') as f:
        #     f.create_dataset("name", data=name)
        #     f.create_dataset("mask", data=train_mask)
        #     f.create_dataset("image", data=train_image)
        #     f.close()

        # cnt+=1
        # print(cnt)
        # f.close()
        # with h5py.File(os.path.join(train_path,j),'w') as f: 
        #     f.create_dataset("mask", data=train_mask_new)
        
# for l in val_mask_list:#/data/guojiayi_files/yjh_data/data_raw/Endovis2017/test label/instrument_dataset_1/TypeSegmentation/  


# for l in val_mask_list:#/data/gjy_files/yjh_data/data_raw/Endovis2017/test label/instrument_dataset_1/TypeSegmentation/
#     bboxes=[]
#     image_path=l.replace('TypeSegmentation','left_frames').replace("test label","test")
#     #/data/gjy_files/yjh_data/data_raw/Endovis2017/test/instrument_dataset_1/left_frames/
#     val_image=cv2.imread(image_path)[h_start: h_start + height, w_start: w_start + width,:]
#     val_mask=cv2.imread(l)[h_start: h_start + height, w_start: w_start + width,0]
#     val_image=cv2.cvtColor(val_image,cv2.COLOR_BGR2RGB)
#     name=l.split('/')[-3]+"_"+l.split('/')[-1].replace('.png','')
#     # val_h5_path=name+'.h5'
    
#     mm={}
#     for K in val_mask.flat:
#         if K not in mm.keys():
#             mm[K]=1
#         else:
#             mm[K]+=1
#     for m in mm.keys():
#         if m!=0:
#             bina_mask=np.where(val_mask==m,1,0).astype(np.uint8) #cv2支持处理uint8
#             # print(bina_mask.max())
#             coords = draw_merged_bboxes(m,
#             bina_mask, val_image,
#             morph_kernel_size=15,  # 增大此值可连接更大间隔
#             dbscan_eps=100         # 增大此值可合并更远区域
#         )   
#             for n in coords:
#                 bboxes.append(n)
#     print(bboxes)
#     # print(mm)
#     val_h5_path=name+'.npy'
#     bboxes=np.array(bboxes)
#     np.save(os.path.join(val_h5_ori,val_h5_path),bboxes)
#     cnt+=1
#     print(cnt)
   
   


   

   
   

        
   
   


   

   
   

        