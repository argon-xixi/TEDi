
import os 
import cv2
import h5py
import numpy as np
path = '/data/yjh_data/data_h5/train/'
output_dir_gt = '/home/gjy/code_yjh/Mask2Former-Simplify-master/gt1/'
output_dir_img = '/home/gjy/code_yjh/Mask2Former-Simplify-master/test1/'

# 创建输出目录
os.makedirs(output_dir_gt, exist_ok=True)
os.makedirs(output_dir_img, exist_ok=True)

# 定义颜色调色板 (RGB格式)
palette = np.array([
    [0, 0, 0],        # class 0 - black
    [255, 0, 0],      # class 1 - red
    [0, 255, 0],      # class 2 - green
    [0, 0, 255],      # class 3 - blue
    [255, 255, 0],    # class 4 - yellow
    [255, 0, 255],    # class 5 - magenta
    [0, 255, 255],    # class 6 - cyan
    [255, 255, 255],  # class 7 - white
], dtype=np.uint8)

for idx, filename in enumerate(os.listdir(path)[:100]):
    path_h5 = os.path.join(path, filename)
    
    with h5py.File(path_h5, 'r') as f:
        image = f['image'][:]  # 假设形状为 (H, W, 3) 或 (H, W)
        mask = f['mask'][:]    # 形状应为 (H, W)
        
        # 转换mask为彩色图像
        color_mask = palette[mask]  # 自动广播为 (H, W, 3)
        
        # 保存图像
        cv2.imwrite(
            os.path.join(output_dir_gt, f'{idx}.png'),
            cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)  # OpenCV需要BGR格式
        )
        
        # 假设image是归一化到[0,1]的浮点数据
        if image.dtype == np.float32:
            
            image = (image ).astype(np.uint8)
            
        cv2.imwrite(
            os.path.join(output_dir_img, f'{idx}.png'),
             cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        )
    f.close()