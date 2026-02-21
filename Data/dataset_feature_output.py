from PIL import Image
import numpy as np
import h5py
from torch.utils import data
import torchvision.transforms.functional as TF
def binary_mask2ori(masks_lists):
    masks=[np.array(mask) for mask in masks_lists]
    fina_masks=np.zeros((1024,1280))
    for n,mask in enumerate(masks):
        fina_masks+=((n+1)*mask)
    fina_masks = fina_masks.astype('int64')
    return fina_masks
    
class SegDataset_feature_output(data.Dataset):
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


    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        if self.cfg.bbox:
            pass  # Add logic for bbox handling if needed

        else:
            namelist=[]
            if self.version == 0:
                path = self.input_paths[index] + ".h5"
            else:
                path = self.input_paths[index] + "_" + str(self.version) + ".h5"
            f = h5py.File(path, "r")
            
            if self.bina_read:
                img_left = f['image_left'][:]
            else:
                img_left = f['image'][:]
            H, W = img_left.shape[:2]
            mask = f['mask'][:].astype('int64')  # Original mask type is int64
            feat = f['feat'][:].transpose((2, 0, 1))
            name = f['name'][()].decode('utf-8')
            f.close()
             # Convert image to PIL for flipping
            img_left_pil = Image.fromarray(img_left)

            # Initialize lists to hold the transformed images and masks
            img_list = [img_left_pil]
            mask_array_list = [mask]
            
            # 生成所有k值的掩码（向量化操作）
            ks = np.arange(1, 8)[:, np.newaxis, np.newaxis]  # 形状变为 (7, 1, 1)
            original_masks = (mask == ks).astype(np.int8)  # 广播比较，生成形状 (7, H, W)
            original_masks = list(original_masks)  # 转换为列表中的二维数组
            original_masks_list=[]
            for k in original_masks:
                mask=Image.fromarray(k)
                original_masks_list.append(mask)

            # Apply vertical flip to the second image
            img_vflip = TF.vflip(img_left_pil)  # Vertical flip
            mask_vflip = [TF.vflip(mask) for mask in original_masks_list]  # Vertical flip
            img_list.append(img_vflip)
            mask_array_list.append(binary_mask2ori(mask_vflip))
            # print(mask_array_list[1].max())
            # Apply horizontal flip to the third image
            img_hflip = TF.hflip(img_left_pil)  # Horizontal flip
            mask_hflip =[TF.hflip(mask) for mask in original_masks_list]   # Horizontal flip
            img_list.append(img_hflip)
            mask_array_list.append(binary_mask2ori(mask_hflip))

            # Apply both vertical and horizontal flips to the fourth image
            img_bothflip = TF.hflip(img_vflip)  # Vertical and Horizontal flip
            mask_bothflip = [TF.hflip(mask) for mask in mask_vflip]  # Vertical and Horizontal flip
            img_list.append(img_bothflip)
            mask_array_list.append(binary_mask2ori(mask_bothflip))

            # Convert all transformed images and masks back to numpy arrays
            img_array_list = [np.array(img) for img in img_list]
           

            # Apply transform_input and transform_target on numpy arrays
            img_array_list = [self.transform_input(img) for img in img_array_list]
            mask_array_list = [self.transform_target(mask) for mask in mask_array_list]

            # Stack images and masks along the first dimension (dim=0)
            img_stacked = np.stack(img_array_list, axis=0)   # Stack images along the first dimension
            mask_stacked = np.stack(mask_array_list, axis=0)  # Stack masks along the first dimension

           
            # Return the stacked images and masks
            mask_stacked = mask_stacked.squeeze(1)
            # print(mask_stacked.max())
            if not self.bina:
                return img_stacked, mask_stacked, feat, name
            else:
                img_right = f['image_right'][:]
                try:
                    flow_l2r = f['flow_l2r'][:]
                except:
                    print(name)
                    print(f.keys())
                img_right = self.transform_input(img_right)
                return (img_stacked, img_right, flow_l2r), mask_stacked, feat, name
