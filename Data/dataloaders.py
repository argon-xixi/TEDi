import numpy as np
import random
import multiprocessing

from sklearn.model_selection import train_test_split
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from Data.dataset_newaug import SegDataset
# from Data.dataset_track_new import SegDataset_track
from Data.dataset_track_new import SegDataset_track
from Data.dataset_feature_output import SegDataset_feature_output
import pandas as pd
import cv2
from time import time
import multiprocessing as mp
import torch
import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, random_split
def seed_worker(worker_id):
    import torch, random, numpy as np
    worker_seed = torch.initial_seed() % 2**32
    # print("worker_seed:",worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def denorm_bchw(x, mean, std):
    """
    x: Tensor [B,C,H,W]
    mean/std: list/tuple 长度 C
    """
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std  = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return x * std + mean 
def check_tensor(img_t, name="img"):
    # img_t: (C,H,W) after normalize
    import torch
    x = img_t.detach().float()
    print(f"== {name} ==")
    print("shape:", tuple(x.shape), "dtype:", x.dtype)
    print("min/max:", float(x.min()), float(x.max()))
    for c in range(x.shape[0]):
        m = float(x[c].mean())
        s = float(x[c].std())
        print(f"  channel {c}: mean={m:.4f}, std={s:.4f}")
def rand_bbox(H, W, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def cutmix_batch(imgs, masks, p=0.5, alpha=1.0):
    """
    imgs:  (B,C,H,W)
    masks: (B,H,W)  long
    """
    a=random.random()
    if a < p:
        return imgs, masks

    B,  H, W = masks.shape
    perm = torch.randperm(B, device=imgs.device)

    lam = np.random.beta(alpha, alpha)
    x1, y1, x2, y2 = rand_bbox(H, W, lam)

    imgs2 = imgs[perm]
    masks2 = masks[perm]+8
    # check_tensor(imgs2[0], "imgs2")

    imgs = imgs.clone()
    masks = masks.clone()
    # print(masks[0].max())
   
    imgs[:, :, y1*4:y2*4, x1*4:x2*4] = imgs2[:, :, y1*4:y2*4, x1*4:x2*4]
    masks[:, y1:y2, x1:x2] = masks2[:, y1:y2, x1:x2]
    # img_denorm=denorm_bchw(imgs, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # cv2.imwrite('cutmix.jpg',img_denorm[0].permute(1,2,0).cpu().numpy()*255)
    # cv2.imwrite('cutmix_mask.jpg',masks[0].cpu().numpy()*255)
   
    
    # 可选：根据真实替换面积修正 lam（有的人会返回 lam 给 loss 用）
    lam_eff = 1.0 - ((x2 - x1) * (y2 - y1) / float(H * W))
    return imgs, masks  # 或 return imgs, masks, lam_eff

# 验证划分结果
def custom_collate_fn(batch):
    img_list=[]
    return_data_list=[]
    feat_sam_list=[]
    name_list=[]
    for i in batch:
        img_list.append(i[0])
        return_data_list.append(i[1])
        feat_sam_list.append(i[2])
        name_list.append(i[3])
    
    return img_list,return_data_list,feat_sam_list,name_list

def get_dataloaders(train_paths,test_paths, cfg,batch_size, num_workers,version,bina):
    g = torch.Generator()
    g.manual_seed(cfg.seed)  # 用 config 里的 seed
    shuffled_list = train_paths.copy()
    transform_input4train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((512, 512), antialias=True),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((512, 512), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


    # transform_target = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Resize((352, 352)), transforms.Grayscale()]
    # )

    transform_target = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((128, 128), interpolation=F._interpolation_modes_from_int(0))] #最近邻插值方法
    )
    if cfg.output_feature:
        train_dataset = SegDataset_feature_output(
            input_paths=train_paths,
            version=version,
            transform_input=transform_input4train,
            transform_target=transform_target,
            cfg=cfg,
            bina=bina,
            bbox=cfg.bbox,
            bina_read=True,
            hflip=True,
            vflip=True,
           
        )
        test_dataset = SegDataset_feature_output(
            input_paths=test_paths,
            version=0,
            transform_input=transform_input4test,
            transform_target=transform_target,
            cfg=cfg,
            bina=bina,
            bbox=cfg.bbox,
            bina_read=True,

        )
    else:
        if cfg.tracker==True or cfg.refiner==True:
            train_dataset = SegDataset_track(
                input_paths=train_paths,
                version=version,
                transform_input=transform_input4train,
                transform_target=transform_target,
                cfg=cfg,
                bina=bina,
                bbox=cfg.bbox,
                bina_read=True,
                hflip=True,
                vflip=True,
                ColorJitter=False,
                RandomRotation=False,
                RandomCrop=False,
                reverse=False,
            )
            

            test_dataset = SegDataset_track(
            input_paths=test_paths,
                version=0,
                transform_input=transform_input4test,
                transform_target=transform_target,
                cfg=cfg,
                bina=bina,
                bbox=cfg.bbox,
                bina_read=True,
                reverse=False,
            )

        else:
            # train_dataset = SegDataset(
            #     input_paths=train_paths,
            #     version=version,
            #     transform_input=transform_input4train,
            #     transform_target=transform_target,
            #     cfg=cfg,
            #     bina=bina,
            #     bbox=cfg.bbox,
            #     bina_read=True,
            #     hflip=False,
            #     vflip=False,
            #     ColorJitter=False,
            #     RandomRotation=False,
            #     RandomCrop=False,
            # )
            train_dataset = SegDataset(
                input_paths=train_paths,
                version=version,
                transform_input=transform_input4train,
                transform_target=transform_target,
                cfg=cfg,
                bina=bina,
                bbox=cfg.bbox,
                bina_read=True,
                hflip=True,
                vflip=True,
                ColorJitter=False,
                RandomRotation=False,
                RandomCrop=False,
            )
            
            test_dataset = SegDataset(
                input_paths=test_paths,
                version=0,
                transform_input=transform_input4test,
                transform_target=transform_target,
                cfg=cfg,
                bina=bina,
                bbox=cfg.bbox,
                bina_read=True,
            )

    if cfg.DDP==True:
        train_datasampler = DistributedSampler(train_dataset,shuffle=True)
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            # num_workers=multiprocessing.Pool()._processes,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            sampler=train_datasampler,
            
        )
        test_datasampler = DistributedSampler(test_dataset)
        test_dataloader = data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            # num_workers=multiprocessing.Pool()._processes,
            num_workers=num_workers,
            sampler=test_datasampler,
            worker_init_fn=seed_worker,
            generator=g,
            
        )
    else:
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            # num_workers=multiprocessing.Pool()._processes,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            # prefetch_factor=2,
            worker_init_fn=seed_worker,
            generator=g,
            
        )
        
        test_dataloader = data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            # num_workers=multiprocessing.Pool()._processes,
            num_workers=num_workers,
            # prefetch_factor=2,
            worker_init_fn=seed_worker,
            generator=g,
            
        )
        

        
    return train_dataloader, test_dataloader


