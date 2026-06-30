import numpy as np
import random
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from Data.dataset_track_new import SegDataset_track
import torch
def seed_worker(worker_id):
    import torch, random, numpy as np
    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)

def denorm_bchw(x, mean, std):

    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std  = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return x * std + mean
def check_tensor(img_t, name="img"):

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

    a=random.random()
    if a < p:
        return imgs, masks

    B,  H, W = masks.shape
    perm = torch.randperm(B, device=imgs.device)

    lam = np.random.beta(alpha, alpha)
    x1, y1, x2, y2 = rand_bbox(H, W, lam)

    imgs2 = imgs[perm]
    masks2 = masks[perm]+8

    imgs = imgs.clone()
    masks = masks.clone()

    imgs[:, :, y1*4:y2*4, x1*4:x2*4] = imgs2[:, :, y1*4:y2*4, x1*4:x2*4]
    masks[:, y1:y2, x1:x2] = masks2[:, y1:y2, x1:x2]

    lam_eff = 1.0 - ((x2 - x1) * (y2 - y1) / float(H * W))
    return imgs, masks

def custom_collate_fn(batch):
    img_list=[]
    return_data_list=[]
    name_list=[]
    for i in batch:
        img_list.append(i[0])
        return_data_list.append(i[1])
        name_list.append(i[3])

    return img_list, return_data_list, name_list

def get_dataloaders(train_paths,test_paths, cfg,batch_size, num_workers,version,bina):
    g = torch.Generator()
    g.manual_seed(cfg.seed)
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

    transform_target = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((128, 128), interpolation=F._interpolation_modes_from_int(0))]
    )
    train_dataset = SegDataset_track(
        input_paths=train_paths,
        version=version,
        transform_input=transform_input4train,
        transform_target=transform_target,
        cfg=cfg,
        bina=bina,
        bbox=False,
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
        bbox=False,
        bina_read=True,
        reverse=False,
    )

    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_dataloader, test_dataloader
