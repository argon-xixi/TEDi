
# import os
# import re
# import argparse
# import h5py
# import numpy as np
# from PIL import Image
# from tqdm import tqdm


# IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
# MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# def read_image_rgb(path: str) -> np.ndarray:
#     img = Image.open(path).convert("RGB")
#     return np.array(img, dtype=np.uint8)


# def read_mask_array(path: str) -> np.ndarray:
#     m = Image.open(path)
#     arr = np.array(m)
#     if arr.ndim == 3:
#         arr = arr[..., 0]
#     return arr


# def normalize_mask_to_0_7(mask: np.ndarray) -> np.ndarray:
#     mask = mask.astype(np.int64)
#     uniq = np.unique(mask)
#     maxv = int(uniq.max())

#     # 已经是0..7
#     if maxv <= 7 and int(uniq.min()) >= 0:
#         return np.clip(mask, 0, 7).astype(np.int64)

#     # 典型：0,32,64,...,224
#     if np.all((uniq % 32) == 0):
#         out = mask // 32
#         return np.clip(out, 0, 7).astype(np.int64)

#     # unique<=8：离散映射（保持0->0）
#     if len(uniq) <= 8:
#         uniq_sorted = sorted(int(x) for x in uniq)
#         if 0 in uniq_sorted:
#             uniq_sorted.remove(0)
#             uniq_sorted = [0] + uniq_sorted
#         mapping = {val: idx for idx, val in enumerate(uniq_sorted)}
#         out = np.vectorize(mapping.get)(mask)
#         return np.clip(out, 0, 7).astype(np.int64)

#     # 否则按比例缩放到0..7
#     out = np.rint(mask / 255.0 * 7.0).astype(np.int64)
#     return np.clip(out, 0, 7).astype(np.int64)


# def extract_frame_id(filename: str) -> str:
#     # frame219.jpg -> 219；frame0001.jpg -> 0001
#     stem = os.path.splitext(filename)[0]
#     m = re.search(r"(\d+)$", stem)
#     return m.group(1) if m else stem


# def find_mask_for_image(masks_dir: str, img_filename: str) -> str | None:
#     """
#     支持 images 是 .jpg，而 mask 是 .png 的情况。
#     用 stem(比如 frame219) 在 masks_dir 里找同stem的文件。
#     """
#     stem = os.path.splitext(img_filename)[0]

#     # 优先常见：png
#     preferred = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
#     for ext in preferred:
#         cand = os.path.join(masks_dir, stem + ext)
#         if os.path.exists(cand):
#             return cand

#     # 最后兜底：遍历目录找同stem
#     for fn in os.listdir(masks_dir):
#         if os.path.splitext(fn)[0] == stem and fn.lower().endswith(MASK_EXTS):
#             return os.path.join(masks_dir, fn)

#     return None


# def list_image_files(images_dir: str):
#     files = [f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)]
#     files.sort()
#     return files


# def process_sequence(seq_path: str, seq_id: int, out_dir: str) -> int:
#     images_dir = os.path.join(seq_path, "images")
#     masks_dir = os.path.join(seq_path, "instruments_masks")

#     if not os.path.isdir(images_dir):
#         print(f"[SKIP] no images dir: {images_dir}")
#         return 0
#     if not os.path.isdir(masks_dir):
#         print(f"[SKIP] no instruments_masks dir: {masks_dir}")
#         return 0

#     img_files = list_image_files(images_dir)
#     if not img_files:
#         print(f"[SKIP] empty images: {images_dir}")
#         return 0

#     os.makedirs(out_dir, exist_ok=True)

#     n_ok = 0
#     n_miss = 0

#     for img_fn in tqdm(img_files, desc=f"instrument_dataset_{seq_id}", leave=False):
#         img_path = os.path.join(images_dir, img_fn)
#         mask_path = find_mask_for_image(masks_dir, img_fn)
#         if mask_path is None:
#             n_miss += 1
#             continue

#         img = read_image_rgb(img_path)
#         mask_raw = read_mask_array(mask_path)
#         mask = normalize_mask_to_0_7(mask_raw)

#         frame_id = extract_frame_id(img_fn)
#         out_name = f"seq_{seq_id}_frame{frame_id}.h5"
#         out_path = os.path.join(out_dir, out_name)
#         name_str = f"seq_{seq_id}_frame{frame_id}"

#         with h5py.File(out_path, "w") as f:
#             f.create_dataset("image_left", data=img, compression="gzip", compression_opts=4)
#             f.create_dataset("mask", data=mask.astype(np.int64), compression="gzip", compression_opts=4)
#             f.create_dataset("name", data=np.string_(name_str))

#         n_ok += 1

#     print(f"[DONE] instrument_dataset_{seq_id}: written={n_ok}, mask_missing={n_miss}")
#     return n_ok


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--root", required=True, help=".../cropped_train_left")
#     ap.add_argument("--out", required=True, help="output directory for per-frame h5")
#     args = ap.parse_args()

#     root = args.root
#     out_root = args.out
#     os.makedirs(out_root, exist_ok=True)

#     # 遍历 instrument_dataset_*
#     seq_dirs = []
#     for d in os.listdir(root):
#         full = os.path.join(root, d)
#         if os.path.isdir(full) and d.startswith("instrument_dataset_"):
#             try:
#                 seq_id = int(d.split("_")[-1])
#             except:
#                 continue
#             seq_dirs.append((seq_id, full))
#     seq_dirs.sort(key=lambda x: x[0])

#     total = 0
#     for seq_id, seq_path in seq_dirs:
#         total += process_sequence(seq_path, seq_id, out_root)

#     print(f"All done. Total frames written: {total}")


# if __name__ == "__main__":
#     main()

# '''
# /data1/yuanjiahong_files/miniconda3/envs/yjh/bin/python /home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/Data/prepare_dataset.py \
#   --root /data1/yuanjiahong_files/bishe/EndoVis2017/data_raw/cropped_train_left \
#   --out  /data1/yuanjiahong_files/bishe/EndoVis2017/data_h5/train
#   '''
# import os
# import h5py
# import torch
# from collections import defaultdict

# h5_dir = "/data/yjh_files/data/Endovis2017/data_h5/train"

# # 每个 sequence 一套统计
# pixel_count_per_class = defaultdict(lambda: defaultdict(int))
# image_count_per_class = defaultdict(lambda: defaultdict(int))

# for fname in os.listdir(h5_dir):
#     if not fname.endswith(".h5"):
#         continue

#     # ===== 从文件名解析 sequence =====
#     # seq_2_frame123.h5 → seq = 2
#     seq = int(fname.split("_")[1])

#     h5_path = os.path.join(h5_dir, fname)

#     with h5py.File(h5_path, "r") as f:
#         mask = f["mask"][:]   # [H, W]

#     mask_t = torch.from_numpy(mask).long()

#     # 当前图像中出现的标签
#     unique_labels = torch.unique(mask_t)

#     for c in unique_labels.tolist():
#         pixel_count = (mask_t == c).sum().item()

#         pixel_count_per_class[seq][c] += pixel_count
#         image_count_per_class[seq][c] += 1
        
# for seq in sorted(pixel_count_per_class.keys()):
#     print(f"\n===== Sequence {seq} label statistics =====")
#     for c in sorted(pixel_count_per_class[seq].keys()):
#         print(
#             f"Label {c}: "
#             f"pixels = {pixel_count_per_class[seq][c]}, "
#             f"images = {image_count_per_class[seq][c]}"
#         )
#     print("==========================================")
# import os
# import re
# import h5py
# import torch
# from collections import defaultdict

# h5_dir = "/data/yjh_files/data/Endovis2017/data_h5/train"

# fold_seq = {
#     0: [1, 3],
#     1: [2, 5],
#     2: [4, 8],
#     3: [6, 7]
# }

# def parse_seq_and_imgid(fname: str):
#     """
#     约定：文件名里第一个数字=序列号，最后一个数字的末3位=图片号
#     例：
#       seq_2_frame123.h5 -> seq=2, imgid=123
#     """
#     nums = re.findall(r"\d+", fname)
#     if len(nums) < 1:
#         raise ValueError(f"Cannot parse seq from filename: {fname}")
#     seq = int(nums[0])
#     imgid = int(nums[-1][-3:]) if len(nums) >= 2 else -1
#     return seq, imgid

# # 收集所有出现过的 sequence
# all_seqs = set()
# all_files = []

# for fname in os.listdir(h5_dir):
#     if not fname.endswith(".h5"):
#         continue
#     seq, imgid = parse_seq_and_imgid(fname)
#     all_seqs.add(seq)
#     all_files.append((fname, seq, imgid))

# all_seqs = sorted(all_seqs)

# # 统计容器：stats[fold]['train'/'test'][label] -> pixels/images
# pixel_stats = defaultdict(lambda: {"train": defaultdict(int), "test": defaultdict(int)})
# image_stats = defaultdict(lambda: {"train": defaultdict(int), "test": defaultdict(int)})

# for fold, test_seqs in fold_seq.items():
#     test_seqs = set(test_seqs)
#     train_seqs = set(all_seqs) - test_seqs

#     for fname, seq, imgid in all_files:
#         if seq in test_seqs:
#             split = "test"
#         elif seq in train_seqs:
#             split = "train"
#         else:
#             # 正常不会发生
#             continue

#         h5_path = os.path.join(h5_dir, fname)
#         with h5py.File(h5_path, "r") as f:
#             mask = f["mask"][:]  # [H, W]

#         mask_t = torch.from_numpy(mask).long().view(-1)

#         # 每个标签的像素数（更快）
#         bc = torch.bincount(mask_t)  # index=label, value=pixel_count
#         present = torch.nonzero(bc, as_tuple=False).view(-1).tolist()

#         for c in present:
#             pixel_stats[fold][split][c] += int(bc[c].item())
#             image_stats[fold][split][c] += 1

# # 输出结果
# def print_split_stats(fold, split, seqs):
#     print(f"\n---- {split.upper()} (seq={sorted(list(seqs))}) ----")
#     labels = sorted(pixel_stats[fold][split].keys())
#     for c in labels:
#         print(f"Label {c}: pixels = {pixel_stats[fold][split][c]}, images = {image_stats[fold][split][c]}")

# for fold in sorted(fold_seq.keys()):
#     test_seqs = set(fold_seq[fold])
#     train_seqs = set(all_seqs) - test_seqs

#     print(f"\n================ Fold {fold} ================")
#     print_split_stats(fold, "train", train_seqs)
#     print_split_stats(fold, "test", test_seqs)
#     print("=============================================")


import os
import h5py
import numpy as np
from tqdm import tqdm

src_dir = "/data/yjh_files/data/Endovis2018/data_h5/test_1"
dst_dir = "/data/yjh_files/data/Endovis2018/data_h5/test"

os.makedirs(dst_dir, exist_ok=True)

for fname in tqdm(os.listdir(src_dir)):
    if not fname.endswith(".h5"):
        continue

    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)

    with h5py.File(src_path, "r") as f:
        img_left = f["image_left"][:]   # 读取时自动解压
        mask = f["mask"][:]
        name = f["name"][()]

    with h5py.File(dst_path, "w") as out:
        # 不指定 compression 即为不压缩
        out.create_dataset("image_left", data=img_left)
        out.create_dataset("mask", data=mask)
        out.create_dataset("name", data=name)

    print(f"Saved uncompressed: {dst_path}")
