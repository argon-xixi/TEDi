# TEDi

Official code repository for **TEDi: Temporal Memory-Enhanced and Denoising Transformer for Surgical Instrument Segmentation**, a MICCAI 2026 paper.

TEDi performs surgical instrument segmentation in endoscopic video by combining a denoising query tracker with a temporal memory bank. The memory-enhanced representation uses preceding frames to improve temporal consistency and robustness in challenging surgical scenes.

![TEDi architecture](fig/Fig2.1%E5%8E%9F%E5%9B%BEHD.png)

## Highlights

- Temporal query memory for video-level instrument segmentation.
- Denoising transformer tracker for stable query propagation.
- Mask2Former segmentation head with a Swin Transformer backbone.
- EndoVis 2017 four-fold evaluation and EndoVis 2018 experiments.
- Training, evaluation, visualization, and efficiency measurement in one entry point.

## Repository Structure

```text
TEDi/
|-- configs/tedi.yaml                 # Main paper configuration
|-- Data/                             # EndoVis temporal clip loader
|-- fig/Fig2.1原图HD.png              # TEDi architecture
|-- modeling/
|   |-- backbone/                     # Swin Transformer backbone
|   |-- dvis/                         # Denoising temporal query tracker
|   |-- memory/                       # Temporal query memory
|   |-- pixel_decoder/                # Multi-scale deformable pixel decoder
|   `-- transformer_decoder/          # Mask2Former query decoder
|-- utils/                            # Losses, matching, and optimization
|-- yjh/                              # EndoVis metrics and visualization
|-- maskformer_train_track.py         # TEDi training/evaluation loop
|-- train.py                          # Public entry point
`-- requirements.txt
```

## Installation

The original experiments used PyTorch 1.10 and CUDA 11.1. Create an environment compatible with your CUDA installation:

```bash
git clone https://github.com/argon-xixi/TEDi.git
cd TEDi

conda create -n tedi python=3.8 -y
conda activate tedi
pip install -r requirements.txt
```

Detectron2 is required by the TEDi data augmentation pipeline but is not included in `requirements.txt`. Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) separately for the selected PyTorch/CUDA combination, then compile the multi-scale deformable attention operator:

```bash
cd modeling/pixel_decoder/ops
sh make.sh
cd ../../..
```

## Data Preparation

Download EndoVis 2017 or EndoVis 2018 from the official challenge pages and follow their licenses. TEDi reads one HDF5 file per frame. Each file must contain:

```text
image_left    RGB image, H x W x 3
mask          integer segmentation mask, H x W
name          UTF-8 frame identifier
```

Frame files must use the naming convention `seq_<id>_frame<id>.h5`. Neighboring frames are assembled into a clip according to `NUM_MEM_FRAMES + NUM_TRACK_FRAMES` in `configs/tedi.yaml`.

Expected EndoVis 2018 layout:

```text
data/endovis2018/
|-- train/
|   `-- seq_1_frame000.h5
`-- test/
    `-- seq_2_frame000.h5
```

For EndoVis 2017, place all HDF5 files in one directory. `train.py` creates the four sequence-level folds used by the experiments.

## Training

EndoVis 2018:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset EndoVis2018 \
  --data-root data/endovis2018 \
  --output-dir outputs/endovis2018 \
  --task tedi_endovis2018
```

EndoVis 2017 fold 0:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset EndoVis2017 \
  --data-root data/endovis2017 \
  --fold 0 \
  --output-dir outputs/endovis2017 \
  --task tedi_endovis2017
```

Use `--pretrained /path/to/checkpoint.pth` to initialize training. Checkpoints and TensorBoard logs are written below `--output-dir`.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset EndoVis2018 \
  --data-root data/endovis2018 \
  --checkpoint /path/to/tedi.pth \
  --infer-only \
  --output-dir outputs/evaluation
```

Evaluation reports EndoVis segmentation metrics and writes prediction overlays below `--output-dir/visualizations/`.

## Efficiency Analysis

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --efficiency \
  --checkpoint /path/to/tedi.pth \
  --eff-T 6 \
  --eff-H 512 \
  --eff-W 512
```

This reports latency with and without temporal memory, peak GPU memory, parameter count, and approximate FLOPs. Custom CUDA operations may not be fully counted by `fvcore`.

## Citation

If TEDi is useful in your research, please cite the paper. The BibTeX record will be added when the MICCAI 2026 publication metadata is available.

## Acknowledgements

This project builds upon [Mask2Former](https://github.com/facebookresearch/Mask2Former), [Mask2Former-Simplify](https://github.com/zzubqh/Mask2Former-Simplify), and temporal query tracking components inspired by [DVIS](https://github.com/zhang-tao-whu/DVIS).
