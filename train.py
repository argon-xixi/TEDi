

import argparse
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from fvcore.common.config import CfgNode as FvCfgNode

from configs.config import Config

FOLD_SEQUENCES = {
    0: (1, 3),
    1: (2, 5),
    2: (4, 8),
    3: (6, 7),
}

# Keep the data-quality exclusions used by the paper experiments reproducible.
EXCLUDED_TRAIN_FRAMES = {
    "seq_2_frame004",
    "seq_4_frame137",
    "seq_6_frame104",
    "seq_6_frame138",
    "seq_10_frame072",
    "seq_10_frame105",
    "seq_15_frame139",
    "seq_16_frame030",
    "seq_16_frame068",
}
EXCLUDED_TEST_FRAMES = {
    "seq_2_frame107",
    "seq_2_frame108",
    "seq_2_frame109",
    "seq_2_frame110",
    "seq_2_frame111",
    "seq_2_frame112",
    "seq_2_frame113",
    "seq_2_frame114",
    "seq_2_frame115",
    "seq_2_frame116",
    "seq_2_frame117",
    "seq_2_frame118",
    "seq_2_frame119",
    "seq_9_frame050",
    "seq_9_frame057",
    "seq_9_frame070",
    "seq_9_frame071",
    "seq_9_frame072",
    "seq_9_frame073",
    "seq_9_frame074",
    "seq_9_frame075",
    "seq_9_frame076",
    "seq_9_frame140",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train TEDi on EndoVis data")
    parser.add_argument("--config", default="configs/tedi.yaml")
    parser.add_argument("--dataset", default="EndoVis2018", choices=("EndoVis2017", "EndoVis2018"))
    parser.add_argument(
        "--data-root",
        "--root",
        dest="data_root",
        default="data/endovis2018",
        help="EndoVis 2017 HDF5 directory, or EndoVis 2018 root containing train/ and test/",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--pretrained", default=None, help="Optional initialization checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Optional evaluation checkpoint")
    parser.add_argument("--fold", type=int, default=0, choices=tuple(FOLD_SEQUENCES))
    parser.add_argument("--task", default="tedi")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=6e-5, dest="lr")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument("--infer-only", action="store_true", dest="inferonly")

    parser.add_argument("--efficiency", action="store_true")
    parser.add_argument("--eff-bs", type=int, default=1)
    parser.add_argument("--eff-T", type=int, default=6)
    parser.add_argument("--eff-H", type=int, default=512)
    parser.add_argument("--eff-W", type=int, default=512)
    parser.add_argument("--eff-warmup", type=int, default=20)
    parser.add_argument("--eff-iters", type=int, default=50)
    parser.add_argument("--eff-amp", action="store_true")
    parser.add_argument("--eff-device", default="cuda", choices=("cuda", "cpu"))
    return parser.parse_args()

def load_config(args):
    raw_cfg = FvCfgNode.load_yaml_with_base(args.config, allow_unsafe=True)
    cfg = Config(raw_cfg)

    cfg.dataset = args.dataset
    cfg.root = args.data_root
    cfg.output_dir = args.output_dir
    cfg.fold = args.fold
    cfg.task = args.task
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.workers = args.workers
    cfg.lr = args.lr
    cfg.seed = args.seed
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    cfg.ngpus = torch.cuda.device_count()
    cfg.inferonly = args.inferonly
    cfg.DDP = False

    cfg.tracker = True
    cfg.refiner = False
    cfg.sam = False
    cfg.bina = False
    cfg.bbox = False
    cfg.output_feature = False
    cfg.resume = False

    cfg.MODEL.TRACKER = FvCfgNode()
    cfg.MODEL.TRACKER.DECODER_LAYERS = 6
    cfg.MODEL.REFINER = FvCfgNode()
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.LR = args.lr
    cfg.TRAIN.CKPT_DIR = str(Path(args.output_dir) / "checkpoints")
    cfg.TRAIN.LOG_DIR = str(Path(args.output_dir) / "logs")
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.pretrained is not None:
        cfg.MODEL.PRETRAINED_WEIGHTS = args.pretrained
    if args.checkpoint is not None:
        cfg.MODEL.INFER_PRETRAINED_WEIGHTS = args.checkpoint

    for name in ("efficiency", "eff_bs", "eff_T", "eff_H", "eff_W", "eff_warmup", "eff_iters", "eff_amp", "eff_device"):
        setattr(cfg, name, getattr(args, name))
    return cfg

def _h5_stems(directory):
    return [str(path.with_suffix("")) for path in sorted(Path(directory).glob("*.h5"))]

def collect_dataset_paths(cfg):
    root = Path(cfg.root).expanduser()
    if cfg.dataset == "EndoVis2017":
        paths = _h5_stems(root)
        test_sequences = set(FOLD_SEQUENCES[cfg.fold])
        train_paths, test_paths = [], []
        for path in paths:
            sequence = int(Path(path).name.split("_")[1])
            (test_paths if sequence in test_sequences else train_paths).append(path)
    else:
        train_paths = [p for p in _h5_stems(root / "train") if Path(p).name not in EXCLUDED_TRAIN_FRAMES]
        test_paths = [p for p in _h5_stems(root / "test") if Path(p).name not in EXCLUDED_TEST_FRAMES]

    if not train_paths or not test_paths:
        raise FileNotFoundError(
            f"No complete train/test split found under {root}. See README.md for the expected HDF5 layout."
        )
    print(f"Dataset: {cfg.dataset}; train={len(train_paths)}; test={len(test_paths)}")
    return train_paths, test_paths

def run_training(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("TEDi training requires a CUDA-capable GPU.")

    from tensorboardX import SummaryWriter
    from maskformer_train_track import MaskFormer_track

    train_paths, test_paths = collect_dataset_paths(cfg)
    log_dir = Path(cfg.TRAIN.LOG_DIR) / cfg.task / str(cfg.fold)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir), flush_secs=15)
    try:
        MaskFormer_track(cfg).train(train_paths, test_paths, cfg.epochs, writer, cfg.fold)
    finally:
        writer.close()

@torch.no_grad()
def run_efficiency_analysis(cfg):
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    from modeling.MaskFormerModel_track_memorybank import MaskFormerModel_track_memorybank

    device = torch.device(cfg.eff_device if torch.cuda.is_available() else "cpu")
    model = MaskFormerModel_track_memorybank(cfg).to(device).eval()
    weight_path = cfg.MODEL.INFER_PRETRAINED_WEIGHTS or cfg.MODEL.PRETRAINED_WEIGHTS
    if weight_path and Path(weight_path).is_file():
        state = torch.load(weight_path, map_location="cpu")
        state = state.get("model", state)
        state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
        model.load_state_dict(state, strict=False)

    inputs = torch.randn(cfg.eff_bs, cfg.eff_T, 3, cfg.eff_H, cfg.eff_W, device=device)

    def forward(use_memory):
        if cfg.eff_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model.forward_infer(inputs, use_memory_bank=use_memory)
        return model.forward_infer(inputs, use_memory_bank=use_memory)

    def benchmark(use_memory):
        for _ in range(cfg.eff_warmup):
            forward(use_memory)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(cfg.eff_iters):
            start = time.perf_counter()
            forward(use_memory)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
        return {
            "mean_ms": statistics.mean(timings),
            "p50_ms": statistics.median(timings),
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
        }

    print("with memory:", benchmark(True))
    print("without memory:", benchmark(False))
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    class InferenceWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, clips):
            return self.wrapped.forward_infer(clips, use_memory_bank=True)

    try:
        flops = FlopCountAnalysis(InferenceWrapper(model), inputs)
        print(f"FLOPs: {flops.total() / 1e9:.3f}G")
        print(flop_count_table(flops, max_depth=3))
    except Exception as exc:
        print(f"FLOP analysis unavailable for one or more custom operators: {exc}")

def main():
    args = parse_args()
    cfg = load_config(args)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if cfg.efficiency:
        run_efficiency_analysis(cfg)
    else:
        run_training(cfg)

if __name__ == "__main__":
    main()
