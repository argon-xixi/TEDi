"""Plot seq_2 query matrix as a stitched-rectangle (heatmap-like) figure.

Input CSV:
  /data/yjh_files/indice/final_1/seq_2_query_matrix.csv

CSV format produced by `tools/eval_endovis2018_seq2_query_matrix.py`:
  header: query, <img1>, <img2>, ...
  each row: query_id(0..9), values...

Visualization requirements:
  - 10 rows (query 0..9)
  - columns = number of images (from 2nd column to end)
  - color mapping: 0=white, 1=blue, 2=red
  - x-axis ticks every 30 columns
  - save to png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Use non-interactive backend for servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv_matrix(csv_path: Path) -> tuple[list[str], np.ndarray]:
    lines = csv_path.read_text().strip().splitlines()
    if not lines:
        raise ValueError(f"Empty CSV: {csv_path}")

    header = [x.strip() for x in lines[0].split(",")]
    if len(header) < 2 or header[0] != "query":
        raise ValueError(f"Unexpected header in {csv_path}: {header[:5]}")
    col_names = header[1:]

    rows = []
    q_ids = []
    for ln in lines[1:]:
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < 2:
            continue
        q_ids.append(int(parts[0]))
        rows.append([int(v) for v in parts[1:]])

    mat = np.array(rows, dtype=np.int64)
    if mat.shape[0] != 10:
        # still allow but warn via exception message? keep strict to avoid silent mismatch
        raise ValueError(f"Expected 10 rows (query 0..9), got {mat.shape[0]} rows")
    # Optionally ensure queries are 0..9
    if sorted(q_ids) != list(range(10)):
        raise ValueError(f"Expected query ids 0..9, got: {q_ids}")
    return col_names, mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default="/data/yjh_files/indice/final_1/seq_2_query_matrix.csv",
        help="Input CSV path",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="/data/yjh_files/indice/final_1/seq_2_query_matrix.png",
        help="Output PNG path",
    )
    ap.add_argument("--tick_step", type=int, default=30, help="x-axis tick step")
    ap.add_argument("--cell_h", type=float, default=0.01, help="cell height in inches scaling")
    ap.add_argument("--cell_w", type=float, default=0.06, help="cell width in inches scaling")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    col_names, mat = read_csv_matrix(csv_path)
    n_rows, n_cols = mat.shape

    # Build RGB image: white/green/yellow
    img = np.zeros((n_rows, n_cols, 3), dtype=np.float32)
    img[:] = (1.0, 1.0, 1.0)  # 0 -> pure white background
    img[mat == 1] = (25/255,176/255,223/255) 
    img[mat == 2] = (248/255,226/255,147/255)  

    # Figure sizing: scale by number of columns/rows
    fig_w = max(6.0, n_cols * args.cell_w)
    fig_h = max(2.0, n_rows * args.cell_h)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    ax.imshow(img, aspect="auto", interpolation="nearest")

    # Y axis: 10 rows
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([f"ID:{i}" for i in range(n_rows)])

    # X axis ticks every 30
    step = max(1, int(args.tick_step))
    xticks = np.arange(0, n_cols, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_xlabel("frame index")

    # Minor grid for better block feeling (optional, light)
    # ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.1, alpha=0.8)
    # ax.grid(False)
    ax.tick_params(which="minor", bottom=False, left=False)

    # remove spines on top/right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", which="both", left=False)
    ax.set_yticklabels([])
    ax.spines["left"].set_visible(False)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"[DONE] saved figure: {out_path} shape={mat.shape}")


if __name__ == "__main__":
    main()
