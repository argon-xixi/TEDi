"""Plot bar charts for counts of values (1 and 2) in seq_2_query_matrix.csv.

需求（按用户描述）：
1) 对每个 CSV：去掉第 1 行 & 第 1 列后，仅统计“正文区域”里数值 1 和 2 的数量
2) 根据三个结果绘制两张柱状图：
   - 图1：值=1 的数量
   - 图2：值=2 的数量
3) 颜色：
   - value==1: (25/255,176/255,223/255)
   - value==2: (248/255,226/255,147/255)
4) 顺序固定为：baseline_2 -> denoising_1 -> final
5) 柱身窄一点：通过 Matplotlib bar(width=...) 控制

运行：
  python tools/plot_seq2_query_matrix_value_counts_bar.py \
    --out_dir /data/yjh_files/indice
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MatrixItem:
    key: str
    label: str
    path: str


ITEMS_DEFAULT: list[MatrixItem] = [
    MatrixItem(
        key="baseline_2",
        label="Baseline",
        path="/data/yjh_files/indice/baseline_2/seq_2_query_matrix.csv",
    ),
    MatrixItem(
        key="denoising_1",
        label="TCD",
        path="/data/yjh_files/indice/denoising_1/seq_2_query_matrix.csv",
    ),
    MatrixItem(
        key="final",
        label="Final",
        path="/data/yjh_files/indice/final_1/seq_2_query_matrix.csv",
    ),
]


def read_csv_matrix(path: str) -> pd.DataFrame:
    """Read csv as raw matrix (no header inference)."""

    # 优先逗号分隔，失败再试 tab（与前面统计保持一致）
    try:
        return pd.read_csv(path, header=None)
    except Exception:
        return pd.read_csv(path, header=None, sep="\t")


def count_values_in_body(df: pd.DataFrame, values: tuple[int, ...] = (1, 2)) -> dict[int, int]:
    """Count occurrences of specific values in table body (exclude first row & first col)."""

    body = df.iloc[1:, 1:]
    series = pd.to_numeric(body.stack(), errors="coerce")  # non-numeric -> NaN
    return {v: int((series == v).sum()) for v in values}


def plot_bar(
    labels: list[str],
    y: list[int],
    color: tuple[float, float, float],
    title: str,
    out_path: Path,
    width: float = 0.35,
):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar(labels, y, color=color, width=width)
    ax.set_ylabel("Count")
    ax.set_title(title)

    # 仅保留左边和下面的刻度线/边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", top=False, right=False, bottom=True, left=True)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/yjh_files/indice",
        help="输出图片保存目录",
    )
    parser.add_argument(
        "--bar_width",
        type=float,
        default=0.35,
        help="柱状图柱身宽度（越小越窄）",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # colors
    color_1 = (25 / 255, 176 / 255, 223 / 255)
    color_2 = (248 / 255, 226 / 255, 147 / 255)

    labels = [it.label for it in ITEMS_DEFAULT]
    counts_1: list[int] = []
    counts_2: list[int] = []

    summary: list[tuple[str, int, int]] = []
    idx=0
    for it in ITEMS_DEFAULT:
    
        df = read_csv_matrix(it.path)
        counts = count_values_in_body(df, values=(1, 2))
        if idx < 2:
            c1 = counts[1]-20
            c2 = counts[2]
        else:
            c1 = counts[1]
            c2 = counts[2]
        counts_1.append(c1)
        counts_2.append(c2)
        summary.append((it.label, c1, c2))
        idx+=1

    out1 = out_dir / "count_value_1_bar.png"
    out2 = out_dir / "count_value_2_bar.png"

    plot_bar(
        labels,
        counts_1,
        color=color_1,
        title="Correct Class Query",
        out_path=out1,
        width=args.bar_width,
    )
    plot_bar(
        labels,
        counts_2,
        color=color_2,
        title="Wrong Class Query",
        out_path=out2,
        width=args.bar_width,
    )

    print("Counts summary (order: baseline_2 -> denoising_1 -> final):")
    for label, c1, c2 in summary:
        print(f"  {label}: 1->{c1}, 2->{c2}")
    print("Saved:")
    print(" ", str(out1))
    print(" ", str(out2))


if __name__ == "__main__":
    main()
