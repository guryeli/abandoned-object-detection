"""
Generates figures from benchmark results.

Usage:
    python -m eval.plot_results --csv benchmark_results.csv --out eval/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; write to a file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Color palette aligned with the bag-state colors (matches visualizer.py)
COLOR_TP = "#2ca02c"   # green
COLOR_FN = "#d62728"   # red
COLOR_FP = "#ff7f0e"   # orange
COLOR_TN = "#7f7f7f"   # gray
COLOR_BAR = "#1f77b4"  # blue
COLOR_REF = "#888"


def _natural_video_key(name: str) -> int:
    """So that video1, video2, ..., video10 sort in the right order."""
    s = "".join(c for c in Path(name).stem if c.isdigit())
    return int(s) if s else 0


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["video_idx"] = df["video"].map(_natural_video_key)
    df = df.sort_values("video_idx").reset_index(drop=True)
    return df


def fig_per_video_outcome(df: pd.DataFrame, out: Path, method_label: str):
    """TP/FN/FP/TN status for each video — colored bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [
        {"TP": COLOR_TP, "FN": COLOR_FN, "FP": COLOR_FP, "TN": COLOR_TN}.get(l, "#999")
        for l in df["label"]
    ]
    heights = [1] * len(df)
    bars = ax.bar(df["video"], heights, color=colors, edgecolor="black", linewidth=0.5)
    for bar, lbl in zip(bars, df["label"]):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.5, lbl,
                ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax.set_yticks([])
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Video")
    ax.set_title(f"{method_label} — Per-Video Result")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_TP, label="TP (correct detection)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_FN, label="FN (missed)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_FP, label="FP (false alarm)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_TN, label="TN (negative)"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = out / "per_video_outcome.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_confusion_matrix(df: pd.DataFrame, out: Path, method_label: str):
    counts = df["label"].value_counts().to_dict()
    tp = counts.get("TP", 0)
    fn = counts.get("FN", 0)
    fp = counts.get("FP", 0)
    tn = counts.get("TN", 0)

    cm = np.array([[tp, fn], [fp, tn]])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        color = "white" if v > cm.max() / 2 else "black"
        ax.text(j, i, str(v), ha="center", va="center",
                fontsize=22, fontweight="bold", color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: ALARM", "Predicted: NO-ALARM"])
    ax.set_yticklabels(["Actual: ABANDONED", "Actual: NORMAL"])
    ax.set_title(f"{method_label} — Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    p = out / "confusion_matrix.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_metrics_bar(df: pd.DataFrame, out: Path, method_label: str):
    counts = df["label"].value_counts().to_dict()
    tp = counts.get("TP", 0)
    fn = counts.get("FN", 0)
    fp = counts.get("FP", 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = ["Precision", "Recall", "F1"]
    values = [precision, recall, f1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(metrics, values, color=[COLOR_TP, COLOR_BAR, "#9467bd"],
                  edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color=COLOR_REF, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Score")
    ax.set_title(f"{method_label} — Overall Metrics  (TP={tp}, FN={fn}, FP={fp})")
    plt.tight_layout()
    p = out / "metrics_bar.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_alarm_timing(df: pd.DataFrame, out: Path, method_label: str):
    """For each video: video duration vs. alarm time.
    If a GT time exists, mark it as a reference.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(df))[::-1]

    # Video duration (gray bar)
    ax.barh(y, df["duration_sec"], color="#dfe3e8", edgecolor="black",
            linewidth=0.4, label="Video duration")

    # A point, if a GT time exists
    has_gt = df["gt_start_sec"].notna()
    if has_gt.any():
        ax.scatter(df.loc[has_gt, "gt_start_sec"], y[has_gt.values],
                   marker="v", s=80, color="black", zorder=3, label="GT abandonment")

    # Alarm time: TP in green, FN as a red X
    alarm_mask = df["any_alarm"].astype(bool)
    if alarm_mask.any():
        ax.scatter(df.loc[alarm_mask, "first_alarm_sec"], y[alarm_mask.values],
                   marker="o", s=110, color=COLOR_TP, edgecolor="black", linewidth=0.6,
                   zorder=4, label="First alarm (TP)")
    miss_mask = ~alarm_mask
    if miss_mask.any():
        # Red X at the end of the duration for FNs
        ax.scatter(df.loc[miss_mask, "duration_sec"], y[miss_mask.values],
                   marker="x", s=110, color=COLOR_FN, linewidth=2.5,
                   zorder=4, label="Missed (FN)")

    ax.set_yticks(y)
    ax.set_yticklabels(df["video"])
    ax.set_xlabel("Seconds (video time)")
    ax.set_title(f"{method_label} — Alarm Timing")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    p = out / "alarm_timing.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_runtime(df: pd.DataFrame, out: Path, method_label: str):
    """Processing speed + wall time (two panels side by side)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1 = axes[0]
    ax1.bar(df["video"], df["speed_fps"], color=COLOR_BAR,
            edgecolor="black", linewidth=0.5)
    ax1.axhline(df["fps"].iloc[0] if len(df) else 30, color=COLOR_REF,
                linestyle="--", linewidth=1, label="Source video FPS")
    ax1.set_ylabel("Processing speed (fps)")
    ax1.set_title(f"{method_label} — Processing Speed")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=30)

    ax2 = axes[1]
    ratio = df["wall_time_sec"] / df["duration_sec"].replace(0, np.nan)
    bars = ax2.bar(df["video"], ratio, color="#9467bd",
                   edgecolor="black", linewidth=0.5)
    ax2.axhline(1.0, color=COLOR_REF, linestyle="--", linewidth=1,
                label="real-time threshold (1×)")
    ax2.set_ylabel("Wall time / video duration  (lower = faster)")
    ax2.set_title(f"{method_label} — Real-time Factor")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=30)
    for bar, r in zip(bars, ratio):
        if pd.notna(r):
            ax2.text(bar.get_x() + bar.get_width() / 2, r + 0.1,
                     f"{r:.1f}×", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    p = out / "runtime.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_summary_table(df: pd.DataFrame, out: Path, method_label: str):
    """Render the results as a compact table figure (for reports)."""
    cols = ["video", "duration_sec", "first_alarm_sec", "label",
            "wall_time_sec", "speed_fps"]
    table_df = df[cols].copy()
    table_df["duration_sec"] = table_df["duration_sec"].round(1)
    table_df["first_alarm_sec"] = table_df["first_alarm_sec"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "—"
    )
    table_df["wall_time_sec"] = table_df["wall_time_sec"].round(1)
    table_df["speed_fps"] = table_df["speed_fps"].round(1)

    fig, ax = plt.subplots(figsize=(11, 0.45 * len(table_df) + 1.5))
    ax.axis("off")
    headers = ["Video", "Duration (s)", "First Alarm (s)", "Result",
               "Wall (s)", "Speed (fps)"]
    cell_text = table_df.values.tolist()
    tbl = ax.table(cellText=cell_text, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    # Color coding: TP green, FN red tones
    for i, lbl in enumerate(table_df["label"]):
        color = {"TP": "#d4edda", "FN": "#f8d7da",
                 "FP": "#fff3cd", "TN": "#e2e3e5"}.get(lbl, "white")
        for j in range(len(headers)):
            tbl[(i + 1, j)].set_facecolor(color)
    # Header
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#343a40")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    plt.title(f"{method_label} — Detailed Results", fontweight="bold", pad=12)
    plt.tight_layout()
    p = out / "summary_table.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _metrics_from_df(df):
    counts = df["label"].value_counts().to_dict()
    tp = counts.get("TP", 0); fn = counts.get("FN", 0)
    fp = counts.get("FP", 0); tn = counts.get("TN", 0)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return dict(TP=tp, FN=fn, FP=fp, TN=tn,
                Precision=prec, Recall=rec, F1=f1)


def fig_compare_two(df_a: pd.DataFrame, label_a: str,
                    df_b: pd.DataFrame, label_b: str,
                    out: Path):
    """Baseline vs Optimized — Precision/Recall/F1 side by side."""
    m_a = _metrics_from_df(df_a)
    m_b = _metrics_from_df(df_b)
    metrics = ["Precision", "Recall", "F1"]
    vals_a = [m_a[k] for k in metrics]
    vals_b = [m_b[k] for k in metrics]

    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - w/2, vals_a, w, label=label_a, color="#888",
                edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, vals_b, w, label=label_b, color=COLOR_TP,
                edgecolor="black", linewidth=0.5)
    for bar, v in list(zip(b1, vals_a)) + list(zip(b2, vals_b)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.18)
    ax.axhline(1.0, color=COLOR_REF, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Score")
    ax.set_title("Comparison: Baseline vs Optimized")
    ax.legend(loc="lower right")
    note = (f"{label_a}: TP={m_a['TP']} FN={m_a['FN']} FP={m_a['FP']}\n"
            f"{label_b}: TP={m_b['TP']} FN={m_b['FN']} FP={m_b['FP']}")
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#ccc"))
    plt.tight_layout()
    p = out / "compare_baseline_vs_optimized.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="benchmark_results.csv")
    parser.add_argument("--out", default="eval/figures")
    parser.add_argument("--label", default="Method 1 (YOLO + DeepSORT + Distance)")
    parser.add_argument("--compare-csv", default=None,
                        help="Second CSV for comparison (baseline). If given, an extra figure is produced.")
    parser.add_argument("--compare-label", default="Baseline")
    args = parser.parse_args()

    df = load(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths = [
        fig_per_video_outcome(df, out, args.label),
        fig_confusion_matrix(df, out, args.label),
        fig_metrics_bar(df, out, args.label),
        fig_alarm_timing(df, out, args.label),
        fig_runtime(df, out, args.label),
        fig_summary_table(df, out, args.label),
    ]

    if args.compare_csv:
        df_b = load(args.compare_csv)
        paths.append(fig_compare_two(df_b, args.compare_label,
                                     df, args.label, out))

    print(f"[OK] {len(paths)} figures generated:")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
