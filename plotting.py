from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from preprocess import resolve_path


def _configure_matplotlib(base_dir: Path):
    # 把 matplotlib 缓存目录放到工作区的独立运行目录，避免用户目录无写权限时出错。
    mpl_config_dir = base_dir / ".mplconfig" / f"run_{os.getpid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.set_loglevel("error")
    import matplotlib.pyplot as plt

    return plt


def plot_velocity_depth_panel(frame: pd.DataFrame, config: dict, base_dir: Path) -> Path | None:
    plots_cfg = config.get("plots", {})
    if not plots_cfg.get("enabled", True):
        return None

    plt = _configure_matplotlib(base_dir)

    # 第一步：筛选出有完整速度对比结果的样本。
    plot_data = frame.loc[frame["compare_valid"]].copy()
    if plot_data.empty:
        raise ValueError("No valid rows available for plotting.")

    # 第二步：解析绘图参数和输出路径。
    output_dir = resolve_path(base_dir, plots_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / plots_cfg["velocity_depth_filename"]

    figure_width = float(plots_cfg.get("figure_width", 20))
    figure_height = float(plots_cfg.get("figure_height", 12))
    dpi = int(plots_cfg.get("dpi", 300))
    axis_label_fontsize = float(plots_cfg.get("axis_label_fontsize", 16))
    tick_label_fontsize = float(plots_cfg.get("tick_label_fontsize", 14))
    title_fontsize = float(plots_cfg.get("title_fontsize", 18))
    legend_fontsize = float(plots_cfg.get("legend_fontsize", 13))
    line_width = float(plots_cfg.get("line_width", 1.8))

    depth = plot_data["depth_m"].to_numpy()

    # 第三步：在一个画布中绘制四个横向排列的深度综合子图。
    fig, axes = plt.subplots(1, 4, figsize=(figure_width, figure_height), sharey=True, constrained_layout=True)

    axes[0].plot(plot_data["vp_log_m_s"], depth, label="Vp_log", lw=line_width)
    axes[0].plot(plot_data["vp_model_m_s"], depth, label="Vp_model", lw=line_width)
    axes[0].set_title("Vp", fontsize=title_fontsize)
    axes[0].set_xlabel("Velocity (m/s)", fontsize=axis_label_fontsize)
    axes[0].set_ylabel("Depth (m)", fontsize=axis_label_fontsize)
    axes[0].legend(fontsize=legend_fontsize)

    axes[1].plot(plot_data["vs_log_m_s"], depth, label="Vs_log", lw=line_width)
    axes[1].plot(plot_data["vs_model_m_s"], depth, label="Vs_model", lw=line_width)
    axes[1].set_title("Vs", fontsize=title_fontsize)
    axes[1].set_xlabel("Velocity (m/s)", fontsize=axis_label_fontsize)
    axes[1].legend(fontsize=legend_fontsize)

    axes[2].plot(plot_data["vp_misfit_m_s"], depth, label="Vp_error", lw=line_width, color="tab:red")
    axes[2].axvline(0.0, color="black", lw=1.0, linestyle="--")
    axes[2].set_title("Vp Error", fontsize=title_fontsize)
    axes[2].set_xlabel("Error (m/s)", fontsize=axis_label_fontsize)
    axes[2].legend(fontsize=legend_fontsize)

    axes[3].plot(plot_data["vs_misfit_m_s"], depth, label="Vs_error", lw=line_width, color="tab:orange")
    axes[3].axvline(0.0, color="black", lw=1.0, linestyle="--")
    axes[3].set_title("Vs Error", fontsize=title_fontsize)
    axes[3].set_xlabel("Error (m/s)", fontsize=axis_label_fontsize)
    axes[3].legend(fontsize=legend_fontsize)

    # 第四步：统一设置深度方向、刻度字号和网格样式。
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

    # 第五步：保存图片并释放内存。
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
