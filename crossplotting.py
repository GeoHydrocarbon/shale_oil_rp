from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess_v2 import resolve_path


def _configure_matplotlib(base_dir: Path):
    mpl_config_dir = base_dir / ".mplconfig" / f"run_{os.getpid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.set_loglevel("error")
    import matplotlib.pyplot as plt

    return plt


def compute_dynamic_elastic_properties(vp_m_s: np.ndarray, vs_m_s: np.ndarray, rho_gcc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rho_safe = np.asarray(rho_gcc, dtype=float)
    vp_safe = np.asarray(vp_m_s, dtype=float)
    vs_safe = np.asarray(vs_m_s, dtype=float)

    g_gpa = rho_safe * vs_safe**2 / 1e6
    k_gpa = rho_safe * (vp_safe**2 - (4.0 / 3.0) * vs_safe**2) / 1e6

    denominator = 3.0 * k_gpa + g_gpa
    young = np.full(len(vp_safe), np.nan)
    poisson = np.full(len(vp_safe), np.nan)

    valid = np.isfinite(k_gpa) & np.isfinite(g_gpa) & np.isfinite(denominator) & (denominator != 0.0)
    young[valid] = 9.0 * k_gpa[valid] * g_gpa[valid] / denominator[valid]
    poisson[valid] = (3.0 * k_gpa[valid] - 2.0 * g_gpa[valid]) / (2.0 * denominator[valid])
    return young, poisson


def build_crossplot_frame(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    crossplot_cfg = config.get("crossplots", {})
    elastic_source = str(crossplot_cfg.get("elastic_source", "fracture_inv")).lower()

    working = frame.copy()
    working["brittle_vol"] = (
        working["QUAZ_vol"].to_numpy(dtype=float)
        + working["DOLM_vol"].to_numpy(dtype=float)
        + working["LIME_vol"].to_numpy(dtype=float)
        + working["PYRITE_vol"].to_numpy(dtype=float)
    )

    if elastic_source == "fracture_inv" and "fracture_inversion_valid" in working.columns:
        elastic_valid = working["fracture_inversion_valid"].to_numpy(dtype=bool)
        vp = np.where(elastic_valid, working["vp_fracture_inv_m_s"], np.nan)
        vs = np.where(elastic_valid, working["vs_fracture_inv_m_s"], np.nan)
    else:
        elastic_valid = working["compare_valid"].to_numpy(dtype=bool)
        vp = np.where(elastic_valid, working["vp_log_m_s"], np.nan)
        vs = np.where(elastic_valid, working["vs_log_m_s"], np.nan)

    rho = working["rho_log_gcc"].to_numpy(dtype=float)
    working["ai_crossplot"] = vp * rho
    working["vpvs_crossplot"] = vp / vs
    young, poisson = compute_dynamic_elastic_properties(vp, vs, rho)
    working["young_crossplot_gpa"] = young
    working["poisson_crossplot"] = poisson
    working["crossplot_valid"] = (
        elastic_valid
        & np.isfinite(working["ai_crossplot"])
        & np.isfinite(working["vpvs_crossplot"])
        & np.isfinite(working["young_crossplot_gpa"])
        & np.isfinite(working["poisson_crossplot"])
        & np.isfinite(working["phi"])
        & np.isfinite(working["so"])
        & np.isfinite(working["toc_clean"])
        & np.isfinite(working["brittle_vol"])
        & np.isfinite(working["fracture_density_inv"])
    )
    return working.loc[working["crossplot_valid"]].copy()


def plot_crossplots(frame: pd.DataFrame, config: dict, base_dir: Path) -> Path | None:
    crossplot_cfg = config.get("crossplots", {})
    if not crossplot_cfg.get("enabled", False):
        return None

    plot_data = build_crossplot_frame(frame, config)
    if plot_data.empty:
        raise ValueError("No valid rows available for crossplotting.")

    plt = _configure_matplotlib(base_dir)
    output_dir = resolve_path(base_dir, crossplot_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / crossplot_cfg["panel_filename"]

    figure_width = float(crossplot_cfg.get("figure_width", 28))
    figure_height = float(crossplot_cfg.get("figure_height", 12))
    dpi = int(crossplot_cfg.get("dpi", 300))
    axis_label_fontsize = float(crossplot_cfg.get("axis_label_fontsize", 14))
    tick_label_fontsize = float(crossplot_cfg.get("tick_label_fontsize", 12))
    title_fontsize = float(crossplot_cfg.get("title_fontsize", 16))
    point_size = float(crossplot_cfg.get("point_size", 18))
    point_alpha = float(crossplot_cfg.get("point_alpha", 0.85))
    cmap = str(crossplot_cfg.get("colormap", "turbo"))

    specs = [
        ("phi", "ai_crossplot", "so", "AI vs Phi", "Porosity", "Acoustic Impedance"),
        ("fracture_density_inv", "ai_crossplot", "so", "AI vs Fracture Density", "Fracture Density", "Acoustic Impedance"),
        ("so", "ai_crossplot", "phi", "AI vs Oil Saturation", "Oil Saturation", "Acoustic Impedance"),
        ("phi", "vpvs_crossplot", "so", "Vp/Vs vs Phi", "Porosity", "Vp/Vs"),
        ("fracture_density_inv", "vpvs_crossplot", "so", "Vp/Vs vs Fracture Density", "Fracture Density", "Vp/Vs"),
        ("so", "vpvs_crossplot", "phi", "Vp/Vs vs Oil Saturation", "Oil Saturation", "Vp/Vs"),
        ("toc_clean", "fracture_density_inv", "brittle_vol", "TOC vs Fracture Density", "TOC", "Fracture Density"),
        ("poisson_crossplot", "young_crossplot_gpa", "fracture_density_inv", "E vs Poisson", "Poisson's Ratio", "Young's Modulus (GPa)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(figure_width, figure_height), constrained_layout=True)
    axes = axes.ravel()

    # 第一步：围绕储集性、含油性、裂缝和脆性四类因素绘制推荐的 8 张交会图。
    for axis, (x_col, y_col, color_col, title, xlabel, ylabel) in zip(axes, specs):
        scatter = axis.scatter(
            plot_data[x_col],
            plot_data[y_col],
            c=plot_data[color_col],
            s=point_size,
            alpha=point_alpha,
            cmap=cmap,
            edgecolors="none",
        )
        axis.set_title(title, fontsize=title_fontsize)
        axis.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        axis.set_ylabel(ylabel, fontsize=axis_label_fontsize)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.tick_params(axis="both", labelsize=tick_label_fontsize)
        colorbar = fig.colorbar(scatter, ax=axis, shrink=0.9)
        colorbar.ax.tick_params(labelsize=tick_label_fontsize)
        colorbar.set_label(color_col, fontsize=axis_label_fontsize - 1)

    # 第二步：保存交会图总面板，便于和论文中的物性-弹性解释图直接对照。
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
