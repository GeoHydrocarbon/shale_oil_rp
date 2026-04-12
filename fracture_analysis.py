from __future__ import annotations

import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess_v2 import resolve_path
from sensitivity_v2 import METRIC_INFO, get_score_weights, score_result_frame, validate_metrics


def _configure_matplotlib(base_dir: Path):
    mpl_config_dir = base_dir / ".mplconfig" / f"run_{os.getpid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.set_loglevel("error")
    import matplotlib.pyplot as plt

    return plt


def load_fracture_modules(base_dir: Path, config: dict):
    repo_path = (base_dir / config["paths"]["rockphypy_repo"]).resolve()
    package_dir = repo_path / "rockphypy"
    if not package_dir.exists():
        raise FileNotFoundError(f"rockphypy package not found: {package_dir}")

    package_name = "rockphypy"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_dir)]
        sys.modules[package_name] = package

    module_paths = {
        "EM": package_dir / "EM.py",
        "Anisotropy": package_dir / "Anisotropy.py",
    }
    loaded = {}
    for short_name, module_path in module_paths.items():
        full_name = f"{package_name}.{short_name}"
        module = sys.modules.get(full_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(full_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load module: {full_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
        setattr(package, short_name, module)
        loaded[short_name] = module

    return loaded["EM"].EM, loaded["Anisotropy"].Anisotropy


def get_host_arrays(frame: pd.DataFrame, fracture_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    host_domain = str(fracture_cfg.get("host_domain", "saturated_background")).lower()
    if host_domain == "saturated_background":
        return (
            frame["K_sat_GPa"].to_numpy(dtype=float),
            frame["G_sat_GPa"].to_numpy(dtype=float),
            frame["rho_model_gcc"].to_numpy(dtype=float),
        )
    if host_domain == "dry_background":
        return (
            frame["K_dry_GPa"].to_numpy(dtype=float),
            frame["G_dry_GPa"].to_numpy(dtype=float),
            ((1.0 - frame["phi"].to_numpy(dtype=float)) * frame["rho_matrix_gcc"].to_numpy(dtype=float)),
        )
    raise ValueError(f"Unsupported fracture.host_domain: {fracture_cfg['host_domain']}")


def get_inclusion_arrays(frame: pd.DataFrame, fracture_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    fill_mode = str(fracture_cfg.get("fill", "fluid")).lower()
    if fill_mode == "fluid":
        ki = frame["K_fluid_GPa"].to_numpy(dtype=float)
        gi = np.zeros(len(frame), dtype=float)
        return ki, gi
    if fill_mode == "dry":
        return np.zeros(len(frame), dtype=float), np.zeros(len(frame), dtype=float)
    raise ValueError(f"Unsupported fracture.fill: {fracture_cfg['fill']}")


def compute_fracture_velocities(
    frame: pd.DataFrame,
    config: dict,
    base_dir: Path,
    crack_density: float,
    dip_angle_deg: float,
) -> dict[str, np.ndarray]:
    fracture_cfg = config["fracture"]
    EM, Anisotropy = load_fracture_modules(base_dir, config)

    host_k, host_g, rho_host = get_host_arrays(frame, fracture_cfg)
    ki, gi = get_inclusion_arrays(frame, fracture_cfg)

    aspect_ratio = float(fracture_cfg["aspect_ratio"])
    incidence_deg = float(fracture_cfg.get("log_incidence_deg", 0.0))
    valid = (
        frame["compare_valid"].to_numpy(dtype=bool)
        & np.isfinite(host_k)
        & np.isfinite(host_g)
        & np.isfinite(rho_host)
        & np.isfinite(ki)
    )

    vp = np.full(len(frame), np.nan)
    vsv = np.full(len(frame), np.nan)
    vsh = np.full(len(frame), np.nan)
    c33 = np.full(len(frame), np.nan)
    c44 = np.full(len(frame), np.nan)

    # 第一步：在当前无裂缝背景上叠加固定密度和倾角的裂缝，计算等效 TI 刚度和速度。
    for index in np.where(valid)[0]:
        c_eff = EM.hudson_cone(
            host_k[index],
            host_g[index],
            ki[index],
            gi[index],
            aspect_ratio,
            crack_density,
            dip_angle_deg,
        )
        vp_km_s, vsh_km_s, vsv_km_s = Anisotropy.vel_azi_VTI(c_eff, rho_host[index], incidence_deg)
        vp[index] = float(vp_km_s) * 1000.0
        vsh[index] = float(vsh_km_s) * 1000.0
        vsv[index] = float(vsv_km_s) * 1000.0
        c33[index] = float(c_eff[2, 2])
        c44[index] = float(c_eff[3, 3])

    return {
        "vp_fracture_m_s": vp,
        "vsv_fracture_m_s": vsv,
        "vsh_fracture_m_s": vsh,
        "fracture_valid": valid,
        "C33_fracture_GPa": c33,
        "C44_fracture_GPa": c44,
    }


def evaluate_fracture_scenario(
    frame: pd.DataFrame,
    config: dict,
    base_dir: Path,
    crack_density: float,
    dip_angle_deg: float,
) -> dict[str, float | int]:
    velocities = compute_fracture_velocities(frame, config, base_dir, crack_density, dip_angle_deg)
    valid = velocities["fracture_valid"] & np.isfinite(velocities["vp_fracture_m_s"]) & np.isfinite(velocities["vsv_fracture_m_s"])

    vp_misfit = velocities["vp_fracture_m_s"][valid] - frame.loc[valid, "vp_log_m_s"].to_numpy(dtype=float)
    vs_misfit = velocities["vsv_fracture_m_s"][valid] - frame.loc[valid, "vs_log_m_s"].to_numpy(dtype=float)

    # 第二步：把裂缝场景转成可比较的误差指标，为后续敏感性和反演打基础。
    return {
        "fracture_compare_rows": int(valid.sum()),
        "vp_rmse": float(np.sqrt(np.nanmean(vp_misfit**2))) if valid.any() else np.nan,
        "vs_rmse": float(np.sqrt(np.nanmean(vs_misfit**2))) if valid.any() else np.nan,
        "vp_bias": float(np.nanmean(vp_misfit)) if valid.any() else np.nan,
        "vs_bias": float(np.nanmean(vs_misfit)) if valid.any() else np.nan,
        "rho_rmse": 0.0,
    }


def build_reference_frame(
    frame: pd.DataFrame,
    config: dict,
    base_dir: Path,
    crack_density: float,
    dip_angle_deg: float,
) -> pd.DataFrame:
    velocities = compute_fracture_velocities(frame, config, base_dir, crack_density, dip_angle_deg)
    result = frame.copy()
    result["fracture_density"] = crack_density
    result["fracture_dip_angle_deg"] = dip_angle_deg
    result["vp_fracture_m_s"] = velocities["vp_fracture_m_s"]
    result["vs_fracture_m_s"] = velocities["vsv_fracture_m_s"]
    result["vsh_fracture_m_s"] = velocities["vsh_fracture_m_s"]
    result["vp_fracture_misfit_m_s"] = result["vp_fracture_m_s"] - result["vp_log_m_s"]
    result["vs_fracture_misfit_m_s"] = result["vs_fracture_m_s"] - result["vs_log_m_s"]
    result["fracture_valid"] = velocities["fracture_valid"]
    result["C33_fracture_GPa"] = velocities["C33_fracture_GPa"]
    result["C44_fracture_GPa"] = velocities["C44_fracture_GPa"]
    return result


def scan_fracture_grid(frame: pd.DataFrame, config: dict, base_dir: Path) -> pd.DataFrame:
    fracture_cfg = config["fracture"]
    density_grid = [float(value) for value in fracture_cfg["density_grid"]]
    dip_grid = [float(value) for value in fracture_cfg["dip_grid_deg"]]
    rows = []

    # 第一步：扫描裂缝密度和裂缝倾角二维网格，观察速度误差的整体变化。
    for crack_density in density_grid:
        for dip_angle_deg in dip_grid:
            metrics = evaluate_fracture_scenario(frame, config, base_dir, crack_density, dip_angle_deg)
            rows.append(
                {
                    "fracture_density": crack_density,
                    "fracture_dip_angle_deg": dip_angle_deg,
                    **metrics,
                }
            )

    result = pd.DataFrame(rows)
    return score_result_frame(result, get_score_weights(config))


def plot_fracture_grid(grid_frame: pd.DataFrame, config: dict, base_dir: Path) -> Path:
    fracture_cfg = config["fracture"]
    metrics = validate_metrics(list(fracture_cfg.get("grid_metrics", ["composite_score", "vp_rmse", "vs_rmse"])), allow_composite=True)
    plt = _configure_matplotlib(base_dir)

    output_dir = resolve_path(base_dir, fracture_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / fracture_cfg["grid_figure_filename"]

    figure_width = float(fracture_cfg.get("figure_width", 16))
    figure_height = float(fracture_cfg.get("figure_height", 10))
    dpi = int(fracture_cfg.get("dpi", 300))
    axis_label_fontsize = float(fracture_cfg.get("axis_label_fontsize", 14))
    tick_label_fontsize = float(fracture_cfg.get("tick_label_fontsize", 12))
    title_fontsize = float(fracture_cfg.get("title_fontsize", 16))

    best_row = grid_frame.sort_values(["composite_score", "vp_rmse", "vs_rmse"], kind="mergesort").iloc[0]
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figure_width, figure_height), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    # 第二步：为综合评分和关键误差指标绘制裂缝密度-倾角热图。
    for axis, metric in zip(axes, metrics):
        pivot = (
            grid_frame.pivot(index="fracture_dip_angle_deg", columns="fracture_density", values=metric)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        image = axis.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="viridis")
        axis.set_title(METRIC_INFO[metric]["label"], fontsize=title_fontsize)
        axis.set_xlabel("Crack Density", fontsize=axis_label_fontsize)
        axis.set_ylabel("Dip Angle (deg)", fontsize=axis_label_fontsize)
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels([f"{value:.3g}" for value in pivot.columns], fontsize=tick_label_fontsize)
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels([f"{value:.0f}" for value in pivot.index], fontsize=tick_label_fontsize)

        best_x_index = list(pivot.columns).index(best_row["fracture_density"])
        best_y_index = list(pivot.index).index(best_row["fracture_dip_angle_deg"])
        axis.scatter(best_x_index, best_y_index, marker="*", s=180, color="white", edgecolor="black")
        fig.colorbar(image, ax=axis, shrink=0.9)

    for axis in axes[len(metrics) :]:
        axis.set_visible(False)

    # 第三步：把当前最优的裂缝密度和倾角组合标到热图标题里。
    fig.suptitle(
        f"Best fracture scenario: density={best_row['fracture_density']}, dip={best_row['fracture_dip_angle_deg']} deg",
        fontsize=title_fontsize + 1,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_fracture_analysis(frame: pd.DataFrame, config: dict, base_dir: Path) -> dict | None:
    fracture_cfg = config.get("fracture", {})
    if not fracture_cfg.get("enabled", False):
        return None

    reference_density = float(fracture_cfg["density"])
    reference_dip_angle_deg = float(fracture_cfg["dip_angle_deg"])

    # 第一步：生成一个基准裂缝场景结果表，便于和无裂缝背景做井段对比。
    reference_frame = build_reference_frame(frame, config, base_dir, reference_density, reference_dip_angle_deg)
    reference_output = resolve_path(base_dir, fracture_cfg["output_reference"])
    reference_output.parent.mkdir(parents=True, exist_ok=True)
    reference_frame.to_excel(reference_output, index=False)

    # 第二步：扫描裂缝密度和倾角二维网格，并输出综合评分结果。
    grid_frame = scan_fracture_grid(frame, config, base_dir)
    best_row = grid_frame.sort_values(["composite_score", "vp_rmse", "vs_rmse"], kind="mergesort").iloc[0]

    recommendation = pd.DataFrame(
        [
            {
                "best_fracture_density": best_row["fracture_density"],
                "best_dip_angle_deg": best_row["fracture_dip_angle_deg"],
                "composite_score": best_row["composite_score"],
                "vp_rmse": best_row["vp_rmse"],
                "vs_rmse": best_row["vs_rmse"],
                "vp_bias": best_row["vp_bias"],
                "vs_bias": best_row["vs_bias"],
            }
        ]
    )

    output_table = resolve_path(base_dir, fracture_cfg["output_table"])
    output_table.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_table) as writer:
        grid_frame.to_excel(writer, sheet_name="fracture_grid", index=False)
        recommendation.to_excel(writer, sheet_name="recommendation", index=False)

    # 第三步：绘制裂缝密度-倾角热图，为后面的裂缝密度反演提供先验。
    grid_figure_path = plot_fracture_grid(grid_frame, config, base_dir)

    return {
        "reference_output": reference_output,
        "output_table": output_table,
        "grid_figure_path": grid_figure_path,
        "summary": {
            "reference_density": reference_density,
            "reference_dip_angle_deg": reference_dip_angle_deg,
            "best_fracture_density": float(best_row["fracture_density"]),
            "best_dip_angle_deg": float(best_row["fracture_dip_angle_deg"]),
            "grid_rows": int(len(grid_frame)),
        },
    }
